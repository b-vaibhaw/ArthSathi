"""
train_finetune.py
==================
ArthaSathi LLM — Supervised Fine-Tuning (SFT)
Stage 1: Instruction fine-tuning (Alpaca format)
Stage 2: Chat fine-tuning (ChatML format)
Supports: Full fine-tuning + LoRA (Parameter-Efficient Fine-Tuning)
LoRA runs on just 6GB VRAM — perfect for Colab T4 free tier.
"""

import os, sys, json, time, math, argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.cuda.amp as amp

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.arthasathi_model import ArthaSathiLLM, ArthaSathiConfig


# ─────────────────────────────────────────────────────────────
# 1. LORA IMPLEMENTATION FROM SCRATCH
# ─────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    LoRA: Low-Rank Adaptation of Large Language Models (Hu et al. 2021)
    
    Instead of updating the full weight matrix W (d × k),
    we learn two small matrices A (d × r) and B (r × k) where r << min(d,k).
    
    During training: W_new = W_original + alpha/r * B @ A
    During inference: weights can be merged: W_merged = W + alpha/r * B @ A
    
    This reduces trainable parameters from d*k to r*(d+k), typically by 99%+
    """

    def __init__(self, linear: nn.Linear, rank: int = 16,
                 alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        in_features  = linear.in_features
        out_features = linear.out_features

        # Frozen original weights
        self.weight = linear.weight
        self.bias   = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # LoRA low-rank matrices
        self.lora_A   = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B   = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling  = alpha / rank
        self.dropout  = nn.Dropout(dropout)

        # Initialize A with Kaiming (standard for random projection)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialized to zero → zero change at start of training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass
        base_out = F.linear(x, self.weight, self.bias)
        # LoRA delta
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out

    def merge_weights(self):
        """Merge LoRA weights into base weights for faster inference"""
        delta = self.scaling * self.lora_B @ self.lora_A
        self.weight.data += delta
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)

    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora(model: ArthaSathiLLM, rank: int = 16, alpha: float = 32.0,
               dropout: float = 0.05, target_modules: List[str] = None) -> int:
    """
    Apply LoRA to target modules (attention projections).
    Returns the count of trainable LoRA parameters.
    """
    if target_modules is None:
        target_modules = ["Wq", "Wk", "Wv", "Wo"]  # Attention matrices

    # Freeze ALL parameters first
    for p in model.parameters():
        p.requires_grad_(False)

    lora_params = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parts  = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr_name = parts[-1]

                # Replace with LoRA version
                lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_module)
                lora_params += lora_module.trainable_params()

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA applied: {trainable/1e6:.2f}M trainable / {total/1e6:.1f}M total "
          f"({100*trainable/total:.2f}%)")
    return trainable


def merge_lora_weights(model: ArthaSathiLLM):
    """Merge all LoRA weights for fast inference"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    print("LoRA weights merged into base model")


# ─────────────────────────────────────────────────────────────
# 2. DATASETS
# ─────────────────────────────────────────────────────────────

class InstructionDataset(Dataset):
    """
    Instruction fine-tuning dataset (Alpaca format).
    Format: {"instruction": ..., "input": ..., "output": ...}
    """

    PROMPT_TEMPLATE = (
        "Below is a financial question. Answer clearly and specifically.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )
    PROMPT_NO_INPUT = (
        "Below is a financial question. Answer clearly and specifically.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )

    def __init__(self, data_file: str, tokenizer_dir: str, max_length: int = 1024):
        from tokenizers import Tokenizer
        self.tokenizer  = Tokenizer.from_file(
            str(Path(tokenizer_dir) / "tokenizer.json")
        )
        self.pad_id     = self.tokenizer.token_to_id("[PAD]") or 0
        self.max_length = max_length
        self.examples   = []

        with open(data_file, encoding="utf-8") as f:
            for line in f:
                try:
                    self.examples.append(json.loads(line))
                except Exception:
                    continue

        print(f"Loaded {len(self.examples):,} instruction examples from {data_file}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        inp_text = ex.get("input", "").strip()
        if inp_text:
            full = self.PROMPT_TEMPLATE.format(
                instruction=ex.get("instruction",""),
                input=inp_text,
                output=ex.get("output",""),
            )
        else:
            full = self.PROMPT_NO_INPUT.format(
                instruction=ex.get("instruction",""),
                output=ex.get("output",""),
            )

        ids = self.tokenizer.encode(full).ids
        ids = ids[:self.max_length]

        # Find where the response starts (compute loss only on response)
        response_marker = self.tokenizer.encode("### Response:\n").ids
        resp_start      = self._find_subseq(ids, response_marker)

        # Create labels: -100 (ignore) for prompt, actual ids for response
        labels = [-100] * len(ids)
        if resp_start != -1:
            response_start = resp_start + len(response_marker)
            labels[response_start:] = ids[response_start:]

        # Pad
        pad_len = self.max_length - len(ids)
        ids     = ids    + [self.pad_id] * pad_len
        labels  = labels + [-100]       * pad_len

        return {
            "input_ids": torch.tensor(ids,    dtype=torch.long),
            "labels":    torch.tensor(labels, dtype=torch.long),
        }

    @staticmethod
    def _find_subseq(seq: list, subseq: list) -> int:
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i:i+len(subseq)] == subseq:
                return i
        return -1


class ChatDataset(Dataset):
    """
    Chat fine-tuning dataset (ChatML format).
    Format: {"text": "<|im_start|>system...user...assistant...", "lang": ...}
    """

    def __init__(self, data_file: str, tokenizer_dir: str, max_length: int = 1024):
        from tokenizers import Tokenizer
        self.tokenizer  = Tokenizer.from_file(
            str(Path(tokenizer_dir) / "tokenizer.json")
        )
        self.pad_id      = self.tokenizer.token_to_id("[PAD]") or 0
        self.ast_id      = self.tokenizer.token_to_id("<|assistant|>") or None
        self.max_length  = max_length
        self.examples    = []

        with open(data_file, encoding="utf-8") as f:
            for line in f:
                try:
                    self.examples.append(json.loads(line))
                except Exception:
                    continue

        print(f"Loaded {len(self.examples):,} chat examples from {data_file}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        text = self.examples[idx]["text"]
        ids  = self.tokenizer.encode(text).ids[:self.max_length]

        # Compute loss only on assistant turns
        labels = self._mask_non_assistant(ids)

        # Pad to max_length
        pad    = self.max_length - len(ids)
        ids    = ids    + [self.pad_id] * pad
        labels = labels + [-100]        * pad

        return {
            "input_ids": torch.tensor(ids,    dtype=torch.long),
            "labels":    torch.tensor(labels, dtype=torch.long),
        }

    def _mask_non_assistant(self, ids: list) -> list:
        """
        Set labels to -100 everywhere except assistant responses.
        This teaches the model to RESPOND, not to repeat instructions.
        """
        labels     = [-100] * len(ids)
        in_assist  = False
        ast_token  = self.ast_id

        for i, tok in enumerate(ids):
            if tok == ast_token:
                in_assist = True
            if in_assist:
                labels[i] = tok
        return labels


# ─────────────────────────────────────────────────────────────
# 3. FINE-TUNING TRAINER
# ─────────────────────────────────────────────────────────────

class FineTuner:
    def __init__(self, pretrain_ckpt: str, config: dict):
        self.cfg    = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = self._load_model(pretrain_ckpt)
        self.opt, self.sched = self._setup_optimizer()
        self.scaler = amp.GradScaler()
        self.step   = 0

    def _load_model(self, ckpt_path: str) -> ArthaSathiLLM:
        print(f"Loading pre-trained model from {ckpt_path}...")
        model = ArthaSathiLLM.from_checkpoint(ckpt_path, device=self.device)

        # Apply LoRA if requested
        if self.cfg.get("use_lora", True):
            apply_lora(
                model,
                rank=self.cfg.get("lora_rank", 16),
                alpha=self.cfg.get("lora_alpha", 32.0),
                dropout=self.cfg.get("lora_dropout", 0.05),
            )
        else:
            # Full fine-tuning — unfreeze all
            for p in model.parameters():
                p.requires_grad_(True)
            print(f"Full fine-tuning: {model.param_count()/1e6:.1f}M trainable params")

        return model

    def _setup_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt   = AdamW(trainable, lr=self.cfg["learning_rate"],
                      betas=(0.9, 0.999), weight_decay=0.01)
        total = self.cfg["max_steps"]
        warm  = self.cfg.get("warmup_steps", 200)
        from torch.optim.lr_scheduler import LambdaLR
        sched = LambdaLR(opt, lambda s: min(s/max(warm,1),
                         0.1 + 0.9*max(0, (total-s)/(total-warm))))
        return opt, sched

    def train(self, dataset: Dataset, output_dir: str = "checkpoints/finetune"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        loader = DataLoader(dataset, batch_size=self.cfg["batch_size"],
                            shuffle=True, num_workers=2, pin_memory=True)
        self.model.train()
        best_loss = float("inf")

        for epoch in range(self.cfg.get("epochs", 3)):
            print(f"\nEpoch {epoch+1}/{self.cfg.get('epochs',3)}")
            epoch_loss = 0.0
            n_batches  = 0

            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)

                self.opt.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", dtype=torch.float16,
                                  enabled=(self.device == "cuda")):
                    # Forward pass — full logits needed for instruction loss
                    logits, _ = self.model(input_ids)

                    # Custom loss: only on non-masked labels
                    loss = F.cross_entropy(
                        logits.view(-1, self.model.config.vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()

                epoch_loss += loss.item()
                n_batches  += 1
                self.step  += 1

                if self.step % 50 == 0:
                    avg = epoch_loss / n_batches
                    lr  = self.sched.get_last_lr()[0]
                    print(f"  step={self.step:5d} loss={loss.item():.4f} "
                          f"avg={avg:.4f} lr={lr:.2e}")

                if self.step % 1000 == 0:
                    self.model.save_checkpoint(
                        f"{output_dir}/ft_step_{self.step}.pt", self.step
                    )

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.model.save_checkpoint(f"{output_dir}/best_ft.pt", self.step)

        # Merge LoRA weights for clean inference model
        if self.cfg.get("use_lora", True):
            merge_lora_weights(self.model)

        self.model.save_checkpoint(f"{output_dir}/final_ft.pt", self.step)
        print(f"\nFine-tuning complete! Best loss: {best_loss:.4f}")
        print(f"Final model saved to {output_dir}/final_ft.pt")


# ─────────────────────────────────────────────────────────────
# 4. QUANTIZATION (for deployment on low-VRAM devices)
# ─────────────────────────────────────────────────────────────

def quantize_model_int8(model: ArthaSathiLLM) -> ArthaSathiLLM:
    """
    Post-training quantization to INT8.
    Reduces model size ~4x, minimal quality loss.
    """
    try:
        import bitsandbytes as bnb
        print("Applying INT8 quantization via bitsandbytes...")
        # Replace linear layers with quantized versions
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
                parent = model
                parts  = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                q_layer = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                )
                q_layer.weight.data = module.weight.data
                setattr(parent, parts[-1], q_layer)
        return model
    except ImportError:
        print("bitsandbytes not available. Skipping INT8 quantization.")
        return model


def quantize_model_4bit(model_path: str, output_path: str):
    """
    Load model with 4-bit quantization using bitsandbytes.
    Reduces a 345M model from ~1.4GB to ~350MB.
    """
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes as bnb
        print("4-bit quantization requires HuggingFace transformers wrapper.")
        print("For pure PyTorch 4-bit: use the INT8 path or GGUF conversion.")
    except ImportError:
        print("Install: pip install bitsandbytes transformers")


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

DEFAULT_FT_CONFIG = {
    "use_lora":       True,
    "lora_rank":      16,
    "lora_alpha":     32.0,
    "lora_dropout":   0.05,
    "learning_rate":  5e-5,       # Lower than pre-training
    "batch_size":     4,
    "grad_accum":     8,
    "epochs":         3,
    "max_steps":      30_000,
    "warmup_steps":   200,
    "ft_type":        "instruction",  # "instruction" or "chat"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_ckpt", required=True,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--data_file",     required=True,
                        help="JSONL file for fine-tuning")
    parser.add_argument("--tok_dir",       default="arthasathi_tokenizer")
    parser.add_argument("--output_dir",    default="checkpoints/finetune")
    parser.add_argument("--ft_type",       default="instruction",
                        choices=["instruction", "chat"])
    parser.add_argument("--use_lora",      action="store_true", default=True)
    parser.add_argument("--lora_rank",     type=int, default=16)
    parser.add_argument("--epochs",        type=int, default=3)
    parser.add_argument("--batch_size",    type=int, default=4)
    args = parser.parse_args()

    cfg = DEFAULT_FT_CONFIG.copy()
    cfg["use_lora"]  = args.use_lora
    cfg["lora_rank"] = args.lora_rank
    cfg["epochs"]    = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["ft_type"]   = args.ft_type

    # Load dataset
    if args.ft_type == "instruction":
        ds = InstructionDataset(args.data_file, args.tok_dir, max_length=1024)
    else:
        ds = ChatDataset(args.data_file, args.tok_dir, max_length=1024)

    # Fine-tune
    finetuner = FineTuner(args.pretrain_ckpt, cfg)
    finetuner.train(ds, args.output_dir)


if __name__ == "__main__":
    main()
