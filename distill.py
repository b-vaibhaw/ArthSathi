"""
distill.py
===========
ArthaSathi — Knowledge Distillation
Teacher: 345M ArthaSathi model (full quality)
Student:  80M distilled model (for mobile / offline)

Distillation reduces model from 345M → ~80M parameters (~75% reduction)
while retaining ~90% of teacher quality.
Enables: Android/iOS app with offline operation, 2x faster inference.

Method: Response distillation (KL divergence on output distributions)
"""

import sys, json, math, time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.cuda.amp as amp

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.arthasathi_model import ArthaSathiLLM, ArthaSathiConfig, get_small_config


# ─────────────────────────────────────────────────────────────
# 1. STUDENT MODEL CONFIG (80M params)
# ─────────────────────────────────────────────────────────────

def get_distill_config() -> ArthaSathiConfig:
    """
    Student model: ~80M parameters
    - 6 layers (vs 24 for teacher)
    - 512 hidden dim (vs 1024 for teacher)
    - 8 heads (vs 16 for teacher)
    - Same vocab size (60K)
    Fits in 300MB, runs in <2s on mobile CPU.
    """
    return ArthaSathiConfig(
        vocab_size=60000,
        context_length=512,     # Shorter context for mobile
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
    )


# ─────────────────────────────────────────────────────────────
# 2. DISTILLATION DATASET
# ─────────────────────────────────────────────────────────────

class DistillDataset(Dataset):
    """
    Loads pre-formatted chat data for distillation.
    Same format as fine-tuning, but used with teacher soft labels.
    """
    def __init__(self, data_file: str, tokenizer_dir: str, max_length: int = 512):
        from tokenizers import Tokenizer
        self.tokenizer  = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
        self.pad_id     = self.tokenizer.token_to_id("[PAD]") or 0
        self.max_length = max_length
        self.examples   = []

        with open(data_file, encoding="utf-8") as f:
            for line in f:
                try:
                    self.examples.append(json.loads(line).get("text",""))
                except Exception:
                    continue
        print(f"Distill dataset: {len(self.examples):,} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids    = self.tokenizer.encode(self.examples[idx]).ids[:self.max_length]
        pad    = self.max_length - len(ids)
        ids   += [self.pad_id] * pad
        return torch.tensor(ids, dtype=torch.long)


# ─────────────────────────────────────────────────────────────
# 3. DISTILLATION TRAINER
# ─────────────────────────────────────────────────────────────

class DistillationTrainer:
    """
    Knowledge distillation: train student to mimic teacher distributions.
    
    Loss = alpha * CE(student, hard_labels) + (1-alpha) * KL(student || teacher)
    
    alpha = 0.3 means 30% standard cross-entropy + 70% distillation loss.
    Temperature T softens distributions: higher T = softer, more transfer.
    """

    def __init__(self, teacher_path: str, student_config: ArthaSathiConfig,
                 device: str = "cuda",
                 temperature: float = 4.0,
                 alpha: float = 0.3):
        self.device = device
        self.T      = temperature
        self.alpha  = alpha

        # Load teacher
        print(f"Loading teacher model from {teacher_path}...")
        self.teacher = ArthaSathiLLM.from_checkpoint(teacher_path, device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        print(f"Teacher: {self.teacher.param_count()/1e6:.1f}M params (frozen)")

        # Create student
        self.student = ArthaSathiLLM(student_config).to(device)
        print(f"Student: {self.student.param_count()/1e6:.1f}M params (training)")

    def distillation_loss(self, student_logits: torch.Tensor,
                           teacher_logits: torch.Tensor,
                           targets: torch.Tensor) -> torch.Tensor:
        """
        Combined distillation + cross-entropy loss.
        
        KL divergence between softened student and teacher distributions.
        Hard CE loss on ground-truth labels.
        """
        # Soft labels from teacher (temperature scaling)
        with torch.no_grad():
            teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)

        student_log_soft = F.log_softmax(student_logits / self.T, dim=-1)

        # KL divergence (distillation loss) — scaled by T^2
        kl_loss = F.kl_div(
            student_log_soft.view(-1, student_logits.size(-1)),
            teacher_soft.view(-1, teacher_soft.size(-1)),
            reduction="batchmean",
        ) * (self.T ** 2)

        # Hard cross-entropy loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=0,  # pad token
        )

        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss

    def train(self, data_file: str, tokenizer_dir: str,
              output_dir: str = "checkpoints/distilled",
              epochs: int = 5, batch_size: int = 8, lr: float = 5e-4):
        """Run the distillation training loop"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        ds     = DistillDataset(data_file, tokenizer_dir, max_length=512)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)

        opt    = AdamW(self.student.parameters(), lr=lr, weight_decay=0.1)
        scaler = amp.GradScaler()

        print(f"\n{'='*50}")
        print("Knowledge Distillation Training")
        print(f"  Teacher: {self.teacher.param_count()/1e6:.0f}M → "
              f"Student: {self.student.param_count()/1e6:.0f}M")
        print(f"  Temperature: {self.T} | Alpha: {self.alpha}")
        print(f"  Epochs: {epochs} | Batch: {batch_size}")
        print(f"{'='*50}\n")

        best_loss = float("inf")
        for epoch in range(epochs):
            self.student.train()
            epoch_loss = 0.0
            n_batches  = 0
            t0         = time.time()

            for batch in loader:
                ids = batch.to(self.device)

                # Split into input and target
                x   = ids[:, :-1]
                y   = ids[:, 1:]

                opt.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", dtype=torch.float16,
                                  enabled=(self.device=="cuda")):
                    # Student forward
                    student_logits, _ = self.student(x)

                    # Teacher forward (no grad, we just need soft labels)
                    with torch.no_grad():
                        teacher_logits, _ = self.teacher(x)

                    loss = self.distillation_loss(student_logits, teacher_logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed  = time.time() - t0
            print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | {elapsed:.0f}s")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.student.save_checkpoint(f"{output_dir}/best_student.pt", epoch)

        self.student.save_checkpoint(f"{output_dir}/final_student.pt", epochs)
        print(f"\nDistillation complete! Best loss: {best_loss:.4f}")
        print(f"Student model: {output_dir}/final_student.pt")
        print(f"Size reduction: {self.teacher.param_count()/self.student.param_count():.1f}x fewer params")
        return self.student


# ─────────────────────────────────────────────────────────────
# 4. ONNX EXPORT (for mobile)
# ─────────────────────────────────────────────────────────────

def export_to_onnx(model: ArthaSathiLLM, output_path: str = "arthasathi_mobile.onnx",
                    context_length: int = 128, batch_size: int = 1):
    """
    Export distilled model to ONNX format for mobile/edge deployment.
    The ONNX model can then be converted to TensorFlow Lite for Android.
    
    Android deployment:
      1. Export to ONNX (this function)
      2. Convert: tf.convert(onnx) → TFLite
      3. Quantize: TFLite dynamic quantization → ~300MB
      4. Run on Android: TFLite interpreter
    
    Expected model size: ~80M params × 4 bytes = ~320MB unquantized
                        ~80MB with INT8 quantization
    """
    import torch.onnx

    model.eval()
    model.clear_cache()

    # Dummy input
    dummy_input = torch.randint(0, model.config.vocab_size, (batch_size, context_length))

    print(f"Exporting model to ONNX: {output_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits":    {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        size_mb = Path(output_path).stat().st_size / 1e6
        print(f"ONNX export successful: {size_mb:.1f}MB → {output_path}")
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def quantize_onnx_int8(onnx_path: str, output_path: str) -> bool:
    """Quantize ONNX model to INT8 (reduces size 4x)"""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
        orig_mb = Path(onnx_path).stat().st_size / 1e6
        quant_mb = Path(output_path).stat().st_size / 1e6
        print(f"INT8 quantization: {orig_mb:.1f}MB → {quant_mb:.1f}MB "
              f"({100*quant_mb/orig_mb:.0f}%)")
        return True
    except ImportError:
        print("pip install onnxruntime onnx")
        return False


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher",    required=True,
                        help="Path to teacher (345M) checkpoint")
    parser.add_argument("--data_file",  required=True,
                        help="Chat JSONL for distillation")
    parser.add_argument("--tok_dir",    default="arthasathi_tokenizer")
    parser.add_argument("--output",     default="checkpoints/distilled")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--export_onnx", action="store_true")
    args = parser.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    student_cfg = get_distill_config()

    trainer = DistillationTrainer(
        teacher_path=args.teacher,
        student_config=student_cfg,
        device=device,
    )
    student = trainer.train(
        data_file=args.data_file,
        tokenizer_dir=args.tok_dir,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if args.export_onnx:
        onnx_path  = f"{args.output}/arthasathi_mobile.onnx"
        quant_path = f"{args.output}/arthasathi_mobile_int8.onnx"
        if export_to_onnx(student, onnx_path):
            quantize_onnx_int8(onnx_path, quant_path)
