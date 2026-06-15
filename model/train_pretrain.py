"""
train_pretrain.py
==================
ArthaSathi LLM — Pre-Training from Scratch
Trains the model on 15B tokens of multilingual Indian language text.
Supports: single GPU (Colab T4/A100), gradient accumulation, mixed precision,
          automatic checkpointing, Colab session recovery.
"""

import os, sys, time, json, math, random, argparse
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.cuda.amp as amp

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.arthasathi_model import ArthaSathiLLM, get_small_config, get_medium_config
from model.arthasathi_model import ArthaSathiConfig


# ─────────────────────────────────────────────────────────────
# 1. TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, data_iter, eval_steps: int = 50, device: str = "cuda") -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with amp.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


# ─────────────────────────────────────────────────────────────
# 2. DATASET LOADER
# ─────────────────────────────────────────────────────────────

class PretrainDataset:
    """
    Memory-efficient streaming dataset for pre-training.
    Reads from JSONL files, tokenizes on-the-fly, yields (input, target) batches.
    Supports multiple files from multiple languages with balancing.
    """

    def __init__(self, data_files: list, tokenizer_dir: str,
                 context_length: int = 1024, batch_size: int = 8):
        self.data_files     = data_files
        self.context_length = context_length
        self.batch_size     = batch_size

        # Load tokenizer
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(
            str(Path(tokenizer_dir) / "tokenizer.json")
        )
        self.pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        self.bos_id = self.tokenizer.token_to_id("[BOS]") or 2
        self.eos_id = self.tokenizer.token_to_id("[EOS]") or 3

        # Token buffer — we accumulate tokens then cut into chunks
        self.buffer: list = []
        self._file_idx = 0
        self._file_handle = None
        self._open_next_file()

    def _open_next_file(self):
        if self._file_handle:
            self._file_handle.close()
        if self._file_idx >= len(self.data_files):
            self._file_idx = 0  # cycle
        self._file_handle = open(self.data_files[self._file_idx],
                                 encoding="utf-8")
        self._file_idx += 1

    def _fill_buffer(self, min_tokens: int = 4096):
        """Read from files until buffer has enough tokens"""
        while len(self.buffer) < min_tokens:
            line = self._file_handle.readline()
            if not line:
                self._open_next_file()
                continue
            try:
                obj  = json.loads(line)
                text = obj.get("text", "")
            except Exception:
                continue
            if not text.strip():
                continue

            # Tokenize
            ids  = [self.bos_id]
            ids += self.tokenizer.encode(text).ids
            ids += [self.eos_id]
            self.buffer.extend(ids)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        """Return a single batch of (input_ids, target_ids)"""
        chunk_size = self.context_length + 1  # +1 for targets
        needed     = self.batch_size * chunk_size

        self._fill_buffer(needed * 2)

        if len(self.buffer) < needed:
            self._fill_buffer(needed * 4)

        xs, ys = [], []
        for _ in range(self.batch_size):
            chunk       = self.buffer[:chunk_size]
            self.buffer = self.buffer[chunk_size:]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])

        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y


# ─────────────────────────────────────────────────────────────
# 3. LEARNING RATE SCHEDULER
# ─────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int,
                                     total_steps: int, min_ratio: float = 0.1):
    """
    Cosine LR schedule with linear warmup.
    - Warmup: LR linearly increases from 0 to max_lr
    - Cosine decay: LR decreases from max_lr to min_ratio * max_lr
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────────────────────

class PreTrainer:
    def __init__(self, config: dict):
        self.cfg    = config
        self.device = self._setup_device()
        self.model  = self._setup_model()
        self.opt, self.sched = self._setup_optimizer()
        self.scaler = amp.GradScaler()
        self.step   = 0
        self.best_loss = float("inf")

    def _setup_device(self) -> str:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        else:
            device = "cpu"
            print("WARNING: No GPU found. Training will be very slow.")
        return device

    def _setup_model(self) -> ArthaSathiLLM:
        size = self.cfg.get("model_size", "small")
        if size == "medium":
            model_cfg = get_medium_config()
        else:
            model_cfg = get_small_config()

        model = ArthaSathiLLM(model_cfg).to(self.device)

        # Load checkpoint if resuming
        ckpt_path = self.cfg.get("resume_from")
        if ckpt_path and Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            self.step = ckpt.get("step", 0)
            print(f"Resumed from step {self.step}")

        # Enable gradient checkpointing (saves memory, 20% slower)
        if self.cfg.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") \
                else print("gradient_checkpointing not available for custom model")

        print(f"Model parameters: {model.param_count()/1e6:.1f}M")
        return model

    def _setup_optimizer(self):
        """
        AdamW with decoupled weight decay.
        Key: do NOT apply weight decay to bias params and layer norm weights.
        """
        decay_params    = [p for n, p in self.model.named_parameters()
                          if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in self.model.named_parameters()
                          if p.requires_grad and p.dim() < 2]

        opt = AdamW(
            [
                {"params": decay_params,    "weight_decay": self.cfg["weight_decay"]},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.cfg["learning_rate"],
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        sched = get_cosine_schedule_with_warmup(
            opt,
            warmup_steps=self.cfg["warmup_steps"],
            total_steps=self.cfg["max_steps"],
        )

        # Load scheduler state if resuming
        ckpt_path = self.cfg.get("resume_from")
        if ckpt_path and Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "optimizer_state_dict" in ckpt:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                sched.load_state_dict(ckpt["scheduler_state_dict"])

        return opt, sched

    def train(self, train_dataset, eval_dataset=None):
        """Main training loop"""
        cfg   = self.cfg
        model = self.model
        opt   = self.opt
        sched = self.sched

        print("\n" + "=" * 60)
        print(f"Pre-Training ArthaSathi LLM")
        print(f"  Steps:       {cfg['max_steps']:,}")
        print(f"  Batch size:  {cfg['batch_size']} x {cfg['grad_accum']} = "
              f"{cfg['batch_size']*cfg['grad_accum']}")
        print(f"  Context len: {cfg['context_length']}")
        print(f"  Max LR:      {cfg['learning_rate']:.0e}")
        print("=" * 60 + "\n")

        model.train()
        train_iter = iter(train_dataset)
        t0         = time.time()
        log_rows   = []

        for step in range(self.step, cfg["max_steps"]):
            self.step = step

            # ── Gradient accumulation ─────────────────────────
            opt.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for micro in range(cfg["grad_accum"]):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataset)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)

                # Mixed precision forward pass
                with amp.autocast(device_type="cuda", dtype=torch.float16,
                                  enabled=(self.device == "cuda")):
                    _, loss = model(x, y)
                    loss    = loss / cfg["grad_accum"]

                self.scaler.scale(loss).backward()
                loss_accum += loss.item()

            # Gradient clipping (prevents exploding gradients)
            self.scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer + scheduler step
            self.scaler.step(opt)
            self.scaler.update()
            sched.step()

            # ── Logging ───────────────────────────────────────
            if step % cfg["log_every"] == 0:
                lr    = sched.get_last_lr()[0]
                dt    = time.time() - t0
                tps   = (cfg["log_every"] * cfg["batch_size"] *
                         cfg["grad_accum"] * cfg["context_length"]) / max(dt, 1)
                t0    = time.time()

                log   = {
                    "step":       step,
                    "loss":       round(loss_accum, 4),
                    "lr":         f"{lr:.2e}",
                    "grad_norm":  round(grad_norm.item(), 3),
                    "tok/s":      int(tps),
                }
                print(f"Step {step:6d} | loss={loss_accum:.4f} | "
                      f"lr={lr:.2e} | grad={grad_norm:.3f} | {int(tps):,} tok/s")
                log_rows.append(log)

                # Save training log
                log_path = Path(cfg["checkpoint_dir"]) / "train_log.jsonl"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(log) + "\n")

            # ── Evaluation ────────────────────────────────────
            if eval_dataset and step % cfg["eval_every"] == 0 and step > 0:
                eval_loss = estimate_loss(model, iter(eval_dataset),
                                          eval_steps=50, device=self.device)
                print(f"  [EVAL] step={step} | eval_loss={eval_loss:.4f}")
                model.train()

                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint(f"best_model.pt", step)

            # ── Checkpointing ─────────────────────────────────
            if step % cfg["save_every"] == 0 and step > 0:
                self.save_checkpoint(f"step_{step:07d}.pt", step)

            # ── Colab auto-save to Drive ───────────────────────
            if step % cfg.get("gdrive_save_every", 5000) == 0 and step > 0:
                self._save_to_gdrive(step)

        # Final save
        self.save_checkpoint("final_pretrain.pt", self.step)
        print(f"\nPre-training complete! Final loss: {loss_accum:.4f}")

    def save_checkpoint(self, filename: str, step: int):
        Path(self.cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        path = Path(self.cfg["checkpoint_dir"]) / filename
        self.model.save_checkpoint(str(path), step, self.opt, self.sched)

        # Keep only last 3 checkpoints (save disk space)
        ckpts = sorted(Path(self.cfg["checkpoint_dir"]).glob("step_*.pt"))
        for old in ckpts[:-3]:
            old.unlink()

    def _save_to_gdrive(self, step: int):
        """Copy checkpoint to Google Drive (for Colab users)"""
        gdrive = self.cfg.get("gdrive_dir")
        if not gdrive or not Path(gdrive).exists():
            return
        try:
            import shutil
            src = Path(self.cfg["checkpoint_dir"])
            dst = Path(gdrive) / "arthasathi_checkpoints"
            dst.mkdir(exist_ok=True)
            # Only copy latest
            latest = src / f"step_{step:07d}.pt"
            if latest.exists():
                shutil.copy2(str(latest), str(dst / "latest.pt"))
                print(f"  → Saved to Google Drive (step {step})")
        except Exception as e:
            print(f"  → GDrive save failed: {e}")


# ─────────────────────────────────────────────────────────────
# 5. DEFAULT CONFIG + MAIN
# ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    "model_size":        "small",    # "small"(117M) or "medium"(345M)
    "context_length":    1024,

    # Training
    "max_steps":         150_000,
    "batch_size":        4,          # Per-GPU batch size (T4: 4, A100: 16)
    "grad_accum":        16,         # Effective batch = 4 * 16 = 64
    "learning_rate":     3e-4,
    "weight_decay":      0.1,
    "warmup_steps":      2_000,

    # Checkpointing
    "checkpoint_dir":    "checkpoints/pretrain",
    "save_every":        5_000,
    "eval_every":        2_500,
    "log_every":         100,
    "gdrive_save_every": 3_000,
    "gdrive_dir":        "/content/drive/MyDrive",  # Colab path

    # Data
    "train_files":       ["formatted/pretrain.jsonl"],
    "tokenizer_dir":     "arthasathi_tokenizer",
    "resume_from":       None,
    "gradient_checkpointing": False,
}


def main():
    parser = argparse.ArgumentParser(description="Pre-train ArthaSathi LLM")
    parser.add_argument("--config",      type=str, help="JSON config file")
    parser.add_argument("--model_size",  default="small", choices=["small","medium","large"])
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--max_steps",   type=int, default=150_000)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--data_files",  nargs="+", default=["formatted/pretrain.jsonl"])
    parser.add_argument("--tok_dir",     default="arthasathi_tokenizer")
    parser.add_argument("--ckpt_dir",    default="checkpoints/pretrain")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    cfg["model_size"]    = args.model_size
    cfg["batch_size"]    = args.batch_size
    cfg["max_steps"]     = args.max_steps
    cfg["resume_from"]   = args.resume_from
    cfg["train_files"]   = args.data_files
    cfg["tokenizer_dir"] = args.tok_dir
    cfg["checkpoint_dir"] = args.ckpt_dir

    # Validate files exist
    for f in cfg["train_files"]:
        if not Path(f).exists():
            print(f"ERROR: training file not found: {f}")
            print("Run data/dataset_pipeline.py first to prepare training data.")
            sys.exit(1)

    # Build dataset
    print(f"\nLoading training data: {cfg['train_files']}")
    train_ds = PretrainDataset(
        data_files=cfg["train_files"],
        tokenizer_dir=cfg["tokenizer_dir"],
        context_length=cfg["context_length"],
        batch_size=cfg["batch_size"],
    )

    # Train
    trainer = PreTrainer(cfg)
    trainer.train(train_ds)


if __name__ == "__main__":
    main()
