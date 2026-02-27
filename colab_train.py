"""
colab_train.py
================
ArthaSathi — Complete Google Colab Training Script
Run this file in Colab to train the LLM from scratch.
Copy-paste sections into separate notebook cells OR run as script.

COLAB SETUP:
  Runtime → Change runtime type → GPU (T4 for small, A100 for medium)
  Connect Google Drive for persistent checkpoints.

STEPS:
  Cell 1: Install dependencies
  Cell 2: Mount Google Drive
  Cell 3: Download datasets
  Cell 4: Train tokenizer
  Cell 5: Pre-train LLM
  Cell 6: Fine-tune LLM
  Cell 7: Test the model
"""

# ══════════════════════════════════════════════════
# CELL 1 — Install Dependencies
# ══════════════════════════════════════════════════
CELL_1_INSTALL = """
# Run this cell FIRST in Colab

!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers tokenizers datasets sentencepiece
!pip install -q faiss-cpu sentence-transformers
!pip install -q openai-whisper gtts librosa soundfile
!pip install -q fastapi uvicorn httpx pydantic
!pip install -q bitsandbytes accelerate peft
!pip install -q PyMuPDF ftfy langdetect

# Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"PyTorch: {torch.__version__}")
"""

# ══════════════════════════════════════════════════
# CELL 2 — Mount Google Drive
# ══════════════════════════════════════════════════
CELL_2_DRIVE = """
from google.colab import drive
drive.mount('/content/drive')

import os, shutil

# Create directories
DRIVE_DIR   = "/content/drive/MyDrive/arthasathi"
COLAB_DIR   = "/content/arthasathi"
CKPT_DIR    = f"{COLAB_DIR}/checkpoints"

os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(COLAB_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)
os.makedirs(f"{COLAB_DIR}/data/raw", exist_ok=True)
os.makedirs(f"{COLAB_DIR}/data/clean", exist_ok=True)
os.makedirs(f"{COLAB_DIR}/data/synthetic", exist_ok=True)
os.makedirs(f"{COLAB_DIR}/data/formatted", exist_ok=True)

print(f"Working directory: {COLAB_DIR}")
print(f"Drive backup:      {DRIVE_DIR}")

# Restore latest checkpoint from Drive if exists
for fname in ["arthasathi_tokenizer", "latest_checkpoint.pt"]:
    src = f"{DRIVE_DIR}/{fname}"
    dst = f"{COLAB_DIR}/{fname}"
    if os.path.exists(src) and not os.path.exists(dst):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"Restored: {fname}")

# Auto-save to Drive every 20 minutes
import threading, time

def auto_backup():
    while True:
        time.sleep(1200)  # 20 minutes
        for item in ["arthasathi_tokenizer", "checkpoints"]:
            src = f"{COLAB_DIR}/{item}"
            dst = f"{DRIVE_DIR}/{item}"
            if os.path.exists(src):
                try:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                except Exception as e:
                    print(f"Backup warning: {e}")
        print(f"[{time.strftime('%H:%M')}] Auto-backed up to Drive")

t = threading.Thread(target=auto_backup, daemon=True)
t.start()
print("Auto-backup thread started (every 20 min)")
"""

# ══════════════════════════════════════════════════
# CELL 3 — Download Datasets
# ══════════════════════════════════════════════════
CELL_3_DATA = """
import sys
sys.path.insert(0, '/content/arthasathi')

from data.dataset_pipeline import DatasetDownloader, DataCleaner

COLAB_DIR = "/content/arthasathi"

# Download datasets (streaming — doesn't fill RAM)
# Set gb_per_lang to control how much data per language
# 0.5 = fast test run | 2.0 = production quality
GB_PER_LANG = 1.0   # Change this — 1.0GB x 9 langs = 9GB raw

downloader = DatasetDownloader(output_dir=f"{COLAB_DIR}/data/raw")
downloader.download_all(gb_per_lang=GB_PER_LANG)

print("\\nDownload complete!")

# Clean the downloaded data
print("\\nCleaning data...")
cleaner = DataCleaner()
cleaner.process_all(
    raw_dir=f"{COLAB_DIR}/data/raw",
    clean_dir=f"{COLAB_DIR}/data/clean"
)
print("Cleaning complete!")
"""

# ══════════════════════════════════════════════════
# CELL 4 — Train Tokenizer
# ══════════════════════════════════════════════════
CELL_4_TOKENIZER = """
import sys, os
sys.path.insert(0, '/content/arthasathi')

from pathlib import Path
from model.tokenizer import train_arthasathi_tokenizer, ArthaSathiTokenizer

COLAB_DIR = "/content/arthasathi"
DRIVE_DIR = "/content/drive/MyDrive/arthasathi"

# Check if tokenizer already trained
TOK_DIR = f"{COLAB_DIR}/arthasathi_tokenizer"
if Path(f"{TOK_DIR}/tokenizer.json").exists():
    print("Tokenizer already trained, loading...")
    tok = ArthaSathiTokenizer(TOK_DIR)
    print(f"Vocab size: {len(tok)}")
else:
    # Collect training files (use a sample — tokenizer training needs ~10GB)
    clean_dir = Path(f"{COLAB_DIR}/data/clean")
    data_files = [str(p) for p in clean_dir.rglob("*.jsonl")][:50]  # Cap at 50 files

    # Convert JSONL to plain text files for tokenizer training
    txt_dir = f"{COLAB_DIR}/data/tokenizer_txt"
    os.makedirs(txt_dir, exist_ok=True)

    import json
    for jf in data_files[:20]:  # Use first 20 files to create text samples
        lang = Path(jf).stem.split("_")[-1]
        out  = f"{txt_dir}/{lang}_{Path(jf).parent.name}.txt"
        if Path(out).exists():
            continue
        with open(jf, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= 100000: break  # 100K lines per file
                try:
                    fout.write(json.loads(line).get("text","") + "\\n")
                except Exception:
                    pass

    txt_files = [str(p) for p in Path(txt_dir).glob("*.txt") if p.stat().st_size > 1000]
    print(f"Training tokenizer on {len(txt_files)} text files...")

    tokenizer = train_arthasathi_tokenizer(
        data_files=txt_files,
        save_dir=TOK_DIR,
        vocab_size=60000,
    )

    # Backup to Drive
    import shutil
    shutil.copytree(TOK_DIR, f"{DRIVE_DIR}/arthasathi_tokenizer", dirs_exist_ok=True)
    print("Tokenizer backed up to Google Drive!")

print("\\nTokenizer ready!")
"""

# ══════════════════════════════════════════════════
# CELL 5 — Pre-Training
# ══════════════════════════════════════════════════
CELL_5_PRETRAIN = """
import sys, os, json, time, math, torch
import torch.amp as amp
from torch.optim import AdamW
from pathlib import Path

sys.path.insert(0, '/content/arthasathi')

from model.arthasathi_model import ArthaSathiLLM, get_small_config, get_medium_config
from model.train_pretrain import PretrainDataset, get_cosine_schedule_with_warmup

COLAB_DIR = "/content/arthasathi"
DRIVE_DIR = "/content/drive/MyDrive/arthasathi"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ─── CHOOSE MODEL SIZE ─────────────────────────────────
# T4 GPU (15GB VRAM):  use small config (117M)
# A100 GPU (40GB VRAM): use medium config (345M)
FREE_T4 = torch.cuda.get_device_properties(0).total_memory < 20e9 if torch.cuda.is_available() else True

if FREE_T4:
    config     = get_small_config()
    BATCH_SIZE = 4
    GRAD_ACCUM = 16   # Effective batch = 64
    MAX_STEPS  = 50000
    print("Using Small config (117M) — Colab T4")
else:
    config     = get_medium_config()
    BATCH_SIZE = 12
    GRAD_ACCUM = 8    # Effective batch = 96
    MAX_STEPS  = 150000
    print("Using Medium config (345M) — A100")

CKPT_DIR  = f"{COLAB_DIR}/checkpoints/pretrain"
TOK_DIR   = f"{COLAB_DIR}/arthasathi_tokenizer"
os.makedirs(CKPT_DIR, exist_ok=True)

# ─── FORMAT PRE-TRAIN DATA ─────────────────────────────
from data.dataset_pipeline import DatasetFormatter

fmt_file = f"{COLAB_DIR}/data/formatted/pretrain.jsonl"
if not Path(fmt_file).exists():
    print("Formatting pre-training data...")
    formatter = DatasetFormatter(max_length=config.context_length)
    formatter.format_pretrain(
        clean_dirs=[f"{COLAB_DIR}/data/clean"],
        output_file=fmt_file,
        target_tokens=5_000_000_000,  # 5B tokens for quick training
    )

# ─── DATASET ───────────────────────────────────────────
print(f"Loading dataset from {fmt_file}...")
train_ds = PretrainDataset(
    data_files=[fmt_file],
    tokenizer_dir=TOK_DIR,
    context_length=config.context_length,
    batch_size=BATCH_SIZE,
)

# ─── MODEL + OPTIMIZER ─────────────────────────────────
# Check if resuming
RESUME_CKPT = f"{CKPT_DIR}/latest.pt"
if Path(RESUME_CKPT).exists():
    print(f"Resuming from checkpoint...")
    model = ArthaSathiLLM.from_checkpoint(RESUME_CKPT, device=DEVICE)
    ckpt  = torch.load(RESUME_CKPT, map_location="cpu")
    start_step = ckpt.get("step", 0)
    print(f"Resuming from step {start_step}")
else:
    model      = ArthaSathiLLM(config).to(DEVICE)
    start_step = 0

decay   = [p for n,p in model.named_parameters() if p.requires_grad and p.dim()>=2]
nodecay = [p for n,p in model.named_parameters() if p.requires_grad and p.dim()<2]
opt     = AdamW([{"params":decay,"weight_decay":0.1},{"params":nodecay,"weight_decay":0.0}],
                 lr=3e-4, betas=(0.9,0.95))
sched   = get_cosine_schedule_with_warmup(opt, warmup_steps=2000, total_steps=MAX_STEPS)
scaler  = amp.GradScaler()

# ─── TRAINING LOOP ─────────────────────────────────────
print(f"\\n{'='*50}")
print(f"Training ArthaSathi LLM")
print(f"  Model:    {model.param_count()/1e6:.1f}M params")
print(f"  Steps:    {MAX_STEPS:,}")
print(f"  Batch:    {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
print(f"  Device:   {DEVICE}")
print(f"{'='*50}\\n")

train_iter = iter(train_ds)
t0         = time.time()
best_loss  = float("inf")

model.train()
for step in range(start_step, MAX_STEPS):
    opt.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro in range(GRAD_ACCUM):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ds)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)
        with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            _, loss = model(x, y)
            loss    = loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        loss_accum += loss.item()

    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()

    if step % 100 == 0:
        lr  = sched.get_last_lr()[0]
        dt  = time.time() - t0
        tps = (100 * BATCH_SIZE * GRAD_ACCUM * config.context_length) / max(dt, 1)
        t0  = time.time()
        print(f"Step {step:6d} | loss={loss_accum:.4f} | lr={lr:.2e} | {int(tps):,} tok/s")

    if step % 2000 == 0 and step > 0:
        # Save checkpoint
        model.save_checkpoint(f"{CKPT_DIR}/step_{step:07d}.pt", step, opt, sched)
        shutil.copy2(f"{CKPT_DIR}/step_{step:07d}.pt", f"{CKPT_DIR}/latest.pt")
        # Backup to Drive
        try:
            shutil.copy2(f"{CKPT_DIR}/latest.pt", f"{DRIVE_DIR}/latest_checkpoint.pt")
            print(f"  → Backed up to Drive (step {step})")
        except Exception as e:
            print(f"  → Drive backup failed: {e}")

print(f"\\nPre-training complete!")
"""

# ══════════════════════════════════════════════════
# CELL 6 — Fine-Tuning
# ══════════════════════════════════════════════════
CELL_6_FINETUNE = """
import sys, os
sys.path.insert(0, '/content/arthasathi')

from model.train_finetune import FineTuner, ChatDataset, InstructionDataset, apply_lora
from data.dataset_pipeline import DatasetFormatter, SyntheticDataGenerator
from pathlib import Path

COLAB_DIR = "/content/arthasathi"
TOK_DIR   = f"{COLAB_DIR}/arthasathi_tokenizer"
CKPT_DIR  = f"{COLAB_DIR}/checkpoints"

# ─── Generate Synthetic Data ───────────────────────────
SYNTH_DIR = f"{COLAB_DIR}/data/synthetic"
os.makedirs(SYNTH_DIR, exist_ok=True)

print("Generating synthetic financial conversations...")
gen = SyntheticDataGenerator()  # Uses local Mistral if GPU available
gen.generate_batch(SYNTH_DIR, n_per_lang=5000, use_llm=False)  # Template-based first

# Format as chat data
fmt = DatasetFormatter(max_length=1024)
chat_file = f"{COLAB_DIR}/data/formatted/chat.jsonl"
fmt.format_chat(SYNTH_DIR, output_file=chat_file)

# ─── Fine-Tune ─────────────────────────────────────────
PRETRAIN_CKPT = f"{CKPT_DIR}/pretrain/latest.pt"
if not Path(PRETRAIN_CKPT).exists():
    PRETRAIN_CKPT = f"{COLAB_DIR}/latest_checkpoint.pt"

if not Path(PRETRAIN_CKPT).exists():
    print("ERROR: Pre-trained checkpoint not found!")
    print("Run Cell 5 (Pre-training) first.")
else:
    ft_config = {
        "use_lora":       True,
        "lora_rank":      16,
        "lora_alpha":     32.0,
        "lora_dropout":   0.05,
        "learning_rate":  5e-5,
        "batch_size":     4,
        "grad_accum":     8,
        "epochs":         3,
        "max_steps":      15000,
        "warmup_steps":   200,
        "ft_type":        "chat",
    }

    ds = ChatDataset(chat_file, TOK_DIR, max_length=1024)
    ft = FineTuner(PRETRAIN_CKPT, ft_config)
    ft.train(ds, output_dir=f"{CKPT_DIR}/finetune")

    print("\\nFine-tuning complete!")
    print(f"Model saved to {CKPT_DIR}/finetune/final_ft.pt")
"""

# ══════════════════════════════════════════════════
# CELL 7 — Test the Model
# ══════════════════════════════════════════════════
CELL_7_TEST = """
import sys, torch
sys.path.insert(0, '/content/arthasathi')

from model.arthasathi_model import ArthaSathiLLM
from tokenizers import Tokenizer
from pathlib import Path

COLAB_DIR = "/content/arthasathi"
TOK_DIR   = f"{COLAB_DIR}/arthasathi_tokenizer"
MODEL_PATH = f"{COLAB_DIR}/checkpoints/finetune/final_ft.pt"

# Fall back to pre-train checkpoint if fine-tune not done yet
if not Path(MODEL_PATH).exists():
    MODEL_PATH = f"{COLAB_DIR}/checkpoints/pretrain/latest.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}...")
model     = ArthaSathiLLM.from_checkpoint(MODEL_PATH, device=DEVICE)
tokenizer = Tokenizer.from_file(f"{TOK_DIR}/tokenizer.json")
bos_id    = tokenizer.token_to_id("[BOS]") or 2

def chat(prompt, language="hi", max_tokens=200):
    sys_prompts = {
        "hi": "Tum ArthaSathi ho. Simple Hindi mein financial advice do.",
        "en": "You are ArthaSathi. Give clear financial advice.",
        "mr": "तुम्ही ArthaSathi आहात. मराठीत आर्थिक सल्ला द्या.",
        "ta": "நீங்கள் ArthaSathi. தமிழில் நிதி ஆலோசனை கொடுங்கள்.",
    }
    sys_p  = sys_prompts.get(language, sys_prompts["en"])
    full   = f"<|im_start|>system\\n{sys_p}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
    ids    = tokenizer.encode(full).ids[-800:]
    x      = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    out    = model.generate(x, max_new_tokens=max_tokens, temperature=0.75, top_p=0.9)
    new    = out[0, len(ids):].tolist()
    return tokenizer.decode(new, skip_special_tokens=True).strip()

# Test in all 9 languages
test_queries = [
    ("Mera HDFC credit card ka 25000 ka debt hai. Kya karna chahiye?", "hi"),
    ("My loan EMI is Rs 5000 but I lost my job. What are my options?", "en"),
    ("माझ्या दुकानाची वार्षिक उलाढाल 15 लाख आहे. GST भरायला हवे का?", "mr"),
    ("என் கடன் 50000 ரூபாய். வட்டி விகிதம் 24%. என்ன செய்வது?", "ta"),
    ("ನನ್ನ ಸಾಲ 30000. ಬ್ಯಾಂಕ್ ಜೊತೆ negotiate ಮಾಡುವುದು ಹೇಗೆ?", "kn"),
]

print("\\n" + "="*55)
print("ArthaSathi Model — Live Test")
print("="*55)
for query, lang in test_queries:
    print(f"\\nLanguage: {lang.upper()}")
    print(f"User: {query}")
    response = chat(query, lang)
    print(f"ArthaSathi: {response[:300]}")
    print("-" * 40)
"""

# ══════════════════════════════════════════════════
# MAIN — Print all cells
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    cells = {
        "CELL 1 — Install":       CELL_1_INSTALL,
        "CELL 2 — Google Drive":  CELL_2_DRIVE,
        "CELL 3 — Data":          CELL_3_DATA,
        "CELL 4 — Tokenizer":     CELL_4_TOKENIZER,
        "CELL 5 — Pre-Train":     CELL_5_PRETRAIN,
        "CELL 6 — Fine-Tune":     CELL_6_FINETUNE,
        "CELL 7 — Test":          CELL_7_TEST,
    }
    print("ArthaSathi — Colab Training Guide")
    print("="*55)
    for title, code in cells.items():
        print(f"\n{'#'*55}")
        print(f"# {title}")
        print(f"{'#'*55}")
        print(code.strip())
    print("\n\nCopy each section into a separate Colab cell and run in order.")
    print("Total training time estimate:")
    print("  Tokenizer:       4-6 hours (CPU)")
    print("  Pre-train 117M:  72 hours (T4 free, 50K steps)")
    print("  Pre-train 345M:  120 hours (A100, 150K steps)")
    print("  Fine-tune LoRA:  8-10 hours (either GPU)")
    print("  Total (medium):  ~6 days spread across Colab sessions")
