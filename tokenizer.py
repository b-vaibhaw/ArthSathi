"""
tokenizer.py
=============
Custom BPE Tokenizer for ArthaSathi — 9 Indian Languages
Vocabulary: 60,000 tokens covering Hindi, English, Marathi, Tamil,
            Kannada, Bhojpuri, Assamese, Bengali, Telugu
            + financial domain special tokens
"""

import os, json, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 1. TRAIN TOKENIZER
# ─────────────────────────────────────────────────────────────

def train_arthasathi_tokenizer(
    data_files:   List[str],
    save_dir:     str = "arthasathi_tokenizer",
    vocab_size:   int = 60000,
    min_freq:     int = 3,
):
    """
    Train a custom BPE tokenizer on multilingual Indian language corpus.

    Args:
        data_files : list of .txt files with training text (one per language ideal)
        save_dir   : directory to save tokenizer files
        vocab_size : target vocabulary size (60K covers all 9 languages well)
        min_freq   : minimum token frequency to include
    """
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace, Sequence as PTSequence, Digits
        from tokenizers.normalizers import Sequence, NFKC, Strip
    except ImportError:
        raise ImportError("pip install tokenizers")

    # ── Special tokens ───────────────────────────────────────
    SPECIAL_TOKENS = [
        "[PAD]",    # 0  padding
        "[UNK]",    # 1  unknown
        "[BOS]",    # 2  beginning of sequence
        "[EOS]",    # 3  end of sequence
        "[SEP]",    # 4  separator (user / assistant boundary)
        "[MASK]",   # 5  masked token (for MLM if needed)
        # ── Conversation roles ────────────────────────────────
        "<|user|>",       # 6
        "<|assistant|>",  # 7
        "<|system|>",     # 8
        # ── ArthaSathi domain tokens (speed up financial reasoning)
        "[DEBT]",         # 9
        "[INCOME]",       # 10
        "[EXPENSE]",      # 11
        "[LOAN]",         # 12
        "[INTEREST]",     # 13
        "[TAX]",          # 14
        "[GST]",          # 15
        "[EMI]",          # 16
        "[AMOUNT]",       # 17
        "[LANG:HI]",      # 18 — Hindi
        "[LANG:EN]",      # 19 — English
        "[LANG:MR]",      # 20 — Marathi
        "[LANG:TA]",      # 21 — Tamil
        "[LANG:KN]",      # 22 — Kannada
        "[LANG:BHO]",     # 23 — Bhojpuri
        "[LANG:AS]",      # 24 — Assamese
        "[LANG:BN]",      # 25 — Bengali
        "[LANG:TE]",      # 26 — Telugu
        "[NEWLINE]",      # 27
    ]

    # ── Normalizer: NFKC unicode normalization ────────────────
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer   = Sequence([NFKC(), Strip()])
    tokenizer.pre_tokenizer = PTSequence([Whitespace(), Digits(individual_digits=True)])

    # ── Trainer ───────────────────────────────────────────────
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=[],   # learn alphabet from data
        continuing_subword_prefix="##",
    )

    print(f"Training BPE tokenizer on {len(data_files)} files  →  vocab_size={vocab_size}")
    tokenizer.train(data_files, trainer)

    # ── Post-processor: adds [BOS] and [EOS] automatically ───
    from tokenizers.processors import TemplateProcessing
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [SEP] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    # Save config
    config = {
        "vocab_size":       vocab_size,
        "special_tokens":   SPECIAL_TOKENS,
        "token_ids":        {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS},
        "pad_token":        "[PAD]",
        "unk_token":        "[UNK]",
        "bos_token":        "[BOS]",
        "eos_token":        "[EOS]",
        "languages":        ["hi", "en", "mr", "ta", "kn", "bho", "as", "bn", "te"],
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nTokenizer saved to {save_dir}/")
    _verify_tokenizer(tokenizer)
    return tokenizer


def _verify_tokenizer(tokenizer):
    """Quick quality check on tokenizer output"""
    test_cases = [
        ("Hindi",    "मेरा कर्ज 50000 रुपये है। मैं कैसे चुकाऊं?"),
        ("English",  "My loan is 50000 rupees. How do I repay?"),
        ("Marathi",  "माझ्या कर्जावर खूप व्याज आहे, काय करू?"),
        ("Tamil",    "என் கடன் 50000 ரூபாய். எப்படி திரும்பச் செலுத்துவது?"),
        ("Kannada",  "ನನ್ನ ಸಾಲ 50000 ರೂಪಾಯಿ. ಹೇಗೆ ತೀರಿಸಲಿ?"),
        ("Bhojpuri", "हमार कर्जा 50000 बा। काहे करी?"),
        ("Assamese", "মোৰ ঋণ 50000 টকা। কেনেকৈ পৰিশোধ কৰিম?"),
        ("Bengali",  "আমার ঋণ 50000 টাকা। কীভাবে পরিশোধ করব?"),
        ("Telugu",   "నా అప్పు 50000 రూపాయలు. ఎలా తీర్చాలి?"),
        ("Code-mix", "bhai mera 12000 ka loan hai aur 8000 salary hai"),
    ]
    print("\n─── Tokenizer Quality Check ───")
    for lang, text in test_cases:
        enc = tokenizer.encode(text)
        print(f"  {lang:10s}: {len(enc.tokens):3d} tokens  | {enc.tokens[:8]}...")
    print("─────────────────────────────────")


# ─────────────────────────────────────────────────────────────
# 2. WRAPPER CLASS (runtime usage)
# ─────────────────────────────────────────────────────────────

class ArthaSathiTokenizer:
    """
    Wrapper around the trained tokenizer with convenient encode/decode methods.
    Handles:
    - Text encoding/decoding
    - ChatML template formatting
    - Batch encoding with padding
    - Special token management
    """

    def __init__(self, tokenizer_dir: str):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError("pip install tokenizers")

        self.tokenizer = Tokenizer.from_file(
            os.path.join(tokenizer_dir, "tokenizer.json")
        )
        with open(os.path.join(tokenizer_dir, "config.json")) as f:
            self.config = json.load(f)

        # Cache special token IDs
        self.pad_id  = self.tokenizer.token_to_id("[PAD]")
        self.unk_id  = self.tokenizer.token_to_id("[UNK]")
        self.bos_id  = self.tokenizer.token_to_id("[BOS]")
        self.eos_id  = self.tokenizer.token_to_id("[EOS]")
        self.sep_id  = self.tokenizer.token_to_id("[SEP]")
        self.usr_id  = self.tokenizer.token_to_id("<|user|>")
        self.ast_id  = self.tokenizer.token_to_id("<|assistant|>")
        self.sys_id  = self.tokenizer.token_to_id("<|system|>")
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Enable padding
        self.tokenizer.enable_padding(
            pad_id=self.pad_id, pad_token="[PAD]"
        )

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode a single string to token IDs"""
        enc = self.tokenizer.encode(text, add_special_tokens=add_special)
        return enc.ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to string"""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    def encode_batch(self, texts: List[str], max_length: int = 1024,
                     padding: bool = True) -> Dict:
        """
        Encode a batch of strings with optional padding.
        Returns dict with input_ids and attention_mask.
        """
        self.tokenizer.enable_padding(
            length=max_length, pad_id=self.pad_id, pad_token="[PAD]"
        )
        self.tokenizer.enable_truncation(max_length=max_length)
        encodings = self.tokenizer.encode_batch(texts)
        return {
            "input_ids":      [e.ids for e in encodings],
            "attention_mask": [e.attention_mask for e in encodings],
        }

    def format_chat(self, messages: List[Dict],
                    system_prompt: Optional[str] = None,
                    language: str = "hi") -> str:
        """
        Format conversation in ChatML template for ArthaSathi.

        messages format: [{"role": "user"|"assistant", "content": "..."}]

        Returns formatted string ready for tokenization.
        """
        system_defaults = {
            "hi":  "Tum ArthaSathi ho — ek financial dost. Hamesha Hindi mein jawab do unless user English mein puchhe. Simple bhasha use karo. Sach batao.",
            "en":  "You are ArthaSathi — a financial companion. Give clear, honest, actionable advice. Speak like a trusted friend, not a bank.",
            "mr":  "तुम्ही ArthaSathi आहात — एक विश्वासू आर्थिक मित्र. मराठीत उत्तर द्या. सोपी भाषा वापरा.",
            "ta":  "நீங்கள் ArthaSathi — ஒரு நம்பகமான நிதி நண்பர். தமிழில் பதிலளிக்கவும்.",
            "kn":  "ನೀವು ArthaSathi — ನಿಮ್ಮ ಆರ್ಥಿಕ ಸ್ನೇಹಿತ. ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
            "bho": "Raho ArthaSathi — ek financial dost. Bhojpuri mein baat karo.",
            "as":  "আপুনি ArthaSathi — এজন বিশ্বাসযোগ্য আৰ্থিক বন্ধু। অসমীয়াত উত্তৰ দিয়ক।",
            "bn":  "তুমি ArthaSathi — একজন আর্থিক বন্ধু। বাংলায় উত্তর দাও।",
            "te":  "మీరు ArthaSathi — ఒక ఆర్థిక మిత్రుడు. తెలుగులో సమాధానం చెప్పండి.",
        }

        sys_text = system_prompt or system_defaults.get(language, system_defaults["en"])
        parts    = [f"<|system|>\n{sys_text}<|EOS|>"]

        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"<|user|>\n{content}<|EOS|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}<|EOS|>")

        # Add assistant prefix for generation
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def encode_chat(self, messages: List[Dict],
                    language: str = "hi",
                    max_length: int = 1024) -> List[int]:
        """Format and encode a full chat conversation"""
        formatted = self.format_chat(messages, language=language)
        ids       = self.encode(formatted, add_special=False)
        return ids[:max_length]

    def __len__(self) -> int:
        return self.vocab_size


# ─────────────────────────────────────────────────────────────
# 3. DATA DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────

def download_and_prepare_tokenizer_data(output_dir: str = "tokenizer_training_data",
                                        sample_gb_per_lang: float = 1.0):
    """
    Download open-source multilingual datasets for tokenizer training.
    Uses streaming to avoid RAM overflow.

    Datasets used:
    - IndicCorp v2 (AI4Bharat) — primary Indian language corpus
    - Wikipedia dumps
    - CC-100 multilingual CommonCrawl
    - mC4 (multilingual C4)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    os.makedirs(output_dir, exist_ok=True)

    # Language configs for each dataset
    INDICCORP_LANGS = {
        "hi":  "hin",  # Hindi
        "mr":  "mar",  # Marathi
        "ta":  "tam",  # Tamil
        "kn":  "kan",  # Kannada
        "as":  "asm",  # Assamese
        "bn":  "ben",  # Bengali
        "te":  "tel",  # Telugu
        "bho": "bho",  # Bhojpuri (limited coverage)
    }

    target_bytes = int(sample_gb_per_lang * 1024**3)

    for lang_code, ic_code in INDICCORP_LANGS.items():
        out_file = os.path.join(output_dir, f"{lang_code}.txt")
        if os.path.exists(out_file):
            print(f"  {lang_code}: already exists, skipping")
            continue

        print(f"\nDownloading IndicCorp v2 [{lang_code}] ...")
        written = 0
        try:
            ds = load_dataset(
                "ai4bharat/IndicCorp",
                ic_code,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            with open(out_file, "w", encoding="utf-8") as f:
                for example in ds:
                    text = example.get("text", "")
                    if len(text.strip()) < 50:
                        continue
                    f.write(text.strip() + "\n")
                    written += len(text.encode("utf-8"))
                    if written >= target_bytes:
                        break
            print(f"  {lang_code}: wrote {written / 1024**2:.1f} MB")
        except Exception as e:
            print(f"  {lang_code}: IndicCorp failed ({e}), trying Wikipedia...")
            _download_wikipedia(lang_code, out_file, target_bytes)

    # English — use a slice of Wikipedia
    en_file = os.path.join(output_dir, "en.txt")
    if not os.path.exists(en_file):
        print("\nDownloading English Wikipedia (sample)...")
        _download_wikipedia("en", en_file, target_bytes)

    all_files = [str(p) for p in Path(output_dir).glob("*.txt") if p.stat().st_size > 1000]
    print(f"\nTokenizer training data ready: {len(all_files)} files in {output_dir}/")
    return all_files


def _download_wikipedia(lang: str, out_file: str, max_bytes: int):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikipedia", f"20231101.{lang}",
                          split="train", streaming=True, trust_remote_code=True)
        written = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for ex in ds:
                text = ex.get("text", "")
                f.write(text + "\n")
                written += len(text.encode("utf-8"))
                if written >= max_bytes:
                    break
        print(f"  {lang} Wikipedia: {written/1024**2:.1f} MB")
    except Exception as e:
        print(f"  {lang} Wikipedia also failed: {e}")


# ─────────────────────────────────────────────────────────────
# 4. MAIN — TRAIN TOKENIZER END TO END
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="tokenizer_training_data")
    parser.add_argument("--save_dir",    default="arthasathi_tokenizer")
    parser.add_argument("--vocab_size",  type=int, default=60000)
    parser.add_argument("--sample_gb",  type=float, default=1.0,
                        help="GB per language for tokenizer training (1GB x 9 = 9GB total)")
    parser.add_argument("--download",   action="store_true",
                        help="Download datasets first")
    args = parser.parse_args()

    if args.download:
        data_files = download_and_prepare_tokenizer_data(
            args.data_dir, args.sample_gb
        )
    else:
        from pathlib import Path
        data_files = [str(p) for p in Path(args.data_dir).glob("*.txt")]
        if not data_files:
            raise FileNotFoundError(
                f"No .txt files in {args.data_dir}. Run with --download first."
            )

    tokenizer = train_arthasathi_tokenizer(
        data_files=data_files,
        save_dir=args.save_dir,
        vocab_size=args.vocab_size,
    )

    print("\n─── Testing trained tokenizer ───")
    tok = ArthaSathiTokenizer(args.save_dir)
    sample = "bhai mera 50000 ka loan hai, kya karna chahiye?"
    ids    = tok.encode(sample)
    back   = tok.decode(ids)
    print(f"Input : {sample}")
    print(f"Tokens: {len(ids)} → {ids[:10]}...")
    print(f"Decode: {back}")

    # Chat format test
    msgs = [
        {"role": "user",      "content": "Mera credit card ka 30000 debt hai. Kaise bharu?"},
        {"role": "assistant", "content": "Dekho bhai, pehle minimum payment karo..."},
        {"role": "user",      "content": "Aur agar paise nahi hain toh?"},
    ]
    chat_ids = tok.encode_chat(msgs, language="hi")
    print(f"\nChat encoded: {len(chat_ids)} tokens")
    print(f"Decoded:\n{tok.decode(chat_ids, skip_special=False)[:300]}...")
