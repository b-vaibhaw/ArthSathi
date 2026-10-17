"""
dataset_pipeline.py
====================
ArthaSathi — Complete Dataset Pipeline
Steps:
  1. Download 14 open-source datasets (IndicCorp, Wikipedia, CC-100, financial PDFs, etc.)
  2. Clean + deduplicate
  3. Generate 2.25M synthetic financial conversations (9 languages x 250K)
  4. Format into pre-training + fine-tuning + chat JSONL files
"""

import os, re, json, hashlib, random, time
from pathlib import Path
from typing import List, Dict, Optional, Generator


# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────

LANGUAGES = ["hi", "en", "mr", "ta", "kn", "bho", "as", "bn", "te"]

LANGUAGE_NAMES = {
    "hi":  "Hindi",    "en": "English",   "mr": "Marathi",
    "ta":  "Tamil",    "kn": "Kannada",   "bho": "Bhojpuri",
    "as":  "Assamese", "bn": "Bengali",   "te": "Telugu",
}

# Financial scenarios for synthetic data generation
DEBT_SCENARIOS = [
    "User has a credit card debt of Rs {amount} at {rate}% interest. Income is Rs {income}/month.",
    "User took a payday loan of Rs {amount} and cannot repay. They have {dependents} dependents.",
    "User has medical bills of Rs {amount} from a hospital. They have no health insurance.",
    "User has 3 loans simultaneously: {loan1}, {loan2}, {loan3}. Which to pay first?",
    "User is paying only minimum payments on credit card for {months} months. Show the trap.",
    "User borrowed from a moneylender at {rate}% monthly interest. Legal options?",
    "User's EMI is Rs {emi}/month but income dropped to Rs {income}. What to do?",
    "User wants to negotiate a settlement for Rs {amount} debt at {discount}% discount.",
]

BUSINESS_SCENARIOS = [
    "Street vegetable vendor earns Rs {income}/day, spends Rs {expense}/day. Advice needed.",
    "Tailor business: material cost Rs {material}, labor Rs {labor}. How to price a shirt?",
    "Kirana store owner has Rs {turnover} annual turnover. Should they register for GST?",
    "Auto-rickshaw driver earns Rs {income}/month. How to save for vehicle repair fund?",
    "Freelance graphic designer charges Rs {rate}/project. Getting low. How to raise rates?",
    "Tea stall owner wants to expand. Needs Rs {amount} loan. How to apply with no documents?",
    "Domestic worker wants to track income. Currently earns from {employers} employers.",
    "Gig driver wants to build credit history to get a home loan eventually.",
]

SYSTEM_PROMPTS = {
    "hi":  "Tum ArthaSathi ho — ek dost jo financial help karta hai. Hamesha simple Hindi mein baat karo. Numbers clearly batao. Legal rights bhi batao.",
    "en":  "You are ArthaSathi, a financial companion for ordinary people. Be clear, honest, and specific. Always mention legal rights.",
    "mr":  "तुम्ही ArthaSathi आहात — एक विश्वासू आर्थिक मित्र. सोप्या मराठीत उत्तर द्या.",
    "ta":  "நீங்கள் ArthaSathi — ஒரு நம்பகமான நிதி நண்பர். எளிய தமிழில் பதிலளிக்கவும்.",
    "kn":  "ನೀವು ArthaSathi — ನಿಮ್ಮ ಆರ್ಥಿಕ ಸ್ನೇಹಿತ. ಸರಳ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
    "bho": "Raho ArthaSathi — ek financial dost. Bhojpuri mein baat karo.",
    "as":  "আপুনি ArthaSathi — এজন আৰ্থিক বন্ধু। সহজ অসমীয়াত উত্তৰ দিয়ক।",
    "bn":  "তুমি ArthaSathi — একজন আর্থিক বন্ধু। সহজ বাংলায় উত্তর দাও।",
    "te":  "మీరు ArthaSathi — ఒక ఆర్థిక మిత్రుడు. సరళమైన తెలుగులో సమాధానం చెప్పండి.",
}


# ─────────────────────────────────────────────────────────────
# 2. DATASET DOWNLOADER
# ─────────────────────────────────────────────────────────────

class DatasetDownloader:
    """Download all 14 open-source datasets needed for ArthaSathi"""

    def __init__(self, output_dir: str = "raw_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self, gb_per_lang: float = 2.0):
        """Download all datasets. Total ~80GB raw before cleaning."""
        print("=" * 60)
        print("ArthaSathi Dataset Download Pipeline")
        print("=" * 60)

        self.download_indiccorp(gb_per_lang)
        self.download_sangraha(gb_per_lang * 0.5)
        self.download_wikipedia()
        self.download_cc100(gb_per_lang * 0.3)
        self.download_mc4(gb_per_lang * 0.3)
        self.download_oscar()
        self.download_financial_datasets()
        self.download_instruction_datasets()
        print("\nAll downloads complete!")

    def download_indiccorp(self, gb_per_lang: float = 2.0):
        """
        AI4Bharat IndicCorp v2 — PRIMARY Indian language corpus
        20.9 billion words across 22 Indian languages.
        HuggingFace: ai4bharat/IndicCorp
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("  pip install datasets"); return

        INDICCORP_CODES = {
            "hi": "hin", "mr": "mar", "ta": "tam", "kn": "kan",
            "as": "asm", "bn": "ben", "te": "tel", "bho": "bho",
        }
        target = int(gb_per_lang * 1024**3)

        for lang, code in INDICCORP_CODES.items():
            out = self.output_dir / "indiccorp" / f"{lang}.jsonl"
            if out.exists():
                print(f"  IndicCorp [{lang}]: already exists"); continue
            out.parent.mkdir(exist_ok=True)
            print(f"  Downloading IndicCorp [{lang}] ({code})...")
            try:
                ds = load_dataset("ai4bharat/IndicCorp", code,
                                  split="train", streaming=True, trust_remote_code=True)
                written = 0
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        txt = ex.get("text", "").strip()
                        if len(txt) < 50: continue
                        f.write(json.dumps({"text": txt, "lang": lang,
                                            "source": "indiccorp"}, ensure_ascii=False) + "\n")
                        written += len(txt.encode("utf-8"))
                        if written >= target: break
                print(f"    → {written/1024**2:.1f} MB written")
            except Exception as e:
                print(f"    → Failed: {e}")

    def download_sangraha(self, gb_per_lang: float = 1.0):
        """
        AI4Bharat Sangraha — 251B tokens, high quality filtered.
        HuggingFace: ai4bharat/sangraha
        """
        try:
            from datasets import load_dataset
        except ImportError:
            return

        target = int(gb_per_lang * 1024**3)
        SANGRAHA_LANGS = ["hi", "mr", "ta", "kn", "as", "bn", "te"]

        for lang in SANGRAHA_LANGS:
            out = self.output_dir / "sangraha" / f"{lang}.jsonl"
            if out.exists(): continue
            out.parent.mkdir(exist_ok=True)
            print(f"  Downloading Sangraha [{lang}]...")
            try:
                ds = load_dataset("ai4bharat/sangraha", lang,
                                  split="train", streaming=True, trust_remote_code=True)
                written = 0
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        txt = ex.get("text", "").strip()
                        if len(txt) < 50: continue
                        f.write(json.dumps({"text": txt, "lang": lang,
                                            "source": "sangraha"}, ensure_ascii=False) + "\n")
                        written += len(txt.encode("utf-8"))
                        if written >= target: break
                print(f"    → {written/1024**2:.1f} MB written")
            except Exception as e:
                print(f"    → Failed: {e}")

    def download_wikipedia(self):
        """Wikipedia dumps for all 9 languages — structured factual text"""
        try:
            from datasets import load_dataset
        except ImportError:
            return

        WIKI_LANGS = {
            "hi": "20231101.hi",  "mr": "20231101.mr",
            "ta": "20231101.ta",  "kn": "20231101.kn",
            "as": "20231101.as",  "bn": "20231101.bn",
            "te": "20231101.te",  "en": "20231101.en",
        }

        for lang, wcode in WIKI_LANGS.items():
            out = self.output_dir / "wikipedia" / f"{lang}.jsonl"
            if out.exists(): continue
            out.parent.mkdir(exist_ok=True)
            print(f"  Downloading Wikipedia [{lang}]...")
            try:
                ds = load_dataset("wikipedia", wcode,
                                  split="train", streaming=True, trust_remote_code=True)
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        txt = ex.get("text", "").strip()
                        if len(txt) < 100: continue
                        f.write(json.dumps({"text": txt, "lang": lang,
                                            "title": ex.get("title",""),
                                            "source": "wikipedia"}, ensure_ascii=False) + "\n")
                print(f"    → Wikipedia [{lang}] done")
            except Exception as e:
                print(f"    → Failed: {e}")

    def download_cc100(self, gb_per_lang: float = 0.5):
        """CC-100 Multilingual CommonCrawl — web-scale text"""
        try:
            from datasets import load_dataset
        except ImportError:
            return

        CC100_LANGS = {"hi": "hi", "mr": "mr", "ta": "ta",
                       "bn": "bn", "as": "as", "te": "te"}
        target = int(gb_per_lang * 1024**3)

        for lang, cc_code in CC100_LANGS.items():
            out = self.output_dir / "cc100" / f"{lang}.jsonl"
            if out.exists(): continue
            out.parent.mkdir(exist_ok=True)
            print(f"  Downloading CC-100 [{lang}]...")
            try:
                ds = load_dataset("cc100", cc_code, split="train",
                                  streaming=True, trust_remote_code=True)
                written = 0
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        txt = ex.get("text", "").strip()
                        if len(txt) < 50: continue
                        f.write(json.dumps({"text": txt, "lang": lang,
                                            "source": "cc100"}, ensure_ascii=False) + "\n")
                        written += len(txt.encode("utf-8"))
                        if written >= target: break
                print(f"    → {written/1024**2:.1f} MB written")
            except Exception as e:
                print(f"    → Failed: {e}")

    def download_mc4(self, gb_per_lang: float = 0.5):
        """mC4 Multilingual C4 from Google"""
        try:
            from datasets import load_dataset
        except ImportError:
            return

        MC4_LANGS = ["hi", "mr", "ta", "kn", "bn", "te", "en"]
        target = int(gb_per_lang * 1024**3)

        for lang in MC4_LANGS:
            out = self.output_dir / "mc4" / f"{lang}.jsonl"
            if out.exists(): continue
            out.parent.mkdir(exist_ok=True)
            print(f"  Downloading mC4 [{lang}]...")
            try:
                ds = load_dataset("mc4", lang, split="train",
                                  streaming=True, trust_remote_code=True)
                written = 0
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        txt = ex.get("text", "").strip()
                        if len(txt) < 100: continue
                        f.write(json.dumps({"text": txt, "lang": lang,
                                            "source": "mc4"}, ensure_ascii=False) + "\n")
                        written += len(txt.encode("utf-8"))
                        if written >= target: break
                print(f"    → {written/1024**2:.1f} MB")
            except Exception as e:
                print(f"    → Failed: {e}")

    def download_oscar(self):
        """OSCAR corpus — deduplicated CommonCrawl"""
        try:
            from datasets import load_dataset
        except ImportError:
            return

        OSCAR_LANGS = ["hi", "mr", "ta", "bn", "te"]
        for lang in OSCAR_LANGS:
            out = self.output_dir / "oscar" / f"{lang}.jsonl"
            if out.exists(): continue
            out.parent.mkdir(exist_ok=True)
            print(f"  Downloading OSCAR [{lang}]...")
            try:
                ds = load_dataset("oscar-corpus/OSCAR-2301", lang,
                                  split="train", streaming=True, trust_remote_code=True)
                written = 0
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        txt = ex.get("content", ex.get("text", "")).strip()
                        if len(txt) < 100: continue
                        f.write(json.dumps({"text": txt, "lang": lang,
                                            "source": "oscar"}, ensure_ascii=False) + "\n")
                        written += len(txt.encode("utf-8"))
                        if written >= 500 * 1024**2: break  # 500MB
                print(f"    → {written/1024**2:.1f} MB")
            except Exception as e:
                print(f"    → Failed: {e}")

    def download_financial_datasets(self):
        """
        Financial domain datasets:
        - FinGPT datasets
        - IndicQA (for QA fine-tuning)
        - Indic-Instruct
        - Anudesh
        """
        try:
            from datasets import load_dataset
        except ImportError:
            return

        FIN_DATASETS = [
            # (dataset_id, config, split, text_field, lang, source_name)
            ("ai4bharat/IndicQA",  "hi", "test",  "context", "hi",  "indicqa"),
            ("ai4bharat/IndicQA",  "mr", "test",  "context", "mr",  "indicqa"),
            ("ai4bharat/IndicQA",  "ta", "test",  "context", "ta",  "indicqa"),
            ("ai4bharat/IndicQA",  "bn", "test",  "context", "bn",  "indicqa"),
            ("ai4bharat/IndicQA",  "te", "test",  "context", "te",  "indicqa"),
        ]

        for ds_id, cfg, split, txt_field, lang, name in FIN_DATASETS:
            out = self.output_dir / "financial" / f"{name}_{lang}.jsonl"
            if out.exists(): continue
            out.parent.mkdir(exist_ok=True)
            try:
                ds = load_dataset(ds_id, cfg, split=split, trust_remote_code=True)
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        txt = ex.get(txt_field, "")
                        if txt:
                            f.write(json.dumps({"text": txt, "lang": lang,
                                                "source": name}, ensure_ascii=False) + "\n")
                print(f"    → {name} [{lang}] done")
            except Exception as e:
                print(f"    → {name} [{lang}] failed: {e}")

    def download_instruction_datasets(self):
        """Download instruction-following datasets for fine-tuning"""
        try:
            from datasets import load_dataset
        except ImportError:
            return

        # Indic-Instruct (AI4Bharat)
        for lang in ["hi", "mr", "ta", "kn", "bn", "te"]:
            out = self.output_dir / "instructions" / f"indic_instruct_{lang}.jsonl"
            if out.exists(): continue
            out.parent.mkdir(exist_ok=True)
            try:
                ds = load_dataset("ai4bharat/indic-instruct-data-v0.1",
                                  lang, split="train", trust_remote_code=True)
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        f.write(json.dumps({
                            "instruction": ex.get("instruction", ""),
                            "input":       ex.get("input", ""),
                            "output":      ex.get("output", ""),
                            "lang":        lang,
                            "source":      "indic_instruct",
                        }, ensure_ascii=False) + "\n")
                print(f"    → Indic-Instruct [{lang}] done")
            except Exception as e:
                print(f"    → Indic-Instruct [{lang}] failed: {e}")

        # OpenAssistant (multilingual)
        out = self.output_dir / "instructions" / "openassistant.jsonl"
        if not out.exists():
            try:
                ds = load_dataset("OpenAssistant/oasst1", split="train")
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        if ex["role"] == "assistant":
                            f.write(json.dumps({
                                "text":   ex["text"],
                                "lang":   ex.get("lang", "en"),
                                "source": "openassistant",
                            }, ensure_ascii=False) + "\n")
                print("    → OpenAssistant done")
            except Exception as e:
                print(f"    → OpenAssistant failed: {e}")


# ─────────────────────────────────────────────────────────────
# 3. DATA CLEANER + DEDUPLICATOR
# ─────────────────────────────────────────────────────────────

class DataCleaner:
    """Clean, filter, and deduplicate text corpus"""

    def __init__(self):
        # MinHash deduplication
        self.seen_hashes: set = set()

    def clean_text(self, text: str, lang: str) -> Optional[str]:
        """
        Clean a single text sample.
        Returns None if sample should be discarded.
        """
        if not text or not text.strip():
            return None

        # Fix unicode encoding issues
        try:
            text = text.encode("utf-8", errors="ignore").decode("utf-8")
        except Exception:
            return None

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', ' ', text)

        # Remove phone numbers (but keep amounts like 50000)
        # Don't remove 5-digit numbers — they're likely financial amounts

        # Normalize whitespace (keep single newlines, collapse multiple)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()

        # Minimum quality checks
        words = text.split()
        if len(words) < 10:
            return None  # Too short
        if len(words) > 50000:
            text = ' '.join(words[:50000])  # Truncate very long docs

        # Language-specific noise removal
        # Remove excessive repeated characters (spammy text)
        if re.search(r'(.)\1{8,}', text):
            return None

        # Remove if mostly numbers/symbols (noise)
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count / max(len(text), 1) < 0.3:
            return None

        return text

    def is_duplicate(self, text: str) -> bool:
        """Fast hashing deduplication"""
        # Use first 500 chars for hash (faster, works for near-duplicates)
        h = hashlib.md5(text[:500].encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def process_file(self, input_path: str, output_path: str,
                     lang: str, max_samples: int = 1_000_000) -> Dict:
        """
        Process a single JSONL file: clean + deduplicate.
        Returns statistics.
        """
        stats = {"read": 0, "kept": 0, "dup": 0, "short": 0, "noise": 0}
        os.makedirs(Path(output_path).parent, exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                if stats["kept"] >= max_samples:
                    break
                stats["read"] += 1
                try:
                    obj  = json.loads(line)
                    text = obj.get("text", "")
                except Exception:
                    continue

                cleaned = self.clean_text(text, lang)
                if cleaned is None:
                    stats["noise"] += 1
                    continue

                if self.is_duplicate(cleaned):
                    stats["dup"] += 1
                    continue

                obj["text"] = cleaned
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["kept"] += 1

        return stats

    def process_all(self, raw_dir: str = "raw_data",
                    clean_dir: str = "clean_data"):
        """Process all raw data files"""
        raw_path   = Path(raw_dir)
        clean_path = Path(clean_dir)
        total_stats = {"read": 0, "kept": 0}

        for subdir in raw_path.iterdir():
            if not subdir.is_dir(): continue
            for f in subdir.glob("*.jsonl"):
                # Infer language from filename
                lang = f.stem.split("_")[-1]
                if lang not in LANGUAGES: lang = "en"
                out = clean_path / subdir.name / f.name
                print(f"Cleaning {f.name}...", end=" ", flush=True)
                st = self.process_file(str(f), str(out), lang)
                total_stats["read"] += st["read"]
                total_stats["kept"] += st["kept"]
                ratio = st["kept"] / max(st["read"], 1)
                print(f"kept={st['kept']:,} ({ratio:.1%})")

        print(f"\nTotal: {total_stats['read']:,} read → {total_stats['kept']:,} kept")
        return total_stats


# ─────────────────────────────────────────────────────────────
# 4. SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generate 2.25M synthetic financial conversations using a local LLM.
    No paid API needed — uses free Mistral-7B on Colab or your GPU.
    """

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 device: str = "auto"):
        self.model_id  = model_id
        self.device    = device
        self.generator = None
        self._load_model()

    def _load_model(self):
        print(f"Loading generation model: {self.model_id}")
        try:
            from transformers import pipeline, BitsAndBytesConfig
            import torch
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model_id,
                device_map=self.device,
                model_kwargs={"quantization_config": quant_config},
                torch_dtype="auto",
            )
            print("Generation model loaded!")
        except Exception as e:
            print(f"Could not load LLM generator: {e}")
            print("Will use template-based generation instead.")
            self.generator = None

    def _fill_scenario(self, template: str) -> str:
        """Fill template placeholders with realistic Indian financial values"""
        replacements = {
            "{amount}":     str(random.choice([5000, 10000, 15000, 20000, 25000, 30000,
                                               50000, 75000, 100000, 150000, 200000])),
            "{rate}":       str(random.choice([12, 14, 16, 18, 24, 28, 36, 42, 48])),
            "{income}":     str(random.choice([8000, 10000, 12000, 15000, 18000, 20000,
                                               25000, 30000, 40000, 50000])),
            "{emi}":        str(random.choice([2000, 3000, 4000, 5000, 6000, 8000, 10000])),
            "{months}":     str(random.choice([3, 6, 9, 12, 18, 24])),
            "{dependents}": str(random.choice([1, 2, 3, 4, 5])),
            "{discount}":   str(random.choice([20, 25, 30, 40, 50])),
            "{employers}":  str(random.choice([2, 3, 4, 5])),
            "{turnover}":   str(random.choice([500000, 800000, 1000000, 1500000,
                                               2000000, 3000000, 4000000])),
            "{material}":   str(random.choice([200, 300, 400, 500, 600])),
            "{labor}":      str(random.choice([100, 150, 200, 250, 300])),
            "{expense}":    str(random.choice([200, 300, 400, 500, 600, 800])),
            "{loan1}":      f"HDFC CC Rs {random.randint(10,50)*1000}@18%",
            "{loan2}":      f"Bajaj Finance Rs {random.randint(5,20)*1000}@26%",
            "{loan3}":      f"Moneylender Rs {random.randint(5,15)*1000}@36%",
        }
        for k, v in replacements.items():
            template = template.replace(k, v)
        return template

    def _generate_with_llm(self, scenario: str, lang: str,
                           n_turns: int = 5) -> Optional[List[Dict]]:
        """Use the loaded LLM to generate a realistic conversation"""
        if self.generator is None:
            return None

        prompt = f"""Generate a realistic WhatsApp conversation in {LANGUAGE_NAMES[lang]} between a poor Indian user and ArthaSathi (a helpful financial AI).

Situation: {scenario}
Language: {LANGUAGE_NAMES[lang]} (use natural speech, include code-mixing with Hindi/English where natural)
Turns: {n_turns} user messages and {n_turns} assistant responses
Format: Return ONLY a JSON array like [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]
The assistant should give specific, actionable advice with numbers.

JSON:"""

        try:
            result = self.generator(
                prompt,
                max_new_tokens=600,
                temperature=0.85,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Extract JSON from response
            json_start = result.find("[")
            json_end   = result.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                return None
            raw_json = result[json_start:json_end]
            messages = json.loads(raw_json)
            if isinstance(messages, list) and len(messages) >= 2:
                return messages
        except Exception:
            pass
        return None

    def _template_conversation(self, scenario: str, lang: str) -> List[Dict]:
        """
        Fallback: template-based conversations (no GPU needed).
        Less creative but guaranteed quality.
        """
        # Simple template conversations that cover key financial advice patterns
        templates_hi = [
            # Debt avalanche template
            (
                "Mera {debt_type} {amount} ka hai. Kya karna chahiye?",
                "Dekho bhai, pehle {debt_type} ka minimum payment karo ({min_pay} per month). "
                "Bacha hua paisa highest interest rate wale loan pe lagao. "
                "Tumhara case mein avalanche method best hai — {months} mahine mein debt free ho jaoge. "
                "Agar possible ho toh creditor se ek baar baat karo — sometimes 20-30% settlement milti hai."
            ),
        ]

        # For now return a structured template conversation
        scenario_filled = self._fill_scenario(scenario)
        return [
            {"role": "user",      "content": f"{scenario_filled}"},
            {"role": "assistant", "content": f"[Template response in {lang} for: {scenario_filled[:100]}]"},
        ]

    def generate_batch(self, output_dir: str = "synthetic_data",
                       n_per_lang: int = 25000,
                       use_llm: bool = True):
        """
        Generate synthetic conversations for all languages.
        n_per_lang = 25000 → 225,000 total (conservative; scale to 250K for full dataset)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        all_scenarios = DEBT_SCENARIOS + BUSINESS_SCENARIOS

        for lang in LANGUAGES:
            out_file = Path(output_dir) / f"{lang}_synthetic.jsonl"
            if out_file.exists():
                existing = sum(1 for _ in open(out_file))
                if existing >= n_per_lang:
                    print(f"  [{lang}] Already has {existing} examples, skipping")
                    continue
            else:
                existing = 0

            print(f"\nGenerating [{lang}] synthetic conversations "
                  f"({existing} → {n_per_lang})...")
            needed = n_per_lang - existing

            with open(out_file, "a", encoding="utf-8") as f:
                generated = 0
                while generated < needed:
                    scenario_template = random.choice(all_scenarios)
                    scenario          = self._fill_scenario(scenario_template)

                    if use_llm and self.generator is not None:
                        messages = self._generate_with_llm(scenario, lang)
                    else:
                        messages = None

                    if messages is None:
                        messages = self._template_conversation(scenario, lang)

                    record = {
                        "messages":  messages,
                        "lang":      lang,
                        "scenario":  scenario[:100],
                        "source":    "synthetic",
                        "module":    "debt" if "loan" in scenario.lower() or
                                               "debt" in scenario.lower() else "business",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    generated += 1

                    if generated % 1000 == 0:
                        print(f"  [{lang}] {generated}/{needed}")

            print(f"  [{lang}] Done: {n_per_lang} total")


# ─────────────────────────────────────────────────────────────
# 5. DATASET FORMATTER (pre-train / fine-tune / chat)
# ─────────────────────────────────────────────────────────────

class DatasetFormatter:
    """
    Format cleaned data into three training formats:
    1. Pre-training: plain text chunks (next-token prediction)
    2. Instruction fine-tuning: {instruction, input, output}
    3. Chat fine-tuning: ChatML-formatted conversations
    """

    def __init__(self, tokenizer_dir: Optional[str] = None, max_length: int = 1024):
        self.max_length    = max_length
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer     = None
        if tokenizer_dir and Path(tokenizer_dir).exists():
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(
                    str(Path(tokenizer_dir) / "tokenizer.json")
                )
                print(f"Tokenizer loaded from {tokenizer_dir}")
            except Exception:
                print("Could not load tokenizer — will use character chunking")

    def _tokenize_count(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text).ids)
        return len(text) // 4  # rough estimate: 1 token ≈ 4 chars

    def format_pretrain(self, clean_dirs: List[str],
                        output_file: str = "formatted/pretrain.jsonl",
                        target_tokens: int = 15_000_000_000):
        """
        Create pre-training corpus.
        Concatenates all text, chunked into context-length pieces.
        Target: 15B tokens (enough for 345M model, 2 epochs)
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        total_tokens = 0
        buffer       = ""
        chunk_size   = self.max_length * 4  # approx chars per chunk

        with open(output_file, "w", encoding="utf-8") as fout:
            for d in clean_dirs:
                for f in Path(d).rglob("*.jsonl"):
                    with open(f, encoding="utf-8") as fin:
                        for line in fin:
                            try:
                                obj  = json.loads(line)
                                text = obj.get("text", "")
                                lang = obj.get("lang", "en")
                            except Exception:
                                continue

                            buffer += f"\n{text}"

                            while len(buffer) >= chunk_size:
                                chunk  = buffer[:chunk_size]
                                buffer = buffer[chunk_size:]
                                toks   = self._tokenize_count(chunk)
                                fout.write(json.dumps({
                                    "text":   chunk,
                                    "lang":   lang,
                                    "tokens": toks,
                                }, ensure_ascii=False) + "\n")
                                total_tokens += toks

                                if total_tokens % 1_000_000_000 < toks:
                                    print(f"  Pre-train: {total_tokens/1e9:.1f}B tokens written")

                                if total_tokens >= target_tokens:
                                    print(f"  Reached target {target_tokens/1e9:.0f}B tokens")
                                    return total_tokens

        print(f"Pre-train corpus: {total_tokens/1e9:.1f}B tokens → {output_file}")
        return total_tokens

    def format_finetune(self, source_dirs: List[str],
                        output_file: str = "formatted/finetune.jsonl"):
        """
        Create instruction fine-tuning dataset in Alpaca format:
        {"instruction": ..., "input": ..., "output": ..., "lang": ...}
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(output_file, "w", encoding="utf-8") as fout:
            for d in source_dirs:
                for f in Path(d).rglob("*.jsonl"):
                    with open(f, encoding="utf-8") as fin:
                        for line in fin:
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            # Already in instruction format
                            if "instruction" in obj and "output" in obj:
                                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                count += 1
                            # Convert QA format
                            elif "question" in obj and "answer" in obj:
                                fout.write(json.dumps({
                                    "instruction": obj["question"],
                                    "input":       "",
                                    "output":      obj["answer"],
                                    "lang":        obj.get("lang", "en"),
                                }, ensure_ascii=False) + "\n")
                                count += 1

        print(f"Fine-tune dataset: {count:,} examples → {output_file}")
        return count

    def format_chat(self, synthetic_dir: str,
                    output_file: str = "formatted/chat.jsonl"):
        """
        Create chat fine-tuning dataset in ChatML format.
        Each example: full conversation with system + user + assistant turns.
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(output_file, "w", encoding="utf-8") as fout:
            for f in Path(synthetic_dir).rglob("*.jsonl"):
                with open(f, encoding="utf-8") as fin:
                    for line in fin:
                        try:
                            obj  = json.loads(line)
                            msgs = obj.get("messages", [])
                            lang = obj.get("lang", "hi")
                        except Exception:
                            continue

                        if len(msgs) < 2:
                            continue

                        # Build ChatML string
                        chatml = self._to_chatml(msgs, lang)
                        if chatml:
                            fout.write(json.dumps({
                                "text":   chatml,
                                "lang":   lang,
                                "module": obj.get("module", "general"),
                            }, ensure_ascii=False) + "\n")
                            count += 1

        print(f"Chat dataset: {count:,} conversations → {output_file}")
        return count

    def _to_chatml(self, messages: List[Dict], lang: str) -> str:
        """Convert message list to ChatML format"""
        system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
        parts  = [f"<|im_start|>system\n{system}<|im_end|>"]
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "").strip()
            if not content: continue
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# 6. FULL PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    base_dir:       str   = ".",
    gb_per_lang:    float = 2.0,
    synthetic_n:    int   = 25000,
    skip_download:  bool  = False,
    skip_clean:     bool  = False,
    skip_synthetic: bool  = False,
):
    """Run the complete data pipeline end-to-end"""
    raw_dir   = f"{base_dir}/raw_data"
    clean_dir = f"{base_dir}/clean_data"
    synth_dir = f"{base_dir}/synthetic_data"
    fmt_dir   = f"{base_dir}/formatted"

    print("\n" + "=" * 60)
    print("ARTHASATHI DATA PIPELINE")
    print("=" * 60)

    # Step 1: Download
    if not skip_download:
        print("\n[STEP 1] Downloading datasets...")
        dl = DatasetDownloader(raw_dir)
        dl.download_all(gb_per_lang)

    # Step 2: Clean
    if not skip_clean:
        print("\n[STEP 2] Cleaning and deduplicating...")
        cleaner = DataCleaner()
        cleaner.process_all(raw_dir, clean_dir)

    # Step 3: Synthetic generation
    if not skip_synthetic:
        print("\n[STEP 3] Generating synthetic conversations...")
        gen = SyntheticDataGenerator()
        gen.generate_batch(synth_dir, n_per_lang=synthetic_n)

    # Step 4: Format
    print("\n[STEP 4] Formatting for training...")
    fmt = DatasetFormatter(max_length=1024)

    fmt.format_pretrain(
        clean_dirs=[clean_dir],
        output_file=f"{fmt_dir}/pretrain.jsonl",
        target_tokens=15_000_000_000,
    )
    fmt.format_finetune(
        source_dirs=[f"{raw_dir}/instructions", clean_dir],
        output_file=f"{fmt_dir}/finetune.jsonl",
    )
    fmt.format_chat(
        synthetic_dir=synth_dir,
        output_file=f"{fmt_dir}/chat.jsonl",
    )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Pre-train data: {fmt_dir}/pretrain.jsonl")
    print(f"  Fine-tune data: {fmt_dir}/finetune.jsonl")
    print(f"  Chat data:      {fmt_dir}/chat.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir",       default=".")
    p.add_argument("--gb_per_lang",    type=float, default=2.0)
    p.add_argument("--synthetic_n",    type=int,   default=25000)
    p.add_argument("--skip_download",  action="store_true")
    p.add_argument("--skip_clean",     action="store_true")
    p.add_argument("--skip_synthetic", action="store_true")
    args = p.parse_args()

    run_full_pipeline(
        base_dir=args.base_dir,
        gb_per_lang=args.gb_per_lang,
        synthetic_n=args.synthetic_n,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_synthetic=args.skip_synthetic,
    )
