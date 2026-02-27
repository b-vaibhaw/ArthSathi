"""
voice_pipeline.py
==================
ArthaSathi — Voice Note Processing Pipeline
WhatsApp voice notes (OGG/MP4) → Transcription (Whisper) → Text
Text → Speech (gTTS / Coqui TTS) → Audio reply
Supports: Hindi, English, Marathi, Tamil, Kannada, Bhojpuri,
          Assamese, Bengali, Telugu
"""

import os, io, json, tempfile, subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────
# 1. LANGUAGE CONFIG
# ─────────────────────────────────────────────────────────────

LANG_CONFIG = {
    "hi":  {"whisper": "hi",  "gtts": "hi",  "name": "Hindi"},
    "en":  {"whisper": "en",  "gtts": "en",  "name": "English"},
    "mr":  {"whisper": "mr",  "gtts": "mr",  "name": "Marathi"},
    "ta":  {"whisper": "ta",  "gtts": "ta",  "name": "Tamil"},
    "kn":  {"whisper": "kn",  "gtts": "kn",  "name": "Kannada"},
    "bn":  {"whisper": "bn",  "gtts": "bn",  "name": "Bengali"},
    "te":  {"whisper": "te",  "gtts": "te",  "name": "Telugu"},
    # Bhojpuri and Assamese — approximate using nearest supported language
    "bho": {"whisper": "hi",  "gtts": "hi",  "name": "Bhojpuri"},
    "as":  {"whisper": "as",  "gtts": "bn",  "name": "Assamese"},
}

# WhatsApp sends voice notes in OGG/OPUS format
SUPPORTED_AUDIO_FORMATS = [".ogg", ".oga", ".opus", ".mp4", ".m4a", ".wav", ".mp3"]


# ─────────────────────────────────────────────────────────────
# 2. AUDIO CONVERSION
# ─────────────────────────────────────────────────────────────

def convert_audio_to_wav(input_bytes: bytes, input_format: str = "ogg") -> bytes:
    """
    Convert any audio format to WAV 16kHz mono (required by Whisper).
    Uses ffmpeg — must be installed: apt-get install ffmpeg
    Falls back to raw bytes if ffmpeg not available.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{input_format}", delete=False) as tmp_in:
            tmp_in.write(input_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path.replace(f".{input_format}", ".wav")

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path,
             "-ar", "16000",   # 16kHz sample rate (Whisper requirement)
             "-ac", "1",       # Mono
             "-f", "wav",
             tmp_out_path],
            capture_output=True, timeout=30
        )

        if result.returncode == 0 and Path(tmp_out_path).exists():
            with open(tmp_out_path, "rb") as f:
                wav_bytes = f.read()
            return wav_bytes
        else:
            print(f"[Voice] ffmpeg conversion failed: {result.stderr.decode()[:200]}")
            return input_bytes

    except FileNotFoundError:
        print("[Voice] ffmpeg not found. Install: apt-get install ffmpeg")
        return input_bytes
    except Exception as e:
        print(f"[Voice] Audio conversion error: {e}")
        return input_bytes
    finally:
        for p in [tmp_in_path, tmp_out_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# 3. SPEECH-TO-TEXT (Whisper)
# ─────────────────────────────────────────────────────────────

class WhisperSTT:
    """
    Whisper-based speech-to-text for all 9 Indian languages.

    Model sizes:
      tiny   — fastest, lowest quality, use for Colab T4
      small  — good balance (recommended)
      medium — best accuracy, use if you have A100 GPU
      large  — best, needs 10GB VRAM

    Fine-tuning datasets for Indian languages:
      - Mozilla Common Voice 15 (hi, mr, ta, bn, te, kn)
      - OpenSLR Indian datasets (resources 53, 64, 65, 66, 103)
      - AI4Bharat IndicSUPERB (12 Indian languages)
      - MUCS 2021 (multilingual + code-switching)
    """

    MODEL_SIZES = {"tiny": 39, "small": 244, "medium": 769, "large": 1550}

    def __init__(self, model_size: str = "small",
                 finetuned_path: Optional[str] = None,
                 device: str = "auto"):
        self.model_size = model_size
        self.model      = None
        self.device     = self._resolve_device(device)
        self._load_model(finetuned_path)

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, finetuned_path: Optional[str]):
        try:
            import whisper
            if finetuned_path and Path(finetuned_path).exists():
                # Load fine-tuned checkpoint
                self.model = whisper.load_model(self.model_size, device=self.device)
                import torch
                ckpt = torch.load(finetuned_path, map_location=self.device)
                self.model.load_state_dict(ckpt)
                print(f"[Voice/STT] Loaded fine-tuned Whisper from {finetuned_path}")
            else:
                self.model = whisper.load_model(self.model_size, device=self.device)
                print(f"[Voice/STT] Loaded Whisper-{self.model_size} "
                      f"({self.MODEL_SIZES[self.model_size]}M params) on {self.device}")
        except ImportError:
            print("[Voice/STT] Whisper not installed. pip install openai-whisper")

    def detect_language(self, audio_bytes: bytes) -> Tuple[str, float]:
        """
        Auto-detect language from audio.
        Returns (language_code, confidence).
        """
        if self.model is None:
            return "hi", 0.5

        try:
            import whisper
            import numpy as np

            wav = convert_audio_to_wav(audio_bytes)
            audio = whisper.load_audio(io.BytesIO(wav))
            audio = whisper.pad_or_trim(audio)
            mel   = whisper.log_mel_spectrogram(audio).to(self.device)

            _, probs = self.model.detect_language(mel)
            lang     = max(probs, key=probs.get)
            conf     = probs[lang]

            # Map to our supported langs
            lang_map = {"hi": "hi", "en": "en", "mr": "mr", "ta": "ta",
                        "kn": "kn", "bn": "bn", "te": "te", "as": "as"}
            lang = lang_map.get(lang, "hi")
            return lang, float(conf)

        except Exception as e:
            print(f"[Voice/STT] Language detection failed: {e}")
            return "hi", 0.5

    def transcribe(self, audio_bytes: bytes,
                   language: Optional[str] = None,
                   auto_detect: bool = True) -> Dict:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes : raw audio bytes (any format, will be converted)
            language    : force a specific language code, or None for auto
            auto_detect : if True and language is None, detect language first

        Returns:
            dict with: text, language, confidence, segments
        """
        if self.model is None:
            return {"text": "", "language": language or "hi",
                    "confidence": 0.0, "error": "Whisper model not loaded"}

        try:
            import whisper

            # Convert to WAV
            wav_bytes = convert_audio_to_wav(audio_bytes)

            # Detect language if not specified
            detected_lang = language
            lang_confidence = 1.0
            if auto_detect and language is None:
                detected_lang, lang_confidence = self.detect_language(audio_bytes)

            # Whisper language code
            whisper_lang = LANG_CONFIG.get(detected_lang, {}).get("whisper", "hi")

            # Transcribe
            audio  = whisper.load_audio(io.BytesIO(wav_bytes))
            result = self.model.transcribe(
                audio,
                language=whisper_lang,
                task="transcribe",
                fp16=(self.device == "cuda"),
                verbose=False,
            )

            text = result.get("text", "").strip()

            return {
                "text":        text,
                "language":    detected_lang,
                "confidence":  lang_confidence,
                "segments":    result.get("segments", []),
                "duration_s":  result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0,
            }

        except Exception as e:
            print(f"[Voice/STT] Transcription failed: {e}")
            return {"text": "", "language": language or "hi",
                    "confidence": 0.0, "error": str(e)}


# ─────────────────────────────────────────────────────────────
# 4. TEXT-TO-SPEECH
# ─────────────────────────────────────────────────────────────

class IndianTTS:
    """
    Text-to-speech for 9 Indian languages.
    Primary: gTTS (Google TTS) — free, good quality for all languages
    Fallback: Coqui TTS (local, offline capable)
    For Bhojpuri/Assamese: uses nearest supported language (Hindi/Bengali)
    """

    def __init__(self, engine: str = "gtts"):
        self.engine = engine
        self._verify()

    def _verify(self):
        if self.engine == "gtts":
            try:
                import gtts
                print("[Voice/TTS] gTTS ready")
            except ImportError:
                print("[Voice/TTS] gTTS not installed. pip install gtts")
                self.engine = "none"

    def synthesize(self, text: str, language: str = "hi",
                   slow: bool = False) -> Optional[bytes]:
        """
        Convert text to speech audio bytes (MP3 format).

        Args:
            text     : text to synthesize
            language : language code (hi, en, mr, ta, kn, bn, te, bho, as)
            slow     : speak slowly (helpful for clarity)

        Returns:
            MP3 audio bytes, or None on failure
        """
        if not text.strip():
            return None

        # Get gTTS language code
        gtts_lang = LANG_CONFIG.get(language, {}).get("gtts", "hi")

        if self.engine == "gtts":
            return self._gtts_synthesize(text, gtts_lang, slow)
        return None

    def _gtts_synthesize(self, text: str, lang: str, slow: bool) -> Optional[bytes]:
        try:
            from gtts import gTTS
            tts    = gTTS(text=text, lang=lang, slow=slow)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            print(f"[Voice/TTS] gTTS failed for lang={lang}: {e}")
            return None

    def synthesize_with_ssml_pause(self, sentences: List[str],
                                    language: str = "hi") -> Optional[bytes]:
        """
        Synthesize multiple sentences with pauses between them.
        Useful for structured financial advice with numbered steps.
        """
        full_text = ". ".join(sentences)
        return self.synthesize(full_text, language)


# ─────────────────────────────────────────────────────────────
# 5. WHISPER FINE-TUNING HELPER
# ─────────────────────────────────────────────────────────────

class WhisperFineTuner:
    """
    Fine-tune Whisper on Indian language speech datasets.

    Datasets to use:
    1. Mozilla Common Voice 15: hi, mr, ta, bn, te, kn (commonvoice.mozilla.org)
    2. OpenSLR Indian datasets:
       - Resource 53: Large Gujarati ASR
       - Resource 64: Large Punjabi ASR
       - Resource 65: Large Tamil ASR
       - Resource 66: Large Telugu ASR
       - Resource 103: Marathi ASR
       - Resource 116-117: Hindi ASR
    3. AI4Bharat IndicSUPERB: HuggingFace ai4bharat/IndicSUPERB
    """

    def __init__(self, base_model: str = "openai/whisper-small",
                 output_dir: str = "whisper_finetuned"):
        self.base_model = base_model
        self.output_dir = output_dir

    def prepare_dataset(self, lang: str, data_source: str = "mozilla_cv") -> Dict:
        """
        Load and prepare a speech dataset for Whisper fine-tuning.
        Returns HuggingFace dataset dict with audio + transcription.
        """
        try:
            from datasets import load_dataset, Audio

            if data_source == "mozilla_cv":
                # Mozilla Common Voice — supported langs: hi, mr, ta, bn, te, kn
                ds = load_dataset("mozilla-foundation/common_voice_15_0",
                                  lang, split="train+validation",
                                  trust_remote_code=True)
                ds = ds.cast_column("audio", Audio(sampling_rate=16000))
                return {"dataset": ds, "audio_col": "audio", "text_col": "sentence"}

            elif data_source == "indicSUPERB":
                ds = load_dataset("ai4bharat/IndicSUPERB",
                                  lang, split="train", trust_remote_code=True)
                return {"dataset": ds, "audio_col": "audio", "text_col": "transcript"}

            else:
                raise ValueError(f"Unknown data source: {data_source}")

        except Exception as e:
            print(f"[WhisperFT] Dataset load failed: {e}")
            return {}

    def finetune(self, lang: str, data_source: str = "mozilla_cv",
                 num_steps: int = 4000, batch_size: int = 8):
        """
        Fine-tune Whisper for a specific Indian language.
        Run this separately for each language that needs improvement.
        Colab A100 recommended: ~2-3 hours per language for 4000 steps.
        """
        try:
            from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                                       Seq2SeqTrainer, Seq2SeqTrainingArguments)
            from datasets import Audio
            import torch

            print(f"[WhisperFT] Fine-tuning Whisper for {lang}...")

            # Load model + processor
            processor = WhisperProcessor.from_pretrained(
                self.base_model, language=LANG_CONFIG[lang]["name"], task="transcribe"
            )
            model = WhisperForConditionalGeneration.from_pretrained(self.base_model)
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens    = []

            # Load data
            data_info = self.prepare_dataset(lang, data_source)
            if not data_info:
                return

            ds       = data_info["dataset"]
            audio_col = data_info["audio_col"]
            text_col  = data_info["text_col"]

            # Preprocessing
            def preprocess(batch):
                audio  = batch[audio_col]
                inputs = processor(
                    audio["array"], sampling_rate=16000, return_tensors="pt"
                )
                batch["input_features"] = inputs.input_features[0]
                with processor.as_target_processor():
                    batch["labels"] = processor(batch[text_col]).input_ids
                return batch

            ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=2)

            # Training arguments
            args = Seq2SeqTrainingArguments(
                output_dir=f"{self.output_dir}/{lang}",
                num_train_epochs=3,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=2,
                learning_rate=1e-5,
                warmup_steps=500,
                max_steps=num_steps,
                gradient_checkpointing=True,
                fp16=torch.cuda.is_available(),
                evaluation_strategy="steps",
                eval_steps=1000,
                save_steps=1000,
                logging_steps=25,
                predict_with_generate=True,
                generation_max_length=225,
                report_to=["tensorboard"],
                load_best_model_at_end=True,
                metric_for_best_model="wer",
                greater_is_better=False,
                push_to_hub=False,
            )

            trainer = Seq2SeqTrainer(
                model=model,
                args=args,
                train_dataset=ds,
                tokenizer=processor.feature_extractor,
            )
            trainer.train()
            trainer.save_model(f"{self.output_dir}/{lang}/final")
            print(f"[WhisperFT] Fine-tuned model saved to {self.output_dir}/{lang}/final")

        except ImportError as e:
            print(f"[WhisperFT] Missing dependency: {e}")
            print("Install: pip install transformers datasets evaluate jiwer")


# ─────────────────────────────────────────────────────────────
# 6. VOICE MESSAGE HANDLER (end-to-end)
# ─────────────────────────────────────────────────────────────

class VoiceHandler:
    """
    End-to-end voice message processor.
    WhatsApp audio → transcription → (optional) audio reply.
    """

    def __init__(self, whisper_size: str = "small",
                 tts_engine: str = "gtts",
                 finetuned_dir: Optional[str] = None):
        self.stt = WhisperSTT(
            model_size=whisper_size,
            finetuned_path=self._find_finetuned(finetuned_dir, None),
        )
        self.tts = IndianTTS(engine=tts_engine)

    def _find_finetuned(self, base_dir: Optional[str], lang: Optional[str]) -> Optional[str]:
        if not base_dir:
            return None
        path = Path(base_dir) / (lang or "") / "final"
        return str(path) if path.exists() else None

    def process_voice_note(self, audio_bytes: bytes,
                            audio_format: str = "ogg",
                            user_language: Optional[str] = None) -> Dict:
        """
        Process a WhatsApp voice note end-to-end.

        Args:
            audio_bytes   : raw audio bytes from WhatsApp
            audio_format  : original format (ogg, mp4, etc.)
            user_language : override language detection

        Returns:
            dict with transcribed text, detected language, confidence
        """
        result = self.stt.transcribe(
            audio_bytes,
            language=user_language,
            auto_detect=(user_language is None),
        )
        return {
            "text":       result.get("text", ""),
            "language":   result.get("language", "hi"),
            "confidence": result.get("confidence", 0.0),
            "duration_s": result.get("duration_s", 0),
            "error":      result.get("error"),
        }

    def generate_audio_reply(self, text: str, language: str = "hi",
                              max_chars: int = 500) -> Optional[bytes]:
        """
        Generate audio reply for the given text.
        Truncates to max_chars for reasonable audio length.
        """
        if len(text) > max_chars:
            # Truncate at sentence boundary
            sentences = text[:max_chars].rsplit(".", 1)
            text      = sentences[0] + "."
        return self.tts.synthesize(text, language)

    def chunk_long_response(self, text: str, language: str,
                             max_chunk: int = 400) -> List[bytes]:
        """
        Split a long response into multiple audio chunks.
        WhatsApp has a ~16MB limit per voice note.
        """
        chunks      = []
        sentences   = text.replace("।", ".").replace("।", ".").split(".")
        current     = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(current) + len(s) > max_chunk and current:
                audio = self.tts.synthesize(current, language)
                if audio:
                    chunks.append(audio)
                current = s
            else:
                current = (current + ". " + s).strip()
        if current:
            audio = self.tts.synthesize(current, language)
            if audio:
                chunks.append(audio)
        return chunks


# ─────────────────────────────────────────────────────────────
# 7. TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from typing import List

    print("=" * 55)
    print("ArthaSathi Voice Pipeline — Test")
    print("=" * 55)

    # Test TTS
    tts = IndianTTS("gtts")
    test_texts = [
        ("Aapka karz 50000 rupaye hai. Pehle credit card chukao.", "hi"),
        ("Your monthly EMI should be Rs 2,500.", "en"),
        ("तुमचे कर्ज 50000 रुपये आहे. आधी क्रेडिट कार्ड भरा.", "mr"),
    ]
    for text, lang in test_texts:
        audio = tts.synthesize(text, lang)
        if audio:
            size = len(audio)
            print(f"  [{lang}] TTS: {size:,} bytes ({size//1024}KB) — '{text[:40]}'")
        else:
            print(f"  [{lang}] TTS failed (install: pip install gtts)")

    print("\nVoice pipeline tests complete!")
    print("\nTo test STT, install: pip install openai-whisper")
    print("Then call: handler = VoiceHandler(); result = handler.process_voice_note(audio_bytes)")
