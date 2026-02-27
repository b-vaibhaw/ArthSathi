# 🏦 ArthaSathi AI — Complete LLM from Scratch
### अर्थसाथी — Your Financial Companion for 700M+ Indians

A **custom-built multilingual LLM** for micro-entrepreneur financial coaching and debt management.
Built from scratch — custom tokenizer, custom transformer, custom training pipeline.

**Languages:** Hindi, English, Marathi, Tamil, Kannada, Bhojpuri, Assamese, Bengali, Telugu  
**Channels:** WhatsApp, SMS, Voice Notes  
**Target users:** Informal workers, micro-entrepreneurs, people in debt crisis

---

## 📁 Project Structure

```
arthasathi/
├── model/
│   ├── arthasathi_model.py    # 🧠 Complete LLM transformer from scratch (345M)
│   ├── tokenizer.py           # 🔤 Custom BPE tokenizer (60K vocab, 9 languages)
│   ├── train_pretrain.py      # 🏋️ Pre-training script (15B tokens)
│   ├── train_finetune.py      # 🎯 Fine-tuning with LoRA
│   └── distill.py             # 📱 Distill 345M → 80M for mobile
├── data/
│   └── dataset_pipeline.py    # 📦 Download 14 datasets + generate 2.25M synthetic
├── rag/
│   └── rag_engine.py          # 🔍 FAISS vector store + user memory (SQLite)
├── engines/
│   └── financial_engines.py   # 💰 Debt, tax, GST, pricing calculators
├── voice/
│   └── voice_pipeline.py      # 🎙️ Whisper STT + gTTS for voice notes
├── api/
│   └── main.py                # 🚀 FastAPI: WhatsApp + SMS + REST endpoints
├── evaluation/
│   └── evaluate.py            # 📊 Perplexity, accuracy, safety, language tests
├── notebooks/
│   └── colab_train.py         # 📓 Google Colab training guide
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
└── requirements.txt
```

---

## ⚡ Quick Start (5 minutes)

```bash
# 1. Clone and install
git clone https://github.com/your-org/arthasathi.git
cd arthasathi
pip install -r requirements.txt

# 2. Run financial engine tests (no model needed)
python evaluation/evaluate.py --mode engines

# 3. Start API server (template responses before training)
python api/main.py
# Open: http://localhost:8000/docs
```

---

## 🗓️ Full Training Timeline

| Phase | What | Time | Hardware |
|-------|------|------|----------|
| 1 | Download + clean 50GB data | 6-8 hrs | Any internet |
| 2 | Train tokenizer (60K BPE) | 4-6 hrs | CPU |
| 3 | Pre-train 117M (proof-of-concept) | 72 hrs | Colab T4 free |
| 4 | Pre-train 345M (production) | 120 hrs | Colab A100 ($12) |
| 5 | Instruction fine-tune (LoRA) | 20 hrs | Colab A100 |
| 6 | Chat fine-tune (LoRA) | 10 hrs | Colab A100 |
| 7 | Build RAG + test | 4 hrs | PC |
| 8 | Evaluate + iterate | 2 weeks | PC + volunteers |

**Total compute cost: ~$25 (Colab A100 pay-as-you-go)**

---

## 🏋️ Training — Step by Step

### Step 1: Prepare Data

```bash
# Download all 14 datasets (~50GB raw)
python data/dataset_pipeline.py \
    --base_dir . \
    --gb_per_lang 2.0 \
    --skip_synthetic  # add later

# This downloads:
# - IndicCorp v2 (AI4Bharat) — 20B token Indian language corpus
# - Sangraha (AI4Bharat) — 251B token filtered corpus
# - Wikipedia (9 languages)
# - CC-100 CommonCrawl
# - mC4, OSCAR
# - Indic-Instruct, IndicQA, OpenAssistant
# - RBI/SEBI/NABARD PDFs (financial domain)
```

### Step 2: Train Tokenizer

```bash
# Train custom BPE tokenizer (60K vocab, 9 Indian languages)
python model/tokenizer.py \
    --data_dir data/clean \
    --save_dir arthasathi_tokenizer \
    --vocab_size 60000

# Test: Hindi "मेरा कर्ज़" → 2-3 tokens (vs 8-15 with English tokenizer)
```

### Step 3: Pre-Training (Colab)

Open `notebooks/colab_train.py` and copy cells into Colab.

```python
# Or run locally if you have 16GB+ VRAM
python model/train_pretrain.py \
    --model_size small \       # 117M — use 'medium' for 345M
    --data_files formatted/pretrain.jsonl \
    --tok_dir arthasathi_tokenizer \
    --batch_size 4 \
    --max_steps 150000 \
    --ckpt_dir checkpoints/pretrain
```

### Step 4: Generate Synthetic Conversations

```bash
# Generate 2.25M financial conversations (25K per language × 9 languages)
# Uses template-based generation (no GPU needed)
# Or use Mistral-7B for higher quality (needs GPU)
python data/dataset_pipeline.py \
    --skip_download \
    --skip_clean \
    --synthetic_n 25000
```

### Step 5: Fine-Tuning

```bash
# Stage 1: Instruction fine-tuning (Alpaca format)
python model/train_finetune.py \
    --pretrain_ckpt checkpoints/pretrain/final_pretrain.pt \
    --data_file formatted/finetune.jsonl \
    --tok_dir arthasathi_tokenizer \
    --ft_type instruction \
    --use_lora \
    --epochs 3

# Stage 2: Chat fine-tuning (ChatML format)
python model/train_finetune.py \
    --pretrain_ckpt checkpoints/finetune/best_ft.pt \
    --data_file formatted/chat.jsonl \
    --ft_type chat \
    --use_lora
```

### Step 6: Evaluate

```bash
# Financial engine accuracy (no model needed)
python evaluation/evaluate.py --mode engines

# Full evaluation (needs trained model)
python evaluation/evaluate.py \
    --mode full \
    --model_path checkpoints/finetune/final_ft.pt \
    --tok_dir arthasathi_tokenizer
```

---

## 🚀 Deployment

### Local / Development

```bash
# Set environment variables
export WA_PHONE_ID=your_whatsapp_phone_id
export WA_ACCESS_TOKEN=your_meta_access_token
export WA_VERIFY_TOKEN=your_secret_token

# Start server
python api/main.py
# → http://localhost:8000

# Expose to internet (for WhatsApp webhook)
ngrok http 8000
# Set webhook URL in Meta Developer Console: https://xxxx.ngrok.io/webhook
```

### Docker Production

```bash
cd deployment
cp .env.example .env
# Edit .env with your WhatsApp + Twilio credentials
docker-compose up -d
```

### WhatsApp Setup (Meta Cloud API)

1. Go to: developers.facebook.com → My Apps → Create App → Business
2. Add "WhatsApp" product
3. Get Phone Number ID and Access Token
4. Set Webhook URL: `https://your-domain.com/webhook`
5. Verify token: match `WA_VERIFY_TOKEN` in your .env
6. Subscribe to: `messages` webhook field

---

## 📱 Mobile Deployment (Offline)

```bash
# Distill 345M → 80M student model
python model/distill.py \
    --teacher checkpoints/finetune/final_ft.pt \
    --data_file formatted/chat.jsonl \
    --tok_dir arthasathi_tokenizer \
    --epochs 5 \
    --export_onnx

# Output: checkpoints/distilled/arthasathi_mobile_int8.onnx (~80MB)
# Convert to TFLite for Android: see deployment/android_guide.md
```

---

## 💬 Example Conversations

**Hindi (Debt Advice):**
```
User:     Bhai mera 3 loan hai — CC 30K@36%, NBFC 20K@26%, salary advance 5K@0%
          Monthly income: 15,000. Kya karu?
ArthaSathi: Dekho bhai, pehle salary advance (5K) jaldi chukao — interest nahi hai.
           Phir CC ka minimum pay karo (900/month), bacha hua sab NBFC pe lagao.
           Is tarike se 14 mahine mein NBFC khatam, phir sab CC pe — total 22 mahine mein debt free!
           Interest bachat: ~8,400 vs agar sab minimum payment karte.
           Kya HDFC se settlement ki baat kare? 60-70% pe ho sakta hai.
```

**Tamil (GST Guidance):**
```
User:     என் கிராசரி கடையின் ஆண்டு விற்பனை 25 லட்சம். GST பதிவு வேண்டுமா?
ArthaSathi: இல்லை! 40 லட்சத்திற்கும் குறைவான விற்பனைக்கு GST பதிவு தேவையில்லை.
           ஆனால் 30+ லட்சம் ஆனால், Composition Scheme எடுங்கள் — வெறும் 1% GST,
           மூன்று மாதத்திற்கு ஒருமுறை filing. வாங்கினவர்களுக்கு invoice கொடுக்கலாம்.
           gst.gov.in-ல் பதிவு செய்யுங்கள் அல்லது Jan Sewa Kendra போங்கள்.
```

**English (Loan Negotiation):**
```
User:     Bank is calling me every day for Rs 45,000 credit card dues. I lost my job.
ArthaSathi: First — they CANNOT call before 8AM or after 8PM. That's RBI rule.
           Ask the caller's name and employee ID (they must give it).
           Now, call the bank's Customer Service and say:
           "I'm facing financial hardship due to job loss. I'd like to discuss
           either a 3-month moratorium or a one-time settlement. Please transfer
           me to your Resolutions team."
           Banks have Rs 45K settlements at 60-70% = you pay ~Rs 27,000-31,500.
           Document everything in writing. Don't agree verbally.
```

---

## 🧠 Model Architecture

```
ArthaSathi LLM (345M parameters)
├── Token Embedding           : 60,000 × 1,024 = 61.4M (weight-tied)
├── 24 × TransformerBlock
│   ├── RMSNorm               : faster than LayerNorm
│   ├── MultiHeadAttention     : 16 heads, RoPE positional encoding
│   │   ├── Q/K/V projections : 1024 × 1024 (no bias)
│   │   └── Output projection
│   ├── RMSNorm
│   └── SwiGLU FFN            : 1024 → 4096 → 1024 (better than GELU)
├── Final RMSNorm
└── LM Head                   : weight-tied with embedding
```

**Key design choices:**
- **RoPE** positional encoding — generalizes to unseen sequence lengths
- **SwiGLU** activations — state-of-the-art for LLMs (LLaMA, PaLM)
- **RMSNorm** — simpler and faster than LayerNorm
- **Weight tying** — saves 61M parameters, improves quality
- **Pre-norm** architecture — more stable training than post-norm
- **Flash Attention** — 2-4x faster attention via PyTorch 2.0

---

## 📊 Target Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| Perplexity | < 15 | Language modeling quality |
| Financial accuracy | > 90% | EMI/tax/GST calculations correct |
| Language coverage | 9/9 | All languages rated ≥ 4/5 by native speakers |
| Hallucination rate | < 5% | Factual claims verified against RBI/GST docs |
| Safety | 0 harmful | No harmful financial advice |
| Latency | < 3s | Response time on single GPU |

---

## 🔑 Environment Variables

```env
# Model
MODEL_PATH=checkpoints/finetune/final_ft.pt
TOKENIZER_DIR=arthasathi_tokenizer
RAG_INDEX_PATH=rag_index
DB_PATH=arthasathi_users.db
DEVICE=cuda

# WhatsApp (Meta Cloud API)
WA_PHONE_ID=your_phone_number_id
WA_ACCESS_TOKEN=your_permanent_token
WA_VERIFY_TOKEN=arthasathi_secret_2024

# SMS (Twilio)
TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_TOKEN=your_auth_token
TWILIO_NUMBER=+1234567890

# Inference
MAX_NEW_TOKENS=300
TEMPERATURE=0.75
TOP_K=50
TOP_P=0.9
```

---

## 💡 Dataset Sources

| Dataset | Source | Content |
|---------|--------|---------|
| IndicCorp v2 | ai4bharat/IndicCorp | 20.9B words, 22 Indian languages |
| Sangraha | ai4bharat/sangraha | 251B tokens, high quality filtered |
| Wikipedia | Various | Factual text, 9 languages |
| CC-100 | FAIR | CommonCrawl, 7 Indian languages |
| mC4 | Google | Multilingual C4 |
| OSCAR | INRIA | Deduplicated CommonCrawl |
| Indic-Instruct | AI4Bharat | Instruction-following data |
| IndicQA | AI4Bharat | QA pairs in 11 languages |
| OpenAssistant | LAION | Multilingual conversations |
| Synthetic | Generated | 2.25M financial conversations |
| RBI PDFs | RBI.org.in | Regulatory circulars |
| GST Circulars | gst.gov.in | Tax regulations |
| NABARD Reports | NABARD | Microfinance guidelines |

---

## 🤝 Contributing

We especially need:
- **Native language reviewers** for quality evaluation (paid)
- **Financial domain experts** for advice accuracy checking
- **Low-income user testers** for real-world feedback

Contact: arthasathi@example.com

---

## 📜 License

MIT License — free to use, modify, and distribute.

**Important:** Financial advice from this model is for informational purposes only.
Always verify with a qualified financial advisor for major decisions.

---

*Built with ❤️ for India's 700 million under-banked citizens.*
*अर्थ = money/meaning | साथी = companion/friend*
