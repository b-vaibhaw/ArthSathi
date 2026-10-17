# ArthaSathi — Independent AI Financial Companion

A from-the-ground-up rebuild of the original ArthaSathi project (which aimed to
pretrain a 345M-parameter LLM from scratch for 700M users). That plan isn't
realistic without a serious GPU budget and months of data work — so this
version instead plugs a **free, self-hosted, open-weight LLM** into a
**deterministic tool-calling agent**, which is buildable and runnable in
hours, costs nothing in API fees, and still gives you a real, independent
platform that you own end to end.

## Why this architecture

| Old plan | This platform |
|---|---|
| Train a 345M-param model from scratch (months, serious GPU $) | Use an existing free open-weight model (Llama 3.1 / Qwen2.5 / Mistral) via [Ollama](https://ollama.com) — self-hosted, zero API cost |
| Hardcoded if/else response templates per language/intent | Real LLM conversation, grounded by tool calls — handles anything, not just the cases someone wrote a template for |
| LLM does arithmetic directly (unreliable for money) | LLM **never** computes a number itself — it calls a deterministic Python tool and reports the exact result |
| Niche: India micro-finance only | Both: India micro-finance (debt, GST, ITR, MUDRA) **and** general personal finance (budgeting, SIP investing, stock basics) |

The two genuinely solid pieces of the old repo — `engines/financial_engines.py`
(debt/tax/business math) and `rag/rag_engine.py` (RAG + per-user memory),
now preserved under [`legacy_from_scratch_llm/`](legacy_from_scratch_llm/) —
are carried forward and extended into `app/`, just reorganized and with more
tools added. The hardcoded chat brain in the old `api/main.py` is replaced by
a real tool-calling agent in `app/agent/`.

**This README and `app/` are now the repo root** — `pip install`, `pytest`,
and `uvicorn` below all run from the top of the repo, no `cd` needed.

## Architecture

```
app/
├── engines/        Deterministic financial math — debt payoff, EMI, tax/GST,
│                   business pricing, SIP/investment projections, simple
│                   stock valuation. No LLM involved; every number here is
│                   exact and auditable.
├── rag/            Knowledge base (GST/RBI/tax rules + general personal
│                   finance facts) with semantic search, plus per-user
│                   SQLite memory (profile, debts, transactions, history).
├── news/           Financial news (RSS) and stock/market quotes (yfinance) —
│                   the only source of "current" facts the agent is allowed
│                   to state.
├── agent/          The brain: tool schemas + dispatcher (tools.py), a
│                   swappable LLM client (llm_client.py — Ollama or a mock
│                   for testing), and the tool-calling loop (orchestrator.py).
└── api/            FastAPI app wiring it together + a small web chat UI
                    (app/api/static/).
```

**How a message flows:** user message → RAG fetches relevant knowledge +
user profile → LLM (via Ollama) decides whether to answer directly or call
one or more tools (EMI calculator, debt strategy, GST check, SIP projection,
stock quote, news, …) → tool results feed back to the LLM → LLM writes the
final natural-language answer in the user's own language, citing the tool's
exact numbers.

This separation is what makes it trustworthy as financial advice: **the math
is always correct regardless of which open-weight model you plug in**, because
the model never does the math — it only explains tool output.

## Setup

### 1. Install Ollama and pull a free model

```bash
# https://ollama.com/download
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &          # starts the local model server (free, no API key)

# Pick one based on your RAM (all free, all support tool-calling):
ollama pull qwen2.5:7b-instruct   # ~8GB RAM, strong multilingual incl. Indian languages (default)
ollama pull llama3.1:8b           # ~8GB RAM, excellent general quality
ollama pull qwen2.5:3b            # ~4GB RAM, lighter machines
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
# Optional, for better retrieval + live news/market data:
pip install sentence-transformers faiss-cpu feedparser yfinance
# then uncomment them in requirements.txt and set USE_* flags in .env
```

### 3. Run

```bash
cp .env.example .env     # adjust OLLAMA_MODEL etc. if needed
./scripts/run_dev.sh
# or directly:
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** — a chat UI with a profile panel (income,
debts) and a list of every tool the agent can call.

### 4. Run the tests (no Ollama needed)

```bash
pytest tests/ -v
```

The test suite runs the full agent loop — orchestrator, tool dispatch,
RAG, memory, and the HTTP API — against `MockLLMClient`, a tiny rule-based
stand-in, so you can verify everything is wired correctly **before**
touching Ollama. Set `LLM_BACKEND=ollama` to use the real model once it's
running.

## Extending it

- **New calculation/tool**: add a method to the right file in `app/engines/`,
  declare its JSON schema + dispatch in `app/agent/tools.py`. The LLM will
  start using it automatically — no prompt engineering needed beyond the
  schema's `description`.
- **More knowledge**: append `(text, metadata)` tuples to `KNOWLEDGE_DOCS` in
  `app/rag/knowledge_base.py`, or drop `*.jsonl` files into a directory and
  pass it to `build_knowledge_base(extra_dir=...)`.
- **Live news/market data**: uncomment `feedparser`/`yfinance` in
  `requirements.txt` and install them — `app/news/market_data.py` already
  calls them, with graceful "unavailable" fallbacks if they're missing or
  offline.
- **Domain fine-tuning** (optional, later): once the free-LLM agent is live
  and you have real conversation logs, you can LoRA-fine-tune an open model
  on them for a more specialized persona/tone. `legacy_from_scratch_llm/tokenizer_training_data/`
  and the original blueprint's synthetic-data approach are a reasonable
  starting point for that — but treat it as a v2 optimization, not a
  prerequisite to shipping.
- **WhatsApp/SMS delivery**: the original blueprint's webhook pattern
  (`legacy_from_scratch_llm/api/main.py`) can be added as another router
  calling `AgentOrchestrator.handle_message()` — the agent itself doesn't
  care which channel it's being called from.

## What changed from the original repo

- `api/main.py`'s ~600-line hardcoded per-language if/else response tree is
  gone, replaced by `app/agent/orchestrator.py` + `app/agent/tools.py`.
- `engines/financial_engines.py` → split into `app/engines/{debt,tax,business,investment}_engine.py`,
  with a new `investment_engine.py` for SIP/budgeting/valuation (general
  personal finance, not just India micro-finance).
- `rag/rag_engine.py` → split into `app/rag/{vector_store,knowledge_base,memory,engine}.py`,
  with the knowledge base extended beyond India-only content.
- New `app/news/market_data.py` for financial news + stock quotes.
- Everything from the "train a 345M-param LLM from scratch" plan —
  `model/`, `data/`, `notebooks/`, `voice/`, `tokenizer_training_data/`,
  `arthasathi_tokenizer/`, the old `api/`, `engines/`, `rag/`, `evaluation/`,
  `deployment/`, the blueprint docs, and the old heavy `requirements.txt`
  (renamed `requirements-legacy-llm-training.txt`) — moved as-is into
  [`legacy_from_scratch_llm/`](legacy_from_scratch_llm/), preserved for
  reference/future fine-tuning but not part of the running platform.
