#!/usr/bin/env bash
# Run the ArthaSathi platform locally against Ollama.
# Prereqs: `ollama serve` running, and `ollama pull $OLLAMA_MODEL` done once.
set -euo pipefail
cd "$(dirname "$0")/.."

export LLM_BACKEND="${LLM_BACKEND:-ollama}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"

echo "Starting ArthaSathi (LLM_BACKEND=$LLM_BACKEND, OLLAMA_MODEL=$OLLAMA_MODEL)"
echo "Open http://localhost:8000 once it says 'Application startup complete.'"
python3 -m uvicorn app.api.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --reload
