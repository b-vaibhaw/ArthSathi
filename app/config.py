"""
config.py
=========
Central configuration for the ArthaSathi platform. Everything is read from
environment variables (with sane free/local defaults) so the same code runs
in dev, on your own server, or in Docker without code changes.
"""

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # ── LLM backend ──────────────────────────────────────────
    # "ollama" = free, self-hosted, open-weight model (recommended).
    # "mock"   = deterministic fake LLM, used for tests / offline dev
    #            when no Ollama server is reachable (e.g. this sandbox).
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # Good free open-weight models with tool-calling support, pick by your RAM:
    #   llama3.1:8b      — ~8GB RAM, best general quality
    #   qwen2.5:7b       — ~8GB RAM, strong multilingual (incl. Indian languages)
    #   mistral:7b       — ~8GB RAM
    #   qwen2.5:3b / phi3.5  — ~4GB RAM, lighter machines
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    LLM_TIMEOUT_S: int = int(os.getenv("LLM_TIMEOUT_S", "120"))
    LLM_MAX_TOOL_ITERATIONS: int = int(os.getenv("LLM_MAX_TOOL_ITERATIONS", "6"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))

    # ── Storage ──────────────────────────────────────────────
    DB_PATH: str = os.getenv("DB_PATH", "arthasathi.db")
    RAG_INDEX_PATH: str = os.getenv("RAG_INDEX_PATH", "rag_index")

    # ── Optional semantic search / news (feature-detected, not required) ──
    USE_SENTENCE_EMBEDDINGS: bool = _bool("USE_SENTENCE_EMBEDDINGS", True)
    USE_LIVE_NEWS: bool = _bool("USE_LIVE_NEWS", True)
    USE_LIVE_MARKET_DATA: bool = _bool("USE_LIVE_MARKET_DATA", True)

    # ── Server ───────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
