"""
engine.py
=========
Top-level RAG entry point: wires VectorStore + knowledge base + UserMemory
together, and does lightweight language detection. This is what the
API layer and the agent orchestrator import.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from app.rag.vector_store import VectorStore
from app.rag.knowledge_base import build_knowledge_base
from app.rag.memory import UserMemory


class RAGEngine:
    def __init__(self, index_path: str = "rag_index", db_path: str = "arthasathi.db",
                 use_embeddings: bool = True):
        self.store = VectorStore(
            index_path=index_path if Path(index_path).exists() else None,
            use_embeddings=use_embeddings,
        )
        self.memory = UserMemory(db_path)
        if len(self.store.documents) == 0:
            build_knowledge_base(self.store)
            self.store.save(index_path)

    def search_knowledge(self, query: str, k: int = 3) -> List[Dict]:
        return self.store.search(query, k=k)

    def get_profile(self, user_id: str) -> Dict:
        return self.memory.get_full_profile(user_id)

    def save_user_message(self, user_id: str, content: str, lang: str = "en"):
        self.memory.add_message(user_id, "user", content, lang)

    def save_assistant_message(self, user_id: str, content: str, lang: str = "en"):
        self.memory.add_message(user_id, "assistant", content, lang)

    def detect_language(self, text: str) -> str:
        """Fast Unicode-range based language detection. Good enough to pick
        a system-prompt language hint; the LLM does the real understanding."""
        counts: Dict[str, int] = {}
        for ch in text:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:
                counts["hi"] = counts.get("hi", 0) + 1
            elif 0x0B80 <= cp <= 0x0BFF:
                counts["ta"] = counts.get("ta", 0) + 1
            elif 0x0C80 <= cp <= 0x0CFF:
                counts["kn"] = counts.get("kn", 0) + 1
            elif 0x0980 <= cp <= 0x09FF:
                counts["bn"] = counts.get("bn", 0) + 1
            elif 0x0C00 <= cp <= 0x0C7F:
                counts["te"] = counts.get("te", 0) + 1
            elif ch.isascii() and ch.isalpha():
                counts["en"] = counts.get("en", 0) + 1
        if not counts:
            return "en"
        return max(counts, key=counts.get)

    def build_context_block(self, user_id: str, user_message: str, max_knowledge_chars: int = 900) -> str:
        """Human-readable context block injected into the system prompt:
        relevant knowledge + user profile summary. Conversation history is
        handled separately as real chat turns, not squeezed into this block."""
        parts = []

        docs = self.search_knowledge(user_message, k=3)
        if docs:
            knowledge = "\n".join(f"- {d['text'][:250]}" for d in docs)[:max_knowledge_chars]
            parts.append(f"[RELEVANT KNOWLEDGE]\n{knowledge}")

        profile = self.get_profile(user_id)
        profile_lines = []
        if profile.get("monthly_income", 0) > 0:
            profile_lines.append(f"Monthly income: Rs {profile['monthly_income']:,.0f}")
        if profile.get("debts"):
            profile_lines.append(f"Active debts ({len(profile['debts'])}), debt-to-income: {profile['debt_to_income_pct']}%")
            for d in profile["debts"][:3]:
                profile_lines.append(f"  - {d['name']}: Rs {d['principal']:,.0f} @ {d['annual_rate']}%")
        if profile_lines:
            parts.append("[USER PROFILE]\n" + "\n".join(profile_lines))

        return "\n\n".join(parts)
