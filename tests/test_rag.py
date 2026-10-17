"""Tests for the RAG/knowledge retrieval and per-user memory layer."""

import tempfile
import os

from app.rag.engine import RAGEngine


def test_knowledge_search_finds_relevant_doc():
    tmp_index = tempfile.mkdtemp()
    tmp_db = tempfile.mktemp(suffix=".db")
    rag = RAGEngine(index_path=tmp_index, db_path=tmp_db, use_embeddings=False)  # keyword fallback, no downloads needed
    results = rag.search_knowledge("GST registration turnover threshold")
    assert len(results) > 0
    assert any("GST" in r["text"] for r in results)


def test_user_memory_roundtrip():
    tmp_index = tempfile.mkdtemp()
    tmp_db = tempfile.mktemp(suffix=".db")
    rag = RAGEngine(index_path=tmp_index, db_path=tmp_db, use_embeddings=False)

    uid = "test_user_1"
    rag.memory.update_user(uid, monthly_income=20000)
    rag.memory.add_debt(uid, "Test Loan", 50000, 24, 2000, "bank")

    profile = rag.get_profile(uid)
    assert profile["monthly_income"] == 20000
    assert len(profile["debts"]) == 1
    assert profile["debts"][0]["name"] == "Test Loan"


def test_language_detection():
    tmp_index = tempfile.mkdtemp()
    tmp_db = tempfile.mktemp(suffix=".db")
    rag = RAGEngine(index_path=tmp_index, db_path=tmp_db, use_embeddings=False)

    assert rag.detect_language("What is my EMI this month?") == "en"
    assert rag.detect_language("मेरा लोन कितना है") == "hi"
    assert rag.detect_language("என் கடன் எவ்வளவு") == "ta"


def test_context_block_includes_profile_once_set():
    tmp_index = tempfile.mkdtemp()
    tmp_db = tempfile.mktemp(suffix=".db")
    rag = RAGEngine(index_path=tmp_index, db_path=tmp_db, use_embeddings=False)

    uid = "test_user_2"
    rag.memory.update_user(uid, monthly_income=15000)
    rag.memory.add_debt(uid, "HDFC CC", 30000, 36, 900, "bank")

    block = rag.build_context_block(uid, "should I pay off my credit card first?")
    assert "HDFC CC" in block
    assert "Rs 15,000" in block
