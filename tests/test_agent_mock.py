"""
End-to-end test of the agent tool-calling loop using MockLLMClient, proving
the orchestrator -> tool dispatch -> engine -> back-to-LLM plumbing works
correctly WITHOUT needing Ollama or a real model (this sandbox has neither
GPU nor network access to pull one). Swap in OllamaClient for real use.
"""

import tempfile

from app.agent.llm_client import MockLLMClient
from app.agent.orchestrator import AgentOrchestrator
from app.rag.engine import RAGEngine


def _make_orchestrator():
    tmp_index = tempfile.mkdtemp()
    tmp_db = tempfile.mktemp(suffix=".db")
    rag = RAGEngine(index_path=tmp_index, db_path=tmp_db, use_embeddings=False)
    return AgentOrchestrator(MockLLMClient(), rag)


def test_emi_question_triggers_emi_tool():
    orch = _make_orchestrator()
    result = orch.handle_message("u1", "What is my EMI for a 100000 loan at 18 percent for 24 months?")
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "emi_calculator"
    assert "error" not in result["tool_calls"][0]["result"]
    assert result["reply"]  # final answer was produced


def test_gst_question_triggers_gst_tool():
    orch = _make_orchestrator()
    result = orch.handle_message("u2", "Do I need GST registration for turnover 2500000?")
    assert result["tool_calls"][0]["tool"] == "gst_registration_check"
    assert result["tool_calls"][0]["result"]["status"] == "NOT_REQUIRED"


def test_plain_chat_with_no_tool_match_still_replies():
    orch = _make_orchestrator()
    result = orch.handle_message("u3", "hello there")
    assert result["tool_calls"] == []
    assert result["reply"]


def test_conversation_history_persists_across_turns():
    orch = _make_orchestrator()
    orch.handle_message("u4", "hello")
    orch.handle_message("u4", "hello again")
    history = orch.rag.memory.get_history("u4", last_n=10)
    # 2 user turns + 2 assistant turns = 4 messages
    assert len(history) == 4
