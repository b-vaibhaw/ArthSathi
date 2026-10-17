"""
End-to-end API test using FastAPI's TestClient, forced onto MockLLMClient
(no Ollama/network needed) via LLM_BACKEND=mock. Proves the HTTP layer,
lifespan startup, and the full chat round-trip all work together.
"""

import os
import tempfile

os.environ["LLM_BACKEND"] = "mock"
os.environ["DB_PATH"] = tempfile.mktemp(suffix=".db")
os.environ["RAG_INDEX_PATH"] = tempfile.mkdtemp()
os.environ["USE_SENTENCE_EMBEDDINGS"] = "false"

from fastapi.testclient import TestClient  # noqa: E402
from app.api.main import app  # noqa: E402


def test_health_check():
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


def test_chat_roundtrip():
    with TestClient(app) as client:
        res = client.post("/api/chat", json={"user_id": "api_test_1", "message": "What is my EMI for 50000 at 12 percent for 12 months?"})
        assert res.status_code == 200
        data = res.json()
        assert data["reply"]
        assert data["tool_calls"][0]["tool"] == "emi_calculator"


def test_profile_set_and_fetch():
    with TestClient(app) as client:
        client.post("/api/profile/api_test_2/income", json={"monthly_income": 25000})
        client.post("/api/profile/api_test_2/debt", json={
            "name": "Test Card", "principal": 20000, "annual_rate": 36, "min_payment": 800, "lender_type": "bank",
        })
        res = client.get("/api/profile/api_test_2")
        profile = res.json()
        assert profile["monthly_income"] == 25000
        assert len(profile["debts"]) == 1


def test_direct_tool_invocation():
    with TestClient(app) as client:
        res = client.post("/api/tools/gst_registration_check", json={"arguments": {"annual_turnover": 2500000}})
        assert res.status_code == 200
        assert res.json()["status"] == "NOT_REQUIRED"


def test_tool_list_endpoint():
    with TestClient(app) as client:
        res = client.get("/api/tools")
        names = {t["name"] for t in res.json()}
        assert "emi_calculator" in names
        assert "get_financial_news" in names


def test_static_frontend_served():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "ArthaSathi" in res.text
