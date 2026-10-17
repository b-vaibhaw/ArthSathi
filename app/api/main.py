"""
ArthaSathi Platform — FastAPI application.

Wires together: the LLM agent orchestrator, the deterministic financial
tools, the RAG knowledge/memory layer, and a small web chat UI.

Run:  uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.agent.llm_client import OllamaClient, MockLLMClient
from app.agent.orchestrator import AgentOrchestrator
from app.agent.tools import TOOL_SCHEMAS, ToolExecutor
from app.rag.engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("arthasathi.api")


class AppState:
    rag: Optional[RAGEngine] = None
    orchestrator: Optional[AgentOrchestrator] = None
    tool_executor: Optional[ToolExecutor] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ArthaSathi platform (LLM_BACKEND=%s, model=%s)",
                settings.LLM_BACKEND, settings.OLLAMA_MODEL)

    state.rag = RAGEngine(index_path=settings.RAG_INDEX_PATH, db_path=settings.DB_PATH,
                           use_embeddings=settings.USE_SENTENCE_EMBEDDINGS)

    if settings.LLM_BACKEND == "ollama":
        llm_client = OllamaClient(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL,
                                   settings.LLM_TIMEOUT_S, settings.LLM_TEMPERATURE)
    else:
        logger.warning("LLM_BACKEND=%s — using MockLLMClient (no real model). "
                        "Set LLM_BACKEND=ollama and run Ollama for real conversations.", settings.LLM_BACKEND)
        llm_client = MockLLMClient()

    state.orchestrator = AgentOrchestrator(llm_client, state.rag, settings.LLM_MAX_TOOL_ITERATIONS)
    state.tool_executor = state.orchestrator.tools

    logger.info("ArthaSathi platform ready. Knowledge docs: %d", len(state.rag.store.documents))
    yield
    logger.info("Shutting down ArthaSathi platform.")


app = FastAPI(title="ArthaSathi", description="Independent AI financial companion", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return PlainTextResponse("ArthaSathi API is running. (static/index.html not found)")


# ─────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Any stable identifier for the user/session")
    message: str


class ChatResponse(BaseModel):
    reply: str
    language: str
    tool_calls: List[Dict] = []


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")
    try:
        result = state.orchestrator.handle_message(req.user_id, req.message)
    except ConnectionError as e:
        raise HTTPException(503, str(e))
    return ChatResponse(**result)


# ─────────────────────────────────────────────────────────────
# Profile
# ─────────────────────────────────────────────────────────────

class DebtIn(BaseModel):
    name: str
    principal: float
    annual_rate: float
    min_payment: float
    lender_type: str = "bank"


class IncomeIn(BaseModel):
    monthly_income: float


@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    return state.rag.get_profile(user_id)


@app.post("/api/profile/{user_id}/debt")
async def add_debt(user_id: str, debt: DebtIn):
    state.rag.memory.add_debt(user_id, debt.name, debt.principal, debt.annual_rate,
                               debt.min_payment, debt.lender_type)
    return state.rag.get_profile(user_id)


@app.post("/api/profile/{user_id}/income")
async def set_income(user_id: str, income: IncomeIn):
    state.rag.memory.update_user(user_id, monthly_income=income.monthly_income)
    return state.rag.get_profile(user_id)


# ─────────────────────────────────────────────────────────────
# Direct tool access ("financial model agent" as a standalone calculator,
# without going through the chat/LLM layer — useful for a UI that wants
# an instant calculator widget, or for scripted/batch use.)
# ─────────────────────────────────────────────────────────────

@app.get("/api/tools")
async def list_tools():
    return [{"name": t["function"]["name"], "description": t["function"]["description"]} for t in TOOL_SCHEMAS]


class ToolInvokeRequest(BaseModel):
    arguments: Dict = {}


@app.post("/api/tools/{tool_name}")
async def invoke_tool(tool_name: str, req: ToolInvokeRequest):
    valid_names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    if tool_name not in valid_names:
        raise HTTPException(404, f"Unknown tool '{tool_name}'")
    result = state.tool_executor.dispatch(tool_name, req.arguments)
    return result


@app.get("/api/health")
async def health():
    return {"status": "ok", "llm_backend": settings.LLM_BACKEND, "model": settings.OLLAMA_MODEL,
            "knowledge_docs": len(state.rag.store.documents) if state.rag else 0}
