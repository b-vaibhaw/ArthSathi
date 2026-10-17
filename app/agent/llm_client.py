"""
llm_client.py
==============
Swappable LLM backends for the agent. The orchestrator only depends on the
`LLMClient` interface (`chat(messages, tools) -> LLMResponse`), so the brain
behind ArthaSathi can be swapped without touching the agent loop, tools, or
API layer.

- OllamaClient: talks to a locally-running Ollama server via its
  OpenAI-compatible `/v1/chat/completions` endpoint. This is the free,
  self-hosted, no-API-key backend this platform is built around — install
  Ollama, `ollama pull <model>`, and point OLLAMA_BASE_URL at it (see README).
- MockLLMClient: a tiny rule-based stand-in used by the test suite and by
  this sandbox (which has no GPU and no network access to pull a real
  model) to prove the tool-calling loop, RAG, and API wiring all work
  correctly end-to-end before you ever touch Ollama.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import requests


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


class LLMClient:
    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, timeout_s: int = 120, temperature: float = 0.4):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature

    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        try:
            r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url}. Is `ollama serve` running, and have you "
                f"run `ollama pull {self.model}`? Original error: {e}"
            ) from e

        data = r.json()
        message = data["choices"][0]["message"]

        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.get("id", fn.get("name", "call")), name=fn.get("name", ""), arguments=args))

        return LLMResponse(content=message.get("content"), tool_calls=tool_calls)


class MockLLMClient(LLMClient):
    """Deterministic rule-based mock. NOT a real language model — it exists
    purely so the rest of the stack (tool dispatch, RAG, memory, API) can be
    tested and demoed without Ollama or a GPU. Swap LLM_BACKEND=ollama for
    real conversational ability."""

    _KEYWORD_TOOL_MAP = [
        (re.compile(r"\bemi\b", re.I), "emi_calculator",
         lambda m: {"principal": _first_number(m, 0, 100000), "annual_rate": _first_number(m, 1, 12),
                    "tenure_months": int(_first_number(m, 2, 24))}),
        (re.compile(r"\b(news|headline)", re.I), "get_financial_news", lambda m: {"max_items": 3}),
        (re.compile(r"\bsip\b", re.I), "sip_future_value",
         lambda m: {"monthly_investment": _first_number(m, 0, 5000), "annual_return_pct": 12, "years": 10}),
        (re.compile(r"\bgst\b", re.I), "gst_registration_check", lambda m: {"annual_turnover": _first_number(m, 0, 2_500_000)}),
        (re.compile(r"\bbudget\b", re.I), "budget_plan", lambda m: {"monthly_income": _first_number(m, 0, 30000)}),
    ]

    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        if has_tool_result:
            last_tool_msg = next(m for m in reversed(messages) if m.get("role") == "tool")
            return LLMResponse(content=(
                f"[mock-llm] Based on the tool result, here's what I found: {last_tool_msg['content'][:300]}"
            ))

        for pattern, tool_name, arg_builder in self._KEYWORD_TOOL_MAP:
            if pattern.search(last_user):
                return LLMResponse(tool_calls=[ToolCall(id="mock-1", name=tool_name, arguments=arg_builder(last_user))])

        return LLMResponse(content=f"[mock-llm] I heard: '{last_user}'. (No real model connected — "
                                    f"set LLM_BACKEND=ollama and run Ollama to get real answers.)")


def _first_number(text: str, index: int, default: float) -> float:
    nums = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    try:
        return float(nums[index])
    except (IndexError, ValueError):
        return default
