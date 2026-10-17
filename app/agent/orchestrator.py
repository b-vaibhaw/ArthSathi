"""
orchestrator.py
================
The agent loop: builds the prompt (persona + RAG context + user profile +
conversation history), calls the LLM, executes any tool calls it asks for,
feeds the results back, and repeats until the model gives a final answer.

This is the piece that replaces the old hardcoded if/else response tree —
the LLM decides what to say and which tools to call; the tools (not the
model) do every calculation and every lookup of "current" information.
"""

import json
import logging
from typing import Dict, List, Optional

from app.agent.llm_client import LLMClient
from app.agent.tools import TOOL_SCHEMAS, ToolExecutor
from app.rag.engine import RAGEngine

logger = logging.getLogger("arthasathi.agent")

SYSTEM_PROMPT = """You are ArthaSathi (अर्थसाथी — "financial companion"), an independent AI financial \
advisor. You help with TWO kinds of users, often the same person over time:
1. General personal finance: budgeting, saving, investing (SIP/lumpsum), debt payoff, understanding \
markets and financial news, basic company/stock valuation.
2. India-specific micro-finance: debt escape for over-indebted borrowers, GST/ITR/tax guidance for \
informal and micro-businesses, pricing and credit-profile help for micro-entrepreneurs.

STRICT RULES:
- NEVER compute a financial number yourself (EMI, tax, SIP growth, valuation, etc). ALWAYS call the \
matching tool and report its exact result. If no tool fits a calculation the user is asking for, say so \
rather than estimating.
- NEVER state a stock price, index level, or news headline from memory — ALWAYS call get_stock_quote or \
get_financial_news for anything "current." If those tools report unavailable, tell the user plainly \
that live data isn't reachable right now instead of guessing.
- For any valuation tool (stock_pe_valuation, simple_dcf), always state the assumptions used (the \
inputs) alongside the result, and make clear it is a simplified estimate, not investment advice to \
act on alone.
- You are not a licensed financial advisor. For large, hard-to-reverse decisions (big investments, \
loan settlements, legal action), say the user should also consult a qualified professional — but still \
give your best concrete analysis using the tools.
- Reply in the same language the user is writing in. Keep numbers in Rs/₹ when the context is India, \
otherwise use whatever currency the user specifies.
- Be concise and concrete: lead with the actionable answer, use the tool's exact numbers, then a short \
explanation. Avoid filler and avoid repeating the user's question back to them.
"""


class AgentOrchestrator:
    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine, max_iterations: int = 6):
        self.llm_client = llm_client
        self.rag = rag_engine
        self.tools = ToolExecutor()
        self.max_iterations = max_iterations

    def handle_message(self, user_id: str, user_message: str) -> Dict:
        language = self.rag.detect_language(user_message)
        self.rag.memory.get_or_create_user(user_id, language)

        context_block = self.rag.build_context_block(user_id, user_message)
        history = self.rag.memory.get_history(user_id, last_n=10)

        messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context_block:
            messages.append({"role": "system", "content": context_block})
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        tool_trace = []
        final_text = None

        for _ in range(self.max_iterations):
            response = self.llm_client.chat(messages, TOOL_SCHEMAS)

            if not response.tool_calls:
                final_text = response.content or ""
                break

            # Record the assistant's tool-call turn, then execute each tool
            # and feed results back as "tool" messages, per OpenAI-style convention.
            messages.append({
                "role": "assistant", "content": response.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                               for tc in response.tool_calls],
            })
            for tc in response.tool_calls:
                result = self.tools.dispatch(tc.name, tc.arguments)
                tool_trace.append({"tool": tc.name, "args": tc.arguments, "result": result})
                logger.info("tool_call name=%s args=%s", tc.name, tc.arguments)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "name": tc.name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            final_text = ("I ran into trouble completing that — too many steps were needed. "
                           "Could you break your question into smaller parts?")

        self.rag.save_user_message(user_id, user_message, language)
        self.rag.save_assistant_message(user_id, final_text, language)

        return {"reply": final_text, "language": language, "tool_calls": tool_trace}
