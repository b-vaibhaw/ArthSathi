"""
tools.py
========
The "financial model agent" layer: every calculation or lookup the LLM is
allowed to use is declared here as a tool with a strict JSON schema, and
dispatched to the deterministic engines in app/engines and app/news.

Design rule: the LLM NEVER computes a financial number itself — it always
calls one of these tools and reports the tool's result. This is what makes
the numbers trustworthy regardless of which open-weight model is plugged in.
"""

from typing import Dict, List, Optional, Any

from app.engines.debt_engine import DebtEngine, Debt
from app.engines.tax_engine import IndianTaxEngine
from app.engines.business_engine import BusinessEngine
from app.engines.investment_engine import InvestmentEngine
from app.news import market_data


# ─────────────────────────────────────────────────────────────
# Tool schemas (OpenAI-compatible "function calling" format —
# supported by Ollama for tool-calling models: llama3.1, qwen2.5, mistral, etc.)
# ─────────────────────────────────────────────────────────────

def _tool(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_DEBT_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "principal": {"type": "number", "description": "Outstanding balance"},
        "annual_rate": {"type": "number", "description": "Annual interest rate, percent"},
        "min_payment": {"type": "number", "description": "Minimum monthly payment"},
        "lender_type": {"type": "string", "enum": ["bank", "nbfc", "moneylender", "medical", "payday", "friend"]},
    },
    "required": ["name", "principal", "annual_rate", "min_payment"],
}

TOOL_SCHEMAS: List[Dict] = [
    _tool("emi_calculator",
          "Calculate the EMI (equated monthly installment) for a loan.",
          {"principal": {"type": "number"}, "annual_rate": {"type": "number"},
           "tenure_months": {"type": "integer"}},
          ["principal", "annual_rate", "tenure_months"]),

    _tool("debt_payoff_strategy",
          "Given a user's list of debts, compute avalanche vs snowball payoff order, "
          "total interest for each, and a recommendation. ALWAYS use this instead of guessing payoff order.",
          {"debts": {"type": "array", "items": _DEBT_ITEM_SCHEMA},
           "extra_monthly_payment": {"type": "number", "description": "Extra cash available beyond minimums, default 0"},
           "monthly_income": {"type": "number", "description": "User's monthly income, for crisis detection, default 0"}},
          ["debts"]),

    _tool("debt_negotiation_brief",
          "Get structured facts (settlement range, legal options, hardship level) for ONE debt, to use "
          "when writing a negotiation script for the user to send their lender.",
          {"debt": _DEBT_ITEM_SCHEMA, "user_income": {"type": "number"}},
          ["debt", "user_income"]),

    _tool("calculate_income_tax",
          "Calculate Indian income tax under the new or old regime for a given annual income.",
          {"annual_income": {"type": "number"}, "regime": {"type": "string", "enum": ["new", "old"]},
           "age": {"type": "integer"}, "deductions": {"type": "number", "description": "Only used under old regime"}},
          ["annual_income"]),

    _tool("presumptive_tax_itr4",
          "Calculate presumptive taxable income and tax for a small business under ITR-4 (Section 44AD), "
          "comparing cash vs digital receipts.",
          {"gross_receipts": {"type": "number"}}, ["gross_receipts"]),

    _tool("itr_form_selector",
          "Recommend which Indian ITR form a user should file based on their income sources.",
          {"income_sources": {"type": "array", "items": {"type": "string",
                                                           "enum": ["salary", "business", "capital_gain", "other"]}},
           "business_turnover": {"type": "number"}},
          ["income_sources"]),

    _tool("gst_registration_check",
          "Check whether a business needs GST registration based on annual turnover, and estimate GST payable.",
          {"annual_turnover": {"type": "number"}, "state": {"type": "string"},
           "supplies_type": {"type": "string", "enum": ["goods", "services"]}},
          ["annual_turnover"]),

    _tool("suggest_business_price",
          "Suggest a selling price for a product/service given costs, target margin, and competition level.",
          {"cost_of_goods": {"type": "number"}, "operating_expenses": {"type": "number"},
           "target_margin_pct": {"type": "number"},
           "competition": {"type": "string", "enum": ["low", "medium", "high"]},
           "business_type": {"type": "string", "enum": ["retail", "service", "manufacturing", "food"]}},
          ["cost_of_goods", "operating_expenses"]),

    _tool("parse_transaction",
          "Parse a short free-text note (e.g. 'sold vegetables for 500, spent 200 on petrol') into a "
          "structured income/expense transaction. Always confirm the parsed result with the user before saving it.",
          {"text": {"type": "string"}, "lang": {"type": "string"}}, ["text"]),

    _tool("sip_future_value",
          "Project the future value of a monthly SIP (systematic investment), compounded monthly.",
          {"monthly_investment": {"type": "number"}, "annual_return_pct": {"type": "number"},
           "years": {"type": "number"}},
          ["monthly_investment", "annual_return_pct", "years"]),

    _tool("lumpsum_future_value",
          "Project the future value of a one-time lumpsum investment.",
          {"principal": {"type": "number"}, "annual_return_pct": {"type": "number"}, "years": {"type": "number"}},
          ["principal", "annual_return_pct", "years"]),

    _tool("required_sip_for_goal",
          "Compute the monthly SIP amount needed to reach a target corpus by a target date.",
          {"target_amount": {"type": "number"}, "annual_return_pct": {"type": "number"}, "years": {"type": "number"}},
          ["target_amount", "annual_return_pct", "years"]),

    _tool("budget_plan",
          "Produce a monthly budget allocation (needs/wants/savings) for a given income.",
          {"monthly_income": {"type": "number"}, "style": {"type": "string", "enum": ["50_30_20", "aggressive_save"]}},
          ["monthly_income"]),

    _tool("emergency_fund_target",
          "Compute the recommended emergency fund size given monthly expenses.",
          {"monthly_expenses": {"type": "number"}, "months_cover": {"type": "integer"}},
          ["monthly_expenses"]),

    _tool("loan_affordability",
          "Compute how much additional EMI a user can safely take on, given income and existing EMI.",
          {"monthly_income": {"type": "number"}, "existing_emi": {"type": "number"},
           "max_dti_pct": {"type": "number", "description": "Max safe debt-to-income %, default 40"}},
          ["monthly_income"]),

    _tool("stock_pe_valuation",
          "Rough P/E-multiple fair-value estimate for a stock. This is a simplified heuristic, not a full valuation.",
          {"eps": {"type": "number"}, "industry_pe": {"type": "number"}, "current_price": {"type": "number"}},
          ["eps", "industry_pe"]),

    _tool("simple_dcf",
          "Simplified single-stage discounted cash flow estimate of enterprise value. Always state the "
          "assumptions (growth rate, discount rate) alongside the result — never present it as precise.",
          {"free_cash_flow": {"type": "number"}, "growth_pct": {"type": "number"},
           "discount_rate_pct": {"type": "number"}, "years": {"type": "integer"},
           "terminal_growth_pct": {"type": "number"}},
          ["free_cash_flow", "growth_pct", "discount_rate_pct"]),

    _tool("get_financial_news",
          "Fetch recent financial news headlines. Use this instead of recalling headlines from memory — "
          "news changes daily and the model's training data is stale.",
          {"topic": {"type": "string", "description": "Optional keyword filter"},
           "feed": {"type": "string", "enum": ["india_business", "india_markets", "world_business"]},
           "max_items": {"type": "integer"}},
          []),

    _tool("get_stock_quote",
          "Fetch a current stock/index quote by ticker (Yahoo Finance symbols, e.g. 'RELIANCE.NS', 'AAPL', '^NSEI'). "
          "Use this instead of recalling a price from memory — prices change constantly.",
          {"ticker": {"type": "string"}}, ["ticker"]),
]


def _debts_from_dicts(items: List[Dict]) -> List[Debt]:
    return [Debt(name=d["name"], principal=d["principal"], annual_rate=d["annual_rate"],
                 min_payment=d["min_payment"], lender_type=d.get("lender_type", "bank")) for d in items]


class ToolExecutor:
    """Dispatches a tool-call name + arguments dict to the right engine method."""

    def __init__(self):
        self.debt_engine = DebtEngine()
        self.tax_engine = IndianTaxEngine()
        self.business_engine = BusinessEngine()
        self.investment_engine = InvestmentEngine()

    def dispatch(self, name: str, args: Dict) -> Dict:
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return {"error": f"Unknown tool '{name}'"}
            return handler(args)
        except Exception as e:  # tool errors must come back to the LLM as data, not crash the agent loop
            return {"error": f"{type(e).__name__}: {e}"}

    # ── Debt ─────────────────────────────────────────────────
    def _tool_emi_calculator(self, a):
        return self.debt_engine.emi_calculator(a["principal"], a["annual_rate"], int(a["tenure_months"]))

    def _tool_debt_payoff_strategy(self, a):
        debts = _debts_from_dicts(a["debts"])
        if not debts:
            return {"error": "No debts provided"}
        return self.debt_engine.recommend_strategy(
            debts, extra=a.get("extra_monthly_payment", 0), income=a.get("monthly_income", 0))

    def _tool_debt_negotiation_brief(self, a):
        d = a["debt"]
        debt = Debt(name=d["name"], principal=d["principal"], annual_rate=d["annual_rate"],
                    min_payment=d["min_payment"], lender_type=d.get("lender_type", "bank"))
        return self.debt_engine.generate_negotiation_brief(debt, a["user_income"])

    # ── Tax / GST ────────────────────────────────────────────
    def _tool_calculate_income_tax(self, a):
        return self.tax_engine.calculate_tax(a["annual_income"], a.get("regime", "new"),
                                              a.get("age", 35), a.get("deductions", 0))

    def _tool_presumptive_tax_itr4(self, a):
        return self.tax_engine.presumptive_tax_itr4(a["gross_receipts"])

    def _tool_itr_form_selector(self, a):
        return self.tax_engine.itr_form_selector(a["income_sources"], a.get("business_turnover", 0))

    def _tool_gst_registration_check(self, a):
        return self.tax_engine.gst_decision(a["annual_turnover"], a.get("state", "general"),
                                             a.get("supplies_type", "goods"))

    # ── Business ─────────────────────────────────────────────
    def _tool_suggest_business_price(self, a):
        return self.business_engine.suggest_price(
            a["cost_of_goods"], a["operating_expenses"], a.get("target_margin_pct", 20),
            a.get("competition", "medium"), a.get("business_type", "retail"))

    def _tool_parse_transaction(self, a):
        result = self.business_engine.parse_voice_transaction(a["text"], a.get("lang", "en"))
        return result or {"error": "Could not find an amount in the text"}

    # ── Investment / budgeting ───────────────────────────────
    def _tool_sip_future_value(self, a):
        return self.investment_engine.sip_future_value(a["monthly_investment"], a["annual_return_pct"], a["years"])

    def _tool_lumpsum_future_value(self, a):
        return self.investment_engine.lumpsum_future_value(a["principal"], a["annual_return_pct"], a["years"])

    def _tool_required_sip_for_goal(self, a):
        return self.investment_engine.required_sip_for_goal(a["target_amount"], a["annual_return_pct"], a["years"])

    def _tool_budget_plan(self, a):
        return self.investment_engine.budget_plan(a["monthly_income"], a.get("style", "50_30_20"))

    def _tool_emergency_fund_target(self, a):
        return self.investment_engine.emergency_fund_target(a["monthly_expenses"], a.get("months_cover", 6))

    def _tool_loan_affordability(self, a):
        return self.investment_engine.loan_affordability(a["monthly_income"], a.get("existing_emi", 0),
                                                           a.get("max_dti_pct", 40))

    def _tool_stock_pe_valuation(self, a):
        return self.investment_engine.pe_based_valuation(a["eps"], a["industry_pe"], a.get("current_price"))

    def _tool_simple_dcf(self, a):
        return self.investment_engine.simple_dcf(a["free_cash_flow"], a["growth_pct"], a["discount_rate_pct"],
                                                   a.get("years", 5), a.get("terminal_growth_pct", 3.0))

    # ── News / market data ───────────────────────────────────
    def _tool_get_financial_news(self, a):
        return market_data.get_financial_news(a.get("topic", ""), a.get("feed", "india_business"),
                                                a.get("max_items", 5))

    def _tool_get_stock_quote(self, a):
        return market_data.get_stock_quote(a["ticker"])
