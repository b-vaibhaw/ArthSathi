"""
advisor.py
==========
Thin convenience layer that instantiates all engines together and exposes
a financial-health summary. The LLM agent (app/agent) calls individual
engine methods as *tools*; this module is mainly used by the API layer for
one-shot "analyze this user" summaries (e.g. a dashboard endpoint).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from app.engines.debt_engine import DebtEngine, Debt
from app.engines.tax_engine import IndianTaxEngine
from app.engines.business_engine import BusinessEngine, Transaction
from app.engines.investment_engine import InvestmentEngine


@dataclass
class UserFinancialProfile:
    user_id: str
    monthly_income: float = 0.0
    debts: List[Debt] = field(default_factory=list)
    transactions: List[Transaction] = field(default_factory=list)
    language: str = "en"


class ArthaSathiAdvisor:
    def __init__(self):
        self.debt_engine = DebtEngine()
        self.tax_engine = IndianTaxEngine()
        self.business_engine = BusinessEngine()
        self.investment_engine = InvestmentEngine()

    def analyze_financial_health(self, profile: UserFinancialProfile) -> Dict:
        monthly_income = profile.monthly_income
        total_debt = sum(d.principal for d in profile.debts)
        total_min_emi = sum(d.min_payment for d in profile.debts)
        debt_to_income = total_min_emi / max(monthly_income, 1)

        score = 100
        if debt_to_income > 0.50:
            score -= 40
        elif debt_to_income > 0.30:
            score -= 20
        elif debt_to_income > 0.15:
            score -= 10
        if total_debt > monthly_income * 12:
            score -= 20
        elif total_debt > monthly_income * 6:
            score -= 10
        if not profile.debts:
            score = min(score + 20, 100)

        health_level = ("CRITICAL" if score < 30 else "POOR" if score < 50 else
                         "FAIR" if score < 70 else "GOOD" if score < 85 else "EXCELLENT")

        recommendations = []
        if debt_to_income > 0.40:
            recommendations.append("Contact lenders now for restructuring — payments exceed a safe share of income.")
        if any(d.annual_rate > 30 for d in profile.debts):
            recommendations.append("High-interest debt (>30%/yr) detected — prioritize paying this first.")
        if monthly_income > 0 and not profile.debts:
            recommendations.append(f"No debt — consider building an emergency fund of Rs {round(monthly_income * 3):,}+.")

        strategy = None
        if profile.debts:
            strategy = self.debt_engine.recommend_strategy(
                profile.debts, extra=max(0, monthly_income * 0.10), income=monthly_income)

        return {
            "health_score": score, "health_level": health_level,
            "summary": {
                "monthly_income": round(monthly_income, 0), "total_debt": round(total_debt, 0),
                "total_monthly_emi": round(total_min_emi, 0), "debt_to_income_pct": round(debt_to_income * 100, 1),
                "n_debts": len(profile.debts),
            },
            "recommendations": recommendations, "debt_strategy": strategy,
        }
