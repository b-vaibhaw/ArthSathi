"""
debt_engine.py
==============
Debt payoff math: EMI calculation, avalanche/snowball strategies,
and negotiation briefs. Pure deterministic math — safe to trust.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Debt:
    name: str
    principal: float        # Outstanding balance
    annual_rate: float      # Annual interest rate (%)
    min_payment: float      # Minimum monthly payment
    lender_type: str = "bank"   # bank | nbfc | moneylender | medical | payday | friend
    loan_id: str = ""

    def __post_init__(self):
        if not self.loan_id:
            self.loan_id = self.name.replace(" ", "_").lower()[:20]


class DebtEngine:
    """All debt-related calculations and strategy generation."""

    @staticmethod
    def emi_calculator(principal: float, annual_rate: float, tenure_months: int) -> Dict:
        """EMI = P * r * (1+r)^n / ((1+r)^n - 1)."""
        if tenure_months <= 0:
            return {"error": "tenure_months must be positive"}
        if annual_rate == 0:
            emi = principal / tenure_months
            return {
                "emi": round(emi, 2),
                "total_amount": round(principal, 2),
                "total_interest": 0.0,
                "principal": round(principal, 2),
                "tenure_months": tenure_months,
                "annual_rate": annual_rate,
            }
        r = annual_rate / 12 / 100
        n = tenure_months
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
        total = emi * n
        return {
            "emi": round(emi, 2),
            "total_amount": round(total, 2),
            "total_interest": round(total - principal, 2),
            "principal": round(principal, 2),
            "tenure_months": tenure_months,
            "annual_rate": annual_rate,
        }

    @staticmethod
    def months_to_payoff(principal: float, annual_rate: float, monthly_payment: float) -> float:
        if monthly_payment <= 0:
            return float("inf")
        r = annual_rate / 12 / 100
        if r == 0:
            return math.ceil(principal / monthly_payment)
        if monthly_payment <= principal * r:
            return float("inf")  # payment too small — debt grows forever
        n = -math.log(1 - (r * principal / monthly_payment)) / math.log(1 + r)
        return max(0.0, n)

    @staticmethod
    def total_interest_paid(principal: float, annual_rate: float, months: float) -> float:
        if months == float("inf") or months > 1200:
            return float("inf")
        r = annual_rate / 12 / 100
        return principal * ((1 + r) ** months - 1)

    def _strategy(self, debts: List[Debt], extra_payment: float, sort_key, name: str, rationale: str) -> Dict:
        sorted_debts = sorted(debts, key=sort_key)
        total_min = sum(d.min_payment for d in debts)
        plan = []
        total_interest = 0.0
        for i, debt in enumerate(sorted_debts):
            extra = extra_payment if i == 0 else 0
            pay = debt.min_payment + extra
            months = self.months_to_payoff(debt.principal, debt.annual_rate, pay)
            capped_months = 9999 if months == float("inf") else months
            interest = self.total_interest_paid(debt.principal, debt.annual_rate, capped_months)
            capped_interest = debt.principal * 100 if interest == float("inf") else interest
            total_interest += capped_interest
            plan.append({
                "rank": i + 1,
                "debt_name": debt.name,
                "balance": round(debt.principal, 2),
                "rate": debt.annual_rate,
                "monthly_pay": round(pay, 2),
                "months_to_free": round(capped_months, 1) if capped_months < 9999 else "unpayable_at_this_rate",
                "interest_paid": round(capped_interest, 2),
            })
        return {
            "strategy": name,
            "rationale": rationale,
            "priority_order": plan,
            "total_min_payment": round(total_min, 2),
            "total_interest": round(total_interest, 2),
        }

    def avalanche_strategy(self, debts: List[Debt], extra_payment: float = 0) -> Dict:
        """Highest interest rate first. Mathematically minimizes total interest paid."""
        return self._strategy(
            debts, extra_payment, lambda d: -d.annual_rate,
            "avalanche", "Highest interest rate first — saves the most money overall.",
        )

    def snowball_strategy(self, debts: List[Debt], extra_payment: float = 0) -> Dict:
        """Smallest balance first. Psychologically motivating — quick wins."""
        return self._strategy(
            debts, extra_payment, lambda d: d.principal,
            "snowball", "Smallest balance first — quick wins build momentum.",
        )

    def recommend_strategy(self, debts: List[Debt], extra: float = 0, income: float = 0) -> Dict:
        if not debts:
            return {"error": "No debts provided"}
        av = self.avalanche_strategy(debts, extra)
        sn = self.snowball_strategy(debts, extra)
        interest_saving = sn["total_interest"] - av["total_interest"]

        total_min = sum(d.min_payment for d in debts)
        is_crisis = income > 0 and (total_min / income) > 0.40

        recommendation, reason = "avalanche", f"Saves the most money (Rs {interest_saving:,.0f} less interest)."
        if is_crisis:
            recommendation, reason = "crisis", "DEBT CRISIS: monthly minimum payments exceed 40% of income. Needs restructuring, not just a payoff order."
        smallest_ratio = min(d.principal for d in debts) / max(max(d.principal for d in debts), 1)
        if smallest_ratio < 0.2 and interest_saving < 5000 and not is_crisis:
            recommendation, reason = "snowball", "Quick win: smallest debt can be cleared fast for very little extra interest cost — good for motivation."

        return {
            "recommended": recommendation,
            "reason": reason,
            "avalanche_result": av,
            "snowball_result": sn,
            "interest_savings_with_avalanche": round(interest_saving, 2),
            "is_crisis": is_crisis,
            "debt_to_income_pct": round(total_min / max(income, 1) * 100, 1) if income > 0 else None,
        }

    def generate_negotiation_brief(self, debt: Debt, user_income: float) -> Dict:
        """Structured facts for the LLM to turn into an actual negotiation script in the user's language."""
        debt_to_income = debt.min_payment / max(user_income, 1)
        hardship_level = "severe" if debt_to_income > 0.40 else "moderate" if debt_to_income > 0.25 else "mild"

        settlement_ranges = {
            "bank": (0.60, 0.85), "nbfc": (0.50, 0.75), "moneylender": (0.40, 0.70),
            "medical": (0.30, 0.60), "payday": (0.40, 0.65), "friend": (0.80, 1.00),
        }
        low, high = settlement_ranges.get(debt.lender_type, (0.60, 0.85))

        legal_options = {
            "bank": ["RBI Banking Ombudsman", "Debt Recovery Tribunal (DRT)", "IBC Section 8 notice"],
            "nbfc": ["RBI NBFC Ombudsman complaint", "Consumer Forum"],
            "moneylender": ["State Moneylenders Act usury complaint", "Police complaint for excessive rates"],
            "medical": ["Hospital financial-assistance program", "State health scheme coverage"],
            "payday": ["RBI Digital Lending Guidelines complaint", "Fair Practices Code complaint"],
        }

        asks = []
        if hardship_level in ("severe", "moderate"):
            asks += ["EMI restructuring (lower monthly payment)", "Moratorium (temporary payment pause)"]
        asks += ["Interest rate reduction", "One-time settlement at a reduced amount"]
        if debt.lender_type == "bank":
            asks.append("Consolidation into a lower-rate personal loan")

        return {
            "debt_name": debt.name,
            "outstanding": round(debt.principal, 2),
            "current_rate": debt.annual_rate,
            "lender_type": debt.lender_type,
            "hardship_level": hardship_level,
            "debt_to_income_pct": round(debt_to_income * 100, 1),
            "settlement_range": {
                "low_offer": round(debt.principal * low, 0),
                "high_offer": round(debt.principal * high, 0),
            },
            "negotiation_asks": asks,
            "legal_options": legal_options.get(debt.lender_type, ["Consumer Forum"]),
            "tone_guidance": "Polite but firm. Cite your rights. Never reveal your walk-away number first.",
        }
