"""
investment_engine.py
=====================
General personal-finance "financial model" tools: SIP / compound-growth
projections, a 50/30/20-style budget planner, loan-affordability check,
and a simple equity valuation helper (P/E based + basic DCF). These extend
ArthaSathi beyond debt/micro-business into broad personal finance, for
users who want investing and planning help rather than debt rescue.

All deterministic math. Valuation helpers are explicitly simplified models —
the engine always returns the assumptions used so the LLM states them as
assumptions, never as facts about a real company.
"""

import math
from typing import Dict, List, Optional


class InvestmentEngine:
    # ── SIP / compound growth ───────────────────────────────

    def sip_future_value(self, monthly_investment: float, annual_return_pct: float,
                          years: float) -> Dict:
        """Future value of a monthly SIP (systematic investment plan),
        compounded monthly. FV = P * [((1+r)^n - 1) / r] * (1+r)."""
        n = round(years * 12)
        r = annual_return_pct / 12 / 100
        if n <= 0:
            return {"error": "years must be positive"}
        if r == 0:
            future_value = monthly_investment * n
        else:
            future_value = monthly_investment * (((1 + r) ** n - 1) / r) * (1 + r)
        invested = monthly_investment * n
        return {
            "monthly_investment": round(monthly_investment, 2),
            "years": years,
            "assumed_annual_return_pct": annual_return_pct,
            "total_invested": round(invested, 2),
            "future_value": round(future_value, 2),
            "wealth_gained": round(future_value - invested, 2),
        }

    def lumpsum_future_value(self, principal: float, annual_return_pct: float, years: float) -> Dict:
        future_value = principal * (1 + annual_return_pct / 100) ** years
        return {
            "principal": round(principal, 2), "years": years,
            "assumed_annual_return_pct": annual_return_pct,
            "future_value": round(future_value, 2),
            "wealth_gained": round(future_value - principal, 2),
        }

    def required_sip_for_goal(self, target_amount: float, annual_return_pct: float, years: float) -> Dict:
        """How much to invest monthly to reach a target corpus."""
        n = round(years * 12)
        r = annual_return_pct / 12 / 100
        if n <= 0:
            return {"error": "years must be positive"}
        if r == 0:
            monthly = target_amount / n
        else:
            monthly = target_amount / ((((1 + r) ** n - 1) / r) * (1 + r))
        return {
            "target_amount": round(target_amount, 2), "years": years,
            "assumed_annual_return_pct": annual_return_pct,
            "required_monthly_sip": round(monthly, 2),
        }

    # ── Budgeting ────────────────────────────────────────────

    def budget_plan(self, monthly_income: float, style: str = "50_30_20") -> Dict:
        """50/30/20 rule: 50% needs, 30% wants, 20% savings/debt payoff.
        Alternate 'aggressive_save' style front-loads savings for debt payoff
        or fast goal funding."""
        if style == "aggressive_save":
            splits = {"needs": 0.50, "wants": 0.15, "savings_or_debt": 0.35}
        else:
            splits = {"needs": 0.50, "wants": 0.30, "savings_or_debt": 0.20}
        plan = {k: round(monthly_income * v, 2) for k, v in splits.items()}
        return {
            "monthly_income": round(monthly_income, 2), "style": style,
            "allocation": plan,
            "note": "Needs = rent/food/EMIs/utilities. Wants = discretionary spending. "
                    "Savings_or_debt = investments, emergency fund, or extra debt payments.",
        }

    def emergency_fund_target(self, monthly_expenses: float, months_cover: int = 6) -> Dict:
        target = monthly_expenses * months_cover
        return {
            "monthly_expenses": round(monthly_expenses, 2), "months_of_cover": months_cover,
            "target_emergency_fund": round(target, 2),
            "tip": "Keep this in a liquid instrument (savings account / liquid mutual fund), not locked investments.",
        }

    def loan_affordability(self, monthly_income: float, existing_emi: float = 0,
                            max_dti_pct: float = 40) -> Dict:
        """How much additional EMI can this income safely support, at a given
        debt-to-income cap (40% is a common lender/affordability guideline)."""
        max_total_emi = monthly_income * max_dti_pct / 100
        headroom = max(0, max_total_emi - existing_emi)
        return {
            "monthly_income": round(monthly_income, 2), "existing_emi": round(existing_emi, 2),
            "max_dti_pct": max_dti_pct,
            "max_affordable_total_emi": round(max_total_emi, 2),
            "additional_emi_headroom": round(headroom, 2),
        }

    # ── Simple equity valuation helpers ─────────────────────

    def pe_based_valuation(self, eps: float, industry_pe: float, current_price: Optional[float] = None) -> Dict:
        """Very simplified fair-value estimate: fair_price = EPS x industry/peer P/E.
        This is a rough heuristic, not a substitute for full fundamental analysis."""
        fair_value = eps * industry_pe
        result = {
            "eps": eps, "industry_or_peer_pe": industry_pe,
            "estimated_fair_value": round(fair_value, 2),
            "method": "P/E multiple (simplified) — NOT a full valuation",
        }
        if current_price is not None:
            result["current_price"] = current_price
            result["current_pe"] = round(current_price / eps, 2) if eps else None
            result["verdict"] = (
                "appears undervalued vs. peers" if current_price < fair_value * 0.9 else
                "appears overvalued vs. peers" if current_price > fair_value * 1.1 else
                "roughly in line with peers"
            )
        return result

    def simple_dcf(self, free_cash_flow: float, growth_pct: float, discount_rate_pct: float,
                    years: int = 5, terminal_growth_pct: float = 3.0) -> Dict:
        """Very simplified single-stage-then-terminal DCF of free cash flow.
        Intended for back-of-envelope estimates and teaching the concept —
        always state the assumptions, never present the output as precise."""
        g = growth_pct / 100
        d = discount_rate_pct / 100
        tg = terminal_growth_pct / 100
        if d <= tg:
            return {"error": "discount_rate must exceed terminal_growth for a stable terminal value"}

        pv_sum = 0.0
        fcf = free_cash_flow
        yearly = []
        for year in range(1, years + 1):
            fcf = fcf * (1 + g)
            pv = fcf / (1 + d) ** year
            pv_sum += pv
            yearly.append({"year": year, "projected_fcf": round(fcf, 2), "present_value": round(pv, 2)})

        terminal_value = fcf * (1 + tg) / (d - tg)
        pv_terminal = terminal_value / (1 + d) ** years
        enterprise_value = pv_sum + pv_terminal

        return {
            "inputs": {"starting_fcf": free_cash_flow, "growth_pct": growth_pct,
                       "discount_rate_pct": discount_rate_pct, "years": years,
                       "terminal_growth_pct": terminal_growth_pct},
            "yearly_projection": yearly,
            "pv_of_projection_period": round(pv_sum, 2),
            "terminal_value": round(terminal_value, 2),
            "pv_of_terminal_value": round(pv_terminal, 2),
            "estimated_enterprise_value": round(enterprise_value, 2),
            "caveat": "Simplified single-stage DCF. Sensitive to growth/discount-rate assumptions — "
                      "treat as a rough estimate, not a precise valuation.",
        }
