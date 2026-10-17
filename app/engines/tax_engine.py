"""
tax_engine.py
=============
Indian income tax calculation (new/old regime, FY2023-24 slabs), ITR form
selection, and GST registration decisions. Deterministic — the LLM narrates,
this computes.

NOTE: Tax slabs change by Finance Act each year. These are FY2023-24 (AY2024-25)
slabs. Update SLABS below (or wire to a RAG doc) when a new Budget changes them —
do not let the LLM guess current slabs from training data, which goes stale.
"""

from typing import List, Dict


class IndianTaxEngine:
    NEW_REGIME_SLABS = [
        (300_000, 0.00), (600_000, 0.05), (900_000, 0.10),
        (1_200_000, 0.15), (1_500_000, 0.20), (float("inf"), 0.30),
    ]
    OLD_REGIME_SLABS = [
        (250_000, 0.00), (500_000, 0.05), (1_000_000, 0.20), (float("inf"), 0.30),
    ]

    def calculate_tax(self, annual_income: float, regime: str = "new",
                       age: int = 35, deductions: float = 0) -> Dict:
        slabs = self.NEW_REGIME_SLABS if regime == "new" else self.OLD_REGIME_SLABS
        if regime == "old" and age >= 60:
            slabs = [(300_000 if age < 80 else 500_000, 0.00)] + slabs[1:]

        taxable_income = max(0, annual_income - (deductions if regime == "old" else 0))
        rebate_limit = 700_000 if regime == "new" else 500_000

        tax, prev_limit, breakdown = 0.0, 0, []
        for limit, rate in slabs:
            if taxable_income <= prev_limit:
                break
            slab_income = min(taxable_income, limit) - prev_limit
            slab_tax = slab_income * rate
            if slab_income > 0:
                breakdown.append({
                    "range": f"{prev_limit:,.0f} - {'above' if limit == float('inf') else f'{limit:,.0f}'}",
                    "rate": f"{int(rate * 100)}%", "income": round(slab_income, 0), "tax": round(slab_tax, 0),
                })
            tax += slab_tax
            prev_limit = limit

        rebate_max = 25_000 if regime == "new" else 12_500
        rebate = min(tax, rebate_max) if taxable_income <= rebate_limit else 0
        tax = max(0, tax - rebate)
        cess = tax * 0.04
        total_tax = tax + cess

        return {
            "annual_income": round(annual_income, 0),
            "taxable_income": round(taxable_income, 0),
            "regime": regime,
            "gross_tax": round(tax, 0),
            "rebate_87A": round(rebate, 0),
            "education_cess": round(cess, 0),
            "total_tax": round(total_tax, 0),
            "effective_rate_pct": round(total_tax / max(annual_income, 1) * 100, 1),
            "monthly_tax": round(total_tax / 12, 0),
            "tax_breakdown": breakdown,
        }

    def itr_form_selector(self, income_sources: List[str], business_turnover: float = 0) -> Dict:
        has_business = "business" in income_sources
        has_capital_gain = "capital_gain" in income_sources

        if has_capital_gain or (has_business and business_turnover > 5_000_000):
            form, rationale = "ITR-3", "Capital gains or large business income — a CA should review this."
        elif has_business and business_turnover <= 5_000_000:
            form = "ITR-4"
            rationale = "Presumptive taxation (Section 44AD) — declare a flat % of turnover as income, no books/audit needed."
        elif "salary" in income_sources and not has_business:
            form, rationale = "ITR-1", "Simple salaried return — pre-filled from Form 16."
        else:
            form, rationale = "ITR-1", "Start with ITR-1; the e-filing portal will redirect if it doesn't fit."

        return {
            "recommended_form": form, "rationale": rationale, "income_sources": income_sources,
            "filing_deadline": "July 31 of the assessment year",
            "penalty_for_late_filing": "Rs 5,000 (Rs 1,000 if total income below Rs 5 lakh)",
            "free_filing_portal": "incometax.gov.in",
        }

    def presumptive_tax_itr4(self, gross_receipts: float) -> Dict:
        """Section 44AD: declare 8% of turnover as income (6% for digital receipts)."""
        presumptive_cash = gross_receipts * 0.08
        presumptive_digital = gross_receipts * 0.06
        tax_cash = self.calculate_tax(presumptive_cash)
        tax_digital = self.calculate_tax(presumptive_digital)
        return {
            "gross_receipts": round(gross_receipts, 0),
            "presumptive_income_if_cash": round(presumptive_cash, 0),
            "presumptive_income_if_digital": round(presumptive_digital, 0),
            "tax_if_cash": tax_cash["total_tax"],
            "tax_if_digital": tax_digital["total_tax"],
            "savings_by_going_digital": round(tax_cash["total_tax"] - tax_digital["total_tax"], 0),
            "tip": "Accept UPI/card payments to declare only 6% of turnover as income instead of 8%.",
        }

    def gst_decision(self, annual_turnover: float, state: str = "general", supplies_type: str = "goods") -> Dict:
        special_states = {"manipur", "mizoram", "nagaland", "tripura", "sikkim",
                           "meghalaya", "arunachal", "uttarakhand", "himachal"}
        if state.lower() in special_states:
            mandatory, composition = 1_000_000, 7_500_000
        elif supplies_type == "services":
            mandatory, composition = 2_000_000, 5_000_000
        else:
            mandatory, composition = 4_000_000, 15_000_000

        if annual_turnover < mandatory:
            status, rate = "NOT_REQUIRED", None
            advice = "Below threshold — no GST registration needed."
        elif annual_turnover <= composition:
            status = "OPTIONAL_COMPOSITION"
            rate = 1.0 if supplies_type == "goods" else 6.0
            advice = f"Optional Composition Scheme: pay a flat {rate}% GST on turnover with simple quarterly filing."
        else:
            status, rate = "MANDATORY", 18.0
            advice = "Mandatory GST registration — get a tax practitioner to file returns."

        gst_payable = (annual_turnover * (rate or 0) / 100) if rate else 0
        return {
            "annual_turnover": round(annual_turnover, 0), "status": status, "gst_rate_pct": rate,
            "advice": advice, "estimated_annual_gst": round(gst_payable, 0),
            "next_step": "Register at gst.gov.in or visit the nearest Jan Seva Kendra.",
        }
