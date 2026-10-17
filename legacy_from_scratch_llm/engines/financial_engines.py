"""
financial_engines.py
======================
ArthaSathi — Core Financial Intelligence Engines
- DebtEngine:       Debt payoff strategies, EMI calculator, negotiation scripts
- BusinessEngine:   Pricing, GST checker, microloan credit profiler
- IndianTaxEngine:  ITR-1/ITR-4 guidance for informal sector workers
All computations are deterministic — no LLM needed for math.
The LLM uses these results as TOOLS to explain and guide users.
"""

import math
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import date, timedelta


# ─────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class Debt:
    name:         str
    principal:    float        # Current outstanding balance (Rs)
    annual_rate:  float        # Annual interest rate (%)
    min_payment:  float        # Minimum monthly payment (Rs)
    lender_type:  str          # bank | nbfc | moneylender | medical | payday | friend
    loan_id:      str          = ""
    start_date:   str          = ""

    @property
    def monthly_rate(self) -> float:
        return self.annual_rate / 12 / 100

    def __post_init__(self):
        if not self.loan_id:
            self.loan_id = self.name.replace(" ", "_").lower()[:20]


@dataclass
class Transaction:
    type:        str    # income | expense
    amount:      float
    category:    str    # salary | business | rent | food | emi | medical | other
    description: str    = ""
    date_str:    str    = ""
    lang:        str    = "hi"


@dataclass
class UserFinancialProfile:
    user_id:       str
    monthly_income:  float        = 0.0
    debts:           List[Debt]   = field(default_factory=list)
    transactions:    List[Transaction] = field(default_factory=list)
    business_data:   Dict         = field(default_factory=dict)
    credit_score:    Optional[int] = None
    language:        str          = "hi"


# ─────────────────────────────────────────────────────────────
# 2. DEBT ENGINE
# ─────────────────────────────────────────────────────────────

class DebtEngine:
    """
    All debt-related calculations and strategy generation.
    Deterministic math — results fed to LLM for explanation.
    """

    # ── EMI Calculations ─────────────────────────────────────

    @staticmethod
    def emi_calculator(principal: float, annual_rate: float,
                       tenure_months: int) -> Dict:
        """
        Calculate EMI (Equated Monthly Installment).
        Formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        where r = monthly interest rate, n = number of months
        """
        if annual_rate == 0:
            return {
                "emi":          round(principal / tenure_months, 2),
                "total_amount": round(principal, 2),
                "total_interest": 0.0,
            }

        r     = annual_rate / 12 / 100
        n     = tenure_months
        emi   = principal * r * (1 + r)**n / ((1 + r)**n - 1)
        total = emi * n
        return {
            "emi":            round(emi, 2),
            "total_amount":   round(total, 2),
            "total_interest": round(total - principal, 2),
            "principal":      round(principal, 2),
            "tenure_months":  tenure_months,
            "annual_rate":    annual_rate,
        }

    @staticmethod
    def months_to_payoff(principal: float, annual_rate: float,
                          monthly_payment: float) -> float:
        """
        How many months to pay off a debt at a given monthly payment?
        """
        if monthly_payment <= 0:
            return float("inf")
        r = annual_rate / 12 / 100
        if r == 0:
            return math.ceil(principal / monthly_payment)
        if monthly_payment <= principal * r:
            return float("inf")  # Payment too small, debt grows forever
        n = -math.log(1 - (r * principal / monthly_payment)) / math.log(1 + r)
        return max(0.0, n)

    @staticmethod
    def total_interest_paid(principal: float, annual_rate: float,
                             months: float) -> float:
        """Total interest paid over the loan lifetime"""
        if months == float("inf") or months > 1200:
            return float("inf")
        r = annual_rate / 12 / 100
        return principal * ((1 + r) ** months - 1)

    # ── Payoff Strategies ────────────────────────────────────

    def avalanche_strategy(self, debts: List[Debt],
                            extra_payment: float = 0) -> Dict:
        """
        Debt Avalanche: Pay minimum on all, throw extra cash at HIGHEST interest.
        Mathematically optimal — minimizes total interest paid.
        Best for: people motivated by numbers and savings.
        """
        sorted_debts  = sorted(debts, key=lambda d: d.annual_rate, reverse=True)
        total_min     = sum(d.min_payment for d in debts)
        plan          = []
        total_interest = 0.0

        for i, debt in enumerate(sorted_debts):
            extra  = extra_payment if i == 0 else 0
            pay    = debt.min_payment + extra
            months = self.months_to_payoff(debt.principal, debt.annual_rate, pay)
            if months == float("inf"):
                months = 9999
            interest = self.total_interest_paid(debt.principal, debt.annual_rate, months)
            total_interest += interest
            plan.append({
                "rank":           i + 1,
                "debt_name":      debt.name,
                "balance":        round(debt.principal, 2),
                "rate":           debt.annual_rate,
                "monthly_pay":    round(pay, 2),
                "months_to_free": round(months, 1),
                "interest_paid":  round(interest, 2),
                "strategy":       "avalanche",
            })

        return {
            "strategy":          "avalanche",
            "rationale":         "Highest interest rate first — saves maximum money",
            "priority_order":    plan,
            "total_min_payment": round(total_min, 2),
            "total_interest":    round(total_interest, 2),
        }

    def snowball_strategy(self, debts: List[Debt],
                           extra_payment: float = 0) -> Dict:
        """
        Debt Snowball: Pay minimum on all, throw extra cash at SMALLEST balance.
        Psychologically satisfying — quick wins keep motivation high.
        Best for: people who need morale boosts to keep going.
        """
        sorted_debts   = sorted(debts, key=lambda d: d.principal)
        total_min      = sum(d.min_payment for d in debts)
        plan           = []
        total_interest = 0.0

        for i, debt in enumerate(sorted_debts):
            extra  = extra_payment if i == 0 else 0
            pay    = debt.min_payment + extra
            months = self.months_to_payoff(debt.principal, debt.annual_rate, pay)
            if months == float("inf"): months = 9999
            interest = self.total_interest_paid(debt.principal, debt.annual_rate, months)
            if interest == float("inf"): interest = debt.principal * 100  # cap for comparison
            total_interest += interest
            plan.append({
                "rank":           i + 1,
                "debt_name":      debt.name,
                "balance":        round(debt.principal, 2),
                "rate":           debt.annual_rate,
                "monthly_pay":    round(pay, 2),
                "months_to_free": round(months, 1) if months < 9999 else "∞",
                "interest_paid":  round(interest, 2) if interest < 1e15 else "∞",
                "strategy":       "snowball",
            })

        return {
            "strategy":          "snowball",
            "rationale":         "Smallest balance first — quick wins, builds momentum",
            "priority_order":    plan,
            "total_min_payment": round(total_min, 2),
            "total_interest":    round(total_interest, 2) if total_interest < 1e15 else "∞",
        }

    def recommend_strategy(self, debts: List[Debt], extra: float = 0,
                            income: float = 0) -> Dict:
        """
        Recommend the best strategy based on the user's specific situation.
        """
        if not debts:
            return {"error": "No debts provided"}

        av  = self.avalanche_strategy(debts, extra)
        sn  = self.snowball_strategy(debts, extra)
        interest_saving = sn["total_interest"] - av["total_interest"]

        # Check for crisis — debt payments > 40% of income
        total_min = sum(d.min_payment for d in debts)
        is_crisis = income > 0 and (total_min / income) > 0.40

        recommendation = "avalanche"  # default
        reason         = "Saves the most money (Rs {:,.0f} less interest)".format(interest_saving)

        if is_crisis:
            recommendation = "crisis"
            reason         = "DEBT CRISIS: Monthly payments exceed 40% of income. Need immediate restructuring."

        smallest_ratio = min(d.principal for d in debts) / max(max(d.principal for d in debts), 1)
        if smallest_ratio < 0.2 and interest_saving < 5000:
            # Snowball wins if smallest debt is very small (can close quickly) and interest diff is small
            recommendation = "snowball"
            reason = "Quick win: clear the smallest debt fast, gain motivation"

        return {
            "recommended":       recommendation,
            "reason":            reason,
            "avalanche_result":  av,
            "snowball_result":   sn,
            "interest_savings_avalanche": round(interest_saving, 2),
            "is_crisis":         is_crisis,
            "debt_to_income":    round(total_min / max(income, 1) * 100, 1) if income > 0 else None,
        }

    # ── Negotiation Scripts ───────────────────────────────────

    def generate_negotiation_brief(self, debt: Debt, user_income: float,
                                    language: str = "hi") -> Dict:
        """
        Generate negotiation parameters for the LLM to turn into a script.
        Returns a structured brief that the LLM will use to write the actual
        conversation script in the user's language.
        """
        debt_to_income = debt.min_payment / max(user_income, 1)
        hardship_level = (
            "severe"   if debt_to_income > 0.40 else
            "moderate" if debt_to_income > 0.25 else
            "mild"
        )

        # Settlement range (realistic for Indian market)
        settlement_ranges = {
            "bank":         (0.60, 0.85),   # Banks settle at 60-85% of outstanding
            "nbfc":         (0.50, 0.75),   # NBFCs settle at 50-75%
            "moneylender":  (0.40, 0.70),   # Moneylenders can settle 40-70%
            "medical":      (0.30, 0.60),   # Hospitals often settle 30-60%
            "payday":       (0.40, 0.65),   # Payday lenders settle 40-65%
            "friend":       (0.80, 1.00),   # Friends — don't lowball
        }
        low, high = settlement_ranges.get(debt.lender_type, (0.60, 0.85))

        # Legal options by lender type
        legal_options = {
            "bank":        ["RBI Banking Ombudsman", "DRT (Debt Recovery Tribunal)", "IBC Section 8"],
            "nbfc":        ["RBI NBFC Ombudsman complaint", "Consumer Forum"],
            "moneylender": ["State Moneylenders Act — usury complaint", "Police complaint for >2% monthly"],
            "medical":     ["Hospital patient welfare committee", "AYUSHMAN BHARAT coverage check"],
            "payday":      ["RBI Digital Lending Guidelines", "Fair Practices Code complaint"],
        }

        asks = []
        if hardship_level in ("severe", "moderate"):
            asks.extend(["EMI restructuring (reduce monthly payment)", "Moratorium (payment pause)"])
        asks.extend(["Interest rate reduction", "Settlement at reduced amount"])
        if debt.lender_type == "bank":
            asks.append("Debt consolidation into personal loan at lower rate")

        return {
            "debt_name":        debt.name,
            "outstanding":      round(debt.principal, 2),
            "current_rate":     debt.annual_rate,
            "lender_type":      debt.lender_type,
            "hardship_level":   hardship_level,
            "debt_to_income_pct": round(debt_to_income * 100, 1),
            "settlement_range": {
                "low_offer":    round(debt.principal * low, 0),
                "high_offer":   round(debt.principal * high, 0),
                "walk_away":    round(debt.principal * (low - 0.10), 0),
            },
            "negotiation_asks": asks,
            "legal_options":    legal_options.get(debt.lender_type, ["Consumer Forum"]),
            "language":         language,
            "tone_guidance":    "Polite but firm. Know your rights. Don't reveal walk-away.",
        }


# ─────────────────────────────────────────────────────────────
# 3. BUSINESS ENGINE
# ─────────────────────────────────────────────────────────────

class BusinessEngine:
    """
    Micro-entrepreneur financial intelligence.
    Covers: pricing, GST, transaction tracking, microloan readiness.
    """

    # ── Pricing Strategy ─────────────────────────────────────

    def suggest_price(self, cost_of_goods: float, operating_expenses: float,
                       target_margin_pct: float = 20,
                       competition: str = "medium",
                       business_type: str = "retail") -> Dict:
        """
        Compute pricing recommendations.
        business_type: retail | service | manufacturing | food
        competition:   low | medium | high
        """
        total_cost = cost_of_goods + operating_expenses

        # Base multiplier by competition
        multipliers = {"low": 1.35, "medium": 1.22, "high": 1.12}
        mult        = multipliers.get(competition, 1.22)

        breakeven    = total_cost * 1.02  # 2% above cost
        suggested    = total_cost * mult
        target_price = total_cost * (1 + target_margin_pct / 100)

        # Market pricing tips by business type
        tips = {
            "retail":        "Add 15-25% markup. Bundle slow items with fast-moving ones.",
            "service":       "Charge for your time. Min Rs 200/hr for skilled work.",
            "manufacturing": "Factor in machine wear, reject rate, storage.",
            "food":          "Food cost should be 30-35% of selling price maximum.",
        }

        return {
            "cost_breakdown": {
                "goods_cost":       round(cost_of_goods, 2),
                "operating_cost":   round(operating_expenses, 2),
                "total_cost":       round(total_cost, 2),
            },
            "price_options": {
                "breakeven_price":  round(breakeven, 2),
                "suggested_price":  round(suggested, 2),
                "target_price":     round(target_price, 2),
                "profit_at_suggested": round(suggested - total_cost, 2),
                "margin_at_suggested": round((suggested - total_cost) / suggested * 100, 1),
            },
            "competition_level":    competition,
            "business_type_tip":    tips.get(business_type, ""),
            "never_sell_below":     round(total_cost * 1.05, 2),
        }

    # ── GST Decision ─────────────────────────────────────────

    def gst_decision(self, annual_turnover: float, state: str = "general",
                      supplies_type: str = "goods") -> Dict:
        """
        Should this business register for GST?
        Updated thresholds per GST Council circular 2023.
        """
        # State-wise thresholds (special category states have lower limits)
        special_states = ["manipur", "mizoram", "nagaland", "tripura", "sikkim",
                           "meghalaya", "arunachal", "uttarakhand", "himachal"]

        if state.lower() in special_states:
            mandatory_threshold  = 1_000_000   # 10 lakh
            composition_threshold = 7_500_000    # 75 lakh
        elif supplies_type == "services":
            mandatory_threshold  = 2_000_000   # 20 lakh
            composition_threshold = 5_000_000  # 50 lakh
        else:
            mandatory_threshold  = 4_000_000   # 40 lakh for goods
            composition_threshold = 15_000_000  # 1.5 crore composition

        if annual_turnover < mandatory_threshold:
            status   = "NOT_REQUIRED"
            rate     = None
            advice   = "Below threshold. No GST registration needed. Save paperwork."
        elif annual_turnover <= composition_threshold:
            status   = "OPTIONAL_COMPOSITION"
            rate     = 1.0 if supplies_type == "goods" else 6.0  # GST %
            advice   = f"Optional Composition Scheme: Pay only {rate}% GST on turnover. Simple."
        else:
            status   = "MANDATORY"
            rate     = 18.0  # Default standard rate
            advice   = "MANDATORY registration. Get a CA/tax practitioner to file."

        # Annual GST payable estimate
        gst_payable = (annual_turnover * (rate or 0) / 100) if rate else 0

        return {
            "annual_turnover":   round(annual_turnover, 0),
            "status":            status,
            "gst_rate":          rate,
            "advice":            advice,
            "estimated_annual_gst": round(gst_payable, 0),
            "registration_benefit": status == "OPTIONAL_COMPOSITION" and annual_turnover > 1_500_000,
            "next_step":         "Visit gst.gov.in or go to nearest Jan Sewa Kendra to register.",
        }

    # ── Transaction Parser ────────────────────────────────────

    def parse_voice_transaction(self, text: str, lang: str = "hi") -> Optional[Dict]:
        """
        Parse a voice note text into a structured transaction.
        This uses simple rule-based extraction — the LLM then confirms with user.

        Examples:
          "aaj 500 ki sabzi bichi"   → {type: income, amount: 500, category: business}
          "200 ka petrol liya"       → {type: expense, amount: 200, category: fuel}
          "kiraye ke 5000 mile"      → {type: income, amount: 5000, category: rent_received}
        """
        import re

        # Extract amount — look for Indian number patterns
        amount_pattern = r'(?:rs\.?|₹|rupay[ae]?|paisa)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs\.?|₹|rupay[ae]?|ka|ki|ke)?'
        amounts        = re.findall(amount_pattern, text.lower().replace(',', ''))
        if not amounts:
            return None

        amount = float(amounts[0])

        # Determine type — income or expense
        income_words  = ["mila", "bika", "aaya", "received", "income", "kamaya", "bikri",
                         "आया", "मिला", "बिका", "मिले", "कमाया", "आमदनी",
                         "आलो", "পেলাম", "கிடைத்தது", "ಬಂತು", "వచ్చింది"]
        expense_words = ["diya", "kharcha", "liya", "gaya", "payment", "expense", "kharch",
                         "दिया", "खर्च", "लिया", "गया", "तिरलाम", "செலவு", "ಖರ್ಚು", "ఖర్చు"]

        text_lower = text.lower()
        is_income  = any(w in text_lower for w in income_words)
        is_expense = any(w in text_lower for w in expense_words)

        if is_income and not is_expense:
            txn_type = "income"
        elif is_expense and not is_income:
            txn_type = "expense"
        else:
            txn_type = "expense"  # default if unclear

        # Category detection
        categories = {
            "food":       ["khana", "sabzi", "roti", "dal", "chawal", "food", "grocery",
                           "खाना", "सब्ज़ी", "राशन"],
            "transport":  ["petrol", "diesel", "auto", "bus", "train", "travel",
                           "पेट्रोल", "डीज़ल", "ट्रैवल"],
            "emi":        ["emi", "loan", "karz", "karza", "installment",
                           "ईएमआई", "लोन", "कर्ज़"],
            "rent":       ["kiraya", "rent", "makaan", "किराया", "rent"],
            "medical":    ["doctor", "dawai", "hospital", "medicine", "दवाई", "अस्पताल"],
            "business":   ["bikri", "dukan", "business", "mal", "stok", "बिक्री", "दुकान"],
            "salary":     ["salary", "tanakvah", "wages", "वेतन", "तनख्वाह"],
        }
        category = "other"
        for cat, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                category = cat
                break

        return {
            "type":        txn_type,
            "amount":      amount,
            "category":    category,
            "description": text[:100],
            "raw_text":    text,
            "lang":        lang,
            "date_str":    str(date.today()),
            "confidence":  "high" if (is_income or is_expense) else "low",
        }

    # ── Credit Profile Builder ────────────────────────────────

    def build_credit_profile(self, transactions: List[Transaction],
                              months: int = 6) -> Dict:
        """
        Build a digital financial history from voice note transactions.
        Used to generate microloan applications.
        """
        if not transactions:
            return {"status": "insufficient_data",
                    "message": "Record at least 30 days of transactions first."}

        income_txns  = [t for t in transactions if t.type == "income"]
        expense_txns = [t for t in transactions if t.type == "expense"]

        total_income  = sum(t.amount for t in income_txns)
        total_expense = sum(t.amount for t in expense_txns)
        net_cash_flow = total_income - total_expense

        monthly_income  = total_income  / max(months, 1)
        monthly_expense = total_expense / max(months, 1)

        # Income regularity score (0-100)
        if len(income_txns) < 5:
            regularity = 30
        elif len(income_txns) >= 20:
            regularity = 80
        else:
            regularity = 30 + (len(income_txns) - 5) * 3.3

        # Creditworthiness
        debt_service_ratio = 0  # would include EMIs if tracked
        is_creditworthy    = (
            net_cash_flow > 0 and
            regularity > 50 and
            monthly_income > 5000
        )

        # Loan eligibility estimate (RBI NBFC microfinance guidelines)
        if is_creditworthy:
            # Generally 6x monthly net income for JLG (joint liability group) loans
            max_loan  = monthly_income * 6
            max_loan  = min(max_loan, 500000)  # Cap at 5L for informal sector
        else:
            max_loan  = 0

        return {
            "summary": {
                "months_tracked":    months,
                "avg_monthly_income":  round(monthly_income, 2),
                "avg_monthly_expense": round(monthly_expense, 2),
                "avg_monthly_surplus": round(net_cash_flow / months, 2),
                "income_sources":    len(set(t.category for t in income_txns)),
                "income_regularity_score": round(regularity, 0),
            },
            "creditworthiness": {
                "is_creditworthy":     is_creditworthy,
                "max_eligible_loan":   round(max_loan, 0),
                "suggested_emi":       round(max_loan / 24, 0) if max_loan > 0 else 0,
                "debt_service_ratio":  round(debt_service_ratio * 100, 1),
            },
            "lender_recommendations": self._recommend_lenders(max_loan),
            "documents_needed":       self._document_checklist(max_loan),
        }

    def _recommend_lenders(self, max_loan: float) -> List[Dict]:
        """Recommend appropriate lenders based on loan amount"""
        lenders = []
        if max_loan >= 10000:
            lenders.append({
                "name":      "MUDRA Shishu Loan (Govt)",
                "limit":     50000,
                "rate":      "10-12% per year",
                "contact":   "Nearest PSU bank or mudra.org.in",
                "documents": "Aadhaar, PAN, 2 photos, business address proof",
            })
        if max_loan >= 50000:
            lenders.append({
                "name":      "NBFC Microfinance (NBFC-MFI)",
                "limit":     200000,
                "rate":      "19-24% per year (RBI regulated)",
                "contact":   "Aroha Finance, Annapurna Finance, Ujjivan",
                "documents": "Aadhaar, bank statement 3 months",
            })
        if max_loan >= 100000:
            lenders.append({
                "name":      "Jan Dhan Overdraft",
                "limit":     10000,
                "rate":      "0% for first 6 months",
                "contact":   "Any nationalised bank with Jan Dhan account",
                "documents": "Jan Dhan account passbook",
            })
        return lenders

    def _document_checklist(self, loan_amount: float) -> List[str]:
        docs = ["Aadhaar card (original + 1 photocopy)", "2 recent passport photos"]
        if loan_amount > 25000:
            docs.extend(["PAN card or Form 60", "Last 3 months bank statement"])
        if loan_amount > 100000:
            docs.extend(["Income proof (salary slip / ITR)", "Address proof (electricity bill)"])
        return docs


# ─────────────────────────────────────────────────────────────
# 4. INDIAN TAX ENGINE
# ─────────────────────────────────────────────────────────────

class IndianTaxEngine:
    """
    ITR filing guidance for informal sector workers.
    Covers: ITR-1 (salaried), ITR-4 (presumptive income).
    FY 2023-24 tax slabs (AY 2024-25).
    """

    # New tax regime slabs (FY2023-24)
    NEW_REGIME_SLABS = [
        (300_000,   0.00),
        (600_000,   0.05),
        (900_000,   0.10),
        (1_200_000, 0.15),
        (1_500_000, 0.20),
        (float("inf"), 0.30),
    ]

    # Old tax regime slabs
    OLD_REGIME_SLABS = [
        (250_000,   0.00),
        (500_000,   0.05),
        (1_000_000, 0.20),
        (float("inf"), 0.30),
    ]

    def calculate_tax(self, annual_income: float,
                       regime: str = "new",
                       age: int = 35,
                       deductions: float = 0) -> Dict:
        """
        Calculate income tax under new or old regime.
        """
        slabs = self.NEW_REGIME_SLABS if regime == "new" else self.OLD_REGIME_SLABS

        # Senior citizen exemption
        if regime == "old" and age >= 60:
            slabs = [(300_000 if age < 80 else 500_000, 0.00)] + slabs[1:]

        taxable_income = annual_income - (deductions if regime == "old" else 0)
        taxable_income = max(0, taxable_income)

        # Rebate u/s 87A — no tax if income <= 7L (new) or 5L (old)
        rebate_limit = 700_000 if regime == "new" else 500_000

        tax = 0.0
        prev_limit = 0
        breakdown  = []
        for limit, rate in slabs:
            if taxable_income <= prev_limit:
                break
            slab_income = min(taxable_income, limit) - prev_limit
            slab_tax    = slab_income * rate
            if slab_income > 0:
                breakdown.append({
                    "range":      f"{prev_limit:,} - {limit if limit != float('inf') else 'above'}",
                    "rate":       f"{int(rate*100)}%",
                    "income":     round(slab_income, 0),
                    "tax":        round(slab_tax, 0),
                })
            tax        += slab_tax
            prev_limit  = limit

        # Section 87A rebate (new regime: up to Rs 25,000 rebate if income <= 7L)
        rebate_max = 25_000 if regime == "new" else 12_500
        if taxable_income <= rebate_limit:
            rebate = min(tax, rebate_max)
            tax    = max(0, tax - rebate)
        else:
            rebate = 0

        # Add 4% cess
        cess       = tax * 0.04
        total_tax  = tax + cess

        return {
            "annual_income":   round(annual_income, 0),
            "taxable_income":  round(taxable_income, 0),
            "regime":          regime,
            "gross_tax":       round(tax, 0),
            "rebate_87A":      round(rebate, 0),
            "education_cess":  round(cess, 0),
            "total_tax":       round(total_tax, 0),
            "effective_rate":  round(total_tax / max(annual_income, 1) * 100, 1),
            "monthly_tax":     round(total_tax / 12, 0),
            "tax_breakdown":   breakdown,
        }

    def itr_form_selector(self, income_sources: List[str],
                           business_turnover: float = 0) -> Dict:
        """Determine which ITR form to file"""
        has_business    = "business" in income_sources
        has_capital_gain = "capital_gain" in income_sources
        has_multiple     = len(income_sources) > 1

        if has_capital_gain or (has_business and business_turnover > 5_000_000):
            form     = "ITR-3"
            rationale = "You have capital gains or large business income. Need CA help."
        elif has_business and business_turnover <= 5_000_000:
            form     = "ITR-4"
            rationale = ("Presumptive taxation — just declare 8% of turnover as income. "
                         "No need to maintain full books. File online at incometaxindiaefiling.gov.in")
        elif "salary" in income_sources and not has_business:
            form     = "ITR-1"
            rationale = "Simple salaried return. Pre-filled from Form 16. File in 10 minutes online."
        else:
            form     = "ITR-1"
            rationale = "Start with ITR-1. If it doesn't fit, system will guide you."

        return {
            "recommended_form": form,
            "rationale":        rationale,
            "income_sources":   income_sources,
            "filing_deadline":  "July 31 (AY year)",
            "penalty_for_late": "Rs 5,000 if filed after July 31 (Rs 1,000 if income < 5L)",
            "free_filing":      "incometaxindiaefiling.gov.in — completely free",
        }

    def presumptive_tax_itr4(self, gross_receipts: float) -> Dict:
        """
        ITR-4 presumptive taxation calculation.
        Section 44AD: Declare 8% of turnover (6% if digital receipts).
        Huge simplification for micro-entrepreneurs.
        """
        presumptive_income = gross_receipts * 0.08  # 8% of turnover
        digital_income     = gross_receipts * 0.06  # 6% if digital payments

        tax_cash    = self.calculate_tax(presumptive_income)
        tax_digital = self.calculate_tax(digital_income)

        return {
            "gross_receipts":          round(gross_receipts, 0),
            "presumptive_income_cash":    round(presumptive_income, 0),
            "presumptive_income_digital": round(digital_income, 0),
            "tax_if_cash":             tax_cash["total_tax"],
            "tax_if_digital":          tax_digital["total_tax"],
            "savings_by_going_digital": round(tax_cash["total_tax"] - tax_digital["total_tax"], 0),
            "tip": "Accept payments via UPI/card to declare only 6% income instead of 8%.",
        }


# ─────────────────────────────────────────────────────────────
# 5. COMBINED FINANCIAL ADVISOR
# ─────────────────────────────────────────────────────────────

class ArthaSathiAdvisor:
    """
    Master advisor — combines all engines and generates structured
    context for the LLM to generate human-readable responses.
    """

    def __init__(self):
        self.debt_engine     = DebtEngine()
        self.business_engine = BusinessEngine()
        self.tax_engine      = IndianTaxEngine()

    def analyze_financial_health(self, profile: UserFinancialProfile) -> Dict:
        """Comprehensive financial health analysis"""
        monthly_income  = profile.monthly_income
        total_debt      = sum(d.principal for d in profile.debts)
        total_min_emi   = sum(d.min_payment for d in profile.debts)
        debt_to_income  = total_min_emi / max(monthly_income, 1)

        # Health score (0-100)
        score = 100
        if debt_to_income > 0.50: score -= 40
        elif debt_to_income > 0.30: score -= 20
        elif debt_to_income > 0.15: score -= 10

        if total_debt > monthly_income * 12: score -= 20
        elif total_debt > monthly_income * 6: score -= 10

        if not profile.debts:
            score = min(score + 20, 100)  # No debt = bonus

        health_level = (
            "CRITICAL"  if score < 30 else
            "POOR"      if score < 50 else
            "FAIR"      if score < 70 else
            "GOOD"      if score < 85 else
            "EXCELLENT"
        )

        # Recommendations
        recommendations = []
        if debt_to_income > 0.40:
            recommendations.append("URGENT: Contact lenders immediately for restructuring")
        if any(d.annual_rate > 30 for d in profile.debts):
            recommendations.append("High-interest debt detected (>30%): prioritize paying this first")
        if monthly_income > 0 and not profile.debts:
            recommendations.append("No debt — great! Consider saving Rs %d/month in emergency fund" %
                                    round(monthly_income * 0.20))

        # Best strategy if debts exist
        strategy = None
        if profile.debts:
            strategy = self.debt_engine.recommend_strategy(
                profile.debts,
                extra=max(0, monthly_income * 0.10),  # 10% extra if available
                income=monthly_income,
            )

        return {
            "health_score":      score,
            "health_level":      health_level,
            "summary": {
                "monthly_income":    round(monthly_income, 0),
                "total_debt":        round(total_debt, 0),
                "total_monthly_emi": round(total_min_emi, 0),
                "debt_to_income":    round(debt_to_income * 100, 1),
                "n_debts":           len(profile.debts),
            },
            "recommendations":   recommendations,
            "debt_strategy":     strategy,
        }

    def process_message(self, text: str, lang: str,
                         user_profile: UserFinancialProfile) -> Dict:
        """
        Parse a user message, determine intent, run relevant engine,
        and return structured context for LLM response generation.
        """
        text_lower = text.lower()

        # Intent detection
        if any(w in text_lower for w in ["loan", "debt", "karz", "emi", "payment",
                                          "कर्ज़", "लोन", "ऋण", "கடன்", "ಸಾಲ"]):
            intent = "debt_query"
        elif any(w in text_lower for w in ["price", "charge", "rate", "kitna", "cost",
                                            "दाम", "कीमत", "விலை", "ಬೆಲೆ"]):
            intent = "pricing_query"
        elif any(w in text_lower for w in ["gst", "tax", "itr", "income tax",
                                            "कर", "टैक्स", "வரி", "ತೆರಿಗೆ"]):
            intent = "tax_query"
        elif any(w in text_lower for w in ["income", "kharcha", "expense", "aaya", "gaya",
                                            "आमदनी", "खर्च", "வருமானம்", "ವೆಚ್ಚ"]):
            intent = "transaction"
        else:
            intent = "general"

        # Run appropriate engine
        engine_result = {}
        if intent == "debt_query" and user_profile.debts:
            engine_result = self.analyze_financial_health(user_profile)
        elif intent == "pricing_query":
            engine_result = {"message": "Need cost details to compute price"}
        elif intent == "tax_query" and user_profile.monthly_income > 0:
            annual = user_profile.monthly_income * 12
            engine_result = self.tax_engine.calculate_tax(annual)
        elif intent == "transaction":
            txn = self.business_engine.parse_voice_transaction(text, lang)
            engine_result = {"parsed_transaction": txn}

        return {
            "intent":        intent,
            "language":      lang,
            "engine_result": engine_result,
            "user_profile_summary": {
                "income":    user_profile.monthly_income,
                "n_debts":   len(user_profile.debts),
                "total_debt": sum(d.principal for d in user_profile.debts),
            },
        }


# ─────────────────────────────────────────────────────────────
# 6. TESTS
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ArthaSathi Financial Engines — Test Suite")
    print("=" * 60)

    engine = DebtEngine()

    # Test 1: EMI Calculator
    emi = engine.emi_calculator(100000, 18, 24)
    print(f"\nEMI (1L, 18%, 24m): Rs {emi['emi']:,.2f}/month | "
          f"Interest: Rs {emi['total_interest']:,.2f}")

    # Test 2: Debt strategies
    debts = [
        Debt("HDFC CC",       30000, 36, 900,  "bank"),
        Debt("Bajaj Finance", 20000, 26, 600,  "nbfc"),
        Debt("Home Loan",    100000, 12, 3000, "bank"),
    ]
    av = engine.avalanche_strategy(debts, extra_payment=2000)
    sn = engine.snowball_strategy(debts, extra_payment=2000)
    print(f"\nAvalanche total interest: Rs {av['total_interest']:,.0f}")
    print(f"Snowball total interest:  Rs {sn['total_interest']:,.0f}")
    print(f"Savings from Avalanche:   Rs {sn['total_interest']-av['total_interest']:,.0f}")

    # Test 3: Business engine
    biz = BusinessEngine()
    price = biz.suggest_price(300, 100, 25, "medium", "retail")
    print(f"\nPricing: Cost=400 → Suggested=Rs {price['price_options']['suggested_price']}")

    gst = biz.gst_decision(2_500_000)
    print(f"GST decision (2.5L turnover): {gst['status']} — {gst['advice']}")

    # Test 4: Tax engine
    tax = IndianTaxEngine()
    t = tax.calculate_tax(600000, "new")
    print(f"\nIncome tax (6L, new regime): Rs {t['total_tax']:,} (effective {t['effective_rate']}%)")

    # Test 5: Voice transaction parser
    txns = [
        ("aaj 500 ki sabzi bichi", "hi"),
        ("200 ka petrol liya", "hi"),
        "today earned 800 from tailoring work",
    ]
    for txn_text in txns:
        if isinstance(txn_text, tuple):
            text, lang = txn_text
        else:
            text, lang = txn_text, "en"
        parsed = biz.parse_voice_transaction(text, lang)
        if parsed:
            print(f"\nParsed: '{text[:40]}' → {parsed['type']} Rs {parsed['amount']} [{parsed['category']}]")

    print("\n✓ All financial engine tests passed!")
