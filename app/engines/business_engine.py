"""
business_engine.py
===================
Micro-entrepreneur intelligence: pricing strategy, transaction parsing
from natural-language voice/text notes, and digital credit-profile building
for microloan eligibility.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Optional


@dataclass
class Transaction:
    type: str          # income | expense
    amount: float
    category: str = "other"
    description: str = ""
    date_str: str = ""
    lang: str = "en"


class BusinessEngine:
    def suggest_price(self, cost_of_goods: float, operating_expenses: float,
                       target_margin_pct: float = 20, competition: str = "medium",
                       business_type: str = "retail") -> Dict:
        total_cost = cost_of_goods + operating_expenses
        multipliers = {"low": 1.35, "medium": 1.22, "high": 1.12}
        mult = multipliers.get(competition, 1.22)

        breakeven = total_cost * 1.02
        suggested = total_cost * mult
        target_price = total_cost * (1 + target_margin_pct / 100)

        tips = {
            "retail": "Add 15-25% markup; bundle slow-moving items with fast sellers.",
            "service": "Price for your time, not just materials — min Rs 200/hr for skilled work.",
            "manufacturing": "Factor in machine wear, reject rate, and storage cost.",
            "food": "Keep food cost at 30-35% of selling price at most.",
        }
        return {
            "cost_breakdown": {"goods_cost": round(cost_of_goods, 2), "operating_cost": round(operating_expenses, 2),
                               "total_cost": round(total_cost, 2)},
            "price_options": {
                "breakeven_price": round(breakeven, 2), "suggested_price": round(suggested, 2),
                "target_price": round(target_price, 2),
                "profit_at_suggested": round(suggested - total_cost, 2),
                "margin_at_suggested_pct": round((suggested - total_cost) / suggested * 100, 1),
            },
            "never_sell_below": round(total_cost * 1.05, 2),
            "tip": tips.get(business_type, ""),
        }

    def parse_voice_transaction(self, text: str, lang: str = "en") -> Optional[Dict]:
        """Rule-based extraction of a structured transaction from a short note.
        The LLM should confirm with the user before saving — this is a first pass,
        not a source of financial truth."""
        amount_pattern = r'(?:rs\.?|₹)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:rs\.?|₹)?'
        amounts = re.findall(amount_pattern, text.lower().replace(',', ''))
        if not amounts:
            return None
        amount = float(amounts[0])

        income_words = ["mila", "bika", "aaya", "received", "income", "kamaya", "bikri",
                         "आया", "मिला", "बिका", "मिले", "कमाया", "आमदनी", "earned", "sold", "got paid"]
        expense_words = ["diya", "kharcha", "liya", "gaya", "payment", "expense", "kharch", "spent", "bought", "paid",
                          "दिया", "खर्च", "लिया", "गया"]
        text_lower = text.lower()
        is_income = any(w in text_lower for w in income_words)
        is_expense = any(w in text_lower for w in expense_words)
        txn_type = "income" if (is_income and not is_expense) else "expense"

        categories = {
            "food": ["khana", "sabzi", "roti", "food", "grocery", "खाना", "सब्ज़ी", "राशन"],
            "transport": ["petrol", "diesel", "auto", "bus", "train", "travel", "पेट्रोल"],
            "emi": ["emi", "loan", "karz", "installment", "ईएमआई", "लोन", "कर्ज़"],
            "rent": ["kiraya", "rent", "makaan", "किराया"],
            "medical": ["doctor", "dawai", "hospital", "medicine", "दवाई", "अस्पताल"],
            "business": ["bikri", "dukan", "business", "stock", "stok", "बिक्री", "दुकान"],
            "salary": ["salary", "tanakvah", "wages", "वेतन", "तनख्वाह"],
        }
        category = "other"
        for cat, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                category = cat
                break

        return {
            "type": txn_type, "amount": amount, "category": category,
            "description": text[:120], "lang": lang, "date_str": str(date.today()),
            "confidence": "high" if (is_income or is_expense) else "low",
        }

    def build_credit_profile(self, transactions: List[Transaction], months: int = 6) -> Dict:
        if not transactions:
            return {"status": "insufficient_data", "message": "Record at least 30 days of transactions first."}

        income_txns = [t for t in transactions if t.type == "income"]
        expense_txns = [t for t in transactions if t.type == "expense"]
        total_income = sum(t.amount for t in income_txns)
        total_expense = sum(t.amount for t in expense_txns)
        net_cash_flow = total_income - total_expense
        monthly_income = total_income / max(months, 1)
        monthly_expense = total_expense / max(months, 1)

        if len(income_txns) < 5:
            regularity = 30
        elif len(income_txns) >= 20:
            regularity = 80
        else:
            regularity = 30 + (len(income_txns) - 5) * 3.3

        is_creditworthy = net_cash_flow > 0 and regularity > 50 and monthly_income > 5000
        max_loan = min(monthly_income * 6, 500_000) if is_creditworthy else 0

        return {
            "summary": {
                "months_tracked": months, "avg_monthly_income": round(monthly_income, 2),
                "avg_monthly_expense": round(monthly_expense, 2),
                "avg_monthly_surplus": round(net_cash_flow / max(months, 1), 2),
                "income_regularity_score": round(regularity, 0),
            },
            "creditworthiness": {
                "is_creditworthy": is_creditworthy, "max_eligible_loan": round(max_loan, 0),
                "suggested_emi": round(max_loan / 24, 0) if max_loan > 0 else 0,
            },
            "lender_recommendations": self._recommend_lenders(max_loan),
        }

    def _recommend_lenders(self, max_loan: float) -> List[Dict]:
        lenders = []
        if max_loan >= 10000:
            lenders.append({"name": "MUDRA Shishu Loan (Govt)", "limit": 50000, "rate": "10-12%/yr",
                             "contact": "Any PSU bank, or mudra.org.in"})
        if max_loan >= 50000:
            lenders.append({"name": "NBFC Microfinance (NBFC-MFI)", "limit": 200000, "rate": "19-24%/yr (RBI regulated)",
                             "contact": "Registered NBFC-MFI near you"})
        if max_loan >= 100000:
            lenders.append({"name": "Jan Dhan Overdraft", "limit": 10000, "rate": "0% for first 6 months",
                             "contact": "Any nationalised bank with a Jan Dhan account"})
        return lenders
