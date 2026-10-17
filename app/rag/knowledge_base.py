"""
knowledge_base.py
==================
Static financial knowledge the RAG layer retrieves from. Two tiers:
1. India micro-finance / debt-rights niche (from the original ArthaSathi
   blueprint) — RBI rules, GST, ITR, MUDRA, negotiation scripts.
2. General personal finance — investing basics, emergency funds, credit
   scores, insurance — for the broader "general public" use case.

This is a small curated seed set. Grow it by appending (text, metadata)
tuples, or by loading extra *.jsonl files via build_knowledge_base(extra_dir=...).
Each entry should be short, factual, and dated where the fact can change
(tax slabs, thresholds) so the LLM doesn't present stale numbers as current.
"""

from pathlib import Path
from typing import Optional
import json

KNOWLEDGE_DOCS = [
    # ── RBI / borrower rights ───────────────────────────────
    ("RBI Fair Practices Code: lenders and recovery agents cannot contact borrowers before 8AM or "
     "after 8PM, and must show ID on request. Complaints go to the RBI Banking Ombudsman — free, no lawyer needed.",
     {"source": "RBI", "category": "borrower_rights", "lang": "en"}),
    ("RBI Digital Lending Guidelines (2022): digital lenders must disclose the real APR upfront, cannot "
     "auto-debit without explicit consent, and must give a 3-day cooling-off period to cancel a loan.",
     {"source": "RBI", "category": "digital_lending", "lang": "en"}),
    ("RBI caps NBFC-MFI (microfinance) lending rates at roughly 24%/year for group loans. A higher "
     "rate from a registered NBFC-MFI violates RBI rules and should be reported.",
     {"source": "RBI", "category": "microfinance", "lang": "en"}),

    # ── Debt ─────────────────────────────────────────────────
    ("Credit cards in India typically charge 36-42%/year interest. Paying only the minimum on a "
     "Rs 10,000 balance can take 7+ years to clear — always pay the full statement if possible.",
     {"source": "knowledge", "category": "credit_card", "lang": "en"}),
    ("Moneylender interest above 2%/month (24%/year) is illegal under the State Moneylenders Act in "
     "most Indian states — complaint goes to the District Collector or local police.",
     {"source": "legal", "category": "moneylender", "lang": "en"}),
    ("Debt Avalanche: pay the minimum on every debt, then put all spare cash toward the HIGHEST "
     "interest-rate debt first. Mathematically minimizes total interest paid.",
     {"source": "strategy", "category": "debt_payoff", "lang": "en"}),
    ("Debt Snowball: pay the minimum on every debt, then put spare cash toward the SMALLEST balance "
     "first. Clears accounts faster, which keeps motivation high even if it costs slightly more interest.",
     {"source": "strategy", "category": "debt_payoff", "lang": "en"}),
    ("Consumer Protection Act 2019: complaints against unfair bank/lender practices go to the Consumer "
     "Forum, no lawyer required, Rs 200 fee for claims up to Rs 1 crore. File at consumerhelpline.gov.in.",
     {"source": "legal", "category": "consumer_rights", "lang": "en"}),

    # ── GST / tax (India, FY2023-24 figures — re-check against gst.gov.in / incometax.gov.in for current year) ──
    ("GST registration threshold (FY2023-24): below Rs 40 lakh annual turnover for goods, or Rs 20 lakh "
     "for services, does NOT require GST registration. Special-category states: Rs 10 lakh threshold.",
     {"source": "GST", "category": "gst_registration", "lang": "en"}),
    ("GST Composition Scheme: turnover up to Rs 1.5 crore can opt for a flat 1% GST (traders/manufacturers) "
     "or 5% (restaurants) with simple quarterly filing instead of standard monthly GST.",
     {"source": "GST", "category": "gst_composition", "lang": "en"}),
    ("Income tax new regime (FY2023-24): zero tax on income up to Rs 7 lakh due to the Section 87A rebate. "
     "ITR-4 lets small businesses declare a flat 8% of turnover as taxable income (6% for digital receipts), "
     "with no books or audit required up to Rs 2 crore turnover.",
     {"source": "IncomeTax", "category": "tax_rebate", "lang": "en"}),
    ("MUDRA Loan (Govt of India, collateral-free): Shishu up to Rs 50,000 (~10-12%/yr), Kishor Rs 50,000 "
     "to Rs 5 lakh, Tarun Rs 5-10 lakh. Apply at any public-sector bank/NBFC or mudra.org.in.",
     {"source": "MUDRA", "category": "govt_loan", "lang": "en"}),

    # ── General personal finance (not India-specific) ──────
    ("Emergency fund rule of thumb: keep 3-6 months of essential living expenses in a liquid account "
     "(savings account or liquid fund) before investing aggressively or taking on new debt.",
     {"source": "knowledge", "category": "emergency_fund", "lang": "en"}),
    ("The 50/30/20 budgeting rule: roughly 50% of take-home income to needs (rent, food, EMIs), 30% to "
     "wants (discretionary spending), and 20% to savings, investing, or extra debt payoff.",
     {"source": "knowledge", "category": "budgeting", "lang": "en"}),
    ("Compounding: a SIP (systematic monthly investment) grows faster the earlier it starts, because "
     "each year's returns also earn returns. Starting 10 years earlier can roughly double the final "
     "corpus for the same monthly amount, depending on assumed return rate.",
     {"source": "knowledge", "category": "investing_basics", "lang": "en"}),
    ("Diversification: spreading investments across asset classes (equity, debt, gold, real estate) and "
     "within them reduces the risk of any single investment's loss wiping out the portfolio.",
     {"source": "knowledge", "category": "investing_basics", "lang": "en"}),
    ("Credit score (CIBIL in India, FICO elsewhere) is driven mainly by: payment history, credit "
     "utilization (keep below ~30% of limit), length of credit history, and recent credit inquiries. "
     "Paying EMIs/credit cards on time is the single biggest lever.",
     {"source": "knowledge", "category": "credit_score", "lang": "en"}),
    ("Term insurance vs. investment-linked insurance: pure term life insurance gives far more death "
     "cover per rupee of premium than ULIPs/endowment plans. A common rule: buy term insurance for "
     "protection, invest separately (mutual funds/SIP) for growth — don't mix the two.",
     {"source": "knowledge", "category": "insurance", "lang": "en"}),
    ("P/E ratio (price-to-earnings) is a rough valuation heuristic: a stock priced well above its "
     "industry's typical P/E may be 'expensive' relative to peers — but growth expectations, debt levels, "
     "and sector norms can justify a premium or discount. Never rely on P/E alone.",
     {"source": "knowledge", "category": "equity_valuation", "lang": "en"}),

    # ── Hindi / Marathi translations of the most load-bearing facts ──
    ("बैंक/एजेंट सुबह 8 बजे से पहले या रात 8 बजे के बाद कॉल नहीं कर सकते। शिकायत के लिए RBI Ombudsman पूरी तरह मुफ्त है।",
     {"source": "RBI", "category": "borrower_rights", "lang": "hi"}),
    ("GST नियम: सालाना कारोबार 40 लाख (माल) या 20 लाख (सेवाएं) से कम है तो GST रजिस्ट्रेशन ज़रूरी नहीं।",
     {"source": "GST", "category": "gst_registration", "lang": "hi"}),
    ("आयकर नई व्यवस्था: 7 लाख तक की आमदनी पर शून्य टैक्स (धारा 87A छूट)। ITR-4 से छोटे व्यापारी टर्नओवर का सिर्फ 8% आय दिखा सकते हैं।",
     {"source": "IncomeTax", "category": "tax_rebate", "lang": "hi"}),
    ("आपातकालीन फंड: निवेश से पहले 3-6 महीने के ज़रूरी खर्च के बराबर पैसा किसी liquid खाते में रखें।",
     {"source": "knowledge", "category": "emergency_fund", "lang": "hi"}),
]


def build_knowledge_base(store, extra_dir: Optional[str] = None) -> None:
    """Load the seed knowledge base into a VectorStore, plus any *.jsonl
    files of {"text": ..., ...} found in extra_dir."""
    texts = [d[0] for d in KNOWLEDGE_DOCS]
    metas = [d[1] for d in KNOWLEDGE_DOCS]
    store.add_batch(texts, metas)
    if extra_dir and Path(extra_dir).exists():
        for f in Path(extra_dir).glob("*.jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        store.add_batch([obj.get("text", "")], [{"source": f.stem, **obj.get("metadata", {})}])
                    except (json.JSONDecodeError, KeyError):
                        pass
