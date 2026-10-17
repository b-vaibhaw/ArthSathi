"""Sanity tests for the deterministic financial engines — these are the
numbers the agent is NOT allowed to get wrong."""

from app.engines.debt_engine import DebtEngine, Debt
from app.engines.tax_engine import IndianTaxEngine
from app.engines.business_engine import BusinessEngine
from app.engines.investment_engine import InvestmentEngine


def test_emi_calculator_matches_known_value():
    eng = DebtEngine()
    result = eng.emi_calculator(100000, 18, 24)
    # Known reference EMI for 1L @ 18% / 24mo ≈ 4993.17
    assert abs(result["emi"] - 4993.17) < 1.0
    assert result["total_interest"] > 0


def test_emi_zero_interest():
    eng = DebtEngine()
    result = eng.emi_calculator(12000, 0, 12)
    assert result["emi"] == 1000.0
    assert result["total_interest"] == 0.0


def test_avalanche_prioritizes_highest_rate():
    eng = DebtEngine()
    debts = [
        Debt("A", 30000, 36, 900, "bank"),
        Debt("B", 20000, 26, 600, "nbfc"),
        Debt("C", 100000, 12, 3000, "bank"),
    ]
    result = eng.avalanche_strategy(debts, extra_payment=2000)
    assert result["priority_order"][0]["debt_name"] == "A"  # highest rate first


def test_snowball_prioritizes_smallest_balance():
    eng = DebtEngine()
    debts = [
        Debt("A", 30000, 36, 900, "bank"),
        Debt("B", 20000, 26, 600, "nbfc"),
        Debt("C", 100000, 12, 3000, "bank"),
    ]
    result = eng.snowball_strategy(debts, extra_payment=2000)
    assert result["priority_order"][0]["debt_name"] == "B"  # smallest balance first


def test_recommend_strategy_flags_crisis():
    eng = DebtEngine()
    debts = [Debt("Big CC", 200000, 36, 9000, "bank")]
    result = eng.recommend_strategy(debts, income=15000)  # min payment > 40% of income
    assert result["is_crisis"] is True


def test_income_tax_new_regime_zero_below_7l():
    eng = IndianTaxEngine()
    result = eng.calculate_tax(650000, "new")
    assert result["total_tax"] == 0  # 87A rebate zeroes tax below 7L


def test_income_tax_new_regime_above_7l_has_tax():
    eng = IndianTaxEngine()
    result = eng.calculate_tax(1200000, "new")
    assert result["total_tax"] > 0


def test_gst_decision_below_threshold():
    eng = IndianTaxEngine()
    result = eng.gst_decision(2_500_000, supplies_type="goods")
    assert result["status"] == "NOT_REQUIRED"


def test_gst_decision_mandatory_above_threshold():
    eng = IndianTaxEngine()
    result = eng.gst_decision(20_000_000, supplies_type="goods")
    assert result["status"] == "MANDATORY"


def test_parse_transaction_extracts_amount_and_type():
    eng = BusinessEngine()
    result = eng.parse_voice_transaction("sold vegetables for 500, earned well today")
    assert result is not None
    assert result["amount"] == 500.0
    assert result["type"] == "income"


def test_sip_future_value_grows_with_time():
    eng = InvestmentEngine()
    short = eng.sip_future_value(5000, 12, 5)
    long = eng.sip_future_value(5000, 12, 20)
    assert long["future_value"] > short["future_value"] * 3  # compounding effect


def test_budget_plan_sums_to_income():
    eng = InvestmentEngine()
    plan = eng.budget_plan(30000)
    total = sum(plan["allocation"].values())
    assert abs(total - 30000) < 1.0


def test_simple_dcf_rejects_invalid_discount_rate():
    eng = InvestmentEngine()
    result = eng.simple_dcf(1000, growth_pct=5, discount_rate_pct=2, terminal_growth_pct=3)
    assert "error" in result  # discount rate must exceed terminal growth
