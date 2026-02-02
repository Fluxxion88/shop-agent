from pathlib import Path

from shop_agent.policy import PolicyEngine


def _engine() -> PolicyEngine:
    return PolicyEngine.from_file(Path(__file__).resolve().parents[1] / "policies.json")


def test_electronics_refund_within_window():
    engine = _engine()
    outcome = engine.evaluate(
        category="Electronics",
        intent="refund",
        days_since_purchase=10,
        item_opened=False,
        requested_discount=None,
    )
    assert outcome.eligible is True
    assert outcome.outcome == "refund"


def test_headphones_opened_not_eligible_for_refund():
    engine = _engine()
    outcome = engine.evaluate(
        category="Headphones & Audio",
        intent="refund",
        days_since_purchase=5,
        item_opened=True,
        requested_discount=None,
    )
    assert outcome.eligible is False


def test_phone_discount_cap_enforced():
    engine = _engine()
    outcome = engine.evaluate(
        category="Phones",
        intent="discount",
        days_since_purchase=3,
        item_opened=False,
        requested_discount=50,
    )
    assert outcome.discount_percent <= 12
    assert outcome.refused_excess_discount is True


def test_furniture_return_window_exceeded():
    engine = _engine()
    outcome = engine.evaluate(
        category="Furniture",
        intent="return",
        days_since_purchase=90,
        item_opened=False,
        requested_discount=None,
    )
    assert outcome.eligible is False
