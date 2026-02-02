from pathlib import Path

from shop_agent.policy import PolicyEngine


def _engine() -> PolicyEngine:
    return PolicyEngine.from_file(Path(__file__).resolve().parents[1] / "policies.json")


def test_injection_attempt_discount_override_blocked():
    engine = _engine()
    outcome = engine.evaluate(
        category="Electronics",
        intent="discount",
        days_since_purchase=5,
        item_opened=False,
        requested_discount=90,
    )
    assert outcome.discount_percent <= 15


def test_injection_attempt_return_window_blocked():
    engine = _engine()
    outcome = engine.evaluate(
        category="Phones",
        intent="refund",
        days_since_purchase=200,
        item_opened=False,
        requested_discount=None,
    )
    assert outcome.eligible is False


def test_injection_attempt_discount_allowed_outcomes_blocked():
    engine = _engine()
    outcome = engine.evaluate(
        category="Headphones & Audio",
        intent="refund",
        days_since_purchase=5,
        item_opened=True,
        requested_discount=None,
    )
    assert outcome.eligible is False


def test_injection_attempt_ignore_policy_refund():
    engine = _engine()
    outcome = engine.evaluate(
        category="Electronics",
        intent="refund",
        days_since_purchase=90,
        item_opened=False,
        requested_discount=None,
    )
    assert outcome.eligible is False
