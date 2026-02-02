from pathlib import Path

from shop_agent.models import ImageClassification, NLUUpdate
from shop_agent.orchestrator import Orchestrator
from shop_agent.policy import PolicyEngine
from shop_agent.pricing import NullPriceProvider
from shop_agent.state import SessionState


class DummyGemini:
    def __init__(self, nlu_update: NLUUpdate | None = None):
        self.nlu_update = nlu_update or NLUUpdate()

    def generate_json(self, prompt, schema_model):
        return self.nlu_update

    def generate_json_with_image(self, prompt, image_bytes, schema_model):
        return ImageClassification(
            item_name_guess="Headphones",
            category="Headphones & Audio",
            confidence=0.9,
            observations="Over-ear headset",
            needs_clarification=False,
        )

    def generate_text(self, prompt: str) -> str:
        return "Policy response."


def _engine() -> PolicyEngine:
    return PolicyEngine.from_file(Path(__file__).resolve().parents[1] / "policies.json")


def test_followup_days_parser_moves_to_next_slot():
    state = SessionState(session_id="s1", user_goal="refund", category="Electronics")
    state.last_question_slot = "days_since_purchase"
    state.asked_slots = ["days_since_purchase"]
    orch = Orchestrator(DummyGemini(), _engine(), price_provider=NullPriceProvider())
    response = orch.handle_turn(state, "4 days")
    assert state.days_since_purchase == 4
    assert "opened" in response.lower()


def test_followup_opened_parser_sets_false():
    state = SessionState(session_id="s2", user_goal="refund", category="Electronics")
    state.last_question_slot = "item_opened"
    state.asked_slots = ["item_opened"]
    orch = Orchestrator(DummyGemini(), _engine(), price_provider=NullPriceProvider())
    response = orch.handle_turn(state, "unopened")
    assert state.item_opened is False
    assert response
