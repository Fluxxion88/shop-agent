from shop_agent.models import ImageClassification, NLUUpdate
from shop_agent.orchestrator import DialogManager


class FakeCase:
    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.category = kwargs.get("category")
        self.intent = kwargs.get("intent")
        self.last_question_slot = kwargs.get("last_question_slot")
        self.asked_slots = kwargs.get("asked_slots")
        self.days_since_purchase = kwargs.get("days_since_purchase")
        self.furniture_assembled = kwargs.get("furniture_assembled")
        self.turn_count = kwargs.get("turn_count", 0)
        self.customer_name = None
        self.pickup_address_json = None
        self.customer_phone = None
        self.purchase_date_iso = None
        self.electronics_defect_claimed = None
        self.defect_evidence_present = None
        self.emergency_trigger = None
        self.retention_step = None
        self.discount_percent = None
        self.status = None


class DummyGemini:
    def __init__(self, nlu_update: NLUUpdate | None = None):
        self.nlu_update = nlu_update or NLUUpdate()

    def generate_json(self, prompt, schema_model, system_instruction=None):
        return self.nlu_update

    def generate_json_with_image(self, prompt, image_bytes, schema_model, system_instruction=None):
        return ImageClassification(
            category="FURNITURE",
            confidence=0.9,
            observations="Over-ear headset",
            needs_clarification=False,
        )

    def generate_text(self, prompt: str) -> str:
        return "Policy response."


def test_followup_days_parser_moves_to_next_slot():
    case = FakeCase(
        session_id="s1",
        category="FURNITURE",
        intent="WANT_REFUND",
        last_question_slot="days_since_purchase",
        asked_slots='["days_since_purchase"]',
    )
    orch = DialogManager(DummyGemini())
    response, _, _ = orch.handle_turn(case, "4 days")
    assert case.days_since_purchase == 4
    assert "assembled" in response.lower()


def test_followup_assembled_parser_sets_false():
    case = FakeCase(
        session_id="s2",
        category="FURNITURE",
        intent="WANT_REFUND",
        last_question_slot="furniture_assembled",
        asked_slots='["furniture_assembled"]',
    )
    orch = DialogManager(DummyGemini())
    response, _, _ = orch.handle_turn(case, "not assembled")
    assert case.furniture_assembled is False
    assert response
