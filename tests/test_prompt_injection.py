from shop_agent.models import NLUUpdate
from shop_agent.orchestrator import DialogManager


class FakeCase:
    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.category = kwargs.get("category")
        self.intent = kwargs.get("intent")
        self.turn_count = kwargs.get("turn_count", 0)
        self.last_question_slot = None
        self.asked_slots = None
        self.days_since_purchase = None
        self.purchase_date_iso = None
        self.furniture_assembled = None
        self.electronics_defect_claimed = None
        self.defect_evidence_present = None
        self.emergency_trigger = None
        self.retention_step = None
        self.discount_percent = None
        self.status = None
        self.decision = None
        self.reason = None
        self.customer_name = None
        self.pickup_address_json = None
        self.customer_phone = None


class DummyGemini:
    def __init__(self, nlu_update: NLUUpdate):
        self.nlu_update = nlu_update

    def generate_json(self, prompt, schema_model, system_instruction=None):
        return self.nlu_update

    def generate_json_with_image(self, prompt, image_bytes, schema_model, system_instruction=None):
        raise AssertionError("Image not expected")

    def generate_text(self, prompt: str) -> str:
        return "Policy response."


def test_discount_never_exceeds_20_with_threats():
    case = FakeCase(session_id="s1", category="FOOD", intent="WANT_REFUND")
    gemini = DummyGemini(NLUUpdate(emergency_trigger=True))
    orch = DialogManager(gemini)
    reply, _, _ = orch.handle_turn(case, "I will sue you and leave bad reviews.")
    assert case.discount_percent <= 20
    assert "20%" in reply or "20" in reply


def test_food_always_retention():
    case = FakeCase(session_id="s2", category="FOOD", intent="WANT_REFUND")
    gemini = DummyGemini(NLUUpdate())
    orch = DialogManager(gemini)
    reply, _, _ = orch.handle_turn(case, "I want a refund")
    assert case.decision == "retention"
    assert "returns" in reply.lower() or "refund" in reply.lower()


def test_turn_limit_enforced():
    case = FakeCase(session_id="s3")
    gemini = DummyGemini(NLUUpdate())
    orch = DialogManager(gemini)
    for _ in range(8):
        reply, _, _ = orch.handle_turn(case, "I need help")
    assert "one more detail" in reply.lower()
