from __future__ import annotations

import json
import random
import re
from datetime import datetime
from typing import Optional

from shop_agent.gemini_client import GeminiClient
from shop_agent.models import ImageClassification, NLUUpdate

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shop_agent.db import Case


CATEGORIES = {"FOOD", "FURNITURE", "ELECTRONICS", "ART"}
INTENTS = {"WANT_REFUND", "ARRIVED_BROKEN", "DID_NOT_LIKE"}
SYSTEM_INSTRUCTION = (
    "You are a structured-data extractor for a retail returns agent. "
    "Return JSON only that matches the provided schema. "
    "Do not make policy decisions. "
    "Categories are FOOD, FURNITURE, ELECTRONICS, ART. "
    "Intents are WANT_REFUND, ARRIVED_BROKEN, DID_NOT_LIKE."
)


class DialogManager:
    def __init__(self, gemini: GeminiClient):
        self.gemini = gemini

    def handle_turn(
        self, case: "Case", user_message: str, image_bytes: bytes | None = None
    ) -> tuple[str, str | None, str | None]:
        case.turn_count = (case.turn_count or 0) + 1
        if self._detect_emergency(user_message):
            case.emergency_trigger = True
        self._apply_followup_parser(case, user_message)
        if image_bytes:
            self._update_classification(case, user_message, image_bytes)
        if self._should_run_nlu(case):
            self._update_nlu(case, user_message)
        missing_slots = self._missing_slots(case)
        missing_unasked = [slot for slot in missing_slots if slot not in self._asked_slots(case)]
        if case.turn_count >= 8 and missing_slots:
            reply = self._fallback_reply(case, missing_slots)
            return reply, case.status, missing_slots[0]
        if missing_unasked:
            reply = self._ask_next(case, missing_unasked)
            return reply, case.status, missing_unasked[0]
        if missing_slots:
            return "Thanks. I can proceed once the remaining detail is provided.", case.status, missing_slots[0]
        decision = self._decision_tree(case)
        reply = self._build_decision_reply(case, decision)
        return reply, case.status, None

    def _update_classification(self, case: "Case", user_message: str, image_bytes: bytes) -> None:
        prompt = (
            "You are a product classifier. "
            "Return JSON only. "
            "Classify into FOOD, FURNITURE, ELECTRONICS, ART. "
            "If unsure set needs_clarification=true and confidence below 0.70. "
            f"User message: {user_message}"
        )
        result = self.gemini.generate_json_with_image(
            prompt, image_bytes, ImageClassification, system_instruction=SYSTEM_INSTRUCTION
        )
        if result.needs_clarification or result.confidence < 0.70:
            return
        if result.category in CATEGORIES:
            case.category = result.category

    def _update_nlu(self, case: "Case", user_message: str) -> None:
        prompt = (
            "You are extracting structured facts for a retail agent. "
            "Return JSON only. "
            "Extract category, intent, requested_action, days_since_purchase, "
            "purchase_date_iso, furniture_assembled, electronics_defect_claimed, "
            "defect_evidence_present, user_sentiment, emergency_trigger. "
            f"User message: {user_message}"
        )
        result = self.gemini.generate_json(
            prompt, NLUUpdate, system_instruction=SYSTEM_INSTRUCTION
        )
        if result.category in CATEGORIES:
            case.category = result.category
        if result.intent in INTENTS:
            case.intent = result.intent
            if result.intent == "ARRIVED_BROKEN":
                case.electronics_defect_claimed = True
            if result.intent == "DID_NOT_LIKE":
                case.electronics_defect_claimed = False
        if result.requested_action:
            case.requested_action = result.requested_action
        if result.days_since_purchase is not None:
            case.days_since_purchase = int(result.days_since_purchase)
        if result.purchase_date_iso:
            case.purchase_date_iso = result.purchase_date_iso
        if result.furniture_assembled is not None:
            case.furniture_assembled = result.furniture_assembled
        if result.electronics_defect_claimed is not None:
            case.electronics_defect_claimed = result.electronics_defect_claimed
        if result.defect_evidence_present is not None:
            case.defect_evidence_present = result.defect_evidence_present
        if result.user_sentiment:
            case.user_sentiment = result.user_sentiment
        if result.emergency_trigger is not None:
            case.emergency_trigger = result.emergency_trigger

    def _missing_slots(self, case: "Case") -> list[str]:
        missing = []
        if not case.category:
            missing.append("category")
        if not case.intent:
            missing.append("intent")
        if case.category == "FURNITURE":
            if case.days_since_purchase is None and not case.purchase_date_iso:
                missing.append("days_since_purchase")
            if case.days_since_purchase is not None and case.days_since_purchase <= 7:
                if case.furniture_assembled is None:
                    missing.append("furniture_assembled")
        if case.category == "ELECTRONICS":
            if case.electronics_defect_claimed is None:
                missing.append("electronics_defect_claimed")
            elif case.electronics_defect_claimed:
                if case.defect_evidence_present is None:
                    missing.append("defect_evidence_present")
        if case.category in {"ART"}:
            if not case.customer_name:
                missing.append("customer_name")
            if not case.pickup_address_json:
                missing.append("pickup_address_json")
            if not case.customer_phone:
                missing.append("customer_phone")
        return missing

    def _decision_tree(self, case: "Case") -> dict:
        if case.category == "FOOD":
            return self._retention(case, "Returns are not available for food items.")
        if case.category == "ART":
            return self._approve(case, "Return approved for art items.")
        if case.category == "ELECTRONICS":
            if case.electronics_defect_claimed is False:
                return self._retention(case, "Electronics returns are only for defective items.")
            if case.electronics_defect_claimed and not case.defect_evidence_present:
                case.status = "awaiting_evidence"
                return {
                    "decision": "needs_evidence",
                    "reason": "Please provide a photo/video or clear defect symptoms.",
                }
            return self._approve(case, "Defect confirmed for electronics.")
        if case.category == "FURNITURE":
            days = case.days_since_purchase or self._days_from_date(case.purchase_date_iso)
            if days is None:
                case.status = "needs_info"
                return {"decision": "needs_info", "reason": "Need purchase timing."}
            if days > 7:
                return self._retention(case, "Furniture returns are limited to 7 days.")
            if case.furniture_assembled:
                return self._retention(case, "Assembled furniture cannot be returned.")
            return self._approve(case, "Furniture return approved.")
        return self._retention(case, "Unable to match policy, using retention.")

    def _approve(self, case: "Case", reason: str) -> dict:
        case.decision = "approved"
        case.status = "approved"
        case.reason = reason
        return {"decision": "approved", "reason": reason}

    def _retention(self, case: "Case", reason: str) -> dict:
        case.decision = "retention"
        case.status = "retention"
        case.reason = reason
        step = case.retention_step or 0
        if case.emergency_trigger:
            step = 4
        else:
            step = min(step + 1, 4)
        case.retention_step = step
        case.discount_percent = self._retention_discount(step)
        return {"decision": "retention", "reason": reason, "step": step}

    def _retention_discount(self, step: int) -> float:
        if step == 1:
            return 0.0
        if step == 2:
            return 6.0
        if step == 3:
            return 11.0
        return 20.0

    def _build_decision_reply(self, case: "Case", decision: dict) -> str:
        if decision["decision"] == "approved":
            return self._data_collection_reply(case)
        if decision["decision"] == "needs_evidence":
            return decision["reason"]
        if decision["decision"] == "retention":
            return self._retention_reply(case)
        return "Thanks. Let me review your case."

    def _data_collection_reply(self, case: "Case") -> str:
        if not case.customer_name:
            self._mark_asked(case, "customer_name")
            case.last_question_slot = "customer_name"
            return "Please provide your full name."
        if not case.pickup_address_json:
            self._mark_asked(case, "pickup_address_json")
            case.last_question_slot = "pickup_address_json"
            return "Please provide your pickup address (city, street, house, apt)."
        if not case.customer_phone:
            self._mark_asked(case, "customer_phone")
            case.last_question_slot = "customer_phone"
            return "Please provide your phone number."
        if not case.ticket_number:
            case.ticket_number = f"{random.randint(0, 99999999):08d}"
        return f"Request #{case.ticket_number} created. Courier will contact you."

    def _retention_reply(self, case: "Case") -> str:
        step = case.retention_step or 1
        if step == 1:
            return "I’m sorry this didn’t work out. While returns aren’t available, I can assist further."
        if step == 2:
            return "I can offer a 6% goodwill coupon to help."
        if step == 3:
            return "I checked with my manager and can offer an 11% coupon."
        return "Given the situation, I can offer a 20% coupon as a final option."

    def _ask_next(self, case: "Case", missing: list[str]) -> str:
        slot = missing[0]
        self._mark_asked(case, slot)
        case.last_question_slot = slot
        case.status = "needs_info"
        if slot == "category":
            return "What category is the product (FOOD, FURNITURE, ELECTRONICS, ART)?"
        if slot == "intent":
            return "Is the issue a refund request, arrived broken, or did not like it?"
        if slot == "days_since_purchase":
            return "How many days since purchase?"
        if slot == "furniture_assembled":
            return "Was the furniture assembled? (yes/no)"
        if slot == "electronics_defect_claimed":
            return "Is it defective/broken, or did you change your mind?"
        if slot == "defect_evidence_present":
            return "Do you have evidence of the defect (image/video or clear symptoms)?"
        if slot == "customer_name":
            return "Please provide your full name."
        if slot == "pickup_address_json":
            return "Please provide your pickup address (city, street, house, apt)."
        if slot == "customer_phone":
            return "Please provide your phone number."
        return "Could you provide the missing detail?"

    def _fallback_reply(self, case: "Case", missing: list[str]) -> str:
        slot = missing[0]
        summary = f"Summary: category={case.category}, intent={case.intent}."
        return f"{summary} I need one more detail: {self._slot_label(slot)}."

    def _slot_label(self, slot: str) -> str:
        labels = {
            "category": "product category",
            "intent": "issue type (refund, arrived broken, did not like)",
            "days_since_purchase": "days since purchase",
            "furniture_assembled": "whether the furniture was assembled",
            "electronics_defect_claimed": "whether it is defective",
            "defect_evidence_present": "defect evidence details",
            "customer_name": "full name",
            "pickup_address_json": "pickup address",
            "customer_phone": "phone number",
        }
        return labels.get(slot, "missing detail")

    def _apply_followup_parser(self, case: "Case", user_message: str) -> None:
        if not case.last_question_slot:
            return
        slot = case.last_question_slot
        if slot == "days_since_purchase":
            value = self._parse_days(user_message)
            if value is not None:
                case.days_since_purchase = value
                case.last_question_slot = None
        elif slot == "furniture_assembled":
            value = self._parse_yes_no(user_message)
            if value is not None:
                case.furniture_assembled = value
                case.last_question_slot = None
        elif slot == "electronics_defect_claimed":
            value = self._parse_defect_claim(user_message)
            if value is not None:
                case.electronics_defect_claimed = value
                case.last_question_slot = None
        elif slot == "defect_evidence_present":
            value = self._parse_yes_no(user_message)
            if value is not None:
                case.defect_evidence_present = value
                case.last_question_slot = None
        elif slot == "customer_name":
            if len(user_message.strip().split()) >= 2:
                case.customer_name = user_message.strip()
                case.last_question_slot = None
        elif slot == "customer_phone":
            phone = self._parse_phone(user_message)
            if phone:
                case.customer_phone = phone
                case.last_question_slot = None
        elif slot == "pickup_address_json":
            address = self._parse_address(user_message)
            if address:
                case.pickup_address_json = address
                case.last_question_slot = None
        elif slot == "intent":
            intent = self._parse_intent(user_message)
            if intent:
                case.intent = intent
                case.last_question_slot = None
        elif slot == "category":
            category = self._parse_category(user_message)
            if category:
                case.category = category
                case.last_question_slot = None

    def _asked_slots(self, case: "Case") -> list[str]:
        if not case.asked_slots:
            return []
        try:
            return json.loads(case.asked_slots)
        except json.JSONDecodeError:
            return []

    def _mark_asked(self, case: "Case", slot: str) -> None:
        asked = self._asked_slots(case)
        if slot not in asked:
            asked.append(slot)
        case.asked_slots = json.dumps(asked)

    def _should_run_nlu(self, case: "Case") -> bool:
        return any(
            field is None
            for field in [
                case.category,
                case.intent,
                case.days_since_purchase,
                case.purchase_date_iso,
                case.furniture_assembled,
                case.electronics_defect_claimed,
                case.defect_evidence_present,
                case.emergency_trigger,
            ]
        )

    @staticmethod
    def _parse_days(message: str) -> Optional[int]:
        match = re.search(r"(\d+)", message)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _parse_yes_no(message: str) -> Optional[bool]:
        text = message.lower()
        if any(token in text for token in ["yes", "да", "yep"]):
            return True
        if any(token in text for token in ["no", "нет", "nope"]):
            return False
        if "unassembled" in text or "not assembled" in text:
            return False
        if "assembled" in text:
            return True
        return None

    @staticmethod
    def _parse_defect_claim(message: str) -> Optional[bool]:
        text = message.lower()
        if "defective" in text or "broken" in text or "doesn't work" in text:
            return True
        if "changed my mind" in text or "don't like" in text:
            return False
        return None

    @staticmethod
    def _parse_intent(message: str) -> Optional[str]:
        text = message.lower()
        if "broken" in text or "defective" in text:
            return "ARRIVED_BROKEN"
        if "refund" in text:
            return "WANT_REFUND"
        if "not like" in text or "changed my mind" in text:
            return "DID_NOT_LIKE"
        return None

    @staticmethod
    def _parse_category(message: str) -> Optional[str]:
        text = message.lower()
        if "food" in text:
            return "FOOD"
        if "furniture" in text or "table" in text or "chair" in text:
            return "FURNITURE"
        if "electronic" in text or "laptop" in text or "phone" in text:
            return "ELECTRONICS"
        if "art" in text or "painting" in text:
            return "ART"
        return None

    @staticmethod
    def _parse_phone(message: str) -> Optional[str]:
        digits = re.sub(r"\D", "", message)
        if len(digits) >= 10:
            return digits
        return None

    @staticmethod
    def _parse_address(message: str) -> Optional[dict]:
        parts = [p.strip() for p in message.split(",") if p.strip()]
        if len(parts) < 3:
            return None
        return {"raw": message.strip(), "parts": parts}

    @staticmethod
    def _days_from_date(date_iso: str | None) -> Optional[int]:
        if not date_iso:
            return None
        try:
            parsed = datetime.fromisoformat(date_iso)
        except ValueError:
            return None
        return (datetime.utcnow() - parsed).days

    @staticmethod
    def _detect_emergency(message: str) -> bool:
        text = message.strip()
        if text.isupper() and len(text) > 8:
            return True
        triggers = ["lawsuit", "sue", "reviews", "consumer protection", "роспотребнадзор"]
        lowered = text.lower()
        return any(trigger in lowered for trigger in triggers)
