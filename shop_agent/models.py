from __future__ import annotations

from pydantic import BaseModel


class ImageClassification(BaseModel):
    category: str
    confidence: float
    observations: str
    needs_clarification: bool


class NLUUpdate(BaseModel):
    category: str | None = None
    intent: str | None = None
    requested_action: str | None = None
    days_since_purchase: int | None = None
    purchase_date_iso: str | None = None
    furniture_assembled: bool | None = None
    electronics_defect_claimed: bool | None = None
    defect_evidence_present: bool | None = None
    user_sentiment: str | None = None
    emergency_trigger: bool | None = None
