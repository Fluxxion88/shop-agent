from __future__ import annotations

from pydantic import BaseModel, Field


class IntentExtraction(BaseModel):
    inferred_intent: str = Field(
        ..., description="One of refund, return, discount, replacement, unknown"
    )
    user_goal_summary: str
    days_since_purchase: int | None
    item_opened: bool | None
    requested_discount: float | None
    missing_info: list[str]


class ImageClassification(BaseModel):
    item_name_guess: str
    category: str
    confidence: float
    observations: str
    needs_clarification: bool


class NLUUpdate(BaseModel):
    user_goal: str | None = None
    user_goal_summary: str | None = None
    category: str | None = None
    days_since_purchase: int | None = None
    item_opened: bool | None = None
    condition: str | None = None
    purchase_price: float | None = None
    amazon_asin: str | None = None
    amazon_url: str | None = None
    requested_discount: float | None = None
