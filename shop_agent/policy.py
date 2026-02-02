from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shop_agent.state import PolicyOutcome


@dataclass
class CategoryPolicy:
    name: str
    return_window_days: int
    allowed_outcomes: List[str]
    discount_cap_percent: float
    tiered_discounts: List[Dict[str, Any]]
    special_constraints: List[str]


class PolicyEngine:
    def __init__(self, policies: Dict[str, CategoryPolicy]):
        self.policies = policies

    @classmethod
    def from_file(cls, path: Path) -> "PolicyEngine":
        payload = json.loads(path.read_text(encoding="utf-8"))
        categories = {}
        for name, data in payload["categories"].items():
            categories[name] = CategoryPolicy(
                name=name,
                return_window_days=data["return_window_days"],
                allowed_outcomes=list(data["allowed_outcomes"]),
                discount_cap_percent=float(data["discount_cap_percent"]),
                tiered_discounts=list(data.get("tiered_discounts", [])),
                special_constraints=list(data.get("special_constraints", [])),
            )
        return cls(categories)

    def evaluate(
        self,
        category: str,
        intent: str,
        days_since_purchase: Optional[int],
        item_opened: Optional[bool],
        requested_discount: Optional[float],
    ) -> PolicyOutcome:
        policy = self.policies[category]
        missing = []
        if days_since_purchase is None:
            missing.append("days_since_purchase")
        if intent == "return" and item_opened is None:
            missing.append("item_opened")

        if missing:
            return PolicyOutcome(
                eligible=False,
                outcome="needs_info",
                discount_percent=0.0,
                reason=f"Missing required info: {', '.join(missing)}.",
            )

        days_since_purchase = int(days_since_purchase)
        within_window = days_since_purchase <= policy.return_window_days

        if intent in {"refund", "return", "replacement"} and not within_window:
            return PolicyOutcome(
                eligible=False,
                outcome="not_eligible",
                discount_percent=0.0,
                reason="Return window exceeded based on store policy.",
            )

        if category == "Headphones & Audio" and intent in {"refund", "return"} and item_opened:
            return PolicyOutcome(
                eligible=False,
                outcome="not_eligible",
                discount_percent=0.0,
                reason="Opened in-ear headphones are not eligible for refund.",
            )

        if intent not in policy.allowed_outcomes:
            return PolicyOutcome(
                eligible=False,
                outcome="not_eligible",
                discount_percent=0.0,
                reason="Requested outcome is not allowed for this category.",
            )

        if intent == "discount":
            discount = self._determine_discount(policy, days_since_purchase)
            refused = False
            if requested_discount is not None and requested_discount > policy.discount_cap_percent:
                refused = True
            if requested_discount is not None:
                discount = min(discount, requested_discount)
            discount = min(discount, policy.discount_cap_percent)
            return PolicyOutcome(
                eligible=True,
                outcome="discount",
                discount_percent=discount,
                reason="Discount determined by policy tiers and caps.",
                refused_excess_discount=refused,
            )

        return PolicyOutcome(
            eligible=True,
            outcome=intent,
            discount_percent=0.0,
            reason="Eligible under store policy.",
        )

    def _determine_discount(self, policy: CategoryPolicy, days_since_purchase: int) -> float:
        for tier in policy.tiered_discounts:
            if days_since_purchase <= tier["max_days"]:
                return float(tier["percent"])
        return float(policy.discount_cap_percent)
