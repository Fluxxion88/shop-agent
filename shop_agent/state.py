from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PolicyOutcome:
    eligible: bool
    outcome: str
    discount_percent: float
    reason: str
    refused_excess_discount: bool = False


@dataclass
class SessionState:
    session_id: str
    inferred_intent: str = "unknown"
    user_goal_summary: str = ""
    category: Optional[str] = None
    item_guess: Optional[str] = None
    confidence: float = 0.0
    missing_info: List[str] = field(default_factory=list)
    last_policy_outcome: Optional[PolicyOutcome] = None
    days_since_purchase: Optional[int] = None
    item_opened: Optional[bool] = None
    requested_discount: Optional[float] = None

    def to_json(self) -> str:
        payload = asdict(self)
        if self.last_policy_outcome is not None:
            payload["last_policy_outcome"] = asdict(self.last_policy_outcome)
        return json.dumps(payload, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        outcome = data.get("last_policy_outcome")
        if outcome:
            data = {**data, "last_policy_outcome": PolicyOutcome(**outcome)}
        return cls(**data)


def session_path(session_id: str, base_dir: Path | None = None) -> Path:
    root = base_dir or Path.cwd()
    sessions_dir = root / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    return sessions_dir / f"{session_id}.json"


def load_session(session_id: str, base_dir: Path | None = None) -> SessionState:
    path = session_path(session_id, base_dir)
    if not path.exists():
        return SessionState(session_id=session_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    return SessionState.from_dict(data)


def save_session(state: SessionState, base_dir: Path | None = None) -> None:
    path = session_path(state.session_id, base_dir)
    path.write_text(state.to_json(), encoding="utf-8")
