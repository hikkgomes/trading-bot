"""Deterministic, persistable risk decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.domain._codec import canonical_hash, json_value, non_empty, timestamp


@dataclass(frozen=True)
class RiskDecision:
    decision_id: str
    scope: str
    accepted: bool
    reason_code: str | None
    evaluated_at: str
    input_snapshot: Mapping[str, Any]
    limits: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", non_empty(self.decision_id, field="decision_id"))
        object.__setattr__(self, "scope", non_empty(self.scope, field="scope"))
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        if self.accepted and self.reason_code is not None:
            raise ValueError("accepted decisions cannot include reason_code")
        if not self.accepted:
            object.__setattr__(
                self, "reason_code", non_empty(self.reason_code or "", field="reason_code")
            )
        object.__setattr__(self, "evaluated_at", timestamp(self.evaluated_at, field="evaluated_at"))
        for attribute in ("input_snapshot", "limits"):
            value = getattr(self, attribute)
            if not isinstance(value, Mapping):
                raise ValueError(f"{attribute} must be an object")
            object.__setattr__(self, attribute, json_value(dict(value), field=attribute))

    @property
    def input_hash(self) -> str:
        return canonical_hash(self.input_snapshot)
