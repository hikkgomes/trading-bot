"""Position lifecycle state shared by paper and live execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.domain._codec import finite, json_value, non_empty, timestamp


class PositionStatus(StrEnum):
    FLAT = "flat"
    ENTRY_PENDING = "entry_pending"
    OPEN = "open"
    REDUCE_PENDING = "reduce_pending"
    EXIT_PENDING = "exit_pending"
    RECOVERY_PENDING = "recovery_pending"
    FLAT_CONFIRMED = "flat_confirmed"


@dataclass(frozen=True)
class Position:
    portfolio_id: str
    instrument_id: str
    quantity: float
    average_entry_price: float
    status: PositionStatus
    updated_at: str
    strategy_contributions: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attribute in ("portfolio_id", "instrument_id"):
            object.__setattr__(
                self, attribute, non_empty(getattr(self, attribute), field=attribute)
            )
        object.__setattr__(self, "quantity", finite(self.quantity, field="quantity"))
        object.__setattr__(
            self,
            "average_entry_price",
            finite(self.average_entry_price, field="average_entry_price", minimum=0.0),
        )
        if self.quantity == 0 and self.status not in {
            PositionStatus.FLAT,
            PositionStatus.FLAT_CONFIRMED,
        }:
            raise ValueError("zero-quantity positions must be flat")
        if self.quantity != 0 and self.status in {
            PositionStatus.FLAT,
            PositionStatus.FLAT_CONFIRMED,
        }:
            raise ValueError("non-zero positions cannot be flat")
        object.__setattr__(self, "updated_at", timestamp(self.updated_at, field="updated_at"))
        if not isinstance(self.strategy_contributions, Mapping):
            raise ValueError("strategy_contributions must be an object")
        object.__setattr__(
            self,
            "strategy_contributions",
            {
                str(key): finite(value, field="strategy contribution")
                for key, value in self.strategy_contributions.items()
            },
        )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), field="metadata"))
