"""Portfolio outputs are target positions, never direct orders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.domain._codec import finite, json_value, non_empty, timestamp


@dataclass(frozen=True)
class TargetPosition:
    portfolio_id: str
    instrument_id: str
    target_quantity: float
    target_notional: float
    target_fraction: float
    strategy_contributions: Mapping[str, float]
    risk_budget: float
    valid_until: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attribute in ("portfolio_id", "instrument_id"):
            object.__setattr__(
                self, attribute, non_empty(getattr(self, attribute), field=attribute)
            )
        for attribute in ("target_quantity", "target_notional"):
            object.__setattr__(self, attribute, finite(getattr(self, attribute), field=attribute))
        fraction = finite(self.target_fraction, field="target_fraction")
        if not -1 <= fraction <= 1:
            raise ValueError("target_fraction must be in [-1, 1]")
        object.__setattr__(self, "target_fraction", fraction)
        object.__setattr__(
            self, "risk_budget", finite(self.risk_budget, field="risk_budget", minimum=0.0)
        )
        if not isinstance(self.strategy_contributions, Mapping):
            raise ValueError("strategy_contributions must be an object")
        contributions = {
            non_empty(key, field="strategy contribution id"): finite(
                value, field="strategy contribution"
            )
            for key, value in self.strategy_contributions.items()
        }
        object.__setattr__(self, "strategy_contributions", contributions)
        object.__setattr__(self, "valid_until", timestamp(self.valid_until, field="valid_until"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), field="metadata"))
