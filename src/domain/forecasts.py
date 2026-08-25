"""The one forecast contract used by all strategy families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.domain._codec import finite, json_value, non_empty, timestamp


class ForecastDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class AlphaForecast:
    strategy_version_id: str
    product_id: str
    instrument_id: str
    direction: ForecastDirection
    score: float
    expected_return: float
    confidence: float
    horizon_seconds: int
    valid_from: str
    valid_until: str
    target_volatility: float
    maximum_position: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attribute in ("strategy_version_id", "product_id", "instrument_id"):
            object.__setattr__(
                self, attribute, non_empty(getattr(self, attribute), field=attribute)
            )
        for attribute in ("score", "confidence"):
            value = finite(getattr(self, attribute), field=attribute, minimum=0.0)
            if value > 1:
                raise ValueError(f"{attribute} must be at most 1")
            object.__setattr__(self, attribute, value)
        object.__setattr__(
            self, "expected_return", finite(self.expected_return, field="expected_return")
        )
        object.__setattr__(
            self,
            "target_volatility",
            finite(self.target_volatility, field="target_volatility", minimum=0.0),
        )
        maximum_position = finite(self.maximum_position, field="maximum_position", minimum=0.0)
        if maximum_position > 1:
            raise ValueError("maximum_position must be at most 1")
        object.__setattr__(self, "maximum_position", maximum_position)
        if isinstance(self.horizon_seconds, bool) or not isinstance(self.horizon_seconds, int):
            raise ValueError("horizon_seconds must be a positive integer")
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be a positive integer")
        valid_from = timestamp(self.valid_from, field="valid_from")
        valid_until = timestamp(self.valid_until, field="valid_until")
        if valid_until <= valid_from:
            raise ValueError("valid_until must be after valid_from")
        object.__setattr__(self, "valid_from", valid_from)
        object.__setattr__(self, "valid_until", valid_until)
        if self.direction is ForecastDirection.FLAT and self.maximum_position != 0:
            raise ValueError("flat forecasts must have maximum_position=0")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), field="metadata"))

    @property
    def signed_strength(self) -> float:
        sign = 1.0 if self.direction is ForecastDirection.LONG else -1.0
        return 0.0 if self.direction is ForecastDirection.FLAT else sign * self.score

    @property
    def utility(self) -> float:
        return max(0.0, self.expected_return) * self.score * self.confidence
