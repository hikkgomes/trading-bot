"""Declared, bounded inputs shared by research scheduling and data acquisition."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.domain._codec import canonical_hash, timeframe_seconds, timestamp


@dataclass(frozen=True)
class ResearchDataPlan:
    product_id: str
    timeframe: str
    instrument_scope: tuple[str, ...]
    minimum_history_days: int = 365
    warmup_bars: int = 200
    maximum_rows: int = 500_000
    maximum_bytes: int = 1_073_741_824
    estimated_bytes_per_row: int = 2048
    batch_rows: int = 4096

    def __post_init__(self) -> None:
        timeframe_seconds(self.timeframe)
        if not self.product_id or not self.instrument_scope:
            raise ValueError("research data plan requires a product and instrument scope")
        if tuple(sorted(set(self.instrument_scope))) != self.instrument_scope:
            raise ValueError("research data plan scope must be sorted and unique")
        for field in ("maximum_rows", "maximum_bytes", "estimated_bytes_per_row", "batch_rows"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"research data plan {field} must be a positive integer")
        for field in ("minimum_history_days", "warmup_bars"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"research data plan {field} must be a non-negative integer")
        self.validate_capacity(len(self.instrument_scope))

    def validate_capacity(self, instruments: int) -> None:
        rows = self.rows_per_instrument * instruments
        if rows > self.maximum_rows or rows * self.estimated_bytes_per_row > self.maximum_bytes:
            raise ValueError(
                "research data plan exceeds its row/byte budget; declare a feasible timeframe or scope"
            )

    @property
    def rows_per_instrument(self) -> int:
        return (
            self.minimum_history_days * 86_400 // timeframe_seconds(self.timeframe)
            + 1
            + self.warmup_bars
        )

    @property
    def required_rows(self) -> int:
        return self.rows_per_instrument * len(self.instrument_scope)

    @property
    def estimated_bytes(self) -> int:
        return self.required_rows * self.estimated_bytes_per_row

    @property
    def plan_id(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "research_data_plan/v1",
            **asdict(self),
            "instrument_scope": list(self.instrument_scope),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ResearchDataPlan:
        data = dict(value)
        if data.pop("schema", None) != "research_data_plan/v1":
            raise ValueError("unsupported research data plan schema")
        data["instrument_scope"] = tuple(data["instrument_scope"])
        return cls(**data)

    def acquisition_interval(self, as_of: str) -> tuple[int, int]:
        interval = timeframe_seconds(self.timeframe) * 1000
        current = dt.datetime.fromisoformat(timestamp(as_of, field="as_of"))
        end = int(current.timestamp() * 1000) // interval * interval
        return end - self.rows_per_instrument * interval, end


def product_data_plan(
    product: Mapping[str, Any], research: Mapping[str, Any], symbols: Sequence[str]
) -> ResearchDataPlan:
    requirements = research.get("data_requirements", {})
    return ResearchDataPlan(
        product_id=str(product["product_id"]),
        timeframe=str(product.get("dataset_timeframe", "1m")),
        instrument_scope=tuple(sorted(set(symbols))),
        minimum_history_days=requirements.get("minimum_history_days", 365),
        warmup_bars=requirements.get("warmup_bars", 200),
        **research.get("data_resource_limits", {}),
    )
