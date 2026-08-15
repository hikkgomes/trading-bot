"""Allowlisted agent context without secrets or protected evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.domain._codec import canonical_hash, json_value, timestamp

ALLOWED_CONTEXT_KEYS = frozenset(
    {
        "strategy_catalogue",
        "feature_catalogue",
        "instrument_universe",
        "development_results",
        "failure_reasons",
        "signal_frequency",
        "cost_model",
        "resource_budget",
        "research_queue",
        "strategy_lineage",
        "public_market_summaries",
    }
)
FORBIDDEN_CONTEXT_MARKERS = (
    "secret",
    "credential",
    "password",
    "api_key",
    "api_secret",
    "approval",
    "protected",
    "holdout",
    "raw_trade",
    "order_book",
)


def _assert_safe(value: Any, *, path: str = "context") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalised = str(key).lower()
            if any(marker in normalised for marker in FORBIDDEN_CONTEXT_MARKERS):
                raise ValueError(f"agent context contains forbidden key: {path}.{key}")
            _assert_safe(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_safe(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class AgentContext:
    created_at: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", timestamp(self.created_at, field="created_at"))
        if not isinstance(self.values, Mapping):
            raise ValueError("agent context must be an object")
        unknown = set(self.values) - ALLOWED_CONTEXT_KEYS
        if unknown:
            raise ValueError(f"agent context contains unsupported keys: {sorted(unknown)}")
        _assert_safe(self.values)
        object.__setattr__(self, "values", json_value(dict(self.values), field="agent context"))

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)
