"""Immutable strategy-definition metadata and source provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.domain._codec import canonical_hash, json_value, non_empty


class StrategySourceType(StrEnum):
    REGISTERED_PYTHON = "registered_python"
    PARAMETER_SEARCH = "parameter_search"
    GENERATED_DSL = "generated_dsl"
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    MACHINE_LEARNING = "machine_learning"
    CROSS_SECTIONAL = "cross_sectional"
    RELATIVE_VALUE = "relative_value"
    MICROSTRUCTURE = "microstructure"
    ENSEMBLE = "ensemble"
    AGENT_GENERATED_PYTHON = "agent_generated_python"


@dataclass(frozen=True)
class StrategyDefinition:
    identity: str
    version: str
    family: str
    product: str
    universe: Mapping[str, Any]
    data_requirements: Mapping[str, Any]
    feature_graph: Mapping[str, Any]
    signal_model: Mapping[str, Any]
    position_model: Mapping[str, Any]
    execution_preferences: Mapping[str, Any]
    risk_policy: Mapping[str, Any]
    validation_policy: Mapping[str, Any]
    source_type: StrategySourceType
    source_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attribute in ("identity", "version", "family", "product", "source_hash"):
            object.__setattr__(
                self, attribute, non_empty(getattr(self, attribute), field=attribute)
            )
        if (
            not self.source_hash.startswith("sha256:")
            or len(self.source_hash) != 71
            or any(character not in "0123456789abcdef" for character in self.source_hash[7:])
        ):
            raise ValueError("source_hash must be a sha256: identity")
        for attribute in (
            "universe",
            "data_requirements",
            "feature_graph",
            "signal_model",
            "position_model",
            "execution_preferences",
            "risk_policy",
            "validation_policy",
            "metadata",
        ):
            value = getattr(self, attribute)
            if not isinstance(value, Mapping):
                raise ValueError(f"{attribute} must be an object")
            object.__setattr__(self, attribute, json_value(dict(value), field=attribute))

    @property
    def strategy_version_id(self) -> str:
        return f"{self.identity}:{self.version}"

    @property
    def definition_hash(self) -> str:
        return canonical_hash(self)
