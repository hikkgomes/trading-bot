"""Immutable strategy-definition metadata and source provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.domain._codec import canonical_hash, json_value, non_empty

_UNIVERSE_FIELDS = frozenset(
    {"type", "symbols", "instrument_ids", "universe_snapshot_id", "snapshot_id", "dynamic"}
)


def _validate_universe(value: Mapping[str, Any], *, product: str) -> None:
    unknown = sorted(set(value) - _UNIVERSE_FIELDS)
    if unknown:
        raise ValueError("universe contains unsupported fields: " + ", ".join(unknown))
    universe_type = value.get("type")
    if universe_type is not None and str(universe_type) not in {"fixed", "point_in_time", "dynamic"}:
        raise ValueError("universe type is unsupported")
    for field_name in ("symbols", "instrument_ids"):
        declared = value.get(field_name)
        if declared is None:
            continue
        if (
            not isinstance(declared, list | tuple)
            or not declared
            or any(not str(item).strip() for item in declared)
            or len({str(item) for item in declared}) != len(declared)
        ):
            raise ValueError(f"universe {field_name} must contain unique non-empty values")
    if universe_type == "point_in_time" and not value.get("universe_snapshot_id"):
        raise ValueError("point-in-time universes need a universe_snapshot_id")
    if universe_type == "fixed" and not value.get("symbols") and not value.get("instrument_ids"):
        raise ValueError("fixed universes need symbols or instrument_ids")
    if product == "btc_accumulation":
        symbols = tuple(str(item).upper() for item in value.get("symbols", ()))
        instrument_ids = tuple(str(item) for item in value.get("instrument_ids", ()))
        if symbols != ("BTCUSDT",) or (
            instrument_ids and instrument_ids != ("binance:spot:BTCUSDT",)
        ):
            raise ValueError("BTC accumulation definitions require BTCUSDT spot only")


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


class MechanismCategory(StrEnum):
    BEHAVIOURAL = "behavioural"
    RISK_PREMIUM = "risk_premium"
    INFORMATION_DIFFUSION = "information_diffusion"
    FORCED_FLOW = "forced_flow"
    LIQUIDITY = "liquidity"
    MARKET_STRUCTURE = "market_structure"
    RELATIVE_VALUE = "relative_value"
    CARRY = "carry"
    EXECUTION = "execution"


@dataclass(frozen=True)
class ResearchThesis:
    """Predeclared, immutable identity for one economic research idea."""

    mechanism_category: MechanismCategory
    market_rationale: str
    expected_causal_chain: tuple[str, ...]
    expected_direction: str
    expected_horizon: str
    required_data: tuple[str, ...]
    permitted_features: tuple[str, ...]
    instrument_universe: tuple[str, ...]
    generalisation_scope: Mapping[str, Any]
    failure_regimes: tuple[str, ...]
    falsification_tests: tuple[str, ...]
    negative_controls: tuple[str, ...]
    execution_capacity_assumptions: Mapping[str, Any]
    parent_thesis_ids: tuple[str, ...]
    cumulative_trial_budget: int
    created_at: str
    creator_identity: str

    def __post_init__(self) -> None:
        from src.domain._codec import timestamp

        for attribute in (
            "market_rationale",
            "expected_direction",
            "expected_horizon",
            "creator_identity",
        ):
            object.__setattr__(
                self, attribute, non_empty(getattr(self, attribute), field=attribute)
            )
        for attribute in (
            "expected_causal_chain",
            "required_data",
            "permitted_features",
            "instrument_universe",
            "failure_regimes",
            "falsification_tests",
            "negative_controls",
            "parent_thesis_ids",
        ):
            values = tuple(non_empty(item, field=attribute) for item in getattr(self, attribute))
            object.__setattr__(self, attribute, values)
        if not self.expected_causal_chain or not self.required_data or not self.instrument_universe:
            raise ValueError(
                "a thesis needs a causal chain, required data, and predeclared universe"
            )
        if self.cumulative_trial_budget < 1:
            raise ValueError("cumulative_trial_budget must be positive")
        for attribute in ("generalisation_scope", "execution_capacity_assumptions"):
            value = getattr(self, attribute)
            if not isinstance(value, Mapping):
                raise ValueError(f"{attribute} must be an object")
            object.__setattr__(self, attribute, json_value(dict(value), field=attribute))
        object.__setattr__(self, "created_at", timestamp(self.created_at, field="created_at"))

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)

    @property
    def thesis_id(self) -> str:
        return self.content_hash


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
        _validate_universe(self.universe, product=self.product)

    @property
    def strategy_version_id(self) -> str:
        return self.definition_hash

    @property
    def definition_hash(self) -> str:
        # ``version`` is a display label. Authority follows executable content,
        # so relabelling an unchanged definition cannot create a new trial or
        # approval identity.
        return canonical_hash(
            {
                "identity": self.identity,
                "family": self.family,
                "product": self.product,
                "universe": self.universe,
                "data_requirements": self.data_requirements,
                "feature_graph": self.feature_graph,
                "signal_model": self.signal_model,
                "position_model": self.position_model,
                "execution_preferences": self.execution_preferences,
                "risk_policy": self.risk_policy,
                "validation_policy": self.validation_policy,
                "source_type": self.source_type,
                "source_hash": self.source_hash,
            }
        )
