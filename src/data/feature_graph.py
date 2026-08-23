"""Versioned causal feature graphs shared by historical and live evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.domain._codec import canonical_hash, json_value, non_empty, timestamp


class FeatureGraphError(RuntimeError):
    pass


class FeatureNodeType(StrEnum):
    BAR_INDICATOR = "bar_indicator"
    MULTI_TIMEFRAME = "multi_timeframe"
    CROSS_SECTIONAL_RANK = "cross_sectional_rank"
    FUNDING_OPEN_INTEREST = "funding_open_interest"
    SPOT_PERPETUAL_BASIS = "spot_perpetual_basis"
    CORRELATION_BETA = "correlation_beta"
    ORDER_BOOK = "order_book"
    TRADE_FLOW = "trade_flow"
    LIQUIDATION = "liquidation"
    FROZEN_ML_FEATURE_SET = "frozen_ml_feature_set"


@dataclass(frozen=True)
class FeatureNode:
    name: str
    node_type: FeatureNodeType
    dependencies: tuple[str, ...]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", non_empty(self.name, field="feature name"))
        object.__setattr__(
            self, "parameters", json_value(dict(self.parameters), field="feature parameters")
        )


@dataclass(frozen=True)
class FeatureGraph:
    version: str
    nodes: tuple[FeatureNode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", non_empty(self.version, field="feature graph version"))
        names = [node.name for node in self.nodes]
        if not names or len(names) != len(set(names)):
            raise FeatureGraphError("feature graph needs unique nodes")
        known: set[str] = set()
        for node in self.nodes:
            unresolved = set(node.dependencies) - known
            if unresolved:
                raise FeatureGraphError(
                    f"feature node {node.name} has unresolved dependencies: {sorted(unresolved)}"
                )
            known.add(node.name)

    @property
    def graph_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class AvailableValue:
    value: Any
    information_time: str
    availability_time: str

    def __post_init__(self) -> None:
        information = timestamp(self.information_time, field="information_time")
        availability = timestamp(self.availability_time, field="availability_time")
        if availability < information:
            raise FeatureGraphError("availability cannot precede information time")
        object.__setattr__(self, "information_time", information)
        object.__setattr__(self, "availability_time", availability)


NodeEvaluator = Callable[[FeatureNode, Mapping[str, Any], Mapping[str, AvailableValue]], Any]


class FeatureGraphEngine:
    def __init__(self, evaluators: Mapping[FeatureNodeType, NodeEvaluator]):
        self.evaluators = dict(evaluators)
        missing = set(FeatureNodeType) - set(self.evaluators)
        if missing:
            raise FeatureGraphError(
                "feature engine has no evaluators for: "
                + ", ".join(sorted(item.value for item in missing))
            )

    def evaluate(
        self,
        graph: FeatureGraph,
        *,
        information_timestamp: str,
        inputs: Mapping[str, AvailableValue],
    ) -> Mapping[str, Any]:
        at = timestamp(information_timestamp, field="information_timestamp")
        unavailable = sorted(name for name, item in inputs.items() if item.availability_time > at)
        if unavailable:
            raise FeatureGraphError(
                "feature inputs are not available at the evaluation timestamp: "
                + ", ".join(unavailable)
            )
        resolved: dict[str, Any] = {}
        for node in graph.nodes:
            dependencies = {name: resolved[name] for name in node.dependencies}
            first = self.evaluators[node.node_type](node, dependencies, inputs)
            second = self.evaluators[node.node_type](node, dependencies, inputs)
            if canonical_hash(first) != canonical_hash(second):
                raise FeatureGraphError(f"feature node is non-deterministic: {node.name}")
            resolved[node.name] = first
        return resolved
