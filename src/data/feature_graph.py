"""Versioned causal feature graphs shared by historical and live evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
import statistics
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
    SEASONALITY = "seasonality"
    SENTIMENT = "sentiment"
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


class FeatureGraphRegistry:
    """Versioned graph registry with dependency-closed graph selection."""

    def __init__(self) -> None:
        self._nodes: dict[str, FeatureNode] = {}

    def register(self, node: FeatureNode) -> None:
        if node.name in self._nodes:
            raise FeatureGraphError(f"feature node already registered: {node.name}")
        self._nodes[node.name] = node

    def graph(
        self, required_nodes: tuple[str, ...], *, version: str = "canonical-features/v2"
    ) -> FeatureGraph:
        if not required_nodes:
            raise FeatureGraphError("an artefact must declare required feature nodes")
        ordered: list[FeatureNode] = []
        visiting: set[str] = set()
        resolved: set[str] = set()

        def add(name: str) -> None:
            if name in resolved:
                return
            if name in visiting:
                raise FeatureGraphError(f"feature dependency cycle at {name}")
            try:
                node = self._nodes[name]
            except KeyError as exc:
                raise FeatureGraphError(f"unknown feature node: {name}") from exc
            visiting.add(name)
            for dependency in node.dependencies:
                add(dependency)
            visiting.remove(name)
            resolved.add(name)
            ordered.append(node)

        for name in required_nodes:
            add(name)
        return FeatureGraph(version=version, nodes=tuple(ordered))

    @classmethod
    def default(cls) -> FeatureGraphRegistry:
        registry = cls()
        specifications = (
            ("bar_return", FeatureNodeType.BAR_INDICATOR, ()),
            ("log_volume", FeatureNodeType.BAR_INDICATOR, ()),
            ("range_fraction", FeatureNodeType.BAR_INDICATOR, ()),
            ("sma", FeatureNodeType.BAR_INDICATOR, ()),
            ("sma_fast", FeatureNodeType.BAR_INDICATOR, ()),
            ("sma_slow", FeatureNodeType.BAR_INDICATOR, ()),
            ("ema", FeatureNodeType.BAR_INDICATOR, ()),
            ("rsi", FeatureNodeType.BAR_INDICATOR, ()),
            ("macd", FeatureNodeType.BAR_INDICATOR, ()),
            ("atr", FeatureNodeType.BAR_INDICATOR, ()),
            ("realised_volatility", FeatureNodeType.BAR_INDICATOR, ("bar_return",)),
            ("bollinger", FeatureNodeType.BAR_INDICATOR, ("sma",)),
            ("keltner", FeatureNodeType.BAR_INDICATOR, ("ema", "atr")),
            ("adx", FeatureNodeType.BAR_INDICATOR, ("atr",)),
            ("supertrend", FeatureNodeType.BAR_INDICATOR, ("atr",)),
            ("breakout", FeatureNodeType.BAR_INDICATOR, ()),
            ("range_state", FeatureNodeType.BAR_INDICATOR, ()),
            ("multi_timeframe", FeatureNodeType.MULTI_TIMEFRAME, ("bar_return",)),
            ("taker_flow", FeatureNodeType.TRADE_FLOW, ()),
            ("funding", FeatureNodeType.FUNDING_OPEN_INTEREST, ()),
            ("open_interest", FeatureNodeType.FUNDING_OPEN_INTEREST, ()),
            ("spot_perpetual_basis", FeatureNodeType.SPOT_PERPETUAL_BASIS, ()),
            ("cross_sectional_rank", FeatureNodeType.CROSS_SECTIONAL_RANK, ()),
            ("beta", FeatureNodeType.CORRELATION_BETA, ()),
            ("correlation", FeatureNodeType.CORRELATION_BETA, ()),
            ("bid_ask_spread", FeatureNodeType.ORDER_BOOK, ()),
            ("depth_imbalance", FeatureNodeType.ORDER_BOOK, ()),
            ("microprice", FeatureNodeType.ORDER_BOOK, ("depth_imbalance",)),
            ("aggressor_flow", FeatureNodeType.TRADE_FLOW, ()),
            ("liquidation_flow", FeatureNodeType.LIQUIDATION, ()),
            ("seasonality", FeatureNodeType.SEASONALITY, ()),
            ("sentiment", FeatureNodeType.SENTIMENT, ()),
            ("frozen_ml_feature_vector", FeatureNodeType.FROZEN_ML_FEATURE_SET, ()),
        )
        for name, node_type, dependencies in specifications:
            parameters = (
                {"period": 20 if name == "sma_fast" else 50} if name.startswith("sma_") else {}
            )
            registry.register(FeatureNode(name, node_type, dependencies, parameters))
        return registry


def default_feature_engine() -> FeatureGraphEngine:
    return FeatureGraphEngine({node_type: _evaluate_default_node for node_type in FeatureNodeType})


def _raw(inputs: Mapping[str, AvailableValue], name: str, default: Any = None) -> Any:
    item = inputs.get(name)
    return default if item is None else item.value


def _series(inputs: Mapping[str, AvailableValue], name: str) -> list[float]:
    value = _raw(inputs, name, ())
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return [float(item) for item in value]
    return [float(value)] if value is not None else []


def _evaluate_default_node(
    node: FeatureNode, dependencies: Mapping[str, Any], inputs: Mapping[str, AvailableValue]
) -> Any:
    close = float(_raw(inputs, "close", 0.0))
    open_price = float(_raw(inputs, "open", close))
    high = float(_raw(inputs, "high", close))
    low = float(_raw(inputs, "low", close))
    volume = max(0.0, float(_raw(inputs, "volume", 0.0)))
    closes = _series(inputs, "close_history") or [close]
    returns = [
        closes[index] / closes[index - 1] - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]
    name = node.name
    if name == "bar_return":
        return close / open_price - 1.0 if open_price > 0 else 0.0
    if name == "log_volume":
        return math.log1p(volume)
    if name == "range_fraction":
        return (high - low) / close if close > 0 else 0.0
    if name in {"sma", "sma_fast", "sma_slow"}:
        return statistics.fmean(closes[-int(node.parameters.get("period", 20)) :])
    if name == "ema":
        period = int(node.parameters.get("period", 20))
        alpha = 2.0 / (period + 1)
        value = closes[0]
        for item in closes[1:]:
            value = alpha * item + (1 - alpha) * value
        return value
    if name == "rsi":
        gains = [max(0.0, value) for value in returns[-14:]]
        losses = [max(0.0, -value) for value in returns[-14:]]
        gain, loss = sum(gains), sum(losses)
        return (
            100.0 if loss == 0 and gain else 100.0 - 100.0 / (1.0 + gain / loss) if loss else 50.0
        )
    if name == "macd":
        return _ema(closes, 12) - _ema(closes, 26)
    if name == "atr":
        return (high - low) / close if close > 0 else 0.0
    if name == "realised_volatility":
        return statistics.pstdev(returns) * math.sqrt(len(returns)) if len(returns) > 1 else 0.0
    if name == "bollinger":
        deviation = statistics.pstdev(closes[-20:]) if len(closes) > 1 else 0.0
        return (close - float(dependencies["sma"])) / deviation if deviation else 0.0
    if name == "keltner":
        width = float(dependencies["atr"])
        return (
            (close - float(dependencies["ema"])) / (close * width) if close > 0 and width else 0.0
        )
    if name in {"adx", "supertrend"}:
        return (close - open_price) / max(high - low, 1e-12)
    if name == "breakout":
        previous_high = max(closes[:-1], default=close)
        return close / previous_high - 1.0 if previous_high > 0 else 0.0
    if name == "range_state":
        return (max(closes) - min(closes)) / close if close > 0 else 0.0
    if name == "multi_timeframe":
        return float(_raw(inputs, "higher_timeframe_return", dependencies["bar_return"]))
    if name in {"taker_flow", "aggressor_flow"}:
        buy = float(_raw(inputs, "taker_buy_volume", 0.0))
        return (2.0 * buy - volume) / volume if volume else 0.0
    if name in {"funding", "open_interest", "beta", "correlation", "liquidation_flow", "sentiment"}:
        return float(_raw(inputs, name, 0.0))
    if name == "spot_perpetual_basis":
        spot, perpetual = (
            float(_raw(inputs, "spot_price", close)),
            float(_raw(inputs, "perpetual_price", close)),
        )
        return perpetual / spot - 1.0 if spot > 0 else 0.0
    if name == "cross_sectional_rank":
        return float(_raw(inputs, "cross_sectional_rank", 0.5))
    if name == "bid_ask_spread":
        bid, ask = float(_raw(inputs, "bid_price", close)), float(_raw(inputs, "ask_price", close))
        return (ask - bid) / ((ask + bid) / 2) if ask + bid > 0 else 0.0
    if name == "depth_imbalance":
        bid, ask = float(_raw(inputs, "bid_depth", 0.0)), float(_raw(inputs, "ask_depth", 0.0))
        return (bid - ask) / (bid + ask) if bid + ask else 0.0
    if name == "microprice":
        bid, ask = float(_raw(inputs, "bid_price", close)), float(_raw(inputs, "ask_price", close))
        bid_depth, ask_depth = (
            float(_raw(inputs, "bid_depth", 0.0)),
            float(_raw(inputs, "ask_depth", 0.0)),
        )
        return (
            (ask * bid_depth + bid * ask_depth) / (bid_depth + ask_depth)
            if bid_depth + ask_depth
            else (bid + ask) / 2
        )
    if name == "seasonality":
        return float(_raw(inputs, "seasonality_score", 0.0))
    if name == "frozen_ml_feature_vector":
        names = tuple(node.parameters.get("feature_names", ()))
        return {feature: float(_raw(inputs, str(feature), 0.0)) for feature in names}
    raise FeatureGraphError(f"no default evaluator for {name}")


def _ema(values: list[float], period: int) -> float:
    alpha = 2.0 / (period + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result
