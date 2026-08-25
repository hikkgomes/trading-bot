"""Versioned causal feature graphs shared by historical and live evaluation."""

from __future__ import annotations

import math
import statistics
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
    SEASONALITY = "seasonality"
    SENTIMENT = "sentiment"
    FROZEN_ML_FEATURE_SET = "frozen_ml_feature_set"


@dataclass(frozen=True)
class FeatureNode:
    name: str
    node_type: FeatureNodeType
    dependencies: tuple[str, ...]
    parameters: Mapping[str, Any]
    required_inputs: tuple[str, ...] = ()
    input_types: Mapping[str, str] = None  # type: ignore[assignment]
    minimum_history: int = 1
    missing_data_policy: str = "fail_closed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", non_empty(self.name, field="feature name"))
        object.__setattr__(
            self, "parameters", json_value(dict(self.parameters), field="feature parameters")
        )
        object.__setattr__(
            self, "required_inputs", tuple(str(item) for item in self.required_inputs)
        )
        object.__setattr__(
            self,
            "input_types",
            json_value(dict(self.input_types or {}), field="feature input types"),
        )
        if self.minimum_history <= 0:
            raise FeatureGraphError("feature minimum_history must be positive")
        if self.missing_data_policy != "fail_closed":
            raise FeatureGraphError("feature nodes must use the fail_closed missing-data policy")


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
            _validate_node_inputs(node, inputs, at)
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
        specifications = {
            "bar_return": (FeatureNodeType.BAR_INDICATOR, (), ("open", "close"), {}, 1),
            "log_volume": (FeatureNodeType.BAR_INDICATOR, (), ("volume",), {}, 1),
            "range_fraction": (FeatureNodeType.BAR_INDICATOR, (), ("high", "low", "close"), {}, 1),
            "sma": (FeatureNodeType.BAR_INDICATOR, (), ("close_history",), {"period": 20}, 20),
            "sma_fast": (FeatureNodeType.BAR_INDICATOR, (), ("close_history",), {"period": 20}, 20),
            "sma_slow": (FeatureNodeType.BAR_INDICATOR, (), ("close_history",), {"period": 50}, 50),
            "ema": (FeatureNodeType.BAR_INDICATOR, (), ("close_history",), {"period": 20}, 20),
            "rsi": (FeatureNodeType.BAR_INDICATOR, (), ("close_history",), {"period": 14}, 15),
            "macd": (
                FeatureNodeType.BAR_INDICATOR,
                (),
                ("close_history",),
                {"fast": 12, "slow": 26},
                26,
            ),
            "atr": (
                FeatureNodeType.BAR_INDICATOR,
                (),
                ("high_history", "low_history", "close_history"),
                {"period": 14},
                15,
            ),
            "realised_volatility": (
                FeatureNodeType.BAR_INDICATOR,
                ("bar_return",),
                ("close_history",),
                {},
                3,
            ),
            "bollinger": (
                FeatureNodeType.BAR_INDICATOR,
                ("sma",),
                ("close_history",),
                {"period": 20},
                20,
            ),
            "keltner": (FeatureNodeType.BAR_INDICATOR, ("ema", "atr"), ("close_history",), {}, 20),
            "adx": (
                FeatureNodeType.BAR_INDICATOR,
                ("atr",),
                ("high_history", "low_history", "close_history"),
                {"period": 14},
                15,
            ),
            "supertrend": (
                FeatureNodeType.BAR_INDICATOR,
                ("atr",),
                ("high_history", "low_history", "close_history"),
                {"period": 10, "multiplier": 3.0},
                11,
            ),
            "breakout": (FeatureNodeType.BAR_INDICATOR, (), ("close_history",), {"period": 20}, 21),
            "range_state": (FeatureNodeType.BAR_INDICATOR, (), ("close_history",), {}, 2),
            "multi_timeframe": (
                FeatureNodeType.MULTI_TIMEFRAME,
                ("bar_return",),
                ("higher_timeframe_return",),
                {},
                1,
            ),
            "taker_flow": (FeatureNodeType.TRADE_FLOW, (), ("taker_buy_volume", "volume"), {}, 1),
            "funding": (FeatureNodeType.FUNDING_OPEN_INTEREST, (), ("funding_rate",), {}, 1),
            "open_interest": (FeatureNodeType.FUNDING_OPEN_INTEREST, (), ("open_interest",), {}, 1),
            "spot_perpetual_basis": (
                FeatureNodeType.SPOT_PERPETUAL_BASIS,
                (),
                ("spot_price", "perpetual_price"),
                {},
                1,
            ),
            "cross_sectional_rank": (
                FeatureNodeType.CROSS_SECTIONAL_RANK,
                (),
                ("cross_sectional_values",),
                {},
                1,
            ),
            "beta": (
                FeatureNodeType.CORRELATION_BETA,
                (),
                ("asset_returns", "benchmark_returns"),
                {},
                2,
            ),
            "correlation": (
                FeatureNodeType.CORRELATION_BETA,
                (),
                ("asset_returns", "benchmark_returns"),
                {},
                2,
            ),
            "bid_ask_spread": (FeatureNodeType.ORDER_BOOK, (), ("bid_price", "ask_price"), {}, 1),
            "depth_imbalance": (FeatureNodeType.ORDER_BOOK, (), ("bid_depth", "ask_depth"), {}, 1),
            "microprice": (
                FeatureNodeType.ORDER_BOOK,
                ("depth_imbalance",),
                ("bid_price", "ask_price", "bid_depth", "ask_depth"),
                {},
                1,
            ),
            "aggressor_flow": (
                FeatureNodeType.TRADE_FLOW,
                (),
                ("aggressor_buy_volume", "aggressor_sell_volume"),
                {},
                1,
            ),
            "liquidation_flow": (
                FeatureNodeType.LIQUIDATION,
                (),
                ("liquidation_buy_volume", "liquidation_sell_volume"),
                {},
                1,
            ),
            "seasonality": (FeatureNodeType.SEASONALITY, (), ("seasonality_score",), {}, 1),
            "sentiment": (FeatureNodeType.SENTIMENT, (), ("sentiment_score",), {}, 1),
            "frozen_ml_feature_vector": (
                FeatureNodeType.FROZEN_ML_FEATURE_SET,
                (),
                ("feature_vector",),
                {},
                1,
            ),
        }
        for name, (
            node_type,
            dependencies,
            required,
            parameters,
            minimum_history,
        ) in specifications.items():
            registry.register(
                FeatureNode(
                    name,
                    node_type,
                    dependencies,
                    parameters,
                    required_inputs=required,
                    input_types={
                        item: (
                            "series"
                            if item.endswith("_history") or item.endswith("_returns")
                            else "mapping"
                            if item in {"cross_sectional_values", "feature_vector"}
                            else "scalar"
                        )
                        for item in required
                    },
                    minimum_history=minimum_history,
                )
            )
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


def _validate_node_inputs(node: FeatureNode, inputs: Mapping[str, AvailableValue], at: str) -> None:
    missing = sorted(set(node.required_inputs) - set(inputs))
    if missing:
        raise FeatureGraphError(
            f"feature node {node.name} is missing required inputs: {', '.join(missing)}"
        )
    for name in node.required_inputs:
        item = inputs[name]
        if item.availability_time > at:
            raise FeatureGraphError(
                f"feature input {name} is not available at the evaluation timestamp"
            )
        expected = str(node.input_types.get(name, "scalar"))
        value = item.value
        if expected == "series":
            if not isinstance(value, list | tuple) or len(value) < node.minimum_history:
                raise FeatureGraphError(
                    f"feature input {name} needs at least {node.minimum_history} historical values"
                )
            try:
                [float(entry) for entry in value]
            except (TypeError, ValueError) as exc:
                raise FeatureGraphError(f"feature input {name} must be numeric series") from exc
        elif expected == "mapping":
            if not isinstance(value, Mapping) or not value:
                raise FeatureGraphError(f"feature input {name} must be a non-empty mapping")
        elif isinstance(value, bool) or not isinstance(value, int | float):
            raise FeatureGraphError(f"feature input {name} must be a numeric scalar")


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
        period = int(node.parameters.get("period", 14))
        changes = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
        gains = [max(0.0, value) for value in changes[-period:]]
        losses = [max(0.0, -value) for value in changes[-period:]]
        average_gain = _wilder_last(gains, period)
        average_loss = _wilder_last(losses, period)
        if average_loss == 0:
            return 100.0 if average_gain else 50.0
        return 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    if name == "macd":
        return _ema(closes, int(node.parameters.get("fast", 12))) - _ema(
            closes, int(node.parameters.get("slow", 26))
        )
    if name == "atr":
        highs, lows = _series(inputs, "high_history"), _series(inputs, "low_history")
        period = int(node.parameters.get("period", 14))
        return _wilder_last(_true_ranges(highs, lows, closes), period)
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
    if name == "adx":
        highs, lows = _series(inputs, "high_history"), _series(inputs, "low_history")
        period = int(node.parameters.get("period", 14))
        true_ranges = _true_ranges(highs, lows, closes)
        plus_dm: list[float] = [0.0]
        minus_dm: list[float] = [0.0]
        for index in range(1, min(len(highs), len(lows))):
            up = highs[index] - highs[index - 1]
            down = lows[index - 1] - lows[index]
            plus_dm.append(up if up > down and up > 0 else 0.0)
            minus_dm.append(down if down > up and down > 0 else 0.0)
        smoothed_tr = _wilder_series(true_ranges, period)
        smoothed_plus = _wilder_series(plus_dm, period)
        smoothed_minus = _wilder_series(minus_dm, period)
        directional: list[float] = []
        for tr, plus, minus in zip(smoothed_tr, smoothed_plus, smoothed_minus, strict=False):
            plus_di = 100.0 * plus / tr if tr > 0 else 0.0
            minus_di = 100.0 * minus / tr if tr > 0 else 0.0
            denominator = plus_di + minus_di
            directional.append(
                100.0 * abs(plus_di - minus_di) / denominator if denominator else 0.0
            )
        return _wilder_last(directional, period)
    if name == "supertrend":
        highs, lows = _series(inputs, "high_history"), _series(inputs, "low_history")
        period = int(node.parameters.get("period", 10))
        multiplier = float(node.parameters.get("multiplier", 3.0))
        atr = _wilder_series(_true_ranges(highs, lows, closes), period)
        length = min(len(highs), len(lows), len(closes), len(atr))
        if not length:
            return 0.0
        upper: list[float] = []
        lower: list[float] = []
        for index in range(length):
            midpoint = (highs[index] + lows[index]) / 2.0
            upper.append(midpoint + multiplier * atr[index])
            lower.append(midpoint - multiplier * atr[index])
        final_upper, final_lower = [upper[0]], [lower[0]]
        direction = 1
        for index in range(1, length):
            final_upper.append(
                upper[index]
                if upper[index] < final_upper[index - 1] or closes[index - 1] > final_upper[index - 1]
                else final_upper[index - 1]
            )
            final_lower.append(
                lower[index]
                if lower[index] > final_lower[index - 1] or closes[index - 1] < final_lower[index - 1]
                else final_lower[index - 1]
            )
            if closes[index] > final_upper[index - 1]:
                direction = 1
            elif closes[index] < final_lower[index - 1]:
                direction = -1
        line = final_lower[-1] if direction > 0 else final_upper[-1]
        return direction * (close - line) / max(close, 1e-12)
    if name == "breakout":
        period = int(node.parameters.get("period", 20))
        previous_high = max(closes[-period - 1 : -1], default=close)
        return close / previous_high - 1.0 if previous_high > 0 else 0.0
    if name == "range_state":
        return (max(closes) - min(closes)) / close if close > 0 else 0.0
    if name == "multi_timeframe":
        return float(_raw(inputs, "higher_timeframe_return"))
    if name == "taker_flow":
        buy = float(_raw(inputs, "taker_buy_volume", 0.0))
        return (2.0 * buy - volume) / volume if volume else 0.0
    if name == "aggressor_flow":
        buy = float(_raw(inputs, "aggressor_buy_volume"))
        sell = float(_raw(inputs, "aggressor_sell_volume"))
        return (buy - sell) / (buy + sell) if buy + sell else 0.0
    if name in {"funding", "open_interest", "sentiment"}:
        source = {
            "funding": "funding_rate",
            "open_interest": "open_interest",
            "sentiment": "sentiment_score",
        }[name]
        return float(_raw(inputs, source))
    if name in {"beta", "correlation"}:
        asset = _series(inputs, "asset_returns")
        benchmark = _series(inputs, "benchmark_returns")
        pairs = tuple(zip(asset, benchmark, strict=False))
        if len(pairs) < 2:
            return 0.0
        asset_mean = statistics.fmean(a for a, _ in pairs)
        benchmark_mean = statistics.fmean(b for _, b in pairs)
        covariance = sum((a - asset_mean) * (b - benchmark_mean) for a, b in pairs)
        variance = sum((b - benchmark_mean) ** 2 for _, b in pairs)
        if name == "correlation":
            asset_variance = sum((a - asset_mean) ** 2 for a, _ in pairs)
            return (
                covariance / math.sqrt(asset_variance * variance)
                if asset_variance and variance
                else 0.0
            )
        return covariance / variance if variance else 0.0
    if name == "spot_perpetual_basis":
        spot, perpetual = (
            float(_raw(inputs, "spot_price", close)),
            float(_raw(inputs, "perpetual_price", close)),
        )
        return perpetual / spot - 1.0 if spot > 0 else 0.0
    if name == "cross_sectional_rank":
        values = sorted(
            float(value) for value in dict(_raw(inputs, "cross_sectional_values")).values()
        )
        current = float(node.parameters.get("current_value", values[-1]))
        return (values.index(current) + 1) / len(values)
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
    if name == "liquidation_flow":
        buy = float(_raw(inputs, "liquidation_buy_volume"))
        sell = float(_raw(inputs, "liquidation_sell_volume"))
        return (buy - sell) / (buy + sell) if buy + sell else 0.0
    if name == "frozen_ml_feature_vector":
        return {
            str(feature): float(value)
            for feature, value in dict(_raw(inputs, "feature_vector")).items()
        }
    raise FeatureGraphError(f"no default evaluator for {name}")


def _ema(values: list[float], period: int) -> float:
    alpha = 2.0 / (period + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result


def _true_ranges(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    length = min(len(highs), len(lows), len(closes))
    if length == 0:
        return []
    ranges = [max(0.0, highs[0] - lows[0])]
    ranges.extend(
        max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        )
        for index in range(1, length)
    )
    return ranges


def _wilder_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    period = max(1, period)
    if len(values) <= period:
        return [statistics.fmean(values)] * len(values)
    result = [statistics.fmean(values[:period])] * period
    current = result[-1]
    for value in values[period:]:
        current = (current * (period - 1) + value) / period
        result.append(current)
    return result


def _wilder_last(values: list[float], period: int) -> float:
    series = _wilder_series(values, period)
    return series[-1] if series else 0.0
