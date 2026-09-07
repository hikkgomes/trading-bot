"""Family-specific strategy contracts that cannot be confused with directional bars."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar

from src.domain._codec import canonical_hash, non_empty, timestamp
from src.domain.forecasts import AlphaForecast
from src.domain.orders import OrderIntent
from src.domain.portfolios import TargetPosition


class SemanticFamily(StrEnum):
    CROSS_SECTIONAL = "cross_sectional"
    RELATIVE_VALUE = "relative_value"
    MICROSTRUCTURE = "microstructure"
    PORTFOLIO_META = "portfolio_meta"
    EXECUTION_POLICY = "execution_policy"


class SemanticEvaluationError(ValueError):
    """A semantic strategy lacks a complete canonical input or output."""


@dataclass(frozen=True)
class PointInTimePanel:
    information_time: str
    instruments: Mapping[str, Mapping[str, float]]
    minimum_score: float = 0.0
    long_count: int = 1
    short_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "information_time", timestamp(self.information_time, field="information_time")
        )
        if len(self.instruments) < 2:
            raise ValueError("cross-sectional panels need at least two instruments")
        if not math.isfinite(float(self.minimum_score)) or self.minimum_score < 0.0:
            raise ValueError("cross-sectional minimum score is invalid")
        if (
            isinstance(self.long_count, bool)
            or isinstance(self.short_count, bool)
            or self.long_count < 1
            or self.short_count < 1
        ):
            raise ValueError("cross-sectional long and short counts must be positive")
        for instrument, values in self.instruments.items():
            if not str(instrument).strip() or not isinstance(values, Mapping) or not values:
                raise ValueError("cross-sectional panel contains an invalid instrument")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in values.values()
            ):
                raise ValueError("cross-sectional panel contains a non-finite value")


@dataclass(frozen=True)
class RankedTargets:
    information_time: str
    scores: Mapping[str, float]
    target_fractions: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "information_time", timestamp(self.information_time, field="information_time")
        )
        if set(self.scores) != set(self.target_fractions):
            raise ValueError("rank scores and target instruments must match")
        if any(
            not math.isfinite(float(value))
            for value in (*self.scores.values(), *self.target_fractions.values())
        ):
            raise ValueError("cross-sectional scores and targets must be finite")
        if abs(sum(self.target_fractions.values())) > 1e-12:
            raise ValueError("cross-sectional targets must be net neutral")
        if sum(abs(value) for value in self.target_fractions.values()) > 1.0 + 1e-12:
            raise ValueError("cross-sectional gross target cannot exceed one")


@dataclass(frozen=True)
class LinkedInstrumentState:
    information_time: str
    legs: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "information_time", timestamp(self.information_time, field="information_time")
        )
        if len(self.legs) < 2:
            raise ValueError("relative-value strategies need linked instruments")
        for instrument, values in self.legs.items():
            if not str(instrument).strip() or not isinstance(values, Mapping):
                raise ValueError("relative-value legs contain an invalid instrument")
            price = values.get("price")
            if price is None or not math.isfinite(float(price)) or float(price) <= 0:
                raise ValueError("relative-value legs need positive prices")
            for name in ("beta", "spread", "causal_spread", "threshold"):
                if values.get(name) is not None and not math.isfinite(float(values[name])):
                    raise ValueError("relative-value leg values must be finite")


@dataclass(frozen=True)
class HedgedTargets:
    information_time: str
    target_notionals: Mapping[str, float]
    hedge_error: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "information_time", timestamp(self.information_time, field="information_time")
        )
        if len(self.target_notionals) < 2:
            raise ValueError("hedged targets need at least two legs")
        if any(
            not str(instrument).strip() or not math.isfinite(float(value))
            for instrument, value in self.target_notionals.items()
        ):
            raise ValueError("hedged targets must contain finite leg notionals")
        if abs(float(self.hedge_error)) > 1e-9:
            raise ValueError("relative-value targets must be hedged at creation")


@dataclass(frozen=True)
class MicrostructureState:
    information_time: str
    sequence: int
    bid_price: float
    ask_price: float
    bid_depth: float
    ask_depth: float
    signed_trade_flow: float
    minimum_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "information_time", timestamp(self.information_time, field="information_time")
        )
        if (
            self.sequence < 0
            or not all(
                math.isfinite(float(value))
                for value in (
                    self.bid_price,
                    self.ask_price,
                    self.bid_depth,
                    self.ask_depth,
                    self.signed_trade_flow,
                    self.minimum_score,
                )
            )
            or self.minimum_score < 0.0
            or self.bid_price <= 0
            or self.ask_price <= self.bid_price
        ):
            raise ValueError("microstructure state has invalid event or book identity")
        if self.bid_depth < 0 or self.ask_depth < 0:
            raise ValueError("order-book depth cannot be negative")


@dataclass(frozen=True)
class MicrostructureForecast:
    information_time: str
    score: float
    expected_direction: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "information_time", timestamp(self.information_time, field="information_time")
        )
        if (
            self.expected_direction not in {-1, 0, 1}
            or not math.isfinite(float(self.score))
            or not -1 <= self.score <= 1
        ):
            raise ValueError("microstructure forecast is outside its contract")


@dataclass(frozen=True)
class ForecastCollection:
    forecasts: tuple[AlphaForecast, ...]

    def __post_init__(self) -> None:
        if not self.forecasts:
            raise ValueError("portfolio meta strategies need forecasts")


@dataclass(frozen=True)
class TargetDeltaBatch:
    targets: tuple[TargetPosition, ...]
    current_quantities: Mapping[str, float]
    decided_at: str

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("execution policies need target deltas")
        object.__setattr__(self, "decided_at", timestamp(self.decided_at, field="decided_at"))
        if any(not math.isfinite(float(value)) for value in self.current_quantities.values()):
            raise ValueError("execution current quantities must be finite")


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class SemanticRegistration(Generic[InputT, OutputT]):
    name: str
    family: SemanticFamily
    input_type: type[InputT]
    output_type: type[OutputT]
    evaluator: Callable[[InputT], OutputT]

    def __post_init__(self) -> None:
        non_empty(self.name, field="semantic strategy name")
        if not callable(self.evaluator):
            raise ValueError("semantic strategy registration needs a concrete evaluator")

    def evaluate(self, value: InputT) -> OutputT:
        if not isinstance(value, self.input_type):
            raise TypeError(f"{self.name} requires {self.input_type.__name__}")
        first = self.evaluator(value)
        second = self.evaluator(value)
        if not isinstance(first, self.output_type):
            raise TypeError(f"{self.name} returned the wrong semantic output")
        if canonical_hash(first) != canonical_hash(second):
            raise RuntimeError(f"{self.name} is non-deterministic")
        return first


class SemanticStrategyRegistry:
    def __init__(self) -> None:
        self._items: dict[str, SemanticRegistration[Any, Any]] = {}

    def register(self, item: SemanticRegistration[Any, Any]) -> None:
        if item.name in self._items:
            raise ValueError(f"duplicate semantic strategy: {item.name}")
        self._items[item.name] = item

    def get(self, name: str) -> SemanticRegistration[Any, Any]:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"semantic strategy has no implementation: {name}") from exc

    def by_family(self, family: SemanticFamily) -> tuple[SemanticRegistration[Any, Any], ...]:
        return tuple(item for item in self._items.values() if item.family is family)


def relative_momentum(panel: PointInTimePanel) -> RankedTargets:
    raw = {
        key: _panel_signal_value(values, preferred=("momentum", "relative_return", "funding_rank"))
        for key, values in panel.instruments.items()
    }
    mean = sum(raw.values()) / len(raw)
    active = {key: value for key, value in raw.items() if abs(value - mean) >= panel.minimum_score}
    ranked = sorted(active.items(), key=lambda item: (item[1], item[0]))
    short = tuple(key for key, _value in ranked[: panel.short_count])
    long = tuple(key for key, _value in ranked[-panel.long_count :])
    long = tuple(key for key in long if key not in short)
    targets = {key: 0.0 for key in raw}
    if long and short:
        targets.update({key: 0.5 / len(long) for key in long})
        targets.update({key: -0.5 / len(short) for key in short})
    ranks = {
        key: float(rank)
        for rank, (key, _) in enumerate(sorted(raw.items(), key=lambda item: (item[1], item[0])))
    }
    return RankedTargets(panel.information_time, ranks, targets)


def funding_adjusted_ranking(panel: PointInTimePanel) -> RankedTargets:
    adjusted = PointInTimePanel(
        panel.information_time,
        {
            key: {
                "momentum": _panel_signal_value(
                    values, preferred=("funding_rank", "relative_return", "momentum")
                )
                - float(values.get("funding", 0.0))
            }
            for key, values in panel.instruments.items()
        },
        minimum_score=panel.minimum_score,
        long_count=panel.long_count,
        short_count=panel.short_count,
    )
    return relative_momentum(adjusted)


def _panel_signal_value(values: Mapping[str, float], *, preferred: tuple[str, ...]) -> float:
    for name in preferred:
        if values.get(name) is not None:
            return float(values[name])
    raise SemanticEvaluationError("cross-sectional panel has no ranking value")


def spot_perpetual_basis(state: LinkedInstrumentState) -> HedgedTargets:
    spot = next((key for key in sorted(state.legs) if "spot" in key.lower()), None)
    perpetual = next((key for key in sorted(state.legs) if "perp" in key.lower()), None)
    if spot is None or perpetual is None:
        raise ValueError("basis strategy requires linked spot and perpetual legs")
    spot_price = float(state.legs[spot]["price"])
    perpetual_price = float(state.legs[perpetual]["price"])
    direction = 1.0 if perpetual_price < spot_price else -1.0
    targets = {spot: direction, perpetual: -direction}
    return HedgedTargets(state.information_time, targets, sum(targets.values()))


def beta_neutral_spread(state: LinkedInstrumentState) -> HedgedTargets:
    first, second = sorted(state.legs)[:2]
    beta = float(state.legs[first].get("beta", 1.0))
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("spread hedge beta must be positive")
    first_price = float(state.legs[first]["price"])
    second_price = float(state.legs[second]["price"])
    spread = state.legs[first].get("causal_spread", state.legs[first].get("spread"))
    if spread is None:
        spread = first_price / second_price - 1.0
    spread = float(spread)
    threshold = abs(float(state.legs[first].get("threshold", 0.0)))
    if abs(spread) <= threshold:
        first_target = 0.0
        second_target = 0.0
    else:
        # A rich first leg is sold and the second leg is bought. The opposite
        # trade is used when the first leg is cheap. The signed hedge ratio is
        # part of the output, so the two legs cannot be collapsed into one
        # directional forecast.
        direction = -1.0 if spread > 0.0 else 1.0
        first_target = direction
        second_target = -direction * beta
    normaliser = abs(first_target) + abs(second_target)
    targets = (
        {first: first_target / normaliser, second: second_target / normaliser}
        if normaliser
        else {first: 0.0, second: 0.0}
    )
    hedge_error = targets[first] * beta + targets[second]
    return HedgedTargets(state.information_time, targets, hedge_error)


def depth_imbalance(state: MicrostructureState) -> MicrostructureForecast:
    total = state.bid_depth + state.ask_depth
    score = (state.bid_depth - state.ask_depth) / total if total else 0.0
    if abs(score) < state.minimum_score:
        score = 0.0
    direction = 1 if score > 0 else -1 if score < 0 else 0
    return MicrostructureForecast(state.information_time, score, direction)


def microprice_displacement(state: MicrostructureState) -> MicrostructureForecast:
    total = state.bid_depth + state.ask_depth
    microprice = (
        (state.ask_price * state.bid_depth + state.bid_price * state.ask_depth) / total
        if total
        else (state.bid_price + state.ask_price) / 2
    )
    mid = (state.bid_price + state.ask_price) / 2
    score = max(-1.0, min(1.0, (microprice - mid) / (state.ask_price - state.bid_price)))
    if abs(score) < state.minimum_score:
        score = 0.0
    direction = 1 if score > 0 else -1 if score < 0 else 0
    return MicrostructureForecast(state.information_time, score, direction)


def correlation_aware_ensemble(value: ForecastCollection) -> tuple[AlphaForecast, ...]:
    from src.portfolio.aggregation import aggregate_forecasts

    return aggregate_forecasts(value.forecasts)


def market_execution(value: TargetDeltaBatch) -> tuple[OrderIntent, ...]:
    from src.execution.order_planner import plan_orders

    return plan_orders(
        value.targets,
        current_quantities=value.current_quantities,
        decided_at=value.decided_at,
    )


SEMANTIC_DEFAULTS = {
    "cross_sectional": "relative_momentum",
    "relative_value": "spot_perpetual_basis",
    "microstructure": "bid_ask_depth_imbalance",
    "ensemble": "correlation_aware_ensemble",
}


def semantic_strategy_name(source_type: str, declared: object = None) -> str:
    name = str(declared or SEMANTIC_DEFAULTS.get(source_type) or "").strip()
    if not name:
        raise SemanticEvaluationError("semantic strategy identity is missing")
    return name


def semantic_input_from_features(name: str, features: Mapping[str, Any]) -> Any:
    """Build the exact typed semantic input from persisted feature inputs."""

    registration = SEMANTIC_STRATEGIES.get(name)
    raw = features.get("semantic_input")
    if isinstance(raw, registration.input_type):
        return raw
    if isinstance(raw, Mapping):
        return _semantic_mapping_input(registration, raw)
    if registration.input_type is PointInTimePanel:
        return _panel_from_features(features)
    if registration.input_type is LinkedInstrumentState:
        return _linked_state_from_features(features)
    if registration.input_type is MicrostructureState:
        return _microstructure_from_features(features)
    if registration.input_type is ForecastCollection:
        return _forecast_collection_from_features(features)
    raise SemanticEvaluationError(f"{name} has no typed canonical input")


def semantic_contract_features(
    name: str, model: Mapping[str, Any], features: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the immutable family contract before semantic evaluation."""

    result = dict(features)
    if name == "beta_neutral_spreads" and model.get("paired_legs") is not None:
        result["linked_instruments"] = _relative_contract_features(model, result)
    if name in {"relative_momentum", "funding_adjusted_ranking"}:
        result["cross_sectional_values"] = _panel_contract_features(model, result)
        panel_contract = model.get("panel_contract")
        parameters = model.get("parameters")
        threshold = (
            panel_contract.get("minimum_score") if isinstance(panel_contract, Mapping) else None
        )
        if threshold is None and isinstance(parameters, Mapping):
            threshold = parameters.get("threshold")
        result["panel_threshold"] = abs(_finite_float(threshold or 0.0, field="panel_threshold"))
        result["panel_long_count"] = _panel_count(
            panel_contract, parameters, "long_count", default=1
        )
        result["panel_short_count"] = _panel_count(
            panel_contract, parameters, "short_count", default=1
        )
    if name in {"bid_ask_depth_imbalance", "microprice_displacement"}:
        _require_microstructure_sequence(result)
        event_contract = model.get("event_contract")
        parameters = model.get("parameters")
        threshold = (
            event_contract.get("minimum_score") if isinstance(event_contract, Mapping) else None
        )
        if threshold is None and isinstance(parameters, Mapping):
            threshold = parameters.get("threshold")
        result["minimum_score"] = abs(_finite_float(threshold or 0.0, field="minimum_score"))
    return result


def _relative_contract_features(
    model: Mapping[str, Any], features: Mapping[str, Any]
) -> dict[str, Any]:
    raw_legs = model.get("paired_legs")
    linked = features.get("linked_instruments")
    if not isinstance(raw_legs, list | tuple) or len(raw_legs) != 2:
        raise SemanticEvaluationError("relative-value contract needs exactly two paired legs")
    if not isinstance(linked, Mapping):
        raise SemanticEvaluationError("relative-value input has no linked instrument panel")
    leg_ids = tuple(
        str(item.get("instrument_id") or "") for item in raw_legs if isinstance(item, Mapping)
    )
    if len(leg_ids) != 2 or any(not item for item in leg_ids) or not set(leg_ids).issubset(linked):
        raise SemanticEvaluationError("linked instruments differ from the paired-leg contract")
    hedge_ratio = _finite_float(model.get("hedge_ratio"), field="hedge_ratio")
    if hedge_ratio <= 0:
        raise SemanticEvaluationError("relative-value hedge ratio must be positive")
    first = sorted(leg_ids)[0]
    threshold = _contract_threshold(model)
    first_price = float(linked[sorted(leg_ids)[0]]["price"])
    second_price = float(linked[sorted(leg_ids)[1]]["price"])
    derived_spread = first_price / second_price - 1.0
    explicit_spread = next(
        (
            value
            for key in ("causal_spread", "spread", "basis")
            if (value := features.get(key)) is not None
        ),
        derived_spread,
    )
    return {
        str(key): {
            **dict(value),
            **(
                {
                    "beta": hedge_ratio,
                    "causal_spread": float(explicit_spread),
                    "threshold": threshold,
                }
                if str(key) == first
                else {}
            ),
        }
        for key, value in linked.items()
        if str(key) in leg_ids and isinstance(value, Mapping)
    }


def _contract_threshold(model: Mapping[str, Any]) -> float:
    parameters = model.get("parameters")
    raw = parameters.get("threshold") if isinstance(parameters, Mapping) else None
    if raw is None:
        raw = model.get("threshold", 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SemanticEvaluationError("relative-value threshold is invalid") from exc
    if not math.isfinite(value):
        raise SemanticEvaluationError("relative-value threshold is invalid")
    return value


def _panel_contract_features(
    model: Mapping[str, Any], features: Mapping[str, Any]
) -> dict[str, Any]:
    scope = model.get("instrument_scope")
    panel = features.get("cross_sectional_values")
    if isinstance(scope, list | tuple) and isinstance(panel, Mapping):
        declared = {str(value) for value in scope}
        observed = {str(key) for key in panel if not str(key).startswith("__")}
        if not declared.issubset(observed):
            raise SemanticEvaluationError("cross-sectional panel is outside its instrument scope")
        return {
            str(key): value
            for key, value in panel.items()
            if str(key).startswith("__") or str(key) in declared
        }
    if not isinstance(panel, Mapping):
        raise SemanticEvaluationError("cross-sectional input has no point-in-time panel")
    return {str(key): value for key, value in panel.items()}


def _require_microstructure_sequence(features: Mapping[str, Any]) -> None:
    if "sequence" not in features:
        raise SemanticEvaluationError("microstructure contract has no event sequence")


def semantic_forecast_from_output(
    output: Any,
    *,
    instrument_id: str,
    position_limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map one shared semantic output to the forecast contract."""

    signed = semantic_signal(output, instrument_id=instrument_id)
    limits = position_limits if isinstance(position_limits, Mapping) else {}
    maximum = _bounded_float(limits.get("maximum_position", limits.get("maximum_fraction", 0.1)))
    target_volatility = _bounded_float(limits.get("target_volatility", 0.1))
    return_scale = _finite_float(limits.get("return_scale", 0.01), field="return_scale")
    direction = "long" if signed > 0.0 else "short" if signed < 0.0 else "flat"
    return {
        "direction": direction,
        "score": abs(signed),
        "expected_return": signed * return_scale,
        "confidence": _bounded_float(limits.get("confidence", 0.7)),
        "target_volatility": target_volatility,
        "maximum_position": maximum if direction != "flat" else 0.0,
        "semantic_signal": signed,
        "semantic_output_hash": canonical_hash(output),
    }


def semantic_signal(output: Any, *, instrument_id: str) -> float:
    if isinstance(output, RankedTargets):
        return _target_value(output.target_fractions, instrument_id)
    if isinstance(output, HedgedTargets):
        return _target_value(output.target_notionals, instrument_id)
    if isinstance(output, MicrostructureForecast):
        return float(output.score)
    if isinstance(output, tuple) and all(isinstance(item, AlphaForecast) for item in output):
        matching = tuple(item for item in output if item.instrument_id == instrument_id)
        if len(matching) != 1:
            raise SemanticEvaluationError(
                "ensemble output must contain exactly one forecast for the instrument"
            )
        return matching[0].signed_strength
    raise SemanticEvaluationError("semantic output cannot produce an instrument forecast")


def _semantic_mapping_input(
    registration: SemanticRegistration[Any, Any], raw: Mapping[str, Any]
) -> Any:
    if registration.input_type is ForecastCollection:
        forecasts = raw.get("forecasts")
        if not isinstance(forecasts, list | tuple):
            raise SemanticEvaluationError("ensemble input has no forecast collection")
        return ForecastCollection(
            tuple(
                item if isinstance(item, AlphaForecast) else AlphaForecast(**dict(item))
                for item in forecasts
            )
        )
    try:
        return registration.input_type(**dict(raw))
    except (TypeError, ValueError) as exc:
        raise SemanticEvaluationError("semantic input does not match its typed contract") from exc


def _panel_from_features(features: Mapping[str, Any]) -> PointInTimePanel:
    raw = features.get("cross_sectional_values")
    if not isinstance(raw, Mapping):
        raise SemanticEvaluationError("cross-sectional input has no point-in-time panel")
    instruments: dict[str, Mapping[str, float]] = {}
    for instrument, value in raw.items():
        if str(instrument).startswith("__"):
            continue
        if isinstance(value, Mapping):
            instruments[str(instrument)] = {
                str(key): _finite_float(item, field=f"panel.{instrument}.{key}")
                for key, item in value.items()
            }
        else:
            instruments[str(instrument)] = {
                "momentum": _finite_float(value, field=f"panel.{instrument}.momentum"),
                "funding": _finite_float(features.get("funding_rate", 0.0), field="funding_rate"),
            }
    information_time = str(features.get("information_time") or "")
    if len(instruments) < 2 or not information_time:
        raise SemanticEvaluationError("cross-sectional input needs two instruments and a timestamp")
    minimum_score = _finite_float(features.get("panel_threshold", 0.0), field="panel_threshold")
    if minimum_score < 0.0:
        raise SemanticEvaluationError("panel_threshold must be non-negative")
    return PointInTimePanel(
        information_time,
        instruments,
        minimum_score=minimum_score,
        long_count=_panel_count_value(features.get("panel_long_count"), field="panel_long_count"),
        short_count=_panel_count_value(
            features.get("panel_short_count"), field="panel_short_count"
        ),
    )


def _panel_count(
    panel_contract: object,
    parameters: object,
    name: str,
    *,
    default: int,
) -> int:
    value = None
    if isinstance(panel_contract, Mapping):
        value = panel_contract.get(name)
    if value is None and isinstance(parameters, Mapping):
        value = parameters.get(name)
    return _panel_count_value(default if value is None else value, field=name)


def _panel_count_value(value: object, *, field: str) -> int:
    if value is None:
        return 1
    if isinstance(value, bool):
        raise SemanticEvaluationError(f"{field} must be a positive integer")
    numeric = _finite_float(value, field=field)
    result = int(numeric)
    if result < 1 or numeric != result:
        raise SemanticEvaluationError(f"{field} must be a positive integer")
    return result


def _linked_state_from_features(features: Mapping[str, Any]) -> LinkedInstrumentState:
    raw = features.get("linked_instruments")
    if isinstance(raw, Mapping):
        legs = {str(key): dict(value) for key, value in raw.items() if isinstance(value, Mapping)}
    else:
        legs = _linked_legs_from_frame(features.get("market_frame"))
        if len(legs) < 2:
            legs = {
                "spot": {"price": _finite_float(features.get("spot_price"), field="spot_price")},
                "perpetual": {
                    "price": _finite_float(
                        features.get("perpetual_price"), field="perpetual_price"
                    ),
                    "beta": _finite_float(features.get("beta", 1.0), field="beta"),
                },
            }
    information_time = str(features.get("information_time") or "")
    if len(legs) < 2 or not information_time:
        raise SemanticEvaluationError("relative-value input needs linked legs and a timestamp")
    return LinkedInstrumentState(information_time, legs)


def _linked_legs_from_frame(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, list | tuple):
        return {}
    legs: dict[str, dict[str, float]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            continue
        instrument_id = str(row.get("instrument_id") or row.get("symbol") or "").strip()
        price = row.get("close", row.get("price"))
        if not instrument_id or price is None:
            continue
        try:
            parsed = float(price)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            legs[instrument_id] = {"price": parsed, "beta": 1.0}
    return legs


def _microstructure_from_features(features: Mapping[str, Any]) -> MicrostructureState:
    information_time = str(features.get("information_time") or "")
    if not information_time:
        raise SemanticEvaluationError("microstructure input has no information timestamp")
    if "sequence" not in features:
        raise SemanticEvaluationError("microstructure input has no exchange event sequence")
    raw_sequence = features["sequence"]
    if isinstance(raw_sequence, bool):
        raise SemanticEvaluationError("microstructure event sequence is invalid")
    try:
        sequence = int(raw_sequence)
    except (TypeError, ValueError) as exc:
        raise SemanticEvaluationError("microstructure event sequence is invalid") from exc
    if sequence < 0 or float(raw_sequence) != sequence:
        raise SemanticEvaluationError("microstructure event sequence is invalid")
    return MicrostructureState(
        information_time=information_time,
        sequence=sequence,
        bid_price=_finite_float(features.get("bid_price"), field="bid_price"),
        ask_price=_finite_float(features.get("ask_price"), field="ask_price"),
        bid_depth=_finite_float(features.get("bid_depth"), field="bid_depth"),
        ask_depth=_finite_float(features.get("ask_depth"), field="ask_depth"),
        signed_trade_flow=_finite_float(
            features.get("signed_trade_flow", 0.0), field="signed_trade_flow"
        ),
        minimum_score=_finite_float(features.get("minimum_score", 0.0), field="minimum_score"),
    )


def _forecast_collection_from_features(features: Mapping[str, Any]) -> ForecastCollection:
    raw = features.get("forecasts") or features.get("semantic_forecasts")
    if not isinstance(raw, list | tuple):
        raise SemanticEvaluationError("ensemble input has no forecast collection")
    return ForecastCollection(
        tuple(
            item if isinstance(item, AlphaForecast) else AlphaForecast(**dict(item)) for item in raw
        )
    )


def _target_value(values: Mapping[str, float], instrument_id: str) -> float:
    key = instrument_id
    if key not in values:
        market_key = "perpetual" if ":futures:" in instrument_id.casefold() else "spot"
        matches = tuple(name for name in values if market_key in str(name).casefold())
        if not matches:
            symbol = _instrument_symbol(instrument_id)
            matches = tuple(
                name for name in values if _compact_identifier(name) == _compact_identifier(symbol)
            )
        if len(matches) == 1:
            key = matches[0]
    if key not in values:
        raise SemanticEvaluationError(f"semantic output has no target for {instrument_id}")
    value = float(values[key])
    if not math.isfinite(value):
        raise SemanticEvaluationError("semantic target is not finite")
    return max(-1.0, min(1.0, value))


def _instrument_symbol(instrument_id: str) -> str:
    parts = instrument_id.split(":")
    if len(parts) >= 3:
        return parts[2]
    return instrument_id.split("/", 1)[0]


def _compact_identifier(value: object) -> str:
    return "".join(character for character in str(value).upper() if character.isalnum())


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SemanticEvaluationError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise SemanticEvaluationError(f"{field} must be finite")
    return result


def _bounded_float(value: Any) -> float:
    return max(0.0, min(1.0, _finite_float(value, field="semantic limit")))


SEMANTIC_STRATEGIES = SemanticStrategyRegistry()
for registration in (
    SemanticRegistration(
        "relative_momentum",
        SemanticFamily.CROSS_SECTIONAL,
        PointInTimePanel,
        RankedTargets,
        relative_momentum,
    ),
    SemanticRegistration(
        "funding_adjusted_ranking",
        SemanticFamily.CROSS_SECTIONAL,
        PointInTimePanel,
        RankedTargets,
        funding_adjusted_ranking,
    ),
    SemanticRegistration(
        "spot_perpetual_basis",
        SemanticFamily.RELATIVE_VALUE,
        LinkedInstrumentState,
        HedgedTargets,
        spot_perpetual_basis,
    ),
    SemanticRegistration(
        "beta_neutral_spreads",
        SemanticFamily.RELATIVE_VALUE,
        LinkedInstrumentState,
        HedgedTargets,
        beta_neutral_spread,
    ),
    SemanticRegistration(
        "bid_ask_depth_imbalance",
        SemanticFamily.MICROSTRUCTURE,
        MicrostructureState,
        MicrostructureForecast,
        depth_imbalance,
    ),
    SemanticRegistration(
        "microprice_displacement",
        SemanticFamily.MICROSTRUCTURE,
        MicrostructureState,
        MicrostructureForecast,
        microprice_displacement,
    ),
    SemanticRegistration(
        "correlation_aware_ensemble",
        SemanticFamily.PORTFOLIO_META,
        ForecastCollection,
        tuple,
        correlation_aware_ensemble,
    ),
    SemanticRegistration(
        "market_execution",
        SemanticFamily.EXECUTION_POLICY,
        TargetDeltaBatch,
        tuple,
        market_execution,
    ),
):
    SEMANTIC_STRATEGIES.register(registration)
