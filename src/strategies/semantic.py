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

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "information_time", timestamp(self.information_time, field="information_time")
        )
        if len(self.instruments) < 2:
            raise ValueError("cross-sectional panels need at least two instruments")


@dataclass(frozen=True)
class RankedTargets:
    information_time: str
    scores: Mapping[str, float]
    target_fractions: Mapping[str, float]

    def __post_init__(self) -> None:
        if set(self.scores) != set(self.target_fractions):
            raise ValueError("rank scores and target instruments must match")
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


@dataclass(frozen=True)
class HedgedTargets:
    information_time: str
    target_notionals: Mapping[str, float]
    hedge_error: float

    def __post_init__(self) -> None:
        if len(self.target_notionals) < 2:
            raise ValueError("hedged targets need at least two legs")
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "information_time", timestamp(self.information_time, field="information_time")
        )
        if self.sequence < 0 or self.bid_price <= 0 or self.ask_price <= self.bid_price:
            raise ValueError("microstructure state has invalid event or book identity")
        if self.bid_depth < 0 or self.ask_depth < 0:
            raise ValueError("order-book depth cannot be negative")


@dataclass(frozen=True)
class MicrostructureForecast:
    information_time: str
    score: float
    expected_direction: int

    def __post_init__(self) -> None:
        if self.expected_direction not in {-1, 0, 1} or not -1 <= self.score <= 1:
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
    raw = {key: float(values["momentum"]) for key, values in panel.instruments.items()}
    mean = sum(raw.values()) / len(raw)
    centred = {key: value - mean for key, value in raw.items()}
    gross = sum(abs(value) for value in centred.values())
    targets = {key: value / gross if gross else 0.0 for key, value in centred.items()}
    ranks = {
        key: float(rank)
        for rank, (key, _) in enumerate(sorted(raw.items(), key=lambda item: (item[1], item[0])))
    }
    return RankedTargets(panel.information_time, ranks, targets)


def funding_adjusted_ranking(panel: PointInTimePanel) -> RankedTargets:
    adjusted = PointInTimePanel(
        panel.information_time,
        {
            key: {"momentum": float(values["momentum"]) - float(values["funding"])}
            for key, values in panel.instruments.items()
        },
    )
    return relative_momentum(adjusted)


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
    first_target = 1.0
    second_target = -beta
    normaliser = abs(first_target) + abs(second_target)
    targets = {first: first_target / normaliser, second: second_target / normaliser}
    hedge_error = targets[first] * beta + targets[second]
    return HedgedTargets(state.information_time, targets, hedge_error)


def depth_imbalance(state: MicrostructureState) -> MicrostructureForecast:
    total = state.bid_depth + state.ask_depth
    score = (state.bid_depth - state.ask_depth) / total if total else 0.0
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
    return PointInTimePanel(information_time, instruments)


def _linked_state_from_features(features: Mapping[str, Any]) -> LinkedInstrumentState:
    raw = features.get("linked_instruments")
    if isinstance(raw, Mapping):
        legs = {str(key): dict(value) for key, value in raw.items() if isinstance(value, Mapping)}
    else:
        legs = {
            "spot": {"price": _finite_float(features.get("spot_price"), field="spot_price")},
            "perpetual": {
                "price": _finite_float(features.get("perpetual_price"), field="perpetual_price"),
                "beta": _finite_float(features.get("beta", 1.0), field="beta"),
            },
        }
    information_time = str(features.get("information_time") or "")
    if len(legs) < 2 or not information_time:
        raise SemanticEvaluationError("relative-value input needs linked legs and a timestamp")
    return LinkedInstrumentState(information_time, legs)


def _microstructure_from_features(features: Mapping[str, Any]) -> MicrostructureState:
    information_time = str(features.get("information_time") or "")
    if not information_time:
        raise SemanticEvaluationError("microstructure input has no information timestamp")
    return MicrostructureState(
        information_time=information_time,
        sequence=int(features.get("sequence", 0)),
        bid_price=_finite_float(features.get("bid_price"), field="bid_price"),
        ask_price=_finite_float(features.get("ask_price"), field="ask_price"),
        bid_depth=_finite_float(features.get("bid_depth"), field="bid_depth"),
        ask_depth=_finite_float(features.get("ask_depth"), field="ask_depth"),
        signed_trade_flow=_finite_float(
            features.get("signed_trade_flow", 0.0), field="signed_trade_flow"
        ),
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
        if len(matches) == 1:
            key = matches[0]
    if key not in values:
        raise SemanticEvaluationError(f"semantic output has no target for {instrument_id}")
    value = float(values[key])
    if not math.isfinite(value):
        raise SemanticEvaluationError("semantic target is not finite")
    return max(-1.0, min(1.0, value))


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
