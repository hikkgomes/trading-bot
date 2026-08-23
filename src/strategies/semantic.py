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
