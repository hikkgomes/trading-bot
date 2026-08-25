"""Typed implementations for advanced alpha, execution, and market-making families."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass

from src.domain._codec import canonical_hash, timestamp


@dataclass(frozen=True)
class PairSeries:
    information_times: tuple[str, ...]
    first_prices: tuple[float, ...]
    second_prices: tuple[float, ...]
    first_funding: tuple[float, ...] = ()
    second_funding: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        size = len(self.information_times)
        if size < 20 or len(self.first_prices) != size or len(self.second_prices) != size:
            raise ValueError("pairs research needs at least 20 synchronised observations")
        times = tuple(
            timestamp(value, field="pair information time") for value in self.information_times
        )
        if tuple(sorted(times)) != times or len(set(times)) != size:
            raise ValueError("pair information times must be unique and chronological")
        if any(
            value <= 0 or not math.isfinite(value)
            for value in (*self.first_prices, *self.second_prices)
        ):
            raise ValueError("pair prices must be finite and positive")
        object.__setattr__(self, "information_times", times)


@dataclass(frozen=True)
class PairSignal:
    hedge_ratio: float
    spread_zscore: float
    direction: int
    stationary: bool
    structural_break: bool
    cost_adjusted_edge: float


@dataclass(frozen=True)
class PairDiscovery:
    first_symbol: str
    second_symbol: str
    correlation: float
    stationary: bool
    hedge_ratio: float
    excursion_count: int


def discover_pairs(
    prices: Mapping[str, tuple[float, ...]], *, minimum_correlation: float = 0.7
) -> tuple[PairDiscovery, ...]:
    """Pre-screen synchronised pairs without consuming protected outcomes."""

    symbols = tuple(sorted(prices))
    discovered: list[PairDiscovery] = []
    for left_index, first in enumerate(symbols):
        for second in symbols[left_index + 1 :]:
            first_prices, second_prices = prices[first], prices[second]
            if len(first_prices) != len(second_prices) or len(first_prices) < 20:
                continue
            first_returns = _returns(first_prices)
            second_returns = _returns(second_prices)
            correlation = _correlation(first_returns, second_returns)
            if correlation < minimum_correlation:
                continue
            times = tuple(f"2020-01-{index + 1:02d}T00:00:00+00:00" for index in range(20))
            sample = PairSeries(times, first_prices[-20:], second_prices[-20:])
            signal = evaluate_pair(sample, entry_zscore=1.0, cost_bps=0.0)
            spreads = _pair_spreads(sample, signal.hedge_ratio)
            deviation = statistics.pstdev(spreads)
            centre = statistics.fmean(spreads)
            excursions = (
                sum(abs(value - centre) >= deviation for value in spreads) if deviation else 0
            )
            discovered.append(
                PairDiscovery(
                    first,
                    second,
                    correlation,
                    signal.stationary,
                    signal.hedge_ratio,
                    excursions,
                )
            )
    return tuple(discovered)


def evaluate_pair(
    series: PairSeries,
    *,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    cost_bps: float = 4.0,
) -> PairSignal:
    x = [math.log(value) for value in series.second_prices]
    y = [math.log(value) for value in series.first_prices]
    x_mean, y_mean = statistics.fmean(x), statistics.fmean(y)
    variance = sum((value - x_mean) ** 2 for value in x)
    hedge = (
        sum((left - x_mean) * (right - y_mean) for left, right in zip(x, y, strict=True)) / variance
        if variance
        else 1.0
    )
    spreads = [right - hedge * left for left, right in zip(x, y, strict=True)]
    window = spreads[-min(60, len(spreads)) :]
    deviation = statistics.pstdev(window)
    zscore = (window[-1] - statistics.fmean(window)) / deviation if deviation else 0.0
    changes = [spreads[index] - spreads[index - 1] for index in range(1, len(spreads))]
    lagged = spreads[:-1]
    stationary = _correlation(lagged, changes) < -0.05
    recent = spreads[-max(5, len(spreads) // 4) :]
    earlier = spreads[: -len(recent)]
    structural_break = bool(earlier) and statistics.pstdev(recent) > 3 * max(
        statistics.pstdev(earlier), 1e-12
    )
    direction = (
        0 if abs(zscore) < exit_zscore or abs(zscore) < entry_zscore else (-1 if zscore > 0 else 1)
    )
    funding = sum(series.first_funding) - hedge * sum(series.second_funding)
    edge = max(0.0, abs(zscore) - entry_zscore) * deviation - cost_bps / 10_000 - abs(funding)
    if not stationary or structural_break or edge <= 0:
        direction = 0
    return PairSignal(hedge, zscore, direction, stationary, structural_break, edge)


def _pair_spreads(series: PairSeries, hedge_ratio: float) -> list[float]:
    return [
        math.log(first) - hedge_ratio * math.log(second)
        for first, second in zip(series.first_prices, series.second_prices, strict=True)
    ]


def _returns(prices: tuple[float, ...]) -> list[float]:
    return [
        math.log(right / left)
        for left, right in zip(prices, prices[1:], strict=False)
        if left > 0 and right > 0
    ]


@dataclass(frozen=True)
class CalendarObservation:
    information_time: str
    return_value: float
    funding_interval: bool = False
    contract_expiry: bool = False


@dataclass(frozen=True)
class SeasonalityModel:
    timezone: str
    declared_buckets: tuple[str, ...]

    def evaluate(
        self, history: tuple[CalendarObservation, ...], current: CalendarObservation
    ) -> float:
        if self.timezone != "UTC" or not self.declared_buckets:
            raise ValueError("seasonality needs a predeclared UTC calendar")
        bucket = _calendar_bucket(current, self.declared_buckets)
        values = [
            item.return_value
            for item in history
            if _calendar_bucket(item, self.declared_buckets) == bucket
            and item.information_time < current.information_time
        ]
        return statistics.fmean(values) if values else 0.0


def _calendar_bucket(value: CalendarObservation, declared: tuple[str, ...]) -> str:
    parsed = __import__("datetime").datetime.fromisoformat(
        timestamp(value.information_time, field="information_time")
    )
    parts = []
    if "hour_of_day" in declared:
        parts.append(f"hour:{parsed.hour}")
    if "day_of_week" in declared:
        parts.append(f"weekday:{parsed.weekday()}")
    if "weekend" in declared:
        parts.append(f"weekend:{parsed.weekday() >= 5}")
    if "funding_interval" in declared:
        parts.append(f"funding:{value.funding_interval}")
    if "month_end" in declared:
        parts.append(
            f"month_end:{(parsed + __import__('datetime').timedelta(days=1)).month != parsed.month}"
        )
    if "quarter_end" in declared:
        parts.append(
            f"quarter_end:{parsed.month in {3, 6, 9, 12} and (parsed + __import__('datetime').timedelta(days=1)).month != parsed.month}"
        )
    if "contract_expiry" in declared:
        parts.append(f"expiry:{value.contract_expiry}")
    return "|".join(parts)


@dataclass(frozen=True)
class VolatilityForecast:
    realised_volatility: float
    forecast_volatility: float
    regime: str
    expansion_score: float
    target_scale: float


@dataclass(frozen=True)
class OptionsCapabilityPolicy:
    enabled: bool = False
    required_surface_snapshot: bool = True

    def authorise(self, *, surface_snapshot_id: str | None) -> bool:
        if not self.enabled:
            return False
        return not self.required_surface_snapshot or bool(
            surface_snapshot_id
            and surface_snapshot_id.startswith("sha256:")
            and len(surface_snapshot_id) == 71
        )


def forecast_volatility(
    returns: tuple[float, ...], *, funding_rate: float = 0.0, target_volatility: float = 0.15
) -> VolatilityForecast:
    if len(returns) < 20:
        raise ValueError("volatility forecasts need at least 20 chronological returns")
    variance = 0.0
    for value in returns:
        variance = 0.94 * variance + 0.06 * value * value
    realised = statistics.pstdev(returns[-20:]) * math.sqrt(365 * 24)
    forecast = math.sqrt(max(0.0, variance)) * math.sqrt(365 * 24) + abs(funding_rate)
    ratio = forecast / max(realised, 1e-12)
    regime = "expansion" if ratio > 1.2 else "compression" if ratio < 0.8 else "stable"
    return VolatilityForecast(
        realised, forecast, regime, ratio - 1.0, min(1.0, target_volatility / max(forecast, 1e-12))
    )


@dataclass(frozen=True)
class PatternForecast:
    change_point_score: float
    motif_score: float
    shapelet_distance: float
    volatility_state: str
    flow_state: str
    direction: int


def recognise_statistical_pattern(
    returns: tuple[float, ...],
    flows: tuple[float, ...],
    *,
    declared_shapelet: tuple[float, ...] = (-1.0, 0.0, 1.0),
) -> PatternForecast:
    if len(returns) < 20 or len(flows) != len(returns) or not declared_shapelet:
        raise ValueError("pattern recognition requires aligned chronological inputs")
    split = len(returns) // 2
    earlier, recent = returns[:split], returns[split:]
    pooled = max(statistics.pstdev(returns), 1e-12)
    change = abs(statistics.fmean(recent) - statistics.fmean(earlier)) / pooled
    width = min(len(declared_shapelet), len(recent))
    observed = recent[-width:]
    scale = max(statistics.pstdev(observed), 1e-12)
    normalised = tuple((value - statistics.fmean(observed)) / scale for value in observed)
    target = declared_shapelet[-width:]
    distance = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(normalised, target, strict=True)) / width
    )
    motif = 1.0 / (1.0 + distance)
    recent_volatility = statistics.pstdev(recent)
    earlier_volatility = max(statistics.pstdev(earlier), 1e-12)
    volatility_state = "expansion" if recent_volatility > 1.2 * earlier_volatility else "stable"
    flow_mean = statistics.fmean(flows[-width:])
    flow_state = "buying" if flow_mean > 0 else "selling" if flow_mean < 0 else "balanced"
    direction = 1 if motif >= 0.5 and flow_mean > 0 else -1 if motif >= 0.5 and flow_mean < 0 else 0
    return PatternForecast(change, motif, distance, volatility_state, flow_state, direction)


@dataclass(frozen=True)
class PointInTimeSentiment:
    publication_time: str
    ingestion_time: str
    source_identity: str
    deduplication_identity: str
    revision: int
    model_version: str
    source_artefact_hash: str
    score: float

    def __post_init__(self) -> None:
        published = timestamp(self.publication_time, field="publication_time")
        ingested = timestamp(self.ingestion_time, field="ingestion_time")
        if published > ingested or self.revision < 0 or not -1 <= self.score <= 1:
            raise ValueError("sentiment record is not point-in-time valid")
        if (
            not self.source_artefact_hash.startswith("sha256:")
            or len(self.source_artefact_hash) != 71
        ):
            raise ValueError("sentiment source artefact must be immutable")


@dataclass(frozen=True)
class MicrostructureState:
    information_time: str
    aggressor_flow: float
    spread_change_bps: float
    depth_change: float
    cancel_add_pressure: float
    liquidation_notional: float
    visible_depth: float
    recent_return: float


@dataclass(frozen=True)
class ScalpingForecast:
    direction: int
    continuation_score: float
    reversal_score: float
    fill_probability: float
    adverse_selection_probability: float
    event_label: str


def microstructure_scalping_signal(state: MicrostructureState) -> ScalpingForecast:
    timestamp(state.information_time, field="information_time")
    if state.visible_depth <= 0 or state.liquidation_notional < 0:
        raise ValueError("microstructure state has invalid depth or liquidation values")
    vacuum = max(0.0, -state.depth_change) + max(0.0, state.spread_change_bps / 10.0)
    forced_flow = state.liquidation_notional / state.visible_depth
    continuation = math.tanh(
        state.aggressor_flow + state.cancel_add_pressure + forced_flow + vacuum
    )
    reversal = math.tanh(-state.recent_return * (1.0 + forced_flow))
    adverse = min(1.0, max(0.0, abs(continuation) * 0.5 + vacuum * 0.1))
    fill_probability = max(0.0, min(1.0, 1.0 - adverse - abs(state.spread_change_bps) / 100.0))
    score = continuation if abs(continuation) >= abs(reversal) else reversal
    label = "continuation" if score == continuation else "reversal"
    return ScalpingForecast(
        1 if score > 0 else -1 if score < 0 else 0,
        continuation,
        reversal,
        fill_probability,
        adverse,
        label,
    )


@dataclass(frozen=True)
class PortfolioMetaState:
    forecasts: Mapping[str, float]
    regimes: Mapping[str, str]
    performance_decay: Mapping[str, float]
    drift: Mapping[str, float]
    correlations: Mapping[str, Mapping[str, float]]
    sleeve_budgets: Mapping[str, float]


@dataclass(frozen=True)
class PortfolioMetaTargets:
    strategy_weights: Mapping[str, float]
    sleeve_allocations: Mapping[str, float]
    suppressed: tuple[str, ...]


def adaptive_portfolio_targets(state: PortfolioMetaState) -> PortfolioMetaTargets:
    names = tuple(sorted(state.forecasts))
    if not names or any(
        name not in state.performance_decay or name not in state.drift for name in names
    ):
        raise ValueError("portfolio meta state is incomplete")
    raw: dict[str, float] = {}
    suppressed: list[str] = []
    for name in names:
        forecast = float(state.forecasts[name])
        conflict = any(
            forecast * float(state.forecasts[other]) < 0
            and abs(float(state.correlations.get(name, {}).get(other, 0.0))) >= 0.8
            for other in names
            if other != name
        )
        decay = max(0.0, min(1.0, float(state.performance_decay[name])))
        drift_multiplier = max(0.0, 1.0 - abs(float(state.drift[name])))
        regime_multiplier = 1.0 if state.regimes.get(name, "active") == "active" else 0.0
        if conflict or not decay or not drift_multiplier or not regime_multiplier:
            suppressed.append(name)
        raw[name] = (
            0.0 if conflict else abs(forecast) * decay * drift_multiplier * regime_multiplier
        )
    total = sum(raw.values())
    weights = {name: value / total if total else 0.0 for name, value in raw.items()}
    sleeve_total = sum(max(0.0, float(value)) for value in state.sleeve_budgets.values())
    sleeves = {
        name: max(0.0, float(value)) / sleeve_total if sleeve_total else 0.0
        for name, value in state.sleeve_budgets.items()
    }
    return PortfolioMetaTargets(weights, sleeves, tuple(sorted(suppressed)))


@dataclass(frozen=True)
class ExecutionSlice:
    sequence: int
    quantity: float
    order_type: str
    participation: float
    limit_price: float | None = None
    post_only: bool = False
    use_sor: bool = False


def execution_schedule(
    policy: str,
    *,
    quantity: float,
    slices: int = 4,
    spread_bps: float = 0.0,
    visible_depth: float = math.inf,
    forecast_volume: float = math.inf,
    urgency: float = 0.5,
) -> tuple[ExecutionSlice, ...]:
    if quantity <= 0 or slices < 1 or not 0 <= urgency <= 1:
        raise ValueError("execution schedule inputs are invalid")
    supported = {
        "market",
        "limit",
        "post_only",
        "twap",
        "vwap",
        "percentage_of_volume",
        "adaptive_urgency",
        "spread_aware",
        "depth_aware",
        "cancel_replace",
        "binance_spot_sor",
    }
    if policy not in supported:
        raise ValueError(f"unsupported execution policy: {policy}")
    if policy in {"market", "limit", "post_only", "binance_spot_sor"}:
        slices = 1
    cap = min(quantity, visible_depth * 0.01, forecast_volume * 0.05)
    executable = quantity if math.isinf(cap) else cap
    weights = [1.0] * slices
    if policy == "vwap":
        weights = [float(index + 1) for index in range(slices)]
    elif policy == "adaptive_urgency":
        weights = [1.0 + urgency * (slices - index) for index in range(slices)]
    total = sum(weights)
    order_type = (
        "market"
        if policy == "market" or (policy == "spread_aware" and spread_bps <= 1)
        else "limit"
    )
    return tuple(
        ExecutionSlice(
            sequence=index,
            quantity=executable * weight / total,
            order_type=order_type,
            participation=executable / forecast_volume
            if math.isfinite(forecast_volume) and forecast_volume
            else 0.0,
            post_only=policy == "post_only",
            use_sor=policy == "binance_spot_sor",
        )
        for index, weight in enumerate(weights)
    )


@dataclass(frozen=True)
class QuoteState:
    information_time: str
    bid: float
    ask: float
    volatility: float
    inventory: float
    inventory_limit: float
    adverse_selection_probability: float
    bid_queue: float
    ask_queue: float


@dataclass(frozen=True)
class QuoteTargets:
    bid_price: float | None
    ask_price: float | None
    bid_quantity: float
    ask_quantity: float
    cancel_existing: bool
    emergency_inventory_exit: bool
    queue_position_estimate: Mapping[str, float]


def market_making_quotes(
    state: QuoteState,
    *,
    minimum_spread_bps: float = 2.0,
    maker_fee_bps: float = 1.0,
    maker_rebate_bps: float = 0.0,
) -> QuoteTargets:
    timestamp(state.information_time, field="information_time")
    if state.bid <= 0 or state.ask <= state.bid or state.inventory_limit <= 0:
        raise ValueError("quote state is invalid")
    emergency = abs(state.inventory) >= state.inventory_limit
    if emergency or state.adverse_selection_probability >= 0.8:
        return QuoteTargets(None, None, 0.0, 0.0, True, emergency, {"bid": 0.0, "ask": 0.0})
    mid = (state.bid + state.ask) / 2
    half_spread = max(
        (state.ask - state.bid) / 2,
        mid * (minimum_spread_bps + maker_fee_bps - maker_rebate_bps) / 20_000,
    )
    half_spread *= 1.0 + max(0.0, state.volatility)
    skew = max(-1.0, min(1.0, state.inventory / state.inventory_limit)) * half_spread
    size = max(0.0, 1.0 - abs(skew / half_spread))
    return QuoteTargets(
        mid - half_spread - skew,
        mid + half_spread - skew,
        size,
        size,
        True,
        False,
        {"bid": 1.0 / (1.0 + state.bid_queue), "ask": 1.0 / (1.0 + state.ask_queue)},
    )


@dataclass(frozen=True)
class StrategyGenome:
    thesis_id: str
    parent_ids: tuple[str, ...]
    lineage_id: str
    cumulative_trial_count: int
    predeclared_universe: tuple[str, ...]
    permitted_features: tuple[str, ...]
    risk_boundaries: Mapping[str, float]
    typed_parameters: Mapping[str, float]

    @property
    def genome_id(self) -> str:
        return canonical_hash(self)

    def mutate(self, changes: Mapping[str, float]) -> StrategyGenome:
        if not set(changes).issubset(self.typed_parameters):
            raise ValueError("genetic mutation cannot add undeclared parameters")
        return StrategyGenome(
            thesis_id=self.thesis_id,
            parent_ids=(self.genome_id,),
            lineage_id=self.lineage_id,
            cumulative_trial_count=self.cumulative_trial_count + 1,
            predeclared_universe=self.predeclared_universe,
            permitted_features=self.permitted_features,
            risk_boundaries=self.risk_boundaries,
            typed_parameters={**self.typed_parameters, **changes},
        )

    def crossover(self, other: StrategyGenome) -> StrategyGenome:
        if (
            other.thesis_id != self.thesis_id
            or other.lineage_id != self.lineage_id
            or other.predeclared_universe != self.predeclared_universe
            or other.permitted_features != self.permitted_features
            or other.risk_boundaries != self.risk_boundaries
        ):
            raise ValueError("genetic crossover cannot cross thesis or risk boundaries")
        keys = tuple(sorted(set(self.typed_parameters) | set(other.typed_parameters)))
        parameters = {}
        for index, key in enumerate(keys):
            preferred = self.typed_parameters if index % 2 == 0 else other.typed_parameters
            fallback = other.typed_parameters if index % 2 == 0 else self.typed_parameters
            parameters[key] = float(preferred[key] if key in preferred else fallback[key])
        return StrategyGenome(
            thesis_id=self.thesis_id,
            parent_ids=(self.genome_id, other.genome_id),
            lineage_id=self.lineage_id,
            cumulative_trial_count=max(self.cumulative_trial_count, other.cumulative_trial_count)
            + 1,
            predeclared_universe=self.predeclared_universe,
            permitted_features=self.permitted_features,
            risk_boundaries=self.risk_boundaries,
            typed_parameters=parameters,
        )


def genetic_fitness(evidence: Mapping[str, object]) -> float:
    if "holdout" in evidence or "protected" in evidence or "forward" in evidence:
        raise ValueError("genetic selection may use development evidence only")
    if evidence.get("stage") != "development":
        raise ValueError("genetic fitness requires development-stage evidence")
    value = evidence.get("cost_adjusted_return")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("genetic fitness needs a measured cost-adjusted return")
    return float(value)


@dataclass(frozen=True)
class MarketMakingCapabilityPolicy:
    paper_enabled: bool = True
    live_enabled: bool = False
    minimum_event_replay_fills: int = 500

    def authorise(self, *, mode: str, event_replay: Mapping[str, object]) -> bool:
        if mode == "paper":
            return self.paper_enabled
        if mode != "live" or not self.live_enabled or event_replay.get("passed") is not True:
            return False
        fills = event_replay.get("fills")
        return (
            isinstance(fills, int)
            and not isinstance(fills, bool)
            and fills >= self.minimum_event_replay_fills
        )


@dataclass(frozen=True)
class MarketMakingReplayEvidence:
    events: int
    fills: int
    gross_spread: float
    fees: float
    rebates: float
    inventory_pnl: float
    maximum_inventory: float
    emergency_exits: int

    @property
    def passed(self) -> bool:
        return self.events > 0 and self.fills > 0 and self.net_pnl > 0

    @property
    def net_pnl(self) -> float:
        return self.gross_spread - self.fees + self.rebates + self.inventory_pnl


def replay_market_making(
    states: tuple[QuoteState, ...],
    *,
    maker_fee_bps: float = 1.0,
    maker_rebate_bps: float = 0.0,
) -> MarketMakingReplayEvidence:
    """Deterministic paper fill model for predeclared event-time quote states."""

    fills = 0
    gross_spread = 0.0
    fees = 0.0
    rebates = 0.0
    emergency_exits = 0
    maximum_inventory = 0.0
    previous_mid: float | None = None
    inventory_pnl = 0.0
    for state in states:
        quotes = market_making_quotes(
            state,
            maker_fee_bps=maker_fee_bps,
            maker_rebate_bps=maker_rebate_bps,
        )
        mid = (state.bid + state.ask) / 2
        if previous_mid is not None:
            inventory_pnl += state.inventory * (mid - previous_mid)
        previous_mid = mid
        maximum_inventory = max(maximum_inventory, abs(state.inventory))
        if quotes.emergency_inventory_exit:
            emergency_exits += 1
            continue
        for price, probability in (
            (quotes.bid_price, quotes.queue_position_estimate["bid"]),
            (quotes.ask_price, quotes.queue_position_estimate["ask"]),
        ):
            if price is None or probability < 0.25:
                continue
            fills += 1
            gross_spread += abs(mid - price)
            fees += price * maker_fee_bps / 10_000
            rebates += price * maker_rebate_bps / 10_000
    return MarketMakingReplayEvidence(
        len(states),
        fills,
        gross_spread,
        fees,
        rebates,
        inventory_pnl,
        maximum_inventory,
        emergency_exits,
    )


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


ADVANCED_IMPLEMENTATIONS = {
    "pairs_trading": evaluate_pair,
    "seasonality": SeasonalityModel.evaluate,
    "volatility_forecast": forecast_volatility,
    "statistical_pattern_recognition": recognise_statistical_pattern,
    "point_in_time_sentiment": PointInTimeSentiment,
    "aggressor_flow_scalping": microstructure_scalping_signal,
    "liquidation_burst_scalping": microstructure_scalping_signal,
    "liquidity_vacuum_scalping": microstructure_scalping_signal,
    "regime_routing": adaptive_portfolio_targets,
    "performance_decay_weighting": adaptive_portfolio_targets,
    "forecast_conflict_suppression": adaptive_portfolio_targets,
    "drift_based_capital_reduction": adaptive_portfolio_targets,
    "sleeve_reallocation": adaptive_portfolio_targets,
    "limit_execution": execution_schedule,
    "post_only_execution": execution_schedule,
    "twap": execution_schedule,
    "vwap": execution_schedule,
    "percentage_of_volume": execution_schedule,
    "adaptive_urgency": execution_schedule,
    "spread_aware_selection": execution_schedule,
    "depth_aware_sizing": execution_schedule,
    "cancel_replace": execution_schedule,
    "binance_spot_sor": execution_schedule,
    "inventory_aware_market_making": market_making_quotes,
}
