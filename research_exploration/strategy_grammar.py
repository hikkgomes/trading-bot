"""Composable, safe strategy grammar for autonomous hypothesis generation.

The grammar emits :class:`~research_exploration.hypothesis_schema.Hypothesis`
objects, never Python source.  That is an important trust boundary: generated
ideas can be serialized, inspected, hashed and evaluated by the existing
causal predicate engine without ``eval``/``exec`` or arbitrary imports.

Unlike the legacy named-family generator, this module composes predicates,
timeframe roles, exits and risk rules into a very large search space.  It also
supports recursive mutation and crossover.  The orchestration layer owns
deduplication, experiment memory and budgets; this module owns only the typed
grammar and its invariants.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from research_exploration.feature_inventory import classify_column
from research_exploration.hypothesis_schema import (
    TF_RANK,
    ExitRule,
    Hypothesis,
    Predicate,
    RiskRule,
)


@dataclass(frozen=True)
class SearchSpace:
    """Product/time-horizon constraints supplied to the grammar."""

    name: str
    product: str
    market: str
    pnl_unit: str
    opportunity_type: str
    base_timeframe: str
    regime_timeframe: str
    setup_timeframe: str
    trigger_timeframe: str
    directions: tuple[str, ...]
    take_profit_range: tuple[float, float]
    stop_loss_range: tuple[float, float]
    horizon_range: tuple[int, int]
    risk_per_trade_range: tuple[float, float]
    max_position_fraction: float
    max_trades_per_day: int | None
    symbol: str = "BTCUSDT"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SearchSpace:
        string_fields = (
            "name",
            "product",
            "market",
            "pnl_unit",
            "opportunity_type",
            "base_timeframe",
            "regime_timeframe",
            "setup_timeframe",
            "trigger_timeframe",
        )
        for field_name in string_fields:
            if not isinstance(payload[field_name], str) or not payload[field_name].strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        directions = payload["directions"]
        if (
            not isinstance(directions, Sequence)
            or isinstance(directions, str | bytes)
            or not directions
            or any(not isinstance(value, str) for value in directions)
        ):
            raise ValueError("directions must be a non-empty list of strings")
        max_position_fraction = payload["max_position_fraction"]
        if isinstance(max_position_fraction, bool) or not isinstance(
            max_position_fraction, int | float
        ):
            raise ValueError("max_position_fraction must be numeric")
        max_position_fraction = float(max_position_fraction)
        if not math.isfinite(max_position_fraction):
            raise ValueError("max_position_fraction must be finite")
        max_trades_per_day = payload.get("max_trades_per_day")
        if max_trades_per_day is not None and (
            isinstance(max_trades_per_day, bool) or not isinstance(max_trades_per_day, int)
        ):
            raise ValueError("max_trades_per_day must be an integer or null")
        return cls(
            name=payload["name"].strip(),
            product=payload["product"].strip(),
            market=payload["market"].strip(),
            pnl_unit=payload["pnl_unit"].strip(),
            opportunity_type=payload["opportunity_type"].strip(),
            base_timeframe=payload["base_timeframe"].strip(),
            regime_timeframe=payload["regime_timeframe"].strip(),
            setup_timeframe=payload["setup_timeframe"].strip(),
            trigger_timeframe=payload["trigger_timeframe"].strip(),
            directions=tuple(value.strip() for value in directions),
            take_profit_range=_float_pair(payload["take_profit_range"], "take_profit_range"),
            stop_loss_range=_float_pair(payload["stop_loss_range"], "stop_loss_range"),
            horizon_range=_int_pair(payload["horizon_range"], "horizon_range"),
            risk_per_trade_range=_float_pair(
                payload["risk_per_trade_range"], "risk_per_trade_range"
            ),
            max_position_fraction=max_position_fraction,
            max_trades_per_day=max_trades_per_day,
            symbol=str(payload.get("symbol", "BTCUSDT")).strip().upper(),
        )

    def __post_init__(self) -> None:
        if not self.name or not self.product or not self.opportunity_type:
            raise ValueError("search-space identity fields cannot be empty")
        if not self.symbol or not self.symbol.isalnum() or not self.symbol.endswith("USDT"):
            raise ValueError("search-space symbol must be an uppercase USDT pair")
        if self.market not in {"spot", "futures"}:
            raise ValueError("search-space market must be spot or futures")
        if self.pnl_unit not in {"btc", "usdt"}:
            raise ValueError("search-space pnl_unit must be btc or usdt")
        if not self.directions or any(value not in {"long", "short"} for value in self.directions):
            raise ValueError("search-space directions must contain long and/or short")
        for label, timeframe in (
            ("base_timeframe", self.base_timeframe),
            ("regime_timeframe", self.regime_timeframe),
            ("setup_timeframe", self.setup_timeframe),
            ("trigger_timeframe", self.trigger_timeframe),
        ):
            if timeframe not in TF_RANK:
                raise ValueError(f"{label} has unknown timeframe {timeframe!r}")
        if not (
            TF_RANK[self.regime_timeframe]
            >= TF_RANK[self.setup_timeframe]
            >= TF_RANK[self.trigger_timeframe]
            >= TF_RANK[self.base_timeframe]
        ):
            raise ValueError(
                "search-space timeframes must satisfy regime >= setup >= trigger >= base"
            )
        for label, bounds in (
            ("take_profit_range", self.take_profit_range),
            ("stop_loss_range", self.stop_loss_range),
            ("risk_per_trade_range", self.risk_per_trade_range),
        ):
            if bounds[0] <= 0 or bounds[1] < bounds[0]:
                raise ValueError(f"{label} must be positive ascending bounds")
        if self.horizon_range[0] <= 0 or self.horizon_range[1] < self.horizon_range[0]:
            raise ValueError("horizon_range must be positive ascending bounds")
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be in (0, 1]")
        if self.max_trades_per_day is not None and self.max_trades_per_day <= 0:
            raise ValueError("max_trades_per_day must be positive when configured")


@dataclass(frozen=True)
class GrammarLimits:
    min_regime_predicates: int = 1
    max_regime_predicates: int = 2
    min_setup_predicates: int = 1
    max_setup_predicates: int = 3
    min_trigger_predicates: int = 1
    max_trigger_predicates: int = 3
    max_total_predicates: int = 7
    max_dynamic_features_per_stage: int = 48

    def __post_init__(self) -> None:
        pairs = (
            (self.min_regime_predicates, self.max_regime_predicates),
            (self.min_setup_predicates, self.max_setup_predicates),
            (self.min_trigger_predicates, self.max_trigger_predicates),
        )
        if any(lo <= 0 or hi < lo for lo, hi in pairs):
            raise ValueError("predicate limits must be positive ascending bounds")
        if self.max_total_predicates < sum(lo for lo, _ in pairs):
            raise ValueError("max_total_predicates is below the required stage minimum")
        if self.max_dynamic_features_per_stage <= 0:
            raise ValueError("max_dynamic_features_per_stage must be positive")


@dataclass(frozen=True)
class PredicateCandidate:
    key: str
    predicate: Predicate
    motifs: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedIdea:
    hypothesis: Hypothesis
    generation_method: str
    grammar_keys: tuple[str, ...]
    motif: str
    parent_hashes: tuple[str, ...] = ()
    adaptation_reasons: tuple[str, ...] = ()


MOTIFS = (
    "trend_following",
    "countertrend_reversion",
    "range_expansion",
    "volatility_transition",
    "orderflow_confirmation",
    "hybrid",
)

DEFAULT_FEATURES = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_20",
        "ema_50",
        "ema_200",
        "sma_20",
        "sma_50",
        "sma_200",
        "adx_14",
        "natr_14",
        "stddev_20",
        "rsi_14",
        "mom_10",
        "macd_macd",
        "macd_macdsignal",
        "bbands_20_middleband",
        "bbands_20_upperband",
        "bbands_20_lowerband",
        "max_20",
        "min_20",
        "max_50",
        "min_50",
        "volume_z_20",
        "taker_buy_ratio",
        "linearreg_slope_20",
    }
)


def _float_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise ValueError(f"{label} values must be numeric")
    result = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} values must be finite")
    return result


def _int_pair(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{label} values must be integers")
    return value[0], value[1]


def _features_for(
    available_features: Mapping[str, Iterable[str]] | None,
    timeframe: str,
) -> set[str]:
    if available_features is None:
        return set(DEFAULT_FEATURES)
    values = available_features.get(timeframe)
    if values is None:
        return set(DEFAULT_FEATURES)
    return {str(value) for value in values if value and value != "timestamp"}


def _has(features: set[str], *names: str) -> bool:
    return all(name in features for name in names)


def _candidate(
    key: str,
    timeframe: str,
    feature: str,
    op: str,
    motifs: Sequence[str],
    **kwargs: Any,
) -> PredicateCandidate:
    return PredicateCandidate(
        key=key,
        predicate=Predicate(timeframe=timeframe, feature=feature, op=op, **kwargs),
        motifs=tuple(motifs),
    )


def _core_candidates(
    stage: str,
    timeframe: str,
    direction: str,
    features: set[str],
    rng: random.Random,
) -> list[PredicateCandidate]:
    long = direction == "long"
    out: list[PredicateCandidate] = []

    if stage == "regime":
        for root in ("ema", "sma"):
            fast, slow = f"{root}_50", f"{root}_200"
            if _has(features, fast, slow):
                out.append(
                    _candidate(
                        f"{stage}:{root}_trend",
                        timeframe,
                        fast,
                        "gt_feature" if long else "lt_feature",
                        ("trend_following", "orderflow_confirmation", "hybrid"),
                        feature_b=slow,
                    )
                )
        for average in ("ema_20", "ema_50", "ema_200", "sma_50", "sma_200"):
            if _has(features, "close", average):
                out.append(
                    _candidate(
                        f"{stage}:price_vs_{average}",
                        timeframe,
                        "close",
                        "gt_feature" if long else "lt_feature",
                        ("trend_following", "countertrend_reversion", "hybrid"),
                        feature_b=average,
                    )
                )
        if "adx_14" in features:
            out.extend(
                [
                    _candidate(
                        f"{stage}:adx_trending:{threshold}",
                        timeframe,
                        "adx_14",
                        "ge",
                        ("trend_following", "range_expansion", "orderflow_confirmation"),
                        reference=float(threshold),
                    )
                    for threshold in rng.sample([18, 20, 22, 25, 28], k=3)
                ]
            )
            out.append(
                _candidate(
                    f"{stage}:adx_quiet",
                    timeframe,
                    "adx_14",
                    "le",
                    ("countertrend_reversion", "volatility_transition"),
                    reference=float(rng.choice([18, 20, 22])),
                )
            )
        if "natr_14" in features:
            out.extend(
                [
                    _candidate(
                        f"{stage}:natr_{side}",
                        timeframe,
                        "natr_14",
                        "q_ge" if side == "high" else "q_le",
                        ("volatility_transition", "range_expansion", "hybrid"),
                        window=rng.choice([90, 120, 180, 240]),
                        quantile=rng.choice([0.25, 0.35, 0.65, 0.75]),
                    )
                    for side in ("low", "high")
                ]
            )
        if "linearreg_slope_20" in features:
            out.append(
                _candidate(
                    f"{stage}:linearreg_direction",
                    timeframe,
                    "linearreg_slope_20",
                    "slope_up" if long else "slope_down",
                    ("trend_following", "hybrid"),
                    lookback=rng.choice([2, 3, 5, 8]),
                    reference=0.0,
                )
            )

    if stage == "setup":
        if "rsi_14" in features:
            out.extend(
                [
                    _candidate(
                        f"{stage}:rsi_pullback:{threshold}",
                        timeframe,
                        "rsi_14",
                        "le" if long else "ge",
                        ("trend_following", "countertrend_reversion", "hybrid"),
                        reference=float(threshold if long else 100 - threshold),
                    )
                    for threshold in rng.sample([25, 30, 35, 40, 45, 48], k=3)
                ]
            )
            low, high = (40.0, 68.0) if long else (32.0, 60.0)
            out.append(
                _candidate(
                    f"{stage}:rsi_continuation_band",
                    timeframe,
                    "rsi_14",
                    "between",
                    ("trend_following", "orderflow_confirmation"),
                    low=low,
                    high=high,
                )
            )
        if _has(features, "close", "bbands_20_lowerband", "bbands_20_upperband"):
            out.append(
                _candidate(
                    f"{stage}:band_extreme",
                    timeframe,
                    "close",
                    "lt_feature" if long else "gt_feature",
                    ("countertrend_reversion", "hybrid"),
                    feature_b="bbands_20_lowerband" if long else "bbands_20_upperband",
                )
            )
        for average in ("ema_20", "ema_50", "sma_20", "sma_50"):
            if _has(features, "close", average):
                out.append(
                    _candidate(
                        f"{stage}:price_posture:{average}",
                        timeframe,
                        "close",
                        "gt_feature" if long else "lt_feature",
                        ("trend_following", "orderflow_confirmation", "hybrid"),
                        feature_b=average,
                    )
                )
        for feature in ("natr_14", "stddev_20"):
            if feature in features:
                out.extend(
                    [
                        _candidate(
                            f"{stage}:{feature}:{side}",
                            timeframe,
                            feature,
                            "q_le" if side == "compressed" else "q_ge",
                            (
                                ("volatility_transition", "range_expansion")
                                if side == "compressed"
                                else ("countertrend_reversion", "hybrid")
                            ),
                            window=rng.choice([60, 90, 120, 180]),
                            quantile=rng.choice([0.25, 0.35, 0.65, 0.75]),
                        )
                        for side in ("compressed", "expanded")
                    ]
                )
        if "mom_10" in features:
            out.append(
                _candidate(
                    f"{stage}:momentum_pause",
                    timeframe,
                    "mom_10",
                    "falling" if long else "rising",
                    ("trend_following", "countertrend_reversion", "hybrid"),
                    lookback=rng.choice([2, 3, 5, 8]),
                )
            )

    if stage == "trigger":
        if _has(features, "close", "open"):
            out.append(
                _candidate(
                    f"{stage}:candle_direction",
                    timeframe,
                    "close",
                    "gt_feature" if long else "lt_feature",
                    ("countertrend_reversion", "orderflow_confirmation", "hybrid"),
                    feature_b="open",
                )
            )
        if _has(features, "macd_macd", "macd_macdsignal"):
            out.append(
                _candidate(
                    f"{stage}:macd_cross",
                    timeframe,
                    "macd_macd",
                    "cross_above" if long else "cross_below",
                    ("trend_following", "countertrend_reversion", "hybrid"),
                    feature_b="macd_macdsignal",
                )
            )
        for average in ("ema_20", "ema_50", "sma_20", "sma_50"):
            if _has(features, "close", average):
                out.append(
                    _candidate(
                        f"{stage}:price_cross:{average}",
                        timeframe,
                        "close",
                        "cross_above" if long else "cross_below",
                        ("trend_following", "countertrend_reversion", "hybrid"),
                        feature_b=average,
                    )
                )
        if "rsi_14" in features:
            out.append(
                _candidate(
                    f"{stage}:rsi_turn",
                    timeframe,
                    "rsi_14",
                    "rising" if long else "falling",
                    ("countertrend_reversion", "trend_following", "hybrid"),
                    lookback=rng.choice([2, 3, 5]),
                )
            )
        if "mom_10" in features:
            out.append(
                _candidate(
                    f"{stage}:momentum_acceleration",
                    timeframe,
                    "mom_10",
                    "slope_up" if long else "slope_down",
                    ("trend_following", "range_expansion", "orderflow_confirmation"),
                    lookback=rng.choice([2, 3, 5]),
                    reference=0.0,
                )
            )
        breakout_feature = "max_20" if long else "min_20"
        if _has(features, "close", breakout_feature):
            out.append(
                _candidate(
                    f"{stage}:prior_range_break",
                    timeframe,
                    "close",
                    "gt_feature" if long else "lt_feature",
                    ("range_expansion", "volatility_transition", "hybrid"),
                    feature_b=breakout_feature,
                    shift_b=1,
                )
            )
        if "volume_z_20" in features:
            out.append(
                _candidate(
                    f"{stage}:volume_confirmation",
                    timeframe,
                    "volume_z_20",
                    "ge",
                    ("range_expansion", "orderflow_confirmation", "hybrid"),
                    reference=rng.choice([0.25, 0.5, 0.75, 1.0, 1.25]),
                )
            )
        if "taker_buy_ratio" in features:
            out.append(
                _candidate(
                    f"{stage}:taker_imbalance",
                    timeframe,
                    "taker_buy_ratio",
                    "ge" if long else "le",
                    ("orderflow_confirmation", "range_expansion", "hybrid"),
                    reference=rng.choice([0.48, 0.5, 0.52, 0.55]),
                )
            )
    return out


def _dynamic_candidates(
    stage: str,
    timeframe: str,
    direction: str,
    features: set[str],
    rng: random.Random,
    limit: int,
) -> list[PredicateCandidate]:
    """Discover additional grammar atoms from the actual parquet schema.

    The discovery remains deliberately typed and conservative: arbitrary
    columns can only be used through causal rolling ranks/slopes, known price
    comparisons, extrema breaks, or discrete candlestick predicates.
    """

    allowed_families = {
        "regime": {"trend_ma", "trend_dmi", "statistic", "volatility", "momentum"},
        "setup": {
            "trend_ma",
            "trend_dmi",
            "momentum",
            "volatility",
            "range_extrema",
            "orderflow",
            "volume",
        },
        "trigger": {
            "trend_ma",
            "momentum",
            "range_extrema",
            "orderflow",
            "volume",
            "candlestick",
        },
    }[stage]
    ignored = DEFAULT_FEATURES | {"open", "high", "low", "close", "volume"}
    pool = [
        feature
        for feature in sorted(features - ignored)
        if classify_column(feature) in allowed_families
    ]
    if len(pool) > limit:
        pool = rng.sample(pool, k=limit)

    long = direction == "long"
    out: list[PredicateCandidate] = []
    for feature in pool:
        family = classify_column(feature)
        if family == "candlestick":
            out.append(
                _candidate(
                    f"dynamic:{stage}:pattern:{feature}",
                    timeframe,
                    feature,
                    "bullish" if long else "bearish",
                    ("countertrend_reversion", "orderflow_confirmation", "hybrid"),
                )
            )
            continue
        if (
            family == "range_extrema"
            and stage == "trigger"
            and feature.startswith(("max_", "min_"))
        ):
            expected = feature.startswith("max_") if long else feature.startswith("min_")
            if expected and "close" in features:
                out.append(
                    _candidate(
                        f"dynamic:{stage}:range_break:{feature}",
                        timeframe,
                        "close",
                        "gt_feature" if long else "lt_feature",
                        ("range_expansion", "volatility_transition", "hybrid"),
                        feature_b=feature,
                        shift_b=1,
                    )
                )
        if family == "trend_ma" and "close" in features:
            out.append(
                _candidate(
                    f"dynamic:{stage}:price_relation:{feature}",
                    timeframe,
                    "close",
                    "gt_feature" if long else "lt_feature",
                    ("trend_following", "countertrend_reversion", "hybrid"),
                    feature_b=feature,
                )
            )
        continuation_op = "q_ge" if long else "q_le"
        if rng.random() < 0.35:
            continuation_op = "q_le" if continuation_op == "q_ge" else "q_ge"
        out.append(
            _candidate(
                f"dynamic:{stage}:rolling_rank:{feature}:{continuation_op}",
                timeframe,
                feature,
                continuation_op,
                (
                    "orderflow_confirmation",
                    "volatility_transition",
                    "countertrend_reversion",
                    "hybrid",
                ),
                window=rng.choice([48, 60, 90, 120, 180, 240]),
                quantile=rng.choice([0.2, 0.3, 0.4, 0.6, 0.7, 0.8]),
            )
        )
        if stage != "regime" or family in {"trend_ma", "statistic", "momentum"}:
            out.append(
                _candidate(
                    f"dynamic:{stage}:slope:{feature}",
                    timeframe,
                    feature,
                    "slope_up" if long else "slope_down",
                    ("trend_following", "orderflow_confirmation", "hybrid"),
                    lookback=rng.choice([2, 3, 5, 8, 13]),
                    reference=0.0,
                )
            )
    return out


def predicate_candidates(
    stage: str,
    timeframe: str,
    direction: str,
    *,
    available_features: Mapping[str, Iterable[str]] | None,
    rng: random.Random,
    limits: GrammarLimits,
) -> list[PredicateCandidate]:
    if stage not in {"regime", "setup", "trigger"}:
        raise ValueError(f"unknown strategy stage {stage!r}")
    features = _features_for(available_features, timeframe)
    candidates = _core_candidates(stage, timeframe, direction, features, rng)
    candidates.extend(
        _dynamic_candidates(
            stage,
            timeframe,
            direction,
            features,
            rng,
            limits.max_dynamic_features_per_stage,
        )
    )
    deduplicated: dict[tuple[Any, ...], PredicateCandidate] = {}
    for item in candidates:
        token = predicate_token(item.predicate, include_values=True)
        deduplicated.setdefault(token, item)
    return list(deduplicated.values())


def _weighted_sample_without_replacement(
    candidates: Sequence[PredicateCandidate],
    count: int,
    *,
    motif: str,
    feedback_weights: Mapping[str, float] | None,
    rng: random.Random,
) -> list[PredicateCandidate]:
    remaining = list(candidates)
    selected: list[PredicateCandidate] = []
    feedback_weights = feedback_weights or {}
    while remaining and len(selected) < count:
        weights = []
        for item in remaining:
            motif_weight = (
                3.0 if motif in item.motifs else (1.5 if "hybrid" in item.motifs else 1.0)
            )
            learned = float(feedback_weights.get(item.key, 1.0))
            weights.append(max(0.05, min(20.0, motif_weight * learned)))
        choice = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
        candidate = remaining.pop(choice)
        # Avoid stacking several tests of exactly the same feature in one stage.
        if any(existing.predicate.feature == candidate.predicate.feature for existing in selected):
            if rng.random() < 0.8:
                continue
        selected.append(candidate)
    return selected


def _stage_bounds(stage: str, limits: GrammarLimits) -> tuple[int, int]:
    return {
        "regime": (limits.min_regime_predicates, limits.max_regime_predicates),
        "setup": (limits.min_setup_predicates, limits.max_setup_predicates),
        "trigger": (limits.min_trigger_predicates, limits.max_trigger_predicates),
    }[stage]


def _stage_timeframe(space: SearchSpace, stage: str) -> str:
    return {
        "regime": space.regime_timeframe,
        "setup": space.setup_timeframe,
        "trigger": space.trigger_timeframe,
    }[stage]


def _random_between(bounds: tuple[float, float], rng: random.Random) -> float:
    lo, hi = bounds
    if math.isclose(lo, hi):
        return lo
    # Log-uniform avoids over-concentrating on the largest risk/exit values.
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _new_exit(space: SearchSpace, rng: random.Random) -> ExitRule:
    tp = _random_between(space.take_profit_range, rng)
    sl = _random_between(space.stop_loss_range, rng)
    horizon = rng.randint(*space.horizon_range)
    return ExitRule(
        take_profit=round(tp, 6),
        stop_loss=round(sl, 6),
        horizon_bars=horizon,
        # The canonical simulator and live runner currently implement fixed
        # TP/SL/time exits.  Do not generate advisory fields they would ignore.
        trail=False,
    )


def _new_risk(space: SearchSpace, rng: random.Random) -> RiskRule:
    risk = _random_between(space.risk_per_trade_range, rng)
    position_fraction = min(
        space.max_position_fraction,
        rng.choice([0.05, 0.075, 0.1, 0.15, 0.2, space.max_position_fraction]),
    )
    return RiskRule(
        risk_per_trade=round(risk, 6),
        max_position_fraction=round(position_fraction, 6),
        max_trades_per_day=space.max_trades_per_day,
        max_daily_loss_r=round(rng.choice([1.0, 1.5, 2.0, 2.5]), 2),
        cooldown_bars=rng.choice([0, 1, 2, 3, 5, 8, 13]),
        min_atr_pct=(rng.choice([0.0005, 0.001, 0.0015]) if rng.random() < 0.35 else None),
        max_atr_pct=(rng.choice([0.015, 0.02, 0.03, 0.05]) if rng.random() < 0.35 else None),
    )


def build_fresh_hypothesis(
    space: SearchSpace,
    *,
    rng: random.Random,
    available_features: Mapping[str, Iterable[str]] | None = None,
    feedback_weights: Mapping[str, float] | None = None,
    limits: GrammarLimits | None = None,
    motif: str | None = None,
) -> GeneratedIdea:
    limits = limits or GrammarLimits()
    motif = motif or rng.choice(MOTIFS)
    if motif not in MOTIFS:
        raise ValueError(f"unknown motif {motif!r}")
    direction = rng.choice(space.directions)
    stages: dict[str, list[Predicate]] = {}
    keys: list[str] = []
    remaining_total = limits.max_total_predicates
    for index, stage in enumerate(("regime", "setup", "trigger")):
        lo, hi = _stage_bounds(stage, limits)
        minimum_for_later = sum(
            _stage_bounds(later, limits)[0] for later in ("regime", "setup", "trigger")[index + 1 :]
        )
        count = rng.randint(lo, min(hi, remaining_total - minimum_for_later))
        candidates = predicate_candidates(
            stage,
            _stage_timeframe(space, stage),
            direction,
            available_features=available_features,
            rng=rng,
            limits=limits,
        )
        selected = _weighted_sample_without_replacement(
            candidates,
            count,
            motif=motif,
            feedback_weights=feedback_weights,
            rng=rng,
        )
        if len(selected) < lo:
            raise ValueError(
                f"{space.name}: grammar has only {len(selected)} usable {stage} predicates"
            )
        stages[stage] = [item.predicate for item in selected]
        keys.extend(item.key for item in selected)
        remaining_total -= len(selected)

    hypothesis = Hypothesis(
        id="GENERATED_PENDING_ID",
        family=f"generated_{motif}",
        idea=(
            f"Autonomously composed {space.opportunity_type} hypothesis using "
            f"{len(keys)} independently testable grammar atoms."
        ),
        market_logic=(
            "Generated from causal, closed-candle building blocks; its rationale is a testable "
            "combination rather than a claim that an edge exists."
        ),
        direction=direction,
        base_timeframe=space.base_timeframe,
        regime_timeframe=space.regime_timeframe,
        setup_timeframe=space.setup_timeframe,
        trigger_timeframe=space.trigger_timeframe,
        regime=stages["regime"],
        setup=stages["setup"],
        trigger=stages["trigger"],
        exit=_new_exit(space, rng),
        risk=_new_risk(space, rng),
        expected_holding=space.opportunity_type.replace("_", " "),
        expected_frequency="unknown until measured on training data",
        invalidation="stop, time exit, risk breaker, or loss of the generated entry conditions",
        tags=["autonomous_generation", "program_synthesis", motif, space.name],
    )
    problems = validate_hypothesis_against_space(
        hypothesis,
        space,
        available_features=available_features,
        limits=limits,
    )
    if problems:
        raise ValueError(f"generated invalid hypothesis: {', '.join(problems)}")
    return GeneratedIdea(
        hypothesis=hypothesis,
        generation_method="grammar_sample",
        grammar_keys=tuple(keys),
        motif=motif,
    )


def _jitter_predicate(predicate: Predicate, rng: random.Random) -> Predicate:
    changes: dict[str, Any] = {}
    if predicate.reference is not None:
        scale = max(abs(float(predicate.reference)), 0.01)
        changes["reference"] = round(
            float(predicate.reference) + rng.uniform(-0.15, 0.15) * scale,
            8,
        )
    if predicate.quantile is not None:
        changes["quantile"] = round(
            min(0.95, max(0.05, float(predicate.quantile) + rng.choice([-0.1, -0.05, 0.05, 0.1]))),
            4,
        )
    if predicate.window is not None:
        changes["window"] = max(
            10, int(round(predicate.window * rng.choice([0.75, 0.9, 1.1, 1.25])))
        )
    if predicate.lookback is not None:
        changes["lookback"] = max(
            1, int(round(predicate.lookback * rng.choice([0.67, 0.8, 1.25, 1.5])))
        )
    if predicate.low is not None and predicate.high is not None:
        width = float(predicate.high) - float(predicate.low)
        shift = rng.uniform(-0.15, 0.15) * max(width, 1.0)
        changes["low"] = round(float(predicate.low) + shift, 8)
        changes["high"] = round(float(predicate.high) + shift, 8)
    return dataclasses.replace(predicate, **changes) if changes else predicate


def _bounded(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def mutate_hypothesis(
    parent: Hypothesis,
    space: SearchSpace,
    *,
    parent_hash: str,
    rng: random.Random,
    available_features: Mapping[str, Iterable[str]] | None = None,
    feedback_weights: Mapping[str, float] | None = None,
    limits: GrammarLimits | None = None,
    failure_reasons: Sequence[str] = (),
) -> GeneratedIdea:
    """Create one recursive descendant while keeping the typed-space contract."""

    limits = limits or GrammarLimits()
    problems = validate_hypothesis_against_space(
        parent,
        space,
        available_features=available_features,
        limits=limits,
    )
    if problems:
        raise ValueError(f"parent does not belong to {space.name}: {', '.join(problems)}")
    direction = parent.direction
    stages = {
        "regime": list(parent.regime),
        "setup": list(parent.setup),
        "trigger": list(parent.trigger),
    }
    keys: list[str] = []
    operations = rng.randint(1, 3)
    exit_rule = parent.exit
    risk_rule = parent.risk

    normalized_reasons = tuple(
        dict.fromkeys(
            str(reason).strip()[:128] for reason in failure_reasons if str(reason).strip()
        )
    )
    reason_text = " ".join(normalized_reasons).lower()
    action_weights = {
        "replace": 3.0,
        "add": 1.5,
        "remove": 1.0,
        "jitter": 2.5,
        "exit": 1.5,
        "risk": 1.0,
    }
    # Failure evidence changes which safe grammar operators are attempted; it
    # never changes the language or its risk bounds.  Sparse signals are
    # simplified, weak edges are recomposed, and fragility/risk failures favor
    # robustness-oriented edits.  Unknown reasons retain the neutral prior.
    if any(token in reason_text for token in ("insufficient", "too_few", "sparse")):
        action_weights.update(remove=4.0, jitter=3.0, exit=2.0, add=0.5)
    if any(token in reason_text for token in ("no_train_edge", "no_validation_edge", "weak_edge")):
        action_weights.update(replace=5.0, jitter=3.5, exit=2.5)
    if any(token in reason_text for token in ("unstable", "fragile", "sensitivity", "drawdown")):
        action_weights.update(remove=3.0, replace=4.0, exit=3.0, risk=3.0)

    action_names = list(action_weights)
    for _ in range(operations):
        action = rng.choices(
            action_names,
            weights=[action_weights[name] for name in action_names],
            k=1,
        )[0]
        stage = rng.choice(["regime", "setup", "trigger"])
        lo, hi = _stage_bounds(stage, limits)
        if action in {"replace", "add"}:
            candidates = predicate_candidates(
                stage,
                _stage_timeframe(space, stage),
                direction,
                available_features=available_features,
                rng=rng,
                limits=limits,
            )
            selected = _weighted_sample_without_replacement(
                candidates,
                1,
                motif="hybrid",
                feedback_weights=feedback_weights,
                rng=rng,
            )
            if selected:
                item = selected[0]
                keys.append(item.key)
                if action == "replace" and stages[stage]:
                    stages[stage][rng.randrange(len(stages[stage]))] = item.predicate
                elif action == "add" and len(stages[stage]) < hi:
                    stages[stage].append(item.predicate)
        elif action == "remove" and len(stages[stage]) > lo:
            stages[stage].pop(rng.randrange(len(stages[stage])))
        elif action == "jitter" and stages[stage]:
            index = rng.randrange(len(stages[stage]))
            stages[stage][index] = _jitter_predicate(stages[stage][index], rng)
        elif action == "exit":
            exit_rule = dataclasses.replace(
                exit_rule,
                take_profit=round(
                    _bounded(
                        exit_rule.take_profit * rng.uniform(0.75, 1.3), space.take_profit_range
                    ),
                    6,
                ),
                stop_loss=round(
                    _bounded(exit_rule.stop_loss * rng.uniform(0.75, 1.3), space.stop_loss_range),
                    6,
                ),
                horizon_bars=int(
                    _bounded(
                        round(exit_rule.horizon_bars * rng.uniform(0.7, 1.4)),
                        (float(space.horizon_range[0]), float(space.horizon_range[1])),
                    )
                ),
            )
        elif action == "risk":
            risk_rule = dataclasses.replace(
                risk_rule,
                risk_per_trade=round(
                    _bounded(
                        risk_rule.risk_per_trade * rng.uniform(0.75, 1.2),
                        space.risk_per_trade_range,
                    ),
                    6,
                ),
                max_position_fraction=min(
                    space.max_position_fraction,
                    max(0.01, risk_rule.max_position_fraction * rng.uniform(0.8, 1.05)),
                ),
            )

    # Enforce the total-complexity cap after additions.
    while sum(len(values) for values in stages.values()) > limits.max_total_predicates:
        removable = [
            stage
            for stage in ("regime", "setup", "trigger")
            if len(stages[stage]) > _stage_bounds(stage, limits)[0]
        ]
        if not removable:
            break
        chosen = rng.choice(removable)
        stages[chosen].pop(rng.randrange(len(stages[chosen])))

    child = dataclasses.replace(
        parent,
        id="GENERATED_PENDING_ID",
        family="generated_evolution",
        idea=f"Autonomous recursive mutation of a prior {space.opportunity_type} experiment.",
        market_logic=(
            "A bounded descendant generated from pre-holdout experiment feedback; it makes no "
            "claim of profitability until the full validation protocol completes."
        ),
        regime=stages["regime"],
        setup=stages["setup"],
        trigger=stages["trigger"],
        exit=exit_rule,
        risk=risk_rule,
        tags=[*dict.fromkeys([*parent.tags, "autonomous_generation", "recursive_mutation"])],
    )
    problems = validate_hypothesis_against_space(
        child,
        space,
        available_features=available_features,
        limits=limits,
    )
    if problems:
        raise ValueError(f"mutation produced invalid hypothesis: {', '.join(problems)}")
    return GeneratedIdea(
        hypothesis=child,
        generation_method="recursive_mutation",
        grammar_keys=tuple(keys),
        motif="evolution",
        parent_hashes=(parent_hash,),
        adaptation_reasons=normalized_reasons,
    )


def crossover_hypotheses(
    first: Hypothesis,
    second: Hypothesis,
    space: SearchSpace,
    *,
    parent_hashes: tuple[str, str],
    rng: random.Random,
    available_features: Mapping[str, Iterable[str]] | None = None,
    feedback_weights: Mapping[str, float] | None = None,
    limits: GrammarLimits | None = None,
) -> GeneratedIdea:
    """Recombine two compatible pre-holdout parents and lightly mutate the child."""

    limits = limits or GrammarLimits()
    for parent in (first, second):
        problems = validate_hypothesis_against_space(
            parent,
            space,
            available_features=available_features,
            limits=limits,
        )
        if problems:
            raise ValueError(f"crossover parent is incompatible: {', '.join(problems)}")
    if first.direction != second.direction:
        raise ValueError("crossover parents must have the same direction")

    stages = {
        stage: list(getattr(rng.choice((first, second)), stage))
        for stage in ("regime", "setup", "trigger")
    }
    exit_rule = ExitRule(
        take_profit=round(math.sqrt(first.exit.take_profit * second.exit.take_profit), 6),
        stop_loss=round(math.sqrt(first.exit.stop_loss * second.exit.stop_loss), 6),
        horizon_bars=round((first.exit.horizon_bars + second.exit.horizon_bars) / 2),
        trail=first.exit.trail and second.exit.trail,
    )
    risk_rule = dataclasses.replace(
        first.risk,
        risk_per_trade=min(first.risk.risk_per_trade, second.risk.risk_per_trade),
        max_position_fraction=min(
            first.risk.max_position_fraction,
            second.risk.max_position_fraction,
            space.max_position_fraction,
        ),
        max_trades_per_day=space.max_trades_per_day,
    )
    base = dataclasses.replace(
        first,
        id="GENERATED_PENDING_ID",
        family="generated_crossover",
        idea=f"Autonomous crossover of two prior {space.opportunity_type} experiments.",
        market_logic=(
            "The child recombines independently testable stages from two compatible parents and "
            "must pass the full validation protocol from the beginning."
        ),
        regime=stages["regime"],
        setup=stages["setup"],
        trigger=stages["trigger"],
        exit=exit_rule,
        risk=risk_rule,
        tags=["autonomous_generation", "crossover", space.name],
    )
    child = mutate_hypothesis(
        base,
        space,
        parent_hash=parent_hashes[0],
        rng=rng,
        available_features=available_features,
        feedback_weights=feedback_weights,
        limits=limits,
    )
    return dataclasses.replace(
        child,
        generation_method="crossover",
        parent_hashes=tuple(parent_hashes),
    )


def predicate_token(predicate: Predicate, *, include_values: bool) -> tuple[Any, ...]:
    values: tuple[Any, ...] = ()
    if include_values:
        values = (
            _rounded(predicate.reference),
            predicate.feature_b,
            predicate.lookback,
            predicate.window,
            _rounded(predicate.quantile),
            _rounded(predicate.low),
            _rounded(predicate.high),
            predicate.shift_b,
        )
    return (predicate.timeframe, predicate.feature, predicate.op, *values)


def _rounded(value: float | None) -> float | None:
    return round(float(value), 8) if value is not None else None


def structural_tokens(hypothesis: Hypothesis, *, include_values: bool = False) -> set[str]:
    tokens: set[str] = {
        f"direction:{hypothesis.direction}",
        f"base:{hypothesis.base_timeframe}",
    }
    for stage in ("regime", "setup", "trigger"):
        for predicate in getattr(hypothesis, stage):
            tokens.add(f"{stage}:{predicate_token(predicate, include_values=include_values)!r}")
    if include_values:
        tokens.update(
            {
                f"tp:{round(hypothesis.exit.take_profit, 6)}",
                f"sl:{round(hypothesis.exit.stop_loss, 6)}",
                f"horizon:{hypothesis.exit.horizon_bars}",
                f"trail:{hypothesis.exit.trail}",
                *{
                    f"risk:{key}:{_rounded(value) if isinstance(value, float) else value}"
                    for key, value in dataclasses.asdict(hypothesis.risk).items()
                },
            }
        )
    return tokens


def structural_similarity(
    first: Hypothesis,
    second: Hypothesis,
    *,
    include_values: bool = False,
) -> float:
    a = structural_tokens(first, include_values=include_values)
    b = structural_tokens(second, include_values=include_values)
    return len(a & b) / len(a | b) if a or b else 1.0


def validate_hypothesis_against_space(
    hypothesis: Hypothesis,
    space: SearchSpace,
    *,
    available_features: Mapping[str, Iterable[str]] | None = None,
    limits: GrammarLimits | None = None,
) -> list[str]:
    limits = limits or GrammarLimits()
    problems: list[str] = []
    expected = {
        "base_timeframe": space.base_timeframe,
        "regime_timeframe": space.regime_timeframe,
        "setup_timeframe": space.setup_timeframe,
        "trigger_timeframe": space.trigger_timeframe,
    }
    for field, value in expected.items():
        if getattr(hypothesis, field) != value:
            problems.append(f"{field}_mismatch")
    if hypothesis.direction not in space.directions:
        problems.append("direction_not_allowed")
    if space.market == "spot" and space.pnl_unit == "btc" and hypothesis.direction != "short":
        problems.append("btc_accumulation_requires_short_dodge_signal")

    total = 0
    for stage in ("regime", "setup", "trigger"):
        predicates = list(getattr(hypothesis, stage))
        lo, hi = _stage_bounds(stage, limits)
        total += len(predicates)
        if not lo <= len(predicates) <= hi:
            problems.append(f"{stage}_predicate_count")
        expected_tf = _stage_timeframe(space, stage)
        seen = set()
        available = _features_for(available_features, expected_tf)
        for predicate in predicates:
            if predicate.timeframe != expected_tf:
                problems.append(f"{stage}_timeframe_mismatch")
            token = predicate_token(predicate, include_values=True)
            if token in seen:
                problems.append(f"{stage}_duplicate_predicate")
            seen.add(token)
            if available_features is not None:
                for feature in (predicate.feature, predicate.feature_b):
                    if feature is not None and feature not in available:
                        problems.append(f"missing_feature:{expected_tf}:{feature}")
    if total > limits.max_total_predicates:
        problems.append("total_predicate_limit")

    if not space.take_profit_range[0] <= hypothesis.exit.take_profit <= space.take_profit_range[1]:
        problems.append("take_profit_out_of_bounds")
    if not space.stop_loss_range[0] <= hypothesis.exit.stop_loss <= space.stop_loss_range[1]:
        problems.append("stop_loss_out_of_bounds")
    if not space.horizon_range[0] <= hypothesis.exit.horizon_bars <= space.horizon_range[1]:
        problems.append("horizon_out_of_bounds")
    if (
        not space.risk_per_trade_range[0]
        <= hypothesis.risk.risk_per_trade
        <= space.risk_per_trade_range[1]
    ):
        problems.append("risk_per_trade_out_of_bounds")
    if not 0 < hypothesis.risk.max_position_fraction <= space.max_position_fraction:
        problems.append("max_position_fraction_out_of_bounds")
    if hypothesis.risk.max_trades_per_day != space.max_trades_per_day:
        problems.append("max_trades_per_day_mismatch")
    if (
        hypothesis.risk.min_atr_pct is not None
        and hypothesis.risk.max_atr_pct is not None
        and hypothesis.risk.min_atr_pct >= hypothesis.risk.max_atr_pct
    ):
        problems.append("invalid_atr_band")
    return sorted(set(problems))
