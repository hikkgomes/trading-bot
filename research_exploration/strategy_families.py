"""Named market-behaviour families and their hypothesis builders.

Each family encodes a *thesis about why an edge could exist* and a template for
turning a concrete (regime, setup, trigger) timeframe choice + direction into a
fully-specified :class:`Hypothesis`. We never emit "all indicators x all
thresholds": every candidate belongs to one of these named behaviours.

Families
--------
A trend_continuation   HTF trend + MTF pullback + LTF re-entry trigger
B volatility_breakout  HTF/MTF compression + LTF range break + volume expansion
C mean_reversion       price stretched too far + reclaim trigger + tight invalidation
D momentum_continuation strong directional pressure + shallow pullback + fast entry
E liquidity_sweep      prior low/high swept then reclaimed, with HTF context
F regime_avoidance     NO-TRADE guards (chop / vol too low / vol too high)

All thresholds are either standard textbook levels (RSI 30/70, ADX ~22-25) or
*causal* rolling quantiles — nothing is fit on the whole sample, so a candidate
is honestly testable out-of-sample.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from research_exploration.hypothesis_schema import (
    ExitRule,
    Hypothesis,
    Predicate,
    RiskRule,
)

# Bars on the trigger TF that correspond to roughly a few hours of holding.
# Used to scale the time-stop sensibly per execution timeframe.
_HORIZON_BY_TF = {"1m": 90, "5m": 36, "15m": 24, "30m": 16, "1h": 12}


def _horizon(base_tf: str) -> int:
    return _HORIZON_BY_TF.get(base_tf, 24)


# --------------------------------------------------------------------------- #
# Family A: trend continuation (pullback re-entry)
# --------------------------------------------------------------------------- #
def trend_continuation(
    regime_tf: str, setup_tf: str, trigger_tf: str, direction: str, idx: int
) -> Hypothesis:
    long = direction == "long"
    regime = [
        Predicate(
            regime_tf,
            "ema_50",
            "gt_feature" if long else "lt_feature",
            feature_b="ema_200",
            note=f"{regime_tf} EMA50 {'>' if long else '<'} EMA200 (trend {'up' if long else 'down'})",
        ),
        Predicate(
            regime_tf, "adx_14", "ge", reference=20.0, note=f"{regime_tf} ADX>=20 (trend has force)"
        ),
    ]
    setup = [
        Predicate(
            setup_tf,
            "close",
            "lt_feature" if long else "gt_feature",
            feature_b="bbands_20_middleband",
            note=f"{setup_tf} pulled back {'below' if long else 'above'} 20-band mean",
        ),
        Predicate(
            setup_tf,
            "rsi_14",
            "le" if long else "ge",
            reference=48.0 if long else 52.0,
            note=f"{setup_tf} RSI cooled ({'<=48' if long else '>=52'})",
        ),
    ]
    trigger = [
        Predicate(
            trigger_tf,
            "macd_macd",
            "cross_above" if long else "cross_below",
            feature_b="macd_macdsignal",
            note=f"{trigger_tf} MACD {'bull' if long else 'bear'} cross",
        ),
        Predicate(
            trigger_tf,
            "volume_z_20",
            "ge",
            reference=0.5,
            note=f"{trigger_tf} volume above average (z>=0.5)",
        ),
    ]
    return Hypothesis(
        id=f"TRENDCONT_{direction.upper()}_{regime_tf}_{setup_tf}_{trigger_tf}_{idx:03d}",
        family="trend_continuation",
        idea=(
            f"In a {regime_tf} {'up' if long else 'down'}trend, {'buy' if long else 'sell'} {setup_tf} "
            f"pullbacks once {trigger_tf} momentum turns back {'up' if long else 'down'} on rising volume."
        ),
        market_logic=(
            "Trend-followers and dip buyers re-enter after controlled pullbacks; the higher "
            "timeframe trend stacks the odds, the trigger times the resumption."
        ),
        direction=direction,
        base_timeframe=trigger_tf,
        regime_timeframe=regime_tf,
        setup_timeframe=setup_tf,
        trigger_timeframe=trigger_tf,
        regime=regime,
        setup=setup,
        trigger=trigger,
        exit=ExitRule(
            take_profit=0.012,
            stop_loss=0.008,
            horizon_bars=_horizon(trigger_tf),
            atr_take_profit=1.8,
            atr_stop_loss=1.2,
        ),
        risk=RiskRule(
            risk_per_trade=0.01, max_trades_per_day=4, cooldown_bars=3, min_atr_pct=0.0010
        ),
        expected_holding="minutes to a few hours",
        expected_frequency="a few setups per day in trending regimes",
        invalidation=(
            f"{regime_tf} trend flips ({'EMA50<EMA200' if long else 'EMA50>EMA200'}) or the "
            f"{setup_tf} pullback deepens past the swing low/high (stop)."
        ),
        tags=["trend", "pullback", "continuation"],
    )


# --------------------------------------------------------------------------- #
# Family B: volatility breakout (compression -> expansion)
# --------------------------------------------------------------------------- #
def volatility_breakout(
    regime_tf: str, setup_tf: str, trigger_tf: str, direction: str, idx: int
) -> Hypothesis:
    long = direction == "long"
    regime = [
        Predicate(
            regime_tf,
            "natr_14",
            "q_le",
            quantile=0.35,
            window=180,
            note=f"{regime_tf} volatility compressed (NATR in bottom 35% of last 180 bars)",
        ),
    ]
    setup = [
        Predicate(
            setup_tf,
            "stddev_20",
            "q_le",
            quantile=0.40,
            window=120,
            note=f"{setup_tf} range tightening (stddev in bottom 40%)",
        ),
        Predicate(
            setup_tf,
            "adx_14",
            "le",
            reference=20.0,
            note=f"{setup_tf} not yet trending (ADX<=20) — coiled",
        ),
    ]
    trigger = [
        Predicate(
            trigger_tf,
            "close",
            "gt_feature" if long else "lt_feature",
            feature_b="max_20" if long else "min_20",
            shift_b=1,
            note=f"{trigger_tf} closes {'above the PRIOR 20-bar high' if long else 'below the PRIOR 20-bar low'} (range break)",
        ),
        Predicate(
            trigger_tf,
            "volume_z_20",
            "ge",
            reference=1.0,
            note=f"{trigger_tf} volume expansion (z>=1.0) confirms the break",
        ),
    ]
    return Hypothesis(
        id=f"VOLBREAK_{direction.upper()}_{regime_tf}_{setup_tf}_{trigger_tf}_{idx:03d}",
        family="volatility_breakout",
        idea=(
            f"After {regime_tf} volatility compresses and {setup_tf} range tightens, trade the first "
            f"{trigger_tf} {'upside' if long else 'downside'} break on a volume expansion."
        ),
        market_logic=(
            "Volatility mean-reverts: long quiet periods build up stop clusters and pent-up "
            "order flow that fuel a directional expansion when the range finally breaks."
        ),
        direction=direction,
        base_timeframe=trigger_tf,
        regime_timeframe=regime_tf,
        setup_timeframe=setup_tf,
        trigger_timeframe=trigger_tf,
        regime=regime,
        setup=setup,
        trigger=trigger,
        exit=ExitRule(
            take_profit=0.018,
            stop_loss=0.007,
            horizon_bars=_horizon(trigger_tf) + 12,
            atr_take_profit=2.5,
            atr_stop_loss=1.0,
        ),
        risk=RiskRule(risk_per_trade=0.01, max_trades_per_day=3, cooldown_bars=6),
        expected_holding="tens of minutes to a few hours",
        expected_frequency="a handful per week (compression is rare)",
        invalidation="price falls back inside the broken range within a few bars (failed break -> exit).",
        tags=["breakout", "squeeze", "volatility"],
    )


# --------------------------------------------------------------------------- #
# Family C: mean reversion (stretched + reclaim)
# --------------------------------------------------------------------------- #
def mean_reversion(
    regime_tf: str, setup_tf: str, trigger_tf: str, direction: str, idx: int
) -> Hypothesis:
    long = direction == "long"  # long = buy oversold dip inside an uptrend
    regime = [
        Predicate(
            regime_tf,
            "ema_50",
            "gt_feature" if long else "lt_feature",
            feature_b="ema_200",
            note=f"{regime_tf} trend still {'up' if long else 'down'} (only fade *with* the trend)",
        ),
    ]
    setup = [
        Predicate(
            setup_tf,
            "close",
            "lt_feature" if long else "gt_feature",
            feature_b="bbands_20_lowerband" if long else "bbands_20_upperband",
            note=f"{setup_tf} stretched outside the {'lower' if long else 'upper'} band",
        ),
        Predicate(
            setup_tf,
            "rsi_14",
            "le" if long else "ge",
            reference=25.0 if long else 75.0,
            note=f"{setup_tf} RSI {'<=25 oversold' if long else '>=75 overbought'}",
        ),
    ]
    trigger = [
        Predicate(
            trigger_tf,
            "rsi_14",
            "rising" if long else "falling",
            lookback=3,
            note=f"{trigger_tf} RSI turning back {'up' if long else 'down'} (reclaim begins)",
        ),
        Predicate(
            trigger_tf,
            "close",
            "gt_feature" if long else "lt_feature",
            feature_b="open",
            note=f"{trigger_tf} {'bullish' if long else 'bearish'} candle confirms the reclaim",
        ),
    ]
    return Hypothesis(
        id=f"MEANREV_{direction.upper()}_{regime_tf}_{setup_tf}_{trigger_tf}_{idx:03d}",
        family="mean_reversion",
        idea=(
            f"When {regime_tf} trend is intact but {setup_tf} stretches to a band extreme and "
            f"{trigger_tf} starts reclaiming, scalp the snap-back."
        ),
        market_logic=(
            "Sharp counter-trend flushes overshoot fair value; once forced selling/buying "
            "exhausts, price reverts to the mean. Only taken *with* the higher-TF trend."
        ),
        direction=direction,
        base_timeframe=trigger_tf,
        regime_timeframe=regime_tf,
        setup_timeframe=setup_tf,
        trigger_timeframe=trigger_tf,
        regime=regime,
        setup=setup,
        trigger=trigger,
        exit=ExitRule(
            take_profit=0.008,
            stop_loss=0.006,
            horizon_bars=max(8, _horizon(trigger_tf) // 2),
            atr_take_profit=1.2,
            atr_stop_loss=1.0,
        ),
        risk=RiskRule(
            risk_per_trade=0.0075, max_trades_per_day=5, cooldown_bars=4, max_atr_pct=0.02
        ),
        expected_holding="a few minutes to ~1 hour (fast scalp)",
        expected_frequency="several per week; clusters around volatility spikes",
        invalidation="price keeps trending through the band without reclaim (stop); or HTF trend flips.",
        tags=["mean_reversion", "scalp", "reclaim"],
    )


# --------------------------------------------------------------------------- #
# Family D: momentum continuation (strong pressure + shallow pullback + fast entry)
# --------------------------------------------------------------------------- #
def momentum_continuation(
    regime_tf: str, setup_tf: str, trigger_tf: str, direction: str, idx: int
) -> Hypothesis:
    long = direction == "long"
    regime = [
        Predicate(
            regime_tf,
            "close",
            "gt_feature" if long else "lt_feature",
            feature_b="ema_50",
            note=f"{regime_tf} price {'above' if long else 'below'} EMA50 (directional pressure)",
        ),
        Predicate(
            regime_tf, "adx_14", "ge", reference=25.0, note=f"{regime_tf} ADX>=25 (strong trend)"
        ),
    ]
    setup = [
        Predicate(
            setup_tf,
            "rsi_14",
            "between",
            low=45.0 if long else 30.0,
            high=70.0 if long else 55.0,
            note=f"{setup_tf} shallow pullback (RSI in continuation zone, not exhausted)",
        ),
        Predicate(
            setup_tf,
            "close",
            "gt_feature" if long else "lt_feature",
            feature_b="ema_20",
            note=f"{setup_tf} still {'above' if long else 'below'} EMA20",
        ),
    ]
    trigger = [
        Predicate(
            trigger_tf,
            "mom_10",
            "slope_up" if long else "slope_down",
            lookback=3,
            reference=0.0,
            note=f"{trigger_tf} momentum re-accelerating",
        ),
        Predicate(
            trigger_tf,
            "volume_z_20",
            "ge",
            reference=0.5,
            note=f"{trigger_tf} participation picking up",
        ),
    ]
    return Hypothesis(
        id=f"MOMCONT_{direction.upper()}_{regime_tf}_{setup_tf}_{trigger_tf}_{idx:03d}",
        family="momentum_continuation",
        idea=(
            f"Strong {regime_tf} momentum + a shallow {setup_tf} pause + a {trigger_tf} re-acceleration "
            f"= continuation entry in the trend direction."
        ),
        market_logic=(
            "Strong trends pause shallowly before continuing as latecomers and trend-followers "
            "keep adding; momentum that refuses to give back much rarely reverses immediately."
        ),
        direction=direction,
        base_timeframe=trigger_tf,
        regime_timeframe=regime_tf,
        setup_timeframe=setup_tf,
        trigger_timeframe=trigger_tf,
        regime=regime,
        setup=setup,
        trigger=trigger,
        exit=ExitRule(
            take_profit=0.010,
            stop_loss=0.006,
            horizon_bars=_horizon(trigger_tf),
            atr_take_profit=1.6,
            atr_stop_loss=1.0,
        ),
        risk=RiskRule(risk_per_trade=0.01, max_trades_per_day=5, cooldown_bars=2),
        expected_holding="minutes to ~2 hours",
        expected_frequency="multiple per day in strong trends, none in chop",
        invalidation="pullback deepens past EMA20 / momentum rolls over before continuation (stop).",
        tags=["momentum", "continuation", "trend"],
    )


# --------------------------------------------------------------------------- #
# Family E: liquidity sweep / failed breakdown
# --------------------------------------------------------------------------- #
def liquidity_sweep(
    regime_tf: str, setup_tf: str, trigger_tf: str, direction: str, idx: int
) -> Hypothesis:
    long = direction == "long"  # long = sweep below support then reclaim
    regime = [
        Predicate(
            regime_tf,
            "ema_50",
            "gt_feature" if long else "lt_feature",
            feature_b="ema_200",
            note=f"{regime_tf} trend {'up' if long else 'down'} — fade the sweep *with* the trend",
        ),
    ]
    setup = [
        Predicate(
            setup_tf,
            "low" if long else "high",
            "lt_feature" if long else "gt_feature",
            feature_b="min_50" if long else "max_50",
            shift_b=1,
            note=f"{setup_tf} {'pierces the PRIOR 50-bar low' if long else 'pierces the PRIOR 50-bar high'} (stop run)",
        ),
    ]
    trigger = [
        Predicate(
            trigger_tf,
            "close",
            "gt_feature" if long else "lt_feature",
            feature_b="min_20" if long else "max_20",
            shift_b=1,
            note=f"{trigger_tf} reclaims back {'above the PRIOR 20-bar low' if long else 'below the PRIOR 20-bar high'}",
        ),
        Predicate(
            trigger_tf,
            "rsi_14",
            "rising" if long else "falling",
            lookback=2,
            note=f"{trigger_tf} momentum flips after the sweep",
        ),
        Predicate(
            trigger_tf,
            "volume_z_20",
            "ge",
            reference=0.8,
            note=f"{trigger_tf} reclaim on real volume",
        ),
    ]
    return Hypothesis(
        id=f"SWEEP_{direction.upper()}_{regime_tf}_{setup_tf}_{trigger_tf}_{idx:03d}",
        family="liquidity_sweep",
        idea=(
            f"A {setup_tf} stop-run through prior {'lows' if long else 'highs'} that immediately reclaims "
            f"on {trigger_tf}, inside a {regime_tf} {'up' if long else 'down'}trend."
        ),
        market_logic=(
            "Resting stops below obvious lows (above highs) are liquidity. Price spikes to grab "
            "them, then snaps back; the failed breakdown traps breakout sellers and fuels reversal."
        ),
        direction=direction,
        base_timeframe=trigger_tf,
        regime_timeframe=regime_tf,
        setup_timeframe=setup_tf,
        trigger_timeframe=trigger_tf,
        regime=regime,
        setup=setup,
        trigger=trigger,
        exit=ExitRule(
            take_profit=0.012,
            stop_loss=0.006,
            horizon_bars=_horizon(trigger_tf),
            atr_take_profit=2.0,
            atr_stop_loss=0.9,
        ),
        risk=RiskRule(risk_per_trade=0.0075, max_trades_per_day=4, cooldown_bars=4),
        expected_holding="minutes to ~2 hours",
        expected_frequency="a few per week around obvious S/R levels",
        invalidation="no reclaim — price holds beyond the swept level (genuine breakout, stop out).",
        tags=["liquidity", "sweep", "reversal", "failed_break"],
    )


# --------------------------------------------------------------------------- #
# Family F: regime avoidance (NO-TRADE guards, not standalone entries)
# --------------------------------------------------------------------------- #
def no_trade_guard_predicates(
    tf: str,
    *,
    chop: bool = True,
    vol_floor: bool = True,
    vol_ceiling: bool = True,
    adx_min: float = 18.0,
    vol_window: int = 240,
) -> list[Predicate]:
    """Family F as *keep-form* predicates — the conditions that must hold for a
    trade to be allowed at all. AND-ing these onto a hypothesis vetoes the hostile
    regimes where edges usually evaporate (chop / dead vol / chaos).

    The grammar has no NOT, so each guard is phrased positively as the *keep*
    condition: anti-chop == ``ADX >= adx_min``; 'not too quiet' == NATR above its
    rolling 10th percentile (range big enough to pay fees); 'not too wild' == NATR
    below its rolling 95th percentile (stops/slippage still survivable). The two
    NATR quantile predicates together express a causal volatility *band*."""
    preds: list[Predicate] = []
    if chop:
        preds.append(
            Predicate(
                tf,
                "adx_14",
                "ge",
                reference=adx_min,
                note=f"{tf} ADX>={adx_min:g} — enough directionality to bother (anti-chop)",
            )
        )
    if vol_floor:
        preds.append(
            Predicate(
                tf,
                "natr_14",
                "q_ge",
                quantile=0.10,
                window=vol_window,
                note=f"{tf} NATR above its rolling bottom-10% — not dead (unpayable after fees)",
            )
        )
    if vol_ceiling:
        preds.append(
            Predicate(
                tf,
                "natr_14",
                "q_le",
                quantile=0.95,
                window=vol_window,
                note=f"{tf} NATR below its rolling top-5% — not chaotic (stop/slippage risk)",
            )
        )
    return preds


def apply_no_trade_guards(hyp: Hypothesis, guard_tf: str | None = None, **kw) -> Hypothesis:
    """Return a copy of ``hyp`` with Family-F no-trade guards AND-ed onto its
    regime stage. This is how family F ("regime avoidance") is actually expressed:
    not as standalone entries but as keep-conditions layered on A–E candidates.

    ``guard_tf`` defaults to the hypothesis's regime timeframe (the natural "are we
    allowed to trade this regime at all?" level). Extra kwargs pass through to
    :func:`no_trade_guard_predicates` (e.g. ``chop=False`` to drop the ADX guard)."""
    tf = guard_tf or hyp.regime_timeframe
    guards = no_trade_guard_predicates(tf, **kw)
    if not guards:
        return hyp
    return Hypothesis(
        id=f"{hyp.id}_G",
        family=hyp.family,
        idea=hyp.idea + " [+regime-avoidance guards]",
        market_logic=hyp.market_logic,
        direction=hyp.direction,
        base_timeframe=hyp.base_timeframe,
        regime_timeframe=hyp.regime_timeframe,
        setup_timeframe=hyp.setup_timeframe,
        trigger_timeframe=hyp.trigger_timeframe,
        regime=[*hyp.regime, *guards],
        setup=list(hyp.setup),
        trigger=list(hyp.trigger),
        exit=hyp.exit,
        risk=hyp.risk,
        expected_holding=hyp.expected_holding,
        expected_frequency=hyp.expected_frequency,
        invalidation=hyp.invalidation
        + " Guards also stand the strategy aside in chop / dead / chaotic vol.",
        tags=[*hyp.tags, "regime_avoidance", "guarded"],
    )


@dataclass
class Family:
    key: str
    title: str
    thesis: str
    builder: Callable[..., Hypothesis]
    default_tf_combos: list[tuple]  # (regime, setup, trigger)
    directions: tuple = ("long", "short")


FAMILIES: dict[str, Family] = {
    "trend_continuation": Family(
        "trend_continuation",
        "A. Trend continuation (pullback re-entry)",
        "HTF trend + MTF pullback + LTF momentum re-entry.",
        trend_continuation,
        [("1d", "4h", "30m"), ("4h", "1h", "15m"), ("4h", "30m", "5m"), ("1h", "30m", "5m")],
    ),
    "volatility_breakout": Family(
        "volatility_breakout",
        "B. Volatility breakout (compression -> expansion)",
        "HTF/MTF volatility compression + LTF range break + volume expansion.",
        volatility_breakout,
        [("4h", "1h", "15m"), ("4h", "30m", "5m"), ("1h", "30m", "5m"), ("1h", "15m", "1m")],
    ),
    "mean_reversion": Family(
        "mean_reversion",
        "C. Mean reversion (stretched + reclaim)",
        "Price stretched to a band extreme + reclaim trigger, with the HTF trend.",
        mean_reversion,
        [("4h", "15m", "5m"), ("4h", "30m", "5m"), ("1h", "15m", "1m"), ("1h", "15m", "5m")],
    ),
    "momentum_continuation": Family(
        "momentum_continuation",
        "D. Momentum continuation (shallow pullback)",
        "Strong directional pressure + shallow pullback + fast continuation entry.",
        momentum_continuation,
        [("4h", "15m", "5m"), ("1h", "15m", "5m"), ("1h", "15m", "1m"), ("30m", "5m", "1m")],
    ),
    "liquidity_sweep": Family(
        "liquidity_sweep",
        "E. Liquidity sweep / failed breakdown",
        "Prior low/high swept then reclaimed, inside an HTF trend.",
        liquidity_sweep,
        [("4h", "30m", "5m"), ("4h", "15m", "5m"), ("1h", "15m", "1m"), ("1h", "30m", "5m")],
    ),
}


def build_family(key: str, direction: str, combo: tuple, idx: int) -> Hypothesis:
    fam = FAMILIES[key]
    regime_tf, setup_tf, trigger_tf = combo
    return fam.builder(regime_tf, setup_tf, trigger_tf, direction, idx)
