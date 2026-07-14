"""Smoke + unit tests for the exploratory research workflow.

These are deliberately lightweight: schema round-trips, predicate semantics, a
small engineered backtest that actually produces trades, the experiment log, and
a metadata-only inventory check (skipped if the parquets aren't present). No
heavy data is loaded.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_exploration import feature_inventory as fi
from research_exploration.evaluate import (
    EvalConfig,
    build_synthetic_aligned_frame,
    effective_rolling_window,
    entry_mask,
    evaluate_hypothesis,
    hypothesis_trades,
    predicate_mask,
    signals_from_hypothesis,
)
from research_exploration.experiment_log import (
    ExperimentRecord,
    already_tested,
    fingerprint,
    load_log,
    log_result,
)
from research_exploration.hypothesis_generator import (
    first_smoke_set,
    generate_batch,
    load_batch,
    position_trading_set,
    write_batch,
)
from research_exploration.hypothesis_schema import (
    ExitRule,
    Hypothesis,
    Predicate,
    RiskRule,
)
from research_exploration.predicate_funnel import (
    REJECTION_REASONS,
    FunnelStep,
    HypothesisFunnel,
    PredicateCoverage,
    analyze_hypothesis,
    classify,
)
from research_exploration.strategy_families import (
    FAMILIES,
    apply_no_trade_guards,
    no_trade_guard_predicates,
)


# --------------------------------------------------------------------------- #
# Feature inventory
# --------------------------------------------------------------------------- #
def test_classify_column():
    cases = {
        "close": "raw_ohlcv",
        "rsi_14": "momentum",
        "ema_50": "trend_ma",
        "adx_14": "trend_dmi",
        "atr_14": "volatility",
        "bbands_20_upperband": "volatility",
        "max_20": "range_extrema",
        "cvd_20": "orderflow",
        "taker_imbalance_ma_20": "orderflow",
        "cdlhammer": "candlestick",
        "ht_dcperiod": "cycle_hilbert",
        "linearreg_slope_50": "statistic",
        "obv": "volume",
        "typprice": "price_transform",
        "sin": "math_scalar",
        "timestamp": "time",
    }
    for col, fam in cases.items():
        assert fi.classify_column(col) == fam, col


def test_feature_root():
    assert fi.feature_root("rsi_14") == "rsi"
    assert fi.feature_root("bbands_20_upperband") == "bbands_20_upperband"  # only trailing _<num>
    assert fi.feature_root("ema") == "ema"


@pytest.mark.skipif(
    not (fi.INDICATOR_DATA_DIR / "BTCUSDT_5m_all_indicators.parquet").exists(),
    reason="indicator parquets not present",
)
def test_inventory_metadata_only():
    inv = fi.build_inventory(timeframes=("5m", "1m", "1d"))
    assert inv["5m"].exists and inv["5m"].num_columns > 700
    assert inv["5m"].orderflow_available is True
    assert inv["1m"].orderflow_available is any(
        fi.classify_column(column) == "orderflow" for column in inv["1m"].columns
    )
    assert inv["5m"].num_rows > inv["1d"].num_rows


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_predicate_validation():
    with pytest.raises(ValueError):
        Predicate("5m", "rsi_14", "not_an_op")
    with pytest.raises(ValueError):
        Predicate("7m", "rsi_14", "gt", reference=1)


def test_hypothesis_timeframe_ordering_enforced():
    good = FAMILIES["trend_continuation"].builder("4h", "30m", "5m", "long", 1)
    assert good.base_timeframe == "5m"
    with pytest.raises(ValueError):
        # regime finer than setup -> invalid
        Hypothesis(
            id="bad",
            family="x",
            idea="",
            market_logic="",
            direction="long",
            base_timeframe="5m",
            regime_timeframe="5m",
            setup_timeframe="30m",
            trigger_timeframe="5m",
            regime=[],
            setup=[],
            trigger=[],
            exit=ExitRule(0.01, 0.01, 10),
        )


def test_hypothesis_roundtrip():
    h = FAMILIES["volatility_breakout"].builder("4h", "30m", "5m", "short", 3)
    d = h.to_dict()
    h2 = Hypothesis.from_dict(d)
    assert h2.to_dict() == d
    assert set(h.feature_columns()) == set(h2.feature_columns())
    assert all(c.startswith("tf_") for c in h.feature_columns())


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def test_generate_batch_is_valid_and_diverse():
    hyps = generate_batch()
    assert len(hyps) >= 20
    assert {h.family for h in hyps} == set(FAMILIES)
    # uses multiple timeframes, never collapses to a single tf
    for h in hyps:
        assert len(h.timeframes()) >= 2
    # ids unique
    assert len({h.id for h in hyps}) == len(hyps)


def test_batch_file_roundtrip(tmp_path: Path):
    hyps = first_smoke_set()
    path = tmp_path / "batch.json"
    write_batch(hyps, path)
    loaded = load_batch(path)
    assert [h.to_dict() for h in loaded] == [h.to_dict() for h in hyps]


# --------------------------------------------------------------------------- #
# Family F: regime-avoidance guards (no-trade conditions)
# --------------------------------------------------------------------------- #
def test_no_trade_guard_predicates_are_keep_form():
    preds = no_trade_guard_predicates("4h")
    assert len(preds) == 3
    assert all(p.timeframe == "4h" for p in preds)
    ops = {p.feature: p.op for p in preds}
    assert ops["adx_14"] == "ge"  # anti-chop keep-form
    assert ops["natr_14"] in {"q_ge", "q_le"}  # vol band via rolling quantiles
    # toggling parts off works
    assert len(no_trade_guard_predicates("1d", chop=False)) == 2


def test_apply_guards_augments_regime_and_roundtrips():
    base = FAMILIES["trend_continuation"].builder("4h", "1h", "15m", "long", 1)
    guarded = apply_no_trade_guards(base)
    assert guarded.id.endswith("_G")
    assert len(guarded.regime) == len(base.regime) + 3
    assert "regime_avoidance" in guarded.tags
    # default guard timeframe is the regime timeframe (no new timeframe introduced)
    assert guarded.timeframes() == base.timeframes()
    # still a valid hypothesis and round-trips through JSON
    d = guarded.to_dict()
    assert Hypothesis.from_dict(d).to_dict() == d


def test_guards_only_make_entries_stricter():
    """A guarded candidate can only ever fire on a subset of the unguarded bars."""
    base = FAMILIES["momentum_continuation"].builder("4h", "15m", "5m", "long", 1)
    guarded = apply_no_trade_guards(base)
    frame = build_synthetic_aligned_frame([base, guarded], n=3000)
    m_base = entry_mask(frame, base).to_numpy()
    m_guard = entry_mask(frame, guarded).to_numpy()
    assert m_guard.sum() <= m_base.sum()
    assert bool((m_guard & ~m_base).any()) is False  # guarded ⊆ unguarded


def test_generate_batch_with_guards_is_stricter_variant():
    plain = generate_batch()
    guarded = generate_batch(with_guards=True)
    assert len(plain) == len(guarded)  # same candidates, guarded
    assert all(h.id.endswith("_G") for h in guarded)
    assert all(len(g.regime) >= len(p.regime) for p, g in zip(plain, guarded, strict=False))


def test_position_trading_set_guarded_variants_stay_btc_short_side():
    plain = position_trading_set()
    guarded = position_trading_set(with_guards=True)

    assert len(plain) == len(guarded)
    assert {h.direction for h in guarded} == {"short"}
    assert {h.base_timeframe for h in guarded} == {"1h", "4h"}
    assert all(h.id.startswith("POS_") and h.id.endswith("_G") for h in guarded)
    assert all("position_btc" in h.tags and "regime_avoidance" in h.tags for h in guarded)
    assert all(len(g.regime) > len(p.regime) for p, g in zip(plain, guarded, strict=False))


# --------------------------------------------------------------------------- #
# Predicate semantics (causal masks on a hand frame)
# --------------------------------------------------------------------------- #
def test_predicate_masks_semantics():
    frame = pd.DataFrame(
        {
            "tf_5m_close": [1.0, 2.0, 3.0, 2.0, 1.0],
            "tf_5m_ema_20": [2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    # cross_above: close goes from <=ema to >ema at index 2
    ca = predicate_mask(frame, Predicate("5m", "close", "cross_above", feature_b="ema_20"))
    assert list(ca) == [False, False, True, False, False]
    # rising over 1 bar
    rs = predicate_mask(frame, Predicate("5m", "close", "rising", lookback=1))
    assert list(rs.fillna(False)) == [False, True, True, False, False]
    # gt reference
    gt = predicate_mask(frame, Predicate("5m", "close", "gt", reference=2.0))
    assert list(gt) == [False, False, True, False, False]


# --------------------------------------------------------------------------- #
# Backtest path produces real trades with the canonical model
# --------------------------------------------------------------------------- #
def _oscillating_frame(n=600):
    t = np.arange(n)
    close = 100 + 5 * np.sin(t / 8.0)  # smooth oscillation -> many crosses
    ema = pd.Series(close).rolling(10, min_periods=1).mean().to_numpy()
    o = np.concatenate([[close[0]], close[:-1]])
    hi = np.maximum(o, close) + 0.5
    lo = np.minimum(o, close) - 0.5
    return pd.DataFrame(
        {
            "tf_5m_open": o,
            "tf_5m_high": hi,
            "tf_5m_low": lo,
            "tf_5m_close": close,
            "tf_5m_ema_20": ema,
        }
    )


def _trigger_only_hypothesis():
    return Hypothesis(
        id="SMOKE_LONG_001",
        family="trend_continuation",
        idea="x",
        market_logic="x",
        direction="long",
        base_timeframe="5m",
        regime_timeframe="5m",
        setup_timeframe="5m",
        trigger_timeframe="5m",
        regime=[],
        setup=[],
        trigger=[Predicate("5m", "close", "cross_above", feature_b="ema_20")],
        exit=ExitRule(take_profit=0.02, stop_loss=0.02, horizon_bars=12),
        risk=RiskRule(),
    )


def test_evaluate_produces_trades_and_metrics():
    frame = _oscillating_frame()
    hyp = _trigger_only_hypothesis()
    m = evaluate_hypothesis(frame, hyp, EvalConfig(min_trades=5))
    assert m["trades"] > 0
    for key in (
        "win_rate",
        "total_return",
        "sharpe",
        "psr",
        "max_drawdown",
        "by_year",
        "exit_reasons",
    ):
        assert key in m
    assert 0.0 <= m["win_rate"] <= 1.0


def test_entry_is_next_bar_no_lookahead():
    frame = _oscillating_frame()
    hyp = _trigger_only_hypothesis()
    sig = signals_from_hypothesis(frame, hyp)
    from src.strategies.backtester import _simulate
    from src.strategies.base import BacktestConfig

    o = frame["tf_5m_open"].to_numpy(float)
    h = frame["tf_5m_high"].to_numpy(float)
    lo = frame["tf_5m_low"].to_numpy(float)
    c = frame["tf_5m_close"].to_numpy(float)
    trades = pd.DataFrame(
        _simulate(
            o,
            h,
            lo,
            c,
            sig.to_numpy(),
            frame.index,
            BacktestConfig(take_profit=0.02, stop_loss=0.02, horizon_bars=12),
        )
    )
    assert not trades.empty
    # every entry is exactly one bar after its signal bar (next-bar-open fill)
    assert (trades["entry_time"] - trades["signal_time"] == 1).all()


def test_synthetic_aligned_frame_has_all_columns():
    hyps = first_smoke_set()
    frame = build_synthetic_aligned_frame(hyps, n=500)
    for h in hyps:
        for col in h.feature_columns():
            assert col in frame.columns, col
        # base OHLC resolvable
        for f in ("open", "high", "low", "close"):
            assert f"tf_{h.base_timeframe}_{f}" in frame.columns


# --------------------------------------------------------------------------- #
# Experiment log
# --------------------------------------------------------------------------- #
def test_experiment_log_roundtrip(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    h = first_smoke_set()[0]
    cfg = {"fee_bps": 5, "slippage_bps": 2}
    fp = fingerprint(h.to_dict(), cfg)
    rec = ExperimentRecord(
        hypothesis_id=h.id,
        family=h.family,
        direction=h.direction,
        fingerprint=fp,
        verdict="reject",
        metrics={"trades": 12, "sharpe": -0.2},
        config=cfg,
        hypothesis=h.to_dict(),
    )
    assert not already_tested(fp, log)
    log_result(rec, log)
    assert already_tested(fp, log)
    rows = load_log(log)
    assert len(rows) == 1 and rows[0]["hypothesis_id"] == h.id
    # fingerprint is stable for identical inputs
    assert fingerprint(h.to_dict(), cfg) == fp


def test_invalid_verdict_rejected():
    with pytest.raises(ValueError):
        ExperimentRecord(
            hypothesis_id="x", family="f", direction="long", fingerprint="abc", verdict="banana"
        )


# --------------------------------------------------------------------------- #
# Semantic fixes: prior-range breakouts + timeframe-aware windows
# --------------------------------------------------------------------------- #
def test_breakout_and_sweep_predicates_use_prior_range_shift():
    vb = FAMILIES["volatility_breakout"].builder("4h", "30m", "5m", "long", 1)
    brk = [p for p in vb.trigger if p.feature_b in ("max_20", "min_20")][0]
    assert brk.shift_b == 1, "breakout trigger must reference the PRIOR range"

    sw = FAMILIES["liquidity_sweep"].builder("4h", "30m", "5m", "long", 1)
    assert [p for p in sw.setup if p.feature_b in ("min_50", "max_50")][0].shift_b == 1
    assert [p for p in sw.trigger if p.feature_b in ("min_20", "max_20")][0].shift_b == 1

    # plain feature comparisons (EMA stack) must NOT be shifted
    tc = FAMILIES["trend_continuation"].builder("4h", "1h", "15m", "long", 1)
    assert [p for p in tc.regime if p.feature_b == "ema_200"][0].shift_b == 0


def test_breakout_threshold_excludes_current_bar():
    n = 30
    close = np.full(n, 100.0)
    close[20] = 110.0  # single spike == the all-time high at that bar
    frame = pd.DataFrame(
        {
            "tf_5m_close": close,
            "tf_5m_max_10": pd.Series(close).rolling(10, min_periods=1).max().to_numpy(),
        }
    )
    # prior-range breakout fires at the spike (110 > prior max 100)
    prior = predicate_mask(
        frame, Predicate("5m", "close", "gt_feature", feature_b="max_10", shift_b=1)
    ).fillna(False)
    assert bool(prior.iloc[20]) is True
    # current-bar-inclusive comparison is the old tautology: close == max -> never True
    incl = predicate_mask(frame, Predicate("5m", "close", "gt_feature", feature_b="max_10")).fillna(
        False
    )
    assert bool(incl.iloc[20]) is False
    # the threshold actually used at the spike is the PRIOR max (100), not the spike (110)
    assert frame["tf_5m_max_10"].shift(1).iloc[20] == 100.0


def test_effective_window_scales_to_native_htf_bars():
    p = Predicate("4h", "natr_14", "q_le", quantile=0.35, window=180)
    assert effective_rolling_window(p, "5m") == 180 * (14400 // 300)  # 8640 base rows
    assert effective_rolling_window(p, "4h") == 180  # same tf -> unscaled
    assert effective_rolling_window(p, None) == 180  # no base -> unscaled
    # finer base -> larger scaled window; coarser predicate tf -> bigger ratio
    p15 = Predicate("1h", "natr_14", "q_le", quantile=0.4, window=120)
    assert effective_rolling_window(p15, "15m") == 120 * 4


def test_funnel_flags_invalid_timeframe_window_when_sample_too_short():
    vb = FAMILIES["volatility_breakout"].builder("4h", "30m", "5m", "long", 1)
    # 4h q_le(window=180) on 5m base => 8640 base rows; a 200-row sample can't hold it
    frame = build_synthetic_aligned_frame([vb], n=200)
    f = analyze_hypothesis(frame, vb, EvalConfig())
    assert f.rejection_reason == "invalid_timeframe_window"
    assert f.killer_predicate  # the offending q-window predicate is named


def test_funnel_runs_on_all_smoke_hypotheses_synthetic():
    hyps = first_smoke_set()
    frame = build_synthetic_aligned_frame(hyps, n=3000)
    for h in hyps:
        f = analyze_hypothesis(frame, h, EvalConfig())
        assert f.rejection_reason in REJECTION_REASONS


# --------------------------------------------------------------------------- #
# Predicate funnel
# --------------------------------------------------------------------------- #
def _blank_funnel(total=100):
    return HypothesisFunnel(
        hypothesis_id="H",
        family="f",
        direction="long",
        base_tf="5m",
        regime_tf="4h",
        setup_tf="15m",
        trigger_tf="5m",
        total_rows=total,
    )


def test_classify_regime_never_fires():
    f = _blank_funnel()
    f.stage_combined = {"regime": 0, "setup": 50, "trigger": 50}
    f.per_predicate = [PredicateCoverage("regime", "4h impossible", "tf_4h_x", 0, 100, 100)]
    classify(f, EvalConfig())
    assert f.rejection_reason == "regime_never_fires"
    assert f.killer_kind == "standalone_zero"
    assert f.killer_stage == "regime"


def test_classify_combination_too_strict():
    f = _blank_funnel()
    f.stage_combined = {"regime": 40, "setup": 30, "trigger": 20}
    f.risk_filtered_rows = 0
    f.per_predicate = [PredicateCoverage("setup", "setup pred", "tf_15m_x", 30, 100, 100)]
    f.cumulative = [
        FunnelStep("regime", "regime pred", 40, 100, 40),
        FunnelStep("setup", "setup pred", 30, 40, 0),
    ]
    classify(f, EvalConfig())
    assert f.rejection_reason == "combination_too_strict"
    assert f.killer_predicate == "setup pred"  # first step that hit zero
    assert f.killer_kind == "combination"  # it fired 30x alone, lethal only combined


def test_classify_enough_signals_but_no_trades():
    f = _blank_funnel()
    f.stage_combined = {"regime": 40, "setup": 30, "trigger": 20}
    f.risk_filtered_rows = 15
    f.trades = 0
    classify(f, EvalConfig())
    assert f.rejection_reason == "enough_signals_but_no_trades"


def _traded_funnel(trades, avg_net, total_ret, signals=100):
    f = _blank_funnel()
    f.stage_combined = {"regime": 40, "setup": 30, "trigger": 20}
    f.cumulative = [FunnelStep("setup", "s", 30, 40, 20)]
    f.risk_filtered_rows = signals
    f.trades = trades
    f.avg_net_return = avg_net
    f.total_return = total_ret
    return f


def test_classify_too_few_vs_negative_vs_positive():
    cfg = EvalConfig(min_trades=30)

    too_few = _traded_funnel(trades=5, avg_net=0.01, total_ret=0.05, signals=25)
    classify(too_few, cfg)
    assert too_few.rejection_reason == "too_few_trades"

    negative = _traded_funnel(trades=60, avg_net=-0.001, total_ret=-0.1)
    classify(negative, cfg)
    assert negative.rejection_reason == "trades_exist_but_negative_expectancy"

    positive = _traded_funnel(trades=60, avg_net=0.002, total_ret=0.1)
    classify(positive, cfg)
    assert positive.rejection_reason == "candidate_positive"


def test_funnel_missing_columns_is_graceful():
    frame = _oscillating_frame()  # has only tf_5m_* cols
    hyp = Hypothesis(
        id="MISS",
        family="f",
        idea="x",
        market_logic="x",
        direction="long",
        base_timeframe="5m",
        regime_timeframe="4h",
        setup_timeframe="5m",
        trigger_timeframe="5m",
        regime=[Predicate("4h", "ema_999", "gt", reference=0.0)],  # column not in frame
        setup=[],
        trigger=[Predicate("5m", "close", "gt", reference=0.0)],
        exit=ExitRule(0.02, 0.02, 12),
    )
    f = analyze_hypothesis(frame, hyp, EvalConfig())
    assert f.rejection_reason == "missing_columns"
    assert "tf_4h_ema_999" in f.missing_columns


def test_funnel_end_to_end_counts_and_killer():
    frame = _oscillating_frame()
    hyp = _trigger_only_hypothesis()
    f = analyze_hypothesis(frame, hyp, EvalConfig(min_trades=5))
    assert f.total_rows == len(frame)
    # the single trigger predicate's standalone coverage matches a direct mask
    direct = int(predicate_mask(frame, hyp.trigger[0]).fillna(False).sum())
    cov = [c for c in f.per_predicate if c.stage == "trigger"][0]
    assert cov.true_rows == direct
    assert f.trades > 0
    assert f.rejection_reason in {
        "too_few_trades",
        "trades_exist_but_negative_expectancy",
        "candidate_positive",
    }
    # a diagnosis was produced: a killer for rejects, or a clean positive with no killer
    assert f.killer_predicate or f.rejection_reason == "candidate_positive"


# --------------------------------------------------------------------------- #
# Validation harness: splits, regimes, sensitivity, OOS windows, gating holdout
# --------------------------------------------------------------------------- #
from research_exploration.validation import (  # noqa: E402
    ValidationConfig,
    _deflated_sharpe_evidence,
    execution_data_stress_check,
    oos_window_stats,
    perturbed_variants,
    regime_breakdown,
    sensitivity_check,
    split_frame,
    tag_regimes,
    validate_batch,
    validate_hypothesis,
)


def _sawtooth(n: int, up: float, down: float, start: float = 100.0) -> np.ndarray:
    """Deterministic alternating up/down closes (net drift = (1+up)(1-down) per pair)."""
    c = [start]
    for i in range(n - 1):
        c.append(c[-1] * (1 + (up if i % 2 == 0 else -down)))
    return np.array(c)


def _ts_frame(closes: np.ndarray) -> pd.DataFrame:
    """5m OHLC frame with a real timestamp column (exercises the real-data path)."""
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(c), freq="5min", tz="UTC"),
            "tf_5m_open": o,
            "tf_5m_high": np.maximum(o, c) * 1.001,
            "tf_5m_low": np.minimum(o, c) * 0.999,
            "tf_5m_close": c,
        }
    )


def _rising_hypothesis(tp=0.02, sl=0.02, horizon=12) -> Hypothesis:
    return Hypothesis(
        id="VAL_LONG_001",
        family="momentum_continuation",
        idea="x",
        market_logic="x",
        direction="long",
        base_timeframe="5m",
        regime_timeframe="5m",
        setup_timeframe="5m",
        trigger_timeframe="5m",
        regime=[],
        setup=[],
        trigger=[Predicate("5m", "close", "rising", lookback=1)],
        exit=ExitRule(take_profit=tp, stop_loss=sl, horizon_bars=horizon),
        risk=RiskRule(),
    )


def _always_long_hypothesis(risk: RiskRule) -> Hypothesis:
    return Hypothesis(
        id="ALWAYS_LONG",
        family="risk_contract",
        idea="exercise stateful risk gates",
        market_logic="always true entry on engineered candles",
        direction="long",
        base_timeframe="5m",
        regime_timeframe="5m",
        setup_timeframe="5m",
        trigger_timeframe="5m",
        regime=[],
        setup=[],
        trigger=[Predicate("5m", "close", "ge", reference=1.0)],
        exit=ExitRule(take_profit=0.01, stop_loss=0.01, horizon_bars=1),
        risk=risk,
    )


def _forced_stop_frame(rows: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC"),
            "tf_5m_open": 100.0,
            "tf_5m_high": 101.0,
            "tf_5m_low": 98.0,
            "tf_5m_close": 100.0,
        }
    )


_RELAXED = ValidationConfig(
    min_trades_train=10,
    min_trades_val=3,
    min_trades_holdout=3,
    oos_windows=4,
    regime_lookback_bars=100,
)


def test_split_frame_is_chronological_and_disjoint():
    frame = _ts_frame(_sawtooth(1000, 0.015, 0.005))
    segs = split_frame(frame, _RELAXED)
    assert len(segs["train"]) == 600 and len(segs["validation"]) == 200
    assert len(segs["holdout"]) == 200
    # contiguous & ordered: train ends where validation starts, etc.
    assert segs["train"].index[-1] + 1 == segs["validation"].index[0]
    assert segs["validation"].index[-1] + 1 == segs["holdout"].index[0]
    t = pd.to_datetime(frame["timestamp"])
    assert t.iloc[599] < t.iloc[600] < t.iloc[800]


def test_hypothesis_trades_apply_live_sizing_daily_stop_and_effective_risk():
    hyp = _always_long_hypothesis(
        RiskRule(
            risk_per_trade=0.005,
            max_position_fraction=0.25,
            max_trades_per_day=8,
            max_daily_loss_r=1.0,
            cooldown_bars=12,
        )
    )

    trades, signals, _ = hypothesis_trades(_forced_stop_frame(), hyp, EvalConfig())

    assert signals > len(trades)
    assert len(trades) == 2  # two sized losses breach the effective -0.5% daily stop
    assert trades["position_size"].tolist() == pytest.approx([0.25, 0.25])
    assert trades["net_return"].to_numpy() == pytest.approx(
        trades["unsized_net_return"].to_numpy() * trades["position_size"].to_numpy()
    )
    metrics = evaluate_hypothesis(_forced_stop_frame(), hyp, EvalConfig())
    assert metrics["effective_risk"] == {
        "risk_per_trade": 0.005,
        "daily_stop_loss": -0.005,
        "max_position_fraction": 0.25,
        "max_consecutive_losses": 3,
        "cooldown_bars": 12,
        "max_trades_per_day": 8,
    }


def test_hypothesis_trades_apply_loss_streak_cooldown_and_daily_trade_cap():
    hyp = _always_long_hypothesis(
        RiskRule(
            risk_per_trade=0.005,
            max_position_fraction=0.25,
            max_trades_per_day=8,
            max_daily_loss_r=100.0,
            cooldown_bars=12,
        )
    )

    trades, _, _ = hypothesis_trades(_forced_stop_frame(), hyp, EvalConfig())

    assert len(trades) == 8
    entries = pd.to_datetime(trades["entry_time"], utc=True)
    assert entries.iloc[3] - entries.iloc[2] >= pd.Timedelta(minutes=60)
    assert entries.iloc[6] - entries.iloc[5] >= pd.Timedelta(minutes=60)


def test_futures_stress_accounts_for_funding_and_delayed_adverse_exit():
    frame = _ts_frame(_sawtooth(500, 0.015, 0.005))
    hypothesis = _rising_hypothesis(horizon=12)

    funded, _, _ = hypothesis_trades(
        frame,
        hypothesis,
        EvalConfig(funding_bps_per_8h=8.0),
    )
    delayed, _, _ = hypothesis_trades(
        frame,
        hypothesis,
        EvalConfig(exit_delay_bars=1, adverse_exit_bps=10.0),
    )

    assert not funded.empty and not delayed.empty
    assert (funded["funding_cost_fraction"] > 0).all()
    assert funded["funding_bps_per_8h"].eq(8.0).all()
    assert delayed["exit_delay_bars"].eq(1).all()
    assert delayed["adverse_exit_bps"].eq(10.0).all()
    assert (
        pd.to_datetime(delayed["exit_time"], utc=True)
        > pd.to_datetime(delayed["original_exit_time"], utc=True)
    ).all()


def test_tag_regimes_is_causal_and_labels_correctly():
    n = 500
    up = _ts_frame(100.0 * 2 ** (np.arange(n) / n))  # doubles over the window
    down = _ts_frame(100.0 * 0.5 ** (np.arange(n) / n))  # halves
    flat = _ts_frame(np.full(n, 100.0))
    for frame, expect in ((up, "bull"), (down, "bear"), (flat, "range")):
        r = tag_regimes(frame, "5m", lookback_bars=100)
        assert (r.iloc[:100] == "unknown").all()  # warmup only
        assert (r.iloc[150:] == expect).all(), expect


def test_perturbed_variants_do_not_mutate_and_cover_exits():
    hyp = FAMILIES["trend_continuation"].builder("4h", "30m", "5m", "long", 1)
    before = hyp.to_dict()
    variants = dict(perturbed_variants(hyp, _RELAXED))
    assert hyp.to_dict() == before  # original untouched
    for label in ("tp_down", "tp_up", "sl_down", "sl_up", "horizon_down", "horizon_up"):
        assert label in variants
        assert variants[label].id == f"{hyp.id}~{label}"
    # family hypotheses carry numeric thresholds -> entry jitters must exist
    assert "entry_stricter" in variants and "entry_looser" in variants
    assert variants["tp_up"].exit.take_profit > hyp.exit.take_profit
    assert variants["horizon_down"].exit.horizon_bars < hyp.exit.horizon_bars


def test_entry_stricter_variant_fires_on_subset_of_bars():
    hyp = FAMILIES["momentum_continuation"].builder("4h", "15m", "5m", "long", 1)
    frame = build_synthetic_aligned_frame([hyp], n=2500)
    stricter = dict(perturbed_variants(hyp, _RELAXED))["entry_stricter"]
    m_base = entry_mask(frame, hyp).to_numpy()
    m_strict = entry_mask(frame, stricter).to_numpy()
    assert bool((m_strict & ~m_base).any()) is False  # stricter ⊆ base


def test_oos_windows_partition_all_trades():
    frame = _ts_frame(_sawtooth(2000, 0.015, 0.005))
    hyp = _rising_hypothesis()
    oos = oos_window_stats(frame, hyp, EvalConfig(), n_windows=5)
    from research_exploration.evaluate import hypothesis_trades

    trades, _, _ = hypothesis_trades(frame, hyp, EvalConfig())
    assert oos["windows"] == 5
    assert sum(w["trades"] for w in oos["window_stats"]) == len(trades)
    assert 0.0 <= oos["pass_rate"] <= 1.0


def test_uptrend_passes_every_stage_and_is_kept():
    frame = _ts_frame(_sawtooth(3000, 0.015, 0.005))  # persistent uptrend
    res = validate_hypothesis(frame, _rising_hypothesis(), _RELAXED)
    assert res["verdict"] == "keep" and res["reasons"] == []
    for seg in ("train", "validation", "holdout"):
        assert res[seg]["total_return"] > 0
    assert res["oos"]["pass_rate"] >= 0.5
    assert res["sensitivity"]["passed"] is True
    assert res["regimes"], "regime breakdown must be populated"
    # real timestamps flowed through: yearly/monthly breakdowns show 2024, not 1970
    assert res["train"]["by_year"][0]["year"] == "2024"
    assert res["train"]["by_month"][0]["month"].startswith("2024-")
    assert res["splits"]["holdout"]["start"] > res["splits"]["train"]["end"]


def test_holdout_must_be_durably_claimed_before_evaluation():
    frame = _ts_frame(_sawtooth(3000, 0.015, 0.005))
    claims = []

    def deny_second_claim(hypothesis, partial_result):
        claims.append((hypothesis.id, partial_result["splits"]["holdout"]))
        return len(claims) == 1

    first = validate_hypothesis(
        frame,
        _rising_hypothesis(),
        _RELAXED,
        before_holdout=deny_second_claim,
    )
    second = validate_hypothesis(
        frame,
        _rising_hypothesis(),
        _RELAXED,
        before_holdout=deny_second_claim,
    )

    assert first["verdict"] == "keep"
    assert second["verdict"] == "inconclusive"
    assert second["reasons"] == ["holdout_already_consumed"]
    assert second["holdout"] is None
    assert len(claims) == 2


def test_negative_holdout_gates_admission():
    """Edge in train+val, crash in the untouched holdout -> REJECT (the old
    pipeline's report-only holdout would have shipped this)."""
    pre = _sawtooth(2400, 0.015, 0.005)  # uptrend train+val
    crash = _sawtooth(601, 0.005, 0.015, start=pre[-1])[1:]  # holdout downtrend
    res = validate_hypothesis(
        _ts_frame(np.concatenate([pre, crash])), _rising_hypothesis(), _RELAXED
    )
    assert res["verdict"] == "reject"
    assert res["reasons"] == ["failed_holdout"]
    assert res["holdout"]["total_return"] <= 0
    assert res["train"]["total_return"] > 0 and res["validation"]["total_return"] > 0


def test_no_train_edge_rejects_without_touching_holdout():
    frame = _ts_frame(_sawtooth(3000, 0.005, 0.015))  # persistent downtrend
    res = validate_hypothesis(frame, _rising_hypothesis(), _RELAXED)
    assert res["verdict"] == "reject" and res["reasons"] == ["no_train_edge"]
    assert res["holdout"] is None  # holdout never evaluated


def test_sensitivity_check_reports_all_variants():
    frame = _ts_frame(_sawtooth(2000, 0.015, 0.005))
    sens = sensitivity_check(frame, _rising_hypothesis(), EvalConfig(), _RELAXED)
    assert len(sens["variants"]) == 6  # exits only (no numeric refs)
    assert sens["pass_fraction"] == 1.0 and sens["passed"] is True


def test_execution_data_stress_check_records_cost_delay_fill_and_gap_variants():
    frame = _ts_frame(_sawtooth(2000, 0.015, 0.005))
    stress = execution_data_stress_check(frame, _rising_hypothesis(), EvalConfig(), _RELAXED)

    assert stress["passed"] is True
    variants = {item["variant"]: item for item in stress["variants"]}
    assert set(variants) == {
        "higher_costs",
        "delayed_adverse_entry",
        "delayed_adverse_exit_funding",
        "missing_bars",
    }
    assert variants["higher_costs"]["eval_overrides"]["fee_bps"] == 10.0
    assert variants["delayed_adverse_entry"]["eval_overrides"] == {
        "fee_bps": 5.0,
        "slippage_bps": 2.0,
        "entry_delay_bars": 1,
        "adverse_fill_bps": 5.0,
        "exit_delay_bars": 0,
        "adverse_exit_bps": 0.0,
        "funding_bps_per_8h": 0.0,
    }
    assert variants["delayed_adverse_exit_funding"]["eval_overrides"] == {
        "fee_bps": 5.0,
        "slippage_bps": 2.0,
        "entry_delay_bars": 0,
        "adverse_fill_bps": 0.0,
        "exit_delay_bars": 1,
        "adverse_exit_bps": 5.0,
        "funding_bps_per_8h": 3.0,
    }
    assert variants["missing_bars"]["rows"] < len(frame)


def test_execution_stress_failure_rejects_before_holdout():
    frame = _ts_frame(_sawtooth(3000, 0.015, 0.005))
    strict = dataclasses.replace(
        _RELAXED,
        stress_cost_multiplier=100.0,
        stress_adverse_fill_bps=500.0,
        min_stress_pass_rate=1.0,
    )
    holdout_claims = []

    result = validate_hypothesis(
        frame,
        _rising_hypothesis(),
        strict,
        before_holdout=lambda *_: holdout_claims.append(True) or True,
    )

    assert result["verdict"] == "reject"
    assert result["reasons"] == ["failed_execution_stress"]
    assert result["stress"]["passed"] is False
    assert result["holdout"] is None
    assert holdout_claims == []


def test_regime_breakdown_totals_match_trades():
    frame = _ts_frame(_sawtooth(2000, 0.015, 0.005))
    hyp = _rising_hypothesis()
    rows = regime_breakdown(frame, hyp, EvalConfig(), _RELAXED)
    from research_exploration.evaluate import hypothesis_trades

    trades, _, _ = hypothesis_trades(frame, hyp, EvalConfig())
    assert sum(r["trades"] for r in rows) == len(trades)
    assert {r["regime"] for r in rows} <= {"bull", "bear", "range", "unknown"}


def test_validate_batch_logs_with_deflated_dsr(tmp_path: Path):
    frame = _ts_frame(_sawtooth(3000, 0.015, 0.005))
    hyps = [_rising_hypothesis()]
    log = tmp_path / "val_log.jsonl"
    results = validate_batch(frame, hyps, _RELAXED, log_path=log)
    assert results[0]["verdict"] == "keep"
    assert results[0]["n_trials"] == len(hyps)
    rows = load_log(log)
    assert len(rows) == 1 and rows[0]["verdict"] == "keep"
    assert rows[0]["config"]["validation"]["train_frac"] == _RELAXED.train_frac
    assert rows[0]["config"]["validation"]["stress_entry_delay_bars"] == 1
    assert rows[0]["config"]["validation"]["stress_exit_delay_bars"] == 1
    assert rows[0]["config"]["validation"]["stress_funding_bps_per_8h"] == 3.0
    assert rows[0]["config"]["eval"]["market"] == "futures"
    assert rows[0]["metrics"]["dsr_method"] == ("autopilot.dsr.expected_max_trial_dispersion/v2")
    assert rows[0]["metrics"]["sr_std_trials"] == 0.0
    assert rows[0]["metrics"]["trial_sharpe_conservative_floor"] == 0.0
    assert "holdout" in rows[0]["metrics"]
    assert {item["variant"] for item in rows[0]["metrics"]["stress"]["variants"]} == {
        "higher_costs",
        "delayed_adverse_entry",
        "delayed_adverse_exit_funding",
        "missing_bars",
    }


def test_validate_batch_preserves_larger_configured_trial_count():
    frame = _ts_frame(_sawtooth(3000, 0.015, 0.005))
    hyps = [_rising_hypothesis()]
    cfg = ValidationConfig(
        min_trades_train=10,
        min_trades_val=3,
        min_trades_holdout=3,
        oos_windows=4,
        regime_lookback_bars=100,
        n_trials=25,
    )

    results = validate_batch(frame, hyps, cfg)

    assert results[0]["n_trials"] == 25
    assert results[0]["sr_std_trials"] > 0


def test_deflated_sharpe_falls_as_cumulative_trial_count_rises():
    returns = np.array([0.012, -0.009, 0.010, -0.008] * 30, dtype=float)
    common = {
        "sr_std_trials": 0.20,
        "trial_sharpe_count": 8,
        "trial_sharpe_observed_std": 0.20,
        "trial_sharpe_conservative_floor": 0.10,
    }

    few_trials = _deflated_sharpe_evidence(
        returns,
        ValidationConfig(n_trials=2, **common),
    )
    many_trials = _deflated_sharpe_evidence(
        returns,
        ValidationConfig(n_trials=200, **common),
    )

    assert many_trials["dsr_deflated"] < few_trials["dsr_deflated"]


def test_validate_batch_checkpoints_each_candidate_before_advancing():
    frame = _ts_frame(_sawtooth(3000, 0.015, 0.005))
    first = _rising_hypothesis()
    second = dataclasses.replace(first, id="SECOND_CANDIDATE")
    checkpointed = []

    def stop_after_first(hypothesis, result):
        checkpointed.append((hypothesis.id, result["verdict"]))
        raise RuntimeError("simulated checkpoint boundary stop")

    with pytest.raises(RuntimeError, match="checkpoint boundary"):
        validate_batch(
            frame,
            [first, second],
            _RELAXED,
            after_candidate=stop_after_first,
        )

    assert checkpointed == [(first.id, "keep")]


def test_validation_config_rejects_bad_fractions():
    with pytest.raises(ValueError):
        ValidationConfig(train_frac=0.8, val_frac=0.3)
    with pytest.raises(ValueError):
        ValidationConfig(train_frac=0.0, val_frac=0.2)
