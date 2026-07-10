import numpy as np
import pandas as pd
import pytest

from src.day_trade_search import StrategyCandidate
from src.discover_patterns import Condition, build_all_conditions
from src.walk_forward import (
    WalkForwardConfig,
    _compute_threshold,
    aggregate_walk_forward_results,
    generate_windows,
    refit_conditions,
    walk_forward_score_candidate,
)


def test_generate_windows_no_overlap():
    cfg = WalkForwardConfig(train_bars=20, test_bars=10, step_bars=10, min_windows=2, embargo_bars=3)
    windows = generate_windows(80, cfg)
    assert len(windows) >= 2
    for train_slice, test_slice in windows:
        assert train_slice.stop <= test_slice.start - cfg.embargo_bars


def test_generate_windows_embargo_zero():
    cfg = WalkForwardConfig(train_bars=20, test_bars=10, step_bars=10, min_windows=2, embargo_bars=0)
    windows = generate_windows(80, cfg)
    assert windows[0][0].stop == windows[0][1].start


def test_generate_windows_min_windows_error():
    cfg = WalkForwardConfig(train_bars=30, test_bars=20, step_bars=20, min_windows=4)
    with pytest.raises(ValueError):
        generate_windows(90, cfg)


def test_refit_conditions_recomputes_quantiles():
    x1 = pd.DataFrame({"tf_5m_rsi_14": np.linspace(10, 20, 50)})
    x2 = pd.DataFrame({"tf_5m_rsi_14": np.linspace(40, 50, 50)})
    candidate = StrategyCandidate(
        direction="long",
        horizon_bars=4,
        conditions=(Condition("tf_5m_rsi_14", "value_ge", 0.0, "desc", threshold_source="quantile", quantile=0.9),),
    )
    c1 = refit_conditions(x1, candidate, "tf_5m_")
    c2 = refit_conditions(x2, candidate, "tf_5m_")
    assert c1.conditions[0].threshold != c2.conditions[0].threshold


def test_compute_threshold_sources():
    df = pd.DataFrame({
        "a": np.linspace(1, 10, 100),
        "b": np.linspace(2, 20, 100),
    })
    assert _compute_threshold(df, Condition("a", "value_ge", 0.0, "d", threshold_source="quantile", quantile=0.8)) > 0
    assert _compute_threshold(df, Condition("a", "delta_ge", 0.0, "d", threshold_source="delta_quantile", quantile=0.9)) >= 0
    assert _compute_threshold(df, Condition("a", "slope_3_ge", 0.0, "d", threshold_source="slope_quantile", quantile=0.9, lookback=3)) >= 0
    assert _compute_threshold(df, Condition("a", "ratio_ge", 0.0, "d", feature_b="b", threshold_source="ratio_quantile", quantile=0.8)) > 0
    assert _compute_threshold(df, Condition("a", "cross_above", 0.0, "d", threshold_source="fixed")) == 0.0


def test_compute_threshold_missing_metadata_raises():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    with pytest.raises(ValueError):
        _compute_threshold(df, Condition("a", "value_ge", 0.0, "d", threshold_source="quantile", quantile=None))
    with pytest.raises(ValueError):
        _compute_threshold(df, Condition("a", "ratio_ge", 0.0, "d", threshold_source="ratio_quantile", quantile=0.8))
    with pytest.raises(ValueError):
        _compute_threshold(df, Condition("a", "x", 0.0, "d", threshold_source=None))
    with pytest.raises(ValueError):
        _compute_threshold(df, Condition("missing", "value_ge", 0.0, "d", threshold_source="fixed"))
    with pytest.raises(ValueError):
        _compute_threshold(df, Condition("a", "slope_3_ge", 0.0, "d", threshold_source="slope_quantile", quantile=0.9))


def test_aggregate_walk_forward_results_counts_all_windows():
    agg = aggregate_walk_forward_results([
        {"test_total_return": 0.1, "test_avg_net_return": 0.01, "test_profit_factor": 1.2, "test_max_drawdown": -0.02, "test_trades": 10, "screened_out": False},
        {"test_total_return": 0.0, "test_avg_net_return": 0.0, "test_profit_factor": 0.0, "test_max_drawdown": 0.0, "test_trades": 0, "screened_out": True},
    ], pass_rate_threshold=0.8)
    assert agg["total_windows"] == 2
    assert agg["scored_windows"] == 1
    assert agg["screened_out_windows"] == 1
    assert agg["pass_rate"] == 0.5


def test_walk_forward_score_candidate_calls_each_fold():
    n = 120
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"), "x": np.arange(n)})
    candidate = StrategyCandidate("long", 4, (Condition("x", "value_ge", 0.0, "x", threshold_source="quantile", quantile=0.8),))
    cfg = WalkForwardConfig(train_bars=40, test_bars=20, step_bars=20, min_windows=3)

    def score_fn(train, test, cand, conf, base_prefix):
        _ = conf, base_prefix
        return {"test_total_return": 1.0 if cand.conditions[0].threshold > 0 else 0.0, "test_avg_net_return": 0.01, "test_profit_factor": 1.1, "test_max_drawdown": -0.01, "test_trades": len(test)}

    windows, agg = walk_forward_score_candidate(df, candidate, object(), cfg, "tf_5m_", score_fn)
    assert len(windows) == 4
    assert agg["total_windows"] == 4


def test_walk_forward_screened_out_windows_reduce_pass_rate():
    n = 120
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"), "x": np.arange(n)})
    candidate = StrategyCandidate("long", 4, (Condition("x", "value_ge", 0.0, "x", threshold_source="quantile", quantile=0.8),))
    cfg = WalkForwardConfig(train_bars=40, test_bars=20, step_bars=20, min_windows=3)
    state = {"i": 0}

    def score_fn(train, test, cand, conf, base_prefix):
        _ = train, test, cand, conf, base_prefix
        state["i"] += 1
        if state["i"] % 2 == 0:
            return {"test_total_return": 0.0, "test_avg_net_return": 0.0, "test_profit_factor": 0.0, "test_max_drawdown": 0.0, "test_trades": 0, "screened_out": True}
        return {"test_total_return": 0.1, "test_avg_net_return": 0.01, "test_profit_factor": 1.1, "test_max_drawdown": -0.01, "test_trades": 20, "screened_out": False}

    _, agg = walk_forward_score_candidate(df, candidate, object(), cfg, "tf_5m_", score_fn)
    assert agg["total_windows"] == 4
    assert agg["screened_out_windows"] == 2
    assert agg["pass_rate"] == 0.5


def test_build_all_conditions_include_threshold_metadata():
    n = 200
    rng = np.random.default_rng(42)
    train = pd.DataFrame({
        "tf_15m_close": rng.normal(size=n).cumsum() + 100,
        "tf_15m_rsi_14": rng.normal(size=n).cumsum() + 50,
        "tf_4h_rsi_14": rng.normal(size=n).cumsum() + 50,
    })
    conds = build_all_conditions(train, ["tf_15m_rsi_14", "tf_4h_rsi_14"])
    assert all(c.threshold_source is not None for c in conds)
    candidate = StrategyCandidate("long", 4, tuple(conds[:20]))
    a = refit_conditions(train.iloc[:120].copy(), candidate, "tf_15m_")
    b = refit_conditions(train.iloc[40:160].copy(), candidate, "tf_15m_")
    any_changed = False
    for ca, cb in zip(a.conditions, b.conditions, strict=False):
        if ca.threshold_source in {"quantile", "delta_quantile", "slope_quantile", "ratio_quantile"}:
            any_changed = any_changed or (ca.threshold != cb.threshold)
        if ca.threshold_source == "fixed":
            assert ca.threshold == 0.0 and cb.threshold == 0.0
    assert any_changed
