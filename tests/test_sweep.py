"""Tests for the batch sweep / compare harness (src.sweep)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sweep import _param_combos, _parse_grid, run_sweep


def _synth(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, n) + 0.001 * np.sin(np.arange(n) / 200.0)
    close = 30_000 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2021-01-01", periods=n, freq="15min", name="timestamp")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(1, 9, n)},
        index=idx,
    )
    df["mom_20"] = df["close"].pct_change(20)
    df["vol_20"] = df["close"].pct_change().rolling(20).std()
    return df


def test_parse_grid_and_combos():
    grid = _parse_grid(["period=7,14,21", "oversold=20,30"])
    assert grid == {"period": [7, 14, 21], "oversold": [20, 30]}
    combos = _param_combos(grid)
    assert len(combos) == 6
    assert {"period": 7, "oversold": 20} in combos


def test_run_sweep_ranks_and_benchmarks():
    df = _synth()
    table = run_sweep(df, ["sma_cross", "macd_trend", "rsi_reversion"], train_fraction=0.6)
    assert len(table) == 3
    for col in ["strategy", "trades", "total_return", "vs_buy_hold", "sharpe", "dsr", "n_trials"]:
        assert col in table.columns
    assert set(table["n_trials"]) == {3}
    assert table["dsr"].between(0.0, 1.0).all()
    assert "buy_and_hold" in table.attrs
    # vs_buy_hold is total_return minus the (single) benchmark for every row.
    bh = table.attrs["buy_and_hold"]
    np.testing.assert_allclose(
        (table["total_return"] - table["vs_buy_hold"]).to_numpy(), bh, atol=1e-9
    )


def test_run_sweep_includes_fittable_strategy():
    df = _synth()
    table = run_sweep(df, ["ml_classifier", "sma_cross"], train_fraction=0.7)
    # Both ran without raising; ml_classifier was fit on the train slice.
    assert set(table["strategy"]) == {"ml_classifier", "sma_cross"}
    assert (table["trades"] >= 0).all()


def test_run_sweep_param_grid():
    df = _synth()
    grids = {"rsi_reversion": {"period": [7, 14]}}
    table = run_sweep(df, ["rsi_reversion"], train_fraction=0.6, grids=grids)
    assert len(table) == 2
    assert set(table["strategy"]) == {"rsi_reversion[period=7]", "rsi_reversion[period=14]"}
    assert set(table["n_trials"]) == {2}


def test_run_sweep_walk_forward_columns():
    df = _synth(n=3600)
    table = run_sweep(
        df,
        ["sma_cross", "macd_trend"],
        train_fraction=0.5,
        walk_forward_windows=4,
    )

    assert set(table["wf_windows"]) == {4}
    assert table["wf_pass_rate"].between(0.0, 1.0).all()
    assert table["wf_trades"].ge(0).all()
    assert table["wf_dsr"].between(0.0, 1.0).all()
    assert table["wf_window_returns"].str.startswith("[").all()


def test_run_sweep_regime_filter_grid():
    df = _synth(n=1600)
    df["tf_1d_regime_id"] = [0] * 800 + [1] * 800
    table = run_sweep(
        df,
        ["regime_filter"],
        train_fraction=0.5,
        grids={"regime_filter": {"strategy": ["sma_cross"], "regime_ids": ["0", "1"]}},
    )

    assert len(table) == 2
    assert set(table["strategy"]) == {
        "regime_filter[strategy=sma_cross,regime_ids=0]",
        "regime_filter[strategy=sma_cross,regime_ids=1]",
    }
    assert set(table["n_trials"]) == {2}
