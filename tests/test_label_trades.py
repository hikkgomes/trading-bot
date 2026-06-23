import warnings

import numpy as np
import pandas as pd

from src.day_trade_search import DayTradeConfig, simulate_day_trades
from src.label_trades import build_trade_labels, compute_tp_sl_labels


def test_compute_tp_sl_labels_shapes():
    n = 30
    open_ = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    out = compute_tp_sl_labels(open_, high, low, close, 0.005, 0.005, 4, "long")
    assert len(out) == 6
    assert all(len(v) == n for v in out)


def test_build_trade_labels_column_naming():
    n = 40
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "tf_5m_open": np.full(n, 100.0),
        "tf_5m_high": np.full(n, 101.0),
        "tf_5m_low": np.full(n, 99.0),
        "tf_5m_close": np.full(n, 100.0),
    })
    out = build_trade_labels(df, [(0.005, 0.003)], [8], ["long"], "tf_5m_")
    assert "label_long_tp50_sl30_h8" in out.columns
    assert "mfe_long_h8_full" in out.columns
    assert "mae_long_tp50_sl30_h8_exit" in out.columns


def test_label_matches_simulation_on_simple_path():
    n = 30
    data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "tf_5m_open": [100.0] * n,
        "tf_5m_high": [100.0, 101.0] + [100.0] * (n - 2),
        "tf_5m_low": [100.0] * n,
        "tf_5m_close": [100.0] * n,
    })
    labels = build_trade_labels(data, [(0.01, 0.01)], [4], ["long"], "tf_5m_")
    signal = pd.Series([True] + [False] * (n - 1))
    config = DayTradeConfig(take_profit=0.01, stop_loss=0.01, fee_bps=0, slippage_bps=0, horizon_bars=4)
    trades = simulate_day_trades(data, signal, "long", config)
    assert bool(labels["label_long_tp100_sl100_h4"].iloc[0]) == (trades["exit_reason"].iloc[0] == "take_profit")


def test_mfe_mae_values_long_and_short():
    n = 8
    open_ = np.array([100, 100, 100, 100, 100, 100, 100, 100], dtype=float)
    high = np.array([100, 101.2, 103, 101, 100, 100, 100, 100], dtype=float)
    low = np.array([100, 99, 98, 99, 100, 100, 100, 100], dtype=float)
    close = np.array([100] * n, dtype=float)
    _, _, mfe_f_l, mae_f_l, _, _ = compute_tp_sl_labels(open_, high, low, close, 0.1, 0.1, 3, "long")
    _, _, mfe_f_s, mae_f_s, _, _ = compute_tp_sl_labels(open_, high, low, close, 0.1, 0.1, 3, "short")
    assert round(float(mfe_f_l[0]), 6) == 0.03
    assert round(float(mae_f_l[0]), 6) == -0.02
    assert round(float(mfe_f_s[0]), 6) == round(100 / 98 - 1, 6)
    assert round(float(mae_f_s[0]), 6) == round(100 / 103 - 1, 6)
    hit_tp, _, _, _, mfe_e_l, mae_e_l = compute_tp_sl_labels(open_, high, low, close, 0.01, 0.05, 3, "long")
    assert bool(hit_tp[0]) is True
    assert round(float(mfe_e_l[0]), 6) != round(float(mfe_f_l[0]), 6)
    assert round(float(mae_e_l[0]), 6) != round(float(mae_f_l[0]), 6)


def test_short_stop_and_time_exit_labels():
    n = 20
    data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "tf_5m_open": [100.0] * n,
        "tf_5m_high": [100.0, 102.0] + [100.0] * (n - 2),
        "tf_5m_low": [100.0] * n,
        "tf_5m_close": [100.0] * n,
    })
    labels = build_trade_labels(data, [(0.01, 0.01)], [4], ["short"], "tf_5m_")
    assert bool(labels["label_short_tp100_sl100_h4"].iloc[0]) is False


def test_exit_columns_differ_across_tp_sl_pairs():
    n = 30
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "tf_5m_open": [100.0] * n,
        "tf_5m_high": [100.0, 101.0, 100.5] + [100.0] * (n - 3),
        "tf_5m_low": [100.0, 99.6, 99.2] + [100.0] * (n - 3),
        "tf_5m_close": [100.0] * n,
    })
    out = build_trade_labels(df, [(0.005, 0.004), (0.01, 0.01)], [4], ["long"], "tf_5m_")
    assert "mfe_long_tp50_sl40_h4_exit" in out.columns
    assert "mfe_long_tp100_sl100_h4_exit" in out.columns
    assert "mfe_long_h4_full" in out.columns


def test_build_trade_labels_does_not_fragment_dataframe():
    n = 40
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "tf_5m_open": np.full(n, 100.0),
        "tf_5m_high": np.full(n, 101.0),
        "tf_5m_low": np.full(n, 99.0),
        "tf_5m_close": np.full(n, 100.0),
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = build_trade_labels(
            df,
            [(0.003, 0.002), (0.005, 0.003), (0.008, 0.004), (0.012, 0.006)],
            [4, 8, 16],
            ["long", "short"],
            "tf_5m_",
        )
    assert not any(isinstance(w.message, pd.errors.PerformanceWarning) for w in caught)
