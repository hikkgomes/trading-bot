import numpy as np
import pandas as pd
import pytest

from src.day_trade_search import DayTradeConfig
from src.model_signals import (
    backtest_model_signals,
    predict_signals,
    train_signal_model,
    walk_forward_model_signals,
)
from src.walk_forward import WalkForwardConfig


def test_predict_signals_positive_ev_filter():
    class _Stub:
        def predict_proba(self, x):
            n = len(x)
            probs = np.zeros((n, 2))
            probs[:, 1] = np.linspace(0.1, 0.9, n)
            probs[:, 0] = 1 - probs[:, 1]
            return probs

    df = pd.DataFrame({"f": [1, 2, 3, 4]})
    mask, prob, ev = predict_signals(_Stub(), df, ["f"], tp=0.01, sl=0.01, fee_cost=0.001, min_ev=0.001)
    assert mask.sum() > 0
    assert (prob >= 0).all()
    assert len(ev) == len(df)


def test_train_signal_model_too_few_rows():
    df = pd.DataFrame({"x": [1], "label": [1]})
    with pytest.raises(ValueError):
        train_signal_model(df, ["x"], "label")


def test_walk_forward_model_signals_oos_rows_and_source_index():
    n = 600
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    df["label_long_tp50_sl30_h8"] = (df["f1"] + rng.normal(scale=0.2, size=n) > 0).astype(int)
    wf = WalkForwardConfig(train_bars=200, test_bars=100, step_bars=100, min_windows=3)
    out = walk_forward_model_signals(df, ["f1", "f2"], "label_long_tp50_sl30_h8", wf, tp=0.005, sl=0.003, fee_cost=0.0, max_features=2)
    assert {"timestamp", "source_index", "signal", "prob", "ev"}.issubset(out.columns)
    assert len(out) == 400


def test_backtest_model_signals_compatible_with_simulator():
    n = 80
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "tf_5m_open": [100.0] * n,
        "tf_5m_high": [100.5] * n,
        "tf_5m_low": [99.5] * n,
        "tf_5m_close": [100.0] * n,
    })
    signals = pd.DataFrame({
        "timestamp": [df.iloc[0]["timestamp"]],
        "source_index": [0],
        "signal": [True],
        "label_column": ["label_long_tp50_sl30_h8"],
    })
    config = DayTradeConfig(take_profit=0.005, stop_loss=0.003, fee_bps=0, slippage_bps=0, horizon_bars=4)
    trades = backtest_model_signals(df, signals, config, "tf_5m_")
    assert isinstance(trades, pd.DataFrame)
