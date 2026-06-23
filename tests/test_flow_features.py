import numpy as np
import pandas as pd

from build_binance_indicator_dataset import FLOW_WINDOWS, build_flow_features


def make_candles(n=200, buy_fraction=0.5, seed=1):
    rng = np.random.default_rng(seed)
    volume = rng.uniform(50, 150, size=n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "open": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "close": np.full(n, 100.0),
        "volume": volume,
        "quote_asset_volume": volume * 100,
        "number_of_trades": rng.integers(10, 100, size=n).astype(float),
        "taker_buy_base_volume": volume * buy_fraction,
        "taker_buy_quote_volume": volume * buy_fraction * 100,
    })


def test_flow_features_missing_columns_returns_empty():
    df = pd.DataFrame({"open": [1.0], "close": [1.0]})
    assert build_flow_features(df) == {}


def test_flow_features_all_buys_saturate_at_one():
    df = make_candles(buy_fraction=1.0)
    features = build_flow_features(df)
    assert features["taker_buy_ratio"].iloc[-1] == 1.0
    assert features["taker_imbalance"].iloc[-1] == 1.0
    for window in FLOW_WINDOWS:
        assert abs(features[f"cvd_{window}"].iloc[-1] - 1.0) < 1e-12


def test_flow_features_all_sells_saturate_at_minus_one():
    df = make_candles(buy_fraction=0.0)
    features = build_flow_features(df)
    assert features["taker_imbalance"].iloc[-1] == -1.0
    for window in FLOW_WINDOWS:
        assert abs(features[f"cvd_{window}"].iloc[-1] + 1.0) < 1e-12


def test_flow_features_balanced_flow_is_neutral():
    df = make_candles(buy_fraction=0.5)
    features = build_flow_features(df)
    assert abs(features["taker_imbalance"].iloc[-1]) < 1e-12
    assert abs(features["cvd_20"].iloc[-1]) < 1e-12


def test_flow_features_have_no_lookahead():
    df = make_candles(n=150)
    features_before = build_flow_features(df)
    mutated = df.copy()
    mutated.loc[mutated.index[-1], ["volume", "taker_buy_base_volume", "number_of_trades"]] = [
        10_000.0, 10_000.0, 5_000.0,
    ]
    features_after = build_flow_features(mutated)
    for name in features_before:
        np.testing.assert_array_equal(
            features_before[name].iloc[:-1].to_numpy(),
            features_after[name].iloc[:-1].to_numpy(),
            err_msg=f"lookahead detected in {name}",
        )


def test_flow_features_zero_volume_yields_nan_not_inf():
    df = make_candles(n=50)
    df.loc[df.index[10], ["volume", "taker_buy_base_volume"]] = 0.0
    df.loc[df.index[11], "number_of_trades"] = 0.0
    features = build_flow_features(df)
    assert np.isnan(features["taker_buy_ratio"].iloc[10])
    assert np.isnan(features["avg_trade_size"].iloc[11])
    for series in features.values():
        assert not np.isinf(series.to_numpy(dtype=float)).any()
