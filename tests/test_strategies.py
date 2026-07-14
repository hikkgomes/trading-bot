"""Tests for the unified strategy framework (registry, backtester, library)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies import BacktestConfig, available, get, run_backtest
from src.strategies.backtester import _simulate
from src.strategies.base import extract_ohlcv


def _synth(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, n) + 0.001 * np.sin(np.arange(n) / 200.0)
    close = 30_000 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2021-01-01", periods=n, freq="15min", name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(1, 9, n)},
        index=idx,
    )


# All rule (non-fittable, non-bridge) strategies: signal-only, need just OHLCV.
_RULE_STRATEGIES = [
    "sma_cross",
    "rsi_reversion",
    "donchian_breakout",
    "multi_tf_trend",
    "macd_trend",
    "supertrend",
    "adx_trend",
    "bollinger_reversion",
    "zscore_reversion",
    "stochastic_reversion",
    "keltner_breakout",
    "atr_channel_breakout",
    "bollinger_squeeze",
    "momentum_roc",
    "candlestick_reversal",
    "swing_structure",
    "btc_cycle_guard",
    "rsi_divergence",
    "regression_channel",
]


# --------------------------------------------------------------------------- registry
def test_registry_contains_library():
    names = available()
    for expected in _RULE_STRATEGIES + [
        "ml_classifier",
        "ml_regressor",
        "condition_grid",
        "regime_filter",
    ]:
        assert expected in names


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get("does_not_exist")


# --------------------------------------------------------------------------- signals
@pytest.mark.parametrize("name", _RULE_STRATEGIES)
def test_rule_strategy_signals_and_backtest(name):
    df = _synth()
    strat = get(name)()
    sig = strat.generate_signals(df)
    assert set(pd.unique(sig.dropna())).issubset({-1, 0, 1})
    assert len(sig) == len(df)
    result = run_backtest(strat, df)
    # Trades are non-overlapping and within bounds.
    if not result.trades.empty:
        # exit can land on the entry bar (intrabar TP/SL) -> holding_bars >= 0.
        assert (result.trades["entry_time"] <= result.trades["exit_time"]).all()
        assert (result.trades["holding_bars"] >= 0).all()
        assert result.trades["direction"].isin({"long", "short"}).all()


# --------------------------------------------------------------------------- engine correctness
def test_backtester_matches_search_engine():
    """The framework engine must reproduce strategy_search.simulate_trades."""
    from src.strategy_search import simulate_trades

    df = _synth(2000, seed=11)
    # Build a tf_15m_-prefixed frame for the search engine + a plain frame for ours.
    tf = df.rename(columns={c: f"tf_15m_{c}" for c in ["open", "high", "low", "close"]})
    tf = tf.reset_index()  # timestamp column

    rng = np.random.default_rng(0)
    mask = pd.Series(rng.random(len(df)) < 0.05, index=df.index)

    cfg = BacktestConfig(
        fee_bps=10, slippage_bps=2, take_profit=0.05, stop_loss=0.03, horizon_bars=96
    )
    search_trades = simulate_trades(
        tf,
        mask.reset_index(drop=True),
        direction="long",
        horizon_bars=96,
        fee_bps=10,
        slippage_bps=2,
        take_profit=0.05,
        stop_loss=0.03,
        pnl_unit="usdt",
    )

    o = extract_ohlcv(df)
    direction = np.where(mask.to_numpy(), 1, 0).astype(int)
    ours = _simulate(o.open, o.high, o.low, o.close, direction, df.index, cfg)

    assert len(ours) == len(search_trades)
    ours_net = np.array([t["net_return"] for t in ours])
    np.testing.assert_allclose(
        ours_net, search_trades["net_return"].to_numpy(), rtol=1e-9, atol=1e-12
    )


def test_btc_pnl_unit_only_shorts_realise():
    df = _synth(1500, seed=5)
    cfg = BacktestConfig(pnl_unit="btc", horizon_bars=48)
    o = extract_ohlcv(df)
    direction = np.zeros(len(df), dtype=int)
    direction[::100] = 1  # longs only
    trades = _simulate(o.open, o.high, o.low, o.close, direction, df.index, cfg)
    # In BTC mode longs realise gross 0 -> net == -cost for every trade.
    assert all(abs(t["gross_return"]) < 1e-12 for t in trades)


def test_canonical_backtester_uses_product_specific_short_return_math():
    open_ = np.array([100.0, 100.0, 90.0])
    high = np.array([100.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 90.0])
    close = np.array([100.0, 100.0, 90.0])
    direction = np.array([-1, 0, 0])
    index = pd.RangeIndex(3)

    usdt_trade = _simulate(
        open_,
        high,
        low,
        close,
        direction,
        index,
        BacktestConfig(
            fee_bps=0,
            slippage_bps=0,
            take_profit=0.5,
            stop_loss=0.5,
            horizon_bars=1,
            pnl_unit="usdt",
        ),
    )[0]
    btc_trade = _simulate(
        open_,
        high,
        low,
        close,
        direction,
        index,
        BacktestConfig(
            fee_bps=0,
            slippage_bps=0,
            take_profit=0.5,
            stop_loss=0.5,
            horizon_bars=1,
            pnl_unit="btc",
        ),
    )[0]

    assert usdt_trade["gross_return"] == pytest.approx(0.10)
    assert btc_trade["gross_return"] == pytest.approx(1 / 9)


# --------------------------------------------------------------------------- indicators
def test_new_indicators_bounds_and_shape():
    from src.strategies import indicators as ind

    df = _synth(800, seed=4)
    close, high, low = df["close"], df["high"], df["low"]

    macd_line, signal_line, hist = ind.macd(close)
    assert len(macd_line) == len(signal_line) == len(hist) == len(df)

    k, d = ind.stochastic(high, low, close)
    valid_k = k.dropna()
    assert ((valid_k >= -1e-6) & (valid_k <= 100 + 1e-6)).all()

    adx_line, plus_di, minus_di = ind.adx(high, low, close)
    valid_adx = adx_line.dropna()
    assert ((valid_adx >= -1e-6) & (valid_adx <= 100 + 1e-6)).all()

    wr = ind.williams_r(high, low, close).dropna()
    assert ((wr >= -100 - 1e-6) & (wr <= 1e-6)).all()

    st = ind.supertrend(high, low, close)
    assert set(pd.unique(st)).issubset({-1, 1})
    lower, mid, upper = ind.bollinger_bands(close)
    assert (upper.dropna() >= lower.dropna()).all()


def test_supertrend_no_lookahead():
    """Supertrend direction at bar i must not change when future bars are altered."""
    from src.strategies import indicators as ind

    df = _synth(500, seed=8)
    full = ind.supertrend(df["high"], df["low"], df["close"])
    cut = 300
    trunc = ind.supertrend(df["high"].iloc[:cut], df["low"].iloc[:cut], df["close"].iloc[:cut])
    # The first `cut` directions must be identical whether or not later bars exist.
    np.testing.assert_array_equal(full.iloc[:cut].to_numpy(), trunc.to_numpy())


def test_swing_levels_no_lookahead():
    """Confirmed swing levels must not change when future bars are appended.

    The pivot uses a centered window, so this guards against leakage: a pivot is
    only emitted `pivot` bars after it forms, so the early values must be stable.
    """
    from src.strategies import indicators as ind

    df = _synth(600, seed=15)
    pivot = 5
    full = ind.last_swing_high(df["high"], pivot)
    cut = 400
    trunc = ind.last_swing_high(df["high"].iloc[:cut], pivot)
    # Compare the region that is fully confirmed in both (exclude the last `pivot`
    # bars of the truncated series, whose pivots can't be confirmed yet).
    safe = cut - pivot
    pd.testing.assert_series_equal(full.iloc[:safe], trunc.iloc[:safe], check_freq=False)


def test_candlestick_patterns_detect():
    from src.strategies import indicators as ind

    # Hand-built bullish engulfing: red bar then a green bar engulfing it.
    o = pd.Series([10.0, 8.5])
    h = pd.Series([10.2, 11.0])
    low = pd.Series([8.8, 8.0])
    c = pd.Series([9.0, 10.5])
    assert bool(ind.bullish_engulfing(o, h, low, c).iloc[1])
    assert not bool(ind.bearish_engulfing(o, h, low, c).iloc[1])


# --------------------------------------------------------------------------- ml
def _synth_with_features(n=3000, seed=9):
    df = _synth(n, seed)
    df["mom_20"] = df["close"].pct_change(20)
    df["mom_50"] = df["close"].pct_change(50)
    df["vol_20"] = df["close"].pct_change().rolling(20).std()
    return df


def test_ml_regressor_fit_predict():
    df = _synth_with_features(seed=12)
    split = int(len(df) * 0.7)
    strat = get("ml_regressor")(horizon=24)
    strat.fit(df.iloc[:split])
    sig = strat.generate_signals(df.iloc[split:])
    assert set(pd.unique(sig.dropna())).issubset({-1, 0, 1})


def test_ml_classifier_fit_predict():
    df = _synth(3000, seed=9)
    df["mom_20"] = df["close"].pct_change(20)
    df["mom_50"] = df["close"].pct_change(50)
    df["vol_20"] = df["close"].pct_change().rolling(20).std()
    split = int(len(df) * 0.7)
    strat = get("ml_classifier")(horizon=24)
    strat.fit(df.iloc[:split])
    sig = strat.generate_signals(df.iloc[split:])
    assert set(pd.unique(sig.dropna())).issubset({-1, 0, 1})


def test_ml_classifier_triple_barrier_target():
    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100],
            "high": [100, 106, 101, 101, 101, 101],
            "low": [100, 99, 96, 99, 99, 99],
            "close": [100, 105, 97, 100, 100, 100],
            "feat": [1, 2, 3, 4, 5, 6],
        }
    )
    strat = get("ml_classifier")(
        horizon=2, label_mode="triple_barrier", label_tp=0.05, label_sl=0.03
    )

    target = strat._target(df)

    assert target.iloc[0] == 1.0
    assert target.iloc[1] == 0.0


def test_ml_classifier_triple_barrier_fit_predict_with_screening():
    df = _synth_with_features(n=1800, seed=22)
    split = int(len(df) * 0.7)
    strat = get("ml_classifier")(
        horizon=16,
        label_mode="triple_barrier",
        label_tp=0.01,
        label_sl=0.01,
        feature_screen="spearman",
        max_features=2,
        model="sklearn",
    )

    strat.fit(df.iloc[:split])
    sig = strat.generate_signals(df.iloc[split:])

    assert strat._features is not None
    assert len(strat._features) <= 2
    assert set(pd.unique(sig.dropna())).issubset({-1, 0, 1})


def test_ml_regressor_triple_barrier_target():
    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100],
            "high": [100, 106, 101, 101, 101, 101],
            "low": [100, 99, 96, 99, 99, 99],
            "close": [100, 105, 97, 100, 100, 100],
            "feat": [1, 2, 3, 4, 5, 6],
        }
    )
    strat = get("ml_regressor")(
        horizon=2, target_mode="triple_barrier", label_tp=0.05, label_sl=0.03
    )

    target = strat._target(df)

    assert target.iloc[0] == 0.05
    assert target.iloc[1] == -0.03


def test_ml_regressor_triple_barrier_fit_predict_with_screening():
    df = _synth_with_features(n=1800, seed=23)
    split = int(len(df) * 0.7)
    strat = get("ml_regressor")(
        horizon=16,
        target_mode="triple_barrier",
        label_tp=0.01,
        label_sl=0.01,
        feature_screen="spearman",
        max_features=2,
        model="sklearn",
    )

    strat.fit(df.iloc[:split])
    sig = strat.generate_signals(df.iloc[split:])

    assert strat._features is not None
    assert len(strat._features) <= 2
    assert set(pd.unique(sig.dropna())).issubset({-1, 0, 1})


# --------------------------------------------------------------------------- condition grid bridge
def test_condition_grid_from_dict():
    from src.discover_patterns import Condition

    df = _synth(1000)
    df["tf_15m_close"] = df["close"]
    cond = Condition(
        feature="tf_15m_close",
        kind="value_ge",
        threshold=float(df["close"].median()),
        description="close >= median",
    )
    strat = get("condition_grid")(conditions=[cond], direction="long")
    sig = strat.generate_signals(df)
    assert set(pd.unique(sig.dropna())).issubset({0, 1})
    assert (sig == 1).sum() >= 1


def test_regime_filter_masks_child_signals():
    df = _synth(800)
    df["tf_1d_regime_id"] = [0] * 400 + [1] * 400
    base = get("sma_cross")(fast=5, slow=20)
    wrapped = get("regime_filter")(
        strategy="sma_cross",
        regime_ids="1",
        child_params={"fast": 5, "slow": 20},
    )

    base_sig = base.generate_signals(df)
    filtered = wrapped.generate_signals(df)

    assert (filtered.iloc[:400] == 0).all()
    assert filtered.iloc[400:].equals(base_sig.iloc[400:])


def test_regime_filter_uses_child_default_config():
    wrapped = get("regime_filter")(strategy="sma_cross", regime_ids="1")

    assert (
        wrapped.resolved_default_config().horizon_bars
        == get("sma_cross").default_config().horizon_bars
    )


def test_regime_filter_requires_regime_column():
    df = _synth(300)
    strat = get("regime_filter")(strategy="sma_cross", regime_ids="0")

    with pytest.raises(ValueError, match="Missing regime column"):
        strat.generate_signals(df)
