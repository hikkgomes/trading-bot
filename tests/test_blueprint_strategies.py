"""Tests for blueprint-derived additions: divergence, regression channel,
Fear & Greed, and the execution position-plan tactics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies import get, run_backtest


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


# --------------------------------------------------------------------------- divergence
def test_divergence_is_lookahead_safe():
    from src.strategies import indicators as ind

    df = _synth(800, seed=21)
    rsi = ind.rsi(df["close"], 14)
    bull_full, bear_full = ind.regular_divergence(df["high"], df["low"], rsi, pivot=5)
    cut = 500
    rsi_t = ind.rsi(df["close"].iloc[:cut], 14)
    bull_t, bear_t = ind.regular_divergence(df["high"].iloc[:cut], df["low"].iloc[:cut], rsi_t, pivot=5)
    safe = cut - 5  # last `pivot` bars can't be confirmed yet in the truncated run
    np.testing.assert_array_equal(bull_full.iloc[:safe].to_numpy(), bull_t.iloc[:safe].to_numpy())
    np.testing.assert_array_equal(bear_full.iloc[:safe].to_numpy(), bear_t.iloc[:safe].to_numpy())


def test_regression_channel_orders_bands():
    from src.strategies import indicators as ind

    df = _synth(600, seed=4)
    lower, mid, upper = ind.regression_channel(df["close"], window=100, num_std=2.0)
    valid = mid.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


@pytest.mark.parametrize("name", ["rsi_divergence", "regression_channel"])
def test_blueprint_rule_strategy_runs(name):
    df = _synth()
    sig = get(name)().generate_signals(df)
    assert set(pd.unique(sig.dropna())).issubset({-1, 0, 1})
    run_backtest(get(name)(), df)  # must not raise


# --------------------------------------------------------------------------- fear & greed
def test_add_fear_greed_column_merges_by_date():
    from src.fear_greed import add_fear_greed_column

    df = _synth(400)
    # Mock the API frame: one value per day spanning the df's dates.
    days = pd.date_range("2021-01-01", periods=10, freq="D", tz="UTC")
    fng = pd.DataFrame({"fear_greed": range(10, 100, 9)}, index=days)
    out = add_fear_greed_column(df, fng)
    assert "fear_greed" in out.columns
    assert out["fear_greed"].notna().all()
    # First day's rows take the first day's value.
    assert out["fear_greed"].iloc[0] == 10


def test_fear_greed_contrarian_signals():
    df = _synth(800)
    # Synthetic sentiment swinging between extreme fear and greed.
    df["fear_greed"] = 50 + 45 * np.sin(np.arange(len(df)) / 50.0)
    sig = get("fear_greed_contrarian")().generate_signals(df)
    assert set(pd.unique(sig.dropna())).issubset({-1, 0, 1})
    assert (sig == 1).sum() >= 1 and (sig == -1).sum() >= 1


def test_fear_greed_contrarian_requires_column():
    df = _synth(200)
    with pytest.raises(KeyError):
        get("fear_greed_contrarian")().generate_signals(df)


# --------------------------------------------------------------------------- position plans
def test_scaled_exit_plan_sums_to_position():
    from src.execution.broker import OrderSide
    from src.execution.position_plan import scaled_exit_plan

    legs = scaled_exit_plan(qty=2.0, range_high=100.0)
    assert all(leg.side == OrderSide.SELL for leg in legs)
    assert abs(sum(leg.qty for leg in legs) - 2.0) < 1e-9
    # Prices are non-decreasing (sell into strength).
    prices = [leg.price for leg in legs]
    assert prices == sorted(prices)


def test_dca_buy_plan_spends_budget_lower_heavy():
    from src.execution.position_plan import dca_buy_plan

    legs = dca_buy_plan(quote_budget=1000.0, low=80.0, high=120.0, levels=4, lower_heavy=True)
    spent = sum(leg.qty * leg.price for leg in legs)
    assert abs(spent - 1000.0) < 1e-6
    # Lower-heavy: the cheapest level carries the largest fraction.
    cheapest = min(legs, key=lambda leg: leg.price)
    assert cheapest.fraction == max(leg.fraction for leg in legs)


def test_stink_bid_plan_prices_below_ref():
    from src.execution.position_plan import stink_bid_plan

    legs = stink_bid_plan(quote_budget=600.0, ref_price=100.0, depths=(0.1, 0.25, 0.4))
    assert all(leg.price < 100.0 for leg in legs)
    assert abs(sum(leg.qty * leg.price for leg in legs) - 600.0) < 1e-6


def test_plan_leg_to_order():
    from src.execution.broker import OrderType
    from src.execution.position_plan import scaled_exit_plan

    leg = scaled_exit_plan(1.0, 100.0)[0]
    order = leg.to_order("BTCUSDT", reduce_only=True)
    assert order.symbol == "BTCUSDT" and order.type == OrderType.LIMIT and order.reduce_only


# --------------------------------------------------------------------------- run_bot macro regime gate
def test_macro_step_aside_trend_break():
    from src.run_bot import compute_macro_step_aside

    # 300 rising daily bars, then a crash on the last bar -> close below the EMA.
    close = pd.Series(list(np.linspace(100, 200, 300)) + [80.0])
    aside, detail = compute_macro_step_aside(close)
    assert aside is True
    assert detail["trend_break"] is True


def test_macro_step_aside_risk_on():
    from src.run_bot import compute_macro_step_aside

    # Steady uptrend, close above its EMA, overheat threshold way out of reach.
    close = pd.Series(np.linspace(100, 200, 300))
    aside, detail = compute_macro_step_aside(close, mayer_top=100.0)
    assert aside is False
    assert detail["trend_break"] is False


def test_macro_step_aside_overheated_threshold():
    from src.run_bot import compute_macro_step_aside

    # Same uptrend but a very low Mayer threshold trips the overheat gate.
    close = pd.Series(np.linspace(100, 200, 300))
    aside, detail = compute_macro_step_aside(close, mayer_top=1.0)
    assert aside is True
    assert detail["overheated"] is True


def test_macro_step_aside_insufficient_history():
    from src.run_bot import compute_macro_step_aside

    aside, detail = compute_macro_step_aside(pd.Series(np.linspace(100, 110, 50)))
    assert aside is False
    assert detail["reason"] == "insufficient_daily_history"
