import pandas as pd

from src.strategy_search import simulate_trades


def test_btc_pnl_scores_short_as_extra_btc_gained():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "tf_15m_open": [100, 100, 90, 90],
            "tf_15m_high": [100, 100, 90, 90],
            "tf_15m_low": [100, 90, 90, 90],
            "tf_15m_close": [100, 90, 90, 90],
        }
    )
    trades = simulate_trades(
        data,
        pd.Series([True, False, False, False]),
        "short",
        horizon_bars=1,
        fee_bps=0,
        slippage_bps=0,
        take_profit=0.1,
        stop_loss=0.1,
        pnl_unit="btc",
    )
    assert round(float(trades["net_return"].iloc[0]), 6) == 0.111111
