import pandas as pd

from src.regime import add_regime_column


def test_add_regime_column_emits_regime_id():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"),
            "tf_1d_close": range(100, 160),
        }
    )
    out = add_regime_column(data)
    assert "tf_1d_regime_id" in out.columns


def test_add_regime_column_handles_15m_repeated_daily_close():
    timestamps = pd.date_range("2024-01-01", periods=96 * 80, freq="15min", tz="UTC")
    daily_values = []
    price = 100.0
    for day in range(80):
        price *= 1.0 + (0.01 if day % 5 else -0.03)
        daily_values.extend([price] * 96)
    data = pd.DataFrame({"timestamp": timestamps, "tf_1d_close": daily_values})
    out = add_regime_column(data)
    regimes = out["tf_1d_regime_id"]
    assert regimes.nunique() > 1
    assert (regimes != -1).sum() > 96
