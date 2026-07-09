import pandas as pd

from src.regime import add_regime_column, add_regime_column_from_daily, tag_regime_file


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


def test_add_regime_column_accepts_timestamp_index():
    data = pd.DataFrame(
        {"close": range(100, 180)},
        index=pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC", name="timestamp"),
    )
    out = add_regime_column(data, price_column="close")
    assert "timestamp" in out.columns
    assert "tf_1d_regime_id" in out.columns


def test_add_regime_column_from_daily_merges_to_intraday_rows():
    intraday = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=96 * 80, freq="15min", tz="UTC"),
            "close": range(96 * 80),
        }
    )
    daily = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC"),
            "close": [100 + i + (10 if i % 9 == 0 else 0) for i in range(80)],
        }
    )
    out = add_regime_column_from_daily(intraday, daily)

    assert len(out) == len(intraday)
    assert "tf_1d_regime_id" in out.columns
    assert (out["tf_1d_regime_id"] != -1).sum() > 96


def test_tag_regime_file_skip_if_missing(tmp_path):
    report = tag_regime_file(
        tmp_path / "missing.parquet",
        tmp_path / "out.parquet",
        skip_if_missing=True,
    )
    assert report["ok"] is True
    assert report["skipped"] is True
    assert report["reason"] == "missing_input"
    assert not (tmp_path / "out.parquet").exists()


def test_tag_regime_file_writes_reportable_output_from_daily(tmp_path):
    intraday_path = tmp_path / "intraday.parquet"
    daily_path = tmp_path / "daily.parquet"
    output_path = tmp_path / "tagged.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=96 * 80, freq="15min", tz="UTC"),
            "close": range(96 * 80),
        }
    ).to_parquet(intraday_path, index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC"),
            "close": [100 + i + (10 if i % 9 == 0 else 0) for i in range(80)],
        }
    ).to_parquet(daily_path, index=False)

    report = tag_regime_file(intraday_path, output_path, daily_input_path=daily_path)

    assert report["ok"] is True
    assert report["skipped"] is False
    assert report["rows"] == 96 * 80
    assert output_path.exists()
    tagged = pd.read_parquet(output_path)
    assert "tf_1d_regime_id" in tagged.columns
