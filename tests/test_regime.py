import pandas as pd

from src.regime import (
    REGIME_LABELS,
    _regime_frame,
    add_regime_column,
    add_regime_column_from_daily,
    tag_regime_file,
)


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


def test_regime_labels_are_causal_prefix_invariant_and_semantically_stable():
    rng = pd.Series(range(600), dtype=float)
    returns = pd.concat(
        [
            pd.Series(0.002 + (rng.iloc[:150] % 7 - 3) * 0.0002),
            pd.Series(-0.002 + (rng.iloc[:150] % 5 - 2) * 0.0003),
            pd.Series((rng.iloc[:150] % 2 * 2 - 1) * 0.025),
            pd.Series((rng.iloc[:150] % 3 - 1) * 0.0002),
        ],
        ignore_index=True,
    )
    close = 100.0 * (1.0 + returns).cumprod()
    daily = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=len(close), freq="D", tz="UTC"),
            "close": close,
        }
    )

    prefix = _regime_frame(daily.iloc[:450], "close").set_index("timestamp")
    extended = _regime_frame(daily, "close").set_index("timestamp").reindex(prefix.index)

    pd.testing.assert_series_equal(
        prefix["tf_1d_regime_id"],
        extended["tf_1d_regime_id"],
    )
    observed = set(prefix["tf_1d_regime_id"].unique())
    assert observed <= set(REGIME_LABELS)
    assert {0, 1, 2, 3} <= observed


def test_direct_daily_close_label_is_visible_only_on_following_day():
    timestamps = pd.date_range("2024-01-01", periods=90, freq="D", tz="UTC")
    close = pd.Series(100.0 * (1.002 ** pd.Series(range(90))), dtype=float)
    daily = pd.DataFrame({"timestamp": timestamps, "close": close})
    raw = _regime_frame(daily, "close").set_index("timestamp")["tf_1d_regime_id"]

    tagged = add_regime_column(daily, price_column="close").set_index("timestamp")

    assert tagged.loc[timestamps[-1], "tf_1d_regime_id"] == raw.loc[timestamps[-2]]
    assert tagged.loc[timestamps[-1], "tf_1d_regime_id"] != -1


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
    assert report["regime_labels"] == {str(key): value for key, value in REGIME_LABELS.items()}
    assert output_path.exists()
    tagged = pd.read_parquet(output_path)
    assert "tf_1d_regime_id" in tagged.columns


def test_tag_regime_file_compact_output_drops_heavy_indicator_columns(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=96 * 10, freq="15min", tz="UTC")
    intraday_path = tmp_path / "intraday.parquet"
    daily_path = tmp_path / "daily.parquet"
    output_path = tmp_path / "compact.parquet"
    close = pd.Series(range(100, 100 + len(timestamps)), dtype=float)
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
            "unused_heavy_indicator": 123.0,
        }
    ).to_parquet(intraday_path, index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "close": range(100, 110),
            "unused_daily_indicator": 456.0,
        }
    ).to_parquet(daily_path, index=False)

    report = tag_regime_file(
        intraday_path,
        output_path,
        daily_input_path=daily_path,
        compact=True,
    )

    assert report["compact"] is True
    assert pd.read_parquet(output_path).columns.tolist() == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tf_1d_regime_id",
    ]
