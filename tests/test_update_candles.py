from argparse import Namespace

import pandas as pd
import pytest

from src import update_candles


def configure_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(update_candles.bbid, "CANDLE_DIR", tmp_path)
    monkeypatch.setattr(update_candles.bbid, "SYMBOL", "BTCUSDT")
    monkeypatch.setattr(update_candles.bbid, "MARKET", "futures")
    monkeypatch.setattr(update_candles.bbid, "TIMEFRAMES", ["1m", "5m", "15m"])


def candle_frame(**overrides):
    values = {
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "volume": [10.0],
        "quote_asset_volume": [1000.0],
        "number_of_trades": [20],
        "taker_buy_base_volume": [5.0],
        "taker_buy_quote_volume": [500.0],
    }
    values.update(overrides)
    return pd.DataFrame(
        values,
        index=pd.date_range("2026-01-01", periods=len(next(iter(values.values()))), freq="1min", tz="UTC", name="timestamp"),
    )


def kline_row(open_time_ms=1767225600000, **overrides):
    row = {
        "open_time": open_time_ms,
        "open": "100.0",
        "high": "101.0",
        "low": "99.0",
        "close": "100.5",
        "volume": "10.0",
        "close_time": open_time_ms + 59999,
        "quote_asset_volume": "1000.0",
        "number_of_trades": "20",
        "taker_buy_base_volume": "5.0",
        "taker_buy_quote_volume": "500.0",
        "ignore": "0",
    }
    row.update(overrides)
    return [row[column] for column in update_candles.bbid.BINANCE_COLUMNS]


class FakeResponse:
    status_code = 200
    text = "ok"

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_fetch_recent_candles_rejects_invalid_numeric_values(monkeypatch):
    monkeypatch.setattr(
        update_candles.requests,
        "get",
        lambda *args, **kwargs: FakeResponse([kline_row(close="not-a-number")]),
    )

    with pytest.raises(ValueError, match="close must be finite numeric"):
        update_candles.fetch_recent_candles("BTCUSDT", "futures", 1767225600000)


def test_fetch_recent_candles_rejects_non_list_payload(monkeypatch):
    monkeypatch.setattr(
        update_candles.requests,
        "get",
        lambda *args, **kwargs: FakeResponse({"unexpected": "payload"}),
    )

    with pytest.raises(ValueError, match="Binance response must be a list"):
        update_candles.fetch_recent_candles("BTCUSDT", "futures", 1767225600000)


def test_fetch_recent_candles_rejects_malformed_rows(monkeypatch):
    monkeypatch.setattr(
        update_candles.requests,
        "get",
        lambda *args, **kwargs: FakeResponse([kline_row()[:-1]]),
    )

    with pytest.raises(ValueError, match="malformed Binance row 0"):
        update_candles.fetch_recent_candles("BTCUSDT", "futures", 1767225600000)


def test_fetch_recent_candles_rejects_invalid_open_time(monkeypatch):
    monkeypatch.setattr(
        update_candles.requests,
        "get",
        lambda *args, **kwargs: FakeResponse([kline_row(open_time="not-a-time")]),
    )

    with pytest.raises(ValueError, match="open_time must be finite and non-negative"):
        update_candles.fetch_recent_candles("BTCUSDT", "futures", 1767225600000)


def test_fetch_recent_candles_rejects_inconsistent_ohlc(monkeypatch):
    monkeypatch.setattr(
        update_candles.requests,
        "get",
        lambda *args, **kwargs: FakeResponse([kline_row(high="98.0")]),
    )

    with pytest.raises(ValueError, match="OHLC values are internally inconsistent"):
        update_candles.fetch_recent_candles("BTCUSDT", "spot", 1767225600000)


def test_fetch_recent_candles_rejects_duplicate_timestamps(monkeypatch):
    monkeypatch.setattr(
        update_candles.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            [
                kline_row(1767225600000),
                kline_row(1767225600000, close="100.7"),
            ]
        ),
    )

    with pytest.raises(ValueError, match="timestamps must be strictly increasing"):
        update_candles.fetch_recent_candles("BTCUSDT", "futures", 1767225600000)


def test_fetch_recent_candles_rejects_missing_1m_interval(monkeypatch):
    monkeypatch.setattr(
        update_candles.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            [
                kline_row(1767225600000),
                kline_row(1767225720000, close="100.7"),
            ]
        ),
    )

    with pytest.raises(ValueError, match="timestamps must be contiguous 1-minute intervals"):
        update_candles.fetch_recent_candles("BTCUSDT", "futures", 1767225600000)


def test_update_1m_candles_rejects_corrupt_existing_seed(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    monkeypatch.setattr(
        update_candles.bbid,
        "load_existing_1m_candles",
        lambda: candle_frame(low=[102.0]),
    )

    def fail_fetch(*args, **kwargs):
        raise AssertionError("fetch should not run with corrupt existing candles")

    monkeypatch.setattr(update_candles, "fetch_recent_candles", fail_fetch)

    with pytest.raises(ValueError, match="existing 1m candles: OHLC values are internally inconsistent"):
        update_candles.update_1m_candles()


def test_update_1m_candles_rejects_existing_seed_gap_before_fetch(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    gapped = candle_frame(
        open=[100.0, 101.0],
        high=[101.0, 102.0],
        low=[99.0, 100.0],
        close=[100.5, 101.5],
        volume=[10.0, 11.0],
        quote_asset_volume=[1000.0, 1100.0],
        number_of_trades=[20, 21],
        taker_buy_base_volume=[5.0, 5.5],
        taker_buy_quote_volume=[500.0, 550.0],
    )
    gapped.index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-01T00:02:00Z"),
        ],
        name="timestamp",
    )
    monkeypatch.setattr(update_candles.bbid, "load_existing_1m_candles", lambda: gapped)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("fetch should not run with gapped existing candles")

    monkeypatch.setattr(update_candles, "fetch_recent_candles", fail_fetch)

    with pytest.raises(ValueError, match="existing 1m candles: timestamps must be contiguous"):
        update_candles.update_1m_candles()


def test_load_existing_1m_candles_rejects_duplicate_seed_timestamps(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    seed_path = tmp_path / "BTCUSDT_1m.parquet"
    seed = candle_frame().reset_index()
    seed = pd.concat([seed, seed], ignore_index=True)
    seed.to_parquet(seed_path)

    with pytest.raises(ValueError, match="stored 1m candles: timestamps must be strictly increasing"):
        update_candles.bbid.load_existing_1m_candles()


def test_run_update_skip_if_missing_exits_without_fetching(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)

    def fail_fetch():
        raise AssertionError("update_1m_candles should not run")

    monkeypatch.setattr(update_candles, "update_1m_candles", fail_fetch)

    report = update_candles.run_update(["5m"], skip_if_missing=True)

    assert report["ok"] is True
    assert report["skipped"] is True
    assert report["reason"] == "missing_seed_dataset"
    assert report["timeframes"] == ["5m"]


def test_run_update_validates_timeframes_before_skip(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="Unknown timeframes"):
        update_candles.run_update(["2m"], skip_if_missing=True)


def test_run_update_can_select_market_before_path_checks(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    selected = {}

    def configure_market(*, market, legacy_fallback):
        selected["market"] = market
        selected["legacy_fallback"] = legacy_fallback
        monkeypatch.setattr(update_candles.bbid, "MARKET", market)
        monkeypatch.setattr(update_candles.bbid, "CANDLE_DIR", tmp_path / market)

    monkeypatch.setattr(update_candles.bbid, "configure_dataset", configure_market)

    report = update_candles.run_update(["5m"], skip_if_missing=True, market="spot")

    assert selected == {"market": "spot", "legacy_fallback": True}
    assert report["market"] == "spot"
    assert report["reason"] == "missing_seed_dataset"
    assert report["candle_path"].endswith("spot/BTCUSDT_1m.parquet")


def test_run_update_filters_indicator_rebuild_timeframes(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    seed_path = tmp_path / "BTCUSDT_1m.parquet"
    seed_path.write_text("seed", encoding="utf-8")
    df_1m = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [10.0, 20.0],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="1min", tz="UTC", name="timestamp"),
    )
    captured = {}

    monkeypatch.setattr(update_candles, "update_1m_candles", lambda: df_1m)
    def build_timeframes(frame, timeframes=None):
        captured["requested_timeframes"] = list(timeframes or [])
        return {tf: frame for tf in timeframes}

    monkeypatch.setattr(update_candles.bbid, "build_timeframes", build_timeframes)

    def capture_indicator_files(datasets):
        captured["timeframes"] = list(datasets)

    monkeypatch.setattr(update_candles.bbid, "build_indicator_files", capture_indicator_files)

    report = update_candles.run_update(["5m", "15m"], skip_if_missing=True)

    assert report["ok"] is True
    assert report["skipped"] is False
    assert report["rows_1m"] == 2
    assert report["timeframes"] == ["5m", "15m"]
    assert captured["requested_timeframes"] == ["5m", "15m"]
    assert captured["timeframes"] == ["5m", "15m"]


def test_build_timeframes_only_writes_requested_timeframes(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    df_1m = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.1, 2.1, 3.1, 4.1, 5.1],
            "low": [0.9, 1.9, 2.9, 3.9, 4.9],
            "close": [1.05, 2.05, 3.05, 4.05, 5.05],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
            "quote_asset_volume": [100.0, 200.0, 300.0, 400.0, 500.0],
            "number_of_trades": [1, 2, 3, 4, 5],
            "taker_buy_base_volume": [5.0, 10.0, 15.0, 20.0, 25.0],
            "taker_buy_quote_volume": [50.0, 100.0, 150.0, 200.0, 250.0],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="1min", tz="UTC", name="timestamp"),
    )

    datasets = update_candles.bbid.build_timeframes(df_1m, timeframes=["5m"])

    assert list(datasets) == ["5m"]
    assert (tmp_path / "BTCUSDT_5m.parquet").exists()
    assert not (tmp_path / "BTCUSDT_1m.parquet").exists()
    assert not (tmp_path / "BTCUSDT_15m.parquet").exists()


def test_run_update_bootstraps_missing_seed_when_requested(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    df_1m = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [10.0, 20.0],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="1min", tz="UTC", name="timestamp"),
    )
    df_1m.attrs["fetched_rows"] = 2
    df_1m.attrs["bootstrap_days"] = 7
    captured = {}

    monkeypatch.setattr(update_candles, "bootstrap_1m_candles", lambda days: df_1m)
    def build_timeframes(frame, timeframes=None):
        captured["requested_timeframes"] = list(timeframes or [])
        return {tf: frame for tf in timeframes}

    monkeypatch.setattr(update_candles.bbid, "build_timeframes", build_timeframes)
    monkeypatch.setattr(
        update_candles.bbid,
        "build_indicator_files",
        lambda datasets: captured.setdefault("timeframes", list(datasets)),
    )

    def fail_update():
        raise AssertionError("update_1m_candles should not run when bootstrapping a missing seed")

    monkeypatch.setattr(update_candles, "update_1m_candles", fail_update)

    report = update_candles.run_update(["5m"], skip_if_missing=True, bootstrap_days=7)

    assert report["ok"] is True
    assert report["skipped"] is False
    assert report["rows_1m"] == 2
    assert report["fetched_rows"] == 2
    assert report["bootstrap_days"] == 7
    assert report["timeframes"] == ["5m"]
    assert captured["requested_timeframes"] == ["5m"]
    assert captured["timeframes"] == ["5m"]


def test_run_update_reports_bootstrap_error_without_rebuilding(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    df_1m = pd.DataFrame()
    df_1m.attrs["fetch_error"] = "no bootstrap candles returned"
    df_1m.attrs["bootstrap_days"] = 3

    monkeypatch.setattr(update_candles, "bootstrap_1m_candles", lambda days: df_1m)

    def fail_build_timeframes(_frame):
        raise AssertionError("build_timeframes should not run after bootstrap failure")

    monkeypatch.setattr(update_candles.bbid, "build_timeframes", fail_build_timeframes)

    report = update_candles.run_update(["5m"], skip_if_missing=True, bootstrap_days=3)

    assert report["ok"] is False
    assert report["reason"] == "fetch_error"
    assert report["error"] == "no bootstrap candles returned"
    assert report["bootstrap_days"] == 3


def test_run_update_reports_empty_bootstrap_without_rebuilding(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    df_1m = pd.DataFrame()
    df_1m.attrs["fetched_rows"] = 0
    df_1m.attrs["bootstrap_days"] = 3

    monkeypatch.setattr(update_candles, "bootstrap_1m_candles", lambda days: df_1m)

    def fail_build_timeframes(_frame):
        raise AssertionError("build_timeframes should not run with an empty 1m seed")

    monkeypatch.setattr(update_candles.bbid, "build_timeframes", fail_build_timeframes)

    report = update_candles.run_update(["5m"], skip_if_missing=True, bootstrap_days=3)

    assert report["ok"] is False
    assert report["skipped"] is True
    assert report["reason"] == "empty_seed_dataset"
    assert report["rows_1m"] == 0
    assert report["bootstrap_days"] == 3
    assert report["timeframes"] == ["5m"]


def test_run_update_rejects_negative_bootstrap_days(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="bootstrap_days must be non-negative"):
        update_candles.run_update(["5m"], bootstrap_days=-1)


def test_bootstrap_rejects_unbounded_window():
    with pytest.raises(ValueError, match="<= 366"):
        update_candles.bootstrap_1m_candles(367)


def test_run_update_reports_fetch_error_without_rebuilding_indicators(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    df_1m = pd.DataFrame(
        {"close": [1.0]},
        index=pd.date_range("2026-01-01", periods=1, freq="1min", tz="UTC", name="timestamp"),
    )
    df_1m.attrs["fetch_error"] = "network unavailable"

    monkeypatch.setattr(update_candles, "update_1m_candles", lambda: df_1m)

    def fail_build_timeframes(_frame):
        raise AssertionError("build_timeframes should not run after fetch failure")

    monkeypatch.setattr(update_candles.bbid, "build_timeframes", fail_build_timeframes)

    report = update_candles.run_update(["15m"], skip_if_missing=False)

    assert report["ok"] is False
    assert report["skipped"] is True
    assert report["reason"] == "fetch_error"
    assert report["error"] == "network unavailable"
    assert report["timeframes"] == ["15m"]


def test_run_update_treats_fetch_error_as_warning_when_existing_seed_is_fresh(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    seed_path = tmp_path / "BTCUSDT_1m.parquet"
    seed_path.write_text("seed", encoding="utf-8")
    df_1m = pd.DataFrame(
        {"close": [1.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=2)],
            name="timestamp",
        ),
    )
    df_1m.attrs["fetch_error"] = "network unavailable"

    monkeypatch.setattr(update_candles, "update_1m_candles", lambda: df_1m)

    def fail_build_timeframes(_frame):
        raise AssertionError("build_timeframes should not run after fetch failure")

    monkeypatch.setattr(update_candles.bbid, "build_timeframes", fail_build_timeframes)

    report = update_candles.run_update(["15m"], skip_if_missing=False)

    assert report["ok"] is True
    assert report["skipped"] is True
    assert report["reason"] == "fetch_error_existing_seed_fresh"
    assert report["warning"] == "network unavailable"
    assert report["rows_1m"] == 1
    assert report["seed_age_seconds"] <= update_candles.DEFAULT_FETCH_ERROR_GRACE_SECONDS


def test_run_update_reports_empty_existing_seed_without_rebuilding(monkeypatch, tmp_path):
    configure_dataset(monkeypatch, tmp_path)
    seed_path = tmp_path / "BTCUSDT_1m.parquet"
    seed_path.write_text("seed", encoding="utf-8")
    df_1m = pd.DataFrame()
    df_1m.attrs["fetched_rows"] = 0

    monkeypatch.setattr(update_candles, "update_1m_candles", lambda: df_1m)

    def fail_build_timeframes(_frame):
        raise AssertionError("build_timeframes should not run with an empty 1m seed")

    monkeypatch.setattr(update_candles.bbid, "build_timeframes", fail_build_timeframes)

    report = update_candles.run_update(["15m"], skip_if_missing=False)

    assert report["ok"] is False
    assert report["skipped"] is True
    assert report["reason"] == "empty_seed_dataset"
    assert report["rows_1m"] == 0
    assert report["timeframes"] == ["15m"]


def test_main_exits_zero_when_report_is_ok(monkeypatch, capsys):
    monkeypatch.setattr(update_candles, "configure_logging", lambda: None)
    monkeypatch.setattr(
        update_candles,
        "parse_args",
        lambda: Namespace(timeframes=["5m"], skip_if_missing=True, market="spot", bootstrap_days=0),
    )
    monkeypatch.setattr(update_candles, "run_update", lambda *args, **kwargs: {"ok": True})

    with pytest.raises(SystemExit) as exc:
        update_candles.main()

    assert exc.value.code == 0
    assert '"ok": true' in capsys.readouterr().out


def test_main_exits_nonzero_when_report_fails(monkeypatch, capsys):
    monkeypatch.setattr(update_candles, "configure_logging", lambda: None)
    monkeypatch.setattr(
        update_candles,
        "parse_args",
        lambda: Namespace(timeframes=["5m"], skip_if_missing=True, market="spot", bootstrap_days=1),
    )
    monkeypatch.setattr(
        update_candles,
        "run_update",
        lambda *args, **kwargs: {"ok": False, "reason": "empty_seed_dataset"},
    )

    with pytest.raises(SystemExit) as exc:
        update_candles.main()

    assert exc.value.code == 1
    assert '"reason": "empty_seed_dataset"' in capsys.readouterr().out
