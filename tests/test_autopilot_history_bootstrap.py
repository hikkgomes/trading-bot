import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.autopilot import history_bootstrap as hb


def candle_frame(start, periods, timeframe="1m"):
    index = pd.date_range(
        start,
        periods=periods,
        freq=hb.TIMEFRAME_DELTAS[timeframe],
        tz="UTC",
        name="timestamp",
    )
    sequence = pd.Series(range(periods), dtype=float).to_numpy()
    return pd.DataFrame(
        {
            "open": 100.0 + sequence,
            "high": 101.0 + sequence,
            "low": 99.0 + sequence,
            "close": 100.5 + sequence,
            "volume": 10.0 + sequence,
            "quote_asset_volume": 1_000.0 + sequence,
            "number_of_trades": 20.0 + sequence,
            "taker_buy_base_volume": 5.0 + sequence,
            "taker_buy_quote_volume": 500.0 + sequence,
        },
        index=index,
    )


def kline_row(open_time_ms):
    return [
        open_time_ms,
        "100",
        "101",
        "99",
        "100.5",
        "10",
        open_time_ms + 59_999,
        "1000",
        20,
        "5",
        "500",
        "0",
    ]


class FakeResponse:
    status_code = 200
    text = "ok"

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def patch_data_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hb,
        "candle_data_dir",
        lambda symbol, market, legacy_fallback: tmp_path / market / symbol,
    )
    monkeypatch.setattr(
        hb,
        "indicator_data_dir",
        lambda symbol, market, legacy_fallback: tmp_path / market / symbol / "indicators",
    )


def write_factory_config(tmp_path, mutate):
    payload = json.loads(Path("config/research_factory.json").read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "research_factory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_plan_uses_configured_search_space_timeframes_and_short_spot_1m_seed():
    now = pd.Timestamp("2026-07-09T12:00:00Z")
    requirements = hb.build_default_requirements(now=now)
    by_key = {(item.market, item.timeframe): item for item in requirements}

    assert set(by_key) == {
        ("futures", "1m"),
        ("futures", "5m"),
        ("futures", "15m"),
        ("futures", "1h"),
        ("futures", "4h"),
        ("futures", "1d"),
        ("spot", "1m"),
        ("spot", "1h"),
        ("spot", "4h"),
        ("spot", "1d"),
        ("spot", "1w"),
    }
    assert by_key[("futures", "5m")].start < pd.Timestamp("2023-01-01T00:00:00Z")
    assert hb.DEFAULT_FEATURES <= by_key[("futures", "5m")].required_features
    assert hb.DEFAULT_FEATURES <= by_key[("spot", "1w")].required_features
    assert by_key[("futures", "5m")].scenario_names == (
        "active_income_day",
        "active_income_scalping",
    )
    spot_seed = by_key[("spot", "1m")]
    assert spot_seed.build_indicators is True
    assert spot_seed.required_features == frozenset({"volume_z_20"})
    assert spot_seed.start == now - pd.Timedelta(days=hb.OPERATIONAL_SEED_DAYS)


def test_factory_timeframe_change_is_reflected_in_plan(tmp_path):
    def use_30m_setup(payload):
        space = next(
            item for item in payload["search_spaces"] if item["name"] == "active_income_day"
        )
        space["setup_timeframe"] = "30m"

    config_path = write_factory_config(tmp_path, use_30m_setup)

    requirements = hb.build_default_requirements(
        config_path=config_path,
        markets=["futures"],
        now="2026-07-09T12:00:00Z",
    )
    by_key = {(item.market, item.timeframe): item for item in requirements}

    assert ("futures", "30m") in by_key
    assert "active_income_day" in by_key[("futures", "30m")].scenario_names


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["search_spaces"][0].update({"analysis_timeframe": "30m"}),
            "unknown fields: analysis_timeframe",
        ),
        (
            lambda payload: payload["search_spaces"][0].update({"setup_timeframe": "2h"}),
            "unknown timeframe '2h'",
        ),
    ],
)
def test_history_plan_strictly_rejects_unrecognized_search_space_timeframes(
    tmp_path,
    mutate,
    message,
):
    config_path = write_factory_config(tmp_path, mutate)

    with pytest.raises(ValueError, match=message):
        hb.build_default_requirements(config_path=config_path)


def test_history_plan_cli_uses_explicit_factory_config(tmp_path, monkeypatch, capsys):
    def use_30m_setup(payload):
        space = next(
            item for item in payload["search_spaces"] if item["name"] == "active_income_day"
        )
        space["setup_timeframe"] = "30m"

    config_path = write_factory_config(tmp_path, use_30m_setup)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "history_bootstrap",
            "--config",
            str(config_path),
            "--market",
            "futures",
            "--plan",
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        hb.main()

    report = json.loads(capsys.readouterr().out)
    assert stopped.value.code == 0
    assert report["research_factory_config"] == str(config_path.resolve())
    assert any(item["timeframe"] == "30m" for item in report["datasets"])


def test_scheduled_partition_exclusion_cannot_hide_new_non_1m_timeframes(tmp_path):
    def use_30m_setup(payload):
        space = next(
            item for item in payload["search_spaces"] if item["name"] == "active_income_day"
        )
        space["setup_timeframe"] = "30m"

    config_path = write_factory_config(tmp_path, use_30m_setup)

    coarse = hb.build_default_requirements(
        config_path=config_path,
        markets=["futures"],
        exclude_timeframes=["1m"],
    )

    assert all(item.timeframe != "1m" for item in coarse)
    assert any(item.timeframe == "30m" for item in coarse)


def test_every_planned_indicator_feature_is_buildable_from_native_ohlcv():
    requirements = hb.build_default_requirements(now="2026-07-09T12:00:00Z")

    for requirement in requirements:
        if not requirement.required_features:
            continue
        frame = candle_frame("2025-01-06T00:00:00Z", 300, requirement.timeframe)
        indicators = hb.bbid.build_indicator_features(
            frame,
            requirement.timeframe,
            required_features=requirement.required_features,
        )
        assert requirement.required_features <= set(indicators.columns), (
            requirement.market,
            requirement.timeframe,
            sorted(requirement.required_features - set(indicators.columns)),
        )


def test_fetch_kline_page_uses_market_endpoint_and_validates_payload():
    captured = {}
    first = int(pd.Timestamp("2026-01-01T00:00:00Z").value // 1_000_000)

    def fake_get(url, *, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return FakeResponse([kline_row(first), kline_row(first + 60_000)])

    frame = hb.fetch_kline_page(
        symbol="BTCUSDT",
        market="futures",
        timeframe="1m",
        start_ms=first,
        end_ms=first + 60_000,
        request_get=fake_get,
    )

    assert captured["url"] == "https://fapi.binance.com/fapi/v1/klines"
    assert captured["params"]["interval"] == "1m"
    assert list(frame.columns) == hb.bbid.CANDLE_COLUMNS[1:]
    assert len(frame) == 2

    with pytest.raises(ValueError, match="response must be a list"):
        hb.fetch_kline_page(
            symbol="BTCUSDT",
            market="spot",
            timeframe="1m",
            start_ms=first,
            end_ms=first,
            request_get=lambda *args, **kwargs: FakeResponse({"bad": True}),
        )


def test_validate_candle_frame_rejects_cadence_gap():
    frame = candle_frame("2026-01-01", 3).iloc[[0, 2]]

    with pytest.raises(ValueError, match="contiguous 1m intervals"):
        hb.validate_candle_frame(frame, "1m", label="test")


def test_sync_requirement_atomically_writes_candles_and_pruned_indicators(
    tmp_path,
    monkeypatch,
):
    patch_data_paths(monkeypatch, tmp_path)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    calls = []

    def fetch_page(**kwargs):
        calls.append(kwargs)
        left = pd.to_datetime(kwargs["start_ms"], unit="ms", utc=True)
        right = pd.to_datetime(kwargs["end_ms"], unit="ms", utc=True)
        periods = int((right - left) / pd.Timedelta(minutes=1)) + 1
        return candle_frame(left, min(periods, kwargs["limit"]))

    requirement = hb.HistoryRequirement(
        market="futures",
        timeframe="1m",
        start=start,
        required_features=frozenset({"rsi_14", "volume_z_20"}),
        scenario_names=("test",),
    )
    result = hb.sync_requirement(
        requirement,
        now="2026-01-01T00:41:00Z",
        request_delay_seconds=0,
        checkpoint_pages=1,
        fetch_page=fetch_page,
    )

    candle_path = tmp_path / "futures" / "BTCUSDT" / "BTCUSDT_1m.parquet"
    indicator_path = (
        tmp_path / "futures" / "BTCUSDT" / "indicators" / "BTCUSDT_1m_all_indicators.parquet"
    )
    assert result["rows"] == 41
    assert candle_path.exists()
    indicators = pd.read_parquet(indicator_path)
    assert {"open", "high", "low", "close", "rsi_14", "volume_z_20"}.issubset(indicators.columns)
    assert "ema_200" not in indicators.columns
    assert not list(candle_path.parent.glob("*.tmp"))

    calls.clear()
    updated = hb.sync_requirement(
        requirement,
        now="2026-01-01T00:43:00Z",
        request_delay_seconds=0,
        fetch_page=fetch_page,
    )
    assert updated["rows"] == 43
    assert len(calls) == 1


def test_failed_download_persists_checkpoint_and_resume_finishes(tmp_path, monkeypatch):
    patch_data_paths(monkeypatch, tmp_path)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    requirement = hb.HistoryRequirement(
        market="spot",
        timeframe="1m",
        start=start,
        required_features=frozenset(),
        scenario_names=("operational",),
        build_indicators=False,
    )
    attempts = 0

    def failing_fetch(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("temporary network failure")
        left = pd.to_datetime(kwargs["start_ms"], unit="ms", utc=True)
        return candle_frame(left, kwargs["limit"])

    with pytest.raises(RuntimeError, match="temporary network failure"):
        hb.sync_requirement(
            requirement,
            now="2026-01-01T16:42:00Z",
            request_delay_seconds=0,
            checkpoint_pages=1,
            fetch_page=failing_fetch,
        )

    checkpoint = tmp_path / "spot" / "BTCUSDT" / ".BTCUSDT_1m.history_checkpoint.parquet"
    assert checkpoint.exists()
    assert len(pd.read_parquet(checkpoint)) == 1_000

    def resumed_fetch(**kwargs):
        left = pd.to_datetime(kwargs["start_ms"], unit="ms", utc=True)
        right = pd.to_datetime(kwargs["end_ms"], unit="ms", utc=True)
        periods = min(int((right - left) / pd.Timedelta(minutes=1)) + 1, kwargs["limit"])
        return candle_frame(left, periods)

    result = hb.sync_requirement(
        requirement,
        now="2026-01-01T16:42:00Z",
        request_delay_seconds=0,
        checkpoint_pages=1,
        fetch_page=resumed_fetch,
    )

    assert result["rows"] == 1_002
    assert not checkpoint.exists()
    manifest = json.loads((tmp_path / "spot" / "BTCUSDT" / ".BTCUSDT_1m.history.json").read_text())
    assert manifest["prefix_complete"] is True


def test_api_page_budget_stops_and_checkpoints_for_next_run(tmp_path, monkeypatch):
    patch_data_paths(monkeypatch, tmp_path)
    requirement = hb.HistoryRequirement(
        market="futures",
        timeframe="1m",
        start=pd.Timestamp("2026-01-01T00:00:00Z"),
        required_features=frozenset(),
        scenario_names=("budget",),
        build_indicators=False,
    )

    def fetch_page(**kwargs):
        left = pd.to_datetime(kwargs["start_ms"], unit="ms", utc=True)
        return candle_frame(left, kwargs["limit"])

    with pytest.raises(RuntimeError, match="API page budget exhausted at 1"):
        hb.sync_requirement(
            requirement,
            now="2026-01-01T16:42:00Z",
            request_delay_seconds=0,
            max_request_pages=1,
            fetch_page=fetch_page,
        )

    checkpoint = tmp_path / "futures" / "BTCUSDT" / ".BTCUSDT_1m.history_checkpoint.parquet"
    assert checkpoint.exists()
    assert len(pd.read_parquet(checkpoint)) == 1_000


def test_empty_incremental_response_cannot_claim_stale_history_is_complete(tmp_path, monkeypatch):
    patch_data_paths(monkeypatch, tmp_path)
    candle_path = tmp_path / "futures" / "BTCUSDT" / "BTCUSDT_1m.parquet"
    candle_path.parent.mkdir(parents=True)
    original = candle_frame("2026-01-01T00:00:00Z", 2)
    original.to_parquet(candle_path)
    requirement = hb.HistoryRequirement(
        market="futures",
        timeframe="1m",
        start=pd.Timestamp("2026-01-01T00:00:00Z"),
        required_features=frozenset(),
        scenario_names=("test",),
        build_indicators=False,
    )

    with pytest.raises(RuntimeError, match="incomplete Binance history"):
        hb.sync_requirement(
            requirement,
            now="2026-01-01T00:05:00Z",
            request_delay_seconds=0,
            fetch_page=lambda **kwargs: pd.DataFrame(columns=hb.bbid.CANDLE_COLUMNS),
        )

    assert len(pd.read_parquet(candle_path)) == 2


def test_stale_manifest_cannot_hide_truncated_history_prefix(tmp_path, monkeypatch):
    patch_data_paths(monkeypatch, tmp_path)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    candle_path = tmp_path / "futures" / "BTCUSDT" / "BTCUSDT_1m.parquet"
    manifest_path = candle_path.parent / ".BTCUSDT_1m.history.json"
    candle_path.parent.mkdir(parents=True)
    complete = candle_frame(start, 6)
    complete.iloc[2:].to_parquet(candle_path)
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "prefix_checked_from": start.isoformat(),
                "prefix_complete": True,
                "first_timestamp": start.isoformat(),
                "last_timestamp": complete.index.max().isoformat(),
                "rows": len(complete),
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fetch_page(**kwargs):
        calls.append(kwargs)
        left = pd.to_datetime(kwargs["start_ms"], unit="ms", utc=True)
        right = pd.to_datetime(kwargs["end_ms"], unit="ms", utc=True)
        periods = int((right - left) / pd.Timedelta(minutes=1)) + 1
        return candle_frame(left, periods)

    result = hb.sync_requirement(
        hb.HistoryRequirement(
            market="futures",
            timeframe="1m",
            start=start,
            required_features=frozenset(),
            scenario_names=("prefix-repair",),
            build_indicators=False,
        ),
        now="2026-01-01T00:06:00Z",
        request_delay_seconds=0,
        fetch_page=fetch_page,
    )

    repaired = pd.read_parquet(candle_path)
    assert result["rows"] == 6
    assert pd.to_datetime(repaired.index.min(), utc=True) == start
    assert any(pd.to_datetime(call["start_ms"], unit="ms", utc=True) == start for call in calls)


def test_run_history_bootstrap_writes_failure_report_with_resume_remediation(
    tmp_path,
    monkeypatch,
):
    patch_data_paths(monkeypatch, tmp_path)
    report_path = tmp_path / "report.json"

    report = hb.run_history_bootstrap(
        markets=["spot"],
        timeframes=["1m"],
        now="2026-01-08T00:00:00Z",
        request_delay_seconds=0,
        fetch_page=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
        report_path=report_path,
    )

    assert report["ok"] is False
    assert report["datasets"][0]["ok"] is False
    assert "checkpoint resumes" in report["datasets"][0]["remediation"]
    assert json.loads(report_path.read_text())["ok"] is False
