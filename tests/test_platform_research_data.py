import pytest

from src.data.bar_window import RecentBarWindow
from src.data.okx_history import download_history
from src.research.dataset_service import DatabaseDatasetBundleService
from src.research.datasets import CanonicalResearchDatasetBuilder
from src.research.executors import DataUnavailableError, _product_accounting
from src.research.funding import attach_settlements, settlement_index


@pytest.mark.parametrize("metadata", [False, True])
def test_declared_minute_cadence_rejects_daily_rows_without_optional_metadata(metadata):
    import datetime as dt

    from src.research.dataset_service import DatasetBundleBuildError, _validate_market_bars

    rows = tuple(
        {
            "instrument_id": "BTC",
            "close_timestamp": (
                dt.datetime(2026, 9, 1, tzinfo=dt.UTC) + dt.timedelta(days=i)
            ).isoformat(),
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
            "volume": 1,
            **({"timeframe": "1m"} if metadata else {}),
        }
        for i in range(8)
    )
    with pytest.raises(DatasetBundleBuildError, match="cadence"):
        _validate_market_bars(rows, market_type="spot", timeframe="1m")


@pytest.mark.parametrize("timeframe,seconds", [("1m", 60), ("1d", 86400)])
def test_declared_cadence_accepts_correct_rows_and_rejects_conflicting_metadata(timeframe, seconds):
    import datetime as dt

    from src.research.dataset_service import DatasetBundleBuildError, _validate_market_bars

    start = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)
    rows = tuple(
        {
            "instrument_id": "BTC",
            "close_timestamp": (start + dt.timedelta(seconds=seconds * i)).isoformat(),
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
            "volume": 1,
        }
        for i in range(8)
    )
    _validate_market_bars(rows, market_type="spot", timeframe=timeframe)
    with pytest.raises(DatasetBundleBuildError, match="timeframe"):
        _validate_market_bars(
            ({**rows[0], "timeframe": "5m"},), market_type="spot", timeframe=timeframe
        )


def test_configured_research_plans_fit_budget_without_changing_declared_cadence():
    from pathlib import Path

    from src.research.data_plan import ResearchDataPlan, product_data_plan
    from src.services.config import load_split_configuration, validate_split_configuration

    config = load_split_configuration(Path("config"))
    validate_split_configuration(config)
    active = next(p for p in config["products"]["products"] if p["product_id"] == "active_income")
    plan = product_data_plan(active, config["research"], ["BTCUSDT", "SOLUSDT"])
    plan.validate_capacity(active["research_max_symbols"])
    assert plan.required_rows == 2 * (365 * 96 + 201)
    assert ResearchDataPlan.from_payload(plan.to_payload()) == plan
    start, end = plan.acquisition_interval("2026-09-10T00:00:00+00:00")
    assert end - start == plan.rows_per_instrument * 900_000
    with pytest.raises(ValueError, match="row/byte budget"):
        product_data_plan(
            {**active, "dataset_timeframe": "1m"}, config["research"], ["BTCUSDT", "SOLUSDT"]
        )


def test_missing_funding_is_unavailable_instead_of_measured_zero():
    rows = [
        {
            "instrument_id": "binance:futures:BTCUSDT:USDT",
            "close": 100,
            "timestamp": "2026-09-07T00:00:00+00:00",
        }
    ]
    payload = CanonicalResearchDatasetBuilder._bar_payload(rows)
    assert payload["funding_available"] is False
    assert payload["funding_rates"] is None
    with pytest.raises(DataUnavailableError, match="funding history"):
        _product_accounting({**payload, "product_id": "active_income"})


def test_native_funding_preserves_each_event_mark_and_excludes_old_settlements():
    instrument = "binance:futures:BTCUSDT:USDT"
    times = [f"2026-09-07T{hour:02}:00:00+00:00" for hour in (0, 8, 16)]
    records = [
        {
            "id": str(index),
            "created_at": "2026-09-08T00:00:00+00:00",
            "payload": {
                "kind": "historical_funding",
                "product_id": "active_income",
                "instrument_id": instrument,
                "funding_time": time,
                "funding_rate": rate,
                "mark_price": mark,
                "source": "native_test_fixture",
            },
        }
        for index, (time, rate, mark) in enumerate(
            zip(times, (0.03, 0.01, -0.02), (99, 110, 120), strict=True)
        )
    ]
    funding = settlement_index(
        records, product_id="active_income", created="2026-09-09T00:00:00+00:00"
    )
    bars = attach_settlements(
        (
            {
                "instrument_id": instrument,
                "open_timestamp": "2026-09-07T01:00:00+00:00",
                "close_timestamp": "2026-09-07T17:00:00+00:00",
                "close": 125,
            },
        ),
        funding,
    )
    assert [event["timestamp"] for event in bars[0]["funding_events"]] == times[1:]
    assert [event["mark_price"] for event in bars[0]["funding_events"]] == [110, 120]
    from src.research.accounting import FuturesResearchAccounting

    report = FuturesResearchAccounting().evaluate(
        initial_cash=1000,
        events=[
            {
                "type": "fill",
                "timestamp": "2026-09-07T01:00:00+00:00",
                "symbol": instrument,
                "side": "buy",
                "quantity": 2,
                "price": 100,
                "fee": 0,
            },
            *bars[0]["funding_events"],
        ],
    )
    assert report.funding_pnl == pytest.approx(-2 * 110 * 0.01 + 2 * 120 * 0.02)


def test_indicative_funding_is_not_a_settlement_and_missing_mark_is_unavailable():
    row = {
        "created_at": "2026-09-07T00:00:00+00:00",
        "payload": {
            "kind": "market_data_input",
            "product_id": "active_income",
            "values": {"funding": 0.01},
        },
    }
    assert (
        settlement_index([row], product_id="active_income", created="2026-09-08T00:00:00+00:00")
        == {}
    )
    row["payload"] = {
        "kind": "historical_funding",
        "product_id": "active_income",
        "funding_rate": 0.01,
    }
    with pytest.raises(ValueError, match="measured mark"):
        settlement_index([row], product_id="active_income", created="2026-09-08T00:00:00+00:00")


@pytest.mark.parametrize("funding_interval_hours", [1, 4, 8])
def test_native_funding_coverage_exhausts_variable_settlement_schedules(
    tmp_path, monkeypatch, funding_interval_hours
):
    import datetime as dt
    from dataclasses import replace

    from sqlalchemy import select

    from src.data.database import PlatformDatabase, funding_entry, historical_bootstrap_progress
    from src.services import historical_bootstrap

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'coverage.sqlite3'}")
    database.create_schema()
    monkeypatch.setattr(historical_bootstrap, "FUNDING_PAGE_SIZE", 2)
    hour = 3_600_000
    times = [13, hour + 27, (1 + funding_interval_hours) * hour + 1]
    range_end_ms = (funding_interval_hours + 2) * hour
    calls = []

    def fetch(*, symbol, start_ms, end_ms):
        calls.append(start_ms)
        return [
            {"symbol": symbol, "fundingTime": time, "fundingRate": "0.001", "markPrice": "100"}
            for time in times
            if start_ms <= time < end_ms
        ][:2]

    service = historical_bootstrap.HistoricalBootstrapService(engine=database.engine, root=tmp_path)
    monkeypatch.setattr(service, "_fetch_funding_page", fetch)
    arguments = {
        "product_id": "active_income",
        "symbol": "BTCUSDT",
        "range_start_ms": 0,
        "range_end_ms": range_end_ms,
        "maximum_pages": 2,
    }
    result = service.bootstrap_funding(**arguments)
    assert result.completed and not result.gaps
    with database.engine.connect() as connection:
        entries = connection.execute(select(funding_entry.c.payload)).scalars().all()
        status = connection.execute(select(historical_bootstrap_progress.c.status)).scalar_one()
        progress_payload = connection.execute(
            select(historical_bootstrap_progress.c.payload)
        ).scalar_one()
    assert status == "completed"
    assert (
        sorted(
            int(dt.datetime.fromisoformat(row["funding_time"]).timestamp() * 1000)
            for row in entries
        )
        == times
    )
    assert progress_payload["coverage_classification"] == "native_query_exhausted"
    assert len(progress_payload["pages"]) == 2
    assert all(page["response_hash"].startswith("sha256:") for page in progress_payload["pages"])
    assert calls == [0, times[1] + 1]
    resumed = service.bootstrap_funding(**arguments)
    assert resumed.completed and resumed.resumed and resumed.rows == 0
    assert calls == [0, times[1] + 1]
    empty = service.bootstrap_funding(
        product_id="active_income",
        symbol="SOLUSDT",
        range_start_ms=range_end_ms + hour,
        range_end_ms=range_end_ms + 2 * hour,
    )
    assert not empty.completed and empty.gaps
    monkeypatch.setattr(service, "_fetch_metadata", lambda **_: {"BTCUSDT"})
    monkeypatch.setattr(service, "bootstrap", lambda **_: replace(result, completed=True, gaps=()))
    monkeypatch.setattr(service, "bootstrap_funding", lambda **_: result)
    combined = service.bootstrap_product(
        product_id="active_income",
        market="futures",
        symbols=("BTCUSDT",),
        timeframe="1h",
        minimum_history_days=1,
        now="1970-01-02T00:00:00+00:00",
    )[0]
    assert combined.completed
    assert combined.gaps == result.gaps


def test_funding_pagination_records_exact_native_page_evidence(tmp_path, monkeypatch):
    from sqlalchemy import select, update

    from src.data.database import PlatformDatabase, funding_entry, historical_bootstrap_progress
    from src.domain._codec import canonical_hash
    from src.research.accounting import FuturesResearchAccounting
    from src.services import historical_bootstrap

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'pages.sqlite3'}")
    database.create_schema()
    service = historical_bootstrap.HistoricalBootstrapService(engine=database.engine, root=tmp_path)
    monkeypatch.setattr(historical_bootstrap, "FUNDING_PAGE_SIZE", 2)
    hour = 3_600_000
    calls = []

    def fetch(*, symbol, start_ms, end_ms):
        calls.append((symbol, start_ms, end_ms))
        return [
            {"symbol": symbol, "fundingTime": time, "fundingRate": "0.001", "markPrice": "100"}
            for time in (hour, 2 * hour, 3 * hour)
            if symbol == "BTCUSDT" and start_ms <= time < end_ms
        ][:2]

    monkeypatch.setattr(service, "_fetch_funding_page", fetch)
    request = {"product_id": "active_income", "symbol": "BTCUSDT", "range_start_ms": 0}
    result = service.bootstrap_funding(**request, range_end_ms=4 * hour)
    assert result.completed and result.rows == 3
    assert calls == [("BTCUSDT", 0, 4 * hour), ("BTCUSDT", 2 * hour + 1, 4 * hour)]
    refreshed = service.bootstrap_funding(**request, range_end_ms=6 * hour)
    assert refreshed.completed and refreshed.rows == 0
    assert calls[-1] == ("BTCUSDT", 4 * hour, 6 * hour)
    with database.engine.connect() as connection:
        records = connection.execute(select(funding_entry)).mappings().all()
        proof = connection.execute(
            select(historical_bootstrap_progress.c.payload).where(
                historical_bootstrap_progress.c.range_end == "1970-01-01T06:00:00+00:00"
            )
        ).scalar_one()
    assert proof["coverage_classification"] == "native_query_exhausted"
    assert proof["request_bounds"] == {"start_ms": 4 * hour, "end_ms": 6 * hour}
    assert proof["pages"] == [
        {
            "request_start_ms": 4 * hour,
            "request_end_ms": 6 * hour - 1,
            "returned_rows": 0,
            "response_hash": canonical_hash([]),
            "first_funding_time_ms": None,
            "last_funding_time_ms": None,
            "next_start_ms": 6 * hour,
        }
    ]
    assert len(proof["pages"][0]["response_hash"]) == 71
    events = settlement_index(
        records, product_id="active_income", created="2099-01-01T00:00:00+00:00"
    )
    symbol = "binance:futures:BTCUSDT:USDT"
    accounting = FuturesResearchAccounting().evaluate(
        initial_cash=1000,
        events=[
            {
                "type": "fill",
                "timestamp": "1970-01-01T00:00:00+00:00",
                "symbol": symbol,
                "side": "buy",
                "quantity": 1,
                "price": 100,
                "fee": 0,
            },
            *events[symbol].values(),
        ],
    )
    assert accounting.funding_pnl == pytest.approx(-0.3)
    missing = service.bootstrap_funding(**request, range_end_ms=12 * hour)
    assert missing.completed and not missing.gaps
    absent = service.bootstrap_funding(**{**request, "symbol": "SOLUSDT"}, range_end_ms=hour)
    assert not absent.completed and absent.gaps
    with database.engine.begin() as connection:
        connection.execute(update(funding_entry).values(created_at="2099-01-01T00:00:00+00:00"))
    future = service.bootstrap_funding(**request, range_end_ms=7 * hour)
    assert not future.completed and future.gaps


def test_funding_precision_correction_preserves_history_without_double_charging(tmp_path):
    import datetime as dt

    from sqlalchemy import insert, select

    from src.data.database import PlatformDatabase, funding_entry
    from src.domain._codec import canonical_hash
    from src.research.accounting import FuturesResearchAccounting
    from src.services.historical_bootstrap import HistoricalBootstrapService

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'funding.sqlite3'}")
    database.create_schema()
    service = HistoricalBootstrapService(engine=database.engine, root=tmp_path)
    instant = "2026-09-07T00:00:00.013000+00:00"
    rows = service._normalise_funding_page(
        [
            {
                "symbol": "BTCUSDT",
                "fundingTime": int(dt.datetime.fromisoformat(instant).timestamp() * 1000),
                "fundingRate": "0.01",
                "markPrice": "100",
            }
        ],
        product_id="active_income",
        symbol="BTCUSDT",
    )
    old = {
        "kind": "historical_funding",
        "product_id": "active_income",
        "instrument_id": rows[0]["instrument_id"],
        "symbol": "BTCUSDT",
        "funding_time": "2026-09-07T00:00:00+00:00",
        "funding_rate": 0.01,
        "mark_price": 100.0,
        "source": "binance_public_funding_rate",
    }
    with database.engine.begin() as connection:
        connection.execute(
            insert(funding_entry).values(
                id=canonical_hash(old), created_at=old["funding_time"], payload=old
            )
        )
    service._record_funding_entries(rows)
    service._record_funding_entries(rows)
    with database.engine.connect() as connection:
        records = tuple(connection.execute(select(funding_entry)).mappings())
    assert len(records) == 2
    assert any(row["payload"] == old for row in records)
    index = settlement_index(
        records, product_id="active_income", created="2026-09-08T00:00:00+00:00"
    )
    events = tuple(index[rows[0]["instrument_id"]].values())
    assert len(events) == 1 and events[0]["timestamp"] == instant
    assert events[0]["source_event_id"] == rows[0]["event_id"]
    correction = next(row for row in records if row["payload"].get("supersedes_record_id"))
    forged = {
        **dict(correction),
        "payload": {**correction["payload"], "supersedes_record_id": "sha256:" + "0" * 64},
    }
    with pytest.raises(ValueError, match="legacy record"):
        settlement_index([forged], product_id="active_income", created="2026-09-08T00:00:00+00:00")
    report = FuturesResearchAccounting().evaluate(
        initial_cash=1000,
        events=[
            {
                "type": "fill",
                "timestamp": "2026-09-07T00:00:00.005000+00:00",
                "symbol": rows[0]["instrument_id"],
                "side": "buy",
                "quantity": 2,
                "price": 100,
                "fee": 0,
            },
            *events,
        ],
    )
    assert report.funding_pnl == pytest.approx(-2)
    assert (
        settlement_index(
            records, product_id="active_income", created="2026-09-07T00:00:00.010000+00:00"
        )
        == {}
    )
    datasets = DatabaseDatasetBundleService(database.engine, tmp_path)
    assert datasets._funding_events("active_income", "2026-09-07T00:00:00.010000+00:00") == {}
    database.dispose()


def test_panel_returns_never_use_previous_instrument_price():
    rows = [
        {"instrument_id": symbol, "close": price, "timestamp": f"2026-09-07T00:0{index}:00+00:00"}
        for index, values in enumerate(((100, 1000), (110, 900)))
        for symbol, price in zip(("A", "B"), values, strict=True)
    ]
    payload = CanonicalResearchDatasetBuilder._bar_payload(rows)
    assert payload["symbol_returns"]["A"] == pytest.approx([0.1])
    assert payload["symbol_returns"]["B"] == pytest.approx([-0.1])
    assert payload["returns"] == pytest.approx([0, 0])
    assert payload["calendar_observations"] == 2
    assert "independent_units" not in payload


def test_future_bar_revision_is_not_available_at_original_decision():
    row = {
        "instrument_id": "A",
        "close": 100,
        "timestamp": "2026-09-07T00:00:00+00:00",
        "availability_time": "2026-09-08T00:00:00+00:00",
    }
    selected = CanonicalResearchDatasetBuilder._select_bars(
        (row,),
        role="development",
        scope={"A"},
        created="2026-09-09T00:00:00+00:00",
        interval={"start": "2026-09-07T00:00:00+00:00", "end": "2026-09-08T00:00:00+00:00"},
    )
    assert selected == []


def test_today_universe_cannot_backfill_historical_membership():
    memberships = (("2026-09-07T00:00:00+00:00", frozenset({"BTC"})),)
    assert not DatabaseDatasetBundleService._eligible_at(
        "BTC", "2025-09-07T00:00:00+00:00", memberships
    )
    assert DatabaseDatasetBundleService._eligible_at(
        "BTC", "2026-09-07T00:00:00+00:00", memberships
    )


def test_stage_features_use_past_warmup_without_trading_it():
    rows = tuple(
        {
            "instrument_id": "BTC",
            "close_timestamp": f"2026-09-07T0{index}:00:00+00:00",
            "availability_time": f"2026-09-07T0{index}:00:00+00:00",
            "close": value,
        }
        for index, value in enumerate((100, 110, 121))
    )
    payload = DatabaseDatasetBundleService._role_payloads(
        rows,
        intervals={
            "development": {
                "start": rows[-1]["close_timestamp"],
                "end": "2026-09-07T03:00:00+00:00",
            }
        },
        memberships=((rows[0]["close_timestamp"], frozenset({"BTC"})),),
        snapshot_ids=("fixture",),
        timeframe="1h",
    )["development"]
    assert payload["feature_rows"][0]["trend"] == pytest.approx(0.1)
    assert len(payload["market_frame"]) == 1
    assert [row["close"] for row in payload["warmup_market_frame"]] == [100, 110]


def test_bounded_window_keeps_latest_unique_bars_and_rejects_conflicts():
    window = RecentBarWindow(2)
    for minute in (3, 1, 2, 3):
        time = f"2026-09-07T00:0{minute}:00+00:00"
        window.add({"close_timestamp": time, "availability_time": time, "close": minute})
    assert [row["close"] for row in window.rows()] == [2, 3]
    with pytest.raises(ValueError, match="conflicting"):
        window.add({**window.rows()[-1], "close": 100})


def test_okx_history_resumes_deduplicates_and_verifies_coverage(tmp_path):
    class Response:
        def __init__(self, rows):
            self.rows = rows

        def raise_for_status(self):
            return None

        def json(self):
            return {"code": "0", "data": self.rows}

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, *, params, timeout):
            self.calls.append(params["after"])
            times = [180000, 120000] if len(self.calls) == 1 else [120000, 60000, 0]
            return Response(
                [[str(time), "100", "101", "99", "100", "1", "1", "100", "1"] for time in times]
            )

    session = Session()
    args = {
        "instrument": "BTC-USDT-SWAP",
        "timeframe": "1m",
        "start_ms": 0,
        "end_ms": 240000,
        "root": tmp_path,
        "maximum_pages": 1,
        "session": session,
    }
    first = download_history(**args)
    assert first["rows"] == 2 and not first["complete"]
    second = download_history(**args)
    assert second["rows"] == 4 and second["complete"]
    assert session.calls == ["240000", "120000"]
    assert download_history(**args) == second
    assert len(session.calls) == 2


def test_okx_funding_archives_resume_preserve_signed_events_and_missing_series(tmp_path):
    import io
    import json
    import zipfile
    from contextlib import nullcontext
    from types import SimpleNamespace

    from src.data.okx_history import download_funding_history

    start = 1746057600000
    calls = []
    failed = False

    def get(url, **_kwargs):
        nonlocal failed
        day = "2025-05-01" if "20250501" in url else "2025-05-02"
        calls.append(day)
        if day == "2025-05-02" and not failed:
            failed = True
            raise OSError("interrupted acquisition")
        rows = (
            [(start, "0.001"), (start, "0.001"), (start + 28_800_000, "-0.002")]
            if day == "2025-05-01"
            else [(start + 57_600_000, "0.0000"), (start + 86_400_000, "0.001")]
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                f"allswap-fundingrates-{day}.csv",
                "instrument_name,funding_rate,funding_time\n"
                + "".join(f"BTC-USDT-SWAP,{rate},{time}\n" for time, rate in rows),
            )
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            iter_content=lambda **_: iter((buffer.getvalue(),)),
        )
        return nullcontext(response)

    args = {
        "instruments": ("BTC-USDT-SWAP", "TON-USDT-SWAP"),
        "start_ms": start,
        "end_ms": start + 86_400_000,
        "root": tmp_path,
        "session": SimpleNamespace(get=get),
    }
    with pytest.raises(OSError, match="interrupted"):
        download_funding_history(**args)
    result = download_funding_history(**args)
    assert [r["funding_rate"] for r in result["records"]] == ["0.001", "-0.002", "0.0000"]
    assert [r["funding_time_ms"] for r in result["records"]] == [
        start,
        start + 28_800_000,
        start + 57_600_000,
    ]
    assert result["missing_instruments"] == ["TON-USDT-SWAP"]
    assert not result["promotable"]
    assert download_funding_history(**args) == result
    assert calls == ["2025-05-01", "2025-05-02", "2025-05-02"]
    source = tmp_path / "allswap-fundingrates-2025-05-01.json"
    payload = json.loads(source.read_text())
    payload["sha256"] = "0" * 64
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        download_funding_history(**args)


@pytest.mark.parametrize("second_rate", ["NaN", "0.02"])
def test_native_funding_rejects_invalid_or_conflicting_events(tmp_path, second_rate):
    import io
    import zipfile
    from contextlib import nullcontext
    from types import SimpleNamespace

    from src.data.okx_history import download_funding_history

    start = 1746057600000
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "allswap-fundingrates-2025-05-01.csv",
            "instrument_name,funding_rate,funding_time\n"
            f"BTC-USDT-SWAP,0.01,{start}\nBTC-USDT-SWAP,{second_rate},{start}\n",
        )
    response = SimpleNamespace(
        raise_for_status=lambda: None, iter_content=lambda **_: iter((buffer.getvalue(),))
    )
    session = SimpleNamespace(get=lambda *_args, **_kwargs: nullcontext(response))
    with pytest.raises(ValueError, match="invalid|conflicting"):
        download_funding_history(
            instruments=("BTC-USDT-SWAP",),
            start_ms=start,
            end_ms=start + 1,
            root=tmp_path,
            session=session,
        )


@pytest.mark.parametrize("complete_coverage", [True, False])
@pytest.mark.parametrize("fixed_scope", [True, False])
def test_bounded_two_market_plan_publishes_complete_warmup_and_funding(
    tmp_path, complete_coverage, fixed_scope
):
    import datetime as dt

    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import insert, select

    from src.data.database import (
        PlatformDatabase,
        cost_model_manifest,
        dataset_snapshot,
        feature_manifest,
        funding_entry,
        historical_bootstrap_progress,
    )
    from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
    from src.domain._codec import canonical_hash
    from src.domain.instruments import Instrument, MarketType
    from src.research.data_plan import ResearchDataPlan
    from src.research.datasets import SqlDatasetBundleRepository
    from src.services.universe_service import DatabaseUniverseService

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'data.sqlite3'}")
    database.create_schema()
    plan = ResearchDataPlan(
        "active_income", "15m", ("BTCUSDT", "SOLUSDT"), minimum_history_days=1, warmup_bars=2
    )
    start = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)
    now = (start + dt.timedelta(minutes=15 * plan.rows_per_instrument)).isoformat()
    available_at = (dt.datetime.fromisoformat(now) + dt.timedelta(days=1)).isoformat()
    build_at = (dt.datetime.fromisoformat(now) + dt.timedelta(days=2)).isoformat()
    instruments = tuple(
        Instrument(
            venue="binance",
            market_type=MarketType.FUTURES,
            base_asset=symbol[:-4],
            quote_asset="USDT",
            settlement_asset="USDT",
            exchange_symbol=symbol,
            price_precision=2,
            quantity_precision=3,
            minimum_quantity=0.001,
            minimum_notional=5,
        )
        for symbol in plan.instrument_scope
    )
    universes = SqlUniverseStore(database.engine)
    universe_id = "bounded-panel"
    observations = tuple(
        InstrumentObservation(instrument, 1000, 1e9, 1000000, 1, 1e9, 0, 0.2, 1e7, 1)
        for instrument in instruments
    )
    if fixed_scope:
        short_history_instrument = Instrument(
            venue="binance",
            market_type=MarketType.FUTURES,
            base_asset="ADA",
            quote_asset="USDT",
            settlement_asset="USDT",
            exchange_symbol="ADAUSDT",
            price_precision=2,
            quantity_precision=3,
            minimum_quantity=0.001,
            minimum_notional=5,
        )
        universes.record_snapshot(
            universe_id="bounded-panel-current",
            observed_at=start.isoformat(),
            observations=(
                *observations,
                InstrumentObservation(
                    short_history_instrument,
                    0.5,
                    5e9,
                    1000000,
                    1,
                    1e9,
                    0,
                    0.2,
                    1e7,
                    1,
                ),
            ),
            policy=UniverseEligibilityPolicy(minimum_listing_age_days=0),
        )
        stale_scope = DatabaseUniverseService(store=universes).resolve_research_scope(
            product_id=plan.product_id,
            source_universe_id="bounded-panel-current",
            selected_at=now,
            maximum_symbols=2,
            minimum_history_days=plan.minimum_history_days,
            specification_hash=plan.plan_id,
        )
        assert stale_scope.snapshot_id is None and stale_scope.symbols == ()
        assert dict(stale_scope.skipped_reasons)["BTCUSDT"] == "universe_observation_stale"
        universes.record_snapshot(
            universe_id="bounded-panel-current",
            observed_at=now,
            observations=tuple(
                InstrumentObservation(instrument=member.instrument, **member.metrics)
                for member in universes.members_at(
                    universe_id="bounded-panel-current", observed_at=now
                )
            ),
            policy=UniverseEligibilityPolicy(minimum_listing_age_days=0),
        )
        scope = DatabaseUniverseService(store=universes).resolve_research_scope(
            product_id=plan.product_id,
            source_universe_id="bounded-panel-current",
            selected_at=now,
            maximum_symbols=2,
            minimum_history_days=plan.minimum_history_days,
            specification_hash=plan.plan_id,
        )
        assert scope == DatabaseUniverseService(store=universes).resolve_research_scope(
            product_id=plan.product_id,
            source_universe_id="bounded-panel-current",
            selected_at=now,
            maximum_symbols=2,
            minimum_history_days=plan.minimum_history_days,
            specification_hash=plan.plan_id,
        )
        universe_id = scope.universe_id
        assert scope.snapshot_id is not None
        assert scope.symbols == plan.instrument_scope
        assert dict(scope.skipped_reasons)["ADAUSDT"] == "insufficient_listing_history"
        no_scope = DatabaseUniverseService(store=universes).resolve_research_scope(
            product_id=plan.product_id,
            source_universe_id="bounded-panel-current",
            selected_at=now,
            maximum_symbols=2,
            minimum_history_days=5000,
            specification_hash=plan.plan_id,
        )
        assert no_scope.snapshot_id is None and no_scope.symbols == ()
        assert universes.members_at(universe_id=universe_id, observed_at=start.isoformat()) == ()
    else:
        older_instrument = Instrument(
            venue="binance",
            market_type=MarketType.FUTURES,
            base_asset="ADA",
            quote_asset="USDT",
            settlement_asset="USDT",
            exchange_symbol="ADAUSDT",
            price_precision=2,
            quantity_precision=3,
            minimum_quantity=0.001,
            minimum_notional=5,
        )
        universes.record_snapshot(
            universe_id="bounded-panel",
            observed_at=(start - dt.timedelta(hours=1)).isoformat(),
            observations=(
                *observations,
                InstrumentObservation(older_instrument, 1000, 1e8, 100000, 1, 1e9, 0, 0.2, 1e7, 1),
            ),
            policy=UniverseEligibilityPolicy(),
        )
        universes.record_snapshot(
            universe_id="bounded-panel",
            observed_at=start.isoformat(),
            observations=(
                *observations,
                InstrumentObservation(older_instrument, 1, 1e8, 100000, 1, 1e9, 0, 0.2, 1e7, 1),
            ),
            policy=UniverseEligibilityPolicy(),
        )
    with database.engine.begin() as connection:
        for table, payload in (
            (feature_manifest, {"market_type": "futures"}),
            (cost_model_manifest, {"product_id": "active_income"}),
        ):
            connection.execute(
                insert(table).values(
                    id=canonical_hash(payload), created_at=start.isoformat(), payload=payload
                )
            )
        for instrument in instruments:
            coverage_end = (
                now if complete_coverage else (start + dt.timedelta(hours=12)).isoformat()
            )
            coverage_end_ms = int(dt.datetime.fromisoformat(coverage_end).timestamp() * 1000)
            funding_times = [
                int((start + dt.timedelta(hours=hour)).timestamp() * 1000)
                for hour in (0, 8, 16, 24)
                if int((start + dt.timedelta(hours=hour)).timestamp() * 1000) < coverage_end_ms
            ]
            raw_funding_page = [
                {
                    "symbol": instrument.exchange_symbol,
                    "fundingTime": funding_time,
                    "fundingRate": "0.001",
                    "markPrice": "100",
                }
                for funding_time in funding_times
            ]
            range_start_ms = int(start.timestamp() * 1000)
            coverage = {
                "product_id": plan.product_id,
                "market": "futures",
                "symbol": instrument.exchange_symbol,
                "timeframe": "funding",
                "range_start": start.isoformat(),
                "range_end": coverage_end,
                "next_start": coverage_end,
                "status": "completed",
                "updated_at": available_at,
                "payload": {
                    "gaps": [],
                    "coverage_classification": "native_query_exhausted",
                    "request_bounds": {"start_ms": range_start_ms, "end_ms": coverage_end_ms},
                    "page_limit": 1000,
                    "rows": len(raw_funding_page),
                    "pages": [
                        {
                            "request_start_ms": range_start_ms,
                            "request_end_ms": coverage_end_ms - 1,
                            "returned_rows": len(raw_funding_page),
                            "response_hash": canonical_hash(raw_funding_page),
                            "first_funding_time_ms": funding_times[0],
                            "last_funding_time_ms": funding_times[-1],
                            "next_start_ms": coverage_end_ms,
                        }
                    ],
                },
            }
            connection.execute(
                insert(historical_bootstrap_progress).values(
                    id=canonical_hash(coverage),
                    content_hash=canonical_hash(coverage),
                    **coverage,
                )
            )
            for funding_time in funding_times:
                time = dt.datetime.fromtimestamp(funding_time / 1000, dt.UTC).isoformat()
                payload = {
                    "kind": "historical_funding",
                    "product_id": "active_income",
                    "instrument_id": instrument.instrument_id,
                    "symbol": instrument.exchange_symbol,
                    "funding_time": time,
                    "funding_rate": 0.001,
                    "mark_price": 100,
                    "source": "isolated_test_fixture",
                }
                connection.execute(
                    insert(funding_entry).values(
                        id=canonical_hash(payload), created_at=available_at, payload=payload
                    )
                )
    for instrument in instruments:
        rows = []
        for index in range(plan.rows_per_instrument):
            opened = start + dt.timedelta(minutes=index * 15)
            closed = opened + dt.timedelta(minutes=15) - dt.timedelta(milliseconds=1)
            rows.append(
                {
                    "instrument_id": instrument.instrument_id,
                    "timeframe": plan.timeframe,
                    "open_time_ms": int(opened.timestamp() * 1000),
                    "close_time_ms": int(closed.timestamp() * 1000),
                    "source_availability_time": closed.isoformat(),
                    "availability_time": available_at,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 10.0,
                }
            )
        future_open = dt.datetime.fromisoformat(now)
        future_close = future_open + dt.timedelta(minutes=15) - dt.timedelta(milliseconds=1)
        rows.append(
            {
                "instrument_id": instrument.instrument_id,
                "timeframe": plan.timeframe,
                "open_time_ms": int(future_open.timestamp() * 1000),
                "close_time_ms": int(future_close.timestamp() * 1000),
                "source_availability_time": future_close.isoformat(),
                "availability_time": available_at,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
            }
        )
        partition = (
            tmp_path / "bars" / "binance" / "futures" / instrument.exchange_symbol / plan.timeframe
        )
        partition.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), partition / "fixture.parquet")
    result = DatabaseDatasetBundleService(database.engine, tmp_path).run(
        product_id=plan.product_id,
        universe_id=universe_id,
        market_type="futures",
        created_at=build_at,
        history_as_of=now,
        timeframe=plan.timeframe,
        data_plan=plan,
    )
    if not complete_coverage:
        assert result.state == "waiting_for_dataset"
        assert result.reason_code == "funding_history_coverage_incomplete"
        with database.engine.connect() as connection:
            assert connection.execute(select(dataset_snapshot)).first() is None
        database.dispose()
        return
    assert result.state == "ready", result
    assert result.instrument_scope == tuple(
        f"binance:futures:{symbol}:USDT" for symbol in plan.instrument_scope
    )
    bundle = SqlDatasetBundleRepository(database.engine).get(result.bundle_id)
    assert bundle.data_plan_id == plan.plan_id
    repository = SqlDatasetBundleRepository(database.engine)
    assert (
        repository.latest_ready(plan.product_id, at=build_at, data_plan_id=plan.plan_id) == bundle
    )
    assert (
        repository.latest_ready(
            plan.product_id,
            at=build_at,
            data_plan_id=canonical_hash("different-timeframe-or-scope"),
        )
        is None
    )
    assert bundle.created_at == build_at
    history_start_ms, history_end_ms = plan.acquisition_interval(now)
    history_start = dt.datetime.fromtimestamp(history_start_ms / 1000, dt.UTC).isoformat()
    history_end = dt.datetime.fromtimestamp(history_end_ms / 1000, dt.UTC).isoformat()
    with database.engine.connect() as connection:
        snapshots = list(connection.execute(select(dataset_snapshot)).mappings())
    stage_intervals = {}
    total_events = 0
    for record in snapshots:
        payload = record["payload"]["payload"]
        intervals = payload["data_plan"]["intervals"]
        stage_intervals[record["payload"]["role"]] = intervals
        assert payload["data_plan_id"] == plan.plan_id
        assert len(intervals) == 4
        assert max(interval["end"] for interval in intervals.values()) == history_end
        assert all(
            history_start <= interval["start"] < interval["end"] <= history_end
            for interval in intervals.values()
        )
        assert payload["point_in_time_universe"] is (not fixed_scope)
        assert payload["universe_selection_mode"] == (
            "fixed_research_scope" if fixed_scope else "point_in_time"
        )
        assert len(payload["warmup_market_frame"]) == 4
        assert {row["instrument_id"] for row in payload["market_frame"]} == {
            i.instrument_id for i in instruments
        }
        assert record["payload"]["availability_timestamp"] == build_at
        for bar in (*payload["market_frame"], *payload["warmup_market_frame"]):
            assert history_start <= bar["close_timestamp"] < history_end
        assert all(
            bar["close_timestamp"] < now
            for bar in (*payload["market_frame"], *payload["warmup_market_frame"])
        )
        assert all(
            row["close_timestamp"] < record["payload"]["interval"]["start"]
            for row in payload["warmup_market_frame"]
        )
        total_events += sum(len(row["funding_events"]) for row in payload["market_frame"])
    assert total_events == 6

    retry_at = (dt.datetime.fromisoformat(build_at) + dt.timedelta(days=1)).isoformat()
    retried = DatabaseDatasetBundleService(database.engine, tmp_path).run(
        product_id=plan.product_id,
        universe_id=universe_id,
        market_type="futures",
        created_at=retry_at,
        history_as_of=now,
        timeframe=plan.timeframe,
        data_plan=plan,
    )
    assert retried.state == "ready", retried
    assert retried.bundle_id == bundle.bundle_id
    retry_bundle = SqlDatasetBundleRepository(database.engine).get(retried.bundle_id)
    assert retry_bundle.created_at == build_at
    with database.engine.connect() as connection:
        retry_snapshots = list(
            connection.execute(
                select(dataset_snapshot).where(dataset_snapshot.c.id.in_(retried.snapshot_ids))
            ).mappings()
        )
    assert {
        row["payload"]["role"]: row["payload"]["payload"]["data_plan"]["intervals"]
        for row in retry_snapshots
    } == stage_intervals
    database.dispose()


def test_fixed_research_scopes_do_not_feed_live_capture_symbols(tmp_path):
    from src.data.database import PlatformDatabase
    from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
    from src.domain._codec import canonical_hash
    from src.domain.instruments import Instrument, MarketType

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'capture-universe.sqlite3'}")
    database.create_schema()
    store = SqlUniverseStore(database.engine)
    selected_at = "2026-09-01T00:00:00+00:00"
    bitcoin = Instrument(
        venue="binance",
        market_type=MarketType.SPOT,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset=None,
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5,
    )
    solana = Instrument(
        venue="binance",
        market_type=MarketType.FUTURES,
        base_asset="SOL",
        quote_asset="USDT",
        settlement_asset="USDT",
        exchange_symbol="SOLUSDT",
        price_precision=2,
        quantity_precision=3,
        minimum_quantity=0.001,
        minimum_notional=5,
    )
    store.record_snapshot(
        universe_id="capture-dynamic",
        observed_at=selected_at,
        observations=(
            InstrumentObservation(bitcoin, 1000, 1e9, 1_000_000, 1, 1e9, 0, 0.2, 1e7, 1),
        ),
        policy=UniverseEligibilityPolicy(),
    )
    store.record_fixed_research_scope(
        universe_id="research-fixed",
        product_id="active_income",
        instruments=(solana,),
        selected_at=selected_at,
        specification_hash=canonical_hash({"scope": "capture-isolation"}),
    )

    assert store.eligible_exchange_symbols(observed_at=selected_at) == ("BTCUSDT",)
    database.dispose()


def test_live_and_native_history_overlap_forms_one_candle_without_losing_source_ids(tmp_path):
    import datetime as dt

    import pyarrow.parquet as pq

    from src.data.parquet_store import PartitionedBarStore, partition_files, upsert_partition_rows
    from src.domain.market_events import MarketEvent, MarketEventType
    from src.research.dataset_service import _ingest_bar_rows, _validate_market_bars

    instrument = "binance:spot:BTCUSDT"
    start = 1_700_000_040_000
    store = PartitionedBarStore(tmp_path)
    native_ids = []
    for index in range(3):
        opened = start + index * 60_000
        closed = opened + 59_999
        available = dt.datetime.fromtimestamp((closed + 1) / 1000, dt.UTC).isoformat()
        candle = {
            "i": "1m",
            "t": opened,
            "T": closed,
            "o": "100",
            "h": "102",
            "l": "99",
            "c": "101",
            "v": "2",
            "x": True,
        }
        event = MarketEvent(
            instrument_id=instrument,
            event_type=MarketEventType.CANDLE,
            exchange_timestamp=available,
            receive_timestamp=available,
            sequence=index,
            payload={"data": {"k": candle}},
        )
        path = store.put(event, venue="binance", market="spot", symbol="BTCUSDT")
        native_ids.append(event.event_id)
        historical = {
            "event_id": f"history-{index}",
            "instrument_id": instrument,
            "open_time_ms": opened,
            "close_time_ms": closed,
            "close_timestamp": dt.datetime.fromtimestamp(closed / 1000, dt.UTC).isoformat(),
            "availability_time": "2026-01-01T00:00:00+00:00",
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 2.0,
            "quote_asset_volume": 202.0,
            "number_of_trades": 5,
            "availability_source": "native_closed_candle",
        }
        upsert_partition_rows(path.parent, (historical,), identity="event_id")
    raw = [
        row for path in partition_files(tmp_path) for row in pq.ParquetFile(path).read().to_pylist()
    ]
    assert set(native_ids).issubset({row["event_id"] for row in raw})
    # Include an older live publisher's whole-second timestamp: integer exchange time wins.
    raw[0]["close_timestamp"] = dt.datetime.fromtimestamp(
        raw[0]["close_time_ms"] // 1000, dt.UTC
    ).isoformat()
    window = RecentBarWindow(20)
    assert _ingest_bar_rows(raw, window, instrument, "2026-02-01T00:00:00+00:00")
    rows = window.rows()
    assert len(rows) == 3
    assert all(row["quote_asset_volume"] == 202 and row["number_of_trades"] == 5 for row in rows)
    assert all(row["availability_time"] < "2026-01-01" for row in rows)
    _validate_market_bars(tuple(rows), market_type="spot", timeframe="1m")
    with pytest.raises(ValueError, match="conflicting"):
        window.add({**rows[0], "quote_asset_volume": 999})
