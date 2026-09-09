import pytest

from src.data.bar_window import RecentBarWindow
from src.data.okx_history import download_history
from src.research.dataset_service import DatabaseDatasetBundleService
from src.research.datasets import CanonicalResearchDatasetBuilder
from src.research.executors import DataUnavailableError, _product_accounting
from src.research.funding import attach_settlements, settlement_index


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
