from __future__ import annotations

import pyarrow.parquet as pq
from sqlalchemy import create_engine, select

from src.data.database import job, job_attempt
from src.data.parquet_store import DurableMarketBatchSpool, PartitionedMarketEventStore
from src.domain.market_events import MarketEvent, MarketEventType
from src.observability.queue_progress import queue_progress
from src.services.market_gateway import DatabaseGatewaySink
from src.services.scheduler import DatabaseJobQueue


def _event(sequence: int) -> MarketEvent:
    return MarketEvent(
        instrument_id="binance:futures:BTCUSDT:USDT",
        event_type=MarketEventType.TRADE,
        exchange_timestamp="2026-08-23T00:00:00+00:00",
        receive_timestamp="2026-08-23T00:00:01+00:00",
        sequence=sequence,
        payload={"data": {"a": sequence}},
    )


def test_spool_publishes_one_atomic_segment_for_multiple_events(tmp_path) -> None:
    spool = DurableMarketBatchSpool(tmp_path / "spool", max_rows=2, max_bytes=10_000)
    assert spool.append(_event(1), venue="binance", market="futures", symbol="BTCUSDT") is None
    segment = spool.append(_event(2), venue="binance", market="futures", symbol="BTCUSDT")
    assert segment is not None and segment.row_count == 2
    rows = spool.read(segment)
    paths = PartitionedMarketEventStore(tmp_path / "parquet").put_batch(
        tuple(
            (MarketEvent(**row["event"]), row["venue"], row["market"], row["symbol"])
            for row in rows
        )
    )
    assert len(paths) == 1
    persisted = pq.ParquetFile(paths[0]).read().to_pylist()
    assert {row["event_id"] for row in persisted} == {_event(1).event_id, _event(2).event_id}
    assert len(persisted) == 2


def test_spool_retention_tolerates_files_removed_during_acknowledgement(tmp_path) -> None:
    import os
    from unittest.mock import Mock

    from src.services.market_gateway import _spool_retention

    root = tmp_path / "retention-spool"
    pending = root / "pending"
    committed = root / "committed"
    pending.mkdir(parents=True)
    committed.mkdir(parents=True)
    vanished = committed / "segment-vanished.jsonl"
    acknowledged = committed / "segment-acknowledged.jsonl"
    unacknowledged = committed / "segment-unacknowledged.jsonl"
    pending_file = pending / "segment-pending.jsonl"
    file_contents = {
        vanished: b"vanished",
        acknowledged: b"acknowledged",
        unacknowledged: b"unacknowledged",
        pending_file: b"pending",
    }
    for path, content in file_contents.items():
        path.write_bytes(content)
        os.utime(path, (0, 0))
    symlink = committed / "segment-symlink.jsonl"
    symlink.symlink_to(unacknowledged)
    is_acknowledged = Mock(
        side_effect=lambda path: (path.unlink(), True)[1]
        if path == vanished
        else path == acknowledged
    )

    result = _spool_retention(
        root,
        retention_seconds=10,
        now=100,
        is_acknowledged=is_acknowledged,
    )

    assert result == {
        "removed_files": 1,
        "removed_bytes": len(file_contents[acknowledged]),
        "total_bytes": len(file_contents[unacknowledged]) + len(file_contents[pending_file]),
    }
    assert not vanished.exists()
    assert not acknowledged.exists()
    assert unacknowledged.read_bytes() == file_contents[unacknowledged]
    assert pending_file.read_bytes() == file_contents[pending_file]
    assert symlink.is_symlink()


def test_new_public_batch_job_is_explicitly_tagged_for_retention(tmp_path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    job.create(engine)
    job_attempt.create(engine)
    queue = DatabaseJobQueue(engine)
    spool = DurableMarketBatchSpool(tmp_path / "spool", max_rows=1, max_bytes=10_000)
    sink = DatabaseGatewaySink(queue, spool=spool)

    segment = spool.append(_event(1), venue="binance", market="futures", symbol="BTCUSDT")
    assert segment is not None
    sink._enqueue_segment(segment)
    job_id = f"work:market_event_batch:{segment.content_hash.removeprefix('sha256:')}"

    with engine.connect() as connection:
        payload = connection.execute(select(job.c.payload).where(job.c.id == job_id)).scalar_one()
    assert payload["queue_retention_policy"] == "public_market_batch_completed_7d/v1"
    progress = queue_progress(engine, now="2026-10-16T00:00:00+00:00")
    assert progress["history_retention"]["completed_public_market_batch_days"] == 7

    legacy_segment = spool.append(_event(2), venue="binance", market="futures", symbol="BTCUSDT")
    assert legacy_segment is not None
    legacy_id = f"work:market_event_batch:{legacy_segment.content_hash.removeprefix('sha256:')}"
    legacy_payload = {
        "segment_path": str(spool.root / "pending" / legacy_segment.path.name),
        "segment_hash": legacy_segment.content_hash,
        "row_count": legacy_segment.row_count,
        "producer_identity": "market-gateway",
    }
    queue.enqueue_if_absent(
        job_id=legacy_id,
        name="market_event_batch_write",
        payload=legacy_payload,
        available_at="2026-10-16T00:00:00+00:00",
        priority=10,
        producer_identity="market-gateway",
    )
    sink._enqueue_segment(legacy_segment)
    with engine.connect() as connection:
        replayed_payload = connection.execute(
            select(job.c.payload).where(job.c.id == legacy_id)
        ).scalar_one()
    assert "queue_retention_policy" not in replayed_payload
    engine.dispose()


def test_history_starts_despite_market_backlog_and_leaves_capture_running(tmp_path):
    from concurrent.futures import Future
    from unittest.mock import Mock

    from src.data.database import PlatformDatabase
    from src.domain._codec import canonical_hash
    from src.services.data_writer import DatabaseMarketDataWriter

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'history-fairness.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    now = "2026-10-16T00:00:00+00:00"
    queue.register_worker(
        worker_id="writer",
        node_id="linux-optiplex",
        role="data-writer",
        capabilities=("historical_bootstrap", "market_event_batch_write"),
        observed_at=now,
    )
    spool = DurableMarketBatchSpool(tmp_path / "spool", max_rows=1, max_bytes=10_000)
    sink = DatabaseGatewaySink(queue, spool=spool)
    segment = spool.append(_event(1), venue="binance", market="futures", symbol="BTCUSDT")
    sink._enqueue_segment(segment)
    with database.engine.begin() as connection:
        connection.execute(job.update().values(available_at=now))
    for identity in ("first-history", "second-history"):
        payload = {
            "product_id": "active_income",
            "producer_identity": identity,
            "market_type": "futures",
            "timeframe": "1h",
            "symbols": ["BTCUSDT"],
            "minimum_history_days": 365,
        }
        queue.enqueue(
            job_id=identity,
            name="historical_bootstrap",
            payload={**payload, "content_hash": canonical_hash(payload)},
            available_at=now,
            priority=0,
        )
    pending = Future()
    executor = Mock()
    executor.submit.return_value = pending
    writer = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="writer",
        root=tmp_path / "market",
        historical_bootstrap_service=Mock(),
        historical_executor=executor,
    )
    assert writer.run_once(now=now)["reason_code"] == "historical_bootstrap_started"
    assert executor.submit.call_args.args[1].job_id == "first-history"
    result = writer.run_once(now=now)
    assert result["reason_code"] == "market_event_batch_written"
    assert result["events"] == 1
    executor.submit.assert_called_once()
    with database.engine.connect() as connection:
        states = dict(connection.execute(select(job.c.id, job.c.state)).all())
    assert states["second-history"] == "pending"
    assert states["first-history"] == "running"
    assert (
        states[f"work:market_event_batch:{segment.content_hash.removeprefix('sha256:')}"]
        == "completed"
    )
    database.dispose()


def test_blocked_quote_snapshot_save_leaves_durable_and_private_paths_writable(tmp_path):
    import datetime as dt
    from threading import Event
    from types import SimpleNamespace
    from unittest.mock import Mock

    from src.data.database import PlatformDatabase
    from src.domain.market_events import MarketEventType
    from src.services.market_gateway import DatabaseMarketGateway

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'quote-write-blocked.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    started, release = Event(), Event()
    snapshot_store = Mock()
    snapshot_store.save_market_inputs.side_effect = lambda *_args, **_kwargs: (
        started.set(),
        release.wait(3),
        ("sha256:" + "a" * 64,),
    )[2]
    gateway = DatabaseMarketGateway(
        queue=queue,
        capture_config=SimpleNamespace(root=tmp_path),
        snapshot_store=snapshot_store,
        product_ids_by_market={"futures": ("active_income",)},
    )
    spool = DurableMarketBatchSpool(tmp_path / "gateway-spool", max_rows=1, max_bytes=10_000)
    sink = DatabaseGatewaySink(queue, spool=spool, coalesce_quotes=True)
    at_time = dt.datetime.now(dt.UTC).replace(microsecond=0)
    at = at_time.isoformat()
    sink.write(
        {
            "market": "futures",
            "stream": "btcusdt@bookTicker",
            "symbol": "BTCUSDT",
            "received_at": at,
            "payload": {
                "e": "bookTicker",
                "E": int(at_time.timestamp() * 1_000),
                "s": "BTCUSDT",
                "b": "100",
                "B": "10",
                "a": "101",
                "A": "20",
            },
        }
    )
    gateway._start_quote_write(sink)
    assert started.wait(2)

    user_event = MarketEvent(
        instrument_id="binance:futures:BTCUSDT:USDT",
        event_type=MarketEventType.ORDER_UPDATE,
        exchange_timestamp=at,
        receive_timestamp=at,
        sequence=1,
        payload={"status": "NEW"},
    )
    sink.write_user(user_event, account_id="fixture-account", market="futures")
    user_job = f"user-stream:{user_event.event_id.removeprefix('sha256:')}"
    sink._write_market_event(_event(4), market="futures", symbol="BTCUSDT")
    sink.flush()
    durable_segment = spool.recoverable()[0]
    states = (
        queue.state(user_job),
        queue.state(f"work:market_event_batch:{durable_segment.stem.removeprefix('segment-')}"),
    )

    release.set()
    assert gateway._quote_write_future is not None
    gateway._quote_write_future.result(timeout=2)
    gateway._collect_quote_write(sink)
    assert states == ("pending", "pending")
    assert gateway._quote_last_reason == "market_quote_snapshot_written"
    assert snapshot_store.save_market_inputs.call_args.kwargs["statement_timeout_seconds"] == 10.0
    assert snapshot_store.save_market_inputs.call_args.kwargs["lock_timeout_seconds"] == 2.0
    quote_health = gateway._quote_persistence_status(sink, now=at)
    assert quote_health["last_persisted_source_time"] == at
    assert quote_health["persisted_source_age_seconds"] == 0.0
    gateway.stop()
    database.dispose()


def test_failed_quote_batch_merge_keeps_newer_event_per_key(tmp_path):
    import datetime as dt
    from types import SimpleNamespace
    from unittest.mock import Mock

    from src.data.binance_market import normalise_public_event
    from src.data.database import PlatformDatabase
    from src.services.market_gateway import DatabaseMarketGateway

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'quote-write-failure.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    snapshot_store = Mock()
    snapshot_store.save_market_inputs.side_effect = RuntimeError("snapshot writer failed")
    gateway = DatabaseMarketGateway(
        queue=queue,
        capture_config=SimpleNamespace(root=tmp_path),
        snapshot_store=snapshot_store,
        product_ids_by_market={"futures": ("active_income",)},
    )
    sink = DatabaseGatewaySink(queue, coalesce_quotes=True)
    at_time = dt.datetime.now(dt.UTC).replace(microsecond=0)
    at = at_time.isoformat()
    sink._write_market_event(
        normalise_public_event(
            market="futures",
            stream="btcusdt@bookTicker",
            receive_timestamp=at,
            payload={
                "e": "bookTicker",
                "E": int(at_time.timestamp() * 1_000),
                "s": "BTCUSDT",
                "b": "100",
                "B": "10",
                "a": "101",
                "A": "20",
            },
        ),
        market="futures",
        symbol="BTCUSDT",
    )
    gateway._start_quote_write(sink)
    newer = normalise_public_event(
        market="futures",
        stream="btcusdt@bookTicker",
        receive_timestamp=at,
        payload={
            "e": "bookTicker",
            "E": int(at_time.timestamp() * 1_000) + 2_000,
            "s": "BTCUSDT",
            "b": "102",
            "B": "10",
            "a": "103",
            "A": "20",
        },
    )
    sink._write_market_event(newer, market="futures", symbol="BTCUSDT")
    assert gateway._quote_write_future is not None
    try:
        gateway._quote_write_future.result(timeout=2)
    except RuntimeError:
        pass
    gateway._collect_quote_write(sink)
    merged = sink.take_quote_batch()

    assert len(merged) == 1
    assert merged[0][2].payload["data"]["b"] == "102"
    assert gateway._quote_last_reason == "market_quote_snapshot_write_failed"
    quote_health = gateway._quote_persistence_status(sink, now=at)
    assert quote_health["last_error_type"] == "RuntimeError"
    assert quote_health["persisted_source_age_seconds"] is None
    gateway.stop()
    database.dispose()


def test_quote_write_failure_marks_service_unhealthy_and_success_recovers(tmp_path, monkeypatch):
    import datetime as dt
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest

    from src.data.binance_market import normalise_public_event
    from src.data.database import PlatformDatabase
    from src.services.config import load_platform_config
    from src.services.health import DatabaseHeartbeatStore
    from src.services.market_gateway import DatabaseGatewaySink, DatabaseMarketGateway
    from src.services.runtime import ServiceRuntime

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'quote-health.sqlite3'}")
    database.create_schema()
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    at = now.isoformat()
    queue = DatabaseJobQueue(database.engine)
    snapshot_store = Mock()
    snapshot_store.save_market_inputs.side_effect = RuntimeError("snapshot write failed")
    gateway = DatabaseMarketGateway(
        queue=queue,
        capture_config=SimpleNamespace(root=tmp_path),
        snapshot_store=snapshot_store,
        product_ids_by_market={"futures": ("active_income",)},
    )
    monkeypatch.setattr(gateway, "start", lambda: None)
    gateway._continuous_status = {"reason_code": "market_gateway_running"}
    sink = DatabaseGatewaySink(queue, coalesce_quotes=True)
    sink._write_market_event(
        normalise_public_event(
            market="futures",
            stream="btcusdt@bookTicker",
            receive_timestamp=at,
            payload={
                "e": "bookTicker",
                "E": int(now.timestamp() * 1_000),
                "s": "BTCUSDT",
                "b": "100",
                "B": "10",
                "a": "101",
                "A": "20",
            },
        ),
        market="futures",
        symbol="BTCUSDT",
    )
    gateway._start_quote_write(sink)
    assert gateway._quote_write_future is not None
    with pytest.raises(RuntimeError, match="snapshot write failed"):
        gateway._quote_write_future.result(timeout=2)
    gateway._collect_quote_write(sink)
    gateway._publish_quote_health(sink, now=at)

    runtime = ServiceRuntime(
        config=load_platform_config(Path("config/platform.json")),
        node_id="linux-optiplex",
        service_name="market-gateway",
        heartbeat_store=DatabaseHeartbeatStore(database.engine),
    )
    failed = runtime.run_once(gateway.poll, observed_at=at)
    assert failed.healthy is False
    assert failed.detail["reason_code"] == "market_gateway_quote_snapshot_write_failed"
    assert failed.detail["quote_persistence"]["last_error_type"] == "RuntimeError"

    snapshot_store.save_market_inputs.side_effect = None
    snapshot_store.save_market_inputs.return_value = ("sha256:" + "a" * 64,)
    gateway._quote_retry_at = 0.0
    gateway._start_quote_write(sink)
    assert gateway._quote_write_future is not None
    gateway._quote_write_future.result(timeout=2)
    gateway._collect_quote_write(sink)
    recovered_at = (now + dt.timedelta(seconds=1)).isoformat()
    gateway._publish_quote_health(sink, now=recovered_at)

    recovered = runtime.run_once(gateway.poll, observed_at=recovered_at)
    assert recovered.healthy is True
    assert recovered.detail["reason_code"] == "market_gateway_running"
    assert recovered.detail["quote_persistence"]["last_error_type"] is None
    gateway.stop()
    database.dispose()


def test_stalled_quote_writer_shutdown_respects_grace_period(tmp_path, monkeypatch):
    import asyncio
    import datetime as dt
    import time
    from threading import Event
    from types import SimpleNamespace
    from unittest.mock import Mock

    from src.data.binance_market import normalise_public_event
    from src.data.database import PlatformDatabase
    from src.services.market_gateway import DatabaseGatewaySink, DatabaseMarketGateway

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'quote-shutdown.sqlite3'}")
    database.create_schema()
    started, release = Event(), Event()
    snapshot_store = Mock()
    snapshot_store.save_market_inputs.side_effect = lambda *_args, **_kwargs: (
        started.set(),
        release.wait(5),
        ("sha256:" + "b" * 64,),
    )[2]
    gateway = DatabaseMarketGateway(
        queue=DatabaseJobQueue(database.engine),
        capture_config=SimpleNamespace(root=tmp_path),
        snapshot_store=snapshot_store,
        product_ids_by_market={"futures": ("active_income",)},
    )
    sink = DatabaseGatewaySink(gateway.queue, coalesce_quotes=True)
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    sink._write_market_event(
        normalise_public_event(
            market="futures",
            stream="btcusdt@bookTicker",
            receive_timestamp=now.isoformat(),
            payload={
                "e": "bookTicker",
                "E": int(now.timestamp() * 1_000),
                "s": "BTCUSDT",
                "b": "100",
                "B": "10",
                "a": "101",
                "A": "20",
            },
        ),
        market="futures",
        symbol="BTCUSDT",
    )
    gateway._start_quote_write(sink)
    assert started.wait(2)
    future = gateway._quote_write_future
    assert future is not None
    monkeypatch.setattr("src.services.market_gateway._MARKET_QUOTE_SHUTDOWN_WAIT_SECONDS", 0.05)

    started_at = time.monotonic()
    asyncio.run(gateway._wait_for_quote_write(sink))
    assert time.monotonic() - started_at < 0.5
    assert future.done() is False
    started_at = time.monotonic()
    gateway.stop()
    assert time.monotonic() - started_at < 0.5
    release.set()
    assert future.result(timeout=2) == (now.isoformat(), 1, 0)
    gateway._quote_executor.shutdown(wait=True, cancel_futures=False)
    database.dispose()


def test_gateway_keeps_unlinked_funding_events_on_the_durable_spool(tmp_path):
    from src.data.database import PlatformDatabase

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'funding-spool.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    spool = DurableMarketBatchSpool(tmp_path / "funding-spool", max_rows=1, max_bytes=10_000)
    sink = DatabaseGatewaySink(queue, spool=spool, coalesce_quotes=True)
    derived = MarketEvent(
        instrument_id="binance:futures:BTCUSDT:USDT",
        event_type=MarketEventType.FUNDING_RATE,
        exchange_timestamp="2026-10-16T00:00:00+00:00",
        receive_timestamp="2026-10-16T00:00:01+00:00",
        sequence=1,
        payload={"source_event_id": "sha256:derived", "data": {"funding_rate": 0.001}},
    )
    unlinked = MarketEvent(
        instrument_id="binance:futures:BTCUSDT:USDT",
        event_type=MarketEventType.FUNDING_RATE,
        exchange_timestamp="2026-10-16T00:00:02+00:00",
        receive_timestamp="2026-10-16T00:00:03+00:00",
        sequence=2,
        payload={"data": {"funding_rate": 0.002}},
    )

    sink._write_market_event(derived, market="futures", symbol="BTCUSDT")
    sink._write_market_event(unlinked, market="futures", symbol="BTCUSDT")
    direct = sink.take_quote_batch()
    sink.flush()
    segment = spool.recoverable()[0]
    persisted = DurableMarketBatchSpool.read(segment)[0]["event"]

    assert len(direct) == 1 and direct[0][2].payload["source_event_id"] == "sha256:derived"
    assert persisted["event_type"] == MarketEventType.FUNDING_RATE.value
    assert persisted["payload"] == unlinked.payload
    database.dispose()
