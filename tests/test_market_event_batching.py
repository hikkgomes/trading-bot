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
