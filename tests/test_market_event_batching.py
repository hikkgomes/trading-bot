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
