from __future__ import annotations

from src.data.parquet_store import DurableMarketBatchSpool, PartitionedMarketEventStore
from src.domain.market_events import MarketEvent, MarketEventType


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
    assert paths[0].name.startswith("batch-")
