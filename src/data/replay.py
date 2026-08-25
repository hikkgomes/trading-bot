"""Deterministic replay of canonical events from immutable Parquet data."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pyarrow.dataset as ds

from src.domain.market_events import MarketEvent, MarketEventType


class ParquetEventReplay:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def events(
        self,
        *,
        venue: str,
        market: str,
        event_type: MarketEventType,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[MarketEvent, ...]:
        partition = (
            self.root / "raw" / venue.lower() / market.lower() / event_type.value / symbol.upper()
        ).resolve()
        if self.root not in partition.parents or not partition.exists():
            return ()
        table = ds.dataset(partition, format="parquet", partitioning="hive").to_table()
        records: list[MarketEvent] = []
        for row in table.to_pylist():
            receive_timestamp = str(row["receive_timestamp"])
            if start is not None and receive_timestamp < start:
                continue
            if end is not None and receive_timestamp >= end:
                continue
            records.append(
                MarketEvent(
                    instrument_id=str(row["instrument_id"]),
                    event_type=MarketEventType(str(row["event_type"])),
                    exchange_timestamp=str(row["exchange_timestamp"]),
                    receive_timestamp=receive_timestamp,
                    sequence=int(row["sequence"]),
                    payload=json.loads(str(row["payload_json"])),
                )
            )
        records.sort(
            key=lambda item: (
                item.receive_timestamp,
                item.exchange_timestamp,
                item.instrument_id,
                item.sequence,
            )
        )
        return tuple(records)

    def merge(self, streams: Iterable[Iterable[MarketEvent]]) -> tuple[MarketEvent, ...]:
        events = [event for stream in streams for event in stream]
        events.sort(
            key=lambda item: (
                item.receive_timestamp,
                item.exchange_timestamp,
                item.instrument_id,
                item.sequence,
            )
        )
        return tuple(events)
