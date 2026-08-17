"""Leased immutable Parquet writer for canonical market events."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from src.data.parquet_store import PartitionedBarStore, PartitionedMarketEventStore
from src.domain._codec import canonical_hash
from src.domain.market_events import MarketEvent, MarketEventType
from src.services.scheduler import DatabaseJobQueue


class DatabaseMarketDataWriter:
    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        root: Path,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.store = PartitionedMarketEventStore(root)
        self.bar_store = PartitionedBarStore(root)
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("market_event_write",),
        )
        if claimed is None:
            return {"reason_code": "market_event_queue_empty"}
        try:
            raw_event = claimed.payload.get("event")
            if not isinstance(raw_event, dict):
                raise ValueError("market-event job has no event object")
            event = MarketEvent(**raw_event)
            venue = str(claimed.payload["venue"])
            market = str(claimed.payload["market"])
            symbol = str(claimed.payload["symbol"])
            expected_prefix = f"{venue.lower()}:{market.lower()}:{symbol.upper()}"
            if not event.instrument_id.startswith(expected_prefix):
                raise ValueError("market-event partition does not match its instrument")
            path = self.store.put(event, venue=venue, market=market, symbol=symbol)
            bar_path = (
                self.bar_store.put(event, venue=venue, market=market, symbol=symbol)
                if _is_closed_candle(event)
                else None
            )
            feature_job_id = self._enqueue_closed_candle_features(
                event=event,
                venue=venue,
                market=market,
                symbol=symbol,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "market_event_write_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "market_event_written",
            "job_id": claimed.job_id,
            "path": str(path),
            "bar_path": str(bar_path) if bar_path is not None else None,
            "feature_job_id": feature_job_id,
        }

    def _enqueue_closed_candle_features(
        self,
        *,
        event: MarketEvent,
        venue: str,
        market: str,
        symbol: str,
    ) -> str | None:
        if not _is_closed_candle(event):
            return None
        raw_data = event.payload["data"]
        candle = raw_data["k"]
        source_event_time = _milliseconds_time(candle.get("t"), field="candle open time")
        source_close_time = _milliseconds_time(candle.get("T"), field="candle close time")
        payload = {
            "instrument_id": event.instrument_id,
            "feature_set_version": "core-bars-v1",
            "source_event_time": source_event_time,
            "source_close_time": source_close_time,
            "availability_time": event.availability_timestamp,
            "inputs": {
                "open": float(candle["o"]),
                "high": float(candle["h"]),
                "low": float(candle["l"]),
                "close": float(candle["c"]),
                "volume": float(candle["v"]),
            },
            "venue": venue,
            "market": market,
            "symbol": symbol,
            "timeframe": str(candle["i"]),
            "source_market_event_id": event.event_id,
        }
        identity = canonical_hash(payload)
        job_id = f"live-feature:{identity.removeprefix('sha256:')}"
        self.queue.enqueue_if_absent(
            job_id=job_id,
            name="live_feature_calculation",
            payload=payload,
            available_at=event.availability_timestamp,
            priority=10,
        )
        return job_id


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()


def _milliseconds_time(value: object, *, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be milliseconds")
    return dt.datetime.fromtimestamp(float(value) / 1_000, dt.UTC).isoformat()


def _is_closed_candle(event: MarketEvent) -> bool:
    raw_data = event.payload.get("data")
    candle = raw_data.get("k") if isinstance(raw_data, dict) else None
    return (
        event.event_type is MarketEventType.CANDLE
        and isinstance(candle, dict)
        and candle.get("x") is True
    )
