"""Content-addressed file storage for Parquet, models, and artefacts.

The interface does not impose a dataframe library. Callers write their Parquet
file with the project tooling then register it atomically through this store.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.feature_store import FeatureValue
from src.domain._codec import canonical_hash, to_primitive
from src.domain.market_events import MarketEvent

_PARTITION_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


class ContentAddressedStore:
    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def digest(path: Path) -> str:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"content must be a regular file: {path}")
        with path.open("rb") as handle:
            return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()

    def put(self, source: Path, *, suffix: str | None = None) -> Path:
        digest = self.digest(source)
        extension = suffix if suffix is not None else source.suffix
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        destination = (
            self.root
            / digest.removeprefix("sha256:")[:2]
            / (digest.removeprefix("sha256:") + extension)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if self.digest(destination) != digest:
                raise RuntimeError(f"content-address collision at {destination}")
            return destination
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
                while chunk := input_handle.read(1024 * 1024):
                    output_handle.write(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if self.digest(destination) != digest:
                    raise RuntimeError(f"content-address collision at {destination}")
        finally:
            temporary.unlink(missing_ok=True)
        return destination


class PartitionedMarketEventStore:
    """Write immutable canonical events into the required Parquet partition layout."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def put(self, event: MarketEvent, *, venue: str, market: str, symbol: str) -> Path:
        parts = {
            "venue": venue.lower(),
            "market": market.lower(),
            "event_type": event.event_type.value,
            "symbol": symbol.upper(),
        }
        if any(not _PARTITION_TOKEN.fullmatch(value) for value in parts.values()):
            raise ValueError("market-event partition contains an unsafe token")
        date = datetime.fromisoformat(event.exchange_timestamp).date().isoformat()
        payload = to_primitive(event)
        event_id = canonical_hash(payload)
        digest = event_id.removeprefix("sha256:")
        destination = (
            self.root
            / "raw"
            / parts["venue"]
            / parts["market"]
            / parts["event_type"]
            / parts["symbol"]
            / f"date={date}"
            / f"{digest}.parquet"
        )
        if destination.exists():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        table = pa.table(
            {
                "event_id": [event_id],
                "instrument_id": [event.instrument_id],
                "event_type": [event.event_type.value],
                "exchange_timestamp": [event.exchange_timestamp],
                "receive_timestamp": [event.receive_timestamp],
                "sequence": [event.sequence],
                "payload_json": [json.dumps(payload["payload"], sort_keys=True)],
            }
        )
        try:
            pq.write_table(table, temporary, compression="zstd")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        return destination


class PartitionedBarStore:
    """Write closed canonical candles into query-efficient bar partitions."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def put(self, event: MarketEvent, *, venue: str, market: str, symbol: str) -> Path:
        raw_data = event.payload.get("data")
        candle = raw_data.get("k") if isinstance(raw_data, dict) else None
        if event.event_type.value != "candle" or not isinstance(candle, dict):
            raise ValueError("bar store accepts only canonical candle events")
        if candle.get("x") is not True:
            raise ValueError("bar store accepts only closed candles")
        parts = {
            "venue": venue.lower(),
            "market": market.lower(),
            "symbol": symbol.upper(),
            "timeframe": str(candle["i"]).lower(),
        }
        if any(not _PARTITION_TOKEN.fullmatch(value) for value in parts.values()):
            raise ValueError("bar partition contains an unsafe token")
        close_time = datetime.fromtimestamp(float(candle["T"]) / 1_000, UTC)
        digest = event.event_id.removeprefix("sha256:")
        destination = (
            self.root
            / "bars"
            / parts["venue"]
            / parts["market"]
            / parts["symbol"]
            / parts["timeframe"]
            / f"date={close_time.date().isoformat()}"
            / f"{digest}.parquet"
        )
        if destination.exists():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        table = pa.table(
            {
                "event_id": [event.event_id],
                "instrument_id": [event.instrument_id],
                "open_time_ms": [int(candle["t"])],
                "close_time_ms": [int(candle["T"])],
                "availability_time": [event.receive_timestamp],
                "open": [float(candle["o"])],
                "high": [float(candle["h"])],
                "low": [float(candle["l"])],
                "close": [float(candle["c"])],
                "volume": [float(candle["v"])],
            }
        )
        try:
            pq.write_table(table, temporary, compression="zstd")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        return destination


class PartitionedFeatureStore:
    """Write deterministic feature batches into immutable Parquet partitions."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def put(
        self,
        values: tuple[FeatureValue, ...],
        *,
        venue: str,
        market: str,
        symbol: str,
        timeframe: str,
    ) -> Path:
        if not values:
            raise ValueError("feature batch cannot be empty")
        feature_sets = {item.feature_set_version for item in values}
        instruments = {item.instrument_id for item in values}
        availability_dates = {
            datetime.fromisoformat(item.availability_time).date().isoformat() for item in values
        }
        if len(feature_sets) != 1 or len(instruments) != 1 or len(availability_dates) != 1:
            raise ValueError("feature batch must share one set, instrument, and availability date")
        parts = {
            "feature_set": next(iter(feature_sets)),
            "venue": venue.lower(),
            "market": market.lower(),
            "symbol": symbol.upper(),
            "timeframe": timeframe.lower(),
        }
        if any(not _PARTITION_TOKEN.fullmatch(value) for value in parts.values()):
            raise ValueError("feature partition contains an unsafe token")
        ordered_values = tuple(sorted(values, key=lambda item: item.feature_name))
        payloads = [to_primitive(item) for item in ordered_values]
        digest = canonical_hash(payloads).removeprefix("sha256:")
        destination = (
            self.root
            / "features"
            / parts["feature_set"]
            / parts["venue"]
            / parts["market"]
            / parts["symbol"]
            / parts["timeframe"]
            / f"date={next(iter(availability_dates))}"
            / f"{digest}.parquet"
        )
        if destination.exists():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        table = pa.table(
            {
                "feature_id": [item.feature_id for item in ordered_values],
                "feature_set_version": [item.feature_set_version for item in ordered_values],
                "feature_name": [item.feature_name for item in ordered_values],
                "instrument_id": [item.instrument_id for item in ordered_values],
                "source_event_time": [item.source_event_time for item in ordered_values],
                "source_close_time": [item.source_close_time for item in ordered_values],
                "availability_time": [item.availability_time for item in ordered_values],
                "value": [item.value for item in ordered_values],
            }
        )
        try:
            pq.write_table(table, temporary, compression="zstd")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        return destination


class PartitionedBacktestStore:
    """Write research result rows as immutable content-addressed Parquet."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def put_rows(
        self,
        *,
        candidate_id: str,
        run_name: str,
        created_at: str,
        rows: list[dict[str, object]],
    ) -> tuple[Path, str]:
        candidate = candidate_id.removeprefix("sha256:")
        if not candidate or not _PARTITION_TOKEN.fullmatch(candidate):
            raise ValueError("backtest candidate ID contains an unsafe token")
        if not _PARTITION_TOKEN.fullmatch(run_name):
            raise ValueError("backtest run name contains an unsafe token")
        clean_rows = [to_primitive(row) for row in rows]
        content_hash = canonical_hash(
            {
                "candidate_id": candidate_id,
                "run_name": run_name,
                "created_at": created_at,
                "rows": clean_rows,
            }
        )
        destination = (
            self.root
            / "backtests"
            / candidate
            / run_name
            / f"{content_hash.removeprefix('sha256:')}.parquet"
        )
        if destination.exists():
            return destination, content_hash
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        table = pa.table(
            {
                "row_number": list(range(len(clean_rows))),
                "candidate_id": [candidate_id] * len(clean_rows),
                "run_name": [run_name] * len(clean_rows),
                "created_at": [created_at] * len(clean_rows),
                "payload_json": [
                    json.dumps(row, sort_keys=True, separators=(",", ":")) for row in clean_rows
                ],
            }
        )
        try:
            pq.write_table(table, temporary, compression="zstd")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        return destination, content_hash
