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
from dataclasses import dataclass
from datetime import UTC, datetime
from collections.abc import Iterable, Mapping
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
                    raise RuntimeError(f"content-address collision at {destination}") from None
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
                "close_timestamp": [event.close_timestamp],
                "receive_timestamp": [event.receive_timestamp],
                "availability_time": [event.availability_time],
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

    def put_batch(
        self,
        events: Iterable[tuple[MarketEvent, str, str, str]],
        *,
        row_group_size: int = 1_024,
    ) -> tuple[Path, ...]:
        """Persist one multi-row Parquet segment per canonical partition."""

        materialised = tuple(events)
        if not materialised:
            raise ValueError("market-event batch cannot be empty")
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive")
        partitions: dict[tuple[str, str, str, str, str], list[MarketEvent]] = {}
        for event, venue, market, symbol in materialised:
            values = (venue.lower(), market.lower(), event.event_type.value, symbol.upper())
            date = datetime.fromisoformat(event.exchange_timestamp).date().isoformat()
            key = (*values, date)
            if any(not _PARTITION_TOKEN.fullmatch(value) for value in values):
                raise ValueError("market-event partition contains an unsafe token")
            partitions.setdefault(key, []).append(event)
        paths: list[Path] = []
        for (venue, market, event_type, symbol, date), partition_events in sorted(partitions.items()):
            ordered = tuple(sorted(partition_events, key=lambda item: (item.sequence, item.event_id)))
            digest = canonical_hash([item.event_id for item in ordered]).removeprefix("sha256:")
            destination = (
                self.root / "raw" / venue / market / event_type / symbol / f"date={date}"
                / f"batch-{digest}.parquet"
            )
            if destination.exists():
                paths.append(destination)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            table = pa.table(
                {
                    "event_id": [item.event_id for item in ordered],
                    "exchange_identity": [item.exchange_identity for item in ordered],
                    "instrument_id": [item.instrument_id for item in ordered],
                    "event_type": [item.event_type.value for item in ordered],
                    "exchange_timestamp": [item.exchange_timestamp for item in ordered],
                    "close_timestamp": [item.close_timestamp for item in ordered],
                    "receive_timestamp": [item.receive_timestamp for item in ordered],
                    "availability_time": [item.availability_time for item in ordered],
                    "sequence": [item.sequence for item in ordered],
                    "payload_json": [
                        json.dumps(dict(item.payload), sort_keys=True, separators=(",", ":"))
                        for item in ordered
                    ],
                }
            )
            try:
                pq.write_table(table, temporary, compression="zstd", row_group_size=row_group_size)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            paths.append(destination)
        return tuple(paths)


@dataclass(frozen=True)
class MarketEventBatchSegment:
    path: Path
    content_hash: str
    row_count: int


class DurableMarketBatchSpool:
    """Durable local spool that publishes complete event segments atomically."""

    def __init__(self, root: Path, *, max_rows: int = 5_000, max_bytes: int = 8 * 1024 * 1024):
        if max_rows <= 0 or max_bytes <= 0:
            raise ValueError("batch spool limits must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self._rows: list[dict[str, object]] = []
        self._bytes = 0

    def append(self, event: MarketEvent, *, venue: str, market: str, symbol: str) -> MarketEventBatchSegment | None:
        row = {
            "venue": venue,
            "market": market,
            "symbol": symbol,
            "event": to_primitive(event),
        }
        encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self._rows.append(row)
        self._bytes += len(encoded)
        if len(self._rows) >= self.max_rows or self._bytes >= self.max_bytes:
            return self.flush()
        return None

    def flush(self) -> MarketEventBatchSegment | None:
        if not self._rows:
            return None
        body = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in self._rows)
        content_hash = canonical_hash(body)
        destination = self.root / f"segment-{content_hash.removeprefix('sha256:')}.jsonl"
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        segment = MarketEventBatchSegment(destination, content_hash, len(self._rows))
        self._rows = []
        self._bytes = 0
        return segment

    @staticmethod
    def read(segment: MarketEventBatchSegment | Path) -> tuple[dict[str, object], ...]:
        path = segment.path if isinstance(segment, MarketEventBatchSegment) else segment
        if path.is_symlink() or not path.is_file():
            raise ValueError("market batch segment must be a regular file")
        rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        if not rows:
            raise ValueError("market batch segment is empty")
        return rows


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
                "availability_time": [event.availability_time],
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
