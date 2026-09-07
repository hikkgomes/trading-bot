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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.feature_store import FeatureValue
from src.domain._codec import canonical_hash, to_primitive
from src.domain.market_events import MarketEvent

_PARTITION_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _fsync_directory(path: Path) -> None:
    """Make a completed atomic rename visible after a power loss."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        _fsync_directory(destination.parent)
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
        directory = (
            self.root
            / "raw"
            / parts["venue"]
            / parts["market"]
            / parts["event_type"]
            / parts["symbol"]
            / f"date={date}"
        )
        existing_paths = tuple(path for path in directory.glob("*.parquet") if path.is_file())
        rows = _merge_event_rows(existing_paths, {event.event_id: _market_event_row(event)})
        destination = directory / f"batch-{canonical_hash(rows).removeprefix('sha256:')}.parquet"
        if destination.exists():
            return destination
        _write_rows(destination, rows, row_group_size=1_024)
        if existing_paths:
            _quarantine_parquet(existing_paths, root=self.root, label="raw")
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
        for (venue, market, event_type, symbol, date), partition_events in sorted(
            partitions.items()
        ):
            unique_events = {event.event_id: event for event in partition_events}
            ordered = tuple(
                sorted(unique_events.values(), key=lambda item: (item.sequence, item.event_id))
            )
            rows = tuple(_market_event_row(item) for item in ordered)
            digest = canonical_hash(rows).removeprefix("sha256:")
            destination = (
                self.root
                / "raw"
                / venue
                / market
                / event_type
                / symbol
                / f"date={date}"
                / f"batch-{digest}.parquet"
            )
            if destination.exists():
                existing_rows = tuple(
                    dict(row) for row in pq.ParquetFile(destination).read().to_pylist()
                )
                if canonical_hash(existing_rows).removeprefix("sha256:") != digest:
                    raise RuntimeError(f"raw partition content hash collision: {destination}")
                paths.append(destination)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            table = pa.table(
                {
                    key: [row[key] for row in rows]
                    for key in rows[0]
                }
            )
            try:
                pq.write_table(table, temporary, compression="zstd", row_group_size=row_group_size)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
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
        for state in ("pending", "claimed", "committed", "deleted"):
            (self.root / state).mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self._rows: list[dict[str, object]] = []
        self._bytes = 0

    def append(
        self, event: MarketEvent, *, venue: str, market: str, symbol: str
    ) -> MarketEventBatchSegment | None:
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
        body = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in self._rows
        )
        content_hash = canonical_hash(body)
        destination = (
            self.root / "pending" / f"segment-{content_hash.removeprefix('sha256:')}.jsonl"
        )
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        segment = MarketEventBatchSegment(destination, content_hash, len(self._rows))
        self._rows = []
        self._bytes = 0
        return segment

    @staticmethod
    def read(segment: MarketEventBatchSegment | Path) -> tuple[dict[str, object], ...]:
        path = segment.path if isinstance(segment, MarketEventBatchSegment) else segment
        if path.is_symlink() or not path.is_file():
            raise ValueError("market batch segment must be a regular file")
        rows = tuple(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
        if not rows:
            raise ValueError("market batch segment is empty")
        return rows

    def claim(self, path: Path) -> Path:
        """Move a durable segment to the claimed state, resolving crash recovery."""

        source = self._resolve(path)
        if source.parent.name == "claimed":
            return source
        if source.parent.name == "committed":
            return source
        destination = self.root / "claimed" / source.name
        if destination.exists():
            source.unlink(missing_ok=True)
            return destination
        os.replace(source, destination)
        _fsync_directory(destination.parent)
        return destination

    def commit(self, path: Path) -> Path:
        """Durably mark a segment committed after all output hashes are present."""

        source = self._resolve(path)
        if source.parent.name == "committed":
            return source
        destination = self.root / "committed" / source.name
        if destination.exists():
            source.unlink(missing_ok=True)
            return destination
        os.replace(source, destination)
        _fsync_directory(destination.parent)
        return destination

    def delete(self, path: Path) -> None:
        """Remove a segment only after its committed output is durable."""

        source = self._resolve(path)
        if source.parent.name != "committed":
            raise ValueError("market batch segment must be committed before deletion")
        destination = self.root / "deleted" / source.name
        if destination.exists():
            source.unlink(missing_ok=True)
        else:
            os.replace(source, destination)
            _fsync_directory(destination.parent)
            destination.unlink(missing_ok=True)
        _fsync_directory(self.root / "deleted")

    def recoverable(self) -> tuple[Path, ...]:
        """Return every segment which still needs its output acknowledged.

        A crash after the output transaction commits but before the queue job
        deletes the segment leaves it in ``committed``.  Replaying that state
        is safe because all downstream writes are content-addressed.
        """

        return tuple(
            sorted(
                path
                for state in ("pending", "claimed", "committed")
                for path in (self.root / state).glob("segment-*.jsonl")
                if path.is_file() and not path.is_symlink()
            )
        )

    def _resolve(self, path: Path) -> Path:
        candidate = path.resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("market batch segment is outside its spool root") from exc
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
        name = candidate.name
        matches = tuple(
            path
            for state in ("pending", "claimed", "committed")
            for path in (self.root / state).glob(name)
            if path.is_file() and not path.is_symlink()
        )
        if len(matches) != 1:
            raise FileNotFoundError(f"market batch segment is unavailable: {path}")
        return matches[0]


@dataclass(frozen=True)
class FeatureBatchSegment:
    path: Path
    content_hash: str
    row_count: int


class DurableFeatureBatchSpool:
    """Durable rolling spool for calculated feature rows."""

    def __init__(
        self,
        root: Path,
        *,
        max_rows: int = 500,
        max_bytes: int = 2 * 1024 * 1024,
        flush_seconds: int = 60,
    ) -> None:
        if max_rows <= 0 or max_bytes <= 0 or flush_seconds <= 0:
            raise ValueError("feature spool limits must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for state in ("pending", "claimed", "committed", "deleted"):
            (self.root / state).mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.flush_seconds = flush_seconds
        self._rows: list[dict[str, object]] = []
        self._bytes = 0
        self._first_available_at: datetime | None = None

    def append(
        self,
        values: Iterable[FeatureValue],
        *,
        venue: str,
        market: str,
        symbol: str,
        timeframe: str,
    ) -> FeatureBatchSegment | None:
        available_values = tuple(values)
        if not available_values:
            raise ValueError("feature spool append cannot be empty")
        for value in available_values:
            row = {
                "venue": venue,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "value": to_primitive(value),
            }
            encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            self._rows.append(row)
            self._bytes += len(encoded)
            available_at = datetime.fromisoformat(value.availability_time)
            self._first_available_at = self._first_available_at or available_at
        current = max(datetime.fromisoformat(value.availability_time) for value in available_values)
        age = (
            current - self._first_available_at
            if self._first_available_at is not None
            else timedelta(0)
        )
        if (
            len(self._rows) >= self.max_rows
            or self._bytes >= self.max_bytes
            or age.total_seconds() >= self.flush_seconds
        ):
            return self.flush()
        return None

    def flush(self) -> FeatureBatchSegment | None:
        if not self._rows:
            return None
        body = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in self._rows
        )
        content_hash = canonical_hash(body)
        destination = (
            self.root / "pending" / f"segment-{content_hash.removeprefix('sha256:')}.jsonl"
        )
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        segment = FeatureBatchSegment(destination, content_hash, len(self._rows))
        self._rows = []
        self._bytes = 0
        self._first_available_at = None
        return segment

    @staticmethod
    def read(segment: FeatureBatchSegment | Path) -> tuple[dict[str, object], ...]:
        path = segment.path if isinstance(segment, FeatureBatchSegment) else segment
        if path.is_symlink() or not path.is_file():
            raise ValueError("feature batch segment must be a regular file")
        rows = tuple(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
        if not rows:
            raise ValueError("feature batch segment is empty")
        return rows

    def claim(self, path: Path) -> Path:
        return self._move(path, "claimed")

    def commit(self, path: Path) -> Path:
        return self._move(path, "committed")

    def delete(self, path: Path) -> None:
        source = self._resolve(path)
        if source.parent.name != "committed":
            raise ValueError("feature batch segment must be committed before deletion")
        destination = self.root / "deleted" / source.name
        if destination.exists():
            source.unlink(missing_ok=True)
        else:
            os.replace(source, destination)
            _fsync_directory(destination.parent)
            destination.unlink(missing_ok=True)
        _fsync_directory(self.root / "deleted")

    def recoverable(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                path
                for state in ("pending", "claimed", "committed")
                for path in (self.root / state).glob("segment-*.jsonl")
                if path.is_file() and not path.is_symlink()
            )
        )

    def _move(self, path: Path, state: str) -> Path:
        source = self._resolve(path)
        if source.parent.name == state or source.parent.name == "committed":
            return source
        destination = self.root / state / source.name
        if destination.exists():
            source.unlink(missing_ok=True)
            return destination
        os.replace(source, destination)
        _fsync_directory(destination.parent)
        return destination

    def _resolve(self, path: Path) -> Path:
        candidate = path.resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("feature batch segment is outside its spool root") from exc
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
        matches = tuple(
            item
            for state in ("pending", "claimed", "committed")
            for item in (self.root / state).glob(candidate.name)
            if item.is_file() and not item.is_symlink()
        )
        if len(matches) != 1:
            raise FileNotFoundError(f"feature batch segment is unavailable: {path}")
        return matches[0]


class PartitionedBarStore:
    """Write closed canonical candles into query-efficient bar partitions."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def put(self, event: MarketEvent, *, venue: str, market: str, symbol: str) -> Path:
        return self.put_batch(((event, venue, market, symbol),))[0]

    def put_batch(
        self,
        events: Iterable[tuple[MarketEvent, str, str, str]],
        *,
        row_group_size: int = 1_024,
    ) -> tuple[Path, ...]:
        materialised = tuple(events)
        if not materialised:
            raise ValueError("bar batch cannot be empty")
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive")
        partitions: dict[
            tuple[str, str, str, str, str], list[tuple[MarketEvent, dict[str, object]]]
        ] = {}
        for event, venue, market, symbol in materialised:
            key, row = _bar_event_row(event, venue=venue, market=market, symbol=symbol)
            partitions.setdefault(key, []).append((event, row))
        return tuple(
            _write_bar_partition(
                self.root,
                key,
                items,
                row_group_size=row_group_size,
            )
            for key, items in sorted(partitions.items())
        )


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
        return self.put_batch(
            values,
            venue=venue,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
        )[0]

    def put_batch(
        self,
        values: Iterable[FeatureValue],
        *,
        venue: str,
        market: str,
        symbol: str,
        timeframe: str,
        row_group_size: int = 1_024,
    ) -> tuple[Path, ...]:
        materialised = tuple(values)
        if not materialised:
            raise ValueError("feature batch cannot be empty")
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive")
        feature_sets = {item.feature_set_version for item in materialised}
        instruments = {item.instrument_id for item in materialised}
        availability_dates = {
            datetime.fromisoformat(item.availability_time).date().isoformat()
            for item in materialised
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
        ordered_values = tuple(
            sorted(materialised, key=lambda item: (item.source_close_time, item.feature_name))
        )
        return (
            _write_feature_partition(
                self.root,
                parts,
                next(iter(availability_dates)),
                ordered_values,
                row_group_size=row_group_size,
            ),
        )


def _market_event_row(event: MarketEvent) -> dict[str, object]:
    payload = to_primitive(event)
    return {
        "event_id": event.event_id,
        "exchange_identity": event.exchange_identity,
        "instrument_id": event.instrument_id,
        "event_type": event.event_type.value,
        "exchange_timestamp": event.exchange_timestamp,
        "close_timestamp": event.close_timestamp,
        "receive_timestamp": event.receive_timestamp,
        "availability_time": event.availability_time,
        "sequence": event.sequence,
        "payload_json": json.dumps(dict(payload["payload"]), sort_keys=True, separators=(",", ":")),
    }


def _merge_event_rows(
    paths: Iterable[Path], incoming: Mapping[str, dict[str, object]]
) -> tuple[dict[str, object], ...]:
    rows = dict(incoming)
    for path in paths:
        for raw in pq.ParquetFile(path).read().to_pylist():
            if isinstance(raw, Mapping):
                identity = str(raw.get("event_id") or canonical_hash(dict(raw)))
                rows.setdefault(identity, dict(raw))
    return tuple(
        row
        for _identity, row in sorted(
            rows.items(),
            key=lambda item: (str(item[1].get("exchange_timestamp") or ""), item[0]),
        )
    )


def _bar_event_row(
    event: MarketEvent, *, venue: str, market: str, symbol: str
) -> tuple[tuple[str, str, str, str, str], dict[str, object]]:
    raw_data = event.payload.get("data")
    candle = raw_data.get("k") if isinstance(raw_data, Mapping) else None
    if event.event_type.value != "candle" or not isinstance(candle, Mapping):
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
    row = {
        "event_id": event.event_id,
        "instrument_id": event.instrument_id,
        "open_time_ms": int(candle["t"]),
        "close_time_ms": int(candle["T"]),
        "close_timestamp": close_time.replace(microsecond=0).isoformat(),
        "availability_time": event.availability_time,
        "open": float(candle["o"]),
        "high": float(candle["h"]),
        "low": float(candle["l"]),
        "close": float(candle["c"]),
        "volume": float(candle["v"]),
        "spread_bps": _optional_float(candle.get("spread_bps")),
        "visible_depth": _optional_float(candle.get("visible_depth")),
        "volatility": _optional_float(candle.get("volatility")),
        "funding": _optional_float(candle.get("funding")),
    }
    key = (
        parts["venue"],
        parts["market"],
        parts["symbol"],
        parts["timeframe"],
        close_time.date().isoformat(),
    )
    return key, row


def _optional_float(value: object) -> float | None:
    return None if value is None else float(cast(Any, value))


def _write_bar_partition(
    root: Path,
    key: tuple[str, str, str, str, str],
    items: list[tuple[MarketEvent, dict[str, object]]],
    *,
    row_group_size: int,
) -> Path:
    venue, market, symbol, timeframe, date = key
    incoming = {event.event_id: row for event, row in items}
    directory = root / "bars" / venue / market / symbol / timeframe / f"date={date}"
    existing_paths = (
        tuple(path for path in sorted(directory.glob("*.parquet")) if path.is_file())
        if directory.exists()
        else ()
    )
    rows = _merge_bar_rows(existing_paths, incoming)
    digest = canonical_hash(rows).removeprefix("sha256:")
    destination = directory / f"batch-{digest}.parquet"
    if destination.exists():
        return destination
    _write_rows(destination, rows, row_group_size=row_group_size)
    if existing_paths:
        _quarantine_parquet(existing_paths, root=root, label="bars")
    return destination


def _merge_bar_rows(
    paths: Iterable[Path], incoming: Mapping[str, dict[str, object]]
) -> tuple[dict[str, object], ...]:
    rows = dict(incoming)
    for path in paths:
        for raw in pq.ParquetFile(path).read().to_pylist():
            if isinstance(raw, Mapping):
                identity = str(raw.get("event_id") or canonical_hash(dict(raw)))
                rows.setdefault(identity, dict(raw))
    return tuple(
        row
        for _identity, row in sorted(
            rows.items(),
            key=lambda item: (
                str(item[1].get("close_timestamp") or item[1].get("close_time_ms") or ""),
                item[0],
            ),
        )
    )


def _write_feature_partition(
    root: Path,
    parts: Mapping[str, str],
    date: str,
    values: tuple[FeatureValue, ...],
    *,
    row_group_size: int,
) -> Path:
    payloads = [to_primitive(item) for item in values]
    digest = canonical_hash(payloads).removeprefix("sha256:")
    directory = (
        root
        / "features"
        / parts["feature_set"]
        / parts["venue"]
        / parts["market"]
        / parts["symbol"]
        / parts["timeframe"]
        / f"date={date}"
    )
    destination = directory / f"batch-{digest}.parquet"
    if destination.exists():
        return destination
    directory.mkdir(parents=True, exist_ok=True)
    existing_paths = tuple(path for path in sorted(directory.glob("*.parquet")) if path.is_file())
    if existing_paths:
        values = _merge_feature_values(existing_paths, values)
        digest = canonical_hash([to_primitive(item) for item in values]).removeprefix("sha256:")
        destination = directory / f"batch-{digest}.parquet"
        if destination.exists():
            _quarantine_parquet(
                (path for path in existing_paths if path != destination),
                root=root,
                label="features",
            )
            return destination
    _write_feature_values(destination, values, row_group_size=row_group_size)
    if existing_paths:
        _quarantine_parquet(existing_paths, root=root, label="features")
    return destination


def _merge_feature_values(
    paths: Iterable[Path], incoming: Iterable[FeatureValue]
) -> tuple[FeatureValue, ...]:
    values: dict[str, FeatureValue] = {item.feature_id: item for item in incoming}
    for path in paths:
        for raw in pq.ParquetFile(path).read().to_pylist():
            if isinstance(raw, Mapping):
                item = FeatureValue(
                    feature_set_version=str(raw["feature_set_version"]),
                    feature_name=str(raw["feature_name"]),
                    instrument_id=str(raw["instrument_id"]),
                    source_event_time=str(raw["source_event_time"]),
                    source_close_time=str(raw["source_close_time"]),
                    availability_time=str(raw["availability_time"]),
                    value=float(raw["value"]),
                )
                values.setdefault(item.feature_id, item)
    return tuple(
        sorted(values.values(), key=lambda item: (item.source_close_time, item.feature_name))
    )


def _write_rows(
    destination: Path,
    rows: tuple[Mapping[str, object], ...],
    *,
    row_group_size: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    table = pa.table(
        {key: [row.get(key) for row in rows] for key in {key for row in rows for key in row}}
    )
    try:
        pq.write_table(table, temporary, compression="zstd", row_group_size=row_group_size)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(destination.parent)


def _write_feature_values(
    destination: Path,
    values: tuple[FeatureValue, ...],
    *,
    row_group_size: int,
) -> None:
    rows = tuple(
        {
            "feature_id": item.feature_id,
            "feature_set_version": item.feature_set_version,
            "feature_name": item.feature_name,
            "instrument_id": item.instrument_id,
            "source_event_time": item.source_event_time,
            "source_close_time": item.source_close_time,
            "availability_time": item.availability_time,
            "value": item.value,
        }
        for item in values
    )
    _write_rows(destination, rows, row_group_size=row_group_size)


def _quarantine_parquet(paths: Iterable[Path], *, root: Path, label: str) -> None:
    """Move superseded active files aside after the replacement is durable."""

    quarantine_root = root / "quarantine" / label
    moved: list[Path] = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        target = quarantine_root / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{target.name}.{uuid.uuid4().hex}")
        os.replace(path, target)
        moved.append(target)
    for directory in {path.parent for path in moved}:
        _fsync_directory(directory)


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
