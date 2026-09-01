"""Build canonical research bundles from immutable Parquet bars."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from sqlalchemy import select

from src.data.database import cost_model_manifest, feature_manifest, risk_snapshot
from src.data.universe import SqlUniverseStore
from src.domain._codec import canonical_hash, timestamp
from src.research.datasets import (
    CORE_RESEARCH_BUNDLE_ROLES,
    CanonicalResearchDatasetBuilder,
    DatasetResolutionError,
)


class DatasetBundleBuildError(RuntimeError):
    """A source partition cannot be used for a canonical research bundle."""


@dataclass(frozen=True)
class DatasetBundleBuildResult:
    state: str
    reason_code: str
    product_id: str
    bundle_id: str | None = None
    snapshot_ids: tuple[str, ...] = ()
    instrument_scope: tuple[str, ...] = ()
    source_partition_hashes: tuple[str, ...] = ()
    detail: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason_code": self.reason_code,
            "product_id": self.product_id,
            "bundle_id": self.bundle_id,
            "snapshot_ids": list(self.snapshot_ids),
            "instrument_scope": list(self.instrument_scope),
            "source_partition_hashes": list(self.source_partition_hashes),
            "detail": self.detail,
        }


class DatabaseDatasetBundleService:
    """Resolve point-in-time inputs and publish one immutable stage bundle."""

    def __init__(
        self,
        engine,
        parquet_root: Path,
        *,
        maximum_rows: int = 250_000,
        timeframe: str = "1m",
        minimum_history_days: int = 0,
    ) -> None:
        if maximum_rows <= 0:
            raise ValueError("maximum_rows must be positive")
        if minimum_history_days < 0:
            raise ValueError("minimum_history_days must be non-negative")
        self.engine = engine
        self.parquet_root = parquet_root.resolve()
        self.maximum_rows = maximum_rows
        self.timeframe = _text(timeframe, field="timeframe")
        self.minimum_history_days = minimum_history_days

    def run(
        self,
        *,
        product_id: str,
        universe_id: str,
        market_type: str,
        created_at: str,
        timeframe: str | None = None,
    ) -> DatasetBundleBuildResult:
        created = timestamp(created_at, field="created_at")
        resolved_universe = self._universe(universe_id, created)
        if resolved_universe is None:
            return self._waiting(product_id, "point_in_time_universe_unavailable")
        universe_snapshot_id, instruments = resolved_universe
        feature_id = self._latest_manifest(
            feature_manifest,
            created=created,
            matches=lambda payload: _matches_feature(payload, market_type),
        )
        cost_id = self._latest_manifest(
            cost_model_manifest,
            created=created,
            matches=lambda payload: str(payload.get("product_id") or "") == product_id,
        )
        if feature_id is None or cost_id is None:
            return self._waiting(product_id, "dataset_manifests_unavailable")
        instrument_scope = tuple(item[0] for item in instruments)
        try:
            bars, source_hashes = self._load_bars(
                instruments,
                market_type=market_type,
                timeframe=timeframe or self.timeframe,
                created=created,
            )
        except DatasetBundleBuildError as exc:
            return DatasetBundleBuildResult(
                state="blocked_dataset",
                reason_code="bar_partition_invalid",
                product_id=product_id,
                detail=str(exc),
            )
        if not bars:
            return self._waiting(product_id, "historical_bars_unavailable")
        if not _history_span_satisfies(bars, self.minimum_history_days):
            return self._waiting(product_id, "historical_history_insufficient")
        try:
            if market_type.lower() == "futures":
                bars = self._attach_funding_rates(
                    product_id=product_id,
                    bars=bars,
                    instrument_scope=instrument_scope,
                    created=created,
                )
            _validate_market_bars(bars, market_type=market_type)
        except DatasetBundleBuildError as exc:
            return DatasetBundleBuildResult(
                state="blocked_dataset",
                reason_code="bar_quality_invalid",
                product_id=product_id,
                detail=str(exc),
            )
        try:
            intervals = _stage_intervals(bars)
            bundle = CanonicalResearchDatasetBuilder(self.engine).build_from_bars(
                product_id,
                bars=bars,
                intervals=intervals,
                universe_snapshot_id=universe_snapshot_id,
                feature_manifest_id=feature_id,
                cost_model_id=cost_id,
                parameter_set_id=_parameter_set_id(product_id),
                instrument_scope=instrument_scope,
                created_at=created,
                source_partition_hashes=source_hashes,
                engine_version="dataset-service/v1",
            )
        except DatasetResolutionError as exc:
            return self._waiting(product_id, _reason_code(str(exc)))
        return DatasetBundleBuildResult(
            state="ready",
            reason_code="canonical_dataset_bundle_ready",
            product_id=product_id,
            bundle_id=bundle.bundle_id,
            snapshot_ids=tuple(bundle.stage_snapshot_ids.values()),
            instrument_scope=instrument_scope,
            source_partition_hashes=source_hashes,
        )

    @staticmethod
    def _waiting(product_id: str, reason_code: str) -> DatasetBundleBuildResult:
        return DatasetBundleBuildResult(
            state="waiting_for_dataset",
            reason_code=reason_code,
            product_id=product_id,
        )

    def _universe(
        self, universe_id: str, observed_at: str
    ) -> tuple[str, tuple[tuple[str, str, str], ...]] | None:
        memberships = SqlUniverseStore(self.engine).members_at(
            universe_id=universe_id,
            observed_at=observed_at,
            eligible_only=True,
        )
        if not memberships:
            return None
        instruments = tuple(
            sorted(
                {
                    (
                        item.instrument.instrument_id,
                        item.instrument.exchange_symbol,
                        item.instrument.venue,
                    )
                    for item in memberships
                }
            )
        )
        return memberships[0].snapshot_id, instruments

    def _latest_manifest(
        self,
        table: Any,
        *,
        created: str,
        matches: Any,
    ) -> str | None:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(table.c.id, table.c.payload)
                .where(table.c.created_at <= created)
                .order_by(table.c.created_at.desc(), table.c.id.desc())
            ).mappings()
            for row in rows:
                payload = row["payload"]
                if isinstance(payload, Mapping) and matches(payload):
                    return str(row["id"])
        return None

    def _load_bars(
        self,
        instruments: tuple[tuple[str, str, str], ...],
        *,
        market_type: str,
        timeframe: str,
        created: str,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
        rows: list[dict[str, Any]] = []
        source_hashes: set[str] = set()
        for instrument_id, symbol, venue in instruments:
            root = self.parquet_root / "bars" / venue.lower() / market_type.lower() / symbol.upper()
            root = root / _text(timeframe, field="timeframe")
            for path in sorted(root.rglob("*.parquet")) if root.is_dir() else ():
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    file_rows = pq.read_table(path).to_pylist()
                except Exception as exc:
                    raise DatasetBundleBuildError(f"cannot read bar partition {path}") from exc
                accepted = [
                    normalised
                    for raw in file_rows
                    if isinstance(raw, Mapping)
                    and (normalised := _normalise_bar(raw, instrument_id, created)) is not None
                ]
                if accepted:
                    rows.extend(accepted)
                    source_hashes.add(_file_identity(path))
        rows.sort(key=lambda row: (str(row["close_timestamp"]), str(row["instrument_id"])))
        if len(rows) > self.maximum_rows:
            rows = rows[-self.maximum_rows :]
        return tuple(rows), tuple(sorted(source_hashes))

    def _attach_funding_rates(
        self,
        *,
        product_id: str,
        bars: tuple[dict[str, Any], ...],
        instrument_scope: tuple[str, ...],
        created: str,
    ) -> tuple[dict[str, Any], ...]:
        funding = self._funding_events(product_id, created)
        missing = sorted(set(instrument_scope) - set(funding))
        if missing:
            raise DatasetBundleBuildError(
                "futures funding history is missing for: " + ", ".join(missing)
            )
        enriched: list[dict[str, Any]] = []
        for row in bars:
            instrument_id = str(row["instrument_id"])
            close = str(row["close_timestamp"])
            values = dict(row)
            values["funding_rate"] = funding[instrument_id].get(close, 0.0)
            enriched.append(values)
        return tuple(enriched)

    def _funding_events(self, product_id: str, created: str) -> dict[str, dict[str, float]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(risk_snapshot.c.payload, risk_snapshot.c.created_at)
                .where(risk_snapshot.c.created_at <= created)
                .order_by(risk_snapshot.c.created_at, risk_snapshot.c.id)
            ).mappings()
        result: dict[str, dict[str, float]] = {}
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, Mapping):
                continue
            if (
                payload.get("kind") != "market_data_input"
                or payload.get("product_id") != product_id
            ):
                continue
            values = payload.get("values")
            if not isinstance(values, Mapping) or values.get("funding") is None:
                continue
            instrument_id = str(payload.get("instrument_id") or "")
            source_time = payload.get("source_event_time", row["created_at"])
            if not instrument_id or source_time is None:
                continue
            result.setdefault(instrument_id, {})[_bar_time(source_time, field="funding_time")] = (
                _signed_number(values["funding"], field="funding")
            )
        return result


def _normalise_bar(
    raw: Mapping[str, Any], instrument_id: str, created: str
) -> dict[str, Any] | None:
    close_value = raw.get("close_timestamp", raw.get("close_time_ms", raw.get("timestamp")))
    if close_value is None:
        return None
    available_value = raw.get("availability_time", raw.get("available_at", close_value))
    close = _bar_time(close_value, field="close_timestamp")
    available = _bar_time(available_value, field="availability_time")
    if close > created or available > created:
        return None
    row = dict(raw)
    row["instrument_id"] = instrument_id
    row["close_timestamp"] = close
    row["availability_time"] = available
    return row


def _history_span_satisfies(rows: tuple[dict[str, Any], ...], minimum_days: int) -> bool:
    if minimum_days == 0:
        return True
    times = [dt.datetime.fromisoformat(str(row["close_timestamp"])) for row in rows]
    return max(times) - min(times) >= dt.timedelta(days=minimum_days)


def _validate_market_bars(rows: tuple[dict[str, Any], ...], *, market_type: str) -> None:
    previous: dict[str, str] = {}
    for index, row in enumerate(rows):
        instrument_id = str(row.get("instrument_id") or "")
        observed = str(row.get("close_timestamp") or "")
        if previous.get(instrument_id, "") >= observed:
            raise DatasetBundleBuildError(
                f"bar timestamps are not strictly increasing at row {index}"
            )
        previous[instrument_id] = observed
        values = {
            name: _positive_number(row.get(name), field=f"bar[{index}].{name}")
            for name in ("open", "high", "low", "close")
        }
        volume = _nonnegative_number(row.get("volume"), field=f"bar[{index}].volume")
        if values["high"] < max(values["open"], values["close"]) or values["low"] > min(
            values["open"], values["close"]
        ):
            raise DatasetBundleBuildError(f"bar[{index}] has inconsistent OHLC values")
        if volume < 0.0:
            raise DatasetBundleBuildError(f"bar[{index}].volume is negative")
        if market_type.lower() == "futures" and "funding_rate" not in row:
            raise DatasetBundleBuildError(f"bar[{index}] has no signed funding rate")


def _positive_number(value: Any, *, field: str) -> float:
    result = _signed_number(value, field=field)
    if result <= 0.0:
        raise DatasetBundleBuildError(f"{field} must be positive")
    return result


def _nonnegative_number(value: Any, *, field: str) -> float:
    result = _signed_number(value, field=field)
    if result < 0.0:
        raise DatasetBundleBuildError(f"{field} must be non-negative")
    return result


def _signed_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise DatasetBundleBuildError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetBundleBuildError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise DatasetBundleBuildError(f"{field} must be finite")
    return result


def _bar_time(value: Any, *, field: str) -> str:
    if isinstance(value, bool):
        raise DatasetBundleBuildError(f"{field} must be a timestamp")
    if isinstance(value, int | float):
        try:
            parsed = dt.datetime.fromtimestamp(float(value) / 1_000, dt.UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise DatasetBundleBuildError(f"{field} is invalid") from exc
        return timestamp(parsed, field=field)
    try:
        return timestamp(str(value), field=field)
    except ValueError as exc:
        raise DatasetBundleBuildError(f"{field} is invalid") from exc


def _stage_intervals(rows: tuple[dict[str, Any], ...]) -> dict[str, dict[str, str]]:
    times = sorted({str(row["close_timestamp"]) for row in rows})
    if len(times) < len(CORE_RESEARCH_BUNDLE_ROLES):
        raise DatasetResolutionError("dataset data_pending: insufficient distinct bar timestamps")
    indexes = (0, len(times) // 4, len(times) // 2, (len(times) * 3) // 4)
    starts = [times[index] for index in indexes]
    final_end = (
        (dt.datetime.fromisoformat(times[-1]) + dt.timedelta(seconds=1))
        .replace(microsecond=0)
        .isoformat()
    )
    ends = [*starts[1:], final_end]
    return {
        role: {"start": start, "end": end}
        for role, start, end in zip(CORE_RESEARCH_BUNDLE_ROLES, starts, ends, strict=True)
    }


def _matches_feature(payload: Mapping[str, Any], market_type: str) -> bool:
    declared = payload.get("market_type")
    return declared is None or str(declared).lower() == market_type.lower()


def _parameter_set_id(product_id: str) -> str:
    return canonical_hash(
        {"schema": "platform.parameter_set/v1", "product_id": product_id, "parameters": {}}
    )


def _file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetBundleBuildError(f"cannot hash bar partition {path}") from exc
    return f"sha256:{digest.hexdigest()}"


def _reason_code(message: str) -> str:
    if "insufficient distinct" in message:
        return "insufficient_distinct_bar_timestamps"
    if "no available bars for role" in message:
        return "historical_bars_incomplete"
    return "canonical_dataset_bundle_invalid"


def _text(value: Any, *, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} cannot be empty")
    return result
