"""Leased immutable Parquet writer for canonical market events."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import select

from src.data.database import job
from src.data.parquet_store import (
    DurableMarketBatchSpool,
    PartitionedBarStore,
    PartitionedMarketEventStore,
)
from src.domain._codec import canonical_hash
from src.domain.market_events import ExchangeSequenceTracker, MarketEvent, MarketEventType
from src.risk.engine import SqlRiskSnapshotStore
from src.services.historical_bootstrap import HistoricalBootstrapError, HistoricalBootstrapService
from src.services.market_gap_recovery import BinanceMarketGapRepair
from src.services.scheduler import DatabaseJobQueue
from src.services.storage_compaction import compact_data_root

_PARTITION_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


class DatabaseMarketDataWriter:
    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        root: Path,
        snapshot_store: SqlRiskSnapshotStore | None = None,
        product_ids_by_market: Mapping[str, Iterable[str]] | None = None,
        gap_repair: Callable[[Mapping[str, Any]], Iterable[MarketEvent]] | None = None,
        historical_bootstrap_service: HistoricalBootstrapService | None = None,
        lease_seconds: int = 60,
        compaction_interval_seconds: int = 3_600,
        compaction_maximum_rows: int = 5_000,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.store = PartitionedMarketEventStore(root)
        self.bar_store = PartitionedBarStore(root)
        self.snapshot_store = snapshot_store
        self.product_ids_by_market = {
            str(market): tuple(str(product_id) for product_id in product_ids)
            for market, product_ids in (product_ids_by_market or {}).items()
        }
        self.lease_seconds = lease_seconds
        self.sequence_tracker = ExchangeSequenceTracker()
        self.gap_repair = gap_repair or BinanceMarketGapRepair()
        self.historical_bootstrap_service = historical_bootstrap_service
        if compaction_interval_seconds <= 0 or compaction_maximum_rows <= 0:
            raise ValueError("storage compaction bounds must be positive")
        self.compaction_interval_seconds = compaction_interval_seconds
        self.compaction_maximum_rows = compaction_maximum_rows
        self._last_compaction_at: dt.datetime | None = None

    def run_once(self, *, now: str) -> dict[str, Any]:
        self._cleanup_completed_spool_segments()
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=(
                "market_event_batch_write",
                "market_event_write",
                "market_data_gap_recovery",
                *(
                    ("historical_bootstrap",)
                    if self.historical_bootstrap_service is not None
                    else ()
                ),
            ),
        )
        if claimed is None:
            compaction = self._maybe_compact(now)
            if compaction is not None:
                return compaction
            return {"reason_code": "market_event_queue_empty"}
        if claimed.name == "historical_bootstrap":
            return self._run_historical_bootstrap(claimed, now=now)
        if claimed.name == "market_data_gap_recovery":
            return self._run_gap_recovery(claimed, now=now)
        if claimed.name == "market_event_batch_write":
            return self._run_batch_write(claimed, now=now)
        return self._run_single_event_write(claimed, now=now)

    def _run_batch_write(self, claimed, *, now: str) -> dict[str, Any]:
        try:
            result = self._write_batch(claimed.payload)
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
        committed_segment = result.pop("committed_segment", None)
        if committed_segment is not None:
            try:
                DurableMarketBatchSpool(
                    Path(str(claimed.payload["segment_path"])).parent.parent
                ).delete(Path(str(committed_segment)))
            except (OSError, ValueError):
                # The queue is already complete and the segment is still
                # recoverable on the next writer cycle.
                pass
        return {"reason_code": "market_event_batch_written", "job_id": claimed.job_id, **result}

    def _run_single_event_write(self, claimed, *, now: str) -> dict[str, Any]:
        try:
            raw_event = claimed.payload.get("event")
            if not isinstance(raw_event, dict):
                raise ValueError("market-event job has no event object")
            event = MarketEvent(**raw_event)
            contiguous = _has_contiguous_sequence(event)
            previous_sequence = self.sequence_tracker.last_sequence(event)
            sequence_status = self.sequence_tracker.observe(event, contiguous=contiguous)
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
            market_snapshot_ids = self._publish_market_snapshots(event=event, market=market)
            feature_job_id = self._enqueue_closed_candle_features(
                event=event,
                venue=venue,
                market=market,
                symbol=symbol,
            )
            gap_recovery_job_id = None
            if sequence_status in {"gap", "out_of_order"}:
                gap_recovery_job_id = self._enqueue_gap_recovery(
                    event=event,
                    venue=venue,
                    market=market,
                    symbol=symbol,
                    previous_sequence=previous_sequence,
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
            "sequence_status": sequence_status,
            "gap_recovery_job_id": gap_recovery_job_id,
            "market_snapshot_ids": list(market_snapshot_ids),
        }

    def _cleanup_completed_spool_segments(self) -> None:
        spool = DurableMarketBatchSpool(self.store.root.parent / "runtime" / "events" / "spool")
        for path in spool.recoverable():
            if path.parent.name != "committed":
                continue
            job_id = f"work:market_event_batch:{path.stem.removeprefix('segment-')}"
            if self.queue.state(job_id) == "completed":
                spool.delete(path)

    def _write_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        segment_path = Path(str(payload["segment_path"]))
        spool = DurableMarketBatchSpool(segment_path.parent.parent)
        segment_path = spool.claim(segment_path)
        rows = DurableMarketBatchSpool.read(segment_path)
        body = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        )
        if canonical_hash(body) != str(payload["segment_hash"]):
            raise ValueError("market batch segment content hash is invalid")
        events = []
        gap_recovery_job_ids: list[str] = []
        for row in rows:
            event = row.get("event")
            if not isinstance(event, dict):
                raise ValueError("market batch row has no event")
            canonical_event = MarketEvent(**event)
            partition_values = (
                str(row.get("venue") or "").lower(),
                str(row.get("market") or "").lower(),
                canonical_event.event_type.value,
                str(row.get("symbol") or "").upper(),
            )
            if any(not _PARTITION_TOKEN.fullmatch(value) for value in partition_values):
                raise ValueError("market batch row contains an unsafe partition token")
            expected_prefix = f"{partition_values[0]}:{partition_values[1]}:{partition_values[3]}"
            if not canonical_event.instrument_id.startswith(expected_prefix):
                raise ValueError("market batch row partition does not match its instrument")
            events.append(
                (
                    canonical_event,
                    str(row["venue"]),
                    str(row["market"]),
                    str(row["symbol"]),
                )
            )
            previous_sequence = self.sequence_tracker.last_sequence(canonical_event)
            sequence_status = self.sequence_tracker.observe(
                canonical_event,
                contiguous=_has_contiguous_sequence(canonical_event),
            )
            if sequence_status in {"gap", "out_of_order"}:
                gap_recovery_job_ids.append(
                    self._enqueue_gap_recovery(
                        event=canonical_event,
                        venue=str(row["venue"]),
                        market=str(row["market"]),
                        symbol=str(row["symbol"]),
                        previous_sequence=previous_sequence,
                    )
                )
        paths = self.store.put_batch(events) if events else ()
        closed_bars = [
            (event, venue, market, symbol)
            for event, venue, market, symbol in events
            if _is_closed_candle(event)
        ]
        if closed_bars:
            self.bar_store.put_batch(closed_bars)
            for event, venue, market, symbol in closed_bars:
                self._enqueue_closed_candle_features(
                    event=event, venue=venue, market=market, symbol=symbol
                )
        snapshot_ids = [
            snapshot_id
            for event, _venue, market, _symbol in events
            for snapshot_id in self._publish_market_snapshots(event=event, market=market)
        ]
        committed = spool.commit(segment_path)
        return {
            "segments": len(paths),
            "events": len(events),
            "paths": [str(path) for path in paths],
            "gap_recovery_job_ids": gap_recovery_job_ids,
            "market_snapshot_ids": snapshot_ids,
            "committed_segment": str(committed),
        }

    def _maybe_compact(self, now: str) -> dict[str, Any] | None:
        if self._storage_consumers_busy():
            return None
        observed = dt.datetime.fromisoformat(now)
        if (
            self._last_compaction_at is not None
            and (observed - self._last_compaction_at).total_seconds()
            < self.compaction_interval_seconds
        ):
            return None
        result = compact_data_root(
            self.store.root,
            maximum_rows=self.compaction_maximum_rows,
        )
        self._last_compaction_at = observed
        return {"reason_code": "storage_compacted", **result}

    def _storage_consumers_busy(self) -> bool:
        with self.queue.engine.connect() as connection:
            rows = connection.execute(
                select(job.c.state).where(
                    job.c.name.in_(
                        (
                            "market_event_batch_write",
                            "market_event_write",
                            "live_feature_calculation",
                            "historical_feature_calculation",
                        )
                    ),
                    job.c.state.in_(("pending", "running")),
                )
            )
            return next(iter(rows), None) is not None

    def _run_historical_bootstrap(self, claimed, *, now: str) -> dict[str, Any]:
        current = claimed

        def heartbeat():
            nonlocal current
            current = self.queue.heartbeat(
                current,
                now=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
                lease_seconds=self.lease_seconds,
            )
            return current

        try:
            if self.historical_bootstrap_service is None:
                raise RuntimeError("historical bootstrap service is not configured")
            reports = self.historical_bootstrap_service.bootstrap_product(
                product_id=str(claimed.payload["product_id"]),
                market=str(claimed.payload["market_type"]),
                symbols=tuple(str(symbol) for symbol in claimed.payload["symbols"]),
                minimum_history_days=int(claimed.payload["minimum_history_days"]),
                timeframe=str(claimed.payload["timeframe"]),
                data_plan=claimed.payload.get("data_plan"),
                now=str(claimed.payload.get("history_as_of") or now),
                heartbeat=heartbeat,
            )
            result = {
                "reason_code": "historical_bootstrap_completed",
                "job_id": claimed.job_id,
                "reports": [report.to_payload() for report in reports],
            }
            if not all(report.completed and report.metadata_fetched for report in reports):
                self.queue.defer(
                    current,
                    available_at=_retry_at(now, min(self.lease_seconds, 300)),
                    reason="historical_bootstrap_incomplete",
                )
                return {
                    **result,
                    "reason_code": "historical_bootstrap_incomplete",
                }
        except Exception as exc:
            retryable = isinstance(
                exc, HistoricalBootstrapError | requests.RequestException | TimeoutError
            )
            if retryable:
                self.queue.defer(
                    current,
                    available_at=_retry_at(now, min(self.lease_seconds, 300)),
                    reason="historical_bootstrap_waiting_for_retry",
                )
            else:
                self.queue.fail(
                    current,
                    completed_at=now,
                    error=f"{type(exc).__name__}: {exc}",
                    retry_at=_retry_at(now, self.lease_seconds),
                )
            return {
                "reason_code": "historical_bootstrap_deferred"
                if retryable
                else "historical_bootstrap_failed",
                "job_id": current.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(current, completed_at=now)
        return result

    def _enqueue_gap_recovery(
        self,
        *,
        event: MarketEvent,
        venue: str,
        market: str,
        symbol: str,
        previous_sequence: int | None,
    ) -> str:
        payload = {
            "venue": venue,
            "market": market,
            "symbol": symbol,
            "instrument_id": event.instrument_id,
            "event_type": event.event_type.value,
            "previous_sequence": previous_sequence,
            "sequence": event.sequence,
            "event": event.payload,
            "observed_at": event.receive_timestamp,
        }
        identity = canonical_hash(payload).removeprefix("sha256:")
        job_id = f"market-gap:{identity}"
        self.queue.enqueue_if_absent(
            job_id=job_id,
            name="market_data_gap_recovery",
            payload=payload,
            available_at=event.receive_timestamp,
            priority=90,
            producer_identity="market-data-writer",
        )
        return job_id

    def _run_gap_recovery(self, claimed, *, now: str) -> dict[str, Any]:
        try:
            repaired = tuple(self.gap_repair(claimed.payload))
            if not repaired:
                raise RuntimeError("REST gap repair returned no events")
            venue = str(claimed.payload["venue"])
            market = str(claimed.payload["market"])
            symbol = str(claimed.payload["symbol"])
            for event in repaired:
                if event.instrument_id != claimed.payload["instrument_id"]:
                    raise ValueError("REST gap repair changed the instrument identity")
            repaired = tuple({event.event_id: event for event in repaired}.values())
            raw_events = tuple((event, venue, market, symbol) for event in repaired)
            raw_paths = self.store.put_batch(raw_events)
            paths = [str(path) for path in raw_paths]
            closed_bars = []
            for event in repaired:
                self.sequence_tracker.observe(
                    event,
                    contiguous=_has_contiguous_sequence(event),
                )
                if _is_closed_candle(event):
                    closed_bars.append((event, venue, market, symbol))
                self._publish_market_snapshots(
                    event=event,
                    market=str(claimed.payload["market"]),
                )
                self._enqueue_closed_candle_features(
                    event=event,
                    venue=venue,
                    market=market,
                    symbol=symbol,
                )
            bar_paths = self.bar_store.put_batch(closed_bars) if closed_bars else ()
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "market_gap_recovery_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "market_gap_recovered",
            "job_id": claimed.job_id,
            "repaired_events": len(repaired),
            "paths": paths,
            "bar_paths": [str(path) for path in bar_paths],
        }

    def _publish_market_snapshots(self, *, event: MarketEvent, market: str) -> tuple[str, ...]:
        if self.snapshot_store is None:
            return ()
        values = _market_snapshot_values(event)
        if values is None:
            return ()
        identities = tuple(
            self.snapshot_store.save(
                {
                    "kind": "market_data_input",
                    "product_id": product_id,
                    "instrument_id": event.instrument_id,
                    "source_event_id": event.event_id,
                    "source_event_type": event.event_type.value,
                    "source_timeframe": event.payload.get("data", {}).get("k", {}).get("i"),
                    "source_open_time_ms": event.payload.get("data", {}).get("k", {}).get("t"),
                    "source_close_time_ms": event.payload.get("data", {}).get("k", {}).get("T"),
                    "source_candle_closed": event.payload.get("data", {}).get("k", {}).get("x"),
                    "source_event_time": event.exchange_timestamp,
                    "availability_time": event.availability_timestamp,
                    "values": values,
                },
                created_at=event.availability_timestamp,
            )
            for product_id in self.product_ids_by_market.get(market, ())
        )

        if _is_closed_candle(event):
            from src.services.paper_protection import has_paper_protection

            for product_id, identity in zip(
                self.product_ids_by_market.get(market, ()), identities, strict=True
            ):
                if has_paper_protection(self.queue.engine, event.instrument_id, product_id):
                    self.queue.enqueue_if_absent(
                        job_id=f"paper-protection:{identity}",
                        name="paper_protection_evaluate",
                        payload={"market_snapshot_id": identity},
                        available_at=event.availability_timestamp,
                        priority=100,
                    )
        return identities

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
        bar_reference = {
            "kind": "partitioned_bar_window",
            "relative_pattern": (
                f"bars/{venue.lower()}/{market.lower()}/{symbol.upper()}/"
                f"{str(candle['i']).lower()}/**/*.parquet"
            ),
            "through_close_time": source_close_time,
            "minimum_history": 1,
            "source_event_ids": [event.event_id],
        }
        references: dict[str, Any] = {
            "bar_window": {
                **bar_reference,
                "content_hash": canonical_hash(bar_reference),
            }
        }
        auxiliary_specs = {
            "higher_timeframe": ("higher_timeframe_bars", ("candle",)),
            "order_book": ("order_book_snapshot", ("best_bid_ask", "depth_update")),
            "trade_flow": ("trade_flow_snapshot", ("trade", "aggregate_trade")),
            "funding_open_interest": (
                "funding_open_interest_snapshot",
                ("funding_rate", "open_interest", "mark_price"),
            ),
            "spot_perpetual": ("spot_perpetual_snapshot", ("candle",)),
            "cross_sectional": ("cross_sectional_snapshot", ("candle",)),
            "correlation_beta": ("correlation_beta_snapshot", ("candle",)),
            "linked_instruments": ("linked_instrument_snapshot", ("candle",)),
            "sentiment": ("sentiment_snapshot", ()),
            "ml_manifest": ("frozen_ml_manifest", ()),
            "liquidation": ("liquidation_snapshot", ("liquidation",)),
        }
        for name, (kind, event_types) in auxiliary_specs.items():
            pattern = (
                f"raw/{venue.lower()}/{market.lower()}/candle/**/*.parquet"
                if name in {"cross_sectional", "correlation_beta", "linked_instruments"}
                else f"raw/{venue.lower()}/**/candle/{symbol.upper()}/**/*.parquet"
                if name == "spot_perpetual"
                else f"raw/{venue.lower()}/{market.lower()}/**/{symbol.upper()}/**/*.parquet"
            )
            reference = {
                "kind": kind,
                "relative_pattern": pattern,
                "event_types": list(event_types),
                "through_close_time": source_close_time,
                "availability_time": event.availability_timestamp,
                "source_event_ids": [],
            }
            references[name] = {
                **reference,
                "content_hash": canonical_hash(reference),
            }
        payload = {
            "instrument_id": event.instrument_id,
            "feature_set_version": "core-bars-v1",
            "source_event_time": source_event_time,
            "source_close_time": source_close_time,
            "availability_time": event.availability_timestamp,
            "input_references": references,
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


def _has_contiguous_sequence(event: MarketEvent) -> bool:
    """Only Binance depth updates expose a contiguous update-id range."""

    if event.event_type is not MarketEventType.DEPTH_UPDATE:
        return False
    raw_data = event.payload.get("data")
    if not isinstance(raw_data, Mapping):
        return False
    return all(
        isinstance(raw_data.get(field), int) and not isinstance(raw_data.get(field), bool)
        for field in ("U", "u")
    )


def _market_snapshot_values(event: MarketEvent) -> dict[str, float] | None:
    raw_data = event.payload.get("data")
    if not isinstance(raw_data, Mapping):
        return None
    if event.event_type is MarketEventType.CANDLE:
        values = _candle_snapshot_values(raw_data)
    elif event.event_type in {MarketEventType.BEST_BID_ASK, MarketEventType.DEPTH_UPDATE}:
        values = _book_snapshot_values(event, raw_data)
    elif event.event_type is MarketEventType.MARK_PRICE:
        values = _mark_snapshot_values(raw_data)
    elif event.event_type is MarketEventType.FUNDING_RATE:
        values = _funding_snapshot_values(raw_data)
    elif event.event_type in {MarketEventType.TRADE, MarketEventType.AGGREGATE_TRADE}:
        values = _trade_snapshot_values(raw_data)
    else:
        return None
    if values is None:
        return None
    if "close" in values and values["close"] <= 0:
        return None
    if values.get("visible_depth", 0.0) < 0:
        return None
    return values


def _market_number(raw_data: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = raw_data.get(name)
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _candle_snapshot_values(raw_data: Mapping[str, Any]) -> dict[str, float] | None:
    candle = raw_data.get("k")
    if not isinstance(candle, Mapping) or candle.get("x") is not True:
        return None
    close = candle.get("c")
    if close is None or isinstance(close, bool):
        return None
    try:
        parsed = float(close)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    values = {"close": parsed}
    for name, source in (("open", "o"), ("high", "h"), ("low", "l"), ("volume", "v")):
        value = _market_number(candle, source)
        if value is not None:
            values[name] = value
    return values


def _book_snapshot_values(
    event: MarketEvent, raw_data: Mapping[str, Any]
) -> dict[str, float] | None:
    bid = _market_number(raw_data, "bid_price", "b")
    ask = _market_number(raw_data, "ask_price", "a")
    bid_depth = _market_number(raw_data, "bid_depth", "B")
    ask_depth = _market_number(raw_data, "ask_depth", "A")
    if event.event_type is MarketEventType.DEPTH_UPDATE:
        bid_level = raw_data.get("b", raw_data.get("bids"))
        ask_level = raw_data.get("a", raw_data.get("asks"))
        if isinstance(bid_level, list) and bid_level:
            bid, bid_depth = _book_level(bid_level[0])
        if isinstance(ask_level, list) and ask_level:
            ask, ask_depth = _book_level(ask_level[0])
    if bid is None or ask is None or bid <= 0 or ask < bid:
        return None
    midpoint = (bid + ask) / 2.0
    return {
        "close": midpoint,
        "spread_bps": (ask - bid) / midpoint * 10_000.0,
        "visible_depth": max(0.0, bid_depth or 0.0) + max(0.0, ask_depth or 0.0),
    }


def _mark_snapshot_values(raw_data: Mapping[str, Any]) -> dict[str, float] | None:
    price = _market_number(raw_data, "mark_price", "markPrice", "p")
    if price is None or price <= 0:
        return None
    values = {"close": price}
    funding = _market_number(raw_data, "funding_rate", "fundingRate", "r")
    if funding is not None:
        values["funding"] = funding
    return values


def _funding_snapshot_values(raw_data: Mapping[str, Any]) -> dict[str, float] | None:
    funding = _market_number(raw_data, "funding_rate", "fundingRate", "r")
    return {"funding": funding} if funding is not None else None


def _trade_snapshot_values(raw_data: Mapping[str, Any]) -> dict[str, float] | None:
    price = _market_number(raw_data, "price", "p")
    return {"close": price} if price is not None and price > 0 else None


def _book_level(value: object) -> tuple[float | None, float | None]:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return None, None
    try:
        price = float(value[0])
        quantity = float(value[1])
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(price) or not math.isfinite(quantity):
        return None, None
    return price, quantity
