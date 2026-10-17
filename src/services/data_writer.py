"""Leased atomic persistence for canonical market events."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import select

from src.data.database import PlatformDatabase, market_input_sources
from src.data.feature_graph import FeatureGraphRegistry, FeatureNodeType
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

_PARTITION_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")
_AUXILIARY_FAMILIES_BY_NODE_TYPE = {
    FeatureNodeType.MULTI_TIMEFRAME: {"higher_timeframe"},
    FeatureNodeType.CROSS_SECTIONAL_RANK: {"cross_sectional"},
    FeatureNodeType.FUNDING_OPEN_INTEREST: {"funding_open_interest"},
    FeatureNodeType.SPOT_PERPETUAL_BASIS: {"spot_perpetual"},
    FeatureNodeType.CORRELATION_BETA: {"correlation_beta", "linked_instruments"},
    FeatureNodeType.ORDER_BOOK: {"order_book"},
    FeatureNodeType.TRADE_FLOW: {"trade_flow"},
    FeatureNodeType.LIQUIDATION: {"liquidation"},
    FeatureNodeType.SENTIMENT: {"sentiment"},
    FeatureNodeType.FROZEN_ML_FEATURE_SET: {"ml_manifest"},
}


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
        historical_executor: Executor | None = None,
        clock: Callable[[], str] | None = None,
        lease_seconds: int = 60,
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
        self.historical_executor = historical_executor
        self.clock = clock or (lambda: dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat())
        self._history_future: Future | None = None
        self._stopping = threading.Event()

    def run_once(self, *, now: str) -> dict[str, Any]:
        if self.snapshot_store is not None:
            self.snapshot_store.expire_market_inputs(at=now)
        if self._history_future is not None and self._history_future.done():
            completed, self._history_future = self._history_future, None
            return completed.result()
        self._cleanup_completed_spool_segments()
        claimed = self._claim_history(now) or self.queue.claim(
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
                    and self._history_future is None
                    else ()
                ),
            ),
        )
        if claimed is None:
            return {"reason_code": "market_event_queue_empty"}
        if claimed.name == "historical_bootstrap":
            if self.historical_executor is not None:
                self._history_future = self.historical_executor.submit(
                    self._run_historical_bootstrap, claimed, now=now
                )
                return {"reason_code": "historical_bootstrap_started", "job_id": claimed.job_id}
            return self._run_historical_bootstrap(claimed, now=now)
        if claimed.name == "market_data_gap_recovery":
            return self._run_gap_recovery(claimed, now=now)
        if claimed.name == "market_event_batch_write":
            return self._run_batch_write(claimed, now=now)
        return self._run_single_event_write(claimed, now=now)

    def _claim_history(self, now: str):
        if (
            self.historical_bootstrap_service is None
            or self.historical_executor is None
            or self._history_future is not None
        ):
            return None
        # Reserve the independent download slot even while the live queue stays busy.
        return self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("historical_bootstrap",),
        )

    def __call__(self) -> dict[str, Any]:
        return self.run_once(now=self.clock())

    def stop(self) -> None:
        self._stopping.set()
        if self.historical_executor is not None:
            self.historical_executor.shutdown(wait=True, cancel_futures=True)

    def _run_batch_write(self, claimed, *, now: str) -> dict[str, Any]:
        try:
            result = self._write_batch(claimed.payload, now=now)
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
            market_snapshot_ids = self._publish_market_snapshots(
                event=event, market=market, retention_at=now
            )
            path = (
                self.store.put(event, venue=venue, market=market, symbol=symbol)
                if self._archive_event(event, market)
                else None
            )
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
            "path": str(path) if path is not None else None,
            "bar_path": str(bar_path) if bar_path is not None else None,
            "feature_job_id": feature_job_id,
            "sequence_status": sequence_status,
            "gap_recovery_job_id": gap_recovery_job_id,
            "market_snapshot_ids": list(market_snapshot_ids),
        }

    def _cleanup_completed_spool_segments(self) -> None:
        spool = DurableMarketBatchSpool(self.store.root.parent / "runtime" / "events" / "spool")
        for path in (spool.root / "committed").glob("segment-*.jsonl"):
            if not path.is_file() or path.is_symlink():
                continue
            job_id = f"work:market_event_batch:{path.stem.removeprefix('segment-')}"
            if self.queue.state(job_id) == "completed":
                spool.delete(path)

    def _write_batch(self, payload: dict[str, Any], *, now: str) -> dict[str, Any]:
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
        snapshot_ids = self._publish_batch_market_snapshots(events, now=now)
        archivable_events = tuple(
            (event, venue, market, symbol)
            for event, venue, market, symbol in events
            if self._archive_event(event, market)
        )
        paths = self.store.put_batch(archivable_events) if archivable_events else ()
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
        committed = spool.commit(segment_path)
        return {
            "segments": len(paths),
            "events": len(events),
            "paths": [str(path) for path in paths],
            "gap_recovery_job_ids": gap_recovery_job_ids,
            "market_snapshot_ids": snapshot_ids,
            "committed_segment": str(committed),
        }

    def _run_historical_bootstrap(self, claimed, *, now: str) -> dict[str, Any]:
        current = claimed

        def heartbeat():
            nonlocal current
            if self._stopping.is_set():
                raise HistoricalBootstrapError("history worker is stopping")
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
            for event in repaired:
                self.sequence_tracker.observe(
                    event,
                    contiguous=_has_contiguous_sequence(event),
                )
                self._publish_market_snapshots(
                    event=event,
                    market=str(claimed.payload["market"]),
                    retention_at=now,
                )
            raw_events = tuple(
                (event, venue, market, symbol)
                for event in repaired
                if self._archive_event(event, market)
            )
            raw_paths = self.store.put_batch(raw_events) if raw_events else ()
            paths = [str(path) for path in raw_paths]
            closed_bars = [
                (event, venue, market, symbol) for event in repaired if _is_closed_candle(event)
            ]
            bar_paths = self.bar_store.put_batch(closed_bars) if closed_bars else ()
            for event, _bar_venue, _bar_market, _bar_symbol in closed_bars:
                self._enqueue_closed_candle_features(
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

    def _publish_batch_market_snapshots(self, events, *, now: str) -> list[str]:
        buffered: list[tuple[dict[str, object], str]] = []
        immutable: list[str] = []
        for event, _venue, market, _symbol in events:
            if not _requires_market_input_buffer(event):
                immutable.extend(self._publish_market_snapshots(event=event, market=market))
                continue
            products = self.product_ids_by_market.get(market, ())
            if self.snapshot_store is None:
                if products:
                    raise RuntimeError("high-rate market input requires the recent snapshot buffer")
                continue
            if not products:
                raise RuntimeError(
                    "high-rate market input has no product scope for snapshot storage"
                )
            values = _market_snapshot_values(event) or {}
            buffered.extend(
                (_market_snapshot_payload(event, product_id, values), event.availability_timestamp)
                for product_id in products
            )
        if buffered:
            assert self.snapshot_store is not None
            immutable.extend(self.snapshot_store.save_market_inputs(buffered, retention_at=now))
        return immutable

    def _publish_market_snapshots(
        self, *, event: MarketEvent, market: str, retention_at: str | None = None
    ) -> tuple[str, ...]:
        store = self.snapshot_store
        product_ids = self.product_ids_by_market.get(market, ())
        if store is None:
            if product_ids and _requires_market_input_buffer(event):
                raise RuntimeError("high-rate market input requires the recent snapshot buffer")
            return ()
        if _requires_market_input_buffer(event) and not product_ids:
            raise RuntimeError("high-rate market input has no product scope for snapshot storage")
        if event.event_type in {MarketEventType.TRADE, MarketEventType.AGGREGATE_TRADE}:
            return ()
        values = _market_snapshot_values(event)
        if values is None and not _requires_market_input_buffer(event):
            return ()
        retained_at = retention_at or event.availability_timestamp
        values = values or {}

        def save(payload: dict[str, object], *, created_at: str) -> str:
            if not _requires_market_input_buffer(event):
                return store.save(payload, created_at=created_at)
            return store.save_market_input(payload, created_at=created_at, retention_at=retained_at)

        identities = tuple(
            save(
                _market_snapshot_payload(event, product_id, values),
                created_at=event.availability_timestamp,
            )
            for product_id in product_ids
        )

        if _is_closed_candle(event):
            self._enqueue_paper_protection_snapshots(event, product_ids, identities)
        return identities

    def _enqueue_paper_protection_snapshots(
        self, event: MarketEvent, product_ids: tuple[str, ...], identities: tuple[str, ...]
    ) -> None:
        from src.services.paper_protection import has_paper_protection

        for product_id, identity in zip(product_ids, identities, strict=True):
            if has_paper_protection(self.queue.engine, event.instrument_id, product_id):
                self.queue.enqueue_if_absent(
                    job_id=f"paper-protection:{identity}",
                    name="paper_protection_evaluate",
                    payload={"market_snapshot_id": identity},
                    available_at=event.availability_timestamp,
                    priority=100,
                )

    def _archive_event(self, event: MarketEvent, market: str) -> bool:
        if not _is_high_rate_event(event):
            return True
        return self.snapshot_store is None and not self.product_ids_by_market.get(market)

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
        required_families = self._active_feature_input_families(event=event, market=market)
        pinned_inputs = self._pin_feature_market_inputs(
            event=event,
            market=market,
            through_close_time=source_close_time,
            required_families=required_families,
        )
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
            reference = {
                "kind": kind,
                "event_types": list(event_types),
                "through_close_time": source_close_time,
                "availability_time": event.availability_timestamp,
                "source_event_ids": [],
            }
            if self.snapshot_store is not None and name in {
                "order_book",
                "funding_open_interest",
            }:
                reference["snapshot_source"] = "market_input"
                reference["source_snapshot_ids"] = pinned_inputs.get(name, [])
            elif self.snapshot_store is not None and name == "trade_flow":
                reference["snapshot_source"] = "disabled"
            else:
                reference["relative_pattern"] = self._legacy_auxiliary_pattern(
                    name=name, venue=venue, market=market, symbol=symbol
                )
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

    @staticmethod
    def _legacy_auxiliary_pattern(*, name: str, venue: str, market: str, symbol: str) -> str:
        if name in {"cross_sectional", "correlation_beta", "linked_instruments"}:
            return f"raw/{venue.lower()}/{market.lower()}/candle/**/*.parquet"
        if name == "spot_perpetual":
            return f"raw/{venue.lower()}/**/candle/{symbol.upper()}/**/*.parquet"
        return f"raw/{venue.lower()}/{market.lower()}/**/{symbol.upper()}/**/*.parquet"

    def _active_feature_input_families(self, *, event: MarketEvent, market: str) -> set[str]:
        product_ids = self.product_ids_by_market.get(market, ())
        if self.snapshot_store is None or not product_ids:
            return set()
        from src.research.canonical import (
            SqlActiveStrategyAssignmentRepository,
            SqlStrategyArtefactRepository,
        )

        assignments = SqlActiveStrategyAssignmentRepository(self.queue.engine)
        artefacts = SqlStrategyArtefactRepository(self.queue.engine)
        registry = FeatureGraphRegistry.default()
        families: set[str] = set()
        for product_id in product_ids:
            for assignment in assignments.active_assignments(
                product_id, at=event.availability_timestamp
            ):
                if assignment.get("instrument_id") not in {None, event.instrument_id}:
                    continue
                artefact = artefacts.get(str(assignment["artefact_hash"]))
                definition = artefact.get("definition")
                graph_spec = (
                    definition.get("feature_graph") if isinstance(definition, Mapping) else None
                )
                if not isinstance(graph_spec, Mapping):
                    continue
                required = graph_spec.get("required_nodes", graph_spec.get("nodes", ()))
                if not isinstance(required, list | tuple) or not required:
                    continue
                nodes = tuple(
                    str(item["name"] if isinstance(item, Mapping) else item) for item in required
                )
                graph = registry.graph(
                    nodes, version=str(graph_spec.get("version") or "canonical-features/v2")
                )
                for node in graph.nodes:
                    families.update(_AUXILIARY_FAMILIES_BY_NODE_TYPE.get(node.node_type, ()))
        return families

    def _pin_feature_market_inputs(
        self,
        *,
        event: MarketEvent,
        market: str,
        through_close_time: str,
        required_families: set[str],
    ) -> dict[str, list[str]]:
        if self.snapshot_store is None:
            return {}
        product_ids = self.product_ids_by_market.get(market, ())
        inputs = {
            "order_book": ("best_bid_ask", "depth_update"),
            "funding_open_interest": ("mark_price", "funding_rate"),
        }
        selected: dict[str, dict[str, str]] = {}
        for family, event_types in inputs.items():
            if family not in required_families:
                continue
            by_event: dict[str, str] = {}
            for product_id in product_ids:
                for event_type in event_types:
                    candidate = self._latest_feature_market_input(
                        event=event,
                        product_id=product_id,
                        event_type=event_type,
                        through_close_time=through_close_time,
                    )
                    if candidate is not None:
                        event_id, identity = candidate
                        by_event.setdefault(event_id, identity)
            selected[family] = by_event
        result: dict[str, list[str]] = {}
        for family, by_event in selected.items():
            identities = sorted(by_event.values())
            for identity in identities:
                self.snapshot_store.get(identity)
            result[family] = identities
        return result

    def _latest_feature_market_input(
        self,
        *,
        event: MarketEvent,
        product_id: str,
        event_type: str,
        through_close_time: str,
    ) -> tuple[str, str] | None:
        with self.queue.engine.connect() as connection:
            rows = connection.execute(
                select(market_input_sources.c.id, market_input_sources.c.payload)
                .where(
                    market_input_sources.c.payload["product_id"].as_string() == product_id,
                    market_input_sources.c.payload["instrument_id"].as_string()
                    == event.instrument_id,
                    market_input_sources.c.payload["source_event_type"].as_string() == event_type,
                    market_input_sources.c.payload["source_event_time"].as_string()
                    <= through_close_time,
                    market_input_sources.c.payload["availability_time"].as_string()
                    <= event.availability_timestamp,
                    market_input_sources.c.created_at <= event.availability_timestamp,
                )
                .order_by(
                    market_input_sources.c.payload["source_event_time"].as_string().desc(),
                    market_input_sources.c.payload["availability_time"].as_string().desc(),
                    market_input_sources.c.id.desc(),
                )
                .limit(100)
            ).mappings()
            for row in rows:
                payload = row["payload"]
                if not isinstance(payload, Mapping):
                    continue
                event_id = payload.get("source_event_id")
                if isinstance(payload.get("source_payload"), Mapping) and isinstance(event_id, str):
                    return event_id, str(row["id"])
        return None


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


def _requires_market_input_buffer(event: MarketEvent) -> bool:
    return event.event_type in {
        MarketEventType.BEST_BID_ASK,
        MarketEventType.DEPTH_UPDATE,
        MarketEventType.MARK_PRICE,
    } or (
        event.event_type is MarketEventType.FUNDING_RATE
        and bool(event.payload.get("source_event_id"))
    )


def _is_high_rate_event(event: MarketEvent) -> bool:
    return (
        _requires_market_input_buffer(event)
        or event.event_type
        in {
            MarketEventType.TRADE,
            MarketEventType.AGGREGATE_TRADE,
        }
        or (event.event_type is MarketEventType.CANDLE and not _is_closed_candle(event))
    )


def _market_snapshot_payload(
    event: MarketEvent, product_id: str, values: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
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
        "values": dict(values),
    }
    if _requires_market_input_buffer(event):
        payload.update(
            {
                "source_sequence": event.sequence,
                "source_receive_time": event.receive_timestamp,
                "source_close_time": event.close_timestamp,
                "source_payload": dict(event.payload),
            }
        )
    return payload


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
    if values.get("visible_depth_quote", 0.0) < 0:
        return None
    return values


def data_writer_cycle(
    *,
    database: PlatformDatabase,
    node_id: str,
    root: Path,
    configuration: Mapping[str, Mapping[str, Any]],
    clock: Callable[[], str] | None = None,
) -> DatabaseMarketDataWriter:
    """Build the dedicated writer without importing the all-service supervisor."""

    from src.services.runtime import utc_now

    worker_clock = clock or utc_now
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:data-writer"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="data-writer",
        capabilities=(
            "market_event_write",
            "market_event_batch_write",
            "market_data_gap_recovery",
            "historical_bootstrap",
        ),
        observed_at=worker_clock(),
    )
    products = {
        str(product["product_id"]): dict(product)
        for product in configuration["products"]["products"]
    }
    accounts = {
        str(account["account_id"]): dict(account)
        for account in configuration["accounts"]["accounts"]
    }
    products_by_market: dict[str, list[str]] = {}
    for product_id, product in products.items():
        account = accounts[str(product["account_id"])]
        market = {"spot": "spot", "usdt_futures": "futures"}.get(str(account["market"]))
        if market is None:
            raise ValueError(f"unsupported account market for {product_id}")
        products_by_market.setdefault(market, []).append(product_id)
    return DatabaseMarketDataWriter(
        queue=queue,
        worker_id=worker_id,
        root=root,
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_ids_by_market=products_by_market,
        historical_bootstrap_service=HistoricalBootstrapService(
            engine=database.engine,
            root=root,
        ),
        historical_executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="history"),
        clock=worker_clock,
        lease_seconds=300,
    )


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
    if bid_depth is None or ask_depth is None or min(bid_depth, ask_depth) < 0:
        return None
    midpoint = (bid + ask) / 2.0
    bid_quote = bid * bid_depth
    ask_quote = ask * ask_depth
    return {
        "close": midpoint,
        "spread_bps": (ask - bid) / midpoint * 10_000.0,
        "bid_depth_quote": bid_quote,
        "ask_depth_quote": ask_quote,
        "visible_depth_quote": min(bid_quote, ask_quote),
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
