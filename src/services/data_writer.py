"""Leased immutable Parquet writer for canonical market events."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from src.data.parquet_store import (
    DurableMarketBatchSpool,
    PartitionedBarStore,
    PartitionedMarketEventStore,
)
from src.domain._codec import canonical_hash
from src.domain.market_events import ExchangeSequenceTracker, MarketEvent, MarketEventType
from src.risk.engine import SqlRiskSnapshotStore
from src.services.market_gap_recovery import BinanceMarketGapRepair
from src.services.scheduler import DatabaseJobQueue

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

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=(
                "market_event_batch_write",
                "market_event_write",
                "market_data_gap_recovery",
            ),
        )
        if claimed is None:
            return {"reason_code": "market_event_queue_empty"}
        if claimed.name == "market_data_gap_recovery":
            return self._run_gap_recovery(claimed, now=now)
        try:
            if claimed.name == "market_event_batch_write":
                result = self._write_batch(claimed.payload)
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": "market_event_batch_written",
                    "job_id": claimed.job_id,
                    **result,
                }
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

    def _write_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        segment_path = Path(str(payload["segment_path"]))
        rows = DurableMarketBatchSpool.read(segment_path)
        body = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        )
        if canonical_hash(body) != str(payload["segment_hash"]):
            raise ValueError("market batch segment content hash is invalid")
        events = []
        gap_recovery_job_ids: list[str] = []
        dropped_rows = 0
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
                dropped_rows += 1
                continue
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
        for event, venue, market, symbol in events:
            if _is_closed_candle(event):
                self.bar_store.put(event, venue=venue, market=market, symbol=symbol)
                self._enqueue_closed_candle_features(
                    event=event, venue=venue, market=market, symbol=symbol
                )
        snapshot_ids = [
            snapshot_id
            for event, _venue, market, _symbol in events
            for snapshot_id in self._publish_market_snapshots(event=event, market=market)
        ]
        return {
            "segments": len(paths),
            "events": len(events),
            "dropped_rows": dropped_rows,
            "paths": [str(path) for path in paths],
            "gap_recovery_job_ids": gap_recovery_job_ids,
            "market_snapshot_ids": snapshot_ids,
        }

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
            paths: list[str] = []
            for event in repaired:
                if event.instrument_id != claimed.payload["instrument_id"]:
                    raise ValueError("REST gap repair changed the instrument identity")
                self.sequence_tracker.observe(
                    event,
                    contiguous=_has_contiguous_sequence(event),
                )
                path = self.store.put(
                    event,
                    venue=str(claimed.payload["venue"]),
                    market=str(claimed.payload["market"]),
                    symbol=str(claimed.payload["symbol"]),
                )
                paths.append(str(path))
                if _is_closed_candle(event):
                    self.bar_store.put(
                        event,
                        venue=str(claimed.payload["venue"]),
                        market=str(claimed.payload["market"]),
                        symbol=str(claimed.payload["symbol"]),
                    )
                self._publish_market_snapshots(
                    event=event,
                    market=str(claimed.payload["market"]),
                )
                self._enqueue_closed_candle_features(
                    event=event,
                    venue=str(claimed.payload["venue"]),
                    market=str(claimed.payload["market"]),
                    symbol=str(claimed.payload["symbol"]),
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
        }

    def _publish_market_snapshots(self, *, event: MarketEvent, market: str) -> tuple[str, ...]:
        if self.snapshot_store is None:
            return ()
        values = _market_snapshot_values(event)
        if values is None:
            return ()
        return tuple(
            self.snapshot_store.save(
                {
                    "kind": "market_data_input",
                    "product_id": product_id,
                    "instrument_id": event.instrument_id,
                    "source_event_id": event.event_id,
                    "source_event_time": event.exchange_timestamp,
                    "availability_time": event.availability_timestamp,
                    "values": values,
                },
                created_at=event.availability_timestamp,
            )
            for product_id in self.product_ids_by_market.get(market, ())
        )

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
            "sentiment": ("sentiment_snapshot", ()),
            "ml_manifest": ("frozen_ml_manifest", ()),
            "liquidation": ("liquidation_snapshot", ("liquidation",)),
        }
        for name, (kind, event_types) in auxiliary_specs.items():
            pattern = (
                f"raw/{venue.lower()}/{market.lower()}/candle/**/*.parquet"
                if name in {"cross_sectional", "correlation_beta"}
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
    values: dict[str, float] = {}

    def number(*names: str) -> float | None:
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

    candle = raw_data.get("k")
    if event.event_type is MarketEventType.CANDLE:
        if not isinstance(candle, Mapping) or candle.get("x") is not True:
            return None
        close = candle.get("c")
        if close is None or isinstance(close, bool):
            return None
        try:
            values["close"] = float(close)
        except (TypeError, ValueError):
            return None
    elif event.event_type in {MarketEventType.BEST_BID_ASK, MarketEventType.DEPTH_UPDATE}:
        bid = number("bid_price", "b")
        ask = number("ask_price", "a")
        bid_depth = number("bid_depth", "B")
        ask_depth = number("ask_depth", "A")
        if event.event_type is MarketEventType.DEPTH_UPDATE:
            bid_levels = raw_data.get("b", raw_data.get("bids"))
            ask_levels = raw_data.get("a", raw_data.get("asks"))
            if isinstance(bid_levels, list) and bid_levels:
                bid, bid_depth = _book_level(bid_levels[0])
            if isinstance(ask_levels, list) and ask_levels:
                ask, ask_depth = _book_level(ask_levels[0])
        if bid is None or ask is None or bid <= 0 or ask < bid:
            return None
        midpoint = (bid + ask) / 2.0
        values.update(
            {
                "close": midpoint,
                "spread_bps": (ask - bid) / midpoint * 10_000.0,
                "visible_depth": max(0.0, bid_depth or 0.0) + max(0.0, ask_depth or 0.0),
            }
        )
    elif event.event_type is MarketEventType.MARK_PRICE:
        price = number("mark_price", "markPrice", "p")
        if price is None or price <= 0:
            return None
        values["close"] = price
        funding = number("funding_rate", "fundingRate", "r")
        if funding is not None:
            values["funding"] = funding
    elif event.event_type is MarketEventType.FUNDING_RATE:
        funding = number("funding_rate", "fundingRate", "r")
        if funding is None:
            return None
        values["funding"] = funding
    elif event.event_type in {MarketEventType.TRADE, MarketEventType.AGGREGATE_TRADE}:
        price = number("price", "p")
        if price is None or price <= 0:
            return None
        values["close"] = price
    else:
        return None

    if "close" in values and values["close"] <= 0:
        return None
    if values.get("visible_depth", 0.0) < 0:
        return None
    return values


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
