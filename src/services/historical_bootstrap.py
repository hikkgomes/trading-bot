"""Resumable canonical Binance history bootstrap for a clean platform install."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from sqlalchemy import insert, select, update

from src.data.database import (
    funding_entry,
    historical_bootstrap_progress,
    instrument,
    instrument_status,
)
from src.data.parquet_store import _fsync_directory
from src.domain._codec import canonical_hash, json_value, timestamp
from src.domain.instruments import Instrument, MarketType
from src.execution.rate_limit import ExchangeRateLimiter, shared_exchange_rate_limiter

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}
ENDPOINTS = {
    "spot": ("https://api.binance.com", "/api/v3/klines"),
    "futures": ("https://fapi.binance.com", "/fapi/v1/klines"),
}
PAGE_SIZE = 1_000
FUNDING_PAGE_SIZE = 1_000


class HistoricalBootstrapError(RuntimeError):
    """History cannot be published as a complete canonical dataset."""


@dataclass(frozen=True)
class HistoricalBootstrapReport:
    product_id: str
    market: str
    symbols: tuple[str, ...]
    timeframe: str
    range_start: str
    range_end: str
    rows: int
    partitions: tuple[str, ...]
    gaps: tuple[Mapping[str, Any], ...]
    resumed: bool
    completed: bool
    funding_rows: int = 0
    funding_partitions: tuple[str, ...] = ()
    metadata_fetched: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "platform.historical_bootstrap/v1",
            "product_id": self.product_id,
            "market": self.market,
            "symbols": list(self.symbols),
            "timeframe": self.timeframe,
            "range_start": self.range_start,
            "range_end": self.range_end,
            "rows": self.rows,
            "partitions": list(self.partitions),
            "gaps": [dict(gap) for gap in self.gaps],
            "resumed": self.resumed,
            "completed": self.completed,
            "funding_rows": self.funding_rows,
            "funding_partitions": list(self.funding_partitions),
            "metadata_fetched": self.metadata_fetched,
        }


class HistoricalBootstrapService:
    """Download public native klines directly into compact canonical partitions."""

    def __init__(
        self,
        *,
        engine,
        root: Path,
        session: requests.Session | None = None,
        rate_limiter: ExchangeRateLimiter | None = None,
        timeout_seconds: float = 15.0,
        page_size: int = PAGE_SIZE,
    ) -> None:
        if timeout_seconds <= 0 or not 1 <= page_size <= PAGE_SIZE:
            raise ValueError("historical bootstrap bounds are invalid")
        self.engine = engine
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.quarantine_root = self.root / "quarantine" / "historical"
        self.session = session or requests.Session()
        self.rate_limiter = rate_limiter or shared_exchange_rate_limiter("binance:history")
        self.timeout_seconds = timeout_seconds
        self.page_size = page_size

    def bootstrap_product(
        self,
        *,
        product_id: str,
        market: str,
        symbols: Iterable[str],
        minimum_history_days: int = 365,
        timeframe: str = "1m",
        now: str | None = None,
        maximum_pages: int = 10_000,
        heartbeat: Callable[[], Any] | None = None,
    ) -> tuple[HistoricalBootstrapReport, ...]:
        if minimum_history_days <= 0 or maximum_pages <= 0:
            raise ValueError("history days and maximum pages must be positive")
        market = str(market).casefold()
        if market not in ENDPOINTS:
            raise ValueError("history bootstrap market is invalid")
        normalised_symbols = tuple(
            sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
        )
        if not normalised_symbols:
            raise ValueError("history bootstrap requires at least one symbol")
        current = dt.datetime.fromisoformat(timestamp(now or dt.datetime.now(dt.UTC), field="now"))
        interval = self._interval(timeframe)
        end = int(current.timestamp() * 1_000) // interval * interval
        start = end - minimum_history_days * 86_400_000 - interval
        metadata = self._fetch_metadata(market=market, symbols=normalised_symbols)
        reports: list[HistoricalBootstrapReport] = []
        for symbol in normalised_symbols:
            report = self.bootstrap(
                product_id=product_id,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                range_start_ms=start,
                range_end_ms=end,
                maximum_pages=maximum_pages,
                heartbeat=heartbeat,
            )
            funding = (
                self.bootstrap_funding(
                    product_id=product_id,
                    symbol=symbol,
                    range_start_ms=start,
                    range_end_ms=end,
                    maximum_pages=maximum_pages,
                    heartbeat=heartbeat,
                )
                if market == "futures"
                else None
            )
            reports.append(
                HistoricalBootstrapReport(
                    **{
                        **report.__dict__,
                        "funding_rows": funding.rows if funding is not None else 0,
                        "funding_partitions": funding.partitions if funding is not None else (),
                        "metadata_fetched": symbol in metadata,
                    }
                )
            )
        return tuple(reports)

    def _fetch_metadata(self, *, market: str, symbols: tuple[str, ...]) -> set[str]:
        base, _ = ENDPOINTS[market]
        metadata_path = "/api/v3/exchangeInfo" if market == "spot" else "/fapi/v1/exchangeInfo"
        self.rate_limiter.acquire()
        response = self.session.get(
            f"{base}{metadata_path}",
            params={"symbol": symbols[0]} if market == "spot" and len(symbols) == 1 else None,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        raw_symbols = payload.get("symbols") if isinstance(payload, Mapping) else None
        if not isinstance(raw_symbols, list):
            return set()
        matched: set[str] = set()
        for item in raw_symbols:
            if not isinstance(item, Mapping):
                continue
            symbol = str(item.get("symbol") or "").upper()
            if symbol not in symbols:
                continue
            self._publish_metadata(market=market, symbol=symbol, payload=dict(item))
            matched.add(symbol)
        return matched

    def _publish_metadata(self, *, market: str, symbol: str, payload: Mapping[str, Any]) -> None:
        instrument_payload = _instrument_payload(market=market, symbol=symbol, payload=payload)
        metadata_hash = canonical_hash(dict(payload))
        clean = {
            "schema": "platform.binance_instrument_metadata/v1",
            "market": market,
            "symbol": symbol,
            "metadata_hash": metadata_hash,
            "metadata": dict(payload),
        }
        digest = canonical_hash(clean).removeprefix("sha256:")
        directory = self.root / "metadata" / "binance" / market / symbol
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"metadata-{digest}.json"
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.tmp")
            try:
                temporary.write_text(
                    json.dumps(clean, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                _fsync_directory(directory)
            finally:
                temporary.unlink(missing_ok=True)
        instrument_id = (
            f"binance:spot:{symbol}" if market == "spot" else f"binance:futures:{symbol}:USDT"
        )
        observed_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        with self.engine.begin() as connection:
            existing = (
                connection.execute(select(instrument).where(instrument.c.id == instrument_id))
                .mappings()
                .first()
            )
            if existing is None:
                connection.execute(
                    insert(instrument).values(
                        id=instrument_id,
                        venue=instrument_payload["venue"],
                        market_type=instrument_payload["market_type"],
                        exchange_symbol=instrument_payload["exchange_symbol"],
                        base_asset=instrument_payload["base_asset"],
                        quote_asset=instrument_payload["quote_asset"],
                        settlement_asset=instrument_payload["settlement_asset"],
                        payload=json_value(instrument_payload, field="instrument metadata"),
                    )
                )
            else:
                for field in (
                    "venue",
                    "market_type",
                    "exchange_symbol",
                    "base_asset",
                    "quote_asset",
                    "settlement_asset",
                ):
                    if existing[field] != instrument_payload[field]:
                        raise HistoricalBootstrapError(
                            f"Binance metadata changed instrument identity: {instrument_id}"
                        )
                if dict(existing["payload"]) != instrument_payload:
                    connection.execute(
                        instrument.update()
                        .where(instrument.c.id == instrument_id)
                        .values(payload=json_value(instrument_payload, field="instrument metadata"))
                    )
            status = str(payload.get("status") or "TRADING")
            identity = canonical_hash(
                {
                    "instrument_id": instrument_id,
                    "metadata_hash": metadata_hash,
                    "status": status,
                }
            )
            if (
                connection.execute(
                    select(instrument_status.c.id).where(instrument_status.c.id == identity)
                ).first()
                is None
            ):
                connection.execute(
                    insert(instrument_status).values(
                        id=identity,
                        instrument_id=instrument_id,
                        observed_at=observed_at,
                        status=status,
                        payload=json_value(clean, field="instrument metadata"),
                    )
                )

    def bootstrap(
        self,
        *,
        product_id: str,
        market: str,
        symbol: str,
        timeframe: str,
        range_start_ms: int,
        range_end_ms: int,
        maximum_pages: int = 10_000,
        heartbeat: Callable[[], Any] | None = None,
    ) -> HistoricalBootstrapReport:
        market = str(market).casefold()
        symbol = str(symbol).upper().strip()
        timeframe = str(timeframe).lower().strip()
        if market not in ENDPOINTS or not symbol.isalnum() or timeframe not in INTERVAL_MS:
            raise ValueError("history bootstrap market, symbol, or timeframe is invalid")
        if range_start_ms >= range_end_ms or maximum_pages <= 0:
            raise ValueError("history bootstrap range is invalid")
        range_start = _millis_timestamp(range_start_ms)
        range_end = _millis_timestamp(range_end_ms)
        progress_id = canonical_hash(
            {
                "product_id": product_id,
                "timeframe": timeframe,
                "market": market,
                "symbol": symbol,
                "range_start": range_start,
                "range_end": range_end,
            }
        )
        progress = self._progress(progress_id)
        cursor = range_start_ms if progress is None else _resume_cursor(progress, range_start_ms)
        resumed = progress is not None
        already_completed = (
            progress is not None
            and progress["status"] == "completed"
            and not _progress_gaps(progress)
            and cursor >= range_end_ms
        )
        if not already_completed:
            self._save_progress(
                progress_id=progress_id,
                product_id=product_id,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                range_start=range_start,
                range_end=range_end,
                next_start=_millis_timestamp(cursor),
                status="running",
                payload={"schema": "platform.historical_bootstrap_progress/v1", "gaps": []},
            )
        if already_completed:
            return HistoricalBootstrapReport(
                product_id=product_id,
                market=market,
                symbols=(symbol,),
                timeframe=timeframe,
                range_start=range_start,
                range_end=range_end,
                rows=0,
                partitions=(),
                gaps=(),
                resumed=True,
                completed=True,
            )
        cursor, rows_downloaded, partitions, gaps = self._download_range(
            progress_id=progress_id,
            product_id=product_id,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            range_start=range_start,
            range_end=range_end,
            cursor=cursor,
            range_end_ms=range_end_ms,
            maximum_pages=maximum_pages,
            heartbeat=heartbeat,
        )
        completed = cursor >= range_end_ms and not gaps
        if not completed:
            self._save_failed(
                progress_id,
                "historical_gap_detected" if gaps else "historical_range_incomplete",
                payload={"gaps": gaps},
            )
        return HistoricalBootstrapReport(
            product_id=product_id,
            market=market,
            symbols=(symbol,),
            timeframe=timeframe,
            range_start=range_start,
            range_end=range_end,
            rows=rows_downloaded,
            partitions=tuple(sorted(partitions)),
            gaps=tuple(gaps),
            resumed=resumed,
            completed=completed,
        )

    def bootstrap_funding(
        self,
        *,
        product_id: str,
        symbol: str,
        range_start_ms: int,
        range_end_ms: int,
        maximum_pages: int = 10_000,
        heartbeat: Callable[[], Any] | None = None,
    ) -> HistoricalBootstrapReport:
        """Download signed Binance funding events at their native timestamps."""

        symbol = str(symbol).upper().strip()
        if not symbol.isalnum() or range_start_ms >= range_end_ms or maximum_pages <= 0:
            raise ValueError("funding bootstrap range is invalid")
        range_start = _millis_timestamp(range_start_ms)
        range_end = _millis_timestamp(range_end_ms)
        progress_id = canonical_hash(
            {
                "product_id": product_id,
                "market": "futures",
                "symbol": symbol,
                "timeframe": "funding",
                "range_start": range_start,
                "range_end": range_end,
            }
        )
        progress = self._progress(progress_id)
        cursor = range_start_ms if progress is None else _resume_cursor(progress, range_start_ms)
        resumed = progress is not None
        if (
            progress is not None
            and progress["status"] == "completed"
            and not _progress_gaps(progress)
            and cursor >= range_end_ms
        ):
            return HistoricalBootstrapReport(
                product_id=product_id,
                market="futures",
                symbols=(symbol,),
                timeframe="funding",
                range_start=range_start,
                range_end=range_end,
                rows=0,
                partitions=(),
                gaps=(),
                resumed=True,
                completed=True,
            )
        self._save_progress(
            progress_id=progress_id,
            product_id=product_id,
            market="futures",
            symbol=symbol,
            timeframe="funding",
            range_start=range_start,
            range_end=range_end,
            next_start=_millis_timestamp(cursor),
            status="running",
            payload={"schema": "platform.historical_bootstrap_progress/v1", "gaps": []},
        )
        rows, partitions, gaps, cursor = self._download_funding_range(
            progress_id=progress_id,
            product_id=product_id,
            symbol=symbol,
            range_start=range_start,
            range_end=range_end,
            cursor=cursor,
            range_end_ms=range_end_ms,
            maximum_pages=maximum_pages,
            heartbeat=heartbeat,
        )
        completed = cursor >= range_end_ms and not gaps
        if not completed:
            self._save_failed(
                progress_id,
                "funding_gap_detected" if gaps else "funding_range_incomplete",
                payload={"gaps": gaps},
            )
        return HistoricalBootstrapReport(
            product_id=product_id,
            market="futures",
            symbols=(symbol,),
            timeframe="funding",
            range_start=range_start,
            range_end=range_end,
            rows=rows,
            partitions=tuple(sorted(partitions)),
            gaps=tuple(gaps),
            resumed=resumed,
            completed=completed,
        )

    def _download_range(
        self,
        *,
        progress_id: str,
        product_id: str,
        market: str,
        symbol: str,
        timeframe: str,
        range_start: str,
        range_end: str,
        cursor: int,
        range_end_ms: int,
        maximum_pages: int,
        heartbeat: Callable[[], Any] | None = None,
    ) -> tuple[int, int, set[str], list[dict[str, Any]]]:
        gaps: list[dict[str, Any]] = []
        partitions: set[str] = set()
        rows_downloaded = 0
        pages = 0
        while cursor < range_end_ms:
            if heartbeat is not None:
                heartbeat()
            pages += 1
            if pages > maximum_pages:
                self._save_failed(progress_id, "maximum_pages_exceeded")
                raise HistoricalBootstrapError("historical bootstrap page budget exhausted")
            raw = self._fetch_page(
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                start_ms=cursor,
                end_ms=range_end_ms,
            )
            page = self._normalise_page(
                raw,
                product_id=product_id,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
            )
            page = [row for row in page if cursor <= int(row["open_time_ms"]) < range_end_ms]
            if not page:
                gaps.append(
                    self._gap(cursor, range_end_ms, timeframe, "exchange_returned_empty_range")
                )
                break
            page.sort(key=lambda row: int(row["open_time_ms"]))
            gaps.extend(_page_gaps(page, cursor, timeframe))
            for day, day_rows in _group_by_day(page):
                partitions.add(
                    str(
                        self._publish_partition(
                            day_rows,
                            market=market,
                            symbol=symbol,
                            timeframe=timeframe,
                            day=day,
                        )
                    )
                )
            rows_downloaded += len(page)
            next_cursor = int(page[-1]["open_time_ms"]) + self._interval(timeframe)
            if next_cursor <= cursor:
                self._save_failed(progress_id, "download_did_not_advance")
                raise HistoricalBootstrapError("historical bootstrap download did not advance")
            cursor = min(next_cursor, range_end_ms)
            self._save_progress(
                progress_id=progress_id,
                product_id=product_id,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                range_start=range_start,
                range_end=range_end,
                next_start=_millis_timestamp(cursor),
                status="running" if cursor < range_end_ms else "completed",
                payload={
                    "schema": "platform.historical_bootstrap_progress/v1",
                    "rows": rows_downloaded,
                    "gaps": gaps,
                    "partitions": sorted(partitions),
                },
            )
            if len(raw) < self.page_size:
                if cursor < range_end_ms:
                    gaps.append(self._gap(cursor, range_end_ms, timeframe, "short_exchange_page"))
                    self._save_progress(
                        progress_id=progress_id,
                        product_id=product_id,
                        market=market,
                        symbol=symbol,
                        timeframe=timeframe,
                        range_start=range_start,
                        range_end=range_end,
                        next_start=_millis_timestamp(cursor),
                        status="running",
                        payload={
                            "schema": "platform.historical_bootstrap_progress/v1",
                            "rows": rows_downloaded,
                            "gaps": gaps,
                            "partitions": sorted(partitions),
                        },
                    )
                break
        return cursor, rows_downloaded, partitions, gaps

    def _download_funding_range(
        self,
        *,
        progress_id: str,
        product_id: str,
        symbol: str,
        range_start: str,
        range_end: str,
        cursor: int,
        range_end_ms: int,
        maximum_pages: int,
        heartbeat: Callable[[], Any] | None = None,
    ) -> tuple[int, set[str], list[dict[str, Any]], int]:
        rows_downloaded = 0
        partitions: set[str] = set()
        gaps: list[dict[str, Any]] = []
        pages = 0
        while cursor < range_end_ms:
            if heartbeat is not None:
                heartbeat()
            pages += 1
            if pages > maximum_pages:
                self._save_failed(progress_id, "funding_maximum_pages_exceeded")
                raise HistoricalBootstrapError("funding bootstrap page budget exhausted")
            raw = self._fetch_funding_page(symbol=symbol, start_ms=cursor, end_ms=range_end_ms)
            page = self._normalise_funding_page(raw, product_id=product_id, symbol=symbol)
            page = [row for row in page if cursor <= int(row["funding_time_ms"]) < range_end_ms]
            if not page:
                gaps.append(self._gap(cursor, range_end_ms, "8h", "exchange_returned_empty_range"))
                break
            page.sort(key=lambda row: int(row["funding_time_ms"]))
            gaps.extend(_funding_gaps(page, cursor))
            for day, day_rows in _group_by_funding_day(page):
                partitions.add(
                    str(self._publish_funding_partition(day_rows, symbol=symbol, day=day))
                )
                self._record_funding_entries(day_rows)
            rows_downloaded += len(page)
            next_cursor = int(page[-1]["funding_time_ms"]) + 1
            if next_cursor <= cursor:
                self._save_failed(progress_id, "funding_download_did_not_advance")
                raise HistoricalBootstrapError("funding bootstrap download did not advance")
            cursor = min(next_cursor, range_end_ms)
            if range_end_ms - cursor < 8 * 3_600_000:
                cursor = range_end_ms
            self._save_progress(
                progress_id=progress_id,
                product_id=product_id,
                market="futures",
                symbol=symbol,
                timeframe="funding",
                range_start=range_start,
                range_end=range_end,
                next_start=_millis_timestamp(cursor),
                status="running" if cursor < range_end_ms else "completed",
                payload={
                    "schema": "platform.historical_bootstrap_progress/v1",
                    "rows": rows_downloaded,
                    "gaps": gaps,
                    "partitions": sorted(partitions),
                },
            )
            if len(raw) < FUNDING_PAGE_SIZE:
                if cursor < range_end_ms and range_end_ms - cursor >= 8 * 3_600_000:
                    gaps.append(self._gap(cursor, range_end_ms, "8h", "short_exchange_page"))
                break
        return rows_downloaded, partitions, gaps, cursor

    def _gap(self, start_ms: int, end_ms: int, timeframe: str, reason_code: str) -> dict[str, Any]:
        return {
            "start": _millis_timestamp(start_ms),
            "end": _millis_timestamp(max(start_ms, end_ms - self._interval(timeframe))),
            "reason_code": reason_code,
        }

    def _fetch_page(
        self, *, market: str, symbol: str, timeframe: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        base, path = ENDPOINTS[market]
        self.rate_limiter.acquire()
        response = self.session.get(
            f"{base}{path}",
            params={
                "symbol": symbol,
                "interval": timeframe,
                "startTime": start_ms,
                "endTime": end_ms - 1,
                "limit": self.page_size,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise HistoricalBootstrapError("Binance history response is not a list")
        return payload

    def _fetch_funding_page(self, *, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        self.rate_limiter.acquire()
        response = self.session.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms - 1,
                "limit": FUNDING_PAGE_SIZE,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise HistoricalBootstrapError("Binance funding response is not a list")
        return payload

    def _normalise_page(
        self,
        raw: Iterable[Any],
        *,
        product_id: str,
        market: str,
        symbol: str,
        timeframe: str,
    ) -> list[dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        interval = self._interval(timeframe)
        downloaded_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        instrument_id = (
            f"binance:spot:{symbol}" if market == "spot" else f"binance:futures:{symbol}:USDT"
        )
        for index, item in enumerate(raw):
            if not isinstance(item, list) or len(item) < 11:
                raise HistoricalBootstrapError(f"history page row {index} is invalid")
            open_time = int(item[0])
            if open_time % interval != 0:
                raise HistoricalBootstrapError("history candle timestamp is not timeframe aligned")
            close_time = int(item[6])
            if close_time != open_time + interval - 1:
                raise HistoricalBootstrapError(
                    "history candle close timestamp is not timeframe aligned"
                )
            number_of_trades = int(item[8])
            if number_of_trades < 0:
                raise HistoricalBootstrapError("history candle trade count is negative")
            values: dict[str, Any] = {
                "event_id": canonical_hash(
                    {
                        "instrument_id": instrument_id,
                        "timeframe": timeframe,
                        "open_time_ms": open_time,
                    }
                ),
                "instrument_id": instrument_id,
                "product_id": product_id,
                "timeframe": timeframe,
                "open_time_ms": open_time,
                "close_time_ms": close_time,
                "close_timestamp": _millis_timestamp(close_time),
                "availability_time": downloaded_at,
                "open": _finite_positive(item[1], "open"),
                "high": _finite_positive(item[2], "high"),
                "low": _finite_positive(item[3], "low"),
                "close": _finite_positive(item[4], "close"),
                "volume": _finite_nonnegative(item[5], "volume"),
                "quote_asset_volume": _finite_nonnegative(item[7], "quote_asset_volume"),
                "number_of_trades": number_of_trades,
                "taker_buy_base_volume": _finite_nonnegative(item[9], "taker_buy_base_volume"),
                "taker_buy_quote_volume": _finite_nonnegative(item[10], "taker_buy_quote_volume"),
            }
            if values["high"] < max(values["open"], values["close"]) or values["low"] > min(
                values["open"], values["close"]
            ):
                raise HistoricalBootstrapError("history candle OHLC values are inconsistent")
            rows[open_time] = values
        return list(rows.values())

    def _normalise_funding_page(
        self, raw: Iterable[Any], *, product_id: str, symbol: str
    ) -> list[dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        downloaded_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        instrument_id = f"binance:futures:{symbol}:USDT"
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise HistoricalBootstrapError(f"funding page row {index} is invalid")
            row_symbol = str(item.get("symbol") or symbol).upper()
            if row_symbol != symbol:
                raise HistoricalBootstrapError("funding response changed symbol")
            funding_time = int(item["fundingTime"])
            if funding_time < 0:
                raise HistoricalBootstrapError("funding timestamp is negative")
            rate = _finite_signed(item["fundingRate"], "funding_rate")
            values = {
                "event_id": canonical_hash(
                    {
                        "product_id": product_id,
                        "instrument_id": instrument_id,
                        "funding_time_ms": funding_time,
                    }
                ),
                "instrument_id": instrument_id,
                "product_id": product_id,
                "symbol": symbol,
                "funding_time_ms": funding_time,
                "funding_timestamp": _millis_timestamp(funding_time),
                "availability_time": downloaded_at,
                "funding_rate": rate,
                "mark_price": _finite_positive(item.get("markPrice", 0.0), "mark_price")
                if item.get("markPrice") not in {None, ""}
                else None,
            }
            rows[funding_time] = values
        return list(rows.values())

    def _publish_partition(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        market: str,
        symbol: str,
        timeframe: str,
        day: str,
    ) -> Path:
        materialised = {int(row["open_time_ms"]): dict(row) for row in rows}
        directory = self.root / "bars" / "binance" / market / symbol / timeframe / f"date={day}"
        directory.mkdir(parents=True, exist_ok=True)
        existing_paths = tuple(sorted(directory.glob("*.parquet")))
        for path in existing_paths:
            try:
                existing = pq.ParquetFile(path).read().to_pylist()
            except Exception as exc:
                raise HistoricalBootstrapError(
                    f"cannot read existing bar partition {path}"
                ) from exc
            for row in existing:
                if isinstance(row, Mapping) and row.get("open_time_ms") is not None:
                    materialised.setdefault(int(row["open_time_ms"]), dict(row))
        ordered = tuple(materialised[key] for key in sorted(materialised))
        content_hash = canonical_hash(ordered).removeprefix("sha256:")
        destination = directory / f"part-{content_hash}.parquet"
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.tmp")
            table = pa.table({key: [row.get(key) for row in ordered] for key in ordered[0]})
            try:
                pq.write_table(table, temporary, compression="zstd", row_group_size=1_024)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        for path in existing_paths:
            if path == destination:
                continue
            self.quarantine_root.mkdir(parents=True, exist_ok=True)
            target = self.quarantine_root / f"{path.name}.{canonical_hash(str(path))[-16:]}"
            if target.exists():
                target = target.with_name(f"{target.name}.{os.urandom(6).hex()}")
            os.replace(path, target)
        _fsync_directory(directory)
        return destination

    def _publish_funding_partition(
        self, rows: Iterable[Mapping[str, Any]], *, symbol: str, day: str
    ) -> Path:
        materialised = {int(row["funding_time_ms"]): dict(row) for row in rows}
        directory = self.root / "funding" / "binance" / "futures" / symbol / f"date={day}"
        directory.mkdir(parents=True, exist_ok=True)
        existing_paths = tuple(sorted(directory.glob("*.parquet")))
        for path in existing_paths:
            for raw in pq.ParquetFile(path).read().to_pylist():
                if isinstance(raw, Mapping) and raw.get("funding_time_ms") is not None:
                    materialised.setdefault(int(raw["funding_time_ms"]), dict(raw))
        ordered = tuple(materialised[key] for key in sorted(materialised))
        destination = directory / f"part-{canonical_hash(ordered).removeprefix('sha256:')}.parquet"
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.tmp")
            try:
                pq.write_table(
                    pa.table({key: [row.get(key) for row in ordered] for key in ordered[0]}),
                    temporary,
                    compression="zstd",
                    row_group_size=1_024,
                )
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        for path in existing_paths:
            if path == destination:
                continue
            target = (
                self.quarantine_root / "funding" / f"{path.name}.{canonical_hash(str(path))[-16:]}"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target = target.with_name(f"{target.name}.{os.urandom(6).hex()}")
            os.replace(path, target)
        _fsync_directory(directory)
        return destination

    def _record_funding_entries(self, rows: Iterable[Mapping[str, Any]]) -> None:
        with self.engine.begin() as connection:
            for row in rows:
                payload = {
                    "kind": "historical_funding",
                    "product_id": row["product_id"],
                    "instrument_id": row["instrument_id"],
                    "symbol": row["symbol"],
                    "funding_time": row["funding_timestamp"],
                    "funding_rate": row["funding_rate"],
                    "mark_price": row["mark_price"],
                    "source": "binance_public_funding_rate",
                }
                identity = canonical_hash(payload)
                existing = connection.execute(
                    select(funding_entry.c.payload).where(funding_entry.c.id == identity)
                ).scalar_one_or_none()
                if existing is None:
                    connection.execute(
                        insert(funding_entry).values(
                            id=identity,
                            created_at=row["funding_timestamp"],
                            payload=json_value(payload, field="historical funding"),
                        )
                    )

    def _progress(self, progress_id: str) -> Mapping[str, Any] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(
                        historical_bootstrap_progress.c.next_start,
                        historical_bootstrap_progress.c.status,
                        historical_bootstrap_progress.c.payload,
                    ).where(historical_bootstrap_progress.c.id == progress_id)
                )
                .mappings()
                .first()
            )
        return row

    def _save_progress(self, **values: Any) -> None:
        payload = json_value(values.pop("payload"), field="history progress payload")
        values.setdefault("updated_at", dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat())
        content = canonical_hash({**values, "payload": payload})
        record = {
            "id": values.pop("progress_id"),
            **values,
            "content_hash": content,
            "payload": payload,
        }
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(historical_bootstrap_progress.c.id).where(
                    historical_bootstrap_progress.c.id == record["id"]
                )
            ).first()
            if existing is None:
                connection.execute(insert(historical_bootstrap_progress).values(**record))
            else:
                connection.execute(
                    update(historical_bootstrap_progress)
                    .where(historical_bootstrap_progress.c.id == record["id"])
                    .values(**{key: value for key, value in record.items() if key != "id"})
                )

    def _save_failed(
        self, progress_id: str, reason: str, *, payload: Mapping[str, Any] | None = None
    ) -> None:
        failed_payload = {"reason_code": reason, **dict(payload or {})}
        updated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        with self.engine.begin() as connection:
            current = (
                connection.execute(
                    select(historical_bootstrap_progress).where(
                        historical_bootstrap_progress.c.id == progress_id
                    )
                )
                .mappings()
                .first()
            )
            if current is None:
                return
            record = {
                "status": "failed",
                "updated_at": updated_at,
                "payload": json_value(failed_payload, field="history failure payload"),
            }
            record["content_hash"] = canonical_hash({"id": progress_id, **dict(current), **record})
            connection.execute(
                update(historical_bootstrap_progress)
                .where(historical_bootstrap_progress.c.id == progress_id)
                .values(**record)
            )

    @staticmethod
    def _interval(timeframe: str) -> int:
        try:
            return INTERVAL_MS[timeframe]
        except KeyError as exc:
            raise ValueError(f"unsupported Binance timeframe: {timeframe}") from exc


HistoricalBootstrap = HistoricalBootstrapService


def _progress_gaps(progress: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    payload = progress.get("payload")
    if payload is None:
        return ()
    if not isinstance(payload, Mapping):
        raise HistoricalBootstrapError("history progress payload is invalid")
    raw_gaps = payload.get("gaps", ())
    if not isinstance(raw_gaps, list | tuple):
        raise HistoricalBootstrapError("history progress gaps are invalid")
    gaps = tuple(item for item in raw_gaps if isinstance(item, Mapping))
    if len(gaps) != len(raw_gaps):
        raise HistoricalBootstrapError("history progress gaps are invalid")
    if any(gap.get("start") is None for gap in gaps):
        raise HistoricalBootstrapError("history progress gaps are invalid")
    return gaps


def _resume_cursor(progress: Mapping[str, Any], range_start_ms: int) -> int:
    cursor = _timestamp_millis(progress["next_start"])
    gap_starts = tuple(_timestamp_millis(gap["start"]) for gap in _progress_gaps(progress))
    if gap_starts:
        cursor = min(cursor, min(gap_starts))
    return max(range_start_ms, cursor)


def _group_by_day(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        day = str(row["close_timestamp"])[:10]
        groups.setdefault(day, []).append(row)
    return tuple((day, tuple(values)) for day, values in sorted(groups.items()))


def _group_by_funding_day(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        day = str(row["funding_timestamp"])[:10]
        groups.setdefault(day, []).append(row)
    return tuple((day, tuple(values)) for day, values in sorted(groups.items()))


def _page_gaps(
    page: Sequence[Mapping[str, Any]], cursor: int, timeframe: str
) -> list[dict[str, Any]]:
    interval = INTERVAL_MS[timeframe]
    gaps: list[dict[str, Any]] = []
    first_open = int(page[0]["open_time_ms"])
    if first_open > cursor:
        gaps.append(
            {
                "start": _millis_timestamp(cursor),
                "end": _millis_timestamp(first_open - interval),
                "reason_code": "missing_exchange_range",
            }
        )
    for previous, current in zip(page, page[1:], strict=False):
        expected = int(previous["open_time_ms"]) + interval
        current_open = int(current["open_time_ms"])
        if current_open > expected:
            gaps.append(
                {
                    "start": _millis_timestamp(expected),
                    "end": _millis_timestamp(current_open - interval),
                    "reason_code": "missing_exchange_range",
                }
            )
    return gaps


def _funding_gaps(page: Sequence[Mapping[str, Any]], cursor: int) -> list[dict[str, Any]]:
    expected = 8 * 3_600_000
    gaps: list[dict[str, Any]] = []
    first = int(page[0]["funding_time_ms"])
    if first > cursor + expected:
        gaps.append(
            {
                "start": _millis_timestamp(cursor),
                "end": _millis_timestamp(first - expected),
                "reason_code": "missing_funding_event",
            }
        )
    for previous, current in zip(page, page[1:], strict=False):
        previous_time = int(previous["funding_time_ms"])
        current_time = int(current["funding_time_ms"])
        if current_time > previous_time + expected:
            gaps.append(
                {
                    "start": _millis_timestamp(previous_time + expected),
                    "end": _millis_timestamp(current_time - expected),
                    "reason_code": "missing_funding_event",
                }
            )
    return gaps


def _instrument_payload(*, market: str, symbol: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert native Binance exchange metadata into the canonical instrument shape."""

    filters = {
        str(item.get("filterType")): item
        for item in payload.get("filters", ())
        if isinstance(item, Mapping) and item.get("filterType")
    }
    price_filter = filters.get("PRICE_FILTER", {})
    lot_filter = filters.get("LOT_SIZE", filters.get("MARKET_LOT_SIZE", {}))
    notional_filter = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
    market_type = MarketType.FUTURES if market == "futures" else MarketType.SPOT
    status = str(payload.get("status") or "TRADING").casefold()
    instrument = Instrument(
        venue="binance",
        market_type=market_type,
        base_asset=str(payload.get("baseAsset") or symbol.removesuffix("USDT")),
        quote_asset=str(payload.get("quoteAsset") or "USDT"),
        settlement_asset=(
            str(payload.get("marginAsset") or payload.get("quoteAsset") or "USDT")
            if market_type is MarketType.FUTURES
            else None
        ),
        exchange_symbol=symbol,
        price_precision=_metadata_precision(
            payload.get("pricePrecision"), price_filter.get("tickSize"), field="price"
        ),
        quantity_precision=_metadata_precision(
            payload.get("quantityPrecision"), lot_filter.get("stepSize"), field="quantity"
        ),
        minimum_quantity=_finite_metadata_number(lot_filter.get("minQty"), 0.0),
        minimum_notional=_finite_metadata_number(
            notional_filter.get("notional", notional_filter.get("minNotional")), 0.0
        ),
        contract_size=_finite_metadata_number(payload.get("contractSize"), 1.0),
        status=status,
    )
    return {
        "venue": instrument.venue,
        "market_type": instrument.market_type.value,
        "base_asset": instrument.base_asset,
        "quote_asset": instrument.quote_asset,
        "settlement_asset": instrument.settlement_asset,
        "exchange_symbol": instrument.exchange_symbol,
        "price_precision": instrument.price_precision,
        "quantity_precision": instrument.quantity_precision,
        "minimum_quantity": instrument.minimum_quantity,
        "minimum_notional": instrument.minimum_notional,
        "contract_size": instrument.contract_size,
        "status": instrument.status,
    }


def _metadata_precision(value: Any, increment: Any, *, field: str) -> int:
    if value is not None:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise HistoricalBootstrapError(f"Binance {field} precision is invalid") from exc
        if result < 0:
            raise HistoricalBootstrapError(f"Binance {field} precision is invalid")
        return result
    if increment is None:
        return 0
    text = str(increment).rstrip("0")
    if "." not in text:
        return 0
    return max(0, len(text.rsplit(".", 1)[1]))


def _finite_metadata_number(value: Any, default: float) -> float:
    if value in {None, ""}:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoricalBootstrapError("Binance instrument numeric metadata is invalid") from exc
    if not math.isfinite(result) or result < 0:
        raise HistoricalBootstrapError("Binance instrument numeric metadata is invalid")
    return result


def _millis_timestamp(value: int) -> str:
    return dt.datetime.fromtimestamp(value / 1_000, dt.UTC).replace(microsecond=0).isoformat()


def _timestamp_millis(value: object) -> int:
    return int(dt.datetime.fromisoformat(str(value)).timestamp() * 1_000)


def _finite_positive(value: Any, field: str) -> float:
    result = _finite_nonnegative(value, field)
    if result <= 0:
        raise HistoricalBootstrapError(f"history {field} must be positive")
    return result


def _finite_nonnegative(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoricalBootstrapError(f"history {field} is not numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise HistoricalBootstrapError(f"history {field} is not finite and non-negative")
    return result


def _finite_signed(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoricalBootstrapError(f"history {field} is not numeric") from exc
    if not math.isfinite(result):
        raise HistoricalBootstrapError(f"history {field} is not finite")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bootstrap canonical Binance history.")
    parser.add_argument(
        "--database-url", default=os.environ.get("TRADING_PLATFORM_DATABASE_URL", "")
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--product", required=True)
    parser.add_argument("--market", choices=tuple(ENDPOINTS), required=True)
    parser.add_argument("--symbol", action="append", required=True)
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("--database-url or TRADING_PLATFORM_DATABASE_URL is required")
    from src.data.database import PlatformDatabase

    database = PlatformDatabase(args.database_url)
    database.assert_migrated()
    reports = HistoricalBootstrapService(engine=database.engine, root=args.root).bootstrap_product(
        product_id=args.product,
        market=args.market,
        symbols=args.symbol,
        timeframe=args.timeframe,
        minimum_history_days=args.days,
    )
    print(json.dumps({"reports": [report.to_payload() for report in reports]}, sort_keys=True))


if __name__ == "__main__":
    main()
