"""Bounded Binance public-event capture for later deterministic replay."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

from src.autopilot.io import write_json_atomic
from src.config import PROJECT_ROOT

SCHEMA = "autopilot.market_event/v1"
STATUS_SCHEMA = "autopilot.event_capture_status/v1"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "event_capture.json"
DEFAULT_STATUS = PROJECT_ROOT / "runtime" / "event_capture_status.json"
ALLOWED_STREAMS = frozenset({"aggTrade", "trade", "bookTicker", "depth20@100ms", "markPrice@1s"})
ALLOWED_HOSTS = frozenset({"fstream.binance.com", "stream.binance.com"})
MAX_EVENT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class EventSource:
    market: str
    url: str
    symbols: tuple[str, ...]
    dynamic_universe: bool
    streams: tuple[str, ...]
    include_liquidations: bool


@dataclass(frozen=True)
class EventCaptureConfig:
    path: Path
    root: Path
    market_universe_report: Path
    max_dynamic_symbols: int
    max_file_bytes: int
    max_total_bytes: int
    retention_seconds: int
    flush_seconds: float
    queue_max_events: int
    sources: tuple[EventSource, ...]


def _project_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    resolved = path.resolve(strict=False) if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} must stay inside the repository") from exc
    return resolved


def _positive_int(value: Any, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ValueError(f"{field} must be an integer in [1, {maximum}]")
    return value


def load_event_capture_config(path: Path = DEFAULT_CONFIG) -> EventCaptureConfig:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"event capture config must be a regular non-symlink file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("event capture config version must be 1")
    allowed = {
        "version",
        "root",
        "market_universe_report",
        "max_dynamic_symbols",
        "max_file_bytes",
        "max_total_bytes",
        "retention_seconds",
        "flush_seconds",
        "queue_max_events",
        "sources",
    }
    if unknown := sorted(set(payload) - allowed):
        raise ValueError(f"event capture config has unknown fields: {', '.join(unknown)}")
    flush_seconds = payload.get("flush_seconds")
    if (
        isinstance(flush_seconds, bool)
        or not isinstance(flush_seconds, int | float)
        or not math.isfinite(float(flush_seconds))
        or not 0.1 <= float(flush_seconds) <= 30
    ):
        raise ValueError("flush_seconds must be in [0.1, 30]")
    sources_payload = payload.get("sources")
    if not isinstance(sources_payload, list) or not sources_payload:
        raise ValueError("event capture sources must be a non-empty list")
    sources: list[EventSource] = []
    for index, item in enumerate(sources_payload):
        if not isinstance(item, dict):
            raise ValueError(f"sources[{index}] must be an object")
        if unknown := sorted(
            set(item)
            - {
                "market",
                "url",
                "symbols",
                "dynamic_universe",
                "streams",
                "include_liquidations",
            }
        ):
            raise ValueError(f"sources[{index}] has unknown fields: {', '.join(unknown)}")
        market = item.get("market")
        if market not in {"spot", "futures"}:
            raise ValueError(f"sources[{index}].market must be spot or futures")
        url = item.get("url")
        parsed = urlparse(str(url))
        if parsed.scheme != "wss" or parsed.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"sources[{index}].url must be an approved Binance WSS endpoint")
        symbols = item.get("symbols")
        if not isinstance(symbols, list) or not symbols:
            raise ValueError(f"sources[{index}].symbols must be a non-empty list")
        normalized_symbols = tuple(str(symbol).upper() for symbol in symbols)
        if any(not symbol.isalnum() or len(symbol) > 30 for symbol in normalized_symbols):
            raise ValueError(f"sources[{index}] contains an invalid symbol")
        streams = item.get("streams")
        if not isinstance(streams, list) or not streams:
            raise ValueError(f"sources[{index}].streams must be a non-empty list")
        if invalid := sorted(set(map(str, streams)) - ALLOWED_STREAMS):
            raise ValueError(f"sources[{index}] has unsupported streams: {', '.join(invalid)}")
        dynamic = item.get("dynamic_universe", False)
        liquidations = item.get("include_liquidations", False)
        if not isinstance(dynamic, bool) or not isinstance(liquidations, bool):
            raise ValueError(f"sources[{index}] boolean fields must be JSON booleans")
        if liquidations and market != "futures":
            raise ValueError("liquidation stream is futures-only")
        sources.append(
            EventSource(
                market=market,
                url=str(url).rstrip("/"),
                symbols=normalized_symbols,
                dynamic_universe=dynamic,
                streams=tuple(map(str, streams)),
                include_liquidations=liquidations,
            )
        )
    max_file_bytes = _positive_int(
        payload.get("max_file_bytes"), field="max_file_bytes", maximum=2 * 1024**3
    )
    max_total_bytes = _positive_int(
        payload.get("max_total_bytes"), field="max_total_bytes", maximum=100 * 1024**3
    )
    if max_total_bytes < max_file_bytes:
        raise ValueError("max_total_bytes must be at least max_file_bytes")
    return EventCaptureConfig(
        path=path.resolve(),
        root=_project_path(payload.get("root"), field="root"),
        market_universe_report=_project_path(
            payload.get("market_universe_report"), field="market_universe_report"
        ),
        max_dynamic_symbols=_positive_int(
            payload.get("max_dynamic_symbols"), field="max_dynamic_symbols", maximum=100
        ),
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        retention_seconds=_positive_int(
            payload.get("retention_seconds"), field="retention_seconds", maximum=365 * 86400
        ),
        flush_seconds=float(flush_seconds),
        queue_max_events=_positive_int(
            payload.get("queue_max_events"), field="queue_max_events", maximum=1_000_000
        ),
        sources=tuple(sources),
    )


def _dynamic_symbols(config: EventCaptureConfig) -> tuple[str, ...]:
    path = config.market_universe_report
    if not path.exists():
        return ()
    if path.is_symlink() or not path.is_file():
        raise ValueError("market universe report must be a regular non-symlink file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = payload.get("eligible_research_symbols") if isinstance(payload, dict) else None
    if not isinstance(symbols, list):
        return ()
    return tuple(str(symbol).upper() for symbol in symbols[: config.max_dynamic_symbols])


def stream_names(source: EventSource, dynamic_symbols: tuple[str, ...] = ()) -> tuple[str, ...]:
    symbols = tuple(
        dict.fromkeys((*source.symbols, *(dynamic_symbols if source.dynamic_universe else ())))
    )
    names = [f"{symbol.lower()}@{stream}" for symbol in symbols for stream in source.streams]
    if source.include_liquidations:
        names.append("!forceOrder@arr")
    return tuple(names)


def normalize_event(*, market: str, stream: str, payload: Any, received_ns: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("market event payload must be an object")
    liquidation_order = payload.get("o") if "forceOrder" in stream else None
    symbol = liquidation_order.get("s") if isinstance(liquidation_order, dict) else payload.get("s")
    event_time_ms = payload.get("E") or payload.get("T")
    return {
        "schema": SCHEMA,
        "received_ns": int(received_ns),
        "received_at": dt.datetime.fromtimestamp(received_ns / 1_000_000_000, dt.UTC).isoformat(),
        "market": market,
        "stream": stream,
        "symbol": str(symbol).upper() if symbol else None,
        "event_time_ms": int(event_time_ms) if event_time_ms is not None else None,
        "payload": payload,
    }


class EventWriter:
    def __init__(self, config: EventCaptureConfig):
        self.config = config
        self.config.root.mkdir(parents=True, exist_ok=True)
        if self.config.root.is_symlink():
            raise ValueError("event capture root must not be a symlink")
        self._handles: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}
        self._opened_hour: dict[str, str] = {}
        self._sequence: dict[str, int] = {}
        self._sizes: dict[str, int] = {}
        self._last_flush = time.monotonic()
        self.events = 0
        self.bytes = 0

    def _path(self, market: str, received_ns: int, additional_bytes: int) -> Path:
        stamp = dt.datetime.fromtimestamp(received_ns / 1_000_000_000, dt.UTC)
        hour = stamp.strftime("%Y%m%dT%H")
        current = self._paths.get(market)
        if current is not None and self._opened_hour.get(market) == hour:
            if self._sizes.get(market, 0) + additional_bytes <= self.config.max_file_bytes:
                return current
        self._close_market(market)
        sequence = self._sequence.get(f"{market}:{hour}", 0)
        while True:
            path = self.config.root / f"{market}_{hour}_{sequence:04d}.jsonl"
            if not path.exists() or path.stat().st_size < self.config.max_file_bytes:
                break
            sequence += 1
        self._sequence[f"{market}:{hour}"] = sequence
        self._paths[market] = path
        self._opened_hour[market] = hour
        self._sizes[market] = path.stat().st_size if path.exists() else 0
        return path

    def write(self, event: Mapping[str, Any]) -> None:
        market = str(event["market"])
        received_ns = int(event["received_ns"])
        line = json.dumps(dict(event), sort_keys=True, separators=(",", ":"), allow_nan=False)
        encoded_bytes = len(line.encode()) + 1
        if encoded_bytes > MAX_EVENT_BYTES:
            raise ValueError("market event exceeds maximum encoded size")
        path = self._path(market, received_ns, encoded_bytes)
        handle = self._handles.get(market)
        if handle is None:
            if path.is_symlink():
                raise ValueError(f"event file must not be a symlink: {path}")
            handle = path.open("a", encoding="utf-8")
            self._handles[market] = handle
        handle.write(line + "\n")
        self._sizes[market] = self._sizes.get(market, 0) + encoded_bytes
        self.events += 1
        self.bytes += len(line) + 1
        if time.monotonic() - self._last_flush >= self.config.flush_seconds:
            self.flush()

    def flush(self) -> None:
        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        self._last_flush = time.monotonic()

    def _close_market(self, market: str) -> None:
        handle = self._handles.pop(market, None)
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    def close(self) -> None:
        for market in list(self._handles):
            self._close_market(market)

    def enforce_retention(self, *, now: float | None = None) -> dict[str, int]:
        now = time.time() if now is None else now
        active = {path.resolve() for path in self._paths.values()}
        files = []
        for path in self.config.root.glob("*_*.jsonl"):
            if path.is_symlink() or not path.is_file() or path.resolve() in active:
                continue
            files.append((path, path.stat()))
        removed = 0
        removed_bytes = 0
        for path, stat_result in files:
            if now - stat_result.st_mtime <= self.config.retention_seconds:
                continue
            path.unlink()
            removed += 1
            removed_bytes += stat_result.st_size
        remaining = [
            (path, path.stat())
            for path in self.config.root.glob("*_*.jsonl")
            if not path.is_symlink() and path.is_file() and path.resolve() not in active
        ]
        total = sum(item.st_size for _, item in remaining) + sum(
            path.stat().st_size for path in self._paths.values() if path.exists()
        )
        for path, stat_result in sorted(remaining, key=lambda item: item[1].st_mtime):
            if total <= self.config.max_total_bytes:
                break
            path.unlink()
            total -= stat_result.st_size
            removed += 1
            removed_bytes += stat_result.st_size
        return {"removed_files": removed, "removed_bytes": removed_bytes, "total_bytes": total}


async def _collect_source(
    session: aiohttp.ClientSession,
    source: EventSource,
    names: tuple[str, ...],
    queue: asyncio.Queue,
    stop: asyncio.Event,
) -> None:
    url = f"{source.url}?streams={'/'.join(names)}"
    delay = 1.0
    while not stop.is_set():
        try:
            async with session.ws_connect(url, heartbeat=30, receive_timeout=90) as websocket:
                delay = 1.0
                async for message in websocket:
                    if stop.is_set():
                        return
                    if message.type != aiohttp.WSMsgType.TEXT:
                        if message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                            break
                        continue
                    wrapper = json.loads(message.data)
                    stream = str(wrapper.get("stream") or "")
                    event = normalize_event(
                        market=source.market,
                        stream=stream,
                        payload=wrapper.get("data"),
                        received_ns=time.time_ns(),
                    )
                    await queue.put(event)
        except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError, ValueError):
            if stop.is_set():
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                pass
            delay = min(30.0, delay * 2)


async def capture(
    config: EventCaptureConfig,
    *,
    max_events: int | None = None,
    max_seconds: float | None = None,
    status_path: Path | None = None,
) -> dict[str, Any]:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:
            pass
    dynamic = _dynamic_symbols(config)
    queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_max_events)
    writer = EventWriter(config)
    started = time.monotonic()
    last_status_write = 0.0
    last_event_ns: int | None = None
    retention = {"removed_files": 0, "removed_bytes": 0, "total_bytes": 0}
    source_status = [
        {"market": source.market, "streams": len(stream_names(source, dynamic))}
        for source in config.sources
    ]
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                _collect_source(session, source, stream_names(source, dynamic), queue, stop)
            )
            for source in config.sources
        ]
        try:
            while not stop.is_set():
                elapsed = time.monotonic() - started
                if status_path is not None and elapsed - last_status_write >= 15:
                    retention = writer.enforce_retention()
                    write_json_atomic(
                        status_path,
                        {
                            "schema": STATUS_SCHEMA,
                            "generated_at": dt.datetime.now(dt.UTC)
                            .replace(microsecond=0)
                            .isoformat(),
                            "ok": writer.events > 0 or elapsed < 90,
                            "running": True,
                            "events": writer.events,
                            "bytes_written": writer.bytes,
                            "queue_size": queue.qsize(),
                            "last_event_received_ns": last_event_ns,
                            "last_event_at": (
                                dt.datetime.fromtimestamp(last_event_ns / 1_000_000_000, dt.UTC)
                                .replace(microsecond=0)
                                .isoformat()
                                if last_event_ns is not None
                                else None
                            ),
                            "sources": source_status,
                            "dynamic_symbols": list(dynamic),
                            "retention": retention,
                        },
                    )
                    last_status_write = elapsed
                remaining = None
                if max_seconds is not None:
                    remaining = max(0.0, max_seconds - (time.monotonic() - started))
                    if remaining <= 0:
                        break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=min(1.0, remaining or 1.0))
                except TimeoutError:
                    continue
                writer.write(event)
                last_event_ns = int(event["received_ns"])
                queue.task_done()
                if max_events is not None and writer.events >= max_events:
                    break
        finally:
            stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
    retention = writer.enforce_retention()
    return {
        "schema": STATUS_SCHEMA,
        "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "ok": True,
        "events": writer.events,
        "bytes_written": writer.bytes,
        "queue_remaining": queue.qsize(),
        "sources": source_status,
        "dynamic_symbols": list(dynamic),
        "last_event_received_ns": last_event_ns,
        "retention": retention,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture bounded Binance public event streams.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate:
        config = load_event_capture_config(args.config)
        print(
            json.dumps(
                {
                    "ok": True,
                    "config": str(config.path),
                    "sources": len(config.sources),
                    "max_total_bytes": config.max_total_bytes,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    try:
        report = asyncio.run(
            capture(
                load_event_capture_config(args.config),
                max_events=args.max_events,
                max_seconds=args.max_seconds,
                status_path=args.status,
            )
        )
    except Exception as exc:
        report = {
            "schema": STATUS_SCHEMA,
            "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json_atomic(args.status, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
