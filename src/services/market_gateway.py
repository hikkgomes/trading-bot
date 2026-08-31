"""Binance public and authenticated streams for the canonical data queue."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import hmac
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from src.autopilot.event_capture import EventCaptureConfig, capture
from src.data.binance_market import normalise_public_event
from src.data.binance_user_stream import normalise_user_event
from src.data.parquet_store import DurableMarketBatchSpool, MarketEventBatchSegment
from src.domain._codec import canonical_hash, to_primitive
from src.domain.market_events import MarketEvent, MarketEventType
from src.services.scheduler import DatabaseJobQueue

_CLOCK_ENDPOINTS = {
    "spot": "https://api.binance.com/api/v3/time",
    "futures": "https://fapi.binance.com/fapi/v1/time",
}
_TESTNET_CLOCK_ENDPOINTS = {
    "spot": "https://testnet.binance.vision/api/v3/time",
    "futures": "https://demo-fapi.binance.com/fapi/v1/time",
}
_USER_STREAM_ENDPOINTS = {
    "futures": (
        "https://fapi.binance.com/fapi/v1/listenKey",
        "wss://fstream.binance.com/private/ws",
    ),
}
_TESTNET_USER_STREAM_ENDPOINTS = {
    "futures": (
        "https://demo-fapi.binance.com/fapi/v1/listenKey",
        "wss://demo-fstream.binance.com/ws",
    ),
}
_SPOT_USER_STREAM = "wss://ws-api.binance.com:443/ws-api/v3?returnRateLimits=false"
_SPOT_TESTNET_USER_STREAM = "wss://ws-api.testnet.binance.vision/ws-api/v3?returnRateLimits=false"


@dataclass(frozen=True)
class UserStreamAccount:
    account_id: str
    market: str
    api_key: str
    api_secret: str
    testnet: bool = False

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> UserStreamAccount | None:
        market = (
            "futures" if payload.get("market") == "usdt_futures" else str(payload.get("market"))
        )
        if market not in {"spot", "futures"}:
            raise ValueError("Binance account market must be spot or USDT futures")
        environment_name = str(payload["api_key_env"])
        api_key = os.environ.get(environment_name, "").strip()
        secret_environment_name = str(payload["api_secret_env"])
        api_secret = os.environ.get(secret_environment_name, "").strip()
        if not api_key and not api_secret:
            return None
        if not api_key or not api_secret:
            raise ValueError("Binance user streams require both API key and secret")
        return cls(
            account_id=str(payload["account_id"]),
            market=market,
            api_key=api_key,
            api_secret=api_secret,
            testnet=str(payload.get("environment") or "production") == "testnet",
        )


class DatabaseGatewaySink:
    """Normalise stream envelopes and enqueue idempotent persistence jobs."""

    def __init__(
        self,
        queue: DatabaseJobQueue,
        *,
        spool: DurableMarketBatchSpool | None = None,
        user_stream_job_name: str = "user_stream_event",
        user_stream_job_prefix: str = "user-stream",
    ):
        self.queue = queue
        self.spool = spool
        self.user_stream_job_name = user_stream_job_name
        self.user_stream_job_prefix = user_stream_job_prefix
        self.events = 0
        self.bytes = 0

    def write(self, event: Mapping[str, Any]) -> None:
        raw_payload = event.get("payload")
        if not isinstance(raw_payload, Mapping):
            raise ValueError("Binance stream envelope has no event object")
        canonical = normalise_public_event(
            market=str(event["market"]),
            stream=str(event["stream"]),
            payload=raw_payload,
            receive_timestamp=str(event["received_at"]),
        )
        symbol = str(event["symbol"])
        self._write_market_event(canonical, market=str(event["market"]), symbol=symbol)
        funding = funding_event_from_mark_price(canonical)
        if funding is not None:
            self._write_market_event(funding, market=str(event["market"]), symbol=symbol)
        self.events += 1
        self.bytes += len(json.dumps(dict(event), sort_keys=True, separators=(",", ":")))

    def write_user(self, event: MarketEvent, *, account_id: str, market: str) -> None:
        payload = to_primitive(event)
        identity = event.event_id
        self.queue.enqueue_if_absent(
            job_id=f"{self.user_stream_job_prefix}:{identity.removeprefix('sha256:')}",
            name=self.user_stream_job_name,
            payload={
                "account_id": account_id,
                "market": market,
                "event": payload,
            },
            available_at=event.receive_timestamp,
            priority=20,
        )
        self.events += 1
        self.bytes += len(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def mark_user_stream_recovery(
        self,
        *,
        account_id: str,
        market: str,
        observed_at: str,
        reason_code: str,
    ) -> str:
        """Queue authenticated REST reconciliation after a stream restart."""

        payload = {
            "account_id": account_id,
            "market": market,
            "recovery_kind": "user_stream_reconnect",
            "reason_code": reason_code,
            "observed_at": observed_at,
        }
        identity = canonical_hash(payload).removeprefix("sha256:")
        job_id = f"{self.user_stream_job_prefix}:recovery:{identity}"
        self.queue.enqueue_if_absent(
            job_id=job_id,
            name="live_order_recovery",
            payload=payload,
            available_at=observed_at,
            priority=100,
            producer_identity=f"{self.user_stream_job_prefix}:recovery",
        )
        return job_id

    def _enqueue_market_event(
        self, event: MarketEvent, *, venue: str, market: str, symbol: str
    ) -> None:
        payload = to_primitive(event)
        identity = canonical_hash(payload)
        self.queue.enqueue_if_absent(
            job_id=f"market-event:{identity.removeprefix('sha256:')}",
            name="market_event_write",
            payload={
                "venue": venue,
                "market": market,
                "symbol": symbol,
                "event": payload,
            },
            available_at=event.receive_timestamp,
            priority=10,
        )

    def _write_market_event(self, event: MarketEvent, *, market: str, symbol: str) -> None:
        if self.spool is None:
            self._enqueue_market_event(event, venue="binance", market=market, symbol=symbol)
            return
        segment = self.spool.append(event, venue="binance", market=market, symbol=symbol)
        if segment is not None:
            self._enqueue_segment(segment)

    def _enqueue_segment(self, segment: MarketEventBatchSegment) -> None:
        self.queue.enqueue_if_absent(
            job_id=f"market-batch:{segment.content_hash.removeprefix('sha256:')}",
            name="market_event_batch_write",
            payload={
                "segment_path": str(segment.path),
                "segment_hash": segment.content_hash,
                "row_count": segment.row_count,
                "producer_identity": "market-gateway",
            },
            available_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            priority=10,
            producer_identity="market-gateway",
        )

    def flush(self) -> None:
        if self.spool is not None:
            segment = self.spool.flush()
            if segment is not None:
                self._enqueue_segment(segment)

    def close(self) -> None:
        self.flush()

    def enforce_retention(self, *, now: float | None = None) -> dict[str, int]:
        del now
        if self.spool is None:
            return {"removed_files": 0, "removed_bytes": 0, "total_bytes": 0}
        return _spool_retention(self.spool.root)


class BinanceUserStreams:
    """Create, keep alive, and reconnect authenticated Binance user streams."""

    def __init__(self) -> None:
        self._listen_keys: dict[str, str] = {}
        self._refreshed_at: dict[str, float] = {}
        self._connected_accounts: set[str] = set()
        self._connection_lock = threading.Lock()

    def connected(self, account_id: str) -> bool:
        with self._connection_lock:
            return account_id in self._connected_accounts

    def _set_connected(self, account_id: str, connected: bool) -> None:
        with self._connection_lock:
            if connected:
                self._connected_accounts.add(account_id)
            else:
                self._connected_accounts.discard(account_id)

    async def capture(
        self,
        *,
        session: aiohttp.ClientSession,
        account: UserStreamAccount,
        sink: DatabaseGatewaySink,
        maximum_seconds: float,
    ) -> int:
        if account.market == "spot":
            return await self._capture_spot(
                session=session,
                account=account,
                sink=sink,
                maximum_seconds=maximum_seconds,
            )
        listen_key = await self._listen_key(session, account)
        _, websocket_root = _stream_endpoints(account)
        deadline = time.monotonic() + maximum_seconds
        events = 0
        async with session.ws_connect(
            f"{websocket_root}/{listen_key}", heartbeat=30, receive_timeout=90
        ) as websocket:
            self._set_connected(account.account_id, True)
            try:
                while (remaining := deadline - time.monotonic()) > 0:
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=remaining)
                    except TimeoutError:
                        break
                    if message.type != aiohttp.WSMsgType.TEXT:
                        if message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                            break
                        continue
                    received_at = dt.datetime.now(dt.UTC).isoformat()
                    payload = json.loads(message.data)
                    if payload.get("e") == "listenKeyExpired":
                        self._listen_keys.pop(account.account_id, None)
                        raise RuntimeError("Binance futures user-stream listen key expired")
                    event = normalise_user_event(
                        account_id=account.account_id,
                        market=account.market,
                        payload=payload,
                        receive_timestamp=received_at,
                    )
                    sink.write_user(event, account_id=account.account_id, market=account.market)
                    events += 1
            finally:
                self._set_connected(account.account_id, False)
        return events

    async def _capture_spot(
        self,
        *,
        session: aiohttp.ClientSession,
        account: UserStreamAccount,
        sink: DatabaseGatewaySink,
        maximum_seconds: float,
    ) -> int:
        request_id = f"subscribe:{account.account_id}"
        timestamp_ms = int(time.time_ns() / 1_000_000)
        parameters: dict[str, str | int] = {
            "apiKey": account.api_key,
            "timestamp": timestamp_ms,
        }
        signing_payload = "&".join(f"{key}={value}" for key, value in sorted(parameters.items()))
        parameters["signature"] = hmac.new(
            account.api_secret.encode(),
            signing_payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        deadline = time.monotonic() + maximum_seconds
        events = 0
        endpoint = _SPOT_TESTNET_USER_STREAM if account.testnet else _SPOT_USER_STREAM
        async with session.ws_connect(endpoint, heartbeat=20, receive_timeout=60) as websocket:
            await websocket.send_json(
                {
                    "id": request_id,
                    "method": "userDataStream.subscribe.signature",
                    "params": parameters,
                }
            )
            confirmation = await asyncio.wait_for(
                websocket.receive_json(), timeout=min(10.0, maximum_seconds)
            )
            if confirmation.get("id") != request_id or confirmation.get("status") != 200:
                raise RuntimeError("Binance spot user-stream subscription was rejected")
            self._set_connected(account.account_id, True)
            try:
                while (remaining := deadline - time.monotonic()) > 0:
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=remaining)
                    except TimeoutError:
                        break
                    if message.type != aiohttp.WSMsgType.TEXT:
                        if message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                            break
                        continue
                    wrapper = json.loads(message.data)
                    payload = wrapper.get("event")
                    if not isinstance(payload, Mapping):
                        continue
                    event_name = str(payload.get("e") or "")
                    if event_name in {"eventStreamTerminated", "serverShutdown"}:
                        raise RuntimeError(f"Binance spot user stream ended: {event_name}")
                    received_at = dt.datetime.now(dt.UTC).isoformat()
                    event = normalise_user_event(
                        account_id=account.account_id,
                        market=account.market,
                        payload=payload,
                        receive_timestamp=received_at,
                    )
                    sink.write_user(event, account_id=account.account_id, market=account.market)
                    events += 1
            finally:
                self._set_connected(account.account_id, False)
        return events

    async def _listen_key(self, session: aiohttp.ClientSession, account: UserStreamAccount) -> str:
        endpoint, _ = _stream_endpoints(account)
        headers = {"X-MBX-APIKEY": account.api_key}
        current = self._listen_keys.get(account.account_id)
        refreshed_at = self._refreshed_at.get(account.account_id, 0.0)
        if current and time.monotonic() - refreshed_at < 1_500:
            return current
        method = session.put if current else session.post
        parameters = {"listenKey": current} if current else None
        async with method(endpoint, headers=headers, params=parameters) as response:
            response.raise_for_status()
            payload = await response.json()
        listen_key = current or str(payload.get("listenKey") or "")
        if not listen_key:
            raise ValueError("Binance did not return a user-stream listen key")
        self._listen_keys[account.account_id] = listen_key
        self._refreshed_at[account.account_id] = time.monotonic()
        return listen_key


def _stream_endpoints(account: UserStreamAccount) -> tuple[str, str]:
    endpoints = _TESTNET_USER_STREAM_ENDPOINTS if account.testnet else _USER_STREAM_ENDPOINTS
    if account.market not in endpoints:
        raise ValueError(f"Binance user-stream endpoints are unavailable for {account.market}")
    return endpoints[account.market]


def _mark_stream_recovery(
    sink: DatabaseGatewaySink | Any,
    *,
    account: UserStreamAccount,
    reason_code: str,
) -> None:
    marker = getattr(sink, "mark_user_stream_recovery", None)
    if not callable(marker):
        return
    marker(
        account_id=account.account_id,
        market=account.market,
        observed_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        reason_code=reason_code,
    )


def funding_event_from_mark_price(event: MarketEvent) -> MarketEvent | None:
    if event.event_type is not MarketEventType.MARK_PRICE:
        return None
    raw_data = event.payload.get("data")
    if not isinstance(raw_data, Mapping) or raw_data.get("r") is None:
        return None
    return MarketEvent(
        instrument_id=event.instrument_id,
        event_type=MarketEventType.FUNDING_RATE,
        exchange_timestamp=event.exchange_timestamp,
        receive_timestamp=event.receive_timestamp,
        sequence=event.sequence,
        payload={
            "data": {
                "funding_rate": float(raw_data["r"]),
                "next_funding_time_ms": raw_data.get("T"),
            },
            "funding_rate": float(raw_data["r"]),
            "next_funding_time_ms": raw_data.get("T"),
            "source_event_id": event.event_id,
        },
    )


def _spool_retention(root: Path, *, retention_seconds: int = 7 * 86_400) -> dict[str, int]:
    now = time.time()
    removed_files = 0
    removed_bytes = 0
    for path in root.glob("segment-*.jsonl"):
        if path.is_symlink() or not path.is_file():
            continue
        if now - path.stat().st_mtime > retention_seconds:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
    total_bytes = sum(
        path.stat().st_size for path in root.glob("segment-*.jsonl") if path.is_file()
    )
    return {
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "total_bytes": total_bytes,
    }


class DatabaseMarketGateway:
    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        capture_config: EventCaptureConfig,
        accounts: tuple[UserStreamAccount, ...] = (),
        universe_symbols: Callable[[], tuple[str, ...]] | None = None,
        maximum_clock_skew_ms: float = 1_000.0,
        testnet: bool = False,
        user_stream_job_name: str = "user_stream_event",
        user_stream_job_prefix: str = "user-stream",
    ) -> None:
        if maximum_clock_skew_ms <= 0:
            raise ValueError("maximum clock skew must be positive")
        self.queue = queue
        self.capture_config = capture_config
        self.accounts = accounts
        self.universe_symbols = universe_symbols or (lambda: ())
        self.maximum_clock_skew_ms = maximum_clock_skew_ms
        self.testnet = testnet
        self.user_stream_job_name = user_stream_job_name
        self.user_stream_job_prefix = user_stream_job_prefix
        self.user_streams = BinanceUserStreams()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[dict[str, Any]] | None = None
        self._continuous_status: dict[str, Any] = {"reason_code": "market_gateway_not_started"}
        self._continuous_error: BaseException | None = None

    def start(self) -> None:
        """Start one persistent event loop for all public and authenticated streams."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._continuous_error = None
        self._thread = threading.Thread(
            target=self._run_continuous_thread,
            name="market-gateway-streams",
            daemon=True,
        )
        self._thread.start()

    def __call__(self) -> dict[str, Any]:
        return self.poll()

    def poll(self) -> dict[str, Any]:
        self.start()
        if self._continuous_error is not None:
            raise RuntimeError("persistent market gateway failed") from self._continuous_error
        return dict(self._continuous_status)

    def stop(self) -> None:
        loop, task = self._loop, self._task
        if loop is not None and task is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                raise RuntimeError("persistent market gateway did not stop cleanly")

    def wait_for_user_stream(self, account_id: str, *, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._continuous_error is not None:
                raise RuntimeError("persistent market gateway failed") from self._continuous_error
            if self.user_streams.connected(account_id):
                return True
            time.sleep(0.05)
        return self.user_streams.connected(account_id)

    def _run_continuous_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._task = loop.create_task(self._run_continuous())
        try:
            loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            self._continuous_error = exc
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._loop = None
            self._task = None

    async def _run_continuous(self) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120)
        sink = DatabaseGatewaySink(
            self.queue,
            user_stream_job_name=self.user_stream_job_name,
            user_stream_job_prefix=self.user_stream_job_prefix,
            spool=DurableMarketBatchSpool(
                self.capture_config.root / "spool",
                max_rows=min(self.capture_config.queue_max_events, 5_000),
                max_bytes=self.capture_config.max_file_bytes,
            ),
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            clocks = await self._clock_status(session)
            public = asyncio.create_task(
                capture(
                    self.capture_config,
                    sink=sink,
                    dynamic_symbols=self.universe_symbols(),
                )
            )
            authenticated = [
                asyncio.create_task(self._capture_user_forever(session, account, sink))
                for account in self.accounts
            ]
            self._continuous_status = {
                "reason_code": "market_gateway_running",
                "events_enqueued": 0,
                "bytes_received": 0,
                "clock_status": clocks,
                "user_stream_accounts": len(self.accounts),
            }
            try:
                while True:
                    await asyncio.sleep(self.capture_config.flush_seconds)
                    sink.flush()
                    self._continuous_status = {
                        **self._continuous_status,
                        "events_enqueued": sink.events,
                        "bytes_received": sink.bytes,
                    }
                    failed = next(
                        (
                            task.exception()
                            for task in (public, *authenticated)
                            if task.done() and not task.cancelled() and task.exception() is not None
                        ),
                        None,
                    )
                    if failed is not None:
                        raise failed
                    if public.done():
                        raise RuntimeError("public market stream stopped unexpectedly")
            finally:
                for task in (public, *authenticated):
                    task.cancel()
                await asyncio.gather(public, *authenticated, return_exceptions=True)
                sink.close()

    async def _capture_user_forever(
        self,
        session: aiohttp.ClientSession,
        account: UserStreamAccount,
        sink: DatabaseGatewaySink,
    ) -> None:
        delay = 1.0
        while True:
            try:
                await self.user_streams.capture(
                    session=session,
                    account=account,
                    sink=sink,
                    maximum_seconds=3_600,
                )
                _mark_stream_recovery(
                    sink,
                    account=account,
                    reason_code="user_stream_reconnect",
                )
                delay = 1.0
            except asyncio.CancelledError:
                raise
            except (
                aiohttp.ClientError,
                TimeoutError,
                RuntimeError,
                ValueError,
                json.JSONDecodeError,
            ):
                self.user_streams._listen_keys.pop(account.account_id, None)
                _mark_stream_recovery(
                    sink,
                    account=account,
                    reason_code="user_stream_disconnect",
                )
                sink.flush()
                await asyncio.sleep(delay)
                delay = min(30.0, delay * 2)

    def run_once(self, *, maximum_seconds: float, maximum_events: int = 5_000) -> dict[str, Any]:
        return asyncio.run(
            self._run_once(
                maximum_seconds=maximum_seconds,
                maximum_events=maximum_events,
            )
        )

    async def _run_once(self, *, maximum_seconds: float, maximum_events: int) -> dict[str, Any]:
        if maximum_seconds <= 0 or maximum_events <= 0:
            raise ValueError("market gateway bounds must be positive")
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120)
        sink = DatabaseGatewaySink(
            self.queue,
            user_stream_job_name=self.user_stream_job_name,
            user_stream_job_prefix=self.user_stream_job_prefix,
            spool=DurableMarketBatchSpool(
                self.capture_config.root / "spool",
                max_rows=min(self.capture_config.queue_max_events, 5_000),
                max_bytes=self.capture_config.max_file_bytes,
            ),
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            clocks = await self._clock_status(session)
            public_task = asyncio.create_task(
                capture(
                    self.capture_config,
                    max_events=maximum_events,
                    max_seconds=maximum_seconds,
                    sink=sink,
                    dynamic_symbols=self.universe_symbols(),
                )
            )
            user_tasks = [
                asyncio.create_task(
                    self.user_streams.capture(
                        session=session,
                        account=account,
                        sink=sink,
                        maximum_seconds=maximum_seconds,
                    )
                )
                for account in self.accounts
            ]
            results = await asyncio.gather(public_task, *user_tasks, return_exceptions=True)
        public_result = results[0]
        if isinstance(public_result, BaseException):
            raise public_result
        user_errors = [
            f"{type(item).__name__}: {item}"
            for item in results[1:]
            if isinstance(item, BaseException)
        ]
        if user_errors:
            raise RuntimeError("; ".join(user_errors))
        return {
            "reason_code": "market_gateway_cycle_completed",
            "events_enqueued": sink.events,
            "bytes_received": sink.bytes,
            "clock_status": clocks,
            "user_stream_accounts": len(self.accounts),
            "user_stream_errors": [],
        }

    async def _clock_status(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        status: dict[str, Any] = {}
        endpoints = _TESTNET_CLOCK_ENDPOINTS if self.testnet else _CLOCK_ENDPOINTS
        for market, endpoint in endpoints.items():
            started_ms = time.time_ns() / 1_000_000
            async with session.get(endpoint) as response:
                response.raise_for_status()
                payload = await response.json()
            completed_ms = time.time_ns() / 1_000_000
            midpoint_ms = (started_ms + completed_ms) / 2
            skew_ms = float(payload["serverTime"]) - midpoint_ms
            if abs(skew_ms) > self.maximum_clock_skew_ms:
                raise RuntimeError(f"Binance {market} clock skew exceeds the configured limit")
            status[market] = {
                "skew_ms": skew_ms,
                "round_trip_ms": completed_ms - started_ms,
            }
        return status
