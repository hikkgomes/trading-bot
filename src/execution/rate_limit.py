"""Process-wide exchange request throttling for execution and recovery."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any


class ExchangeRateLimiter:
    """Reserve request slots at a fixed minimum interval.

    The limiter is deliberately independent of an exchange SDK.  This keeps
    public REST repair, authenticated SDK calls, and tests on the same policy.
    Reservation happens before the request starts, so concurrent workers cannot
    accidentally burst through the exchange limit.
    """

    def __init__(
        self,
        minimum_interval_seconds: float = 0.05,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
        shared_budget=None,
        background: bool = False,
    ) -> None:
        try:
            minimum_interval_seconds = float(minimum_interval_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("exchange request interval must be numeric") from exc
        if minimum_interval_seconds < 0 or not _finite(minimum_interval_seconds):
            raise ValueError("exchange request interval must be finite and non-negative")
        self.minimum_interval_seconds = minimum_interval_seconds
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self.shared_budget = shared_budget
        self.background = background

    def acquire(self, *, cost: float = 1.0, funding: bool = False) -> None:
        """Wait until this request owns the next exchange request slot."""

        if not _finite(cost) or cost <= 0:
            raise ValueError("request cost must be finite and positive")
        while True:
            with self._lock:
                now = self._clock()
                delay = max(0.0, self._next_allowed - now)
                if delay == 0.0 and self.shared_budget is not None:
                    delay = self.shared_budget.reserve(
                        cost, background=self.background, funding=funding
                    )
                if delay == 0.0:
                    self._next_allowed = now + self.minimum_interval_seconds * cost
                    return
            self._sleeper(min(delay, 1.0))

    def observe_response(self, status: int, headers) -> None:
        if status not in {418, 429}:
            return
        values = {str(key).lower(): value for key, value in (headers or {}).items()}
        delay = max(60.0, float(values.get("retry-after", 60)))
        if self.shared_budget is not None:
            self.shared_budget.backoff(delay)
        with self._lock:
            self._next_allowed = max(self._next_allowed, self._clock() + delay)


class RateLimitedExchangeClient:
    """Proxy that throttles every network-shaped SDK method used by the bot."""

    def __init__(self, client: Any, limiter: ExchangeRateLimiter) -> None:
        self._client = client
        self._limiter = limiter
        self._weighted_requests = callable(getattr(client, "fetch2", None))
        if self._weighted_requests:
            self._install_request_budget()

    def _install_request_budget(self):
        original = self._client.fetch2

        @wraps(original)
        def request(
            path, api="public", method="GET", params=None, headers=None, body=None, config=None
        ):
            cost = self._client.calculate_rate_limiter_cost(
                api, method, path, params or {}, config or {}
            )
            # CCXT's Binance spot costs use a fifth of an IP weight. Charging the
            # same factor to futures leaves additional capacity for recovery.
            self._limiter.acquire(
                cost=max(1.0, float(cost) * 5), funding=path in {"fundingRate", "fundingInfo"}
            )
            try:
                return original(path, api, method, params or {}, headers, body, config or {})
            except Exception:
                response_headers = getattr(self._client, "last_response_headers", {}) or {}
                if any(str(key).lower() == "retry-after" for key in response_headers):
                    self._limiter.observe_response(429, response_headers)
                raise

        self._client.fetch2 = request

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if not callable(attribute) or not _requires_rate_limit(name):
            return attribute

        @wraps(attribute)
        def limited(*args: Any, **kwargs: Any) -> Any:
            if not self._weighted_requests:
                self._limiter.acquire()
            return attribute(*args, **kwargs)

        return limited


_NETWORK_METHOD_PREFIXES = (
    "fetch",
    "create",
    "cancel",
    "edit",
    "set_",
    "enable_demo",
    "close",
    "transfer",
    "withdraw",
    "request",
    "fapiPrivate",
    "dapiPrivate",
    "sapi",
    "papi",
)


def _requires_rate_limit(name: str) -> bool:
    return name == "load_markets" or name.startswith(_NETWORK_METHOD_PREFIXES)


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


_LIMITERS: dict[str, ExchangeRateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_exchange_rate_limiter(
    key: str, *, minimum_interval_seconds: float = 0.05
) -> ExchangeRateLimiter:
    """Return the process-wide limiter for one venue/environment bucket."""

    normalized = str(key).strip()
    if not normalized:
        raise ValueError("exchange rate limiter key must be non-empty")
    try:
        requested_interval = float(minimum_interval_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("exchange request interval must be numeric") from exc
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(normalized)
        if limiter is None:
            limiter = _new_limiter(normalized, requested_interval)
            _LIMITERS[normalized] = limiter
        elif requested_interval > limiter.minimum_interval_seconds:
            # A second client may request a stricter policy.  Keeping the
            # larger interval preserves safety for all clients sharing a key.
            limiter.minimum_interval_seconds = requested_interval
        return limiter


def _new_limiter(key, interval):
    from src.execution.request_budget import SharedRequestBudget

    root = os.environ.get("TRADING_EXCHANGE_RATE_LIMIT_DIR")
    shared = None
    if root and key.startswith("binance"):
        environment = "testnet" if key.endswith(":True") else "production"
        shared = SharedRequestBudget(Path(root) / f"binance-{environment}.json")
    return ExchangeRateLimiter(
        interval, shared_budget=shared, background="public" in key or key.endswith(":history")
    )


def observe_http_response(limiter, response):
    status = getattr(response, "status_code", getattr(response, "status", 200))
    if status in {418, 429}:
        limiter.observe_response(status, response.headers)
