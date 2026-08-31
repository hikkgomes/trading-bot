"""Process-wide exchange request throttling for execution and recovery."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from functools import wraps
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

    def acquire(self) -> None:
        """Wait until this request owns the next exchange request slot."""

        while True:
            with self._lock:
                now = self._clock()
                delay = max(0.0, self._next_allowed - now)
                if delay == 0.0:
                    self._next_allowed = now + self.minimum_interval_seconds
                    return
            self._sleeper(delay)


class RateLimitedExchangeClient:
    """Proxy that throttles every network-shaped SDK method used by the bot."""

    def __init__(self, client: Any, limiter: ExchangeRateLimiter) -> None:
        self._client = client
        self._limiter = limiter

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if not callable(attribute) or not _requires_rate_limit(name):
            return attribute

        @wraps(attribute)
        def limited(*args: Any, **kwargs: Any) -> Any:
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
            limiter = ExchangeRateLimiter(requested_interval)
            _LIMITERS[normalized] = limiter
        elif requested_interval > limiter.minimum_interval_seconds:
            # A second client may request a stricter policy.  Keeping the
            # larger interval preserves safety for all clients sharing a key.
            limiter.minimum_interval_seconds = requested_interval
        return limiter
