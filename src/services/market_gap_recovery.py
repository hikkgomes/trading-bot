"""Bounded public REST repair for exchange market-stream sequence gaps."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from typing import Any

import requests

from src.data.binance_market import normalise_public_event
from src.domain.market_events import MarketEvent, MarketEventType
from src.execution.rate_limit import ExchangeRateLimiter, shared_exchange_rate_limiter

_REST_ENDPOINTS = {
    "spot": "https://api.binance.com",
    "futures": "https://fapi.binance.com",
}
_INTERVALS_MS = {
    "1s": 1_000,
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


class MarketGapRepairError(RuntimeError):
    """The exchange did not provide a safe REST repair for a stream gap."""


class BinanceMarketGapRepair:
    """Fetch bounded public REST data for a canonical gap-recovery command."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        base_urls: Mapping[str, str] | None = None,
        timeout_seconds: float = 10.0,
        clock_ms: Callable[[], int] | None = None,
        rate_limiter: ExchangeRateLimiter | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("gap repair timeout must be positive")
        urls = dict(base_urls or _REST_ENDPOINTS)
        if set(urls) != set(_REST_ENDPOINTS):
            raise ValueError("gap repair base URLs must define spot and futures")
        self.session = session or requests.Session()
        self.base_urls = {market: str(url).rstrip("/") for market, url in urls.items()}
        self.timeout_seconds = timeout_seconds
        self.clock_ms = clock_ms or (lambda: int(time.time_ns() / 1_000_000))
        self.rate_limiter = rate_limiter or shared_exchange_rate_limiter(
            "binance-public-market-repair"
        )

    def __call__(self, request: Mapping[str, Any]) -> tuple[MarketEvent, ...]:
        market = str(request.get("market") or "")
        symbol = str(request.get("symbol") or "").upper()
        if market not in self.base_urls or not symbol.isalnum():
            raise MarketGapRepairError("gap recovery request has an invalid market or symbol")
        event_type = MarketEventType(str(request.get("event_type") or ""))
        handlers = {
            MarketEventType.DEPTH_UPDATE: self._depth,
            MarketEventType.AGGREGATE_TRADE: self._aggregate_trades,
            MarketEventType.CANDLE: self._candles,
            MarketEventType.MARK_PRICE: self._mark_price,
            MarketEventType.BEST_BID_ASK: self._book_ticker,
        }
        handler = handlers.get(event_type)
        if handler is None:
            raise MarketGapRepairError(
                f"REST gap repair is not supported for {event_type.value} events"
            )
        return handler(request, market=market, symbol=symbol)

    def _get(self, *, market: str, path: str, params: Mapping[str, Any]) -> Any:
        self.rate_limiter.acquire()
        response = self.session.get(
            f"{self.base_urls[market]}{path}",
            params=dict(params),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _depth(
        self, request: Mapping[str, Any], *, market: str, symbol: str
    ) -> tuple[MarketEvent, ...]:
        path = "/api/v3/depth" if market == "spot" else "/fapi/v1/depth"
        payload = self._get(market=market, path=path, params={"symbol": symbol, "limit": 1_000})
        if not isinstance(payload, Mapping) or not isinstance(payload.get("lastUpdateId"), int):
            raise MarketGapRepairError("REST depth response has no update identity")
        raw = {
            "e": "depthUpdate",
            "E": self.clock_ms(),
            "s": symbol,
            "U": payload["lastUpdateId"],
            "u": payload["lastUpdateId"],
            "b": payload.get("bids", []),
            "a": payload.get("asks", []),
        }
        return (
            self._normalise(
                raw, market=market, stream=f"{symbol.lower()}@depth20@100ms", request=request
            ),
        )

    def _aggregate_trades(
        self, request: Mapping[str, Any], *, market: str, symbol: str
    ) -> tuple[MarketEvent, ...]:
        previous = _optional_int(request.get("previous_sequence"))
        current = _optional_int(request.get("sequence"))
        params: dict[str, Any] = {"symbol": symbol, "limit": 1_000}
        if previous is not None:
            params["fromId"] = previous + 1
        if current is not None and current > 0:
            params["toId"] = current - 1
        if previous is not None and current is not None and previous + 1 >= current:
            return ()
        payload = self._get(
            market=market,
            path="/api/v3/aggTrades" if market == "spot" else "/fapi/v1/aggTrades",
            params=params,
        )
        if not isinstance(payload, list):
            raise MarketGapRepairError("REST aggregate-trade response is not a list")
        return tuple(
            self._normalise(
                {
                    "e": "aggTrade",
                    "E": int(item["T"]),
                    "s": symbol,
                    "a": int(item["a"]),
                    "p": item["p"],
                    "q": item["q"],
                    "f": item.get("f"),
                    "l": item.get("l"),
                    "T": int(item["T"]),
                    "m": item.get("m"),
                    "M": item.get("M"),
                },
                market=market,
                stream=f"{symbol.lower()}@aggTrade",
                request=request,
            )
            for item in payload
            if isinstance(item, Mapping) and {"a", "p", "q", "T"}.issubset(item)
        )

    def _candles(
        self, request: Mapping[str, Any], *, market: str, symbol: str
    ) -> tuple[MarketEvent, ...]:
        source = request.get("event")
        source_data = source.get("data") if isinstance(source, Mapping) else None
        source_kline = source_data.get("k") if isinstance(source_data, Mapping) else None
        if not isinstance(source_kline, Mapping):
            raise MarketGapRepairError("candle gap request has no source kline")
        interval = str(source_kline.get("i") or _stream_interval(source))
        interval_ms = _INTERVALS_MS.get(interval)
        if interval_ms is None:
            raise MarketGapRepairError(f"unsupported candle interval: {interval}")
        open_time = _required_int(source_kline.get("t"), "candle open time")
        close_time = _required_int(source_kline.get("T"), "candle close time")
        path = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
        payload = self._get(
            market=market,
            path=path,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": max(0, open_time - interval_ms),
                "endTime": close_time,
                "limit": 1_000,
            },
        )
        if not isinstance(payload, list):
            raise MarketGapRepairError("REST kline response is not a list")
        events: list[MarketEvent] = []
        for row in payload:
            if not isinstance(row, list) or len(row) < 7:
                continue
            if not all(isinstance(row[index], int | float) for index in (0, 6)):
                continue
            raw = {
                "e": "kline",
                "E": int(row[6]),
                "s": symbol,
                "k": {
                    "t": int(row[0]),
                    "T": int(row[6]),
                    "s": symbol,
                    "i": interval,
                    "o": row[1],
                    "c": row[4],
                    "h": row[2],
                    "l": row[3],
                    "v": row[5],
                    "x": int(row[6]) <= self.clock_ms(),
                },
            }
            events.append(
                self._normalise(
                    raw, market=market, stream=f"{symbol.lower()}@kline_{interval}", request=request
                )
            )
        return tuple(events)

    def _mark_price(
        self, request: Mapping[str, Any], *, market: str, symbol: str
    ) -> tuple[MarketEvent, ...]:
        if market != "futures":
            raise MarketGapRepairError("mark-price REST repair is futures-only")
        payload = self._get(market=market, path="/fapi/v1/premiumIndex", params={"symbol": symbol})
        if not isinstance(payload, Mapping) or payload.get("markPrice") is None:
            raise MarketGapRepairError("REST mark-price response has no mark price")
        raw = {
            "e": "markPriceUpdate",
            "E": self.clock_ms(),
            "s": symbol,
            "p": payload["markPrice"],
            "r": payload.get("lastFundingRate", 0.0),
            "T": payload.get("nextFundingTime", self.clock_ms()),
        }
        return (
            self._normalise(
                raw, market=market, stream=f"{symbol.lower()}@markPrice@1s", request=request
            ),
        )

    def _book_ticker(
        self, request: Mapping[str, Any], *, market: str, symbol: str
    ) -> tuple[MarketEvent, ...]:
        path = "/api/v3/ticker/bookTicker" if market == "spot" else "/fapi/v1/ticker/bookTicker"
        payload = self._get(market=market, path=path, params={"symbol": symbol})
        if not isinstance(payload, Mapping) or not {"bidPrice", "askPrice"}.issubset(payload):
            raise MarketGapRepairError("REST book-ticker response has no bid and ask")
        raw = {
            "e": "bookTicker",
            "E": self.clock_ms(),
            "s": symbol,
            "b": payload["bidPrice"],
            "a": payload["askPrice"],
            "B": payload.get("bidQty", 0.0),
            "A": payload.get("askQty", 0.0),
        }
        return (
            self._normalise(
                raw, market=market, stream=f"{symbol.lower()}@bookTicker", request=request
            ),
        )

    @staticmethod
    def _normalise(
        payload: Mapping[str, Any],
        *,
        market: str,
        stream: str,
        request: Mapping[str, Any],
    ) -> MarketEvent:
        return normalise_public_event(
            market=market,
            stream=stream,
            payload=payload,
            receive_timestamp=str(request["observed_at"]),
        )


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _required_int(value: object, field: str) -> int:
    result = _optional_int(value)
    if result is None:
        raise MarketGapRepairError(f"{field} is not an integer")
    return result


def _stream_interval(source: object) -> str:
    stream = str(source.get("stream") if isinstance(source, Mapping) else "")
    match = re.search(r"@kline_([A-Za-z0-9]+)$", stream)
    return match.group(1) if match else ""
