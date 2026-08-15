"""Normalise Binance public stream envelopes into canonical market events."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

from src.domain.market_events import MarketEvent, MarketEventType


def _time_from_ms(value: object, *, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Binance event time must be milliseconds")
    return dt.datetime.fromtimestamp(float(value) / 1_000, dt.UTC).isoformat()


def _event_type(stream: str, payload: Mapping[str, Any]) -> MarketEventType:
    event_name = str(payload.get("e") or "")
    normalised = stream.lower()
    if "forceorder" in normalised or event_name == "forceOrder":
        return MarketEventType.LIQUIDATION
    if "aggtrade" in normalised or event_name == "aggTrade":
        return MarketEventType.AGGREGATE_TRADE
    if normalised.endswith("@trade") or event_name == "trade":
        return MarketEventType.TRADE
    if "bookticker" in normalised or event_name == "bookTicker":
        return MarketEventType.BEST_BID_ASK
    if "depth" in normalised or event_name == "depthUpdate":
        return MarketEventType.DEPTH_UPDATE
    if "markprice" in normalised or event_name == "markPriceUpdate":
        return MarketEventType.MARK_PRICE
    if "kline" in normalised or event_name == "kline":
        return MarketEventType.CANDLE
    raise ValueError(f"unsupported Binance public event: {stream or event_name}")


def normalise_public_event(
    *,
    market: str,
    stream: str,
    payload: Mapping[str, Any],
    receive_timestamp: str,
) -> MarketEvent:
    """Convert one combined-stream event without changing its raw payload."""
    if market not in {"spot", "futures"}:
        raise ValueError("Binance market must be spot or futures")
    liquidation = payload.get("o") if isinstance(payload.get("o"), Mapping) else {}
    symbol = str(payload.get("s") or liquidation.get("s") or "").upper()
    if not symbol or not symbol.isalnum():
        raise ValueError("Binance public event has no valid symbol")
    event_time = payload.get("E", payload.get("T"))
    sequence_value = next(
        (
            payload[key]
            for key in ("u", "a", "t", "E", "T")
            if isinstance(payload.get(key), int) and not isinstance(payload.get(key), bool)
        ),
        0,
    )
    settlement = ":USDT" if market == "futures" else ""
    return MarketEvent(
        instrument_id=f"binance:{market}:{symbol}{settlement}",
        event_type=_event_type(stream, payload),
        exchange_timestamp=_time_from_ms(event_time, fallback=receive_timestamp),
        receive_timestamp=receive_timestamp,
        sequence=int(sequence_value),
        payload={"stream": stream, "data": dict(payload)},
    )
