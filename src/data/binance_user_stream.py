"""Normalise authenticated Binance user-stream events without retaining secrets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.data.binance_market import _time_from_ms
from src.domain.market_events import MarketEvent, MarketEventType


def normalise_user_event(
    *,
    account_id: str,
    market: str,
    payload: Mapping[str, Any],
    receive_timestamp: str,
) -> MarketEvent:
    if market not in {"spot", "futures"}:
        raise ValueError("Binance user-stream market must be spot or futures")
    event_name = str(payload.get("e") or "")
    order = payload.get("o") if isinstance(payload.get("o"), Mapping) else {}
    symbol = str(payload.get("s") or order.get("s") or "").upper()
    if event_name in {"executionReport", "ORDER_TRADE_UPDATE"}:
        execution_type = str(payload.get("x") or order.get("x") or "")
        event_type = (
            MarketEventType.FILL_UPDATE
            if execution_type == "TRADE"
            else MarketEventType.ORDER_UPDATE
        )
        if not symbol:
            raise ValueError("Binance order update has no symbol")
        settlement = ":USDT" if market == "futures" else ""
        identity = f"binance:{market}:{symbol}{settlement}"
    elif event_name in {"outboundAccountPosition", "ACCOUNT_UPDATE", "balanceUpdate"}:
        event_type = MarketEventType.ACCOUNT_BALANCE
        identity = account_id
    else:
        raise ValueError(f"unsupported Binance user event: {event_name}")
    event_time = payload.get("E", payload.get("T"))
    sequence = payload.get("u", payload.get("T", payload.get("E", 0)))
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        sequence = 0
    return MarketEvent(
        instrument_id=identity,
        event_type=event_type,
        exchange_timestamp=_time_from_ms(event_time, fallback=receive_timestamp),
        receive_timestamp=receive_timestamp,
        sequence=sequence,
        payload={"event": event_name, "data": dict(payload)},
    )
