"""Chronologically explicit market and account events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.domain._codec import canonical_hash, json_value, non_empty, timestamp


class MarketEventType(StrEnum):
    CANDLE = "candle"
    TRADE = "trade"
    AGGREGATE_TRADE = "aggregate_trade"
    BEST_BID_ASK = "best_bid_ask"
    DEPTH_UPDATE = "depth_update"
    LIQUIDATION = "liquidation"
    MARK_PRICE = "mark_price"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    ACCOUNT_BALANCE = "account_balance"
    ORDER_UPDATE = "order_update"
    FILL_UPDATE = "fill_update"


@dataclass(frozen=True)
class MarketEvent:
    instrument_id: str
    event_type: MarketEventType
    exchange_timestamp: str
    receive_timestamp: str
    sequence: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", MarketEventType(self.event_type))
        object.__setattr__(
            self, "instrument_id", non_empty(self.instrument_id, field="instrument_id")
        )
        object.__setattr__(
            self,
            "exchange_timestamp",
            timestamp(self.exchange_timestamp, field="exchange_timestamp"),
        )
        object.__setattr__(
            self, "receive_timestamp", timestamp(self.receive_timestamp, field="receive_timestamp")
        )
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be an object")
        object.__setattr__(self, "payload", json_value(dict(self.payload), field="payload"))

    @property
    def availability_timestamp(self) -> str:
        """The earliest time a live strategy is permitted to use this event."""
        return self.receive_timestamp

    @property
    def event_id(self) -> str:
        return canonical_hash(
            {
                "instrument_id": self.instrument_id,
                "event_type": self.event_type.value,
                "exchange_timestamp": self.exchange_timestamp,
                "receive_timestamp": self.receive_timestamp,
                "sequence": self.sequence,
                "payload": dict(self.payload),
            }
        )
