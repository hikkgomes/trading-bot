"""Chronologically explicit market and account events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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


@dataclass
class ExchangeSequenceTracker:
    """Detect duplicate and discontinuous exchange-native sequence values."""

    _last: dict[tuple[str, str], int] = field(default_factory=dict)

    def observe(self, event: MarketEvent) -> str:
        key = (event.instrument_id, event.event_type.value)
        previous = self._last.get(key)
        if previous is not None and event.sequence <= previous:
            if event.sequence == previous:
                return "duplicate"
            return "out_of_order"
        status = "gap" if previous is not None and event.sequence > previous + 1 else "ok"
        self._last[key] = event.sequence
        return status


@dataclass(frozen=True)
class MarketEvent:
    instrument_id: str
    event_type: MarketEventType
    exchange_timestamp: str
    receive_timestamp: str
    sequence: int
    payload: Mapping[str, Any]
    close_timestamp: str | None = None
    availability_time: str | None = None

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
        close_timestamp = self.close_timestamp or self.exchange_timestamp
        availability_time = self.availability_time or self.receive_timestamp
        close_timestamp = timestamp(close_timestamp, field="close_timestamp")
        availability_time = timestamp(availability_time, field="availability_time")
        if availability_time < self.receive_timestamp:
            raise ValueError("availability_time cannot precede receive_timestamp")
        object.__setattr__(self, "close_timestamp", close_timestamp)
        object.__setattr__(self, "availability_time", availability_time)

    @property
    def availability_timestamp(self) -> str:
        """The earliest time a live strategy is permitted to use this event."""
        return str(self.availability_time)

    @property
    def event_id(self) -> str:
        # Receive and availability times describe this observation, not the
        # exchange event.  They must not turn a reconnect duplicate into a new
        # event identity.
        return canonical_hash(
            {
                "instrument_id": self.instrument_id,
                "event_type": self.event_type.value,
                "exchange_timestamp": self.exchange_timestamp,
                "close_timestamp": self.close_timestamp,
                "sequence": self.sequence,
                "payload": dict(self.payload),
            }
        )

    @property
    def exchange_identity(self) -> str:
        """Stable exchange-native identity used for duplicate detection."""

        raw_value = self.payload.get("data")
        if not isinstance(raw_value, Mapping):
            raw_value = self.payload.get("event")
        raw: Mapping[str, Any] = raw_value if isinstance(raw_value, Mapping) else {}
        native = {key: raw[key] for key in ("a", "i", "t", "u", "U", "T", "E") if key in raw}
        return canonical_hash(
            {
                "instrument_id": self.instrument_id,
                "event_type": self.event_type.value,
                "sequence": self.sequence,
                "native": native,
            }
        )
