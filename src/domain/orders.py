"""Durable order and fill state contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from src.domain._codec import finite, json_value, non_empty, timestamp


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    PERSISTED = "persisted"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    RECOVERY_REQUIRED = "recovery_required"
    RECONCILED = "reconciled"


TERMINAL_ORDER_STATUSES = frozenset(
    {OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.RECONCILED}
)


@dataclass(frozen=True)
class OrderIntent:
    order_id: str
    portfolio_id: str
    instrument_id: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    created_at: str
    limit_price: float | None = None
    reduce_only: bool = False
    depends_on_order_id: str | None = None
    group_id: str | None = None
    strategy_contributions: Mapping[str, float] = field(default_factory=dict)
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: float = 0.0
    average_fill_price: float | None = None
    fee: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attribute in ("order_id", "portfolio_id", "instrument_id"):
            object.__setattr__(
                self, attribute, non_empty(getattr(self, attribute), field=attribute)
            )
        object.__setattr__(self, "quantity", finite(self.quantity, field="quantity", minimum=0.0))
        if self.quantity == 0:
            raise ValueError("quantity must be positive")
        object.__setattr__(self, "created_at", timestamp(self.created_at, field="created_at"))
        if self.order_type is OrderType.LIMIT:
            if self.limit_price is None:
                raise ValueError("limit orders require limit_price")
        if self.limit_price is not None:
            object.__setattr__(
                self, "limit_price", finite(self.limit_price, field="limit_price", minimum=0.0)
            )
            if self.limit_price == 0:
                raise ValueError("limit_price must be positive")
        object.__setattr__(
            self,
            "filled_quantity",
            finite(self.filled_quantity, field="filled_quantity", minimum=0.0),
        )
        if self.filled_quantity > self.quantity + 1e-12:
            raise ValueError("filled_quantity cannot exceed quantity")
        if self.average_fill_price is not None:
            object.__setattr__(
                self,
                "average_fill_price",
                finite(self.average_fill_price, field="average_fill_price", minimum=0.0),
            )
        object.__setattr__(self, "fee", finite(self.fee, field="fee", minimum=0.0))
        if self.group_id is not None:
            object.__setattr__(self, "group_id", non_empty(self.group_id, field="group_id"))
        if self.depends_on_order_id is not None:
            dependency = non_empty(self.depends_on_order_id, field="depends_on_order_id")
            if dependency == self.order_id:
                raise ValueError("an order cannot depend on itself")
            object.__setattr__(self, "depends_on_order_id", dependency)
        if not isinstance(self.strategy_contributions, Mapping):
            raise ValueError("strategy_contributions must be an object")
        object.__setattr__(
            self,
            "strategy_contributions",
            {
                str(key): finite(value, field="strategy contribution")
                for key, value in self.strategy_contributions.items()
            },
        )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), field="metadata"))

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_ORDER_STATUSES or self.status is OrderStatus.FILLED

    def with_status(self, status: OrderStatus, **changes: Any) -> OrderIntent:
        return replace(self, status=status, **changes)


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    instrument_id: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    occurred_at: str
    fee_asset: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attribute in ("fill_id", "order_id", "instrument_id"):
            object.__setattr__(
                self, attribute, non_empty(getattr(self, attribute), field=attribute)
            )
        for attribute in ("quantity", "price"):
            value = finite(getattr(self, attribute), field=attribute, minimum=0.0)
            if value == 0:
                raise ValueError(f"{attribute} must be positive")
            object.__setattr__(self, attribute, value)
        object.__setattr__(self, "fee", finite(self.fee, field="fee", minimum=0.0))
        object.__setattr__(self, "occurred_at", timestamp(self.occurred_at, field="occurred_at"))
        if self.fee_asset is not None:
            object.__setattr__(
                self, "fee_asset", non_empty(self.fee_asset, field="fee_asset").upper()
            )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), field="metadata"))
