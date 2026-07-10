"""Broker abstraction for futures execution.

A single interface the algo-trading system can target regardless of venue. Two
implementations ship:

* ``PaperBroker``  — simulated fills (default; safe for development and paper
  autopilot cycles).
* ``CcxtBroker``   — any ccxt-supported futures exchange (Binance USDM, Bybit,
  OKX, ...). Live order placement is gated behind explicit env switches.

Positions are signed: ``qty > 0`` is long, ``qty < 0`` is short, ``0`` is flat.
Balances are in the quote currency (e.g. USDT).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class ProtectiveOrderStatus(str, Enum):
    """Normalized lifecycle for an exchange-native protective stop."""

    OPEN = "open"
    TRIGGERED = "triggered"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    qty: float  # absolute base-asset quantity (always positive)
    type: OrderType = OrderType.MARKET
    price: float | None = None  # required for LIMIT
    reduce_only: bool = False
    client_id: str | None = None


@dataclass
class Fill:
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ProtectiveOrder:
    """Validated exchange-native reduce-only stop state.

    ``filled_qty`` and ``average_price`` are populated when ``status`` is
    ``TRIGGERED``.  They intentionally live on the stop record so a bot can
    adopt an exchange-triggered close after a restart without inventing a
    synthetic fill.
    """

    symbol: str
    side: OrderSide
    qty: float
    trigger_price: float
    status: ProtectiveOrderStatus
    order_id: str
    client_id: str
    filled_qty: float = 0.0
    average_price: float | None = None
    fee: float = 0.0


@dataclass(frozen=True)
class OpenOrderIdentity:
    """Sanitized identity for an exchange order that is still open.

    The raw exchange response is intentionally not retained: preflight and
    entry gates only need enough evidence to identify an unexpected order
    without copying arbitrary exchange payloads into reports or logs.
    """

    symbol: str
    order_id: str
    client_id: str
    status: str
    conditional: bool


@dataclass
class Position:
    symbol: str
    qty: float = 0.0  # signed
    avg_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < 1e-12

    @property
    def side(self) -> OrderSide | None:
        if self.is_flat:
            return None
        return OrderSide.BUY if self.qty > 0 else OrderSide.SELL


class Broker(ABC):
    """Minimal synchronous broker interface."""

    name: str = "broker"

    @abstractmethod
    def get_price(self, symbol: str) -> float: ...

    @abstractmethod
    def get_balance(self) -> float:
        """Free quote-currency balance (e.g. USDT)."""

    @abstractmethod
    def get_position(self, symbol: str) -> Position: ...

    @abstractmethod
    def place_order(self, order: Order) -> Fill: ...

    def normalize_order_qty(
        self,
        symbol: str,
        qty: float,
        *,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> float:
        """Return the venue-valid quantity for a prospective order.

        The default is an identity hook so existing paper/custom brokers remain
        source compatible. Live adapters should apply exchange precision and
        minimum-order filters before callers persist a write-ahead intent.
        """

        return qty

    def normalize_order_price(self, symbol: str, price: float) -> float:
        """Return the venue-valid price/trigger value (identity by default)."""

        return price

    def supports_native_protective_stops(self) -> bool:
        """Whether this broker can place and reconcile native stop orders.

        Defaulting to ``False`` keeps paper and third-party broker adapters
        source-compatible while making live callers opt in explicitly.
        """

        return False

    def verify_one_way_position_mode(self, symbol: str) -> bool:
        """Read and verify that futures positions use one-way mode.

        The default fails closed. Live futures adapters must implement this as
        a read-only query; connected preflight must not change account settings.
        """

        raise NotImplementedError(
            f"{self.name} cannot verify one-way futures position mode for {symbol}."
        )

    def list_open_orders(
        self,
        symbol: str,
        *,
        conditional: bool,
    ) -> tuple[OpenOrderIdentity, ...]:
        """Return sanitized open-order identities for ``symbol``.

        The default deliberately fails closed. A live entry or connected
        preflight must not infer that an account has no outstanding orders
        merely because a custom broker has no read implementation.
        """

        order_kind = "conditional" if conditional else "regular"
        raise NotImplementedError(f"{self.name} cannot verify {order_kind} open orders.")

    def place_protective_stop(
        self,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        trigger_price: float,
        client_id: str,
    ) -> ProtectiveOrder:
        raise NotImplementedError(f"{self.name} does not support native protective stops.")

    def get_protective_stop(
        self,
        *,
        symbol: str,
        order_id: str | None,
        client_id: str,
    ) -> ProtectiveOrder:
        raise NotImplementedError(f"{self.name} does not support native protective stops.")

    def cancel_protective_stop(
        self,
        *,
        symbol: str,
        order_id: str | None,
        client_id: str,
    ) -> ProtectiveOrder:
        raise NotImplementedError(f"{self.name} does not support native protective stops.")

    def close_position(self, symbol: str) -> Fill | None:
        """Market-close any open position on ``symbol``. Default impl uses
        ``place_order`` with a reduce-only market order."""
        pos = self.get_position(symbol)
        if pos.is_flat:
            return None
        side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
        return self.place_order(
            Order(
                symbol=symbol, side=side, qty=abs(pos.qty), type=OrderType.MARKET, reduce_only=True
            )
        )
