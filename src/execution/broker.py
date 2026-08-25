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
    exchange_order_id: str | None = None
    client_order_id: str | None = None
    fee_asset: str | None = None


@dataclass(frozen=True)
class BrokerOrderAcknowledgement:
    exchange_order_id: str
    client_order_id: str
    status: str
    submitted_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class BrokerOrderState:
    exchange_order_id: str
    client_order_id: str
    status: str
    filled_quantity: float
    average_price: float | None


@dataclass(frozen=True)
class BrokerFill:
    trade_id: str
    exchange_order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    occurred_at: float


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


@dataclass(frozen=True)
class FuturesPositionIdentity:
    """Sanitized non-flat position from a whole futures account inventory.

    Entry and production-readiness gates use this deliberately small record so
    an adapter cannot accidentally copy arbitrary authenticated exchange
    payloads into logs.  ``qty`` is signed using the same convention as
    :class:`Position`.
    """

    symbol: str
    qty: float
    avg_price: float


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

    def submit_order(self, order: Order) -> BrokerOrderAcknowledgement:
        """Submit without assigning fill authority to the REST response.

        Legacy adapters receive a compatible acknowledgement from their
        immediate fill. Live adapters must override this method.
        """

        fill = self.place_order(order)
        exchange_order_id = str(fill.exchange_order_id or "")
        client_order_id = str(fill.client_order_id or order.client_id or "")
        if not exchange_order_id or not client_order_id:
            raise RuntimeError("broker acknowledgement has no order identity")
        return BrokerOrderAcknowledgement(
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
            status="filled",
            submitted_at=fill.timestamp,
        )

    def query_order(
        self, *, symbol: str, exchange_order_id: str, client_order_id: str
    ) -> BrokerOrderState:
        raise NotImplementedError(f"{self.name} cannot query exchange order state")

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

    def list_account_open_orders(
        self,
        *,
        conditional: bool,
    ) -> tuple[OpenOrderIdentity, ...]:
        """Return every active futures order in the authenticated account.

        This is intentionally separate from the symbol-scoped method: passing
        a configured symbol to an exchange endpoint cannot prove that a
        dedicated execution account has no exposure elsewhere.  The default
        fails closed for adapters that have not implemented a validated
        whole-account read.
        """

        order_kind = "conditional" if conditional else "regular"
        raise NotImplementedError(
            f"{self.name} cannot verify account-wide {order_kind} open orders."
        )

    def list_account_futures_positions(self) -> tuple[FuturesPositionIdentity, ...]:
        """Return every sanitized non-flat futures position in the account.

        Implementations must parse the complete authenticated response and
        fail on malformed records instead of dropping records they do not
        understand.
        """

        raise NotImplementedError(f"{self.name} cannot verify account-wide futures positions.")

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
