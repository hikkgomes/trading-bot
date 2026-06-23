"""Broker abstraction for futures execution.

A single interface the algo-trading system can target regardless of venue. Two
implementations ship:

* ``PaperBroker``  — simulated fills (default; safe for development + the
  current paper-trading cron jobs).
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
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    qty: float  # absolute base-asset quantity (always positive)
    type: OrderType = OrderType.MARKET
    price: Optional[float] = None  # required for LIMIT
    reduce_only: bool = False
    client_id: Optional[str] = None


@dataclass
class Fill:
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Position:
    symbol: str
    qty: float = 0.0  # signed
    avg_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < 1e-12

    @property
    def side(self) -> Optional[OrderSide]:
        if self.is_flat:
            return None
        return OrderSide.BUY if self.qty > 0 else OrderSide.SELL


class Broker(ABC):
    """Minimal synchronous broker interface."""

    name: str = "broker"

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_balance(self) -> float:
        """Free quote-currency balance (e.g. USDT)."""

    @abstractmethod
    def get_position(self, symbol: str) -> Position:
        ...

    @abstractmethod
    def place_order(self, order: Order) -> Fill:
        ...

    def close_position(self, symbol: str) -> Optional[Fill]:
        """Market-close any open position on ``symbol``. Default impl uses
        ``place_order`` with a reduce-only market order."""
        pos = self.get_position(symbol)
        if pos.is_flat:
            return None
        side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
        return self.place_order(
            Order(symbol=symbol, side=side, qty=abs(pos.qty), type=OrderType.MARKET, reduce_only=True)
        )
