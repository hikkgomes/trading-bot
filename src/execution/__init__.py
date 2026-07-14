"""Futures execution layer.

    from src.execution import PaperBroker, Order, OrderSide

    broker = PaperBroker(price_source=lambda s: 30_000.0, starting_balance=1_000)
    broker.place_order(Order(symbol="BTCUSDT", side=OrderSide.BUY, qty=0.01))
    print(broker.equity())

The ccxt-backed live/testnet adapter is imported lazily (it needs the optional
``ccxt`` package): ``from src.execution.ccxt_broker import CcxtBroker``.
"""

from src.execution.broker import (
    Broker,
    Fill,
    FuturesPositionIdentity,
    OpenOrderIdentity,
    Order,
    OrderSide,
    OrderType,
    Position,
    ProtectiveOrder,
    ProtectiveOrderStatus,
)
from src.execution.config import ExchangeConfig
from src.execution.paper import PaperBroker, binance_mark_price

__all__ = [
    "Broker",
    "Order",
    "OrderSide",
    "OrderType",
    "Fill",
    "FuturesPositionIdentity",
    "OpenOrderIdentity",
    "Position",
    "ProtectiveOrder",
    "ProtectiveOrderStatus",
    "PaperBroker",
    "binance_mark_price",
    "ExchangeConfig",
]
