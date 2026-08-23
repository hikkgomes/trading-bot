"""Canonical order-contract adapter for an approved live broker."""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping

from src.domain.instruments import Instrument
from src.domain.orders import Fill, OrderIntent
from src.domain.orders import OrderSide as DomainOrderSide
from src.execution.broker import (
    Broker,
    Order,
)
from src.execution.broker import (
    OrderSide as BrokerOrderSide,
)
from src.execution.broker import (
    OrderType as BrokerOrderType,
)
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager


class BrokerExecutionVenue:
    """Apply durable intents to a live/testnet broker after write-ahead persistence."""

    def __init__(
        self,
        *,
        order_manager: OrderManager,
        position_manager: PositionManager,
        broker: Broker,
        instruments: Mapping[str, Instrument],
    ) -> None:
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.broker = broker
        self.instruments = dict(instruments)

    def submit(self, intent: OrderIntent) -> Fill:
        instrument = self.instruments.get(intent.instrument_id)
        if instrument is None or not instrument.is_tradable:
            raise ValueError(f"instrument is not approved for execution: {intent.instrument_id}")
        existing = {item.order_id: item for item in self.order_manager.all()}
        if intent.order_id not in existing:
            self.order_manager.create(intent)
        elif existing[intent.order_id] != intent:
            raise ValueError(f"order identity collision: {intent.order_id}")
        client_order_id = bounded_client_order_id(intent)
        self.order_manager.bind_client_order_id(intent.order_id, client_order_id)
        self.order_manager.persist_for_submission(intent.order_id)
        self.order_manager.submitted(intent.order_id)
        broker_order = Order(
            symbol=instrument.exchange_symbol,
            side=BrokerOrderSide(intent.side.value),
            qty=intent.quantity,
            type=BrokerOrderType(intent.order_type.value),
            price=intent.limit_price,
            reduce_only=intent.reduce_only,
            client_id=client_order_id,
        )
        try:
            broker_fill = self.broker.place_order(broker_order)
        except Exception:
            self.order_manager.recovery_required(intent.order_id)
            raise
        occurred_at = (
            dt.datetime.fromtimestamp(broker_fill.timestamp, dt.UTC)
            .replace(microsecond=0)
            .isoformat()
        )
        fill = Fill(
            fill_id="fill_"
            + hashlib.sha256(
                f"{intent.order_id}|{broker_fill.timestamp}|{broker_fill.qty}".encode()
            ).hexdigest()[:24],
            order_id=intent.order_id,
            instrument_id=intent.instrument_id,
            side=DomainOrderSide(broker_fill.side.value),
            quantity=broker_fill.qty,
            price=broker_fill.price,
            fee=broker_fill.fee,
            occurred_at=occurred_at,
            fee_asset=broker_fill.fee_asset or instrument.quote_asset,
            metadata={
                "broker": self.broker.name,
                "simulated": False,
                "exchange_order_id": broker_fill.exchange_order_id,
                "client_order_id": broker_fill.client_order_id or client_order_id,
            },
        )
        updated = self.order_manager.apply_fill(fill)
        self.position_manager.apply_fill(
            updated.portfolio_id,
            fill,
            contributions=dict(updated.strategy_contributions),
        )
        return fill


def bounded_client_order_id(intent: OrderIntent) -> str:
    """Create a deterministic venue-safe ID from the complete intent identity."""

    material = (
        f"{intent.order_id}|{intent.instrument_id}|{intent.created_at}|{intent.quantity:.12f}"
    )
    return "c" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:35]
