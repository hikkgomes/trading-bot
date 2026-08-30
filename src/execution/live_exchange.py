"""Canonical order-contract adapter for an approved live broker."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from src.domain.instruments import Instrument
from src.domain.orders import OrderIntent, OrderStatus
from src.execution.broker import (
    Broker,
    BrokerOrderState,
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


@dataclass(frozen=True)
class SubmissionAcknowledgement:
    order_id: str
    client_order_id: str
    exchange_order_id: str
    status: str


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

    def submit(self, intent: OrderIntent) -> SubmissionAcknowledgement:
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
            response = self.broker.submit_order(broker_order)
        except Exception:
            self.order_manager.recovery_required(intent.order_id)
            raise
        exchange_order_id = str(response.exchange_order_id or "").strip()
        if not exchange_order_id:
            self.order_manager.recovery_required(intent.order_id)
            raise RuntimeError("exchange acknowledgement has no order ID")
        acknowledged_client_id = str(response.client_order_id or client_order_id)
        if acknowledged_client_id != client_order_id:
            self.order_manager.recovery_required(intent.order_id)
            raise RuntimeError("exchange acknowledgement changed the client order ID")
        self.order_manager.bind_exchange_acknowledgement(
            intent.order_id,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
        )
        return SubmissionAcknowledgement(
            order_id=intent.order_id,
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
            status="acknowledged",
        )

    def cancel(self, intent: OrderIntent) -> BrokerOrderState:
        """Cancel a previously acknowledged regular order and verify the result."""
        instrument = self.instruments.get(intent.instrument_id)
        if instrument is None:
            raise ValueError(f"instrument is not approved for cancellation: {intent.instrument_id}")
        exchange_order_id = str(intent.metadata.get("exchange_order_id") or "")
        client_order_id = str(intent.metadata.get("client_order_id") or "")
        if not exchange_order_id and not client_order_id:
            raise ValueError(f"order {intent.order_id} has no exchange cancellation identity")
        current = self.order_manager.get(intent.order_id)
        if current.status in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}:
            self.order_manager.request_cancel(intent.order_id)
        state = self.broker.cancel_order(
            symbol=instrument.exchange_symbol,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
        )
        if str(state.status).lower() in {"open", "new", "accepted", "partially_filled"}:
            self.order_manager.recovery_required(intent.order_id)
            raise RuntimeError(f"exchange order {intent.order_id} remains open after cancellation")
        if self.order_manager.get(intent.order_id).status is OrderStatus.CANCEL_PENDING:
            self.order_manager.cancelled(intent.order_id)
        return state


def bounded_client_order_id(intent: OrderIntent) -> str:
    """Create a deterministic venue-safe ID from the complete intent identity."""

    material = (
        f"{intent.order_id}|{intent.instrument_id}|{intent.created_at}|{intent.quantity:.12f}"
    )
    return "c" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:35]
