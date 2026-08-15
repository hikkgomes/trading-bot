"""Durable order lifecycle with partial-fill and recovery states."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine

from src.data.database import exchange_order, order_intent
from src.data.database import fill as fill_table
from src.domain._codec import to_primitive
from src.domain.orders import Fill, OrderIntent, OrderSide, OrderStatus, OrderType

_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.CREATED: frozenset({OrderStatus.VALIDATED, OrderStatus.REJECTED}),
    OrderStatus.VALIDATED: frozenset({OrderStatus.PERSISTED, OrderStatus.REJECTED}),
    OrderStatus.PERSISTED: frozenset({OrderStatus.SUBMITTED, OrderStatus.RECOVERY_REQUIRED}),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.RECOVERY_REQUIRED,
        }
    ),
    OrderStatus.ACKNOWLEDGED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.RECOVERY_REQUIRED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.RECOVERY_REQUIRED,
        }
    ),
    OrderStatus.CANCEL_PENDING: frozenset(
        {
            OrderStatus.CANCELLED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.RECOVERY_REQUIRED,
        }
    ),
    OrderStatus.CANCELLED: frozenset({OrderStatus.RECONCILED}),
    OrderStatus.REJECTED: frozenset({OrderStatus.RECONCILED}),
    OrderStatus.EXPIRED: frozenset({OrderStatus.RECONCILED}),
    OrderStatus.FILLED: frozenset({OrderStatus.RECONCILED}),
    OrderStatus.RECOVERY_REQUIRED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.RECONCILED,
        }
    ),
    OrderStatus.RECONCILED: frozenset(),
}


class JsonlOrderStore:
    """Append-only local durability adapter suitable for paper and tests.

    Production deployments can replace this with a PostgreSQL implementation
    while keeping the order-manager contract unchanged.
    """

    def __init__(self, path: Path):
        self.path = path

    def append(self, *, event_type: str, intent: OrderIntent, fill: Fill | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event_type": event_type, "intent": to_primitive(intent)}
        if fill is not None:
            payload["fill"] = to_primitive(fill)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError(f"order journal must be a regular file: {self.path}")
        events: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid order journal JSON at line {line_number}: {self.path}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"order journal event must be an object at line {line_number}")
                events.append(payload)
        return tuple(events)


class OrderEventStore(Protocol):
    def append(self, *, event_type: str, intent: OrderIntent, fill: Fill | None = None) -> None: ...

    def read(self) -> tuple[dict[str, Any], ...]: ...


class SqlOrderStore:
    """PostgreSQL-backed append-only order lifecycle."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, *, event_type: str, intent: OrderIntent, fill: Fill | None = None) -> None:
        payload: dict[str, Any] = {"event_type": event_type, "intent": to_primitive(intent)}
        if fill is not None:
            payload["fill"] = to_primitive(fill)
        with self.engine.begin() as connection:
            exists = connection.execute(
                select(order_intent.c.id).where(order_intent.c.id == intent.order_id)
            ).first()
            if event_type == "created":
                if exists:
                    raise ValueError(f"duplicate order_id: {intent.order_id}")
                connection.execute(
                    insert(order_intent).values(
                        id=intent.order_id,
                        created_at=intent.created_at,
                        payload=to_primitive(intent),
                    )
                )
            elif not exists:
                raise ValueError(f"order intent does not exist: {intent.order_id}")
            sequence = (
                int(
                    connection.execute(
                        select(func.coalesce(func.max(exchange_order.c.sequence), -1)).where(
                            exchange_order.c.order_id == intent.order_id
                        )
                    ).scalar_one()
                )
                + 1
            )
            event_id = f"{intent.order_id}:{sequence}"
            connection.execute(
                insert(exchange_order).values(
                    id=event_id,
                    order_id=intent.order_id,
                    sequence=sequence,
                    created_at=intent.created_at,
                    status=event_type,
                    payload=payload,
                )
            )
            if fill is not None:
                connection.execute(
                    insert(fill_table).values(
                        id=fill.fill_id,
                        order_id=intent.order_id,
                        created_at=fill.occurred_at,
                        payload=to_primitive(fill),
                    )
                )

    def read(self) -> tuple[dict[str, Any], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(exchange_order.c.payload).order_by(
                    exchange_order.c.created_at,
                    exchange_order.c.order_id,
                    exchange_order.c.sequence,
                )
            ).scalars()
            return tuple(dict(payload) for payload in rows)


def _intent_from_dict(payload: object) -> OrderIntent:
    if not isinstance(payload, dict):
        raise ValueError("order journal intent must be an object")
    values = dict(payload)
    values["side"] = OrderSide(values["side"])
    values["order_type"] = OrderType(values["order_type"])
    values["status"] = OrderStatus(values["status"])
    return OrderIntent(**values)


def _fill_from_dict(payload: object) -> Fill:
    if not isinstance(payload, dict):
        raise ValueError("order journal fill must be an object")
    values = dict(payload)
    values["side"] = OrderSide(values["side"])
    return Fill(**values)


class OrderManager:
    """Own order state and persist every transition before side effects."""

    def __init__(self, store: OrderEventStore):
        self.store = store
        self._orders: dict[str, OrderIntent] = {}
        self._fills: dict[str, list[Fill]] = {}
        self._fill_sequence: list[Fill] = []
        self._replay()

    def reload(self) -> None:
        """Refresh the local view before work leased by another process."""
        self._orders.clear()
        self._fills.clear()
        self._fill_sequence.clear()
        self._replay()

    def _replay(self) -> None:
        seen_fills: set[str] = set()
        for line_number, event in enumerate(self.store.read(), start=1):
            event_type = event.get("event_type")
            intent = _intent_from_dict(event.get("intent"))
            previous = self._orders.get(intent.order_id)
            if previous is None:
                if event_type != "created" or intent.status is not OrderStatus.CREATED:
                    raise ValueError(
                        f"order journal starts {intent.order_id} without a created event "
                        f"at line {line_number}"
                    )
            elif event_type == "fill":
                if intent != previous:
                    raise ValueError(
                        f"fill event snapshot differs from order state at line {line_number}"
                    )
            else:
                if intent.status not in _ALLOWED_TRANSITIONS[previous.status]:
                    raise ValueError(
                        f"invalid replayed order transition {previous.status.value}->"
                        f"{intent.status.value} at line {line_number}"
                    )
                if event_type != intent.status.value:
                    raise ValueError(f"order journal event type mismatch at line {line_number}")
            self._orders[intent.order_id] = intent
            if event_type == "fill":
                fill = _fill_from_dict(event.get("fill"))
                if fill.order_id != intent.order_id:
                    raise ValueError(f"fill order identity mismatch at line {line_number}")
                if fill.fill_id in seen_fills:
                    raise ValueError(f"duplicate fill identity at line {line_number}")
                seen_fills.add(fill.fill_id)
                self._fills.setdefault(fill.order_id, []).append(fill)
                self._fill_sequence.append(fill)

    def get(self, order_id: str) -> OrderIntent:
        return self._orders[order_id]

    def all(self) -> tuple[OrderIntent, ...]:
        return tuple(self._orders[key] for key in sorted(self._orders))

    def create(self, intent: OrderIntent) -> OrderIntent:
        if intent.order_id in self._orders:
            raise ValueError(f"duplicate order_id: {intent.order_id}")
        if intent.status is not OrderStatus.CREATED:
            raise ValueError("new order intents must have CREATED status")
        self.store.append(event_type="created", intent=intent)
        self._orders[intent.order_id] = intent
        return intent

    def transition(self, order_id: str, status: OrderStatus, **changes: object) -> OrderIntent:
        current = self.get(order_id)
        if status not in _ALLOWED_TRANSITIONS[current.status]:
            raise ValueError(f"invalid order transition {current.status.value}->{status.value}")
        updated = replace(current, status=status, **changes)
        self.store.append(event_type=status.value, intent=updated)
        self._orders[order_id] = updated
        return updated

    def persist_for_submission(self, order_id: str) -> OrderIntent:
        current = self.get(order_id)
        if current.status is OrderStatus.CREATED:
            self.transition(order_id, OrderStatus.VALIDATED)
        current = self.get(order_id)
        if current.status is OrderStatus.VALIDATED:
            return self.transition(order_id, OrderStatus.PERSISTED)
        if current.status is OrderStatus.PERSISTED:
            return current
        raise ValueError(f"order {order_id} is not ready for submission")

    def submitted(self, order_id: str) -> OrderIntent:
        return self.transition(order_id, OrderStatus.SUBMITTED)

    def acknowledged(self, order_id: str) -> OrderIntent:
        return self.transition(order_id, OrderStatus.ACKNOWLEDGED)

    def apply_fill(self, fill: Fill) -> OrderIntent:
        current = self.get(fill.order_id)
        existing = next(
            (item for item in self._fills.get(fill.order_id, ()) if item.fill_id == fill.fill_id),
            None,
        )
        if existing is not None:
            if existing != fill:
                raise ValueError(f"fill identity collision: {fill.fill_id}")
            return current
        if current.is_terminal:
            raise ValueError(f"cannot fill terminal order {fill.order_id}")
        if fill.instrument_id != current.instrument_id or fill.side is not current.side:
            raise ValueError("fill identity does not match order")
        if fill.quantity > current.remaining_quantity + 1e-12:
            raise ValueError("fill exceeds remaining order quantity")
        total_quantity = current.filled_quantity + fill.quantity
        average = (
            fill.price
            if current.average_fill_price is None
            else (current.average_fill_price * current.filled_quantity + fill.price * fill.quantity)
            / total_quantity
        )
        status = (
            OrderStatus.FILLED
            if abs(total_quantity - current.quantity) <= 1e-12
            else OrderStatus.PARTIALLY_FILLED
        )
        if current.status is OrderStatus.SUBMITTED:
            self.acknowledged(fill.order_id)
        updated = self.transition(
            fill.order_id,
            status,
            filled_quantity=total_quantity,
            average_fill_price=average,
            fee=current.fee + fill.fee,
        )
        self.store.append(event_type="fill", intent=updated, fill=fill)
        self._fills.setdefault(fill.order_id, []).append(fill)
        self._fill_sequence.append(fill)
        return updated

    def request_cancel(self, order_id: str) -> OrderIntent:
        return self.transition(order_id, OrderStatus.CANCEL_PENDING)

    def cancelled(self, order_id: str) -> OrderIntent:
        return self.transition(order_id, OrderStatus.CANCELLED)

    def recovery_required(self, order_id: str) -> OrderIntent:
        return self.transition(order_id, OrderStatus.RECOVERY_REQUIRED)

    def reconcile(self, order_id: str) -> OrderIntent:
        return self.transition(order_id, OrderStatus.RECONCILED)

    def fills_for(self, order_id: str) -> tuple[Fill, ...]:
        return tuple(self._fills.get(order_id, ()))

    def all_fills(self) -> tuple[Fill, ...]:
        return tuple(self._fill_sequence)

    def restore(self, events: Iterable[OrderIntent]) -> None:
        """Restore latest snapshots supplied by an authoritative event store."""
        for intent in events:
            existing = self._orders.get(intent.order_id)
            if existing is None or intent.status != existing.status:
                self._orders[intent.order_id] = intent
