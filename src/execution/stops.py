"""Durable protective stop contracts shared by paper and live execution."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine

from src.data.database import protective_stop as protective_stop_table
from src.domain._codec import canonical_hash, non_empty, timestamp, to_primitive
from src.domain.orders import OrderSide


class StopStatus(StrEnum):
    ACTIVE = "active"
    PROTECTED = "protected"
    REPLACE_PENDING = "replace_pending"
    CONFIRMATION_FAILED = "confirmation_failed"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    RECONCILED = "reconciled"


@dataclass(frozen=True)
class ProtectiveStop:
    stop_id: str
    portfolio_id: str
    instrument_id: str
    exit_side: OrderSide
    quantity: float
    trigger_price: float
    created_at: str
    status: StopStatus = StopStatus.ACTIVE
    native_order_id: str | None = None
    triggered_at: str | None = None
    entry_order_id: str | None = None
    native_client_id: str | None = None
    protected_quantity: float = 0.0
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "exit_side", OrderSide(self.exit_side))
        object.__setattr__(self, "status", StopStatus(self.status))
        for field_name in ("stop_id", "portfolio_id", "instrument_id"):
            object.__setattr__(
                self, field_name, non_empty(getattr(self, field_name), field=field_name)
            )
        if self.quantity <= 0 or self.trigger_price <= 0:
            raise ValueError("stop quantity and trigger price must be positive")
        object.__setattr__(self, "created_at", timestamp(self.created_at, field="created_at"))
        if self.native_order_id is not None:
            object.__setattr__(
                self,
                "native_order_id",
                non_empty(self.native_order_id, field="native_order_id"),
            )
        if self.triggered_at is not None:
            object.__setattr__(
                self, "triggered_at", timestamp(self.triggered_at, field="triggered_at")
            )
        if self.entry_order_id is not None:
            object.__setattr__(
                self, "entry_order_id", non_empty(self.entry_order_id, field="entry_order_id")
            )
        if self.native_client_id is not None:
            object.__setattr__(
                self,
                "native_client_id",
                non_empty(self.native_client_id, field="native_client_id"),
            )
        if self.protected_quantity < 0 or self.protected_quantity > self.quantity + 1e-12:
            raise ValueError("protected quantity must be between zero and stop quantity")
        if self.failure_reason is not None:
            object.__setattr__(
                self, "failure_reason", non_empty(self.failure_reason, field="failure_reason")
            )

    def is_triggered_by(self, price: float) -> bool:
        if self.status not in {
            StopStatus.ACTIVE,
            StopStatus.PROTECTED,
            StopStatus.REPLACE_PENDING,
            StopStatus.CONFIRMATION_FAILED,
        }:
            return False
        return (
            price <= self.trigger_price
            if self.exit_side is OrderSide.SELL
            else price >= self.trigger_price
        )


class JsonlStopStore:
    def __init__(self, path: Path):
        self.path = path

    def append(self, stop: ProtectiveStop) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(asdict(stop), sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> tuple[ProtectiveStop, ...]:
        if not self.path.exists():
            return ()
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("stop journal must be a regular file")
        events: list[ProtectiveStop] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                payload = json.loads(line)
                payload["exit_side"] = OrderSide(payload["exit_side"])
                payload["status"] = StopStatus(payload["status"])
                events.append(ProtectiveStop(**payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid stop event at line {line_number}") from exc
        return tuple(events)


class StopStore(Protocol):
    def append(self, stop: ProtectiveStop) -> None: ...

    def read(self) -> tuple[ProtectiveStop, ...]: ...


class SqlStopStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, stop: ProtectiveStop) -> None:
        payload = to_primitive(stop)
        event_id = canonical_hash(payload)
        with self.engine.begin() as connection:
            if connection.execute(
                select(protective_stop_table.c.id).where(protective_stop_table.c.id == event_id)
            ).first():
                return
            sequence = (
                int(
                    connection.execute(
                        select(func.coalesce(func.max(protective_stop_table.c.sequence), -1)).where(
                            protective_stop_table.c.stop_id == stop.stop_id
                        )
                    ).scalar_one()
                )
                + 1
            )
            connection.execute(
                insert(protective_stop_table).values(
                    id=event_id,
                    stop_id=stop.stop_id,
                    sequence=sequence,
                    created_at=stop.triggered_at or stop.created_at,
                    status=stop.status.value,
                    payload=payload,
                )
            )

    def read(self) -> tuple[ProtectiveStop, ...]:
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(protective_stop_table.c.payload).order_by(
                    protective_stop_table.c.created_at,
                    protective_stop_table.c.stop_id,
                    protective_stop_table.c.sequence,
                )
            ).scalars()
            events: list[ProtectiveStop] = []
            for payload in payloads:
                values = dict(payload)
                values["exit_side"] = OrderSide(values["exit_side"])
                values["status"] = StopStatus(values["status"])
                events.append(ProtectiveStop(**values))
            return tuple(events)


class StopManager:
    def __init__(self, store: StopStore):
        self.store = store
        self._stops: dict[str, ProtectiveStop] = {}
        for stop in store.read():
            self._stops[stop.stop_id] = stop

    def create(self, stop: ProtectiveStop) -> ProtectiveStop:
        if stop.stop_id in self._stops:
            raise ValueError(f"duplicate stop_id: {stop.stop_id}")
        if stop.status is not StopStatus.ACTIVE:
            raise ValueError("new stops must be active")
        self.store.append(stop)
        self._stops[stop.stop_id] = stop
        return stop

    def get(self, stop_id: str) -> ProtectiveStop:
        return self._stops[stop_id]

    def for_entry_order(self, entry_order_id: str) -> tuple[ProtectiveStop, ...]:
        return tuple(
            self._stops[key]
            for key in sorted(self._stops)
            if self._stops[key].entry_order_id == entry_order_id
        )

    def active(self) -> tuple[ProtectiveStop, ...]:
        return tuple(
            self._stops[key]
            for key in sorted(self._stops)
            if self._stops[key].status
            in {
                StopStatus.ACTIVE,
                StopStatus.PROTECTED,
                StopStatus.REPLACE_PENDING,
                StopStatus.CONFIRMATION_FAILED,
            }
        )

    def mark_protected(
        self,
        stop_id: str,
        *,
        native_order_id: str,
        native_client_id: str,
        protected_quantity: float,
    ) -> ProtectiveStop:
        current = self._stops[stop_id]
        updated = replace(
            current,
            status=StopStatus.PROTECTED,
            native_order_id=non_empty(native_order_id, field="native_order_id"),
            native_client_id=non_empty(native_client_id, field="native_client_id"),
            protected_quantity=protected_quantity,
            failure_reason=None,
        )
        self.store.append(updated)
        self._stops[stop_id] = updated
        return updated

    def mark_failure(self, stop_id: str, *, reason: str) -> ProtectiveStop:
        current = self._stops[stop_id]
        updated = replace(
            current,
            status=StopStatus.CONFIRMATION_FAILED,
            failure_reason=non_empty(reason, field="failure_reason"),
        )
        self.store.append(updated)
        self._stops[stop_id] = updated
        return updated

    def resize(self, stop_id: str, *, quantity: float) -> ProtectiveStop:
        current = self._stops[stop_id]
        if quantity <= 0:
            raise ValueError("resized stop quantity must be positive")
        updated = replace(
            current,
            quantity=quantity,
            protected_quantity=0.0,
            native_order_id=None,
            status=StopStatus.REPLACE_PENDING,
        )
        self.store.append(updated)
        self._stops[stop_id] = updated
        return updated

    def triggered(self, stop_id: str, *, triggered_at: str) -> ProtectiveStop:
        current = self._stops[stop_id]
        if current.status not in {
            StopStatus.ACTIVE,
            StopStatus.PROTECTED,
            StopStatus.REPLACE_PENDING,
            StopStatus.CONFIRMATION_FAILED,
        }:
            raise ValueError("only active stops can trigger")
        updated = replace(
            current,
            status=StopStatus.TRIGGERED,
            triggered_at=timestamp(triggered_at, field="triggered_at"),
        )
        self.store.append(updated)
        self._stops[stop_id] = updated
        return updated

    def cancel(self, stop_id: str) -> ProtectiveStop:
        current = self._stops[stop_id]
        if current.status is StopStatus.CANCELLED:
            return current
        if current.status not in {
            StopStatus.ACTIVE,
            StopStatus.PROTECTED,
            StopStatus.REPLACE_PENDING,
            StopStatus.CONFIRMATION_FAILED,
        }:
            raise ValueError("only active stops can be cancelled")
        updated = replace(current, status=StopStatus.CANCELLED)
        self.store.append(updated)
        self._stops[stop_id] = updated
        return updated

    def evaluate(
        self, prices: dict[str, float], *, triggered_at: str
    ) -> tuple[ProtectiveStop, ...]:
        triggered: list[ProtectiveStop] = []
        for stop in self.active():
            price = prices.get(stop.instrument_id)
            if price is not None and stop.is_triggered_by(float(price)):
                triggered.append(self.triggered(stop.stop_id, triggered_at=triggered_at))
        return tuple(triggered)
