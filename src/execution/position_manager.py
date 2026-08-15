"""Apply fills to multi-symbol portfolio positions."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from src.data.database import position as position_table
from src.data.database import position_event
from src.domain._codec import canonical_hash, to_primitive
from src.domain.orders import Fill, OrderSide
from src.domain.positions import Position, PositionStatus

if TYPE_CHECKING:
    from src.execution.order_manager import OrderManager


class PositionStore(Protocol):
    def save(self, position: Position) -> None: ...

    def load(self) -> tuple[Position, ...]: ...


class SqlPositionStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    @staticmethod
    def _position_id(item: Position) -> str:
        return canonical_hash(
            {"portfolio_id": item.portfolio_id, "instrument_id": item.instrument_id}
        )

    def save(self, position: Position) -> None:
        payload = to_primitive(position)
        position_id = self._position_id(position)
        event_id = canonical_hash(payload)
        with self.engine.begin() as connection:
            exists = connection.execute(
                select(position_table.c.id).where(position_table.c.id == position_id)
            ).first()
            if exists:
                connection.execute(
                    update(position_table)
                    .where(position_table.c.id == position_id)
                    .values(created_at=position.updated_at, payload=payload)
                )
            else:
                connection.execute(
                    insert(position_table).values(
                        id=position_id,
                        created_at=position.updated_at,
                        payload=payload,
                    )
                )
            if (
                connection.execute(
                    select(position_event.c.id).where(position_event.c.id == event_id)
                ).first()
                is None
            ):
                connection.execute(
                    insert(position_event).values(
                        id=event_id,
                        created_at=position.updated_at,
                        payload=payload,
                    )
                )

    def load(self) -> tuple[Position, ...]:
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(position_table.c.payload).order_by(position_table.c.id)
            ).scalars()
            positions: list[Position] = []
            for payload in payloads:
                values = dict(payload)
                values["status"] = PositionStatus(values["status"])
                positions.append(Position(**values))
            return tuple(positions)


class PositionManager:
    def __init__(self, store: PositionStore | None = None) -> None:
        self.store = store
        persisted = store.load() if store is not None else ()
        self._positions: dict[tuple[str, str], Position] = {
            (item.portfolio_id, item.instrument_id): item for item in persisted
        }

    def reload(self) -> None:
        """Refresh durable positions changed by another service process."""
        if self.store is None:
            return
        self._positions = {
            (item.portfolio_id, item.instrument_id): item for item in self.store.load()
        }

    def get(self, portfolio_id: str, instrument_id: str) -> Position:
        key = (portfolio_id, instrument_id)
        return self._positions.get(
            key,
            Position(
                portfolio_id=portfolio_id,
                instrument_id=instrument_id,
                quantity=0.0,
                average_entry_price=0.0,
                status=PositionStatus.FLAT_CONFIRMED,
                updated_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            ),
        )

    def current_quantities(self, portfolio_id: str) -> dict[str, float]:
        return {
            instrument_id: position.quantity
            for (portfolio, instrument_id), position in self._positions.items()
            if portfolio == portfolio_id
        }

    def reconcile_position(
        self,
        *,
        portfolio_id: str,
        instrument_id: str,
        quantity: float,
        average_entry_price: float,
        updated_at: str,
        contributions: dict[str, float] | None = None,
    ) -> Position:
        """Replace one local position with an authoritative reconciled balance."""
        status = PositionStatus.OPEN if quantity != 0 else PositionStatus.FLAT_CONFIRMED
        position = Position(
            portfolio_id=portfolio_id,
            instrument_id=instrument_id,
            quantity=quantity,
            average_entry_price=average_entry_price if quantity != 0 else 0.0,
            status=status,
            updated_at=updated_at,
            strategy_contributions=contributions or {},
            metadata={
                "source": "authoritative_reconciliation",
                "reference_entry_price": average_entry_price if quantity != 0 else 0.0,
            },
        )
        self._positions[(portfolio_id, instrument_id)] = position
        if self.store is not None:
            self.store.save(position)
        return position

    def apply_fill(
        self, portfolio_id: str, fill: Fill, *, contributions: dict[str, float] | None = None
    ) -> Position:
        current = self.get(portfolio_id, fill.instrument_id)
        base_fee_quantity = float(fill.metadata.get("base_fee_quantity") or 0.0)
        signed_fill = (
            fill.quantity - base_fee_quantity
            if fill.side is OrderSide.BUY
            else -(fill.quantity + base_fee_quantity)
        )
        quantity = current.quantity + signed_fill
        reference_fill_price = float(fill.metadata.get("reference_price") or fill.price)
        current_reference_price = float(
            current.metadata.get("reference_entry_price") or current.average_entry_price
        )
        if quantity == 0 or (current.quantity > 0 > quantity) or (current.quantity < 0 < quantity):
            average = fill.price if quantity != 0 else 0.0
            reference_average = reference_fill_price if quantity != 0 else 0.0
        elif current.quantity == 0 or (current.quantity > 0) == (signed_fill > 0):
            average = (
                fill.price
                if current.quantity == 0
                else (
                    current.average_entry_price * abs(current.quantity)
                    + fill.price * abs(signed_fill)
                )
                / abs(quantity)
            )
            reference_average = (
                reference_fill_price
                if current.quantity == 0
                else (
                    current_reference_price * abs(current.quantity)
                    + reference_fill_price * abs(signed_fill)
                )
                / abs(quantity)
            )
        else:
            average = current.average_entry_price
            reference_average = current_reference_price
        status = PositionStatus.OPEN if quantity != 0 else PositionStatus.FLAT_CONFIRMED
        position = Position(
            portfolio_id=portfolio_id,
            instrument_id=fill.instrument_id,
            quantity=quantity,
            average_entry_price=average,
            status=status,
            updated_at=fill.occurred_at,
            strategy_contributions=contributions or dict(current.strategy_contributions),
            metadata={**dict(current.metadata), "reference_entry_price": reference_average},
        )
        self._positions[(portfolio_id, fill.instrument_id)] = position
        if self.store is not None:
            self.store.save(position)
        return position

    def all(self) -> tuple[Position, ...]:
        return tuple(self._positions[key] for key in sorted(self._positions))

    def recover_from_orders(self, order_manager: OrderManager) -> None:
        """Reconstruct positions from the durable order and fill journal."""
        self._positions.clear()
        for fill in order_manager.all_fills():
            order = order_manager.get(fill.order_id)
            self.apply_fill(
                order.portfolio_id,
                fill,
                contributions=dict(order.strategy_contributions),
            )
