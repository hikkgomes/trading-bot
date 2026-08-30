"""State machine for pairs, carry, baskets, and other multi-leg orders."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine

from src.data.database import order_group as order_group_table
from src.domain._codec import canonical_hash, to_primitive
from src.domain.orders import OrderIntent
from src.domain.portfolios import TargetPosition
from src.execution.order_planner import plan_orders


class OrderGroupStatus(StrEnum):
    PLANNED = "planned"
    PRIMARY_SUBMITTED = "primary_submitted"
    PRIMARY_PARTIAL = "primary_partial"
    HEDGE_SUBMITTED = "hedge_submitted"
    HEDGED = "hedged"
    ACTIVE = "active"
    EXITING = "exiting"
    FLAT = "flat"
    RECOVERY = "recovery"


_TRANSITIONS: dict[OrderGroupStatus, frozenset[OrderGroupStatus]] = {
    OrderGroupStatus.PLANNED: frozenset(
        {OrderGroupStatus.PRIMARY_SUBMITTED, OrderGroupStatus.RECOVERY}
    ),
    OrderGroupStatus.PRIMARY_SUBMITTED: frozenset(
        {
            OrderGroupStatus.PRIMARY_PARTIAL,
            OrderGroupStatus.HEDGE_SUBMITTED,
            OrderGroupStatus.RECOVERY,
        }
    ),
    OrderGroupStatus.PRIMARY_PARTIAL: frozenset(
        {OrderGroupStatus.HEDGE_SUBMITTED, OrderGroupStatus.RECOVERY}
    ),
    OrderGroupStatus.HEDGE_SUBMITTED: frozenset(
        {OrderGroupStatus.HEDGED, OrderGroupStatus.RECOVERY}
    ),
    OrderGroupStatus.HEDGED: frozenset(
        {OrderGroupStatus.ACTIVE, OrderGroupStatus.EXITING, OrderGroupStatus.RECOVERY}
    ),
    OrderGroupStatus.ACTIVE: frozenset({OrderGroupStatus.EXITING, OrderGroupStatus.RECOVERY}),
    OrderGroupStatus.EXITING: frozenset({OrderGroupStatus.FLAT, OrderGroupStatus.RECOVERY}),
    OrderGroupStatus.FLAT: frozenset(),
    OrderGroupStatus.RECOVERY: frozenset({OrderGroupStatus.EXITING, OrderGroupStatus.FLAT}),
}


@dataclass(frozen=True)
class OrderGroup:
    group_id: str
    portfolio_id: str
    primary_order_id: str
    hedge_order_ids: tuple[str, ...]
    status: OrderGroupStatus = OrderGroupStatus.PLANNED
    metadata: dict[str, object] = field(default_factory=dict)

    def transition(self, status: OrderGroupStatus) -> OrderGroup:
        if status not in _TRANSITIONS[self.status]:
            raise ValueError(f"invalid order-group transition {self.status.value}->{status.value}")
        return replace(self, status=status)


@dataclass(frozen=True)
class OrderGroupPlan:
    group: OrderGroup
    orders: tuple[OrderIntent, ...]


@dataclass(frozen=True)
class MultiLegRecoveryPlan:
    group_id: str
    action: str
    target_quantities: Mapping[str, float]


@dataclass(frozen=True)
class HedgeRelease:
    orders: tuple[OrderIntent, ...]
    primary_filled_quantity: float
    hedge_error: float


class JsonlOrderGroupStore:
    def __init__(self, path: Path):
        self.path = path

    def append(self, group: OrderGroup) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(asdict(group), sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> tuple[OrderGroup, ...]:
        if not self.path.exists():
            return ()
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("order-group journal must be a regular file")
        groups: list[OrderGroup] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                payload = json.loads(line)
                payload["hedge_order_ids"] = tuple(payload["hedge_order_ids"])
                payload["status"] = OrderGroupStatus(payload["status"])
                groups.append(OrderGroup(**payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid order group at line {line_number}") from exc
        return tuple(groups)


class OrderGroupStore(Protocol):
    def append(self, group: OrderGroup) -> None: ...

    def read(self) -> tuple[OrderGroup, ...]: ...


class SqlOrderGroupStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, group: OrderGroup) -> None:
        payload = to_primitive(group)
        event_id = canonical_hash(payload)
        created_at = str(group.metadata.get("created_at") or "1970-01-01T00:00:00+00:00")
        with self.engine.begin() as connection:
            if connection.execute(
                select(order_group_table.c.id).where(order_group_table.c.id == event_id)
            ).first():
                return
            sequence = (
                int(
                    connection.execute(
                        select(func.coalesce(func.max(order_group_table.c.sequence), -1)).where(
                            order_group_table.c.group_id == group.group_id
                        )
                    ).scalar_one()
                )
                + 1
            )
            connection.execute(
                insert(order_group_table).values(
                    id=event_id,
                    group_id=group.group_id,
                    sequence=sequence,
                    created_at=created_at,
                    status=group.status.value,
                    payload=payload,
                )
            )

    def read(self) -> tuple[OrderGroup, ...]:
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(order_group_table.c.payload).order_by(
                    order_group_table.c.created_at,
                    order_group_table.c.group_id,
                    order_group_table.c.sequence,
                )
            ).scalars()
            groups: list[OrderGroup] = []
            for payload in payloads:
                values = dict(payload)
                values["hedge_order_ids"] = tuple(values["hedge_order_ids"])
                values["status"] = OrderGroupStatus(values["status"])
                groups.append(OrderGroup(**values))
            return tuple(groups)


class OrderGroupManager:
    def __init__(self, store: OrderGroupStore):
        self.store = store
        self._groups: dict[str, OrderGroup] = {}
        for group in store.read():
            self._groups[group.group_id] = group

    def reload(self) -> None:
        """Refresh group state written by another service process."""
        self._groups.clear()
        for group in self.store.read():
            self._groups[group.group_id] = group

    def create(self, group: OrderGroup) -> OrderGroup:
        if group.group_id in self._groups:
            raise ValueError(f"duplicate group_id: {group.group_id}")
        self.store.append(group)
        self._groups[group.group_id] = group
        return group

    def get(self, group_id: str) -> OrderGroup:
        return self._groups[group_id]

    def transition(self, group_id: str, status: OrderGroupStatus) -> OrderGroup:
        updated = self.get(group_id).transition(status)
        self.store.append(updated)
        self._groups[group_id] = updated
        return updated

    def recovery_plan(self, group_id: str) -> MultiLegRecoveryPlan:
        group = self.get(group_id)
        raw_targets = group.metadata.get("target_quantities")
        if not isinstance(raw_targets, dict):
            raise ValueError("order group has no target quantities")
        targets = {str(key): float(value) for key, value in raw_targets.items()}
        policy = str(group.metadata.get("recovery_policy") or "unwind")
        if policy == "hedge":
            action = "complete_hedge"
        elif policy == "unwind":
            action = "unwind_to_flat"
            targets = {instrument_id: 0.0 for instrument_id in targets}
        else:
            raise ValueError(f"unsupported order-group recovery policy: {policy}")
        return MultiLegRecoveryPlan(
            group_id=group_id,
            action=action,
            target_quantities=targets,
        )


def plan_order_group(
    targets: Iterable[TargetPosition],
    *,
    current_quantities: Mapping[str, float],
    decided_at: str,
    recovery_policy: str = "unwind",
    prices: Mapping[str, float] | None = None,
) -> OrderGroupPlan:
    materialised = tuple(targets)
    if len(materialised) < 2:
        raise ValueError("multi-leg order groups require at least two targets")
    if recovery_policy not in {"hedge", "unwind"}:
        raise ValueError("recovery_policy must be hedge or unwind")
    group_id = canonical_hash(
        {
            "portfolio_id": materialised[0].portfolio_id,
            "targets": [
                {
                    "instrument_id": item.instrument_id,
                    "target_quantity": item.target_quantity,
                }
                for item in materialised
            ],
            "decided_at": decided_at,
        }
    )
    raw_orders = plan_orders(
        materialised,
        current_quantities=current_quantities,
        decided_at=decided_at,
        prices=prices,
    )
    if len(raw_orders) < 2:
        raise ValueError("multi-leg targets must produce at least two orders")
    primary_order_id = raw_orders[0].order_id
    orders = tuple(
        replace(
            order,
            group_id=group_id,
            depends_on_order_id=(primary_order_id if index else order.depends_on_order_id),
        )
        for index, order in enumerate(raw_orders)
    )
    group = OrderGroup(
        group_id=group_id,
        portfolio_id=materialised[0].portfolio_id,
        primary_order_id=primary_order_id,
        hedge_order_ids=tuple(order.order_id for order in orders[1:]),
        metadata={
            "created_at": decided_at,
            "target_quantities": {
                item.instrument_id: item.target_quantity for item in materialised
            },
            "recovery_policy": recovery_policy,
        },
    )
    return OrderGroupPlan(group=group, orders=orders)


def release_hedges_from_primary_fill(
    plan: OrderGroupPlan,
    *,
    primary_filled_quantity: float,
    hedge_filled_quantities: Mapping[str, float],
) -> HedgeRelease:
    """Resize hedge legs from authoritative primary trade quantities."""

    if primary_filled_quantity <= 0:
        raise ValueError("primary fill quantity must be positive")
    primary = plan.orders[0]
    if primary_filled_quantity > primary.quantity + 1e-12:
        raise ValueError("primary fill exceeds the planned primary quantity")
    scale = primary_filled_quantity / primary.quantity
    hedges = tuple(
        replace(
            order,
            quantity=order.quantity * scale,
            order_id=canonical_hash(
                {
                    "planned_order_id": order.order_id,
                    "primary_filled_quantity": primary_filled_quantity,
                }
            ),
            metadata={
                **dict(order.metadata),
                "planned_order_id": order.order_id,
                "primary_fill_scale": scale,
            },
        )
        for order in plan.orders[1:]
    )
    required = sum(order.quantity for order in hedges)
    filled = sum(
        max(0.0, float(hedge_filled_quantities.get(order.instrument_id, 0.0))) for order in hedges
    )
    error = 0.0 if required == 0 else max(0.0, (required - filled) / required)
    return HedgeRelease(hedges, primary_filled_quantity, error)


def deterministic_unwind_orders(
    plan: OrderGroupPlan,
    *,
    actual_signed_quantities: Mapping[str, float],
    decided_at: str,
) -> tuple[OrderIntent, ...]:
    """Flatten only the actual residual exposure when a hedge cannot complete."""

    targets = tuple(
        TargetPosition(
            portfolio_id=plan.group.portfolio_id,
            instrument_id=instrument_id,
            target_quantity=0.0,
            target_notional=0.0,
            target_fraction=0.0,
            strategy_contributions={"multi_leg_recovery": 1.0},
            risk_budget=0.0,
            valid_until=decided_at,
            metadata={"reason": "multi_leg_recovery_unwind"},
        )
        for instrument_id, quantity in sorted(actual_signed_quantities.items())
        if abs(float(quantity)) > 1e-12
    )
    if not targets:
        return ()
    return tuple(
        replace(order, reduce_only=True, group_id=plan.group.group_id)
        for order in plan_orders(
            targets,
            current_quantities=actual_signed_quantities,
            decided_at=decided_at,
        )
    )
