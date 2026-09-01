"""Deterministic recovery plans for unknown or mismatched exchange state."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import reconciliation_event
from src.domain._codec import canonical_hash, timestamp, to_primitive
from src.execution.reconciler import ReconciliationResult


class RecoveryActionType(StrEnum):
    CANCEL_UNKNOWN_ORDER = "cancel_unknown_order"
    RECONCILE_ORDER = "reconcile_order"
    RECONCILE_POSITION = "reconcile_position"
    EMERGENCY_FLATTEN = "emergency_flatten"


@dataclass(frozen=True)
class RecoveryAction:
    action_type: RecoveryActionType
    target: str
    quantity: float | None = None


@dataclass(frozen=True)
class RecoveryPlan:
    plan_id: str
    created_at: str
    actions: tuple[RecoveryAction, ...]
    reason_code: str
    requires_operator_review: bool


class JsonlRecoveryStore:
    def __init__(self, path: Path):
        self.path = path

    def append(self, plan: RecoveryPlan) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(asdict(plan), sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> tuple[RecoveryPlan, ...]:
        if not self.path.exists():
            return ()
        plans: list[RecoveryPlan] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                payload = json.loads(line)
                payload["actions"] = tuple(
                    RecoveryAction(
                        action_type=RecoveryActionType(item["action_type"]),
                        target=item["target"],
                        quantity=item.get("quantity"),
                    )
                    for item in payload["actions"]
                )
                plans.append(RecoveryPlan(**payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid recovery plan at line {line_number}") from exc
        return tuple(plans)


class RecoveryStore(Protocol):
    def append(self, plan: RecoveryPlan) -> None: ...

    def read(self) -> tuple[RecoveryPlan, ...]: ...


class SqlRecoveryStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, plan: RecoveryPlan) -> None:
        payload = {
            "record_type": "recovery_plan",
            "plan": to_primitive(plan),
        }
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(reconciliation_event.c.payload).where(
                    reconciliation_event.c.id == plan.plan_id
                )
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != payload:
                    raise ValueError("recovery plan identity collision")
                return
            connection.execute(
                insert(reconciliation_event).values(
                    id=plan.plan_id,
                    created_at=plan.created_at,
                    payload=payload,
                )
            )

    def read(self) -> tuple[RecoveryPlan, ...]:
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(reconciliation_event.c.payload).order_by(
                    reconciliation_event.c.created_at,
                    reconciliation_event.c.id,
                )
            ).scalars()
            plans: list[RecoveryPlan] = []
            for payload in payloads:
                if payload.get("record_type") != "recovery_plan":
                    continue
                values = dict(payload["plan"])
                values["actions"] = tuple(
                    RecoveryAction(
                        action_type=RecoveryActionType(item["action_type"]),
                        target=item["target"],
                        quantity=item.get("quantity"),
                    )
                    for item in values["actions"]
                )
                plans.append(RecoveryPlan(**values))
            return tuple(plans)

    def resolve(self, plan_id: str, *, resolved_at: str, verification_hash: str) -> str:
        plan_id = str(plan_id).strip()
        if not plan_id:
            raise ValueError("recovery plan identity cannot be empty")
        resolved_at = timestamp(resolved_at, field="recovery resolution time")
        verification_hash = str(verification_hash).strip()
        if not verification_hash.startswith("sha256:") or len(verification_hash) != 71:
            raise ValueError("recovery verification hash must be a sha256 identity")
        with self.engine.begin() as connection:
            if (
                connection.execute(
                    select(reconciliation_event.c.id).where(reconciliation_event.c.id == plan_id)
                ).first()
                is None
            ):
                raise KeyError(f"recovery plan does not exist: {plan_id}")
            payload = {
                "record_type": "recovery_resolution",
                "recovery_plan_id": plan_id,
                "resolved_at": resolved_at,
                "verification_hash": verification_hash,
            }
            identity = canonical_hash(payload)
            existing = connection.execute(
                select(reconciliation_event.c.payload).where(reconciliation_event.c.id == identity)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != payload:
                    raise ValueError("recovery resolution identity collision")
                return identity
            connection.execute(
                insert(reconciliation_event).values(
                    id=identity,
                    created_at=resolved_at,
                    payload=payload,
                )
            )
            return identity


def plan_recovery(
    result: ReconciliationResult,
    *,
    created_at: str,
    store: RecoveryStore | None = None,
) -> RecoveryPlan | None:
    if not result.recovery_required:
        return None
    actions = [
        RecoveryAction(RecoveryActionType.CANCEL_UNKNOWN_ORDER, order_id)
        for order_id in result.unknown_orders
    ]
    actions.extend(
        RecoveryAction(RecoveryActionType.RECONCILE_ORDER, order_id)
        for order_id in result.missing_exchange_orders
    )
    actions.extend(
        RecoveryAction(RecoveryActionType.EMERGENCY_FLATTEN, symbol, quantity)
        for symbol, quantity in sorted(result.unknown_positions.items())
    )
    actions.extend(
        RecoveryAction(
            RecoveryActionType.RECONCILE_POSITION,
            symbol,
            values["exchange_quantity"],
        )
        for symbol, values in sorted(result.quantity_mismatches.items())
        if symbol not in result.unknown_positions
    )
    created_at = timestamp(created_at, field="created_at")
    material = {
        "created_at": created_at,
        "actions": [asdict(item) for item in actions],
    }
    plan = RecoveryPlan(
        plan_id=canonical_hash(material),
        created_at=created_at,
        actions=tuple(actions),
        reason_code="exchange_state_mismatch",
        requires_operator_review=True,
    )
    if store is not None:
        store.append(plan)
    return plan
