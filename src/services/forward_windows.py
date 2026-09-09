"""Prospective paper evidence scopes, independent of position and cash state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import insert, select

from src.data.database import (
    active_strategy_assignment,
    control_event,
    forward_qualification_window,
    strategy_definition,
    strategy_version,
)
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.research.canonical import _identity
from src.research.objectives import objective_unit
from src.services.execution_identity import execution_engine_digest


class ForwardWindowError(ValueError):
    """The requested evidence scope cannot provide current qualification."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


def current_window(
    connection, *, strategy_version_id: str, product_id: str, artefact_hash: str, at: str
) -> dict[str, Any] | None:
    row = (
        connection.execute(
            select(forward_qualification_window)
            .where(
                forward_qualification_window.c.strategy_version_id == strategy_version_id,
                forward_qualification_window.c.product_id == product_id,
                forward_qualification_window.c.artefact_hash == artefact_hash,
                forward_qualification_window.c.opened_at
                <= timestamp(at, field="window lookup time"),
            )
            .order_by(
                forward_qualification_window.c.opened_at.desc(),
                forward_qualification_window.c.id.desc(),
            )
            .limit(1)
        )
        .mappings()
        .first()
    )
    return _decode_window(row)


def _decode_window(row) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row["payload"])
    if (
        payload.get("schema") != "platform.forward_qualification_window/v1"
        or payload.get("purpose") != "operational_forward"
        or payload.get("confidence_authority") is not False
        or canonical_hash(payload.get("policy")) != payload.get("policy_hash")
        or canonical_hash(payload) != row["id"]
    ) or any(
        payload.get(field) != row[field]
        for field in (
            "strategy_version_id",
            "product_id",
            "artefact_hash",
            "opened_at",
            "previous_window_id",
            "source_event_id",
        )
    ):
        raise ForwardWindowError("forward_window_corrupt")
    return {**payload, "window_id": row["id"]}


def open_window(
    connection,
    *,
    strategy_version_id: str,
    product_id: str,
    artefact_hash: str,
    opened_at: str,
    source_event_id: str,
    policy: Mapping[str, Any],
    reason: str,
    previous_window_id: str | None = None,
) -> dict[str, Any]:
    from src.research.canonical import _window_decision_limits

    _window_decision_limits(policy)
    opened_at = timestamp(opened_at, field="forward window start")
    artefact_hash = _identity(artefact_hash, field="window artefact hash")
    existing = current_window(
        connection,
        strategy_version_id=strategy_version_id,
        product_id=product_id,
        artefact_hash=artefact_hash,
        at=opened_at,
    )
    if existing is not None and existing["source_event_id"] == source_event_id:
        return existing
    if previous_window_id is None and existing is not None:
        return existing
    if previous_window_id is not None and (
        existing is None
        or existing["window_id"] != previous_window_id
        or existing["opened_at"] >= opened_at
    ):
        raise ValueError("forward window predecessor is stale or its start is not prospective")
    definition = connection.execute(
        select(strategy_definition.c.definition)
        .join(strategy_version, strategy_version.c.definition_id == strategy_definition.c.id)
        .where(strategy_version.c.id == strategy_version_id)
    ).scalar_one()
    unit = objective_unit(product_id)
    if unit is None:
        raise ValueError("forward window product has no accounting unit")
    payload = json_value(
        {
            "schema": "platform.forward_qualification_window/v1",
            "purpose": "operational_forward",
            "strategy_version_id": strategy_version_id,
            "product_id": product_id,
            "artefact_hash": artefact_hash,
            "opened_at": opened_at,
            "source_event_id": non_empty(source_event_id, field="window source event"),
            "previous_window_id": previous_window_id,
            "policy": dict(policy),
            "policy_hash": canonical_hash(policy),
            "engine_hash": execution_engine_digest(),
            "reason": non_empty(reason, field="window reason"),
            "confidence_authority": False,
            "family_evidence": {
                "schema": "platform.forward_family_evidence/v1",
                "family": definition["family"],
                "objective_unit": unit,
                "sample_unit": "calendar_intervals_and_completed_campaigns",
            },
        },
        field="forward window",
    )
    identity = canonical_hash(payload)
    connection.execute(
        insert(forward_qualification_window).values(
            id=identity,
            strategy_version_id=strategy_version_id,
            product_id=product_id,
            artefact_hash=artefact_hash,
            opened_at=opened_at,
            previous_window_id=previous_window_id,
            source_event_id=payload["source_event_id"],
            payload=payload,
        )
    )
    return {"window_id": identity, **payload}


def window_for_source(connection, source_event_id: str) -> dict[str, Any] | None:
    row = (
        connection.execute(
            select(forward_qualification_window).where(
                forward_qualification_window.c.source_event_id == source_event_id
            )
        )
        .mappings()
        .one_or_none()
    )
    return _decode_window(row)


def require_current_window(
    connection,
    *,
    window_id: str,
    strategy_version_id: str,
    product_id: str,
    artefact_hash: str,
    at: str,
) -> dict[str, Any]:
    window = current_window(
        connection,
        strategy_version_id=strategy_version_id,
        product_id=product_id,
        artefact_hash=artefact_hash,
        at=at,
    )
    if window is None or window["window_id"] != window_id:
        raise ForwardWindowError(
            "forward_window_superseded" if window else "forward_window_missing"
        )
    if window["engine_hash"] != execution_engine_digest():
        raise ForwardWindowError("forward_window_engine_changed")
    return window


def restart_operator_window(engine, payload: Mapping[str, Any], *, now: str) -> dict[str, Any]:
    required = {
        "request_id",
        "strategy_version_id",
        "product_id",
        "artefact_hash",
        "previous_window_id",
        "requested_by",
        "repair_reference",
        "reason_code",
    }
    if set(payload) != required:
        raise ValueError("forward window restart has an invalid field set")
    request = {field: non_empty(payload[field], field=field) for field in required}
    if request["reason_code"] not in {
        "market_data_repaired",
        "forecast_dependency_repaired",
        "execution_reconciliation_repaired",
    }:
        raise ValueError("forward window restart requires an operational repair reason")
    now = timestamp(now, field="forward window restart time")
    source_id = canonical_hash(
        {"action": "forward_window_restart", "request_id": request["request_id"]}
    )
    with engine.begin() as connection:
        existing = connection.execute(
            select(control_event.c.payload).where(control_event.c.id == source_id)
        ).scalar_one_or_none()
        if existing is not None:
            if existing.get("request") != request:
                raise ValueError("forward window restart identity collision")
            window = window_for_source(connection, source_id)
            if window is None:
                raise ForwardWindowError("forward_window_corrupt")
            return window
        _require_paper_scope(connection, request, now=now)
        current = require_current_window(
            connection,
            window_id=request["previous_window_id"],
            strategy_version_id=request["strategy_version_id"],
            product_id=request["product_id"],
            artefact_hash=request["artefact_hash"],
            at=now,
        )
        connection.execute(
            insert(control_event).values(
                id=source_id,
                created_at=now,
                payload={
                    "schema": "platform.forward_window_restart/v1",
                    "request": request,
                    "repair_evidence": "operator_attestation",
                    "risk_controls_changed": False,
                },
            )
        )
        return open_window(
            connection,
            strategy_version_id=request["strategy_version_id"],
            product_id=request["product_id"],
            artefact_hash=request["artefact_hash"],
            opened_at=now,
            source_event_id=source_id,
            policy=current["policy"],
            reason=request["reason_code"],
            previous_window_id=current["window_id"],
        )


def _require_paper_scope(connection, request: Mapping[str, str], *, now: str) -> None:
    modes = set(
        connection.execute(
            select(active_strategy_assignment.c.execution_mode).where(
                active_strategy_assignment.c.product_id == request["product_id"],
                active_strategy_assignment.c.strategy_version_id == request["strategy_version_id"],
                active_strategy_assignment.c.artefact_hash == request["artefact_hash"],
                active_strategy_assignment.c.active.is_(True),
                active_strategy_assignment.c.assigned_at <= now,
            )
        ).scalars()
    )
    if modes != {"paper"}:
        raise PermissionError("forward window restart requires an active paper-only scope")
