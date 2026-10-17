"""Durable service heartbeats for platform health and control."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, insert, select
from sqlalchemy.engine import Engine

from src.data.database import service_heartbeat
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.services.alerting import AlertRecord, AlertSeverity, AlertSink


@dataclass(frozen=True)
class ServiceHealth:
    service_name: str
    node_id: str
    observed_at: str
    healthy: bool
    payload: dict[str, Any]


def publish_external_heartbeat(
    store: DatabaseHeartbeatStore,
    *,
    sink: AlertSink | None,
    node_id: str,
    required_services: tuple[str, ...],
    now: str,
    maximum_age_seconds: int = 60,
) -> str:
    """Send only current health; an off-host monitor detects missing pings."""
    if sink is None:
        return "not_configured"
    if not required_services or maximum_age_seconds <= 0:
        raise ValueError("external health requires services and a positive freshness limit")
    current = dt.datetime.fromisoformat(timestamp(now, field="now"))
    healthy = {
        item.service_name
        for item in store.latest()
        if item.node_id == node_id
        and item.healthy
        and 0
        <= (current - dt.datetime.fromisoformat(item.observed_at)).total_seconds()
        <= maximum_age_seconds
    }
    if not set(required_services).issubset(healthy):
        return "withheld_unhealthy"
    record = AlertRecord(
        alert_id=canonical_hash({"node": node_id, "observed_at": now}),
        event_type="platform_heartbeat",
        severity=AlertSeverity.INFO,
        dedupe_key=node_id,
        target=node_id,
        message="Critical platform services are reachable",
        emitted_at=now,
    )
    try:
        sink.send(record)
    except (OSError, RuntimeError):
        return "delivery_failed"
    return "delivered"


class DatabaseHeartbeatStore:
    def __init__(self, engine: Engine, *, retention_per_service: int = 32):
        self.engine = engine
        if retention_per_service <= 0:
            raise ValueError("retention_per_service must be positive")
        self.retention_per_service = retention_per_service

    def record(
        self,
        *,
        service_name: str,
        node_id: str,
        observed_at: str,
        healthy: bool,
        payload: dict[str, Any] | None = None,
    ) -> ServiceHealth:
        service_name = non_empty(service_name, field="service_name")
        node_id = non_empty(node_id, field="node_id")
        observed_at = timestamp(observed_at, field="observed_at")
        stored_payload = json_value(payload or {}, field="payload")
        identity = canonical_hash(
            {
                "service_name": service_name,
                "node_id": node_id,
                "observed_at": observed_at,
                "healthy": bool(healthy),
                "payload": stored_payload,
            }
        ).removeprefix("sha256:")
        with self.engine.begin() as connection:
            heartbeat_id = f"heartbeat:{identity}"
            existing = connection.execute(
                select(service_heartbeat.c.payload).where(service_heartbeat.c.id == heartbeat_id)
            ).scalar_one_or_none()
            if existing is None:
                connection.execute(
                    insert(service_heartbeat).values(
                        id=heartbeat_id,
                        service_name=service_name,
                        node_id=node_id,
                        observed_at=observed_at,
                        healthy=bool(healthy),
                        payload=stored_payload,
                    )
                )
            elif dict(existing) != stored_payload:
                raise ValueError("heartbeat identity collision")
            stale_ids = connection.execute(
                select(service_heartbeat.c.id)
                .where(
                    service_heartbeat.c.service_name == service_name,
                    service_heartbeat.c.node_id == node_id,
                )
                .order_by(service_heartbeat.c.observed_at.desc(), service_heartbeat.c.id.desc())
                .offset(self.retention_per_service)
            ).scalars()
            stale_ids = tuple(stale_ids)
            if stale_ids:
                connection.execute(
                    delete(service_heartbeat).where(service_heartbeat.c.id.in_(stale_ids))
                )
        return ServiceHealth(service_name, node_id, observed_at, bool(healthy), stored_payload)

    def latest(self, *, include_payload: bool = True) -> tuple[ServiceHealth, ...]:
        columns = (
            service_heartbeat.c.service_name,
            service_heartbeat.c.node_id,
            service_heartbeat.c.observed_at,
            service_heartbeat.c.healthy,
        )
        statement = select(
            *columns,
            *([service_heartbeat.c.payload] if include_payload else []),
        ).order_by(
            service_heartbeat.c.observed_at.desc(),
            service_heartbeat.c.healthy.asc(),
            service_heartbeat.c.id.desc(),
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings()
            latest_by_service: dict[tuple[str, str], ServiceHealth] = {}
            for row in rows:
                key = (row["node_id"], row["service_name"])
                if key not in latest_by_service:
                    latest_by_service[key] = ServiceHealth(
                        service_name=row["service_name"],
                        node_id=row["node_id"],
                        observed_at=row["observed_at"],
                        healthy=bool(row["healthy"]),
                        payload=dict(row["payload"]) if include_payload else {},
                    )
        return tuple(latest_by_service[key] for key in sorted(latest_by_service))

    def stale(self, *, now: str, maximum_age_seconds: int) -> tuple[ServiceHealth, ...]:
        if maximum_age_seconds <= 0:
            raise ValueError("maximum_age_seconds must be positive")
        current = dt.datetime.fromisoformat(timestamp(now, field="now"))
        cutoff = current - dt.timedelta(seconds=maximum_age_seconds)
        return tuple(
            heartbeat
            for heartbeat in self.latest()
            if dt.datetime.fromisoformat(heartbeat.observed_at) < cutoff
        )
