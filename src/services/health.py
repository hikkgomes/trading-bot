"""Durable service heartbeats for platform health and control."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import service_heartbeat
from src.domain._codec import json_value, non_empty, timestamp


@dataclass(frozen=True)
class ServiceHealth:
    service_name: str
    node_id: str
    observed_at: str
    healthy: bool
    payload: dict[str, Any]


class DatabaseHeartbeatStore:
    def __init__(self, engine: Engine):
        self.engine = engine

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
        identity = uuid.uuid4().hex
        with self.engine.begin() as connection:
            connection.execute(
                insert(service_heartbeat).values(
                    id=f"heartbeat:{identity}",
                    service_name=service_name,
                    node_id=node_id,
                    observed_at=observed_at,
                    healthy=bool(healthy),
                    payload=stored_payload,
                )
            )
        return ServiceHealth(service_name, node_id, observed_at, bool(healthy), stored_payload)

    def latest(self) -> tuple[ServiceHealth, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(service_heartbeat).order_by(service_heartbeat.c.observed_at.desc())
            ).mappings()
            latest_by_service: dict[tuple[str, str], ServiceHealth] = {}
            for row in rows:
                key = (row["node_id"], row["service_name"])
                if key not in latest_by_service:
                    latest_by_service[key] = ServiceHealth(
                        service_name=row["service_name"],
                        node_id=row["node_id"],
                        observed_at=row["observed_at"],
                        healthy=bool(row["healthy"]),
                        payload=dict(row["payload"]),
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
