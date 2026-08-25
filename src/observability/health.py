"""Aggregate service and platform health from authoritative state."""

from __future__ import annotations

from dataclasses import dataclass

from src.services.config import PlatformConfig
from src.services.health import DatabaseHeartbeatStore, ServiceHealth


@dataclass(frozen=True)
class PlatformHealth:
    healthy: bool
    reason_code: str
    missing_services: tuple[str, ...]
    stale_services: tuple[str, ...]
    unhealthy_services: tuple[str, ...]
    heartbeats: tuple[ServiceHealth, ...]


def assess_platform_health(
    *,
    config: PlatformConfig,
    store: DatabaseHeartbeatStore,
    now: str,
    maximum_age_seconds: int,
) -> PlatformHealth:
    heartbeats = store.latest()
    expected = {(node.node_id, service) for node in config.nodes for service in node.services}
    present = {(item.node_id, item.service_name) for item in heartbeats}
    missing = tuple(f"{node}:{service}" for node, service in sorted(expected - present))
    stale_records = store.stale(now=now, maximum_age_seconds=maximum_age_seconds)
    stale = tuple(
        f"{item.node_id}:{item.service_name}"
        for item in sorted(stale_records, key=lambda item: (item.node_id, item.service_name))
    )
    unhealthy = tuple(
        f"{item.node_id}:{item.service_name}" for item in heartbeats if not item.healthy
    )
    if missing:
        reason = "service_heartbeats_missing"
    elif stale:
        reason = "service_heartbeats_stale"
    elif unhealthy:
        reason = "service_unhealthy"
    else:
        reason = "platform_healthy"
    return PlatformHealth(
        healthy=not (missing or stale or unhealthy),
        reason_code=reason,
        missing_services=missing,
        stale_services=stale,
        unhealthy_services=unhealthy,
        heartbeats=heartbeats,
    )
