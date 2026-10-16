"""Common lifecycle and node-authority checks for long-running services."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.services.config import ORDER_SUBMISSION_SERVICES, PlatformConfig
from src.services.health import DatabaseHeartbeatStore, ServiceHealth
from src.services.release import observed_release

Work = Callable[[], Mapping[str, Any] | None]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ServiceCycle:
    service_name: str
    node_id: str
    healthy: bool
    observed_at: str
    detail: dict[str, Any]


class ServiceRuntime:
    def __init__(
        self,
        *,
        config: PlatformConfig,
        node_id: str,
        service_name: str,
        heartbeat_store: DatabaseHeartbeatStore,
        heartbeat_interval_seconds: float = 0,
    ) -> None:
        self.config = config
        self.node = config.assert_service_assignment(node_id=node_id, service=service_name)
        self.service_name = service_name
        self.heartbeat_store = heartbeat_store
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.last_health: ServiceHealth | None = None

    @property
    def can_submit_orders(self) -> bool:
        return (
            self.service_name in ORDER_SUBMISSION_SERVICES
            and self.node.operating_system == "linux"
            and self.node.production_authority
        )

    def assert_order_submission_authority(self) -> None:
        if not self.can_submit_orders:
            raise PermissionError(
                f"service {self.service_name} on {self.node.node_id} cannot submit orders"
            )

    def run_once(self, work: Work, *, observed_at: str | None = None) -> ServiceCycle:
        try:
            result = work()
            detail = dict(result or {})
            detail.setdefault("reason_code", "cycle_completed")
            healthy = not bool(
                detail.get("error_type")
                or detail.get("error")
                or str(detail["reason_code"]).endswith("_failed")
            )
        except Exception as exc:
            detail = {
                "reason_code": "service_cycle_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            healthy = False
        observed_at = observed_at or utc_now()
        self._record_health(
            service_name=self.service_name,
            node_id=self.node.node_id,
            observed_at=observed_at,
            healthy=healthy,
            payload={**detail, "release": observed_release()},
        )
        return ServiceCycle(
            service_name=self.service_name,
            node_id=self.node.node_id,
            healthy=healthy,
            observed_at=observed_at,
            detail=detail,
        )

    def _record_health(self, *, observed_at, healthy, **fields):
        if self.last_health is not None and self.last_health.healthy == healthy:
            elapsed = (
                dt.datetime.fromisoformat(observed_at)
                - dt.datetime.fromisoformat(self.last_health.observed_at)
            ).total_seconds()
            if 0 <= elapsed < self.heartbeat_interval_seconds:
                return
        self.last_health = self.heartbeat_store.record(
            observed_at=observed_at, healthy=healthy, **fields
        )

    def heartbeat(
        self,
        *,
        observed_at: str | None = None,
        healthy: bool = True,
        payload: dict[str, Any] | None = None,
    ) -> ServiceHealth:
        return self.heartbeat_store.record(
            service_name=self.service_name,
            node_id=self.node.node_id,
            observed_at=observed_at or utc_now(),
            healthy=healthy,
            payload={
                **(payload or {"reason_code": "service_alive"}),
                "release": observed_release(),
            },
        )
