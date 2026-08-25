"""Common lifecycle and node-authority checks for long-running services."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.services.config import ORDER_SUBMISSION_SERVICES, PlatformConfig
from src.services.health import DatabaseHeartbeatStore, ServiceHealth

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
    ) -> None:
        self.config = config
        self.node = config.assert_service_assignment(node_id=node_id, service=service_name)
        self.service_name = service_name
        self.heartbeat_store = heartbeat_store

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
        observed_at = observed_at or utc_now()
        try:
            result = work()
            detail = dict(result or {})
            detail.setdefault("reason_code", "cycle_completed")
            healthy = True
        except Exception as exc:
            detail = {
                "reason_code": "service_cycle_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            healthy = False
        self.heartbeat_store.record(
            service_name=self.service_name,
            node_id=self.node.node_id,
            observed_at=observed_at,
            healthy=healthy,
            payload=detail,
        )
        return ServiceCycle(
            service_name=self.service_name,
            node_id=self.node.node_id,
            healthy=healthy,
            observed_at=observed_at,
            detail=detail,
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
            payload=payload or {"reason_code": "service_alive"},
        )
