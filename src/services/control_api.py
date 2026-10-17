"""Database-backed control plane and local HTTP API."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import control_event
from src.domain._codec import json_value, non_empty, timestamp, to_primitive
from src.observability.reports import read_persisted_platform_report
from src.services.alerting import AlertSeverity, SqlAlertService
from src.services.config import PLATFORM_PROCESS_SERVICES
from src.services.health import DatabaseHeartbeatStore
from src.services.scheduler import DatabaseJobQueue

if TYPE_CHECKING:
    from src.services.agent_transport import AgentReviewTransport

_HEALTHZ_CACHE_SECONDS = 5.0
_HEALTHZ_FRESHNESS_SECONDS = 60.0
_HEALTHZ_PROCESS_GROUPS = (
    "market-gateway",
    "data-writer",
    "trading-runtime",
    "scheduler-operations",
)
_HEALTHZ_REQUIRED_SERVICES = tuple(
    sorted(
        {
            service
            for group in _HEALTHZ_PROCESS_GROUPS
            for service in PLATFORM_PROCESS_SERVICES[group]
            if service != "report-worker"
        }
        | {"research-worker", "evaluation-worker"}
    )
)
_HEALTHZ_REPORT_REASONS = {
    "missing": "report_missing",
    "stale": "report_stale",
    "invalid": "report_invalid",
}


@dataclass(frozen=True)
class ControlState:
    target: str
    paused: bool
    reason_code: str
    changed_at: str
    mode: str = "run"


class ControlMode(StrEnum):
    RUN = "run"
    BLOCK_NEW_RISK = "block_new_risk"
    MANAGEMENT_ONLY = "management_only"
    EMERGENCY_FLATTEN = "emergency_flatten"
    SUSPENDED = "suspended"


class DatabaseControlPlane:
    def __init__(
        self,
        engine: Engine,
        heartbeat_store: DatabaseHeartbeatStore,
        *,
        configuration: dict[str, dict[str, Any]] | None = None,
        alerts: SqlAlertService | None = None,
        report_root: Path | None = None,
    ):
        self.engine = engine
        self.heartbeat_store = heartbeat_store
        self.configuration = json_value(configuration or {}, field="control configuration")
        self.alerts = alerts
        self.report_root = Path(report_root) if report_root is not None else None
        self._healthz_lock = threading.Lock()
        self._healthz_cache: dict[str, Any] | None = None
        self._healthz_cached_at = 0.0
        self._healthz_cached_node_id: str | None = None

    def healthz(self, *, node_id: str, now: str | None = None) -> dict[str, Any]:
        with self._healthz_lock:
            current = monotonic()
            if (
                self._healthz_cache is not None
                and self._healthz_cached_node_id == node_id
                and current - self._healthz_cached_at < _HEALTHZ_CACHE_SECONDS
            ):
                return _copy_healthz_result(self._healthz_cache)
            try:
                checked_at = timestamp(now or datetime.now(UTC).isoformat(), field="checked_at")
                result = self._build_healthz(checked_at, node_id=node_id)
            except Exception:
                checked_at = timestamp(datetime.now(UTC).isoformat(), field="checked_at")
                result = _healthz_result(
                    checked_at,
                    {"health_check_unavailable"},
                    set(_HEALTHZ_REQUIRED_SERVICES),
                )
            self._healthz_cache = result
            self._healthz_cached_at = monotonic()
            self._healthz_cached_node_id = node_id
            return _copy_healthz_result(result)

    def _build_healthz(self, checked_at: str, *, node_id: str) -> dict[str, Any]:
        current = datetime.fromisoformat(checked_at)
        reasons, unhealthy_services = _heartbeat_health(
            self.heartbeat_store, current=current, node_id=node_id
        )
        snapshot = read_persisted_platform_report(self.report_root, now=checked_at)
        reasons.update(_report_health_reasons(snapshot))
        return _healthz_result(checked_at, reasons, unhealthy_services)

    def set_paused(
        self,
        *,
        target: str,
        paused: bool,
        reason_code: str,
        requested_by: str,
        changed_at: str,
    ) -> ControlState:
        target = non_empty(target, field="target")
        reason_code = non_empty(reason_code, field="reason_code")
        requested_by = non_empty(requested_by, field="requested_by")
        changed_at = timestamp(changed_at, field="changed_at")
        return self.set_mode(
            target=target,
            mode=ControlMode.MANAGEMENT_ONLY if paused else ControlMode.RUN,
            reason_code=reason_code,
            requested_by=requested_by,
            changed_at=changed_at,
            confirm_resume=not paused,
        )

    def set_mode(
        self,
        *,
        target: str,
        mode: ControlMode | str,
        reason_code: str,
        requested_by: str,
        changed_at: str,
        confirm_resume: bool = False,
    ) -> ControlState:
        target = _control_target(target)
        mode = ControlMode(mode)
        reason_code = non_empty(reason_code, field="reason_code")
        requested_by = non_empty(requested_by, field="requested_by")
        changed_at = timestamp(changed_at, field="changed_at")
        if mode is ControlMode.RUN and not confirm_resume:
            raise PermissionError("explicit resume confirmation is required")
        payload = json_value(
            {
                "target": target,
                "action": "resume" if mode is ControlMode.RUN else "mode_change",
                "mode": mode.value,
                "reason_code": reason_code,
                "requested_by": requested_by,
            },
            field="payload",
        )
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            + changed_at.encode()
        ).hexdigest()
        control_id = f"control:{digest}"
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(control_event.c.payload).where(control_event.c.id == control_id)
            ).scalar_one_or_none()
            if existing is None:
                connection.execute(
                    insert(control_event).values(
                        id=control_id,
                        created_at=changed_at,
                        payload=payload,
                    )
                )
        if mode is ControlMode.EMERGENCY_FLATTEN:
            self._enqueue_control_job(
                name="emergency_flatten",
                control_id=control_id,
                target=target,
                reason_code=reason_code,
                requested_by=requested_by,
                changed_at=changed_at,
            )
        if self.alerts is not None:
            try:
                self.alerts.emit(
                    event_type="control_mode_changed",
                    severity=(
                        AlertSeverity.CRITICAL
                        if mode is ControlMode.EMERGENCY_FLATTEN
                        else AlertSeverity.WARNING
                    ),
                    dedupe_key=f"control:{control_id}",
                    target=target,
                    message=f"control mode changed to {mode.value}",
                    emitted_at=changed_at,
                    payload={
                        "mode": mode.value,
                        "reason_code": reason_code,
                        "requested_by": requested_by,
                    },
                    cooldown_seconds=0,
                )
            except Exception:
                pass
        return ControlState(
            target, mode is not ControlMode.RUN, reason_code, changed_at, mode.value
        )

    def cancel_all_entry_orders(
        self,
        *,
        target: str,
        reason_code: str,
        requested_by: str,
        changed_at: str,
    ) -> dict[str, Any]:
        state = self.set_mode(
            target=target,
            mode=ControlMode.BLOCK_NEW_RISK,
            reason_code=reason_code,
            requested_by=requested_by,
            changed_at=changed_at,
        )
        control_id = (
            "control-cancel:"
            + hashlib.sha256(
                f"{state.target}|{state.changed_at}|{requested_by}".encode()
            ).hexdigest()
        )
        self._enqueue_control_job(
            name="cancel_entry_orders",
            control_id=control_id,
            target=state.target,
            reason_code=reason_code,
            requested_by=requested_by,
            changed_at=state.changed_at,
        )
        return {**state.__dict__, "control_id": control_id, "job_name": "cancel_entry_orders"}

    def _enqueue_control_job(
        self,
        *,
        name: str,
        control_id: str,
        target: str,
        reason_code: str,
        requested_by: str,
        changed_at: str,
    ) -> None:
        payload = {
            "control_id": control_id,
            "target": target,
            "reason_code": reason_code,
            "requested_by": requested_by,
            "changed_at": changed_at,
            "producer_identity": "database-control-plane",
        }
        job_id = f"control:{name}:{hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}"
        DatabaseJobQueue(self.engine).enqueue_if_absent(
            job_id=job_id,
            name=name,
            payload=payload,
            available_at=changed_at,
            priority=250,
        )

    def states(self) -> tuple[ControlState, ...]:
        latest: dict[str, ControlState] = {}
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(control_event).order_by(control_event.c.created_at.desc())
            ).mappings()
            for row in rows:
                payload = dict(row["payload"])
                target = str(payload.get("target") or "")
                action = payload.get("action")
                if (
                    target
                    and target not in latest
                    and (action in {"pause", "resume", "mode_change"})
                ):
                    mode = str(payload.get("mode") or "")
                    if not mode:
                        mode = (
                            ControlMode.MANAGEMENT_ONLY.value
                            if action == "pause"
                            else ControlMode.RUN.value
                        )
                    latest[target] = ControlState(
                        target=target,
                        paused=mode != ControlMode.RUN.value,
                        reason_code=str(payload.get("reason_code") or "unknown"),
                        changed_at=row["created_at"],
                        mode=mode,
                    )
        return tuple(latest[key] for key in sorted(latest))

    def is_paused(self, target: str) -> bool:
        return any(state.target == target and state.paused for state in self.states())

    def effective_mode(self, *, product_id: str, strategy_id: str | None = None) -> ControlMode:
        candidates = {"global", product_id, f"product:{product_id}"}
        if strategy_id:
            candidates.add(f"strategy:{strategy_id}")
        modes = [ControlMode(state.mode) for state in self.states() if state.target in candidates]
        return max(modes, key=_mode_rank, default=ControlMode.RUN)

    def blocks_new_risk(self, *, product_id: str, strategy_id: str | None = None) -> bool:
        return (
            self.effective_mode(product_id=product_id, strategy_id=strategy_id)
            is not ControlMode.RUN
        )

    def service_is_paused(self, service: str) -> bool:
        if service in _CRITICAL_SERVICES:
            return False
        return self.is_paused(service) or self.is_paused(f"service:{service}")

    def status(self, *, now: str | None = None) -> dict[str, Any]:
        snapshot = read_persisted_platform_report(self.report_root, now=now)
        report = snapshot.get("report")
        response = dict(report) if isinstance(report, Mapping) else {}
        operations = response.get("operations")
        operations = dict(operations) if isinstance(operations, Mapping) else {}
        operations["heartbeats"] = [
            {
                "service_name": heartbeat.service_name,
                "node_id": heartbeat.node_id,
                "observed_at": heartbeat.observed_at,
                "healthy": heartbeat.healthy,
                "payload": dict(heartbeat.payload),
            }
            for heartbeat in self.heartbeat_store.latest()
        ]
        response["operations"] = operations
        return {
            **response,
            "schema": "platform.control_status/v1",
            "controls": [state.__dict__ for state in self.states()],
            "report_snapshot": {key: value for key, value in snapshot.items() if key != "report"},
        }

    def configuration_view(self) -> dict[str, Any]:
        return {
            "schema": "platform.configuration/v1",
            "configuration": _redact_configuration(self.configuration),
        }

    def report(self, *, now: str | None = None) -> dict[str, Any]:
        snapshot = read_persisted_platform_report(self.report_root, now=now)
        return {
            "schema": "platform.report/v1",
            "report": snapshot.get("report"),
            "report_snapshot": {key: value for key, value in snapshot.items() if key != "report"},
        }

    def agent_reviews(self) -> dict[str, Any]:
        from src.agents.store import SqlAgentStore

        return {
            "schema": "platform.agent_reviews/v1",
            "reviews": SqlAgentStore(self.engine).records("review")[-100:],
        }

    def ingest_agent_proposal(self, payload: dict[str, Any]) -> dict[str, Any]:
        from src.agents.openclaw_bridge import OpenClawAgentBridge
        from src.agents.store import SqlAgentStore

        proposal = OpenClawAgentBridge(
            store=SqlAgentStore(self.engine),
            queue=DatabaseJobQueue(self.engine),
        ).ingest(payload)
        return {
            "schema": "platform.agent_intake/v1",
            "proposal": to_primitive(proposal),
            "content_hash": proposal.content_hash,
        }

    def restart_forward_window(self, payload: dict[str, Any]) -> dict[str, Any]:
        from src.services.forward_windows import restart_operator_window
        from src.services.runtime import utc_now

        window = restart_operator_window(self.engine, payload, now=utc_now())
        return {"window_id": window["window_id"], "opened_at": window["opened_at"]}

    def forward_windows(self) -> dict[str, Any]:
        from src.data.database import forward_qualification_window

        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    select(
                        forward_qualification_window.c.id, forward_qualification_window.c.payload
                    )
                    .order_by(
                        forward_qualification_window.c.opened_at.desc(),
                        forward_qualification_window.c.id.desc(),
                    )
                    .limit(100)
                )
                .mappings()
                .all()
            )
        return {
            "schema": "platform.forward_windows/v1",
            "windows": [{"window_id": row["id"], "payload": row["payload"]} for row in rows],
        }


_CRITICAL_SERVICES = frozenset(
    {
        "market-gateway",
        "account-reconciliation",
        "execution-engine",
        "paper-engine",
        "control-api",
        "platform-scheduler",
    }
)


def _mode_rank(mode: ControlMode) -> int:
    return {
        ControlMode.RUN: 0,
        ControlMode.BLOCK_NEW_RISK: 1,
        ControlMode.MANAGEMENT_ONLY: 2,
        ControlMode.SUSPENDED: 3,
        ControlMode.EMERGENCY_FLATTEN: 4,
    }[mode]


def _control_target(target: str) -> str:
    clean = non_empty(target, field="target")
    if clean == "global" or clean.startswith(("product:", "strategy:", "service:")):
        return clean
    return clean


def _healthz_result(
    checked_at: str, reason_codes: set[str], unhealthy_services: set[str]
) -> dict[str, Any]:
    return {
        "schema": "platform.healthz/v1",
        "ok": not reason_codes,
        "checked_at": checked_at,
        "reason_codes": sorted(reason_codes),
        "unhealthy_services": sorted(unhealthy_services),
    }


def _copy_healthz_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(result),
        "reason_codes": list(result["reason_codes"]),
        "unhealthy_services": list(result["unhealthy_services"]),
    }


def _heartbeat_health(
    store: DatabaseHeartbeatStore, *, current: datetime, node_id: str
) -> tuple[set[str], set[str]]:
    latest_by_service: dict[str, tuple[datetime, bool]] = {}
    for heartbeat in store.latest(include_payload=False):
        service = heartbeat.service_name
        if heartbeat.node_id != node_id or service not in _HEALTHZ_REQUIRED_SERVICES:
            continue
        observed = datetime.fromisoformat(
            timestamp(heartbeat.observed_at, field="heartbeat observed_at")
        )
        previous = latest_by_service.get(service)
        if (
            previous is None
            or observed > previous[0]
            or (observed == previous[0] and not heartbeat.healthy)
        ):
            latest_by_service[service] = (observed, heartbeat.healthy)

    reasons: set[str] = set()
    unhealthy_services: set[str] = set()
    for service in _HEALTHZ_REQUIRED_SERVICES:
        latest = latest_by_service.get(service)
        if latest is None:
            reasons.add("heartbeat_missing")
            unhealthy_services.add(service)
            continue
        observed, healthy = latest
        age = (current - observed).total_seconds()
        if age < 0 or age > _HEALTHZ_FRESHNESS_SECONDS:
            reasons.add("heartbeat_stale")
            unhealthy_services.add(service)
        elif not healthy:
            reasons.add("heartbeat_unhealthy")
            unhealthy_services.add(service)
    return reasons, unhealthy_services


def _report_health_reasons(snapshot: Mapping[str, Any]) -> set[str]:
    status = snapshot.get("snapshot_status")
    if status != "fresh":
        return {_HEALTHZ_REPORT_REASONS.get(str(status), "report_invalid")}
    report = snapshot.get("report")
    operations = report.get("operations") if isinstance(report, Mapping) else None
    if not isinstance(operations, Mapping):
        return {"report_invalid"}
    return _report_operations_health_reasons(operations)


def _report_operations_health_reasons(operations: Mapping[str, Any]) -> set[str]:
    reasons = _report_resource_reasons(operations)
    slis = operations.get("slis")
    if not isinstance(slis, Mapping):
        reasons.add("report_invalid")
        return reasons
    reasons.update(_report_sli_reasons(operations, slis))
    return reasons


def _report_resource_reasons(operations: Mapping[str, Any]) -> set[str]:
    reasons: set[str] = set()
    for section, missing, invalid, unhealthy in (
        ("backups", "backup_status_missing", "backup_status_invalid", "backup_unhealthy"),
        ("disk", "disk_status_missing", "disk_status_invalid", "disk_unhealthy"),
    ):
        value = operations.get(section)
        if value is None:
            reasons.add(missing)
        elif not isinstance(value, Mapping) or not isinstance(value.get("healthy"), bool):
            reasons.add(invalid)
        elif value["healthy"] is False:
            reasons.add(unhealthy)
    return reasons


def _report_sli_reasons(operations: Mapping[str, Any], slis: Mapping[str, Any]) -> set[str]:
    reasons: set[str] = set()
    _add_count_reason(reasons, slis.get("unresolved_recovery_count"), "recovery_unresolved")
    for name, reason in (
        ("stale_account_authority", "account_authority_stale"),
        ("stale_market_data", "market_data_stale"),
        ("missing_risk_data", "risk_data_missing"),
    ):
        value = slis.get(name)
        if not isinstance(value, Mapping):
            reasons.add("report_invalid")
        else:
            _add_count_reason(reasons, value.get("count"), reason)
    conflicts = slis.get("execution_authority_conflicts")
    if not isinstance(conflicts, Mapping) or not isinstance(conflicts.get("conflict"), bool):
        reasons.add("report_invalid")
    elif conflicts["conflict"]:
        reasons.add("execution_authority_conflict")
    _add_count_reason(
        reasons, operations.get("alert_delivery_failure_count"), "alert_delivery_failed"
    )
    return reasons


def _add_count_reason(reasons: set[str], value: Any, reason: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        reasons.add("report_invalid")
    elif value < 0:
        reasons.add("report_invalid")
    elif value > 0:
        reasons.add(reason)


class ControlHTTPServer(ThreadingHTTPServer):
    control_plane: DatabaseControlPlane
    bearer_token: str
    node_id: str
    agent_transport: AgentReviewTransport | None = None


class ControlRequestHandler(BaseHTTPRequestHandler):
    """Authenticated HTTP adapter around one database control plane."""

    def _authorised(self) -> bool:
        server = cast(ControlHTTPServer, self.server)
        return hmac.compare_digest(
            self.headers.get("Authorization", ""), f"Bearer {server.bearer_token}"
        )

    def _agent_post(self) -> None:
        from src.services.runtime import utc_now

        transport = cast(ControlHTTPServer, self.server).agent_transport
        if transport is None or not hmac.compare_digest(
            self.headers.get("Authorization", ""), f"Bearer {transport.key.hex()}"
        ):
            self._reply(HTTPStatus.UNAUTHORIZED, {"reason_code": "unauthorised"})
            return
        try:
            payload = _request_payload(self)
            if self.path == "/agent/requests/next":
                if payload:
                    raise ValueError("request polling takes no arguments")
                response = transport.next_request(now=utc_now())
            else:
                response = transport.respond(payload, now=utc_now())
        except (UnicodeDecodeError, PermissionError, ValueError, KeyError) as exc:
            self._reply(
                HTTPStatus.BAD_REQUEST, {"reason_code": "invalid_request", "error": str(exc)}
            )
            return
        self._reply(HTTPStatus.OK, response)

    def _reply(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            if not _is_loopback_peer(self.client_address):
                self._reply(HTTPStatus.FORBIDDEN, {"reason_code": "loopback_only"})
                return
            server = cast(ControlHTTPServer, self.server)
            response = server.control_plane.healthz(node_id=server.node_id)
            self._reply(
                HTTPStatus.OK if response["ok"] else HTTPStatus.SERVICE_UNAVAILABLE,
                response,
            )
            return
        if not self._authorised():
            self._reply(HTTPStatus.UNAUTHORIZED, {"reason_code": "unauthorised"})
            return
        route_response = _get_route(cast(ControlHTTPServer, self.server).control_plane, self.path)
        if route_response is None:
            self._reply(HTTPStatus.NOT_FOUND, {"reason_code": "not_found"})
            return
        self._reply(HTTPStatus.OK, route_response)

    def do_POST(self) -> None:  # noqa: N802
        if self.path in {"/agent/requests/next", "/agent/responses"}:
            self._agent_post()
            return
        if not self._authorised():
            self._reply(HTTPStatus.UNAUTHORIZED, {"reason_code": "unauthorised"})
            return
        if self.path not in _POST_ROUTES:
            self._reply(HTTPStatus.NOT_FOUND, {"reason_code": "not_found"})
            return
        try:
            response = _post_route(
                cast(ControlHTTPServer, self.server).control_plane,
                self.path,
                _request_payload(self),
            )
        except (UnicodeDecodeError, PermissionError, ValueError, json.JSONDecodeError) as exc:
            self._reply(
                HTTPStatus.BAD_REQUEST,
                {"reason_code": "invalid_request", "error": str(exc)},
            )
            return
        self._reply(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: object) -> None:
        return


def build_control_server(
    *,
    bind: tuple[str, int],
    control_plane: DatabaseControlPlane,
    bearer_token: str,
    agent_transport_key: bytes | None = None,
    node_id: str = "agent-transport",
) -> ControlHTTPServer:
    if not bearer_token:
        raise ValueError("control API bearer token cannot be empty")
    transport = None
    if agent_transport_key is not None:
        from src.services.agent_transport import AgentReviewTransport

        if hmac.compare_digest(bearer_token, agent_transport_key.hex()):
            raise ValueError("agent and operator credentials must be distinct")
        transport = AgentReviewTransport(
            DatabaseJobQueue(control_plane.engine), key=agent_transport_key, node_id=node_id
        )
    server = ControlHTTPServer(bind, ControlRequestHandler)
    server.control_plane = control_plane
    server.bearer_token = bearer_token
    server.node_id = node_id
    server.agent_transport = transport
    return server


_POST_ROUTES = frozenset(
    {
        "/pause",
        "/resume",
        "/block-new-risk",
        "/management-only",
        "/suspend-strategy",
        "/emergency-flatten",
        "/cancel-all-entry-orders",
        "/agent/proposals",
        "/paper/restart-forward-window",
    }
)


def _get_route(control_plane: DatabaseControlPlane, path: str) -> dict[str, Any] | None:
    routes = {
        "/health": control_plane.status,
        "/status": control_plane.status,
        "/configuration": control_plane.configuration_view,
        "/reports": control_plane.report,
        "/agent/reviews": control_plane.agent_reviews,
        "/paper/windows": control_plane.forward_windows,
    }
    handler = routes.get(path)
    return handler() if handler is not None else None


def _is_loopback_peer(address: tuple[str, int]) -> bool:
    try:
        peer = ipaddress.ip_address(address[0])
    except ValueError:
        return False
    if isinstance(peer, ipaddress.IPv6Address) and peer.ipv4_mapped is not None:
        peer = peer.ipv4_mapped
    return peer.is_loopback


def _request_payload(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    size = int(handler.headers.get("Content-Length", "0"))
    if not 1 <= size <= 16_384:
        raise ValueError("request size is invalid")
    payload = json.loads(handler.rfile.read(size))
    if not isinstance(payload, dict):
        raise ValueError("request must be an object")
    return payload


def _post_route(
    control_plane: DatabaseControlPlane, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    if path == "/agent/proposals":
        return control_plane.ingest_agent_proposal(payload)
    if path == "/paper/restart-forward-window":
        return control_plane.restart_forward_window(payload)
    target = _post_target(path, payload)
    if path == "/cancel-all-entry-orders":
        return control_plane.cancel_all_entry_orders(
            target=target,
            reason_code=str(payload.get("reason_code") or ""),
            requested_by=str(payload.get("requested_by") or ""),
            changed_at=str(payload.get("changed_at") or ""),
        )
    return control_plane.set_mode(
        target=target,
        mode=_post_mode(path),
        reason_code=str(payload.get("reason_code") or ""),
        requested_by=str(payload.get("requested_by") or ""),
        changed_at=str(payload.get("changed_at") or ""),
        confirm_resume=bool(payload.get("confirm_resume", False)),
    ).__dict__


def _post_target(path: str, payload: Mapping[str, Any]) -> str:
    if path != "/suspend-strategy":
        return str(payload.get("target") or "")
    strategy_id = str(payload.get("strategy_id") or "").strip()
    if not strategy_id:
        raise ValueError("strategy_id is required")
    return f"strategy:{strategy_id}"


def _post_mode(path: str) -> ControlMode:
    return {
        "/pause": ControlMode.MANAGEMENT_ONLY,
        "/resume": ControlMode.RUN,
        "/block-new-risk": ControlMode.BLOCK_NEW_RISK,
        "/management-only": ControlMode.MANAGEMENT_ONLY,
        "/suspend-strategy": ControlMode.SUSPENDED,
        "/emergency-flatten": ControlMode.EMERGENCY_FLATTEN,
    }[path]


def _redact_configuration(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in ("password", "secret", "token", "credential")):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(child_key): _redact_configuration(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_configuration(item) for item in value]
    return value
