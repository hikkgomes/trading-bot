"""Database-backed control plane and local HTTP API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import control_event
from src.domain._codec import json_value, non_empty, timestamp, to_primitive
from src.observability.reports import DatabasePlatformReport
from src.services.alerting import AlertSeverity, SqlAlertService
from src.services.health import DatabaseHeartbeatStore
from src.services.scheduler import DatabaseJobQueue


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
    ):
        self.engine = engine
        self.heartbeat_store = heartbeat_store
        self.configuration = json_value(configuration or {}, field="control configuration")
        self.alerts = alerts

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

    def status(self) -> dict[str, Any]:
        report = DatabasePlatformReport(self.engine).build()
        return {
            **report,
            "schema": "platform.control_status/v1",
            "controls": [state.__dict__ for state in self.states()],
        }

    def configuration_view(self) -> dict[str, Any]:
        return {
            "schema": "platform.configuration/v1",
            "configuration": _redact_configuration(self.configuration),
        }

    def report(self) -> dict[str, Any]:
        return {
            "schema": "platform.report/v1",
            "report": DatabasePlatformReport(self.engine).build(),
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


def build_control_server(
    *,
    bind: tuple[str, int],
    control_plane: DatabaseControlPlane,
    bearer_token: str,
) -> ThreadingHTTPServer:
    if not bearer_token:
        raise ValueError("control API bearer token cannot be empty")

    class Handler(BaseHTTPRequestHandler):
        def _authorised(self) -> bool:
            return self.headers.get("Authorization") == f"Bearer {bearer_token}"

        def _reply(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorised():
                self._reply(HTTPStatus.UNAUTHORIZED, {"reason_code": "unauthorised"})
                return
            routes = {
                "/health": control_plane.status,
                "/status": control_plane.status,
                "/configuration": control_plane.configuration_view,
                "/reports": control_plane.report,
                "/agent/reviews": control_plane.agent_reviews,
            }
            handler = routes.get(self.path)
            if handler is None:
                self._reply(HTTPStatus.NOT_FOUND, {"reason_code": "not_found"})
                return
            self._reply(HTTPStatus.OK, handler())

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorised():
                self._reply(HTTPStatus.UNAUTHORIZED, {"reason_code": "unauthorised"})
                return
            if self.path not in {
                "/pause",
                "/resume",
                "/block-new-risk",
                "/management-only",
                "/suspend-strategy",
                "/emergency-flatten",
                "/cancel-all-entry-orders",
                "/agent/proposals",
            }:
                self._reply(HTTPStatus.NOT_FOUND, {"reason_code": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 16_384:
                    raise ValueError("request size is invalid")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("request must be an object")
                if self.path == "/agent/proposals":
                    response = control_plane.ingest_agent_proposal(payload)
                else:
                    target = str(payload.get("target") or "")
                    if self.path == "/suspend-strategy":
                        strategy_id = str(payload.get("strategy_id") or "").strip()
                        if not strategy_id:
                            raise ValueError("strategy_id is required")
                        target = f"strategy:{strategy_id}"
                    if self.path == "/cancel-all-entry-orders":
                        response = control_plane.cancel_all_entry_orders(
                            target=target,
                            reason_code=str(payload.get("reason_code") or ""),
                            requested_by=str(payload.get("requested_by") or ""),
                            changed_at=str(payload.get("changed_at") or ""),
                        )
                    else:
                        mode = {
                            "/pause": ControlMode.MANAGEMENT_ONLY,
                            "/resume": ControlMode.RUN,
                            "/block-new-risk": ControlMode.BLOCK_NEW_RISK,
                            "/management-only": ControlMode.MANAGEMENT_ONLY,
                            "/suspend-strategy": ControlMode.SUSPENDED,
                            "/emergency-flatten": ControlMode.EMERGENCY_FLATTEN,
                        }[self.path]
                        response = control_plane.set_mode(
                            target=target,
                            mode=mode,
                            reason_code=str(payload.get("reason_code") or ""),
                            requested_by=str(payload.get("requested_by") or ""),
                            changed_at=str(payload.get("changed_at") or ""),
                            confirm_resume=bool(payload.get("confirm_resume", False)),
                        ).__dict__
            except (UnicodeDecodeError, PermissionError, ValueError, json.JSONDecodeError) as exc:
                self._reply(
                    HTTPStatus.BAD_REQUEST,
                    {"reason_code": "invalid_request", "error": str(exc)},
                )
                return
            self._reply(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(bind, Handler)


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
