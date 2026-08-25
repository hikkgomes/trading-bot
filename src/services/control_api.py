"""Database-backed control plane and local HTTP API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import control_event
from src.domain._codec import json_value, non_empty, timestamp, to_primitive
from src.observability.reports import DatabasePlatformReport
from src.services.health import DatabaseHeartbeatStore
from src.services.scheduler import DatabaseJobQueue


@dataclass(frozen=True)
class ControlState:
    target: str
    paused: bool
    reason_code: str
    changed_at: str


class DatabaseControlPlane:
    def __init__(
        self,
        engine: Engine,
        heartbeat_store: DatabaseHeartbeatStore,
        *,
        configuration: dict[str, dict[str, Any]] | None = None,
    ):
        self.engine = engine
        self.heartbeat_store = heartbeat_store
        self.configuration = json_value(configuration or {}, field="control configuration")

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
        payload = json_value(
            {
                "target": target,
                "action": "pause" if paused else "resume",
                "reason_code": reason_code,
                "requested_by": requested_by,
            },
            field="payload",
        )
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            + changed_at.encode()
        ).hexdigest()
        with self.engine.begin() as connection:
            connection.execute(
                insert(control_event).values(
                    id=f"control:{digest}",
                    created_at=changed_at,
                    payload=payload,
                )
            )
        return ControlState(target, paused, reason_code, changed_at)

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
                if target and target not in latest and action in {"pause", "resume"}:
                    latest[target] = ControlState(
                        target=target,
                        paused=action == "pause",
                        reason_code=str(payload.get("reason_code") or "unknown"),
                        changed_at=row["created_at"],
                    )
        return tuple(latest[key] for key in sorted(latest))

    def is_paused(self, target: str) -> bool:
        return any(state.target == target and state.paused for state in self.states())

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
            if self.path not in {"/pause", "/resume", "/agent/proposals"}:
                self._reply(HTTPStatus.NOT_FOUND, {"reason_code": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 16_384:
                    raise ValueError("request size is invalid")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("request must be an object")
                response = (
                    control_plane.ingest_agent_proposal(payload)
                    if self.path == "/agent/proposals"
                    else control_plane.set_paused(
                        target=str(payload.get("target") or ""),
                        paused=self.path == "/pause",
                        reason_code=str(payload.get("reason_code") or ""),
                        requested_by=str(payload.get("requested_by") or ""),
                        changed_at=str(payload.get("changed_at") or ""),
                    ).__dict__
                )
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
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
