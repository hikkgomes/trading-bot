"""Local, bounded IPC broker for tool-free OpenClaw research turns."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pwd
import selectors
import signal
import socket
import socketserver
import sqlite3
import stat
import struct
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from src.agents.context import AgentContext
from src.agents.openclaw_bridge import build_agent_proposal
from src.domain._codec import canonical_hash, timestamp

REQUEST_SCHEMA = "openclaw.model_broker_request/v1"
MAX_REQUEST_BYTES = 1_048_576
MAX_MESSAGE_BYTES = 120_000
MAX_OUTPUT_BYTES = 1_048_576
MAX_PROMPT_BYTES = 16_384
MAX_TURN_SECONDS = 480
MAX_NEW_ATTEMPTS_PER_HOUR = 4
_REQUIRED_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "state",
        "review_id",
        "proposal_id",
        "context",
        "context_hash",
        "transport",
        "response_required",
        "requested_at",
        "expires_at",
    }
)


class ModelBrokerError(ValueError):
    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


def peer_credentials_available() -> bool:
    return hasattr(socket, "SO_PEERCRED") or hasattr(socket.socket, "getpeereid")


def _peer_uid(connection: socket.socket) -> int:
    if hasattr(connection, "getpeereid"):
        return int(connection.getpeereid()[0])
    if hasattr(socket, "SO_PEERCRED"):
        credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        _pid, uid, _gid = struct.unpack("3i", credentials)
        return uid
    raise ModelBrokerError("peer_credentials_unavailable")


def _read_exact(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = connection.recv(length - len(chunks))
        if not chunk:
            raise ModelBrokerError("incomplete_frame")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_frame(connection: socket.socket, *, maximum_bytes: int) -> dict[str, Any]:
    length = struct.unpack("!I", _read_exact(connection, 4))[0]
    if length > maximum_bytes:
        raise ModelBrokerError("request_too_large")
    value = json.loads(_read_exact(connection, length))
    if not isinstance(value, dict):
        raise ModelBrokerError("invalid_request")
    return value


def _write_frame(
    connection: socket.socket,
    value: Mapping[str, Any],
    *,
    maximum_bytes: int,
    too_large_reason: str = "response_too_large",
) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(data) > maximum_bytes:
        raise ModelBrokerError(too_large_reason)
    connection.sendall(struct.pack("!I", len(data)) + data)


def request_model_proposal(
    socket_path: Path, request: dict[str, Any], *, timeout: int
) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("sealed agent request must be an object")
    envelope = {"schema": REQUEST_SCHEMA, "request": request}
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try:
        connection.connect(str(socket_path))
        greeting = _read_frame(connection, maximum_bytes=MAX_OUTPUT_BYTES)
        if greeting.get("state") != "ready":
            raise ModelBrokerError(str(greeting.get("reason_code") or "broker_failed"))
        _write_frame(
            connection,
            envelope,
            maximum_bytes=MAX_REQUEST_BYTES,
            too_large_reason="request_too_large",
        )
        result = _read_frame(connection, maximum_bytes=MAX_OUTPUT_BYTES)
    finally:
        connection.close()
    if result.get("state") != "proposal" or not isinstance(result.get("proposal"), dict):
        raise ModelBrokerError(str(result.get("reason_code") or "broker_failed"))
    return result["proposal"]


class _BrokerLedger:
    def __init__(self, path: Path):
        expanded = path.expanduser()
        if expanded.is_symlink():
            raise ValueError("model broker ledger cannot be a symlink")
        self.path = expanded.resolve()
        if not self.path.is_absolute() or not self.path.parent.is_dir():
            raise ValueError("model broker ledger directory must exist")
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS model_attempt (attempted_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS model_response_cache ("
                "cache_key TEXT PRIMARY KEY, request_hash TEXT NOT NULL, "
                "prompt_hash TEXT NOT NULL, expires_at TEXT NOT NULL, "
                "proposal_json TEXT NOT NULL)"
            )
        os.chmod(self.path, 0o600)

    def reserve(
        self,
        *,
        cache_key: str,
        request_hash: str,
        prompt_hash: str,
        now: str,
        expires_at: str,
    ) -> tuple[str, dict[str, Any] | None]:
        cutoff = (dt.datetime.fromisoformat(now) - dt.timedelta(hours=1)).isoformat()
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM model_attempt WHERE attempted_at <= ?", (cutoff,))
            connection.execute("DELETE FROM model_response_cache WHERE expires_at <= ?", (now,))
            cached = connection.execute(
                "SELECT proposal_json FROM model_response_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if cached is not None:
                connection.commit()
                proposal = json.loads(cached[0])
                if not isinstance(proposal, dict):
                    raise ModelBrokerError("cache_invalid")
                return "cached", proposal
            attempts = connection.execute("SELECT count(*) FROM model_attempt").fetchone()[0]
            if attempts >= MAX_NEW_ATTEMPTS_PER_HOUR:
                connection.commit()
                return "rate_limited", None
            connection.execute("INSERT INTO model_attempt (attempted_at) VALUES (?)", (now,))
            connection.commit()
        return "reserved", None

    def cache_success(
        self,
        *,
        cache_key: str,
        request_hash: str,
        prompt_hash: str,
        expires_at: str,
        proposal: dict[str, Any],
    ) -> None:
        proposal_json = json.dumps(proposal, sort_keys=True, separators=(",", ":"))
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO model_response_cache "
                "(cache_key, request_hash, prompt_hash, expires_at, proposal_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (cache_key, request_hash, prompt_hash, expires_at, proposal_json),
            )


class OpenClawModelBroker:
    def __init__(
        self,
        *,
        executable: Path,
        config_path: Path,
        state_dir: Path,
        agent_id: str,
        prompt_path: Path,
        ledger_path: Path,
    ) -> None:
        self.executable = _trusted_executable(executable)
        self.config_path = _trusted_file(config_path, field="OpenClaw config")
        self.state_dir = state_dir.expanduser().resolve(strict=True)
        if not self.state_dir.is_dir():
            raise ValueError("OpenClaw state directory must exist")
        self.agent_id = agent_id
        self.prompt_path = _trusted_file(prompt_path, field="research prompt")
        prompt_bytes = self.prompt_path.read_bytes()
        if len(prompt_bytes) > MAX_PROMPT_BYTES:
            raise ValueError("research prompt exceeds its size limit")
        self.prompt = prompt_bytes.decode("utf-8")
        self.prompt_hash = canonical_hash(self.prompt)
        self.ledger = _BrokerLedger(ledger_path)
        self.environment = self._validated_environment()

    def _validated_environment(self) -> dict[str, str]:
        from src.agents.consumer import isolated_agent_environment

        return isolated_agent_environment(
            self.config_path, agent_id=self.agent_id, state_dir=self.state_dir
        )

    def validate_request(self, request: Any) -> tuple[str, str, str]:
        if not isinstance(request, dict) or set(request) != _REQUIRED_REQUEST_FIELDS:
            raise ModelBrokerError("invalid_request")
        request_bytes = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        if len(request_bytes) > MAX_REQUEST_BYTES:
            raise ModelBrokerError("request_too_large")
        if (
            request.get("schema") != "platform.agent_review_request/v1"
            or request.get("state") != "requested"
            or request.get("transport") != "postgresql_agent_review"
            or request.get("response_required") is not True
            or not isinstance(request.get("review_id"), str)
            or not request["review_id"].strip()
            or request.get("proposal_id") != request["review_id"]
        ):
            raise ModelBrokerError("invalid_request")
        requested_at = _parse_time(request["requested_at"])
        expires_at = _parse_time(request["expires_at"])
        now = _parse_time(_utc_now())
        duration = expires_at - requested_at
        remaining = expires_at - now
        if (
            requested_at > now + dt.timedelta(seconds=5)
            or duration <= dt.timedelta(0)
            or duration > dt.timedelta(hours=1)
            or remaining <= dt.timedelta(seconds=5)
        ):
            raise ModelBrokerError("request_expired_or_out_of_bounds")
        context = AgentContext(created_at=request["requested_at"], values=request["context"])
        if request.get("context_hash") != context.content_hash:
            raise ModelBrokerError("context_hash_mismatch")
        request_hash = canonical_hash(request)
        cache_key = canonical_hash({"request_hash": request_hash, "prompt_hash": self.prompt_hash})
        return request_hash, cache_key, timestamp(expires_at, field="expires_at")

    def proposal(self, envelope: dict[str, Any]) -> dict[str, Any]:
        if set(envelope) != {"schema", "request"} or envelope.get("schema") != REQUEST_SCHEMA:
            raise ModelBrokerError("invalid_request")
        request = envelope["request"]
        request_hash, cache_key, expires_at = self.validate_request(request)
        now = timestamp(_utc_now(), field="now")
        state, cached = self.ledger.reserve(
            cache_key=cache_key,
            request_hash=request_hash,
            prompt_hash=self.prompt_hash,
            now=now,
            expires_at=expires_at,
        )
        if state == "cached" and cached is not None:
            return {"state": "proposal", "proposal": cached}
        if state == "rate_limited":
            raise ModelBrokerError("rate_limited")
        proposal = self._run_agent(request, expires_at=expires_at)
        self._validate_proposal_binding(proposal, request)
        self.ledger.cache_success(
            cache_key=cache_key,
            request_hash=request_hash,
            prompt_hash=self.prompt_hash,
            expires_at=expires_at,
            proposal=proposal,
        )
        return {"state": "proposal", "proposal": proposal}

    @staticmethod
    def _validate_proposal_binding(proposal: dict[str, Any], request: dict[str, Any]) -> None:
        review_id = request["review_id"]
        provenance = proposal.get("provenance")
        context = request.get("context")
        if (
            not isinstance(provenance, dict)
            or provenance.get("review_id") != review_id
            or not isinstance(context, Mapping)
            or not isinstance(context.get("product_id"), str)
            or proposal.get("product_id") != context["product_id"]
        ):
            raise ModelBrokerError("proposal_request_binding_mismatch")
        try:
            built = build_agent_proposal(proposal)
        except (TypeError, ValueError) as exc:
            raise ModelBrokerError("invalid_model_proposal") from exc
        requested_at = timestamp(request["requested_at"], field="requested_at")
        accepted_at = timestamp(_utc_now(), field="accepted_at")
        if not requested_at <= built.created_at <= accepted_at:
            raise ModelBrokerError("proposal_request_binding_mismatch")

    def _run_agent(self, request: dict[str, Any], *, expires_at: str) -> dict[str, Any]:
        remaining = (_parse_time(expires_at) - _parse_time(_utc_now())).total_seconds()
        timeout = min(MAX_TURN_SECONDS, int(remaining) - 5)
        if timeout <= 0:
            raise ModelBrokerError("request_expired_or_out_of_bounds")
        message = self.prompt + "\n\nSealed review request:\n" + json.dumps(request, sort_keys=True)
        if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ModelBrokerError("request_too_large_for_openclaw_cli")
        command = [
            str(self.executable),
            "agent",
            "--local",
            "--agent",
            self.agent_id,
            "--session-id",
            str(uuid.uuid4()),
            "--message",
            message,
            "--json",
            "--timeout",
            str(timeout),
        ]
        try:
            output = _run_bounded_process(
                command,
                environment=self.environment,
                cwd=Path(self.environment["HOME"]),
                timeout=timeout,
            )
        except (OSError, ModelBrokerError, UnicodeError, ValueError) as exc:
            if isinstance(exc, ModelBrokerError):
                raise
            raise ModelBrokerError("model_execution_failed") from exc
        try:
            result = json.loads(output)
            payloads = result.get("payloads")
            if not isinstance(payloads, list) or len(payloads) != 1:
                raise ValueError
            proposal = json.loads(payloads[0]["text"])
            if not isinstance(proposal, dict):
                raise ValueError
            return proposal
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelBrokerError("invalid_model_response") from exc


def _run_bounded_process(
    command: list[str], *, environment: dict[str, str], cwd: Path, timeout: int
) -> str:
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    try:
        outputs = _read_bounded_output(process, deadline=deadline)
        if process.wait(timeout=max(0.1, deadline - time.monotonic())) != 0:
            raise ModelBrokerError("model_execution_failed")
    except (OSError, subprocess.TimeoutExpired):
        raise ModelBrokerError("model_execution_failed") from None
    finally:
        _stop_process(process)
    try:
        return outputs["stdout"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelBrokerError("invalid_model_response") from exc


def _read_bounded_output(
    process: subprocess.Popen[bytes], *, deadline: float
) -> dict[str, bytearray]:
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    outputs: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelBrokerError("model_timeout")
            for key, _mask in selector.select(min(remaining, 1.0)):
                chunk = os.read(key.fd, min(65_536, MAX_OUTPUT_BYTES + 1))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if sum(len(value) for value in outputs.values()) + len(chunk) > MAX_OUTPUT_BYTES:
                    raise ModelBrokerError("model_output_too_large")
                outputs[key.data].extend(chunk)
    finally:
        selector.close()
    return outputs


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
        process.wait()
    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        process.stderr.close()


def _trusted_file(path: Path, *, field: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid not in {0, os.geteuid()}:
        raise ValueError(f"{field} must be a regular trusted file")
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise ValueError(f"{field} cannot be group or world writable")
    for parent in (resolved.parent, *resolved.parent.parents):
        parent_info = parent.stat()
        if parent_info.st_uid not in {0, os.geteuid()}:
            raise ValueError(f"{field} has an untrusted parent directory")
        protected_sticky = parent_info.st_uid == 0 and parent_info.st_mode & stat.S_ISVTX
        if stat.S_IMODE(parent_info.st_mode) & 0o022 and not protected_sticky:
            raise ValueError(f"{field} has a writable parent directory")
    return resolved


def _trusted_executable(path: Path) -> Path:
    executable = _trusted_file(path, field="OpenClaw executable")
    if not os.access(executable, os.X_OK):
        raise ValueError("OpenClaw executable is not executable")
    return executable


def _parse_time(value: Any) -> dt.datetime:
    if not isinstance(value, str):
        raise ModelBrokerError("invalid_request_time")
    try:
        return dt.datetime.fromisoformat(
            timestamp(value, field="request timestamp", preserve_subseconds=True)
        )
    except ValueError as exc:
        raise ModelBrokerError("invalid_request_time") from exc


def _utc_now() -> str:
    from src.services.runtime import utc_now

    return utc_now()


class _ModelBrokerRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = cast(OpenClawModelBrokerServer, self.server)
        connection: socket.socket = self.request
        try:
            connection.settimeout(5)
            uid = server.peer_uid_reader(connection)
            if uid not in server.allowed_uids:
                raise ModelBrokerError("unauthorised_peer")
            _write_frame(connection, {"state": "ready"}, maximum_bytes=MAX_OUTPUT_BYTES)
            envelope = _read_frame(connection, maximum_bytes=MAX_REQUEST_BYTES)
            result = server.broker.proposal(envelope)
        except ModelBrokerError as exc:
            result = {"state": "error", "reason_code": exc.reason_code}
        except (OSError, UnicodeError, ValueError):
            result = {"state": "error", "reason_code": "invalid_request"}
        except Exception:
            result = {"state": "error", "reason_code": "broker_failed"}
        try:
            connection.settimeout(None)
            _write_frame(connection, result, maximum_bytes=MAX_OUTPUT_BYTES)
        except (OSError, ModelBrokerError):
            return


class OpenClawModelBrokerServer(socketserver.UnixStreamServer):
    request_queue_size = 8

    def __init__(
        self,
        socket_path: Path,
        broker: OpenClawModelBroker,
        *,
        allowed_uids: frozenset[int],
        peer_uid_reader: Callable[[socket.socket], int] = _peer_uid,
    ) -> None:
        self.socket_path = socket_path.expanduser().absolute()
        self.broker = broker
        self.allowed_uids = allowed_uids
        self.peer_uid_reader = peer_uid_reader
        if not self.allowed_uids:
            raise ValueError("model broker requires at least one allowed peer uid")
        if not self.socket_path.parent.is_dir():
            raise ValueError("model broker socket directory must exist")
        parent_mode = stat.S_IMODE(self.socket_path.parent.stat().st_mode)
        if parent_mode & 0o022:
            raise ValueError("model broker socket directory cannot be writable by peers")
        super().__init__(str(self.socket_path), _ModelBrokerRequestHandler, bind_and_activate=True)

    def server_bind(self) -> None:
        if self.socket_path.exists() or self.socket_path.is_symlink():
            info = self.socket_path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
                raise ValueError("refusing to replace a non-owned model broker socket")
            self.socket_path.unlink()
        super().server_bind()
        os.chmod(self.socket_path, 0o660)

    def server_close(self) -> None:
        super().server_close()
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(info.st_mode) and info.st_uid == os.geteuid():
            self.socket_path.unlink()


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be configured")
    return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--agent", choices=("research",), required=True)
    parser.add_argument("--allowed-user", default="trading-agent")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    broker = OpenClawModelBroker(
        executable=_required_path("TRADING_PLATFORM_OPENCLAW_EXECUTABLE"),
        config_path=_required_path("TRADING_PLATFORM_OPENCLAW_CONFIG"),
        state_dir=_required_path("TRADING_PLATFORM_OPENCLAW_STATE_DIR"),
        agent_id=args.agent,
        prompt_path=args.prompt,
        ledger_path=args.ledger,
    )
    if args.validate:
        print(json.dumps({"state": "valid"}, sort_keys=True))
        return
    if not peer_credentials_available():
        raise RuntimeError("model broker requires Unix peer credentials")
    uid = pwd.getpwnam(args.allowed_user).pw_uid
    server = OpenClawModelBrokerServer(args.socket, broker, allowed_uids=frozenset({uid}))
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
