"""One leased review consumer with a tool-free, isolated OpenClaw turn."""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.agents.context import AgentContext
from src.agents.openclaw_bridge import build_agent_proposal
from src.agents.transport import sign_review_response
from src.domain._codec import canonical_hash


def post_review_api(base_url: str, path: str, payload: dict[str, Any], *, key: bytes) -> dict:
    url = urlsplit(base_url)
    if url.username or url.password or url.path not in {"", "/"} or url.query or url.fragment:
        raise ValueError("agent API URL must contain only scheme, host and port")
    if not url.hostname:
        raise ValueError("agent API URL requires a host")
    connection: http.client.HTTPConnection
    if url.scheme == "https":
        connection = http.client.HTTPSConnection(url.hostname, url.port, timeout=30)
    elif url.scheme == "http" and url.hostname in {"127.0.0.1", "::1", "localhost"}:
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
    else:
        raise ValueError("agent API requires HTTPS or loopback HTTP")
    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={"Authorization": f"Bearer {key.hex()}", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read(1_048_577)
        if response.status != 200 or len(body) > 1_048_576:
            raise RuntimeError(f"agent API failed with status {response.status}")
        result = json.loads(body)
        if not isinstance(result, dict):
            raise ValueError("agent API response must be an object")
        return result
    finally:
        connection.close()


def isolated_agent_environment(config_path: Path, *, agent_id: str) -> dict[str, str]:
    """Validate the installed 2026.6 agent contract before sending any context."""
    config = json.loads(config_path.read_text())
    agents = config.get("agents", {})
    entries = [entry for entry in agents.get("list", []) if entry.get("id") == agent_id]
    if len(entries) != 1 or agent_id == "main":
        raise ValueError("a dedicated OpenClaw research agent is required")
    agent = entries[0]
    if agent.get("tools", {}).get("deny") != ["*"]:
        raise ValueError("OpenClaw research agent must deny every tool")
    sandbox = agent.get("sandbox", {})
    if sandbox.get("mode") != "all" or sandbox.get("workspaceAccess") != "none":
        raise ValueError("OpenClaw research agent requires sandboxing without workspace access")
    workspace = Path(agent.get("workspace", "")).expanduser().resolve()
    default = (
        Path(agents.get("defaults", {}).get("workspace", "~/.openclaw/workspace"))
        .expanduser()
        .resolve()
    )
    if not agent.get("workspace") or not workspace.is_dir() or workspace == default:
        raise ValueError("OpenClaw research agent requires a separate workspace")
    return {
        "HOME": str(Path.home()),
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "OPENCLAW_CONFIG_PATH": str(config_path.resolve()),
        "OPENCLAW_STATE_DIR": str(config_path.resolve().parent),
    }


def consume_review(
    *, api_url: str, key: bytes, executable: Path, config_path: Path, agent_id: str, prompt: str
) -> dict[str, Any]:
    from src.services.runtime import utc_now

    environment = isolated_agent_environment(config_path, agent_id=agent_id)
    request = post_review_api(api_url, "/agent/requests/next", {}, key=key)["request"]
    if request is None:
        return {"state": "idle"}
    remaining = (
        dt.datetime.fromisoformat(request["expires_at"]) - dt.datetime.fromisoformat(utc_now())
    ).total_seconds()
    if remaining <= 5:
        return {"state": "expired"}
    timeout = min(480, int(remaining) - 5)
    context = AgentContext(created_at=request["requested_at"], values=request["context"])
    if request["context_hash"] != context.content_hash:
        raise ValueError("agent request context hash does not match")
    message = prompt + "\n\nSealed review request:\n" + json.dumps(request, sort_keys=True)
    completed = subprocess.run(
        [
            str(executable.resolve()),
            "agent",
            "--local",
            "--agent",
            agent_id,
            "--session-id",
            str(uuid.uuid4()),
            "--message",
            message,
            "--json",
            "--timeout",
            str(timeout),
        ],
        env=environment,
        cwd=config_path.resolve().parent,
        capture_output=True,
        text=True,
        timeout=timeout + 5,
        check=False,
    )
    if completed.returncode != 0 or len(completed.stdout) > 1_048_576:
        raise RuntimeError("isolated OpenClaw turn failed")
    result = json.loads(completed.stdout)
    payloads = result.get("payloads")
    if not isinstance(payloads, list) or len(payloads) != 1:
        raise ValueError("OpenClaw must return exactly one JSON proposal")
    proposal = json.loads(payloads[0]["text"])
    build_agent_proposal(proposal)
    envelope = sign_review_response(request, proposal, key=key)
    response = post_review_api(api_url, "/agent/responses", envelope, key=key)
    return {**response, "request_hash": canonical_hash(request)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--prompt", required=True, type=Path)
    args = parser.parse_args()
    key = bytes.fromhex(os.environ["TRADING_PLATFORM_AGENT_TRANSPORT_KEY"])
    if len(key) < 32:
        raise ValueError("agent transport key requires at least 32 bytes")
    result = consume_review(
        api_url=args.api_url,
        key=key,
        executable=args.executable,
        config_path=args.config,
        agent_id=args.agent,
        prompt=args.prompt.read_text(),
    )
    print(json.dumps(result, sort_keys=True))


def consume_configured_review() -> dict[str, Any]:
    key = bytes.fromhex(os.environ["TRADING_PLATFORM_AGENT_TRANSPORT_KEY"])
    if len(key) < 32:
        raise ValueError("agent transport key requires at least 32 bytes")
    return consume_review(
        api_url=os.environ.get("TRADING_PLATFORM_AGENT_API_URL", "http://127.0.0.1:8088"),
        key=key,
        executable=Path(os.environ["TRADING_PLATFORM_OPENCLAW_EXECUTABLE"]),
        config_path=Path(os.environ["TRADING_PLATFORM_OPENCLAW_CONFIG"]),
        agent_id=os.environ.get("TRADING_PLATFORM_OPENCLAW_AGENT", "research"),
        prompt=Path("config/openclaw_daily_review_prompt.md").read_text(),
    )


if __name__ == "__main__":
    main()
