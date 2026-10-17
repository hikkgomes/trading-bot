"""One leased review consumer with a tool-free, isolated OpenClaw turn."""

from __future__ import annotations

import datetime as dt
import http.client
import json
import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.agents.context import AgentContext
from src.agents.openclaw_bridge import build_agent_proposal
from src.agents.transport import sign_review_response
from src.domain._codec import canonical_hash


def _validate_agent_directories(
    agents: dict[str, Any], agent: dict[str, Any], state_path: Path
) -> Path:
    workspace_value = agent.get("workspace")
    agent_dir_value = agent.get("agentDir")
    if not isinstance(workspace_value, str) or not workspace_value:
        raise ValueError("OpenClaw research agent requires a separate workspace")
    if not isinstance(agent_dir_value, str) or not agent_dir_value:
        raise ValueError("OpenClaw research agent requires a separate agent directory")
    if not Path(workspace_value).is_absolute() or not Path(agent_dir_value).is_absolute():
        raise ValueError("OpenClaw research workspace paths must be absolute")
    workspace = Path(workspace_value).resolve()
    agent_dir = Path(agent_dir_value).resolve()
    defaults = agents.get("defaults", {})
    default_workspace = (
        Path(defaults.get("workspace", "~/.openclaw/workspace")).expanduser().resolve()
    )
    default_agent_dir_value = defaults.get("agentDir")
    default_agent_dir = (
        Path(default_agent_dir_value).expanduser().resolve()
        if isinstance(default_agent_dir_value, str) and default_agent_dir_value
        else state_path / "agents" / "main" / "agent"
    )
    if not workspace.is_dir() or workspace == default_workspace:
        raise ValueError("OpenClaw research agent requires a separate workspace")
    if (
        not agent_dir.is_dir()
        or agent_dir in {default_agent_dir, workspace}
        or not workspace.is_relative_to(state_path)
        or not agent_dir.is_relative_to(state_path)
    ):
        raise ValueError("OpenClaw research agent requires a separate agent directory")
    return workspace


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


def isolated_agent_environment(
    config_path: Path, *, agent_id: str, state_dir: Path
) -> dict[str, str]:
    """Validate the installed 2026.6 agent contract before sending any context."""
    config_path = config_path.resolve(strict=True)
    state_path = state_dir.expanduser().resolve(strict=True)
    if config_path == state_path / "openclaw.json":
        raise ValueError("OpenClaw research requires a separate configuration")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("OpenClaw research config must be an object")
    agents = config.get("agents", {})
    entries = [entry for entry in agents.get("list", []) if entry.get("id") == agent_id]
    if len(entries) != 1 or agent_id == "main":
        raise ValueError("a dedicated OpenClaw research agent is required")
    agent = entries[0]
    if agent.get("tools", {}).get("deny") != ["*"]:
        raise ValueError("OpenClaw research agent must deny every tool")
    if agent.get("skills") != []:
        raise ValueError("OpenClaw research agent must disable every skill")
    sandbox = agent.get("sandbox", {})
    if sandbox.get("mode") != "off" or sandbox.get("workspaceAccess") != "none":
        raise ValueError("tool-free research uses the restricted OS service, without Docker access")
    if config.get("hooks", {}).get("enabled") is not False:
        raise ValueError("OpenClaw research must disable hooks")
    _validate_provider_plugins(config, state_path)
    workspace = _validate_agent_directories(agents, agent, state_path)
    return {
        "HOME": str(workspace),
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "OPENCLAW_CONFIG_PATH": str(config_path.resolve()),
        "OPENCLAW_STATE_DIR": str(state_path),
        "OPENCLAW_DISABLE_BUNDLED_SOURCE_OVERLAYS": "1",
    }


def _validate_provider_plugins(config: dict[str, Any], state_path: Path) -> None:
    plugins = config.get("plugins", {})
    if plugins == {"enabled": False}:
        return
    expected = {
        "enabled": True,
        "allow": ["openai"],
        "load": {"paths": []},
        "entries": {"openai": {"enabled": True}},
    }
    if plugins != expected:
        raise ValueError("research permits only the bundled native OpenAI provider")
    provider = config.get("models", {}).get("providers", {}).get("openai", {})
    if (
        provider.get("auth") != "oauth"
        or provider.get("api") != "openai-chatgpt-responses"
        or provider.get("baseUrl") != "https://chatgpt.com/backend-api/codex"
        or provider.get("agentRuntime") != {"id": "openclaw"}
        or any(key in provider for key in ("apiKey", "headers"))
    ):
        raise ValueError("research requires the embedded native OAuth provider")
    models = config.get("agents", {}).get("defaults", {}).get("models", {})
    declarations = [*models.values(), *provider.get("models", [])]
    if not declarations or any(
        item.get("agentRuntime") != {"id": "openclaw"} for item in declarations
    ):
        raise ValueError("every research model must use the embedded runtime")
    _validate_private_state_binding(config, state_path)
    _reject_installed_provider_override(state_path)


def _validate_private_state_binding(config: dict[str, Any], state_path: Path) -> None:
    configured = os.environ.get("TRADING_PLATFORM_OPENCLAW_PRIVATE_STATE")
    if not configured:
        raise ValueError("research requires a private global state mount")
    private = Path(configured).resolve(strict=True)
    if private.is_relative_to(state_path) or not (state_path / "state").samefile(private):
        raise ValueError("research global state is not isolated")
    session_store = Path(str(config.get("session", {}).get("store", "")))
    expected = private.parent / "sessions" / "{agentId}" / "sessions.json"
    if session_store != expected:
        raise ValueError("research session lookup must remain private")


def _reject_installed_provider_override(state_path: Path) -> None:
    database = state_path / "state" / "openclaw.sqlite"
    if not database.exists():
        # A fresh private mount has no installed plugins; native startup creates its index.
        return
    if database.is_symlink() or not database.is_file():
        raise ValueError("native plugin registry is unavailable")
    try:
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT install_records_json, plugins_json FROM installed_plugin_index "
                "WHERE index_key='installed-plugin-index'"
            ).fetchone()
        if row is None:
            return
        installs, plugins = json.loads(row[0]), json.loads(row[1])
        if not isinstance(installs, dict) or not isinstance(plugins, list):
            raise ValueError("native plugin registry is invalid")
        if "openai" in installs or any(
            plugin.get("pluginId") == "openai" and plugin.get("installRecord") is not None
            for plugin in plugins
        ):
            raise ValueError("research provider is overridden by an installed plugin")
    except (sqlite3.Error, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("native plugin registry cannot be verified") from exc


def consume_review(*, api_url: str, key: bytes, broker_socket: Path) -> dict[str, Any]:
    from src.services.runtime import utc_now

    if not broker_socket.is_socket():
        return {"state": "broker_unavailable"}
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
    from src.agents.model_broker import ModelBrokerError, request_model_proposal

    try:
        proposal = request_model_proposal(broker_socket, request, timeout=timeout + 5)
    except ModelBrokerError as exc:
        if exc.reason_code == "rate_limited":
            return {"state": "deferred", "reason_code": "model_hourly_limit"}
        raise
    build_agent_proposal(proposal)
    envelope = sign_review_response(request, proposal, key=key)
    response = post_review_api(api_url, "/agent/responses", envelope, key=key)
    return {**response, "request_hash": canonical_hash(request)}


def main() -> None:
    print(json.dumps(consume_configured_review(), sort_keys=True))


def consume_configured_review() -> dict[str, Any]:
    key = bytes.fromhex(os.environ["TRADING_PLATFORM_AGENT_TRANSPORT_KEY"])
    if len(key) < 32:
        raise ValueError("agent transport key requires at least 32 bytes")
    return consume_review(
        api_url=os.environ.get("TRADING_PLATFORM_AGENT_API_URL", "http://127.0.0.1:8088"),
        key=key,
        broker_socket=Path(os.environ["TRADING_PLATFORM_MODEL_BROKER_SOCKET"]),
    )


if __name__ == "__main__":
    main()
