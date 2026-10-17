from __future__ import annotations

import datetime as dt
import json
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.agents.compiler import AgentCompilationError, compile_openclaw_candidate_payload
from src.agents.openclaw_bridge import OpenClawAgentBridge, build_agent_proposal
from src.agents.store import SqlAgentStore
from src.agents.transport import sign_review_response
from src.data.database import PlatformDatabase
from src.domain._codec import canonical_hash
from src.research.datasets import RESEARCH_BUNDLE_ROLES, CanonicalResearchDatasetBuilder
from src.research.store import SqlResearchStore
from src.services.agent_worker import (
    DatabaseAgentJobHandlers,
    adaptive_research_report,
    build_agent_review_context,
)
from src.services.control_api import DatabaseControlPlane, build_control_server
from src.services.health import DatabaseHeartbeatStore
from src.services.scheduler import DatabaseJobQueue

NOW = dt.datetime(2026, 8, 30, tzinfo=dt.UTC).isoformat()


def _payload(product_id: str = "active_income") -> dict:
    return {
        "schema": "openclaw.agent_proposal/v1",
        "source": "openclaw",
        "proposal_id": "openclaw-typed-1",
        "role": "researcher",
        "action": "create_dsl",
        "product_id": product_id,
        "created_at": NOW,
        "thesis": {
            "mechanism_category": "behavioural",
            "market_rationale": "Persistent demand imbalance can continue across bars.",
            "expected_causal_chain": ["demand imbalance", "trend", "subsequent return"],
            "expected_direction": "signed",
            "expected_horizon": "15m to 3d",
            "required_data": ["closed_ohlcv_bars"],
            "permitted_features": ["trend", "returns"],
            "instrument_universe": ["BTCUSDT", "ETHUSDT"],
            "generalisation_scope": {
                "product": product_id,
                "family": "time_series",
                "predeclared": True,
            },
            "failure_regimes": ["structural break"],
            "falsification_tests": ["chronological holdout"],
            "negative_controls": ["block_permutation"],
            "execution_capacity_assumptions": {"maximum_participation": 0.01},
            "parent_thesis_ids": [],
            "cumulative_trial_budget": 4,
        },
        "provenance": {
            "rule": {
                "feature": "trend",
                "operator": "gt",
                "threshold": 0.0,
                "direction": "signed",
            },
            "evidence_type": "swing",
        },
    }


def _bundle(database: PlatformDatabase, *, created_at=NOW):
    identities = {
        "universe": canonical_hash({"universe": "active"}),
        "feature": canonical_hash({"feature": "v1"}),
        "cost": canonical_hash({"cost": "v1"}),
        "parameters": canonical_hash({"parameters": "v1"}),
    }
    intervals = {
        role: {
            "start": f"2026-08-{20 + index:02d}T00:00:00+00:00",
            "end": f"2026-08-{21 + index:02d}T00:00:00+00:00",
        }
        for index, role in enumerate(RESEARCH_BUNDLE_ROLES)
    }
    return CanonicalResearchDatasetBuilder(database.engine).build(
        "active_income",
        intervals=intervals,
        payload_by_role={role: {"bars": [{"close": 100.0}]} for role in RESEARCH_BUNDLE_ROLES},
        universe_snapshot_id=identities["universe"],
        feature_manifest_id=identities["feature"],
        cost_model_id=identities["cost"],
        parameter_set_id=identities["parameters"],
        instrument_scope=("BTCUSDT", "ETHUSDT"),
        availability_timestamp=created_at,
        created_at=created_at,
    )


def test_openclaw_typed_thesis_rejects_live_fields() -> None:
    payload = _payload()
    payload["thesis"]["accepted"] = True
    with pytest.raises(ValueError, match="unsupported fields"):
        build_agent_proposal(payload)


def test_consumer_defers_the_hourly_limit_without_retrying_or_fabricating_a_response(monkeypatch):
    from unittest.mock import Mock

    from src.agents import consumer, model_broker
    from src.agents.context import AgentContext

    context = AgentContext(created_at=NOW, values={"product_id": "active_income"})
    request = {
        "requested_at": NOW,
        "expires_at": (dt.datetime.fromisoformat(NOW) + dt.timedelta(minutes=10)).isoformat(),
        "context": dict(context.values),
        "context_hash": context.content_hash,
    }
    post = Mock(return_value={"request": request})
    model = Mock(side_effect=model_broker.ModelBrokerError("rate_limited"))
    monkeypatch.setattr(consumer, "post_review_api", post)
    monkeypatch.setattr(model_broker, "request_model_proposal", model)
    monkeypatch.setattr("src.services.runtime.utc_now", lambda: NOW)
    broker_socket = Mock()
    broker_socket.is_socket.return_value = True
    assert consumer.consume_review(
        api_url="http://127.0.0.1:8088", key=b"fixture-key", broker_socket=broker_socket
    ) == {"state": "deferred", "reason_code": "model_hourly_limit"}
    model.assert_called_once()
    post.assert_called_once_with(
        "http://127.0.0.1:8088", "/agent/requests/next", {}, key=b"fixture-key"
    )

    model.side_effect = model_broker.ModelBrokerError("invalid_model_proposal")
    with pytest.raises(model_broker.ModelBrokerError, match="invalid_model_proposal"):
        consumer.consume_review(
            api_url="http://127.0.0.1:8088", key=b"fixture-key", broker_socket=broker_socket
        )


def test_openclaw_typed_thesis_compiles_only_through_trusted_dsl(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'openclaw.sqlite3'}")
    database.create_schema()
    bundle = _bundle(database)
    proposal = build_agent_proposal(_payload())
    candidate = compile_openclaw_candidate_payload(
        proposal,
        bundle=bundle,
        submitted_at=NOW,
    )
    assert candidate.definition.source_type.value == "generated_dsl"
    assert candidate.dataset_plan is not None
    assert candidate.definition.metadata["trusted_compiler"] == "openclaw-trusted-compiler/v1"
    unsafe = _payload()
    unsafe["provenance"]["rule"]["feature"] = "secret_feature"
    unsafe_proposal = build_agent_proposal(unsafe)
    with pytest.raises(AgentCompilationError, match="outside its thesis"):
        compile_openclaw_candidate_payload(unsafe_proposal, bundle=bundle, submitted_at=NOW)


def test_openclaw_compiler_rejects_specialised_or_hedged_scalar_theses(tmp_path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'capabilities.sqlite3'}")
    database.create_schema()
    bundle = _bundle(database)
    for family, direction in (
        ("relative_value", "signed"),
        ("cross_sectional", "signed"),
        ("machine_learning", "signed"),
        ("microstructure", "signed"),
        ("meta_strategy", "signed"),
        ("execution", "signed"),
        ("market_making", "signed"),
        ("time_series", "hedged"),
        ("time_series", "market_neutral"),
    ):
        payload = _payload()
        payload["thesis"]["generalisation_scope"]["family"] = family
        payload["provenance"]["rule"]["direction"] = direction
        with pytest.raises(AgentCompilationError, match="directional DSL"):
            compile_openclaw_candidate_payload(
                build_agent_proposal(payload), bundle=bundle, submitted_at=NOW
            )
    payload = _payload()
    payload["thesis"]["expected_direction"] = "hedged"
    with pytest.raises(AgentCompilationError, match="directional DSL"):
        compile_openclaw_candidate_payload(
            build_agent_proposal(payload), bundle=bundle, submitted_at=NOW
        )
    database.dispose()


def test_openclaw_http_transport_compiles_a_candidate_into_the_normal_queue(
    tmp_path, monkeypatch
) -> None:
    import os
    from pathlib import Path
    from unittest.mock import Mock

    from test_platform_candidate_lifecycle import _bundle as market_bundle

    from src.agents.consumer import consume_review
    from src.agents.model_broker import (
        ModelBrokerError,
        OpenClawModelBroker,
        OpenClawModelBrokerServer,
        peer_credentials_available,
        request_model_proposal,
    )
    from src.research.datasets import CanonicalDatasetResolver, SqlCanonicalDatasetRepository
    from src.services.config import load_split_configuration
    from src.services.research_jobs import DatabaseResearchJobHandlers
    from src.services.scheduler import PlatformScheduler, ScheduleSpec

    now = "2026-09-07T00:00:00+00:00"
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'openclaw-worker.sqlite3'}")
    database.create_schema()
    market_bundle(database)
    queue = DatabaseJobQueue(database.engine)
    monkeypatch.setattr("src.services.agent_worker.utc_now", lambda: now)
    monkeypatch.setattr("src.services.runtime.utc_now", lambda: now)
    handlers = DatabaseAgentJobHandlers(
        queue=queue,
        store=SqlAgentStore(database.engine),
        code_workflow=SimpleNamespace(),
        maximum_runtime_seconds=60,
    )
    review = handlers.agent_review(
        SimpleNamespace(job_id="http-review", payload={"product_id": "active_income"}), lambda: None
    )
    key = b"scoped-agent-transport-test-key!!"
    server = build_control_server(
        bind=("127.0.0.1", 0),
        control_plane=DatabaseControlPlane(
            database.engine, DatabaseHeartbeatStore(database.engine)
        ),
        bearer_token="operator-only-token",
        agent_transport_key=key,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    broker_server = None
    broker_thread = None
    socket_dir = None

    def post(path, payload):
        request = Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key.hex()}", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    try:
        for path in (
            "/emergency-flatten",
            "/agent/proposals",
            "/reports",
            "/paper/restart-forward-window",
        ):
            with pytest.raises(HTTPError) as denied:
                post(path, {})
            assert denied.value.code == 401
        request = post("/agent/requests/next", {})["request"]
        assert request["review_id"] == review["review_id"]
        assert post("/agent/requests/next", {})["request"] is None
        payload = _payload()
        payload["created_at"] = request["requested_at"]
        payload["thesis"]["instrument_universe"] = ["BTCUSDT"]
        payload["provenance"]["review_id"] = review["review_id"]
        envelope = sign_review_response(request, payload, key=key)
        tampered = {**envelope, "proposal": {**payload, "product_id": "btc_accumulation"}}
        with pytest.raises(HTTPError) as denied:
            post("/agent/responses", tampered)
        assert denied.value.code == 400
        state_dir = tmp_path / "openclaw-state"
        state_dir.mkdir()
        workspace = state_dir / "research-workspace"
        workspace.mkdir()
        agent_dir = state_dir / "agents" / "research" / "agent"
        agent_dir.mkdir(parents=True)
        main_workspace = state_dir / "workspace"
        main_workspace.mkdir()
        main_agent_dir = state_dir / "agents" / "main" / "agent"
        main_agent_dir.mkdir(parents=True)
        agent_config = tmp_path / "openclaw.json"
        agent_config.write_text(
            json.dumps(
                {
                    "plugins": {"enabled": False},
                    "hooks": {"enabled": False},
                    "agents": {
                        "defaults": {
                            "workspace": str(main_workspace),
                            "agentDir": str(main_agent_dir),
                        },
                        "list": [
                            {
                                "id": "research",
                                "workspace": str(workspace),
                                "agentDir": str(agent_dir),
                                "skills": [],
                                "tools": {"deny": ["*"]},
                                "sandbox": {"mode": "off", "workspaceAccess": "none"},
                            }
                        ],
                    },
                }
            )
        )
        executable = tmp_path / "openclaw-test-process"
        response_path = tmp_path / "fake-model-response.json"
        invalid_payload = {
            **payload,
            "provenance": {**payload["provenance"], "review_id": "wrong-review"},
        }
        response_path.write_text(json.dumps({"payloads": [{"text": json.dumps(invalid_payload)}]}))
        base_payload = json.dumps(payload)
        call_log = tmp_path / "model-calls.log"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys, uuid\n"
            "assert sys.argv[1:3] == ['agent', '--local']\n"
            "assert '--deliver' not in sys.argv and '--model' not in sys.argv\n"
            "assert sys.argv[sys.argv.index('--agent') + 1] == 'research'\n"
            "uuid.UUID(sys.argv[sys.argv.index('--session-id') + 1])\n"
            "message = sys.argv[sys.argv.index('--message') + 1]\n"
            "sealed_request = json.loads(message.split('Sealed review request:\\n', 1)[1])\n"
            "assert 'Return one research proposal as JSON.' in message\n"
            "assert 'Sealed review request:' in message\n"
            f"assert os.environ['OPENCLAW_STATE_DIR'] == {str(state_dir)!r}\n"
            f"assert os.environ['OPENCLAW_CONFIG_PATH'] == {str(agent_config)!r}\n"
            f"assert os.environ['HOME'] == {str(workspace)!r} and os.getcwd() == {str(workspace)!r}\n"
            "assert not any(key.startswith(('TRADING_', 'EXCHANGE_')) for key in os.environ)\n"
            f"open({str(call_log)!r}, 'a').write(sys.argv[sys.argv.index('--session-id') + 1] + '\\n')\n"
            f"fixed_response = json.load(open({str(response_path)!r}))\n"
            "if fixed_response is None:\n"
            f"    model_proposal = json.loads({base_payload!r})\n"
            "    model_proposal['created_at'] = sealed_request['requested_at']\n"
            "    model_proposal['provenance'] = dict(model_proposal['provenance'], review_id=sealed_request['review_id'])\n"
            "    fixed_response = {'payloads': [{'text': json.dumps(model_proposal)}]}\n"
            "print(json.dumps(fixed_response))\n"
        )
        executable.chmod(0o700)
        agent_config.chmod(0o600)
        prompt_path = tmp_path / "prompt.md"
        prompt_path.write_text("Return one research proposal as JSON.")
        prompt_path.chmod(0o600)
        ledger_path = tmp_path / "model-broker.sqlite3"
        socket_dir = Path("/tmp") / f"model-broker-{os.urandom(4).hex()}"
        socket_dir.mkdir(mode=0o700)
        socket_path = socket_dir / "broker.sock"
        broker = OpenClawModelBroker(
            executable=executable,
            config_path=agent_config,
            state_dir=state_dir,
            agent_id="research",
            prompt_path=prompt_path,
            ledger_path=ledger_path,
        )
        peer_uid_reader = None
        if not peer_credentials_available():
            peer_uid_reader = Mock(return_value=os.getuid())

        server_arguments = (
            {"peer_uid_reader": peer_uid_reader} if peer_uid_reader is not None else {}
        )
        broker_server = OpenClawModelBrokerServer(
            socket_path,
            broker,
            allowed_uids=frozenset({os.getuid()}),
            **server_arguments,
        )
        broker_thread = threading.Thread(target=broker_server.serve_forever, daemon=True)
        broker_thread.start()
        monkeypatch.setenv("TRADING_PLATFORM_AGENT_TRANSPORT_KEY", key.hex())
        monkeypatch.setenv("EXCHANGE_API_KEY", "fixture-only-secret")
        now = (dt.datetime.fromisoformat(now) + dt.timedelta(minutes=11)).isoformat()
        with pytest.raises(ModelBrokerError, match="proposal_request_binding_mismatch"):
            request_model_proposal(socket_path, request, timeout=5)
        response_path.write_text("null")
        assert request_model_proposal(socket_path, request, timeout=5) == payload
        received = consume_review(
            api_url=f"http://127.0.0.1:{server.server_port}",
            key=key,
            broker_socket=socket_path,
        )
        assert request_model_proposal(socket_path, request, timeout=5) == payload
        with pytest.raises(ModelBrokerError, match="invalid_request"):
            request_model_proposal(socket_path, {}, timeout=5)
        for index in range(1, 3):
            review_id = f"synthetic-review-{index}"
            next_request = {**request, "review_id": review_id, "proposal_id": review_id}
            synthetic = request_model_proposal(socket_path, next_request, timeout=5)
            assert synthetic["provenance"]["review_id"] == review_id
        limited_request = {**request, "review_id": "synthetic-review-3"}
        limited_request["proposal_id"] = limited_request["review_id"]
        with pytest.raises(ModelBrokerError, match="rate_limited"):
            request_model_proposal(socket_path, limited_request, timeout=5)
        sessions = call_log.read_text().splitlines()
        assert len(sessions) == 4 and len(set(sessions)) == 4
        replay = post("/agent/responses", envelope)
        assert replay["proposal_id"] == received["proposal_id"]
        assert received["state"] == "responded"
        broker_server.shutdown()
        broker_server.server_close()
        broker_thread.join(timeout=5)

        broker = OpenClawModelBroker(
            executable=executable,
            config_path=agent_config,
            state_dir=state_dir,
            agent_id="research",
            prompt_path=prompt_path,
            ledger_path=ledger_path,
        )
        broker_server = OpenClawModelBrokerServer(
            socket_path,
            broker,
            allowed_uids=frozenset({os.getuid()}),
            **server_arguments,
        )
        broker_thread = threading.Thread(target=broker_server.serve_forever, daemon=True)
        broker_thread.start()
        assert request_model_proposal(socket_path, request, timeout=5) == payload
        assert len(call_log.read_text().splitlines()) == 4
        with pytest.raises(ModelBrokerError, match="rate_limited"):
            request_model_proposal(
                socket_path,
                {
                    **request,
                    "review_id": "synthetic-review-4",
                    "proposal_id": "synthetic-review-4",
                },
                timeout=5,
            )

        broker_server.shutdown()
        broker_server.server_close()
        broker_thread.join(timeout=5)
        denied_broker_server = OpenClawModelBrokerServer(
            socket_path,
            broker,
            allowed_uids=frozenset({os.getuid() + 1}),
            **server_arguments,
        )
        broker_server = denied_broker_server
        broker_thread = threading.Thread(target=broker_server.serve_forever, daemon=True)
        broker_thread.start()
        with pytest.raises(ModelBrokerError, match="unauthorised_peer"):
            request_model_proposal(socket_path, request, timeout=5)
    finally:
        if broker_thread is not None and broker_thread.is_alive():
            broker_server.shutdown()
            broker_server.server_close()
            broker_thread.join(timeout=5)
        if socket_dir is not None:
            socket_dir.rmdir()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    queue.register_worker(
        worker_id="agent-worker",
        node_id="node",
        role="agent-sandbox",
        capabilities=("agent_research",),
        observed_at=now,
    )
    claimed = queue.claim(
        worker_id="agent-worker",
        now=now,
        lease_seconds=60,
        names=("agent_research",),
    )
    assert claimed is not None
    result = DatabaseAgentJobHandlers(
        queue=queue,
        store=SqlAgentStore(database.engine),
        code_workflow=SimpleNamespace(),
        maximum_runtime_seconds=60,
    ).agent_research(claimed, lambda: claimed)
    assert result["candidate_id"]
    candidate = SqlResearchStore(database.engine).get_candidate(result["candidate_id"])
    assert candidate.provider == "openclaw_trusted_compiler"
    dispositions = SqlAgentStore(database.engine).records("disposition")
    assert dispositions[-1]["live_eligible"] is False
    assert dispositions[-1]["candidate_id"] == candidate.candidate_id
    configuration = load_split_configuration(Path(__file__).resolve().parents[1] / "config")
    product = next(
        row for row in configuration["products"]["products"] if row["product_id"] == "active_income"
    )
    monkeypatch.setattr(
        "src.services.scheduler.AUTONOMOUS_SCHEDULES", (ScheduleSpec("candidate_evaluation", 1),)
    )
    queue.register_worker(
        worker_id="research",
        node_id="fixture",
        role="research",
        capabilities=("evaluate_candidate",),
        observed_at=now,
    )
    PlatformScheduler(
        engine=database.engine, products={"active_income": product}, node_id="fixture"
    ).run_once(now=now)
    evaluation_job = queue.claim(
        worker_id="research", now=now, lease_seconds=60, names=("evaluate_candidate",)
    )
    assert evaluation_job is not None
    evaluation = DatabaseResearchJobHandlers(
        SqlResearchStore(database.engine),
        configuration=configuration,
        dataset_resolver=CanonicalDatasetResolver(SqlCanonicalDatasetRepository(database.engine)),
    ).evaluate_candidate(evaluation_job, lambda: evaluation_job)
    assert evaluation["candidate_id"] == candidate.candidate_id
    assert not evaluation.get("deferred"), evaluation
    assert evaluation["outcome_category"] in {"accepted", "rejected"}, evaluation
    assert SqlResearchStore(database.engine).runs(candidate.candidate_id)
    assert any(
        row.get("state") == "evaluated" and row.get("candidate_id") == candidate.candidate_id
        for row in SqlAgentStore(database.engine).records("review")
    )


def test_healthz_is_loopback_minimal_and_keeps_operator_routes_authenticated(
    tmp_path, monkeypatch, capsys
):
    from unittest.mock import Mock

    from scripts.check_platform_health import main as health_check_main
    from src.data.atomic import write_json_atomic
    from src.services.config import PLATFORM_PROCESS_SERVICES
    from src.services.control_api import _is_loopback_peer

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'healthz-healthy.sqlite3'}")
    database.create_schema()
    now = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    heartbeats = DatabaseHeartbeatStore(database.engine)
    required_services = {
        service
        for group in ("market-gateway", "data-writer", "trading-runtime", "scheduler-operations")
        for service in PLATFORM_PROCESS_SERVICES[group]
        if service != "report-worker"
    } | {"research-worker", "evaluation-worker"}
    for service in required_services:
        heartbeats.record(
            service_name=service,
            node_id="local",
            observed_at=now,
            healthy=True,
            payload={"private_marker": "must-not-appear"},
        )
    latest = Mock(wraps=heartbeats.latest)
    monkeypatch.setattr(heartbeats, "latest", latest)
    report_root = tmp_path / "runtime" / "reports"
    report_root.mkdir(parents=True)
    write_json_atomic(
        report_root / "latest.json",
        {
            "schema": "platform.operator_report/v1",
            "generated_at": now,
            "trading": {},
            "research": {},
            "products": {},
            "operations": {
                "disk": {"healthy": True},
                "backups": {"healthy": True},
                "alert_delivery_failure_count": 0,
                "slis": {
                    "unresolved_recovery_count": 0,
                    "stale_account_authority": {"count": 0},
                    "stale_market_data": {"count": 0},
                    "missing_risk_data": {"count": 0},
                    "execution_authority_conflicts": {"conflict": False},
                },
            },
        },
    )
    server = build_control_server(
        bind=("127.0.0.1", 0),
        control_plane=DatabaseControlPlane(database.engine, heartbeats, report_root=report_root),
        bearer_token="operator-health-test-token",
        node_id="local",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/healthz"
    try:
        with urlopen(url, timeout=2) as response:
            assert response.status == 200
            body = response.read().decode()
        payload = json.loads(body)
        assert set(payload) == {
            "schema",
            "ok",
            "checked_at",
            "reason_codes",
            "unhealthy_services",
        }
        assert payload["schema"] == "platform.healthz/v1"
        assert payload["ok"] is True
        assert payload["unhealthy_services"] == []
        assert "must-not-appear" not in body
        assert latest.call_count == 1
        assert latest.call_args.kwargs == {"include_payload": False}

        assert health_check_main(["--url", url]) == 0
        assert capsys.readouterr().out == "NO_REPLY\n"
        assert latest.call_count == 1
        for path in ("/health", "/status", "/reports", "/configuration", "/agent/reviews"):
            with pytest.raises(HTTPError) as denied:
                urlopen(f"http://127.0.0.1:{server.server_port}{path}", timeout=2)
            assert denied.value.code == 401
        mutation = Request(
            f"http://127.0.0.1:{server.server_port}/pause",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as denied:
            urlopen(mutation, timeout=2)
        assert denied.value.code == 401
        assert _is_loopback_peer(("127.0.0.1", 0)) is True
        assert _is_loopback_peer(("203.0.113.1", 0)) is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        database.dispose()


def test_heartbeat_same_time_tie_prefers_unhealthy_record(tmp_path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'heartbeat-tie.sqlite3'}")
    database.create_schema()
    store = DatabaseHeartbeatStore(database.engine)
    observed_at = "2026-08-30T00:00:00+00:00"
    store.record(
        service_name="research-worker",
        node_id="fixture",
        observed_at=observed_at,
        healthy=True,
        payload={"state": "healthy"},
    )
    store.record(
        service_name="research-worker",
        node_id="fixture",
        observed_at=observed_at,
        healthy=False,
        payload={"state": "unhealthy"},
    )

    latest = store.latest()

    assert len(latest) == 1
    assert latest[0].healthy is False
    assert latest[0].payload == {"state": "unhealthy"}
    database.dispose()


def test_healthz_fails_on_stale_inputs_critical_slis_and_missing_status(tmp_path, capsys):
    from scripts.check_platform_health import main as health_check_main
    from src.data.atomic import write_json_atomic
    from src.services.config import PLATFORM_PROCESS_SERVICES

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'healthz-failures.sqlite3'}")
    database.create_schema()
    now_dt = dt.datetime.now(dt.UTC).replace(microsecond=0)
    now = now_dt.isoformat()
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    required_services = {
        service
        for group in ("market-gateway", "data-writer", "trading-runtime", "scheduler-operations")
        for service in PLATFORM_PROCESS_SERVICES[group]
        if service != "report-worker"
    } | {"research-worker", "evaluation-worker"}
    for service in required_services:
        if service == "research-worker":
            continue
        observed_at = now
        if service == "portfolio-engine":
            observed_at = (now_dt - dt.timedelta(seconds=61)).isoformat()
        heartbeat_store.record(
            service_name=service,
            node_id="local",
            observed_at=observed_at,
            healthy=service != "evaluation-worker",
        )
    heartbeat_store.record(
        service_name="research-worker",
        node_id="remote",
        observed_at=now,
        healthy=True,
    )
    report_root = tmp_path / "runtime" / "reports"
    report_root.mkdir(parents=True)
    report = {
        "schema": "platform.operator_report/v1",
        "generated_at": now,
        "trading": {},
        "research": {},
        "products": {},
        "operations": {
            "disk": {"healthy": False},
            "backups": {"healthy": False},
            "alert_delivery_failure_count": 0,
            "slis": {
                "unresolved_recovery_count": 0,
                "stale_account_authority": {"count": 0},
                "stale_market_data": {"count": 0},
                "missing_risk_data": {"count": 1},
                "execution_authority_conflicts": {"conflict": False},
            },
        },
    }
    destination = report_root / "latest.json"
    write_json_atomic(destination, report)
    server = build_control_server(
        bind=("127.0.0.1", 0),
        control_plane=DatabaseControlPlane(
            database.engine, heartbeat_store, report_root=report_root
        ),
        bearer_token="operator-health-test-token",
        node_id="local",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/healthz"
    try:
        with pytest.raises(HTTPError) as failed:
            urlopen(url, timeout=2)
        assert failed.value.code == 503
        first = json.loads(failed.value.read())
        assert {
            "heartbeat_stale",
            "heartbeat_missing",
            "heartbeat_unhealthy",
            "risk_data_missing",
            "backup_unhealthy",
            "disk_unhealthy",
        }.issubset(first["reason_codes"])
        assert first["unhealthy_services"] == [
            "evaluation-worker",
            "portfolio-engine",
            "research-worker",
        ]

        stale_report = {
            **report,
            "generated_at": (now_dt - dt.timedelta(seconds=1_000)).isoformat(),
        }
        write_json_atomic(destination, stale_report)
        server.control_plane = DatabaseControlPlane(
            database.engine, heartbeat_store, report_root=report_root
        )
        with pytest.raises(HTTPError) as failed:
            urlopen(url, timeout=2)
        assert failed.value.code == 503
        stale = json.loads(failed.value.read())
        assert "report_stale" in stale["reason_codes"]
        assert health_check_main(["--url", url]) == 1
        assert capsys.readouterr().err == (
            "Platform health check failed: report stale; service heartbeat missing; "
            "service heartbeat stale; service unhealthy.\n"
        )

        incomplete_report = {
            **report,
            "operations": {
                key: value
                for key, value in report["operations"].items()
                if key not in {"disk", "backups"}
            },
        }
        write_json_atomic(destination, incomplete_report)
        server.control_plane = DatabaseControlPlane(
            database.engine, heartbeat_store, report_root=report_root
        )
        with pytest.raises(HTTPError) as failed:
            urlopen(url, timeout=2)
        assert failed.value.code == 503
        incomplete = json.loads(failed.value.read())
        assert {"backup_status_missing", "disk_status_missing"}.issubset(incomplete["reason_codes"])

        negative_count_report = {
            **report,
            "operations": {
                **report["operations"],
                "slis": {
                    **report["operations"]["slis"],
                    "missing_risk_data": {"count": -1},
                },
            },
        }
        write_json_atomic(destination, negative_count_report)
        server.control_plane = DatabaseControlPlane(
            database.engine, heartbeat_store, report_root=report_root
        )
        with pytest.raises(HTTPError) as failed:
            urlopen(url, timeout=2)
        assert failed.value.code == 503
        negative = json.loads(failed.value.read())
        assert "report_invalid" in negative["reason_codes"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        database.dispose()


def test_healthz_database_failure_is_sanitised_and_failure_cache_is_bounded(
    tmp_path, monkeypatch, capsys
):
    from unittest.mock import Mock

    from scripts.check_platform_health import main as health_check_main
    from src.services.control_api import _HEALTHZ_REQUIRED_SERVICES

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'healthz-database-error.sqlite3'}")
    heartbeats = DatabaseHeartbeatStore(database.engine)
    failing_latest = Mock(side_effect=RuntimeError("private database error marker"))
    monkeypatch.setattr(heartbeats, "latest", failing_latest)
    server = build_control_server(
        bind=("127.0.0.1", 0),
        control_plane=DatabaseControlPlane(database.engine, heartbeats),
        bearer_token="operator-health-test-token",
        node_id="local",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/healthz"
    try:
        with pytest.raises(HTTPError) as failed:
            urlopen(url, timeout=2)
        assert failed.value.code == 503
        first_body = failed.value.read().decode()
        first = json.loads(first_body)
        assert first["reason_codes"] == ["health_check_unavailable"]
        assert first["unhealthy_services"] == list(_HEALTHZ_REQUIRED_SERVICES)
        assert "private database error marker" not in first_body

        with pytest.raises(HTTPError) as failed:
            urlopen(url, timeout=2)
        assert failed.value.code == 503
        second_body = failed.value.read().decode()
        assert json.loads(second_body) == first
        assert failing_latest.call_count == 1
        assert failing_latest.call_args.kwargs == {"include_payload": False}

        assert health_check_main(["--url", url]) == 1
        assert capsys.readouterr().err == (
            "Platform health check failed: health check unavailable.\n"
        )
        assert failing_latest.call_count == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        database.dispose()


def test_health_cli_rejects_remote_urls_and_oversized_responses(monkeypatch, capsys):
    from unittest.mock import MagicMock, Mock

    from scripts import check_platform_health

    opener = Mock()
    build_opener = Mock(return_value=opener)
    monkeypatch.setattr(check_platform_health, "build_opener", build_opener)
    assert check_platform_health.main(["--url", "http://203.0.113.1:8088/healthz"]) == 1
    build_opener.assert_not_called()

    response = MagicMock()
    response.status = 200
    response.read.return_value = b"x" * (check_platform_health.MAX_RESPONSE_BYTES + 1)
    response.__enter__.return_value = response
    opener.open.return_value = response
    assert check_platform_health.main(["--url", "http://127.0.0.1:9999/healthz"]) == 1
    opener.open.assert_called_once_with("http://127.0.0.1:9999/healthz", timeout=10)
    response.read.assert_called_once_with(check_platform_health.MAX_RESPONSE_BYTES + 1)
    response.read.reset_mock()
    response.read.return_value = json.dumps(
        {
            "schema": "platform.healthz/v1",
            "ok": False,
            "reason_codes": ["private account details"],
        }
    ).encode()
    assert check_platform_health.main(["--url", "http://127.0.0.1:9999/healthz"]) == 1
    output = capsys.readouterr()
    assert output.err == (
        "Platform health check failed.\n"
        "Platform health check failed.\n"
        "Platform health check failed.\n"
    )
    assert "private account details" not in output.err


def test_health_cli_does_not_follow_loopback_redirects(tmp_path, capsys):
    from functools import partial
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    from scripts.check_platform_health import main as health_check_main

    root = tmp_path / "redirect-server"
    (root / "healthz").mkdir(parents=True)
    (root / "healthz" / "index.html").write_text(
        json.dumps({"schema": "platform.healthz/v1", "ok": True}), encoding="utf-8"
    )
    server = HTTPServer(
        ("127.0.0.1", 0),
        partial(SimpleHTTPRequestHandler, directory=str(root)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/healthz"
    try:
        assert health_check_main(["--url", url]) == 1
        output = capsys.readouterr()
        assert output.out == ""
        assert output.err.splitlines()[-1] == "Platform health check failed."
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_data_blocked_agent_job_resumes_with_a_later_bundle_without_spending_retries(
    tmp_path, monkeypatch
):
    from sqlalchemy import func, select

    from src.data.database import experiment, job
    from src.services.research_worker import ResearchWorker

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'resumption.sqlite3'}")
    database.create_schema()
    queue, store = DatabaseJobQueue(database.engine), SqlAgentStore(database.engine)
    proposal = OpenClawAgentBridge(store=store, queue=queue).ingest(_payload())
    clock = [NOW]
    monkeypatch.setattr("src.services.agent_worker.utc_now", lambda: clock[0])
    monkeypatch.setattr("src.services.research_worker.utc_now", lambda: clock[0])
    for name in ("EXCHANGE_API_KEY", "EXCHANGE_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    queue.register_worker(
        worker_id="agent",
        node_id="fixture",
        role="agent-sandbox",
        capabilities=("agent_research",),
        observed_at=NOW,
    )
    handlers = DatabaseAgentJobHandlers(
        queue=queue, store=store, code_workflow=SimpleNamespace(), maximum_runtime_seconds=60
    )
    worker = ResearchWorker(
        runtime=SimpleNamespace(
            node=SimpleNamespace(operating_system="linux"), can_submit_orders=False
        ),
        queue=queue,
        worker_id="agent",
        handlers={"agent_research": handlers.agent_research},
        lease_seconds=30,
    )
    waiting = worker.run_once(now=clock[0])
    assert waiting["reason_code"] == "research_job_deferred"
    assert waiting["outcome_category"] == "blocked_data"
    with database.engine.connect() as connection:
        pending = (
            connection.execute(select(job).where(job.c.id == f"agent:{proposal.proposal_id}"))
            .mappings()
            .one()
        )
    assert pending["state"] == "pending" and pending["attempts"] == 0
    clock[0] = (dt.datetime.fromisoformat(NOW) + dt.timedelta(minutes=1)).isoformat()
    bundle = _bundle(database, created_at=clock[0])
    completed = worker.run_once(now=clock[0])
    candidate_id = completed["candidate_id"]
    assert store.dataset_binding(proposal.proposal_id)["bundle_id"] == bundle.bundle_id
    candidate = SqlResearchStore(database.engine).get_candidate(candidate_id)
    assert candidate.submitted_at == clock[0]
    clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(minutes=1)).isoformat()
    _bundle(database, created_at=clock[0])
    replay = handlers.agent_research(
        SimpleNamespace(payload={"proposal_id": proposal.proposal_id}), lambda: None
    )
    assert replay["candidate_id"] == candidate_id
    with database.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(experiment)).scalar_one() == 1
        assert (
            connection.execute(
                select(job.c.state).where(job.c.id == f"agent:{proposal.proposal_id}")
            ).scalar_one()
            == "completed"
        )
    database.dispose()


def test_control_api_heartbeats_and_closes_both_servers_on_sigterm(tmp_path, monkeypatch) -> None:
    import os
    import signal
    import socket
    from unittest.mock import Mock

    from src.services import supervisor

    ports = []
    for _ in range(2):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        ports.append(probe.getsockname()[1])
        probe.close()
    control_port, metrics_port = ports
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'control-api-lifecycle.sqlite3'}")
    database.create_schema()
    original_dispose = database.dispose
    dispose = Mock(wraps=original_dispose)
    monkeypatch.setattr(database, "dispose", dispose)
    heartbeats = []
    runtime = Mock()
    runtime.heartbeat.side_effect = lambda *, payload: (
        heartbeats.append(payload),
        os.kill(os.getpid(), signal.SIGTERM) if len(heartbeats) == 2 else None,
    )

    monkeypatch.setenv("CONTROL_API_TEST_TOKEN", "operator-test-token")
    context = {
        "database": database,
        "config": SimpleNamespace(
            paths={"parquet": str(tmp_path)},
            metrics={
                "bind": f"127.0.0.1:{metrics_port}",
                "heartbeat_seconds": 0.05,
            },
        ),
        "heartbeat_store": DatabaseHeartbeatStore(database.engine),
        "runtime": runtime,
        "split": {},
        "alerts": None,
    }
    args = SimpleNamespace(
        control_bind=f"127.0.0.1:{control_port}",
        control_token_env="CONTROL_API_TEST_TOKEN",
        node="fixture",
    )

    assert supervisor._serve_control_api(args, context) == 0
    assert [item["reason_code"] for item in heartbeats] == [
        "control_api_started",
        "control_api_serving",
    ]
    assert runtime.heartbeat.call_count == 2
    assert dispose.call_count == 1
    for port in (control_port, metrics_port):
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_metrics_server_requires_the_operator_bearer_token() -> None:
    from unittest.mock import Mock
    from urllib.error import HTTPError

    from src.observability.metrics import build_metrics_server

    provider = Mock(return_value="private_research_metric 1\n")
    server = build_metrics_server(
        bind=("127.0.0.1", 0),
        provider=provider,
        bearer_token="operator-metrics-test-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/metrics"
    try:
        with pytest.raises(HTTPError) as denied:
            urlopen(url, timeout=2)
        assert denied.value.code == 401
        provider.assert_not_called()
        request = Request(url, headers={"Authorization": "Bearer operator-metrics-test-token"})
        with urlopen(request, timeout=2) as response:
            assert response.status == 200
            assert response.read().decode() == "private_research_metric 1\n"
        provider.assert_called_once_with()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_control_api_metrics_server_failure_closes_control_socket_and_database(
    tmp_path, monkeypatch
) -> None:
    import socket
    from unittest.mock import Mock

    from src.services import supervisor

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    control_port = probe.getsockname()[1]
    probe.close()
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'metrics-bind-failure.sqlite3'}")
    database.create_schema()
    dispose = Mock(wraps=database.dispose)
    monkeypatch.setattr(database, "dispose", dispose)
    monkeypatch.setattr(
        supervisor, "build_metrics_server", Mock(side_effect=OSError("bind failed"))
    )
    monkeypatch.setenv("CONTROL_API_TEST_TOKEN", "operator-test-token")
    config = SimpleNamespace(
        paths={"parquet": str(tmp_path)},
        metrics={"bind": "127.0.0.1:9108", "heartbeat_seconds": 1},
    )
    context = {
        "database": database,
        "config": config,
        "heartbeat_store": DatabaseHeartbeatStore(database.engine),
        "runtime": Mock(),
        "split": {},
        "alerts": None,
    }
    args = SimpleNamespace(
        control_bind=f"127.0.0.1:{control_port}",
        control_token_env="CONTROL_API_TEST_TOKEN",
        node="fixture",
    )

    with pytest.raises(OSError, match="bind failed"):
        supervisor._serve_control_api(args, context)

    assert dispose.call_count == 1
    with socket.socket() as closed_probe:
        assert closed_probe.connect_ex(("127.0.0.1", control_port)) != 0


def test_scheduled_review_producer_reader_and_retry_share_one_context_identity(
    tmp_path, monkeypatch
):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'review.sqlite3'}")
    database.create_schema()
    queue, store = DatabaseJobQueue(database.engine), SqlAgentStore(database.engine)
    handlers = DatabaseAgentJobHandlers(
        queue=queue, store=store, code_workflow=SimpleNamespace(), maximum_runtime_seconds=60
    )
    clock = [NOW]
    monkeypatch.setattr("src.services.agent_worker.utc_now", lambda: clock[0])
    claimed = SimpleNamespace(
        job_id="scheduled-review-1", payload={"product_id": "active_income", "available_at": NOW}
    )
    first = handlers.agent_review(claimed, lambda: None)
    bridge = OpenClawAgentBridge(store=store, queue=queue)
    request = bridge.review_request(first["review_id"])
    assert request["context_hash"] == first["context_hash"]
    assert request["context"]["data_availability"]["hypothesis_inputs_ready"] is False
    assert request["context"]["strategy_catalogue"]["families"] == []
    clock[0] = (dt.datetime.fromisoformat(NOW) + dt.timedelta(minutes=1)).isoformat()
    assert handlers.agent_review(claimed, lambda: None) == first
    assert len(store.records("review")) == 1
    claimed.job_id = "scheduled-review-2"
    second = handlers.agent_review(claimed, lambda: None)
    assert second["review_id"] != first["review_id"]
    assert bridge.review_request(second["review_id"])["context_hash"] == second["context_hash"]
    database.dispose()


def test_scheduled_reviews_cover_both_products_without_duplicate_jobs(tmp_path, monkeypatch):
    from pathlib import Path

    from sqlalchemy import select

    from src.data.database import job
    from src.services.config import load_split_configuration
    from src.services.scheduler import PlatformScheduler, ScheduleSpec

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'review-products.sqlite3'}")
    database.create_schema()
    configuration = load_split_configuration(Path(__file__).resolve().parents[1] / "config")
    products = {row["product_id"]: row for row in configuration["products"]["products"]}
    monkeypatch.setattr(
        "src.services.scheduler.AUTONOMOUS_SCHEDULES", (ScheduleSpec("agent_review", 3600),)
    )
    scheduler = PlatformScheduler(engine=database.engine, products=products, node_id="fixture")
    scheduler.run_once(now=NOW)
    scheduler.run_once(now=NOW)
    with database.engine.connect() as connection:
        payloads = (
            connection.execute(select(job.c.payload).where(job.c.name == "agent_review"))
            .scalars()
            .all()
        )
    assert sorted(payload["product_id"] for payload in payloads) == [
        "active_income",
        "btc_accumulation",
    ]
    database.dispose()


def test_scheduled_agent_review_exposes_adaptive_failures_without_holdout_data() -> None:
    context = build_agent_review_context(
        {
            "product_id": "active_income",
            "research": {
                "funnel": {
                    "top_rejection_reasons": {"negative_return": 3},
                    "feature_family_concentration": {"time_series": 4},
                    "exact_duplicate_rate": 0.25,
                    "near_duplicate_rate": 0.10,
                    "candidates_generated": 8,
                    "candidates_evaluated": 5,
                    "candidates_never_evaluated_count": 3,
                    "theses_generated": 7,
                    "cumulative_trial_count": 11,
                    "candidates_rejected_by_stage": {"development": 3},
                }
            },
        },
        created_at=NOW,
        product_id="active_income",
    )

    assert context.values["failure_reasons"] == {"negative_return": 3}
    assert "holdout_outcomes" not in context.values
    assert "protected_holdout" not in str(context.values).lower()
    assert context.values["data_availability"]["hypothesis_inputs_ready"] is False


def test_review_context_uses_supported_product_catalogue_without_protected_payloads(tmp_path):
    from sqlalchemy import delete, insert

    from src.data.database import (
        dataset_snapshot,
        experiment,
        experiment_run,
        instrument,
        strategy_definition,
        strategy_version,
        validation_stage,
    )
    from src.research.economic_series import ECONOMIC_ENGINE_VERSION

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'review-context.sqlite3'}")
    database.create_schema()
    scopes = {
        "btc_accumulation": ("binance:spot:BTCUSDT",),
        "active_income": (
            "binance:futures:BTCUSDT:USDT",
            "binance:futures:ETHUSDT:USDT",
        ),
    }
    instrument_rows = [
        {
            "id": instrument_id,
            "venue": "binance",
            "market_type": "spot" if product_id == "btc_accumulation" else "futures",
            "exchange_symbol": instrument_id.split(":")[2],
            "base_asset": instrument_id.split(":")[2].removesuffix("USDT"),
            "quote_asset": "USDT",
            "settlement_asset": "USDT" if product_id == "active_income" else None,
            "payload": {},
        }
        for product_id, instrument_ids in scopes.items()
        for instrument_id in instrument_ids
    ]
    bundles = {}
    for product_id, instrument_ids in scopes.items():
        bundles[product_id] = CanonicalResearchDatasetBuilder(database.engine).build(
            product_id,
            intervals={
                role: {
                    "start": f"2026-08-{20 + index:02d}T00:00:00+00:00",
                    "end": f"2026-08-{21 + index:02d}T00:00:00+00:00",
                }
                for index, role in enumerate(RESEARCH_BUNDLE_ROLES)
            },
            payload_by_role={
                role: {
                    "timeframe": "1h" if product_id == "btc_accumulation" else "15m",
                    "bars": [{"close": 100.0}],
                    "raw_marker": "PROTECTED_PAYLOAD_SENTINEL",
                }
                for role in RESEARCH_BUNDLE_ROLES
            },
            universe_snapshot_id=canonical_hash({"universe": product_id}),
            feature_manifest_id=canonical_hash({"features": "review"}),
            cost_model_id=canonical_hash({"costs": "review"}),
            parameter_set_id=canonical_hash({"parameters": "review"}),
            instrument_scope=instrument_ids,
            availability_timestamp=NOW,
            created_at=NOW,
        )

    definitions = [
        {
            "id": "review-active-definition",
            "identity": "review-active-identity",
            "product_id": "active_income",
            "source_type": "generated_dsl",
            "source_hash": canonical_hash({"definition": "active"}),
            "definition": {
                "family": "time_series",
                "feature_graph": {"required_nodes": ["trend", "returns", "secret_feature"]},
                "metadata": {"api_secret": "PRIVATE_FIELD_SENTINEL"},
            },
        },
        {
            "id": "review-btc-definition",
            "identity": "review-btc-identity",
            "product_id": "btc_accumulation",
            "source_type": "generated_dsl",
            "source_hash": canonical_hash({"definition": "btc"}),
            "definition": {
                "family": "time_series",
                "feature_graph": {"required_nodes": ["secret_feature"]},
                "metadata": {"api_secret": "PRIVATE_FIELD_SENTINEL"},
            },
        },
    ]
    versions = [
        {
            "id": f"{definition['id']}-version",
            "definition_id": definition["id"],
            "version": "review-v1",
            "created_at": NOW,
            "payload": {},
        }
        for definition in definitions
    ]
    candidate_rows = [
        {
            "id": f"review-{product_id}-candidate",
            "strategy_version_id": (
                "review-active-definition-version"
                if product_id == "active_income"
                else "review-btc-definition-version"
            ),
            "provider": "fixture",
            "state": "submitted",
            "submitted_at": NOW,
            "dataset_snapshot_hashes": [],
            "dataset_bundle_id": bundles[product_id].bundle_id,
            "metadata": {"raw_price": "RAW_PRICE_SENTINEL"},
        }
        for product_id in scopes
    ]
    run_rows = [
        {"id": f"review-{product_id}-{stage}-run", "created_at": NOW, "payload": {}}
        for product_id, stages in {
            "active_income": ("screening", "development"),
            "btc_accumulation": ("protected_holdout",),
        }.items()
        for stage in stages
    ]
    stage_rows = []
    for product_id, stages in {
        "active_income": (("screening", True, None), ("development", False, "negative_return")),
        "btc_accumulation": (("protected_holdout", False, "protected_result_secret"),),
    }.items():
        for stage, accepted, reason in stages:
            stage_rows.append(
                {
                    "id": f"review-{product_id}-{stage}",
                    "experiment_id": f"review-{product_id}-candidate",
                    "stage": stage,
                    "source_run_id": f"review-{product_id}-{stage}-run",
                    "evaluated_at": NOW,
                    "state": "evaluated",
                    "accepted": accepted,
                    "reason_code": reason,
                    "evidence_hash": canonical_hash({"stage": stage}),
                    "evaluation_identity": f"review-{product_id}-{stage}-identity",
                    "engine_version": ECONOMIC_ENGINE_VERSION,
                    "outcome_category": "accepted" if accepted else "rejected",
                    "payload": {
                        "raw_price": "RAW_PRICE_SENTINEL",
                        "protected_result": "PROTECTED_RESULT_SENTINEL",
                        "api_secret": "PRIVATE_FIELD_SENTINEL",
                    },
                }
            )

    with database.engine.begin() as connection:
        connection.execute(insert(instrument), instrument_rows)
        connection.execute(insert(strategy_definition), definitions)
        connection.execute(insert(strategy_version), versions)
        connection.execute(insert(experiment), candidate_rows)
        connection.execute(insert(experiment_run), run_rows)
        connection.execute(insert(validation_stage), stage_rows)

    active_report = adaptive_research_report(database.engine, product_id="active_income", at=NOW)
    btc_report = adaptive_research_report(database.engine, product_id="btc_accumulation", at=NOW)
    active_context = build_agent_review_context(
        active_report, created_at=NOW, product_id="active_income"
    )
    btc_context = build_agent_review_context(
        btc_report, created_at=NOW, product_id="btc_accumulation"
    )

    active = active_context.values
    assert active["product_id"] == "active_income"
    assert active["data_availability"]["hypothesis_inputs_ready"] is True
    assert active["data_availability"]["available_data"] == ["closed_ohlcv_bars"]
    assert active["instrument_universe"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert active["instrument_universe"]["venue"] == "binance"
    assert active["instrument_universe"]["market_type"] == "futures"
    assert active["instrument_universe"]["settlement_asset"] == "USDT"
    assert active["strategy_catalogue"]["evidence_type"] == "swing"
    assert active["strategy_catalogue"]["families"] == ["mean_reversion", "time_series"]
    assert active["feature_catalogue"]["supported_by_family"] == {
        "mean_reversion": ["normalised_price_deviation", "oscillator", "range_state"],
        "time_series": ["breakout", "realised_volatility", "returns", "trend"],
    }
    assert active["feature_catalogue"]["publicly_observed_by_stage"]["development"] == {
        "time_series": ["returns", "trend"]
    }
    assert active["development_results"]["development_evaluated"] == 1
    assert active["failure_reasons"] == {"negative_return": 1}

    btc = btc_context.values
    assert btc["product_id"] == "btc_accumulation"
    assert btc["data_availability"]["hypothesis_inputs_ready"] is True
    assert btc["instrument_universe"]["symbols"] == ["BTCUSDT"]
    assert btc["instrument_universe"]["venue"] == "binance"
    assert btc["instrument_universe"]["market_type"] == "spot"
    assert btc["strategy_catalogue"]["evidence_type"] == "btc_allocation"
    assert btc["strategy_catalogue"]["families"]
    assert btc["feature_catalogue"]["supported_by_family"] == {
        "mean_reversion": ["normalised_price_deviation", "oscillator", "range_state"],
        "time_series": ["breakout", "realised_volatility", "returns", "trend"],
    }
    assert btc["development_results"]["development_evaluated"] == 0
    assert btc["feature_catalogue"]["publicly_observed_by_stage"]["development"] == {}
    mismatched = build_agent_review_context(
        active_report, created_at=NOW, product_id="btc_accumulation"
    )
    assert "product_id" not in mismatched.values
    assert mismatched.values["data_availability"]["hypothesis_inputs_ready"] is False
    assert mismatched.values["strategy_catalogue"]["families"] == []
    for context in (active, btc):
        serialised = json.dumps(context)
        for sentinel in (
            "PRIVATE_FIELD_SENTINEL",
            "RAW_PRICE_SENTINEL",
            "PROTECTED_PAYLOAD_SENTINEL",
            "PROTECTED_RESULT_SENTINEL",
            "secret_feature",
        ):
            assert sentinel not in serialised
    before_bundle = adaptive_research_report(
        database.engine, product_id="active_income", at="2026-08-29T23:59:59+00:00"
    )
    assert before_bundle["research"]["product_inputs"]["hypothesis_inputs_ready"] is False
    with database.engine.begin() as connection:
        connection.execute(
            delete(dataset_snapshot).where(
                dataset_snapshot.c.id == bundles["btc_accumulation"].stage_snapshot_ids["screening"]
            )
        )
    missing_stage = adaptive_research_report(database.engine, product_id="btc_accumulation", at=NOW)
    assert missing_stage["research"]["product_inputs"]["hypothesis_inputs_ready"] is False
    database.dispose()


def test_native_research_profile_rejects_plugin_overrides(tmp_path, monkeypatch):
    import sqlite3

    from src.agents.consumer import isolated_agent_environment

    state = tmp_path / "state-root"
    workspace = state / "research-workspace"
    agent = state / "agents/research/agent"
    workspace.mkdir(parents=True)
    agent.mkdir(parents=True)
    private = tmp_path / "private" / "openclaw-state"
    private.mkdir(parents=True)
    (state / "state").symlink_to(private, target_is_directory=True)
    monkeypatch.setenv("TRADING_PLATFORM_OPENCLAW_PRIVATE_STATE", str(private))
    registry = state / "state/openclaw.sqlite"
    with sqlite3.connect(registry) as connection:
        connection.execute(
            "CREATE TABLE installed_plugin_index (index_key TEXT, install_records_json TEXT, plugins_json TEXT)"
        )
        connection.execute(
            "INSERT INTO installed_plugin_index VALUES ('installed-plugin-index', '{}', '[]')"
        )
    config = {
        "plugins": {
            "enabled": True,
            "allow": ["openai"],
            "load": {"paths": []},
            "entries": {"openai": {"enabled": True}},
        },
        "hooks": {"enabled": False},
        "models": {
            "providers": {
                "openai": {
                    "auth": "oauth",
                    "api": "openai-chatgpt-responses",
                    "baseUrl": "https://chatgpt.com/backend-api/codex",
                    "agentRuntime": {"id": "openclaw"},
                    "models": [{"id": "gpt-5.6-luna", "agentRuntime": {"id": "openclaw"}}],
                }
            }
        },
        "agents": {
            "defaults": {"models": {"openai/gpt-5.6-luna": {"agentRuntime": {"id": "openclaw"}}}},
            "list": [
                {
                    "id": "research",
                    "workspace": str(workspace),
                    "agentDir": str(agent),
                    "tools": {"deny": ["*"]},
                    "skills": [],
                    "sandbox": {"mode": "off", "workspaceAccess": "none"},
                }
            ],
        },
    }
    config["session"] = {"store": str(private.parent / "sessions/{agentId}/sessions.json")}
    path = tmp_path / "research.json"
    path.write_text(json.dumps(config))
    environment = isolated_agent_environment(path, agent_id="research", state_dir=state)
    assert environment["OPENCLAW_DISABLE_BUNDLED_SOURCE_OVERLAYS"] == "1"
    config["plugins"]["load"]["paths"] = ["/tmp/custom-provider"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="only the bundled"):
        isolated_agent_environment(path, agent_id="research", state_dir=state)
    config["plugins"]["load"]["paths"] = []
    path.write_text(json.dumps(config))
    with sqlite3.connect(registry) as connection:
        connection.execute(
            "UPDATE installed_plugin_index SET install_records_json=?",
            (json.dumps({"openai": {"source": "npm"}}),),
        )
    with pytest.raises(ValueError, match="overridden"):
        isolated_agent_environment(path, agent_id="research", state_dir=state)
