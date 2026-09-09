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
from src.services.agent_worker import DatabaseAgentJobHandlers, build_agent_review_context
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
    from pathlib import Path

    from test_platform_candidate_lifecycle import _bundle as market_bundle

    from src.agents.consumer import consume_review
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
        workspace = tmp_path / "research-workspace"
        workspace.mkdir()
        agent_config = tmp_path / "openclaw.json"
        agent_config.write_text(
            json.dumps(
                {
                    "agents": {
                        "list": [
                            {
                                "id": "research",
                                "workspace": str(workspace),
                                "tools": {"deny": ["*"]},
                                "sandbox": {"mode": "all", "workspaceAccess": "none"},
                            }
                        ]
                    }
                }
            )
        )
        executable = tmp_path / "openclaw-test-process"
        output = json.dumps({"payloads": [{"text": json.dumps(payload)}]})
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys, uuid\n"
            "assert '--local' in sys.argv and '--deliver' not in sys.argv\n"
            "uuid.UUID(sys.argv[sys.argv.index('--session-id') + 1])\n"
            "assert not any(key.startswith(('TRADING_', 'EXCHANGE_')) for key in os.environ)\n"
            f"print({output!r})\n"
        )
        executable.chmod(0o700)
        monkeypatch.setenv("TRADING_PLATFORM_AGENT_TRANSPORT_KEY", key.hex())
        monkeypatch.setenv("EXCHANGE_API_KEY", "fixture-only-secret")
        now = (dt.datetime.fromisoformat(now) + dt.timedelta(minutes=11)).isoformat()
        received = consume_review(
            api_url=f"http://127.0.0.1:{server.server_port}",
            key=key,
            executable=executable,
            config_path=agent_config,
            agent_id="research",
            prompt="Return one research proposal as JSON.",
        )
        replay = post("/agent/responses", envelope)
        assert replay["proposal_id"] == received["proposal_id"]
        assert received["state"] == "responded"
    finally:
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
            "research": {
                "funnel": {
                    "top_rejection_reasons": {"negative_return": 3},
                    "feature_family_concentration": {"time_series": 4},
                    "exact_duplicate_rate": 0.25,
                    "near_duplicate_rate": 0.10,
                    "missing_stage_datasets": {"robustness": 1},
                    "candidates_generated": 8,
                    "candidates_evaluated": 5,
                    "candidates_never_evaluated": ["candidate-1"],
                    "theses_generated": 7,
                    "cumulative_trial_count": 11,
                    "candidates_rejected_by_stage": {"development": 3},
                    "first_blocked_stage": {"development": 3},
                    "jobs_dead_letter": 1,
                }
            }
        },
        created_at=NOW,
    )

    assert context.values["failure_reasons"] == {"negative_return": 3}
    assert "holdout_outcomes" not in context.values
    assert "protected_holdout" not in str(context.values).lower()
