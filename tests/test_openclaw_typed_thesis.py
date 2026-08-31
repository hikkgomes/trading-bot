from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from src.agents.compiler import AgentCompilationError, compile_openclaw_candidate_payload
from src.agents.openclaw_bridge import OpenClawAgentBridge, build_agent_proposal
from src.agents.store import SqlAgentStore
from src.data.database import PlatformDatabase
from src.domain._codec import canonical_hash
from src.research.datasets import RESEARCH_BUNDLE_ROLES, CanonicalResearchDatasetBuilder
from src.research.store import SqlResearchStore
from src.services.agent_worker import DatabaseAgentJobHandlers, build_agent_review_context
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


def _bundle(database: PlatformDatabase):
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
        availability_timestamp=NOW,
        created_at=NOW,
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


def test_openclaw_worker_compiles_a_candidate_into_the_normal_queue(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'openclaw-worker.sqlite3'}")
    database.create_schema()
    _bundle(database)
    queue = DatabaseJobQueue(database.engine)
    OpenClawAgentBridge(
        store=SqlAgentStore(database.engine),
        queue=queue,
    ).ingest(_payload())
    queue.register_worker(
        worker_id="agent-worker",
        node_id="node",
        role="agent-sandbox",
        capabilities=("agent_research",),
        observed_at=NOW,
    )
    claimed = queue.claim(
        worker_id="agent-worker",
        now=NOW,
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
