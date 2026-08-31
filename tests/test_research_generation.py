from __future__ import annotations

import datetime as dt
from dataclasses import replace

from src.data.database import PlatformDatabase
from src.domain._codec import canonical_hash
from src.research.coordinator import ResearchCoordinator
from src.research.datasets import RESEARCH_BUNDLE_ROLES, CanonicalResearchDatasetBuilder
from src.research.generation import (
    CAMPAIGNS,
    GenerationAllocator,
    GenerationFeedback,
    HypothesisGenerator,
    SqlGenerationFeedbackStore,
    SqlHypothesisMemory,
    build_hypothesis,
    hypothesis_signature,
    semantic_distance,
)
from src.research.store import SqlResearchStore
from src.research.theses import SqlThesisRegistry, ThesisError, ThesisRegistry
from src.services.research_jobs import DatabaseResearchJobHandlers
from src.services.scheduler import ClaimedJob

NOW = dt.datetime(2026, 8, 23, tzinfo=dt.UTC).isoformat()


def test_campaign_catalogue_covers_the_declared_research_families() -> None:
    names = {campaign.name for campaign in CAMPAIGNS}

    assert {
        "btc_risk_off_reentry",
        "futures_breakout",
        "futures_mean_reversion",
        "cross_sectional_momentum",
        "funding_carry",
        "basis_convergence",
        "pairs_mean_reversion",
        "event_microstructure",
        "ensemble_regime",
    } <= names
    assert {campaign.evidence_type for campaign in CAMPAIGNS} >= {
        "btc_allocation",
        "swing",
        "intraday",
        "cross_sectional",
        "funding_carry",
        "pairs",
        "scalping",
    }
    assert all(campaign.required_data for campaign in CAMPAIGNS)


def _database(tmp_path) -> PlatformDatabase:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'generation.sqlite3'}")
    database.create_schema()
    return database


def test_campaign_allocator_keeps_exploration_and_reduces_failed_family(tmp_path) -> None:
    campaigns = CAMPAIGNS[:3]
    feedback = tuple(
        GenerationFeedback(
            campaign="btc_pullback_reversion",
            outcome="rejected",
            reason_code="negative_control_failed",
            observed_at=NOW,
        )
        for _ in range(4)
    )
    allocations = GenerationAllocator(minimum_exploration_fraction=0.3).allocate(
        campaigns,
        total_budget=12,
        feedback=feedback,
    )
    by_name = {item.campaign: item for item in allocations}
    assert sum(item.trials for item in allocations) == 12
    assert all(item.trials >= 1 for item in allocations)
    assert by_name["btc_pullback_reversion"].trials < by_name["btc_trend_breakout"].trials


def test_generation_feedback_is_append_only_and_idempotent(tmp_path) -> None:
    database = _database(tmp_path)
    store = SqlGenerationFeedbackStore(database.engine)
    feedback = GenerationFeedback(
        campaign="futures_trend_following",
        outcome="duplicate_near",
        candidate_id=canonical_hash({"candidate": "duplicate"}),
        semantic_signature=canonical_hash({"semantic": "trend"}),
        distance=0.125,
        observed_at=NOW,
        metadata={"source": "test"},
    )
    assert store.append(feedback) == feedback.feedback_id
    assert store.append(feedback) == feedback.feedback_id
    assert store.load(campaign=feedback.campaign) == (feedback,)


def test_hypothesis_memory_detects_exact_and_near_duplicates(tmp_path) -> None:
    database = _database(tmp_path)
    campaign = CAMPAIGNS[0]
    snapshots = (canonical_hash({"snapshot": "screening"}),)
    first = build_hypothesis(
        campaign,
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=snapshots,
        submitted_at=NOW,
    )
    second = build_hypothesis(
        campaign,
        variant=1,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=snapshots,
        submitted_at=NOW,
    )
    SqlThesisRegistry(database.engine).register(first.thesis)
    coordinator = ResearchCoordinator(SqlResearchStore(database.engine))
    coordinator.submit(first.candidate)
    memory = SqlHypothesisMemory(database.engine)
    near = memory.find(second.candidate)
    assert near is not None
    assert near.kind == "near"
    assert near.candidate_id == first.candidate.candidate_id
    assert hypothesis_signature(first.candidate.definition) != hypothesis_signature(
        second.candidate.definition
    )
    assert 0.0 < semantic_distance(first.candidate.definition, second.candidate.definition) <= 0.2
    exact = memory.find(first.candidate)
    assert exact is not None
    assert exact.kind == "exact"


def test_generated_thesis_and_candidate_are_product_and_universe_bound() -> None:
    hypothesis = build_hypothesis(
        CAMPAIGNS[-1],
        variant=2,
        instrument_universe=("ETHUSDT", "BTCUSDT"),
        dataset_snapshot_hashes=(canonical_hash({"snapshot": "event"}),),
        submitted_at=NOW,
        universe_snapshot_id=canonical_hash({"universe": "point-in-time"}),
    )
    assert hypothesis.thesis.generalisation_scope["product"] == "active_income"
    assert hypothesis.candidate.definition.universe["type"] == "point_in_time"
    assert hypothesis.candidate.definition.signal_model["rule"]["direction"] == "signed"
    assert hypothesis.semantic_signature == hypothesis_signature(hypothesis.candidate.definition)


def test_hypothesis_generator_records_and_skips_an_exact_duplicate(tmp_path) -> None:
    database = _database(tmp_path)
    feedback_store = SqlGenerationFeedbackStore(database.engine)
    first = build_hypothesis(
        CAMPAIGNS[0],
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=(canonical_hash({"snapshot": "screening"}),),
        submitted_at=NOW,
    )
    SqlThesisRegistry(database.engine).register(first.thesis)
    ResearchCoordinator(SqlResearchStore(database.engine)).submit(first.candidate)
    generated = HypothesisGenerator(
        product="btc_accumulation",
        instrument_universe=("BTCUSDT",),
        memory=SqlHypothesisMemory(database.engine),
        feedback_store=feedback_store,
    ).generate(
        dataset_snapshot_hashes=first.candidate.dataset_snapshot_hashes,
        submitted_at=NOW,
        total_budget=1,
        campaigns=(CAMPAIGNS[0],),
    )
    assert generated == ()
    assert feedback_store.load(campaign=CAMPAIGNS[0].name)[0].outcome == "duplicate_exact"


def test_generation_worker_persists_a_typed_dataset_plan(tmp_path) -> None:
    database = _database(tmp_path)
    identity = {
        "universe": canonical_hash({"universe": "btc"}),
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
    bundle = CanonicalResearchDatasetBuilder(database.engine).build(
        "btc_accumulation",
        intervals=intervals,
        payload_by_role={role: {"bars": [{"close": 100.0}]} for role in RESEARCH_BUNDLE_ROLES},
        universe_snapshot_id=identity["universe"],
        feature_manifest_id=identity["feature"],
        cost_model_id=identity["cost"],
        parameter_set_id=identity["parameters"],
        instrument_scope=("BTCUSDT",),
        availability_timestamp="2026-08-30T00:00:00+00:00",
        created_at=NOW,
    )
    claimed = ClaimedJob(
        job_id="generation-job",
        name="generate_hypotheses",
        payload={
            "product_id": "btc_accumulation",
            "instrument_universe": ["BTCUSDT"],
            "dataset_snapshot_hashes": list(bundle.stage_snapshot_ids.values()),
            "dataset_bundle_id": bundle.bundle_id,
            "universe_snapshot_id": bundle.universe_snapshot_id,
            "submitted_at": "2026-08-30T00:00:00+00:00",
            "generation_budget": 1,
        },
        worker_id="generation-worker",
        attempt=1,
        lease_expires_at="2026-08-30T00:01:00+00:00",
    )
    result = DatabaseResearchJobHandlers(SqlResearchStore(database.engine)).generate_hypotheses(
        claimed,
        lambda: claimed,
    )
    assert result["candidate_count"] == 1
    candidate = SqlResearchStore(database.engine).get_candidate(result["candidate_ids"][0])
    assert candidate.dataset_bundle_id == bundle.bundle_id
    assert candidate.dataset_plan is not None
    assert (
        candidate.dataset_plan.protected_holdout_snapshot_id
        == bundle.stage_snapshot_ids["protected_holdout"]
    )


def test_catalogue_registration_waits_without_a_ready_dataset() -> None:
    database = PlatformDatabase("sqlite+pysqlite:///:memory:")
    database.create_schema()
    claimed = ClaimedJob(
        job_id="catalogue-job",
        name="register_strategy_catalogue",
        payload={
            "product_id": "btc_accumulation",
            "instrument_universe": ["BTCUSDT"],
            "dataset_snapshot_hashes": [],
            "catalogue_submitted_at": NOW,
        },
        worker_id="catalogue-worker",
        attempt=1,
        lease_expires_at="2026-08-23T00:01:00+00:00",
    )

    result = DatabaseResearchJobHandlers(
        SqlResearchStore(database.engine)
    ).register_strategy_catalogue(
        claimed,
        lambda: claimed,
    )

    assert result == {
        "product_id": "btc_accumulation",
        "state": "waiting_for_dataset",
        "reason_code": "canonical_dataset_bundle_unavailable",
    }


def test_in_memory_thesis_budget_is_shared_by_descendants() -> None:
    root = build_hypothesis(
        CAMPAIGNS[0],
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=(canonical_hash({"snapshot": "root"}),),
        submitted_at=NOW,
    ).thesis
    root = replace(root, cumulative_trial_budget=2)
    child = replace(root, parent_thesis_ids=(root.thesis_id,))
    registry = ThesisRegistry()
    registry.register(root)
    registry.register(child)
    registry.claim_trial(thesis_id=root.thesis_id, candidate_id="root-trial", lineage_id="root")
    registry.claim_trial(
        thesis_id=child.thesis_id,
        candidate_id="child-trial",
        lineage_id="child",
    )
    try:
        registry.claim_trial(
            thesis_id=child.thesis_id,
            candidate_id="third-trial",
            lineage_id="child-2",
        )
    except ThesisError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("descendant thesis must share its parent budget")


def test_sql_thesis_budget_is_shared_by_descendants(tmp_path) -> None:
    database = _database(tmp_path)
    root = build_hypothesis(
        CAMPAIGNS[0],
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=(canonical_hash({"snapshot": "root"}),),
        submitted_at=NOW,
    ).thesis
    root = replace(root, cumulative_trial_budget=2)
    child = replace(root, parent_thesis_ids=(root.thesis_id,))
    registry = SqlThesisRegistry(database.engine)
    registry.register(root)
    registry.register(child)
    registry.claim_trial(
        thesis_id=root.thesis_id,
        candidate_id=canonical_hash({"candidate": "root"}),
        lineage_id=canonical_hash({"lineage": "root"}),
        claimed_at=NOW,
    )
    registry.claim_trial(
        thesis_id=child.thesis_id,
        candidate_id=canonical_hash({"candidate": "child"}),
        lineage_id=canonical_hash({"lineage": "child"}),
        claimed_at=NOW,
    )
    try:
        registry.claim_trial(
            thesis_id=child.thesis_id,
            candidate_id=canonical_hash({"candidate": "third"}),
            lineage_id=canonical_hash({"lineage": "third"}),
            claimed_at=NOW,
        )
    except ThesisError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("descendant thesis must share its parent budget")
