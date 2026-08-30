from __future__ import annotations

import datetime as dt
from dataclasses import replace

from src.data.database import PlatformDatabase
from src.domain._codec import canonical_hash
from src.research.coordinator import ResearchCoordinator
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

NOW = dt.datetime(2026, 8, 23, tzinfo=dt.UTC).isoformat()


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
