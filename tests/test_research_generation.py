from __future__ import annotations

import datetime as dt
from dataclasses import replace

from src.data.database import PlatformDatabase
from src.domain._codec import canonical_hash
from src.research.catalogue import registered_strategy_candidate_page
from src.research.coordinator import ResearchCoordinator, TrialCapacityError
from src.research.datasets import RESEARCH_BUNDLE_ROLES, CanonicalResearchDatasetBuilder
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.generation import (
    CAMPAIGNS,
    BtcAllocationLimits,
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
            candidate_id=f"candidate-{index}",
            metadata={
                "stage": "development",
                "economic_engine": ECONOMIC_ENGINE_VERSION,
                "net_return": -0.1,
            },
        )
        for index in range(4)
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
        next(campaign for campaign in CAMPAIGNS if campaign.name == "cross_sectional_momentum"),
        variant=2,
        instrument_universe=("ETHUSDT", "BTCUSDT"),
        dataset_snapshot_hashes=(canonical_hash({"snapshot": "event"}),),
        submitted_at=NOW,
        universe_snapshot_id=canonical_hash({"universe": "point-in-time"}),
    )
    assert hypothesis.thesis.generalisation_scope["product"] == "active_income"
    assert hypothesis.candidate.definition.universe["type"] == "point_in_time"
    assert hypothesis.candidate.definition.signal_model["kind"] == "ranked_long_short_panel"
    assert hypothesis.candidate.definition.signal_model["instrument_scope"] == [
        "binance:futures:ETHUSDT:USDT",
        "binance:futures:BTCUSDT:USDT",
    ]
    assert hypothesis.semantic_signature == hypothesis_signature(hypothesis.candidate.definition)


def test_active_income_directional_generation_isolated_but_cross_sectional_stays_panel() -> None:
    snapshots = (canonical_hash({"snapshot": "scope"}),)
    directional = next(item for item in CAMPAIGNS if item.name == "futures_trend_following")
    generated = HypothesisGenerator(
        product="active_income", instrument_universe=("BTCUSDT", "ETHUSDT")
    ).generate(
        dataset_snapshot_hashes=snapshots,
        submitted_at=NOW,
        total_budget=2,
        campaigns=(directional,),
    )
    assert len(generated) == 2
    assert {tuple(item.candidate.definition.universe["symbols"]) for item in generated} == {
        ("BTCUSDT",),
        ("ETHUSDT",),
    }

    panel = next(item for item in CAMPAIGNS if item.name == "cross_sectional_momentum")
    panel_candidate = (
        HypothesisGenerator(product="active_income", instrument_universe=("BTCUSDT", "ETHUSDT"))
        .generate(
            dataset_snapshot_hashes=snapshots,
            submitted_at=NOW,
            total_budget=1,
            campaigns=(panel,),
        )[0]
        .candidate
    )
    assert tuple(panel_candidate.definition.universe["symbols"]) == ("BTCUSDT", "ETHUSDT")


def test_btc_allocation_uses_minimum_risk_cap_and_mutations_preserve_it(tmp_path) -> None:
    limits = BtcAllocationLimits.from_mapping(
        {
            "product_max_tactical_fraction": 0.3,
            "instrument_maximum_fraction": 0.2,
            "strategy_maximum_fraction": 0.25,
            "sleeve_maximum_fraction": 0.5,
            "minimum_btc_fraction": 0.7,
        }
    )
    campaign = next(item for item in CAMPAIGNS if item.name == "btc_trend_breakout")
    snapshots = (canonical_hash({"snapshot": "btc-limits"}),)
    first = build_hypothesis(
        campaign,
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=snapshots,
        submitted_at=NOW,
        btc_allocation_limits=limits,
    )
    definition = first.candidate.definition
    assert limits.maximum_tactical_fraction == 0.2
    assert definition.signal_model["maximum_tactical_fraction"] == 0.2
    assert definition.position_model["maximum_tactical_fraction"] == 0.2
    assert definition.risk_policy["btc_allocation_limits"] == limits.to_payload()
    lower_cap = BtcAllocationLimits.from_mapping(
        {**limits.to_payload(), "product_max_tactical_fraction": 0.1}
    )
    lower = build_hypothesis(
        campaign,
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=snapshots,
        submitted_at=NOW,
        btc_allocation_limits=lower_cap,
    )
    assert first.semantic_signature != lower.semantic_signature

    database = _database(tmp_path)
    SqlThesisRegistry(database.engine).register(first.thesis)
    ResearchCoordinator(SqlResearchStore(database.engine)).submit(first.candidate)
    mutation = HypothesisGenerator(
        product="btc_accumulation",
        instrument_universe=("BTCUSDT",),
        memory=SqlHypothesisMemory(database.engine),
        btc_allocation_limits=limits,
    ).generate(
        dataset_snapshot_hashes=snapshots,
        submitted_at=NOW,
        total_budget=1,
        campaigns=(campaign,),
        parent_candidates=(first.candidate,),
    )[0]
    assert mutation.candidate.definition.position_model["maximum_tactical_fraction"] == 0.2
    assert (
        mutation.candidate.definition.signal_model["btc_allocation_limits"] == limits.to_payload()
    )
    database.dispose()


def test_catalogue_pages_round_robin_families_and_symbols() -> None:
    snapshots = (canonical_hash({"snapshot": "catalogue-page"}),)
    first, cursor, first_closed = registered_strategy_candidate_page(
        product="active_income",
        dataset_snapshot_hashes=snapshots,
        instrument_universe=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        cursor=0,
        limit=12,
        submitted_at=NOW,
    )
    second, next_cursor, second_closed = registered_strategy_candidate_page(
        product="active_income",
        dataset_snapshot_hashes=snapshots,
        instrument_universe=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        cursor=cursor,
        limit=12,
        submitted_at=NOW,
    )
    assert 0 < len(first) <= 12
    assert 0 < len(second) <= 12
    assert len({item.candidate_id for item in (*first, *second)}) == len(first) + len(second)
    assert (
        len(
            {
                (
                    str(item.definition.metadata["manifest_family"]),
                    str(item.definition.universe["symbols"][0]),
                )
                for item in first
            }
        )
        >= 6
    )
    assert len({str(item.definition.universe["symbols"][0]) for item in first}) == 3
    assert cursor == 12
    assert next_cursor == 24
    assert not first_closed and next_cursor > cursor
    assert isinstance(second_closed, bool)


def test_catalogue_resume_cursor_keeps_filtered_raw_slots() -> None:
    from src.research.catalogue import (
        _registered_strategy_candidates,
        _round_robin_candidates,
        registered_strategy_catalogue_eligibility,
    )

    arguments = {
        "product": "active_income",
        "dataset_snapshot_hashes": (canonical_hash({"snapshot": "catalogue-resume"}),),
        "instrument_universe": ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        "submitted_at": NOW,
    }
    raw_candidates = _round_robin_candidates(_registered_strategy_candidates(**arguments))
    for cursor in (3, 9):
        limit = 6
        expected = tuple(
            candidate.candidate_id
            for candidate in raw_candidates[cursor : cursor + limit]
            if registered_strategy_catalogue_eligibility(
                candidate.definition.identity,
                parameters=candidate.definition.signal_model.get("parameters"),
            )["eligible"]
        )
        page, next_cursor, closed = registered_strategy_candidate_page(
            **arguments, cursor=cursor, limit=limit
        )

        assert tuple(candidate.candidate_id for candidate in page) == expected
        assert next_cursor == min(cursor + limit, len(raw_candidates))
        assert closed is (next_cursor == len(raw_candidates))


def test_registration_reserves_catalogue_and_total_epoch_capacity(tmp_path) -> None:
    database = _database(tmp_path)
    store = SqlResearchStore(database.engine, epoch_trial_budget=2, daily_trial_budget=10)
    campaign = next(item for item in CAMPAIGNS if item.name == "futures_trend_following")
    snapshots = (canonical_hash({"snapshot": "unbundled-budget"}),)
    candidates = tuple(
        build_hypothesis(
            campaign,
            variant=variant,
            instrument_universe=("BTCUSDT",),
            dataset_snapshot_hashes=snapshots,
            submitted_at=NOW,
        )
        for variant in range(3)
    )
    SqlThesisRegistry(database.engine).register(candidates[0].thesis)
    coordinator = ResearchCoordinator(store)
    coordinator.submit(candidates[0].candidate, producer="catalogue")
    try:
        coordinator.submit(candidates[1].candidate, producer="catalogue")
    except TrialCapacityError as exc:
        assert "80%" in str(exc)
    else:
        raise AssertionError("catalogue registration must leave epoch capacity available")
    coordinator.submit(candidates[1].candidate, producer="adaptive")
    try:
        coordinator.submit(candidates[2].candidate, producer="adaptive")
    except TrialCapacityError as exc:
        assert "epoch" in str(exc)
    else:
        raise AssertionError("unique candidate registration must enforce the epoch limit")
    database.dispose()


def test_finite_closure_preserves_admission_epoch_and_rejects_started_round(tmp_path) -> None:
    from sqlalchemy import select
    from test_platform_candidate_lifecycle import _bundle, _candidate

    from src.data.database import research_epoch
    from src.research.rounds import ResearchRoundCoordinator

    database = _database(tmp_path)
    rounds = ResearchRoundCoordinator(database.engine)
    bundle = _bundle(database)
    candidate = _candidate(database, bundle)
    with database.engine.connect() as connection:
        before = connection.execute(select(research_epoch)).mappings().one()
    epoch_id = rounds.close_finite(
        bundle.bundle_id,
        product_id="active_income",
        candidate_ids=[candidate.candidate_id],
        at=NOW,
    )
    with database.engine.connect() as connection:
        closed = connection.execute(select(research_epoch)).mappings().one()
    assert epoch_id == before["id"] == closed["id"]
    assert closed["budget"] == before["budget"]
    assert closed["consumed"] == before["consumed"] == 1
    assert closed["round_cursor"] == before["round_cursor"] == 0
    assert closed["state"] == "closed_for_selection"
    assert closed["payload"]["closure_reason"] == "finite_campaign"
    assert closed["payload"]["finite_campaign_id"]
    assert (
        rounds.close_finite(
            bundle.bundle_id,
            product_id="active_income",
            candidate_ids=[candidate.candidate_id],
            at=NOW,
        )
        == epoch_id
    )

    adaptive_bundle = _bundle(database, source_revision=1)
    adaptive_candidate = _candidate(database, adaptive_bundle)
    request = rounds.claim(
        {
            "product_id": "active_income",
            "dataset_bundle_id": adaptive_bundle.bundle_id,
            "generation_budget": 1,
            "producer": "catalogue",
        },
        now=NOW,
        epoch_budget=100,
        daily_budget=100,
    )
    assert request is not None
    try:
        rounds.close_finite(
            adaptive_bundle.bundle_id,
            product_id="active_income",
            candidate_ids=[adaptive_candidate.candidate_id],
            at=NOW,
        )
    except ValueError as exc:
        assert "adaptive epoch" in str(exc)
    else:
        raise AssertionError("a started research round must reject finite closure")
    database.dispose()


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
    assert len(generated) == 1
    assert generated[0].candidate.candidate_id != first.candidate.candidate_id
    assert any(
        item.outcome == "duplicate_exact" and item.candidate_id == first.candidate.candidate_id
        for item in feedback_store.load(campaign=CAMPAIGNS[0].name)
    )
    assert SqlThesisRegistry(database.engine).lineage_trial_count(first.thesis.thesis_id) == 1


def test_hypothesis_generator_derives_a_bounded_mutation_from_a_duplicate(tmp_path) -> None:
    database = _database(tmp_path)
    snapshots = (canonical_hash({"snapshot": "screening"}),)
    first = build_hypothesis(
        CAMPAIGNS[2],
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=snapshots,
        submitted_at=NOW,
    )
    SqlThesisRegistry(database.engine).register(first.thesis)
    ResearchCoordinator(SqlResearchStore(database.engine)).submit(first.candidate)

    generated = HypothesisGenerator(
        product="active_income",
        instrument_universe=("BTCUSDT",),
        memory=SqlHypothesisMemory(database.engine),
    ).generate(
        dataset_snapshot_hashes=snapshots,
        submitted_at=NOW,
        total_budget=1,
        campaigns=(CAMPAIGNS[2],),
        parent_candidates=(first.candidate,),
    )

    assert len(generated) == 1
    assert generated[0].candidate.provider == "bounded_hypothesis_mutation"
    assert generated[0].candidate.definition.metadata["generated_method"] == "mutation"
    assert generated[0].thesis.parent_thesis_ids == (first.thesis.thesis_id,)


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
        instrument_scope=("binance:spot:BTCUSDT",),
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
        lease_token="fixture",
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
        lease_token="fixture",
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
        "_defer": True,
        "_defer_reason": "canonical_dataset_bundle_unavailable",
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
