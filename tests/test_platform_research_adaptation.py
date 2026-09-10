from collections import Counter

import pytest

from src.data.database import PlatformDatabase
from src.domain._codec import canonical_hash
from src.domain.strategies import StrategySourceType
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.executors import ProviderExecutorRegistry, UnsupportedExecutorError
from src.research.generation import (
    CAMPAIGNS,
    GenerationAllocator,
    GenerationError,
    GenerationFeedback,
    HypothesisGenerator,
    SqlGenerationFeedbackStore,
    build_hypothesis,
)
from src.research.providers import provider_candidate


@pytest.mark.parametrize(
    "family,direction",
    [
        (family, "signed")
        for family in (
            "cross_sectional",
            "relative_value",
            "microstructure",
            "machine_learning",
            "meta_strategy",
            "execution",
            "market_making",
        )
    ]
    + [("time_series", "market_neutral"), ("time_series", "hedged")],
)
def test_scalar_dsl_cannot_bypass_specialised_execution_contracts(family, direction):
    identity = canonical_hash({"family": family, "direction": direction})
    candidate = provider_candidate(
        identity="scalar-capability-check",
        version="1",
        family=family,
        product="active_income",
        thesis_id=identity,
        lineage_id=identity,
        provider="fixture",
        source_type=StrategySourceType.GENERATED_DSL,
        source_payload={
            "kind": "typed_rule",
            "rule": {
                "feature": "trend",
                "operator": "gt",
                "threshold": 0,
                "direction": direction,
            },
        },
        dataset_snapshot_hashes=(identity,),
        submitted_at="2026-09-07T00:00:00+00:00",
    )
    with pytest.raises(UnsupportedExecutorError, match="directional DSL") as error:
        ProviderExecutorRegistry.default().execute(candidate, {})
    assert error.value.outcome_category.value == "unsupported"


def test_direct_relative_value_submission_cannot_bypass_capability_gate():
    identity = canonical_hash("fixture")
    candidate = provider_candidate(
        identity="beta_neutral_spreads",
        version="1",
        family="relative_value",
        product="active_income",
        thesis_id=identity,
        lineage_id=identity,
        provider="agent",
        source_type=StrategySourceType.RELATIVE_VALUE,
        source_payload={"semantic_strategy": "beta_neutral_spreads"},
        dataset_snapshot_hashes=(identity,),
        submitted_at="2026-09-07T00:00:00+00:00",
    )
    with pytest.raises(UnsupportedExecutorError, match="frozen hedge estimates") as error:
        ProviderExecutorRegistry.default().execute(candidate, {})
    assert error.value.outcome_category.value == "unsupported"


def test_directional_dsl_cannot_omit_the_runtime_protective_policy():
    from src.domain._codec import to_primitive
    from src.research.executors import execute_generated_dsl
    from src.services.artefact_dispatcher import ArtefactDispatcher, ArtefactDispatchError

    identity = canonical_hash("unprotected-fixture")
    candidate = provider_candidate(
        identity="unprotected-fixture",
        version="1",
        family="time_series",
        product="active_income",
        thesis_id=identity,
        lineage_id=identity,
        provider="fixture",
        source_type=StrategySourceType.GENERATED_DSL,
        source_payload={
            "kind": "typed_rule",
            "rule": {"feature": "trend", "operator": "gt", "threshold": 0, "direction": "long"},
        },
        dataset_snapshot_hashes=(identity,),
        submitted_at="2026-09-07T00:00:00+00:00",
    )
    context = {"feature_rows": [{"trend": 1}], "dataset_snapshot_ids": [identity]}
    with pytest.raises(UnsupportedExecutorError, match="protective execution policy"):
        ProviderExecutorRegistry.default().execute(candidate, context)
    with pytest.raises(UnsupportedExecutorError, match="protective execution policy"):
        execute_generated_dsl(candidate, context)
    artefact = {
        "definition": to_primitive(candidate.definition),
        "definition_hash": candidate.definition.definition_hash,
        "position_limits": {"maximum_position": 0.1},
    }
    artefact["artefact_hash"] = canonical_hash(artefact)
    with pytest.raises(ArtefactDispatchError, match="protective execution policy"):
        ArtefactDispatcher.default().evaluate({"trend": 1}, artefact)


@pytest.mark.parametrize(
    "name", ["basis_convergence", "pairs_mean_reversion", "funding_carry", "event_microstructure"]
)
def test_unsupported_campaign_cannot_be_enabled_by_explicit_selection(name):
    campaign = next(item for item in CAMPAIGNS if item.name == name)
    arguments = {
        "dataset_snapshot_hashes": (canonical_hash("fixture"),),
        "submitted_at": "2026-09-07T00:00:00+00:00",
    }
    generator = HypothesisGenerator(
        product="active_income", instrument_universe=("BTCUSDT", "ETHUSDT")
    )
    assert generator.generate(**arguments, total_budget=1, campaigns=(campaign,)) == ()
    with pytest.raises(GenerationError, match="unsupported campaign"):
        build_hypothesis(
            campaign, variant=0, instrument_universe=("BTCUSDT", "ETHUSDT"), **arguments
        )


def test_single_slot_budget_rotates_after_allocator_restart(tmp_path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'rotation.sqlite3'}")
    database.create_schema()
    store = SqlGenerationFeedbackStore(database.engine)
    selected = []
    for iteration in range(9):
        allocations = GenerationAllocator().allocate(
            CAMPAIGNS[:3], total_budget=1, feedback=store.load()
        )
        winner = next(item.campaign for item in allocations if item.trials)
        selected.append(winner)
        store.append(
            GenerationFeedback(
                winner,
                "generated",
                "2026-09-07T00:00:00+00:00",
                candidate_id=f"candidate-{iteration}",
            )
        )
    assert Counter(selected) == {campaign.name: 3 for campaign in CAMPAIGNS[:3]}
    database.dispose()


def test_registration_and_missing_data_do_not_change_alpha_score():
    observed = "2026-09-07T00:00:00+00:00"
    neutral = [
        GenerationFeedback("campaign", outcome, observed, candidate_id="candidate")
        for outcome in ("generated", "accepted", "data_unavailable", "duplicate_exact")
    ]
    assert GenerationAllocator._score(neutral) == 1
    economic = GenerationFeedback(
        "campaign",
        "accepted",
        observed,
        candidate_id="candidate",
        metadata={
            "stage": "development",
            "economic_engine": ECONOMIC_ENGINE_VERSION,
            "net_return": 0.01,
        },
    )
    assert GenerationAllocator._score([*neutral, economic, economic]) == 2


@pytest.mark.parametrize("crash_after_insert", [False, True])
def test_scheduler_round_restarts_without_duplicate_trials_and_uses_real_feedback(
    tmp_path, monkeypatch, crash_after_insert
):
    import datetime as dt
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy import select
    from test_platform_candidate_lifecycle import NOW, _bundle

    from src.data.database import (
        experiment,
        job,
        research_epoch,
        research_round,
        thesis_trial,
        validation_stage,
    )
    from src.research.datasets import CanonicalDatasetResolver, SqlCanonicalDatasetRepository
    from src.research.store import SqlResearchStore
    from src.services.config import load_split_configuration
    from src.services.research_jobs import DatabaseResearchJobHandlers
    from src.services.research_worker import ResearchWorker
    from src.services.scheduler import DatabaseJobQueue, PlatformScheduler, ScheduleSpec

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'rounds.sqlite3'}")
    database.create_schema()
    product = {
        "product_id": "active_income",
        "dataset_timeframe": "1h",
        "universe_id": "lifecycle-fixture",
        "generation_budget": 1,
    }
    config = {
        "search_budgets": {"maximum_candidates_per_epoch": 2, "maximum_candidates_per_day": 100}
    }
    from src.research.data_plan import product_data_plan

    bundle = _bundle(database, data_plan_id=product_data_plan(product, config, ["BTCUSDT"]).plan_id)
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="round-worker",
        node_id="fixture",
        role="research",
        capabilities=("generate_hypotheses", "evaluate_candidate"),
        observed_at=NOW,
    )
    handlers = DatabaseResearchJobHandlers(
        SqlResearchStore(database.engine),
        configuration=load_split_configuration(Path(__file__).resolve().parents[1] / "config"),
        dataset_resolver=CanonicalDatasetResolver(SqlCanonicalDatasetRepository(database.engine)),
    )
    worker = ResearchWorker(
        runtime=SimpleNamespace(
            node=SimpleNamespace(operating_system="linux"), can_submit_orders=False
        ),
        queue=queue,
        worker_id="round-worker",
        handlers={
            "generate_hypotheses": handlers.generate_hypotheses,
            "evaluate_candidate": handlers.evaluate_candidate,
        },
    )
    clock = [NOW]
    monkeypatch.setattr("src.services.research_worker.utc_now", lambda: clock[0])
    monkeypatch.setattr("src.services.research_jobs.utc_now", lambda: clock[0])
    monkeypatch.setattr(
        "src.services.scheduler.AUTONOMOUS_SCHEDULES",
        (ScheduleSpec("candidate_generation", 1), ScheduleSpec("candidate_evaluation", 1)),
    )
    scheduler = PlatformScheduler(
        engine=database.engine,
        products={"active_income": product},
        node_id="fixture",
        research_configuration=config,
    )
    scheduler.run_once(now=clock[0])
    if crash_after_insert:
        original = SqlResearchStore.save_candidate

        def interrupted(store, candidate):
            original(store, candidate)
            raise RuntimeError("simulated interruption after candidate insert")

        monkeypatch.setattr(SqlResearchStore, "save_candidate", interrupted)
        failed = worker.run_once(now=clock[0])
        assert failed["reason_code"] == "research_job_failed"
        with database.engine.connect() as connection:
            assert connection.execute(select(experiment.c.id)).first() is None
            assert connection.execute(select(thesis_trial.c.id)).first() is None
            assert connection.execute(select(research_epoch.c.consumed)).scalar_one() == 0
        monkeypatch.setattr(SqlResearchStore, "save_candidate", original)
        clock[0] = (dt.datetime.fromisoformat(NOW) + dt.timedelta(seconds=121)).isoformat()
    first = worker.run_once(now=clock[0])
    assert first["reason_code"] == "research_job_completed", first
    assert first["round_number"] == 1 and len(first["candidate_ids"]) == 1
    first_id = first["candidate_ids"][0]
    completed_at = clock[0]
    for tick in range(1, 6):
        clock[0] = (
            dt.datetime.fromisoformat(completed_at) + dt.timedelta(seconds=tick)
        ).isoformat()
        scheduler = PlatformScheduler(
            engine=database.engine,
            products={"active_income": product},
            node_id="fixture",
            research_configuration=config,
        )
        scheduler.run_once(now=clock[0])
        result = worker.run_once(now=clock[0])
        assert result["reason_code"] != "research_job_failed", result
        if result.get("round_number") == 2:
            break
    assert result.get("round_number") == 2, result
    assert set(result["candidate_ids"]) - {first_id}
    with database.engine.connect() as connection:
        rounds = list(connection.execute(select(research_round)).mappings())
        candidates = list(connection.execute(select(experiment.c.id)).scalars())
        trials = list(connection.execute(select(thesis_trial.c.candidate_id)).scalars())
        stages = list(
            connection.execute(
                select(validation_stage).where(validation_stage.c.experiment_id == first_id)
            ).mappings()
        )
        generation_jobs = list(
            connection.execute(select(job).where(job.c.name == "generate_hypotheses")).mappings()
        )
        epoch = connection.execute(select(research_epoch)).mappings().one()
    assert len(rounds) == len(generation_jobs) == 2
    assert len(candidates) == len(trials) == len(set(trials)) == 2
    assert stages and any(row["payload"].get("evidence") for row in stages)
    assert epoch["consumed"] == 2 and epoch["state"] == "closed_for_selection"
    assert epoch["dataset_bundle_id"] == bundle.bundle_id
    # Replaying a completed round has exactly its original financial/research identity.
    from src.research.rounds import ResearchRoundCoordinator

    replay = ResearchRoundCoordinator(database.engine).complete(
        rounds[0]["id"], produce=lambda *_: pytest.fail("completed round was executed twice")
    )
    assert replay["candidate_ids"] == first["candidate_ids"]
    database.dispose()
