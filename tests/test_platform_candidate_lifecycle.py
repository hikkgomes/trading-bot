"""A synthetic correctness path through the real research and paper scheduler."""

import datetime as dt
import json
import math
import os
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url

from src.accounting.paper_account import configured_paper_accounts
from src.data.binance_market import normalise_public_event
from src.data.database import (
    PlatformDatabase,
    active_strategy_assignment,
    alpha_forecast,
    fill,
    forward_paper_decision,
    forward_paper_observation,
    forward_paper_summary,
    forward_qualification_window,
    holdout_claim,
    position,
    reconciliation_event,
    selection_cohort,
    selection_cohort_result,
)
from src.data.parquet_store import PartitionedBarStore
from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.domain._codec import canonical_hash, timeframe_seconds, to_primitive
from src.domain.instruments import Instrument, MarketType
from src.research.catalogue import registered_strategy_candidates, registered_strategy_theses
from src.research.cohorts import CohortPending, prepare_cohort
from src.research.coordinator import ResearchCoordinator
from src.research.datasets import (
    CORE_RESEARCH_BUNDLE_ROLES,
    CandidateDatasetPlan,
    CanonicalDatasetResolver,
    CanonicalResearchDatasetBuilder,
    SqlCanonicalDatasetRepository,
)
from src.research.executors import DataUnavailableError
from src.research.rounds import ResearchRoundCoordinator
from src.research.store import SqlResearchStore
from src.research.theses import SqlThesisRegistry
from src.services import supervisor
from src.services.config import load_split_configuration
from src.services.health import DatabaseHeartbeatStore
from src.services.promotion import (
    DatabasePromotionWorker,
    PromotionPolicy,
    SqlPromotionPolicyStore,
    SqlPromotionStore,
)
from src.services.research_jobs import DatabaseResearchJobHandlers
from src.services.scheduler import DatabaseJobQueue, PlatformScheduler, ScheduleSpec

NOW = "2026-09-07T00:00:00+00:00"
INSTRUMENT = "binance:futures:BTCUSDT:USDT"


def test_invalidated_forward_assignment_completes_without_retry_or_reschedule(tmp_path):
    from sqlalchemy import insert

    from src.data.database import (
        active_strategy_assignment,
        evidence_invalidation,
        forward_paper_observation,
        job,
        job_attempt,
        strategy_definition,
        strategy_version,
    )
    from src.services.forward_observation import DatabaseForwardObservationWorker

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'invalidated-forward.sqlite3'}")
    database.create_schema()
    definition_id = canonical_hash("invalidated-definition")
    version_id = canonical_hash("invalidated-version")
    artefact_hash = canonical_hash("invalidated-artefact")
    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_definition).values(
                id=definition_id,
                identity="invalidated-fixture",
                product_id="active_income",
                source_type="generated_dsl",
                source_hash=definition_id,
                definition={"product": "active_income"},
            )
        )
        connection.execute(
            insert(strategy_version).values(
                id=version_id,
                definition_id=definition_id,
                version="1",
                created_at=NOW,
                payload={},
            )
        )
    assignment_id = "fixture-assignment"
    with database.engine.begin() as connection:
        connection.execute(
            insert(active_strategy_assignment).values(
                id=assignment_id,
                product_id="active_income",
                portfolio_id="fixture-portfolio",
                sleeve_id="default",
                strategy_version_id=version_id,
                instrument_id=INSTRUMENT,
                universe_id=None,
                assignment_scope_id=f"instrument:{INSTRUMENT}",
                artefact_hash=artefact_hash,
                lifecycle_state="forward_paper",
                execution_mode="paper",
                capital_limit=100.0,
                risk_budget=1.0,
                assigned_at=NOW,
                active_until=None,
                assigned_by="fixture operator",
                assignment_reason="fixture history",
                active=True,
                payload={},
            )
        )
    with database.engine.begin() as connection:
        connection.execute(
            forward_paper_observation.insert().values(
                id=canonical_hash("old-forward-observation"),
                strategy_version_id=version_id,
                product_id="active_income",
                instrument_id=INSTRUMENT,
                observed_at=NOW,
                artefact_hash=artefact_hash,
                observation_hash=canonical_hash("old-forward-observation-content"),
                payload={},
            )
        )
        connection.execute(
            evidence_invalidation.insert().values(
                id=f"artefact:{artefact_hash}",
                created_at=NOW,
                payload={"reason_code": "stale_economic_evidence"},
            )
        )
    product = {"product_id": "active_income"}
    scheduler = PlatformScheduler(
        engine=database.engine, products={"active_income": product}, node_id="fixture"
    )
    assert scheduler._forward_jobs("active_income", NOW, NOW) == ()
    assert scheduler._summary_jobs("active_income", NOW, NOW) == ()

    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="forward-worker",
        node_id="fixture",
        role="runtime",
        capabilities=("forward_paper_observation",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="invalidated-forward-observation",
        name="forward_paper_observation",
        payload={
            "assignment_id": assignment_id,
            "strategy_version_id": version_id,
            "product_id": "active_income",
            "instrument_id": INSTRUMENT,
            "artefact_hash": artefact_hash,
            "evaluation_time": NOW,
        },
        available_at=NOW,
    )
    result = DatabaseForwardObservationWorker(
        engine=database.engine, queue=queue, worker_id="forward-worker"
    ).run_once(now=NOW)

    assert result["reason_code"] == "forward_observation_stopped_artefact_invalidated"
    with database.engine.connect() as connection:
        job_state = connection.execute(
            select(job.c.state).where(job.c.id == "invalidated-forward-observation")
        ).scalar_one()
        attempt = connection.execute(
            select(job_attempt.c.status, job_attempt.c.error).where(
                job_attempt.c.job_id == "invalidated-forward-observation"
            )
        ).one()
    assert job_state == "completed"
    assert attempt.status == "completed" and attempt.error is None
    assert scheduler._forward_jobs("active_income", NOW, NOW) == ()
    database.dispose()


def test_scheduler_quarantines_nonpromotable_diagnostics_but_keeps_explicit_work(tmp_path):
    from sqlalchemy import insert, update

    from src.data.database import (
        experiment,
        job,
        job_quarantine,
        strategy_artefact,
    )

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'diagnostic-scheduler.sqlite3'}")
    database.create_schema()
    bundle = _bundle(database)
    candidate = _candidate(database, bundle)
    with database.engine.begin() as connection:
        connection.execute(
            update(experiment)
            .where(experiment.c.id == candidate.candidate_id)
            .values(state="legacy_import")
        )
    diagnostic = replace(
        candidate,
        dataset_bundle_id=None,
        dataset_plan=None,
        metadata={**dict(candidate.metadata), "diagnostic": True, "promotable": False},
        definition=replace(
            candidate.definition,
            metadata={
                **dict(candidate.definition.metadata),
                "diagnostic": True,
                "promotable": False,
            },
        ),
    )
    ResearchCoordinator(SqlResearchStore(database.engine)).submit(diagnostic)
    stored = next(
        item
        for item in SqlResearchStore(database.engine).load_candidates()
        if item.candidate_id == diagnostic.candidate_id
    )
    assert stored.metadata.get("diagnostic") is True
    snapshot_id = bundle.stage_snapshot_ids["screening"]
    resolved_snapshot = CanonicalDatasetResolver(
        SqlCanonicalDatasetRepository(database.engine)
    ).resolve(snapshot_id, expected={})
    snapshot_payload = {
        "role": resolved_snapshot.role,
        "feature_manifest_id": resolved_snapshot.feature_manifest_hash,
        "cost_model_id": resolved_snapshot.cost_model_hash,
        "parameter_set_id": resolved_snapshot.parameter_set_hash,
        "availability_timestamp": resolved_snapshot.availability_timestamp,
    }
    scheduler = PlatformScheduler(
        engine=database.engine,
        products={"active_income": {"product_id": "active_income"}},
        node_id="fixture",
    )
    evaluation_payload = scheduler._research_request(
        candidate_id=diagnostic.candidate_id,
        snapshot_id=snapshot_id,
        snapshot_payload=snapshot_payload,
        requested_stage="screening",
        evaluated_at=NOW,
    )
    queue = DatabaseJobQueue(database.engine)
    for job_id, producer in (
        ("automatic-diagnostic", "platform-scheduler:fixture"),
        ("explicit-diagnostic", "operator-diagnostic"),
    ):
        queue.enqueue(
            job_id=job_id,
            name="evaluate_candidate",
            payload=evaluation_payload,
            available_at=NOW,
            producer_identity=producer,
        )
    artefact_hash = canonical_hash("diagnostic-artefact")
    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_artefact).values(
                id=artefact_hash,
                created_at=NOW,
                payload={
                    "product_id": "active_income",
                    "strategy_version_id": diagnostic.definition.strategy_version_id,
                    "artefact_hash": artefact_hash,
                    "metadata": {"diagnostic": True, "promotable": False},
                },
            )
        )

    assert scheduler._candidate_evaluation_jobs("active_income", NOW, NOW) == ()
    assert scheduler._promotion_jobs("active_income", NOW, NOW) == ()
    with database.engine.connect() as connection:
        state = connection.execute(
            select(experiment.c.state).where(experiment.c.id == diagnostic.candidate_id)
        ).scalar_one()
        queued = {
            row.id: row.state
            for row in connection.execute(
                select(job.c.id, job.c.state).where(
                    job.c.id.in_(("automatic-diagnostic", "explicit-diagnostic"))
                )
            )
        }
        quarantine_reason = connection.execute(
            select(job_quarantine.c.reason).where(job_quarantine.c.job_id == "automatic-diagnostic")
        ).scalar_one()
    assert state == "queued"
    assert queued == {"automatic-diagnostic": "quarantined", "explicit-diagnostic": "pending"}
    assert quarantine_reason == "non_promotable_diagnostic_excluded_from_autonomous_evaluation"
    database.dispose()


@pytest.fixture(params=["sqlite", "postgresql"])
def lifecycle_database(request, tmp_path):
    if request.param == "sqlite":
        database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'lifecycle.sqlite3'}")
        database.create_schema()
        yield database
        database.dispose()
        return
    url = os.environ.get("TRADING_PLATFORM_DATABASE_URL", "")
    if not url.startswith("postgresql"):
        pytest.skip("requires isolated PostgreSQL for the full candidate lifecycle")
    base = make_url(url)
    name = "candidate_lifecycle_" + uuid.uuid4().hex[:16]
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    database = None
    with admin.connect() as connection:
        connection.execute(text(f"CREATE DATABASE {name}"))
    try:
        database = PlatformDatabase(base.set(database=name).render_as_string(hide_password=False))
        database.migrate()
        yield database
    finally:
        if database is not None:
            database.dispose()
        with admin.connect() as connection:
            connection.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
        admin.dispose()


def _bundle(database, *, source_revision=0, extra_market_scale=None, data_plan_id=None):
    instrument = Instrument(
        venue="binance",
        market_type=MarketType.FUTURES,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset="USDT",
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5,
    )
    instruments = (instrument,)
    if extra_market_scale is not None:
        instruments += (replace(instrument, base_asset="SOL", exchange_symbol="SOLUSDT"),)
    snapshot_id = SqlUniverseStore(database.engine).record_snapshot(
        universe_id="lifecycle-fixture",
        observed_at="2025-01-01T00:00:00+00:00",
        observations=tuple(
            InstrumentObservation(
                instrument=item,
                listing_age_days=2000,
                quote_volume=1000000000,
                trade_count=1000000,
                spread_bps=1,
                open_interest=100000000,
                funding_rate=0,
                realised_volatility=0.2,
                depth_notional=10000000,
                data_completeness=1,
            )
            for item in instruments
        ),
        policy=UniverseEligibilityPolicy(),
    )
    intervals, payloads = {}, {}
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    for part, role in enumerate(CORE_RESEARCH_BUNDLE_ROLES):
        rows = []
        for index in range(800):
            time = start + dt.timedelta(hours=part * 800 + index)
            price = 100 * math.exp(0.15 * math.sin(index * math.pi / 20))
            rows.append(
                {
                    "instrument_id": INSTRUMENT,
                    "timestamp": time.isoformat(),
                    "open_timestamp": time.isoformat(),
                    "close_timestamp": time.isoformat(),
                    "availability_time": time.isoformat(),
                    "open": price,
                    "close": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "volume": 100000,
                    "funding_rate": 0,
                    "funding_events": [],
                }
            )
        intervals[role] = {
            "start": rows[0]["timestamp"],
            "end": (time + dt.timedelta(hours=1)).isoformat(),
        }
        if extra_market_scale is not None:
            rows.extend(
                [
                    {
                        **row,
                        "instrument_id": instruments[1].instrument_id,
                        **{
                            key: row[key] * extra_market_scale
                            for key in ("open", "high", "low", "close")
                        },
                        "funding_rate": extra_market_scale / 1000,
                        "funding_events": [
                            {
                                "type": "funding",
                                "timestamp": row["timestamp"],
                                "symbol": instruments[1].instrument_id,
                                "event_id": row["timestamp"],
                                "mark_price": row["close"] * extra_market_scale,
                                "funding_rate": extra_market_scale / 1000,
                            }
                        ],
                    }
                    for row in rows.copy()
                ]
            )
            rows.reverse()
        payloads[role] = CanonicalResearchDatasetBuilder._bar_payload(rows, timeframe="1h")
        payloads[role]["source_revision"] = source_revision
        if data_plan_id:
            payloads[role]["data_plan_id"] = data_plan_id
    return CanonicalResearchDatasetBuilder(database.engine).build(
        "active_income",
        intervals=intervals,
        payload_by_role=payloads,
        universe_snapshot_id=snapshot_id,
        feature_manifest_id=canonical_hash("fixture-features"),
        cost_model_id=canonical_hash("fixture-costs"),
        parameter_set_id=canonical_hash("fixture-parameters"),
        instrument_scope=tuple(item.instrument_id for item in instruments),
        created_at=NOW,
        availability_timestamp=NOW,
    )


def _candidate(database, bundle):
    theses = registered_strategy_theses(product="active_income", instrument_universe=("BTCUSDT",))
    candidate = next(
        item
        for item in registered_strategy_candidates(
            product="active_income",
            instrument_universe=("BTCUSDT",),
            dataset_snapshot_hashes=tuple(bundle.stage_snapshot_ids.values()),
            submitted_at=NOW,
            dataset_bundle_id=bundle.bundle_id,
            universe_snapshot_id=bundle.universe_snapshot_id,
        )
        if item.definition.identity == "sma_cross"
    )
    candidate = replace(
        candidate,
        definition=replace(
            candidate.definition,
            signal_model={
                **candidate.definition.signal_model,
                "parameters": {"fast": 2, "slow": 4, "allow_short": True},
            },
        ),
        dataset_plan=CandidateDatasetPlan.from_bundle(bundle),
    )
    SqlThesisRegistry(database.engine).register(theses["sma_cross"])
    ResearchCoordinator(SqlResearchStore(database.engine)).submit(candidate)
    return candidate


@pytest.mark.parametrize("approved_research", [False, True])
def test_fresh_candidate_reaches_paper_fill_and_honest_forward_decision(
    tmp_path, monkeypatch, approved_research, lifecycle_database
):
    database = lifecycle_database
    bundle = _bundle(database)
    candidate = _candidate(database, bundle)
    assert candidate.definition.data_requirements["bars"] == "1h"
    ResearchRoundCoordinator(database.engine).close_finite(
        bundle.bundle_id, product_id="active_income", candidate_ids=[candidate.candidate_id], at=NOW
    )
    configuration = load_split_configuration(Path(__file__).resolve().parents[1] / "config")
    for configured_product in configuration["products"]["products"]:
        if configured_product["product_id"] == "active_income":
            configured_product["dataset_timeframe"] = candidate.definition.data_requirements["bars"]
    from src.data.database import research_calibration, research_policy_approval
    from src.research.evaluation import EvidencePolicy
    from src.research.policy_approval import (
        _qualified_result,
        research_qualification,
        revoke_policy,
    )
    from src.research.policy_identity import research_policy_engine_digest

    if approved_research:
        calibration_id = canonical_hash({"fixture": "reviewed calibration"})
        approval = {
            "schema": "research.policy_approval/v1",
            "method": "selection_adjusted_block_bootstrap/v2",
            "status": "approved",
            "approved_by": "test operator",
            "approved_at": NOW,
            "configuration_hash": canonical_hash(configuration["research"]),
            "policy_hash": EvidencePolicy.from_configuration(
                configuration["research"]["evidence_policy"]
            ).policy_hash,
            "engine_hash": research_policy_engine_digest(),
            "nominal_test_alpha": 0.05,
            "calibration_id": calibration_id,
        }
        with database.engine.begin() as connection:
            connection.execute(
                research_calibration.insert().values(
                    id=calibration_id, created_at=NOW, payload={"fixture": "reviewed calibration"}
                )
            )
            connection.execute(
                research_policy_approval.insert().values(
                    id=canonical_hash(approval), created_at=NOW, payload=approval
                )
            )
    product = next(
        row for row in configuration["products"]["products"] if row["product_id"] == "active_income"
    )
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="research",
        node_id="fixture",
        role="research",
        capabilities=("evaluate_candidate",),
        observed_at=NOW,
    )
    handlers = DatabaseResearchJobHandlers(
        SqlResearchStore(database.engine),
        configuration=configuration,
        dataset_resolver=CanonicalDatasetResolver(SqlCanonicalDatasetRepository(database.engine)),
    )
    monkeypatch.setattr(
        "src.services.scheduler.AUTONOMOUS_SCHEDULES",
        (ScheduleSpec("candidate_evaluation", 1), ScheduleSpec("promotion_evaluation", 1)),
    )
    scheduler = PlatformScheduler(
        engine=database.engine, products={"active_income": product}, node_id="fixture"
    )
    clock = [NOW]
    monkeypatch.setattr("src.services.research_jobs.utc_now", lambda clock=clock: clock[0])
    for index, stage in enumerate(("screening", "development", "robustness", "protected")):
        clock[0] = (dt.datetime.fromisoformat(NOW) + dt.timedelta(seconds=index)).isoformat()
        scheduler.run_once(now=clock[0])
        claimed = queue.claim(
            worker_id="research", now=clock[0], lease_seconds=60, names=("evaluate_candidate",)
        )
        assert claimed is not None, stage
        assert claimed.payload["candidate_id"] == candidate.candidate_id
        assert claimed.payload["requested_stage"] == stage
        result = handlers.evaluate_candidate(claimed, lambda claimed=claimed: claimed)
        if result.get("reason_code") == "cohort_pending":
            with database.engine.connect() as connection:
                assert connection.execute(select(holdout_claim)).first() is None
                assert len(connection.execute(select(selection_cohort)).all()) == 1
            clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)).isoformat()
            queue.defer(claimed, available_at=clock[0], reason=result["_defer_reason"])
            claimed = queue.claim(
                worker_id="research", now=clock[0], lease_seconds=60, names=("evaluate_candidate",)
            )
            assert claimed is not None and claimed.attempt == 1
            result = handlers.evaluate_candidate(claimed, lambda claimed=claimed: claimed)
        if stage == "protected" and not approved_research:
            from src.data.database import selection_cohort_open

            assert result["reason_code"] == "cohort_pending", result
            assert "research_policy_approval_required" in result["_defer_reason"]
            with database.engine.connect() as connection:
                assert connection.execute(select(holdout_claim)).first() is None
                assert connection.execute(select(selection_cohort_open)).first() is None
                assert connection.execute(select(active_strategy_assignment)).first() is None
            database.dispose()
            return
        assert result["accepted"], result
        queue.complete(claimed, completed_at=clock[0], outcome=result)
    with database.engine.connect() as connection:
        assert connection.execute(select(active_strategy_assignment)).first() is None
    assert result["artefact_hash"]
    with database.engine.connect() as connection:
        inference = connection.execute(select(selection_cohort_result.c.payload)).scalar_one()
    assert inference["members"] == [candidate.candidate_id]
    assert inference["draws"] == 2000
    assert inference["qualification_authority"] is False
    with database.engine.connect() as connection:
        qualification = research_qualification(
            connection,
            artefact_hash=result["artefact_hash"],
            strategy_version_id=candidate.definition.strategy_version_id,
            product_id="active_income",
            configuration_hash=canonical_hash(configuration["research"]),
            at=clock[0],
        )
    assert qualification["qualified"] is approved_research, qualification
    if not approved_research:
        assert qualification["reason_code"] == "research_policy_unapproved"
    else:
        with database.engine.connect() as connection:
            cohort = connection.execute(select(selection_cohort.c.payload)).scalar_one()
            changed_configuration = research_qualification(
                connection,
                artefact_hash=result["artefact_hash"],
                strategy_version_id=candidate.definition.strategy_version_id,
                product_id="active_income",
                configuration_hash=canonical_hash({}),
                at=clock[0],
            )
        assert changed_configuration["qualified"] is False
        rejected = _qualified_result(
            cohort,
            {**inference, "adjusted_pvalues": {candidate.candidate_id: 1.0}},
            candidate.candidate_id,
            {**approval, "approval_id": canonical_hash(approval)},
        )
        assert rejected["qualified"] is False
        assert rejected["reason_code"] == "research_selection_threshold"
    policy_store = SqlPromotionPolicyStore(database.engine)
    policy = next(
        row
        for row in configuration["promotion"]["policies"]
        if row["policy_id"] == product["promotion_policy_id"]
    )
    policy_store.put(
        policy["policy_id"],
        PromotionPolicy(
            **{
                key: value
                for key, value in policy.items()
                if key in PromotionPolicy.__dataclass_fields__
            }
        ),
        created_at=NOW,
    )
    queue.register_worker(
        worker_id="promotion",
        node_id="fixture",
        role="promotion",
        capabilities=("promotion_evaluation",),
        observed_at=clock[0],
    )
    clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)).isoformat()
    scheduler.run_once(now=clock[0])
    clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(minutes=10)).isoformat()
    promoted = DatabasePromotionWorker(
        queue=queue,
        worker_id="promotion",
        store=SqlPromotionStore(database.engine),
        policy_store=policy_store,
        strict_identity_contract=True,
        paper_accounts=configured_paper_accounts(database.engine, configuration),
    ).run_once(now=clock[0])
    assert promoted["reason_code"] == "forward_paper_promoted", promoted
    with database.engine.connect() as connection:
        assignments = connection.execute(select(active_strategy_assignment)).mappings().all()
    assert len(assignments) == 1
    assert assignments[0]["execution_mode"] == "paper"
    assert assignments[0]["assigned_at"] == clock[0]
    with database.engine.connect() as connection:
        window = connection.execute(select(forward_qualification_window)).mappings().one()
    assert window["opened_at"] == clock[0]
    assert window["source_event_id"] == promoted["promotion_event_id"]
    assert window["payload"]["family_evidence"]["family"] == candidate.definition.family
    assert assignments[0]["risk_budget"] > 0
    from src.portfolio.books import paper_book_account

    state = paper_book_account(database.engine, assignments[0]["payload"]["book_id"]).state()
    assert state is not None and state.balances == {"USDT": 10000.0}
    assert state.observed_at == assignments[0]["assigned_at"]
    restarted = PlatformScheduler(
        engine=database.engine, products={"active_income": product}, node_id="fixture"
    )
    clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)).isoformat()
    restarted.run_once(now=clock[0])
    assert (
        queue.claim(
            worker_id="promotion", now=clock[0], lease_seconds=60, names=("promotion_evaluation",)
        )
        is None
    )
    with database.engine.connect() as connection:
        assert len(connection.execute(select(active_strategy_assignment)).all()) == 1
        assert len(connection.execute(select(forward_qualification_window)).all()) == 1
    services = _forward_services(database, configuration, queue, clock, tmp_path, monkeypatch)
    with database.engine.connect() as connection:
        observations = (
            connection.execute(select(forward_paper_observation.c.payload)).scalars().all()
        )
    assert observations
    assert (
        min(row["observation"]["facts"]["window_start"] for row in observations)
        == assignments[0]["assigned_at"]
    )
    _repair_window_cycle(database, configuration, queue, clock, services, assignments[0], window)
    if approved_research:
        revoked_at = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)).isoformat()
        revoke_policy(
            database.engine,
            approval_id=canonical_hash(approval),
            revoked_by="test operator",
            revoked_at=revoked_at,
        )
        with database.engine.connect() as connection:
            revoked = research_qualification(
                connection,
                artefact_hash=assignments[0]["artefact_hash"],
                strategy_version_id=candidate.definition.strategy_version_id,
                product_id="active_income",
                configuration_hash=canonical_hash(configuration["research"]),
                at=revoked_at,
            )
        assert revoked["qualified"] is False
        assert revoked["reason_code"] == "research_policy_unapproved"
    database.dispose()


def _forward_services(database, configuration, queue, clock, root, monkeypatch):
    monkeypatch.setattr(supervisor, "utc_now", lambda clock=clock: clock[0])
    common = {"database": database, "configuration": configuration, "node_id": "fixture"}
    history = tuple(
        _candle((dt.datetime.fromisoformat(NOW) - dt.timedelta(hours=index)).isoformat(), 100, 100)
        for index in reversed(range(200))
    )
    PartitionedBarStore(root / "market").put_batch(
        (event, "binance", "futures", "BTCUSDT") for event in history
    )
    from src.risk.engine import SqlRiskSnapshotStore

    snapshots = SqlRiskSnapshotStore(database.engine)
    for historical in history[-4:-1]:
        candle = historical.payload["data"]["k"]
        snapshots.save(
            {
                "kind": "market_data_input",
                "product_id": "active_income",
                "instrument_id": historical.instrument_id,
                "source_event_id": historical.event_id,
                "source_event_type": "candle",
                "source_candle_closed": True,
                "source_timeframe": candle["i"],
                "source_close_time_ms": candle["T"],
                "availability_time": historical.availability_timestamp,
                "values": {"close": float(candle["c"])},
            },
            created_at=historical.availability_timestamp,
        )
    writer = supervisor._data_writer_cycle(**common, root=root / "market")
    feature = supervisor._feature_cycle(
        database=database,
        node_id="fixture",
        service_name="feature-service",
        parquet_root=root / "market",
    )
    strategy = supervisor._strategy_evaluator_cycle(database=database, node_id="fixture")
    trading = (
        supervisor._account_reconciliation_cycle(database=database, configuration=configuration),
        supervisor._portfolio_state_cycle(**common),
        supervisor._portfolio_cycle(**common),
        supervisor._risk_cycle(**common),
        supervisor._product_coordination_cycle(database=database, node_id="fixture"),
        supervisor._execution_cycle(**common),
        supervisor._paper_cycle(**common),
    )
    outcomes = []
    for index, price in enumerate((100, 100, 100, 101, 103, 105)):
        clock[0] = (
            (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(hours=1))
            .replace(minute=0, second=0)
            .isoformat()
        )
        previous = (100, 100, 100, 100, 101, 103)[index]
        _write_market_support(queue, writer, clock[0], index, price)
        event = _candle(clock[0], price, previous)
        queue.enqueue(
            job_id=f"lifecycle-market-{index}",
            name="market_event_write",
            payload={
                "venue": "binance",
                "market": "futures",
                "symbol": "BTCUSDT",
                "event": to_primitive(event),
            },
            available_at=clock[0],
        )
        written = writer()
        assert written["reason_code"] == "market_event_written", written
        trading[-1]()
        featured = feature()
        assert featured["reason_code"] == "features_persisted", featured
        evaluated = strategy()
        assert evaluated["reason_code"] == "alpha_forecast_persisted", evaluated
        DatabaseHeartbeatStore(database.engine).record(
            service_name="strategy-evaluator",
            node_id="fixture",
            observed_at=clock[0],
            healthy=True,
            payload={"completed_job": evaluated["job_id"]},
        )
        results = [trading[0]()]
        results.extend(trading[1]() for _ in range(3))
        results.extend(cycle() for cycle in trading[2:])
        assert not any(result.get("error_type") for result in results), json.dumps(
            results, default=str
        )
        outcomes.append(results)
    with database.engine.connect() as connection:
        forecasts = connection.execute(select(alpha_forecast.c.payload)).scalars().all()
        fills = connection.execute(select(fill.c.payload)).scalars().all()
    assert any(row["direction"] == "long" for row in forecasts)
    assert fills, outcomes
    assert len(fills) == 1, fills
    assert abs(fills[0]["quantity"] * 101 - 1000) < 1e-8
    clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(minutes=1)).isoformat()
    queue.enqueue(
        job_id="wrong-timeframe",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(_candle(clock[0], 105, 105, timeframe="1m")),
        },
        available_at=clock[0],
    )
    assert writer()["reason_code"] == "market_event_written"
    assert feature()["reason_code"] == "features_persisted"
    assert strategy()["reason_code"] == "strategy_evaluation_queue_empty"
    _forward_evidence(database, configuration, clock, monkeypatch)
    return writer, feature, strategy, trading


def _repair_window_cycle(database, configuration, queue, clock, services, assignment, window):
    from src.research.canonical import CanonicalEvidenceError, SqlForwardEvidenceRepository
    from src.services.forward_windows import restart_operator_window

    def advance():
        clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(minutes=1)).isoformat()

    def observe(identity):
        queue.enqueue(
            job_id=identity,
            name="forward_paper_observation",
            available_at=clock[0],
            payload={
                "assignment_id": assignment["id"],
                "strategy_version_id": assignment["strategy_version_id"],
                "product_id": "active_income",
                "instrument_id": INSTRUMENT,
                "artefact_hash": assignment["artefact_hash"],
                "evaluation_time": clock[0],
            },
        )
        result = supervisor._promotion_cycle(
            database=database, node_id="fixture", configuration=configuration
        )()
        assert result["reason_code"] == "forward_observation_recorded", result

    advance()
    incident_id = "fixture-forward-feed-gap"
    with database.engine.begin() as connection:
        connection.execute(
            reconciliation_event.insert().values(
                id=incident_id,
                created_at=clock[0],
                payload={
                    "product_id": "active_income",
                    "instrument_id": INSTRUMENT,
                    "data_gap": True,
                },
            )
        )
    advance()
    observe("old-window-after-gap")
    evidence = SqlForwardEvidenceRepository(database.engine)
    scope = {
        "strategy_version_id": assignment["strategy_version_id"],
        "product_id": "active_income",
        "artefact_hash": assignment["artefact_hash"],
    }
    old_summary_id, old_summary = evidence.build_summary(**scope, observed_at=clock[0])
    assert incident_id in old_summary.data_gap_ids
    from src.portfolio.books import paper_book_account

    ledger = paper_book_account(database.engine, assignment["payload"]["book_id"])
    state_before = ledger.state()
    with database.engine.connect() as connection:
        positions_before = connection.execute(select(position.c.payload)).scalars().all()
    advance()
    request = {
        **scope,
        "request_id": "repair-fixture",
        "previous_window_id": window["id"],
        "requested_by": "fixture operator",
        "repair_reference": incident_id,
        "reason_code": "market_data_repaired",
    }
    restarted = restart_operator_window(database.engine, request, now=clock[0])
    assert restarted["window_id"] != window["id"]
    assert restart_operator_window(database.engine, request, now=clock[0]) == restarted
    later = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)).isoformat()
    assert restart_operator_window(database.engine, request, now=later) == restarted
    with pytest.raises(ValueError, match="identity collision"):
        restart_operator_window(
            database.engine, {**request, "repair_reference": "changed"}, now=later
        )
    with pytest.raises(ValueError, match="forward_window_superseded"):
        restart_operator_window(
            database.engine, {**request, "request_id": "stale-window"}, now=later
        )
    with pytest.raises(ValueError, match="operational repair"):
        restart_operator_window(
            database.engine, {**request, "reason_code": "loss_reset"}, now=later
        )
    assert ledger.state() == state_before
    with database.engine.connect() as connection:
        assert connection.execute(select(position.c.payload)).scalars().all() == positions_before
        assert (
            connection.execute(
                select(reconciliation_event.c.id).where(reconciliation_event.c.id == incident_id)
            ).scalar_one()
            == incident_id
        )
        assert len(connection.execute(select(forward_qualification_window)).all()) == 2
    decision_id = evidence.append_decision(
        summary_id=old_summary_id, decided_at=clock[0], accepted=True
    )
    with database.engine.connect() as connection:
        decision = connection.execute(
            select(forward_paper_decision.c.payload).where(
                forward_paper_decision.c.id == decision_id
            )
        ).scalar_one()
    assert decision["accepted"] is False and decision["reason_code"] == "forward_window_superseded"
    with pytest.raises(CanonicalEvidenceError, match="no forward observations"):
        evidence.build_summary(**scope, observed_at=clock[0])
    writer, feature, strategy, trading = services
    clock[0] = (
        (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(hours=1))
        .replace(minute=0, second=0)
        .isoformat()
    )
    _write_market_support(queue, writer, clock[0], 99, 105)
    queue.enqueue(
        job_id="post-repair-market",
        name="market_event_write",
        available_at=clock[0],
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(_candle(clock[0], 105, 105)),
        },
    )
    assert writer()["reason_code"] == "market_event_written"
    assert feature()["reason_code"] == "features_persisted"
    evaluated = strategy()
    assert evaluated["reason_code"] == "alpha_forecast_persisted", evaluated
    DatabaseHeartbeatStore(database.engine).record(
        service_name="strategy-evaluator",
        node_id="fixture",
        observed_at=clock[0],
        healthy=True,
        payload={"completed_job": evaluated["job_id"]},
    )
    results = [cycle() for cycle in trading]
    assert not any(result.get("error_type") for result in results), json.dumps(results, default=str)
    advance()
    observe("new-window-first-observation")
    _, new_summary = evidence.build_summary(**scope, observed_at=clock[0])
    assert new_summary.qualification_window_id == restarted["window_id"]
    assert incident_id not in new_summary.data_gap_ids
    assert set(new_summary.observation_ids).isdisjoint(old_summary.observation_ids)
    assert new_summary.net_pnl == pytest.approx(0.0)
    _resume_paper_cycle(database, configuration, queue, clock, services, assignment)


def _resume_paper_cycle(database, configuration, queue, clock, services, assignment):
    from src.portfolio.books import paper_book_account
    from src.research.canonical import (
        SqlActiveStrategyAssignmentRepository,
        SqlForwardEvidenceRepository,
    )

    ledger = paper_book_account(database.engine, assignment["payload"]["book_id"])
    state_before = ledger.state()
    with database.engine.connect() as connection:
        positions_before = connection.execute(select(position.c.payload)).scalars().all()
    worker = DatabasePromotionWorker(
        queue=queue,
        worker_id="promotion",
        store=SqlPromotionStore(database.engine),
        policy_store=SqlPromotionPolicyStore(database.engine),
        strict_identity_contract=True,
        paper_accounts=configured_paper_accounts(database.engine, configuration),
    )
    for transition, expected in (("suspend", "suspended"), ("resume", "forward_paper")):
        clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(minutes=1)).isoformat()
        queue.enqueue(
            job_id=f"window-{transition}",
            name="promotion_evaluation",
            available_at=clock[0],
            payload={
                "strategy_version_id": assignment["strategy_version_id"],
                "artefact_hash": assignment["artefact_hash"],
                "requested_transition": transition,
                "requested_capital": 0,
                "evaluated_at": clock[0],
            },
        )
        result = worker.run_once(now=clock[0])
        assert result["accepted"] is True and result["next_state"] == expected, result
    assert ledger.state() == state_before
    with database.engine.connect() as connection:
        assert connection.execute(select(position.c.payload)).scalars().all() == positions_before
        assert len(connection.execute(select(forward_qualification_window)).all()) == 3
    active = SqlActiveStrategyAssignmentRepository(database.engine).active_assignments(
        "active_income", at=clock[0]
    )
    assert len(active) == 1
    resumed = active[0]
    assert resumed["execution_mode"] == "paper"
    assert resumed["risk_budget"] <= assignment["risk_budget"]
    writer, feature, strategy, trading = services
    clock[0] = (
        (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(hours=1))
        .replace(minute=0, second=0)
        .isoformat()
    )
    _write_market_support(queue, writer, clock[0], 100, 105)
    queue.enqueue(
        job_id="post-resume-market",
        name="market_event_write",
        available_at=clock[0],
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(_candle(clock[0], 105, 105)),
        },
    )
    assert writer()["reason_code"] == "market_event_written"
    assert feature()["reason_code"] == "features_persisted"
    evaluated = strategy()
    assert evaluated["reason_code"] == "alpha_forecast_persisted", evaluated
    DatabaseHeartbeatStore(database.engine).record(
        service_name="strategy-evaluator",
        node_id="fixture",
        observed_at=clock[0],
        healthy=True,
        payload={"completed_job": evaluated["job_id"]},
    )
    results = [trading[0]()]
    results.extend(trading[1]() for _ in range(3))
    results.extend(cycle() for cycle in trading[2:])
    assert not any(result.get("error_type") for result in results), json.dumps(results, default=str)
    clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(minutes=1)).isoformat()
    queue.enqueue(
        job_id="resumed-observation",
        name="forward_paper_observation",
        available_at=clock[0],
        payload={
            "assignment_id": resumed["id"],
            "strategy_version_id": resumed["strategy_version_id"],
            "product_id": "active_income",
            "instrument_id": INSTRUMENT,
            "artefact_hash": resumed["artefact_hash"],
            "evaluation_time": clock[0],
        },
    )
    result = supervisor._promotion_cycle(
        database=database, node_id="fixture", configuration=configuration
    )()
    assert result["reason_code"] == "forward_observation_recorded", result
    with database.engine.connect() as connection:
        observed = connection.execute(
            select(forward_paper_observation.c.payload).where(
                forward_paper_observation.c.id == result["observation_id"]
            )
        ).scalar_one()
    assert observed["observation"]["position"] is not None
    previous_position = next(
        row
        for row in positions_before
        if row["instrument_id"] == INSTRUMENT
        and row["portfolio_id"] == assignment["payload"]["position_portfolio_id"]
    )
    with database.engine.connect() as connection:
        current_positions = connection.execute(select(position.c.payload)).scalars().all()
    current_position = next(
        row
        for row in current_positions
        if row["instrument_id"] == INSTRUMENT
        and row["portfolio_id"] == assignment["payload"]["position_portfolio_id"]
    )
    assert observed["observation"]["position"]["quantity"] == current_position["quantity"]
    assert (
        current_position["metadata"]["reference_entry_price"]
        == previous_position["metadata"]["reference_entry_price"]
    )
    _, summary = SqlForwardEvidenceRepository(database.engine).build_summary(
        strategy_version_id=resumed["strategy_version_id"],
        product_id="active_income",
        artefact_hash=resumed["artefact_hash"],
        observed_at=clock[0],
    )
    state_after = ledger.state()
    equities = [
        state.balances["USDT"]
        + sum(
            quantity * (105 - state.position_entry_prices[instrument])
            for instrument, quantity in state.positions.items()
        )
        for state in (state_before, state_after)
    ]
    assert summary.net_pnl == pytest.approx(equities[1] - equities[0])
    assert summary.net_pnl <= 0
    assert summary.completed_episodes == 0
    from src.data.assignment_scope import financial_assignment_ids
    from src.services.portfolio_service import SqlPortfolioRepository

    other = SqlActiveStrategyAssignmentRepository(database.engine).assign(
        product_id="active_income",
        portfolio_id=resumed["portfolio_id"],
        strategy_version_id=resumed["strategy_version_id"],
        artefact_hash=resumed["artefact_hash"],
        lifecycle_state="forward_paper",
        execution_mode="paper",
        capital_limit=0.01,
        risk_budget=0.01,
        assigned_at=clock[0],
        assigned_by="fixture",
        instrument_id=INSTRUMENT,
        sleeve_id="different-sleeve",
    )
    owners = financial_assignment_ids(database.engine, resumed["id"], at=clock[0])
    assert assignment["id"] in owners and resumed["id"] in owners and other not in owners
    with pytest.raises(ValueError, match="ownership requires reconciliation"):
        SqlPortfolioRepository(database.engine).prior_strategy_state(
            assignment_id=other,
            version_id=resumed["strategy_version_id"],
            portfolio_id=resumed["payload"]["position_portfolio_id"],
            instrument_id=INSTRUMENT,
            before=clock[0],
            interval_seconds=3600,
        )


def _forward_evidence(database, configuration, clock, monkeypatch):
    monkeypatch.setattr(
        "src.services.scheduler.AUTONOMOUS_SCHEDULES",
        (
            ScheduleSpec("forward_paper_observation", 300),
            ScheduleSpec("forward_paper_summary", 1),
        ),
    )
    product = next(
        row for row in configuration["products"]["products"] if row["product_id"] == "active_income"
    )
    scheduler = PlatformScheduler(
        engine=database.engine, products={"active_income": product}, node_id="fixture"
    )
    cycle = supervisor._promotion_cycle(
        database=database, node_id="fixture", configuration=configuration
    )
    clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)).isoformat()
    scheduler.run_once(now=clock[0])
    observed = cycle()
    assert observed["reason_code"] == "forward_observation_recorded", observed
    clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)).isoformat()
    scheduler.run_once(now=clock[0])
    summarised = cycle()
    assert summarised["reason_code"] != "forward_summary_failed", summarised
    with database.engine.connect() as connection:
        observations = connection.execute(select(forward_paper_observation)).mappings().all()
        summaries = connection.execute(select(forward_paper_summary)).mappings().all()
        decisions = connection.execute(select(forward_paper_decision)).mappings().all()
    assert len(observations) == len(summaries) == 1, summarised
    assert summaries[0]["payload"]["completed_attempts"] == 1
    assert summaries[0]["payload"]["valid_attempts"] == 1
    assert summaries[0]["payload"]["fill_rate"] == 1
    assert summaries[0]["payload"]["model_drift"] is None
    assert summaries[0]["payload"]["model_drift_applicable"] is False
    assert summaries[0]["payload"]["observed_observations"] == 6
    assert summaries[0]["payload"]["expected_observations"] == 6
    assert summaries[0]["payload"]["data_uptime"] == 1
    assert (
        abs(summaries[0]["payload"]["net_pnl"] - summaries[0]["payload"]["objective_excess"]) < 1e-8
    )
    assert len(decisions) == 1 and decisions[0]["accepted"] is False
    assert decisions[0]["reason_code"] == "forward_evidence_duration_insufficient"
    assert decisions[0]["payload"]["research_state"] == "economically_inconclusive"


def _candle(at, price, previous, *, timeframe="1h"):
    millis = int(dt.datetime.fromisoformat(at).timestamp() * 1000)
    return normalise_public_event(
        market="futures",
        stream=f"btcusdt@kline_{timeframe}",
        receive_timestamp=at,
        payload={
            "e": "kline",
            "E": millis,
            "s": "BTCUSDT",
            "k": {
                "t": millis - timeframe_seconds(timeframe) * 1000,
                "T": millis - 1,
                "i": timeframe,
                "o": str(previous),
                "h": str(max(price, previous) * 1.001),
                "l": str(min(price, previous) * 0.999),
                "c": str(price),
                "v": "100000",
                "x": True,
            },
        },
    )


def test_cohort_waits_for_all_trials_and_cannot_expand_after_freeze(tmp_path, monkeypatch):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'cohort.sqlite3'}")
    database.create_schema()
    bundle = _bundle(database)
    first = _candidate(database, bundle)
    second = replace(
        first,
        definition=replace(
            first.definition,
            signal_model={
                **first.definition.signal_model,
                "parameters": {"fast": 3, "slow": 5, "allow_short": True},
            },
        ),
    )
    ResearchCoordinator(SqlResearchStore(database.engine)).submit(second)
    with pytest.raises(CohortPending, match="epoch must close"):
        prepare_cohort(database.engine, first, now=NOW, policy_hash="not-opened")
    ResearchRoundCoordinator(database.engine).close_finite(
        bundle.bundle_id,
        product_id="active_income",
        candidate_ids=[first.candidate_id, second.candidate_id],
        at=NOW,
    )
    configuration = load_split_configuration(Path(__file__).resolve().parents[1] / "config")
    product = next(
        row for row in configuration["products"]["products"] if row["product_id"] == "active_income"
    )
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="cohort",
        node_id="fixture",
        role="research",
        capabilities=("evaluate_candidate",),
        observed_at=NOW,
    )
    handlers = DatabaseResearchJobHandlers(
        SqlResearchStore(database.engine),
        configuration=configuration,
        dataset_resolver=CanonicalDatasetResolver(SqlCanonicalDatasetRepository(database.engine)),
    )
    monkeypatch.setattr(
        "src.services.scheduler.AUTONOMOUS_SCHEDULES", (ScheduleSpec("candidate_evaluation", 1),)
    )
    scheduler = PlatformScheduler(
        engine=database.engine, products={"active_income": product}, node_id="fixture"
    )
    clock = [NOW]
    monkeypatch.setattr("src.services.research_jobs.utc_now", lambda clock=clock: clock[0])
    for step, stage in enumerate(("screening", "development", "robustness")):
        clock[0] = (dt.datetime.fromisoformat(NOW) + dt.timedelta(seconds=step)).isoformat()
        scheduler.run_once(now=clock[0])
        for index in range(2):
            claimed = queue.claim(
                worker_id="cohort", now=clock[0], lease_seconds=60, names=("evaluate_candidate",)
            )
            assert claimed is not None and claimed.payload["requested_stage"] == stage
            result = handlers.evaluate_candidate(claimed, lambda claimed=claimed: claimed)
            assert result["accepted"], result
            queue.complete(claimed, completed_at=clock[0], outcome=result)
            if stage == "robustness" and index == 0:
                with pytest.raises(CohortPending, match="waits for trial"):
                    prepare_cohort(
                        database.engine,
                        first,
                        now=clock[0],
                        policy_hash=handlers.evidence_policy.policy_hash,
                    )
                with database.engine.connect() as connection:
                    assert connection.execute(select(selection_cohort)).first() is None
    with pytest.raises(CohortPending, match="next clock tick"):
        prepare_cohort(
            database.engine, first, now=clock[0], policy_hash=handlers.evidence_policy.policy_hash
        )
    clock[0] = (dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)).isoformat()
    cohort = prepare_cohort(
        database.engine, first, now=clock[0], policy_hash=handlers.evidence_policy.policy_hash
    )
    assert set(cohort["members"]) == {first.candidate_id, second.candidate_id}
    late = replace(
        first,
        definition=replace(
            first.definition,
            signal_model={
                **first.definition.signal_model,
                "parameters": {"fast": 4, "slow": 6, "allow_short": True},
            },
        ),
    )
    with pytest.raises(DataUnavailableError, match="outside the already frozen"):
        prepare_cohort(
            database.engine, late, now=clock[0], policy_hash=handlers.evidence_policy.policy_hash
        )
    scheduler.run_once(now=clock[0])
    for _ in range(2):
        claimed = queue.claim(
            worker_id="cohort", now=clock[0], lease_seconds=60, names=("evaluate_candidate",)
        )
        assert claimed is not None and claimed.payload["requested_stage"] == "protected"
        result = handlers.evaluate_candidate(claimed, lambda claimed=claimed: claimed)
        assert result["accepted"], result
        queue.complete(claimed, completed_at=clock[0], outcome=result)
    with database.engine.connect() as connection:
        result = connection.execute(select(selection_cohort_result.c.payload)).scalar_one()
    assert set(result["adjusted_pvalues"]) == {first.candidate_id, second.candidate_id}
    assert result["qualification_authority"] is False
    revised = _candidate(database, _bundle(database, source_revision=1))
    ResearchRoundCoordinator(database.engine).close_finite(
        revised.dataset_bundle_id,
        product_id="active_income",
        candidate_ids=[revised.candidate_id],
        at=clock[0],
    )
    with pytest.raises(DataUnavailableError, match="overlaps an already opened cohort"):
        prepare_cohort(
            database.engine, revised, now=clock[0], policy_hash=handlers.evidence_policy.policy_hash
        )
    database.dispose()


def _write_market_support(queue, writer, at, index, price):
    millis = int(dt.datetime.fromisoformat(at).timestamp() * 1000)
    messages = (
        (
            "btcusdt@bookTicker",
            {
                "e": "bookTicker",
                "E": millis,
                "s": "BTCUSDT",
                "b": str(price - 0.01),
                "a": str(price + 0.01),
                "B": "100000",
                "A": "100000",
            },
        ),
        (
            "btcusdt@markPrice@1s",
            {
                "e": "markPriceUpdate",
                "E": millis,
                "s": "BTCUSDT",
                "p": str(price),
                "r": "0",
                "T": millis + 28800000,
            },
        ),
    )
    for kind, (stream, payload) in enumerate(messages):
        event = normalise_public_event(
            market="futures", stream=stream, receive_timestamp=at, payload=payload
        )
        queue.enqueue(
            job_id=f"lifecycle-support-{index}-{kind}",
            name="market_event_write",
            payload={
                "venue": "binance",
                "market": "futures",
                "symbol": "BTCUSDT",
                "event": to_primitive(event),
            },
            available_at=at,
        )
        result = writer()
        assert result["reason_code"] == "market_event_written", result

    for service_name in ("market-gateway", "data-writer"):
        DatabaseHeartbeatStore(queue.engine).record(
            service_name=service_name,
            node_id="fixture",
            observed_at=at,
            healthy=True,
            payload={"reason_code": "fixture_public_events_persisted"},
        )


def test_protected_economic_result_ignores_other_market_and_excludes_exhausted_worker(
    tmp_path, monkeypatch
):
    from src.data.database import holdout_outcome

    protected_results = []
    for scale in (1, 1000):
        database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / f'scope-{scale}.sqlite3'}")
        database.create_schema()
        bundle = _bundle(database, extra_market_scale=scale)
        candidate = _candidate(database, bundle)
        excluded = replace(
            candidate,
            definition=replace(
                candidate.definition,
                signal_model={
                    **candidate.definition.signal_model,
                    "parameters": {"fast": 3, "slow": 5, "allow_short": True},
                },
            ),
        )
        ResearchCoordinator(SqlResearchStore(database.engine)).submit(excluded)
        ResearchRoundCoordinator(database.engine).close_finite(
            bundle.bundle_id,
            product_id="active_income",
            candidate_ids=[candidate.candidate_id, excluded.candidate_id],
            at=NOW,
        )
        product = {"product_id": "active_income", "dataset_timeframe": "1h"}
        scheduler = PlatformScheduler(
            engine=database.engine, products={"active_income": product}, node_id="fixture"
        )
        scheduler._mark_candidate_dead_letter(excluded.candidate_id, "development", NOW)
        queue = DatabaseJobQueue(database.engine)
        queue.register_worker(
            worker_id="scope-worker",
            node_id="fixture",
            role="research",
            capabilities=("evaluate_candidate",),
            observed_at=NOW,
        )
        import hashlib

        from test_platform_validation_contracts import _approval_calibration_fixture

        from src.research.policy_approval import approve_policy

        configuration = load_split_configuration(Path(__file__).resolve().parents[1] / "config")
        calibration = tmp_path / f"scope-calibration-{scale}.json"
        calibration.write_text(json.dumps(_approval_calibration_fixture(configuration["research"])))
        approve_policy(
            database.engine,
            receipt_path=calibration,
            expected_hash="sha256:" + hashlib.sha256(calibration.read_bytes()).hexdigest(),
            configuration=configuration["research"],
            approved_by="test operator",
            approved_at=NOW,
        )
        handlers = DatabaseResearchJobHandlers(
            SqlResearchStore(database.engine),
            configuration=configuration,
            dataset_resolver=CanonicalDatasetResolver(
                SqlCanonicalDatasetRepository(database.engine)
            ),
        )
        monkeypatch.setattr(
            "src.services.scheduler.AUTONOMOUS_SCHEDULES",
            (ScheduleSpec("candidate_evaluation", 1),),
        )
        clock = [NOW]
        monkeypatch.setattr("src.services.research_jobs.utc_now", lambda clock=clock: clock[0])
        protected = False
        for tick in range(6):
            clock[0] = (dt.datetime.fromisoformat(NOW) + dt.timedelta(seconds=tick)).isoformat()
            scheduler.run_once(now=clock[0])
            claimed = queue.claim(
                worker_id="scope-worker",
                now=clock[0],
                lease_seconds=60,
                names=("evaluate_candidate",),
            )
            assert claimed is not None
            result = handlers.evaluate_candidate(claimed, lambda claimed=claimed: claimed)
            if result.get("reason_code") == "cohort_pending":
                queue.defer(
                    claimed,
                    available_at=(
                        dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)
                    ).isoformat(),
                    reason=result["_defer_reason"],
                )
                continue
            assert result["accepted"], result
            queue.complete(claimed, completed_at=clock[0], outcome=result)
            if result["stage"] == "protected":
                protected = True
                break
        assert protected
        with database.engine.connect() as connection:
            cohort = connection.execute(select(selection_cohort.c.payload)).scalar_one()
            outcome = connection.execute(select(holdout_outcome.c.payload)).scalar_one()
        assert cohort["excluded_trials"][excluded.candidate_id] == "infrastructure_exhausted"
        protected_results.append(outcome)
        database.dispose()
    left, right = protected_results
    assert left["metrics"] == right["metrics"]
    assert left["selection_stream"] == right["selection_stream"]
    for outcome in protected_results:
        receipt = outcome["execution_receipt"]
        assert receipt["evaluation_input"]["instrument_scope"] == [INSTRUMENT]
        assert receipt["evaluation_input"]["dataset_roles"] == ["protected_holdout"]
