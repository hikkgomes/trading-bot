from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError


@pytest.mark.skipif(
    not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL roles",
)
def test_privilege_fixture_is_explicitly_enabled() -> None:
    # Role grants are applied by the owner Alembic revision. This test remains
    # opt-in because local developer databases do not normally contain roles.
    assert os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") == "1"


def _as_role(role: str, statement: str, parameters: dict | None = None):
    engine = create_engine(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    try:
        with engine.begin() as connection:
            connection.execute(text(f"SET LOCAL ROLE {role}"))
            result = connection.execute(text(statement), parameters or {})
            return result.all() if result.returns_rows else []
    finally:
        engine.dispose()


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="requires PostgreSQL role fixture",
)
def test_service_login_cannot_hide_runtime_membership_by_switching_role():
    from src.services.database_roles import check_connection_role

    role = "role_boundary_" + uuid.uuid4().hex[:16]
    password = uuid.uuid4().hex
    admin = create_engine(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    service = None
    try:
        with admin.begin() as connection:
            connection.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'"))
            connection.execute(text(f"GRANT trading_research TO {role}"))
        service = create_engine(
            make_url(os.environ["TRADING_PLATFORM_DATABASE_URL"]).set(
                username=role, password=password
            )
        )
        with service.connect() as connection:
            check_connection_role(connection, "trading_research")
        with admin.begin() as connection:
            connection.execute(text(f"ALTER ROLE {role} CREATEROLE"))
        with service.connect() as connection:
            with pytest.raises(PermissionError, match="only to service domain"):
                check_connection_role(connection, "trading_research")
        with admin.begin() as connection:
            connection.execute(text(f"ALTER ROLE {role} NOCREATEROLE"))
            connection.execute(text(f"GRANT trading_runtime TO {role}"))
        with service.begin() as connection:
            with pytest.raises(PermissionError, match="only to service domain"):
                check_connection_role(connection, "trading_research")
            connection.execute(text("SET LOCAL ROLE trading_research"))
            with pytest.raises(PermissionError, match="only to service domain"):
                check_connection_role(connection, "trading_research")
    finally:
        if service is not None:
            service.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP ROLE IF EXISTS {role}"))
        admin.dispose()


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="requires PostgreSQL role fixture",
)
def test_evaluator_completes_research_without_runtime_membership(monkeypatch, tmp_path):
    from test_platform_candidate_lifecycle import NOW, _bundle, _candidate

    from src.data.database import PlatformDatabase
    from src.research.datasets import CanonicalDatasetResolver, SqlCanonicalDatasetRepository
    from src.research.store import SqlResearchStore
    from src.services.config import load_split_configuration
    from src.services.database_roles import check_connection_role
    from src.services.research_jobs import DatabaseResearchJobHandlers
    from src.services.scheduler import DatabaseJobQueue, PlatformScheduler, ScheduleSpec

    token = uuid.uuid4().hex[:16]
    name, role, password = "evaluation_" + token, "evaluator_" + token, uuid.uuid4().hex
    base = make_url(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    database = evaluator = None
    try:
        with admin.connect() as connection:
            connection.execute(text(f"CREATE DATABASE {name}"))
            connection.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'"))
            connection.execute(text(f"GRANT trading_evaluator TO {role}"))
        database = PlatformDatabase(base.set(database=name).render_as_string(hide_password=False))
        database.migrate()
        bundle = _bundle(database)
        candidate = _candidate(database, bundle)
        from dataclasses import replace

        from sqlalchemy import select, update

        from src.data.database import experiment, research_epoch
        from src.research.coordinator import ResearchCoordinator
        from src.research.rounds import ResearchRoundCoordinator

        evaluator = create_engine(base.set(database=name, username=role, password=password))
        with evaluator.connect() as connection:
            check_connection_role(connection, "trading_evaluator")
        probe = replace(
            candidate,
            definition=replace(
                candidate.definition,
                signal_model={
                    **candidate.definition.signal_model,
                    "parameters": {"fast": 3, "slow": 5, "allow_short": True},
                },
            ),
        )
        ResearchCoordinator(SqlResearchStore(evaluator)).submit(probe)
        with evaluator.begin() as connection:
            assert (
                connection.execute(
                    select(research_epoch.c.consumed).where(
                        research_epoch.c.dataset_bundle_id == bundle.bundle_id
                    )
                ).scalar_one()
                == 2
            )
            connection.execute(
                update(experiment)
                .where(experiment.c.id == probe.candidate_id)
                .values(state="unsupported")
            )
        ResearchRoundCoordinator(database.engine).close_finite(
            bundle.bundle_id,
            product_id="active_income",
            candidate_ids=[candidate.candidate_id, probe.candidate_id],
            at=NOW,
        )
        queue = DatabaseJobQueue(evaluator)
        queue.register_worker(
            worker_id="evaluation",
            node_id="fixture",
            role="evaluation-worker",
            capabilities=("evaluate_candidate",),
            observed_at=NOW,
        )
        configuration = load_split_configuration(Path(__file__).resolve().parents[1] / "config")
        product = next(
            p for p in configuration["products"]["products"] if p["product_id"] == "active_income"
        )
        product["dataset_timeframe"] = "1h"
        import hashlib

        from test_platform_validation_contracts import _approval_calibration_fixture

        from src.research.policy_approval import approve_policy

        calibration_path = tmp_path / "evaluator-calibration.json"
        calibration_path.write_text(
            json.dumps(_approval_calibration_fixture(configuration["research"]))
        )
        approve_policy(
            database.engine,
            receipt_path=calibration_path,
            expected_hash="sha256:" + hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            configuration=configuration["research"],
            approved_by="test operator",
            approved_at=NOW,
        )
        handlers = DatabaseResearchJobHandlers(
            SqlResearchStore(evaluator),
            configuration=configuration,
            dataset_resolver=CanonicalDatasetResolver(SqlCanonicalDatasetRepository(evaluator)),
        )
        monkeypatch.setattr(
            "src.services.scheduler.AUTONOMOUS_SCHEDULES",
            (ScheduleSpec("candidate_evaluation", 1),),
        )
        scheduler = PlatformScheduler(
            engine=database.engine, products={"active_income": product}, node_id="fixture"
        )
        clock = [NOW]
        monkeypatch.setattr("src.services.research_jobs.utc_now", lambda: clock[0])
        for index, stage in enumerate(("screening", "development", "robustness", "protected")):
            clock[0] = (
                dt.datetime.fromisoformat(NOW) + dt.timedelta(seconds=index * 2)
            ).isoformat()
            scheduler.run_once(now=clock[0])
            claimed = queue.claim(
                worker_id="evaluation",
                now=clock[0],
                lease_seconds=60,
                names=("evaluate_candidate",),
            )
            assert claimed is not None, stage
            result = handlers.evaluate_candidate(claimed, lambda claimed=claimed: claimed)
            if result.get("reason_code") == "cohort_pending":
                clock[0] = (
                    dt.datetime.fromisoformat(clock[0]) + dt.timedelta(seconds=1)
                ).isoformat()
                queue.defer(claimed, available_at=clock[0], reason=result["_defer_reason"])
                claimed = queue.claim(
                    worker_id="evaluation",
                    now=clock[0],
                    lease_seconds=60,
                    names=("evaluate_candidate",),
                )
                result = handlers.evaluate_candidate(claimed, lambda claimed=claimed: claimed)
            assert result["accepted"], result
            queue.complete(claimed, completed_at=clock[0], outcome=result)
        assert result["artefact_hash"]
        from src.services.control_api import DatabaseControlPlane
        from src.services.health import DatabaseHeartbeatStore

        owner_controls = DatabaseControlPlane(
            database.engine, DatabaseHeartbeatStore(database.engine)
        )
        evaluator_controls = DatabaseControlPlane(evaluator, DatabaseHeartbeatStore(evaluator))
        assert not evaluator_controls.service_is_paused("evaluation-worker")
        owner_controls.set_paused(
            target="service:evaluation-worker",
            paused=True,
            reason_code="fixture_pause",
            requested_by="operator",
            changed_at=NOW,
        )
        assert evaluator_controls.service_is_paused("evaluation-worker")
        heartbeats = DatabaseHeartbeatStore(evaluator, retention_per_service=2)
        for second in range(3):
            heartbeats.record(
                service_name="evaluation-worker",
                node_id="fixture",
                observed_at=f"2026-08-23T00:00:0{second}+00:00",
                healthy=True,
            )
        assert heartbeats.latest()[0].observed_at == "2026-08-23T00:00:02+00:00"
        with database.engine.begin() as connection:
            connection.execute(
                text("""UPDATE experiment SET metadata=jsonb_set(metadata,
                    '{evaluation_deferred}',
                    '{"stage":"protected","reason_code":"private_fixture"}'::jsonb)
                    WHERE id=:id"""),
                {"id": candidate.candidate_id},
            )
            connection.execute(
                text("""INSERT INTO experiment
                    (id, strategy_version_id, provider, state, submitted_at,
                        dataset_snapshot_hashes, dataset_bundle_id, metadata)
                    SELECT 'legacy-fixture', strategy_version_id, provider, 'legacy_import',
                    submitted_at, dataset_snapshot_hashes, NULL, metadata - '_dataset_plan'
                    FROM experiment WHERE id=:id"""),
                {"id": candidate.candidate_id},
            )
        with database.engine.begin() as connection:
            connection.execute(text("SET LOCAL ROLE trading_research"))
            assert (
                connection.execute(
                    text("SELECT count(*) FROM job WHERE name='evaluate_candidate'")
                ).scalar_one()
                == 0
            )
            assert (
                connection.execute(
                    text("SELECT count(*) FROM validation_stage WHERE stage='protected'")
                ).scalar_one()
                == 0
            )
        adaptive = create_engine(
            base.set(database=name), connect_args={"options": "-c role=trading_research"}
        )
        try:
            from src.services.agent_worker import adaptive_research_report

            report = adaptive_research_report(adaptive)
            assert report["research"]["funnel"]["candidates_generated"] == 2
            restored = SqlResearchStore(adaptive).get_candidate(candidate.candidate_id)
            assert restored.candidate_id == candidate.candidate_id
            assert "evaluation_deferred" not in restored.metadata
            with adaptive.connect() as connection:
                assert (
                    connection.execute(
                        text("SELECT state FROM adaptive_candidate WHERE id=:id"),
                        {"id": candidate.candidate_id},
                    ).scalar_one()
                    == "queued"
                )
            for column in ("state", "metadata"):
                with adaptive.connect() as connection:
                    with pytest.raises(DBAPIError, match="permission denied"):
                        connection.execute(text(f"SELECT {column} FROM experiment"))
            controls = DatabaseControlPlane(adaptive, DatabaseHeartbeatStore(adaptive))
            assert not controls.service_is_paused("research-worker")
            assert not controls.service_is_paused("evaluation-worker")
            assert report["research"]["funnel"]["candidates_evaluated"] == 1
            public_queue = DatabaseJobQueue(adaptive)
            public_queue.register_worker(
                worker_id="research",
                node_id="fixture",
                role="research-worker",
                capabilities=("bounded_backtest",),
                observed_at=NOW,
            )
            assert public_queue.enqueue_if_absent(
                job_id="public-work",
                name="bounded_backtest",
                payload={"fixture": True},
                available_at=NOW,
            )
            assert not public_queue.enqueue_if_absent(
                job_id="public-work",
                name="bounded_backtest",
                payload={"fixture": True},
                available_at=NOW,
            )
            claimed = public_queue.claim(
                worker_id="research", now=NOW, lease_seconds=60, names=("bounded_backtest",)
            )
            assert claimed is not None
            public_queue.complete(claimed, completed_at=NOW)
        finally:
            adaptive.dispose()
    finally:
        if evaluator is not None:
            evaluator.dispose()
        if database is not None:
            database.dispose()
        with admin.connect() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {name}"))
            connection.execute(text(f"DROP ROLE IF EXISTS {role}"))
        admin.dispose()


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="requires PostgreSQL role fixture",
)
def test_every_platform_role_has_only_its_declared_authority() -> None:
    run_id = uuid.uuid4().hex
    research_job_id = f"role-test-research-job-{run_id}"
    agent_proposal_id = f"role-test-agent-proposal-{run_id}"
    _as_role("trading_runtime", "SELECT id FROM balance_snapshot LIMIT 1")
    buffer_id = f"role-test-market-input-{run_id}"
    market_payload = {
        "kind": "market_data_input",
        "product_id": "active_income",
        "instrument_id": "binance:futures:BTCUSDT:USDT",
        "source_event_type": "best_bid_ask",
    }
    statement = (
        "INSERT INTO market_input_buffer (id, product_id, instrument_id, event_type, created_at, payload) "
        "VALUES (:id, 'active_income', 'binance:futures:BTCUSDT:USDT', 'best_bid_ask', "
        "CURRENT_TIMESTAMP, CAST(:payload AS jsonb))"
    )
    _as_role("trading_runtime", statement, {"id": buffer_id, "payload": json.dumps(market_payload)})
    with pytest.raises(DBAPIError):
        _as_role(
            "trading_runtime",
            "UPDATE market_input_buffer SET payload=payload WHERE id=:id",
            {"id": buffer_id},
        )
    with pytest.raises(DBAPIError):
        _as_role(
            "trading_runtime",
            statement,
            {
                "id": buffer_id + "-invalid",
                "payload": json.dumps({**market_payload, "kind": "balance_snapshot"}),
            },
        )
    with pytest.raises(DBAPIError):
        _as_role(
            "trading_research", "DELETE FROM market_input_buffer WHERE id=:id", {"id": buffer_id}
        )
    _as_role("trading_runtime", "DELETE FROM market_input_buffer WHERE id=:id", {"id": buffer_id})
    _as_role("trading_research", "SELECT id FROM instrument LIMIT 1")
    with pytest.raises(DBAPIError):
        _as_role("trading_research", "SELECT id FROM balance_snapshot LIMIT 1")
    with pytest.raises(DBAPIError):
        _as_role(
            "trading_research",
            "INSERT INTO job (id, name, state, priority, available_at, attempts, "
            "producer_identity, content_hash, payload) VALUES "
            "('forbidden', 'evaluate_candidate', 'queued', 0, CURRENT_TIMESTAMP, 0, "
            "'research', 'sha256:' || repeat('1', 64), '{}'::jsonb)",
        )
    rows = _as_role(
        "trading_research",
        "SELECT submit_typed_research_job(CAST(:request AS jsonb))",
        {
            "request": json.dumps(
                {
                    "id": research_job_id,
                    "name": "evaluate_candidate",
                    "payload": {},
                    "state": "pending",
                    "attempts": 0,
                    "max_attempts": 3,
                    "available_at": "2026-09-09T00:00:00+00:00",
                    "priority": 0,
                    "producer_identity": "research:test",
                    "content_hash": "sha256:" + "1" * 64,
                }
            )
        },
    )
    assert rows == [(True,)]

    _as_role(
        "trading_agent",
        "INSERT INTO agent_proposal (id, created_at, payload) "
        "VALUES (:id, CURRENT_TIMESTAMP, '{}'::jsonb)",
        {"id": agent_proposal_id},
    )
    with pytest.raises(DBAPIError):
        _as_role("trading_agent", "SELECT id FROM strategy_approval LIMIT 1")
    with pytest.raises(DBAPIError):
        _as_role(
            "trading_agent",
            "SELECT submit_typed_research_job(CAST(:request AS jsonb))",
            {"request": json.dumps({"id": "forbidden", "name": "live_order_submit"})},
        )

    with pytest.raises(DBAPIError):
        _as_role("trading_platform_owner", "SELECT id FROM balance_snapshot LIMIT 1")


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="requires PostgreSQL role fixture",
)
def test_runtime_cannot_write_human_live_approval() -> None:
    statement = "INSERT INTO strategy_approval SELECT * FROM strategy_approval WHERE FALSE"
    assert _as_role("trading_operator", statement) == []
    _as_role("trading_runtime", "SELECT id FROM strategy_approval LIMIT 1")
    with pytest.raises(DBAPIError) as rejected:
        _as_role("trading_runtime", statement)
    assert rejected.value.orig.sqlstate == "42501"
    assert _as_role(
        "trading_runtime", "SELECT pg_has_role(current_user, 'trading_operator', 'MEMBER')"
    ) == [(False,)]


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="requires PostgreSQL role fixture",
)
def test_only_operator_can_approve_research_policy():
    for table in ("research_calibration", "research_policy_approval"):
        statement = f"INSERT INTO {table} SELECT * FROM {table} WHERE FALSE"
        assert _as_role("trading_operator", statement) == []
        for role in ("trading_runtime", "trading_research", "trading_agent"):
            with pytest.raises(DBAPIError) as rejected:
                _as_role(role, statement)
            assert rejected.value.orig.sqlstate == "42501"


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="requires PostgreSQL role fixture",
)
@pytest.mark.parametrize("phase", ["holdout", "forward"])
def test_adaptive_roles_cannot_read_protected_data_or_legacy_holdout_metrics(phase):
    run_id = uuid.uuid4().hex
    ids = {
        role: f"visibility-{role}-{run_id}"
        for role in ("development", "protected_holdout", "forward_observation")
    }
    engine = create_engine(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    with engine.begin() as connection:
        for role, identity in ids.items():
            connection.execute(
                text(
                    "INSERT INTO dataset_snapshot(id, created_at, payload) VALUES (:id, CURRENT_TIMESTAMP, jsonb_build_object('role', CAST(:role AS text), 'rows', jsonb_build_array(123)))"
                ),
                {"id": identity, "role": role},
            )
        connection.execute(
            text(
                "INSERT INTO experiment_run(id, created_at, payload) VALUES (:id, CURRENT_TIMESTAMP, CAST(:payload AS jsonb))"
            ),
            {
                "id": run_id,
                "payload": json.dumps({"phase": phase, "metrics": {"net_return": 123}}),
            },
        )
        connection.execute(
            text(
                "INSERT INTO experiment_metric(id, created_at, payload) VALUES (:id, CURRENT_TIMESTAMP, CAST(:payload AS jsonb))"
            ),
            {"id": run_id, "payload": json.dumps({"run_id": run_id, "value": 123})},
        )
    parameters = {
        "development": ids["development"],
        "protected": ids["protected_holdout"],
        "forward": ids["forward_observation"],
    }
    query = "SELECT id FROM dataset_snapshot WHERE id IN (:development, :protected, :forward) ORDER BY id"
    assert _as_role("trading_research", query, parameters) == [(ids["development"],)]
    assert len(_as_role("trading_runtime", query, parameters)) == 3
    assert (
        _as_role(
            "trading_research", "SELECT payload FROM experiment_run WHERE id=:id", {"id": run_id}
        )
        == []
    )
    assert (
        _as_role(
            "trading_research", "SELECT payload FROM experiment_metric WHERE id=:id", {"id": run_id}
        )
        == []
    )
    for role in ("trading_research", "trading_agent"):
        with pytest.raises(DBAPIError):
            _as_role(role, "SELECT payload FROM holdout_outcome LIMIT 1")
        with pytest.raises(DBAPIError):
            _as_role(role, "SELECT payload FROM legacy_research_archive LIMIT 1")
    engine.dispose()


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="requires PostgreSQL role fixture",
)
def test_real_result_writer_does_not_expose_protected_stage_feedback():
    from test_platform_candidate_lifecycle import NOW, _bundle, _candidate

    from src.data.database import PlatformDatabase
    from src.research.coordinator import ResearchCoordinator
    from src.research.store import SqlResearchStore

    database = PlatformDatabase(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    candidate = _candidate(database, _bundle(database))
    coordinator = ResearchCoordinator(SqlResearchStore(database.engine))

    def passed(_candidate):
        return True, None, {"fixture_metric": 123}

    def rejected(_candidate):
        return False, "fixture_rejection", {"fixture_metric": 123}

    for development, protected in ((rejected, passed), (passed, rejected), (passed, passed)):
        coordinator.evaluate(
            candidate.candidate_id,
            screening=passed,
            development=development,
            robustness=passed,
            protected=protected,
            evaluated_at=NOW,
        )
    query = "SELECT state FROM validation_result WHERE experiment_id=:id ORDER BY state"
    parameters = {"id": candidate.candidate_id}
    assert _as_role("trading_runtime", query, parameters) == [
        ("development_rejected",),
        ("forward_paper",),
        ("protected_rejected",),
    ]
    assert _as_role("trading_research", query, parameters) == [("development_rejected",)]
    database.dispose()


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="requires PostgreSQL role fixture",
)
def test_runtime_prunes_only_expired_tagged_public_market_batch_jobs():
    token = uuid.uuid4().hex[:16]
    worker_id = f"retention-worker-{token}"
    cases = {
        "expired": (
            "market_event_batch_write",
            "completed",
            8,
            True,
            "completed",
            "released",
            False,
        ),
        "recent": (
            "market_event_batch_write",
            "completed",
            1,
            True,
            "completed",
            "released",
            False,
        ),
        "legacy": (
            "market_event_batch_write",
            "completed",
            8,
            False,
            "completed",
            "released",
            False,
        ),
        "financial": (
            "financial_live_followup",
            "completed",
            8,
            True,
            "completed",
            "released",
            False,
        ),
        "research": ("research_development", "completed", 8, True, "completed", "released", False),
        "user_stream": ("user_stream_event", "completed", 8, True, "completed", "released", False),
        "pending": ("market_event_batch_write", "pending", 8, True, None, None, False),
        "running": ("market_event_batch_write", "running", 8, True, "running", "active", False),
        "dead_letter": (
            "market_event_batch_write",
            "dead_letter",
            8,
            True,
            "failed",
            "released",
            False,
        ),
        "audited": (
            "market_event_batch_write",
            "completed",
            8,
            True,
            "completed",
            "released",
            True,
        ),
    }
    job_ids = {key: f"market-retention-{key}-{token}" for key in cases}
    now = dt.datetime.now(dt.UTC)
    engine = create_engine(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    try:
        with engine.begin() as connection:
            connection.execute(
                text("""INSERT INTO worker
                    (id, node_id, role, last_heartbeat, status, capabilities, payload)
                    VALUES (:id, 'retention-test', 'market-writer', :now, 'ready', '[]', '{}')"""),
                {"id": worker_id, "now": now},
            )
            for key, (
                name,
                state,
                age_days,
                tagged,
                attempt_status,
                lease_status,
                audited,
            ) in cases.items():
                job_id = job_ids[key]
                completed_at = now - dt.timedelta(days=age_days)
                payload = (
                    {"queue_retention_policy": "public_market_batch_completed_7d/v1"}
                    if tagged
                    else {}
                )
                content_hash = "sha256:" + uuid.uuid4().hex + uuid.uuid4().hex
                connection.execute(
                    text("""INSERT INTO job
                        (id, name, state, priority, available_at, attempts, max_attempts,
                            producer_identity, content_hash, payload)
                        VALUES (:id, :name, :state, 0, :available_at, :attempts, 3,
                            'retention-fixture', :content_hash, CAST(:payload AS jsonb))"""),
                    {
                        "id": job_id,
                        "name": name,
                        "state": state,
                        "available_at": completed_at,
                        "attempts": 0 if attempt_status is None else 1,
                        "content_hash": content_hash,
                        "payload": json.dumps(payload),
                    },
                )
                if attempt_status is None:
                    continue
                attempt_id = f"{job_id}:1"
                finished = None if attempt_status == "running" else completed_at
                connection.execute(
                    text("""INSERT INTO job_attempt
                        (id, job_id, worker_id, started_at, completed_at, status, payload)
                        VALUES (:id, :job_id, :worker_id, :started_at, :completed_at,
                            :status, '{}')"""),
                    {
                        "id": attempt_id,
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "started_at": completed_at,
                        "completed_at": finished,
                        "status": attempt_status,
                    },
                )
                connection.execute(
                    text("""INSERT INTO worker_lease
                        (id, job_id, worker_id, expires_at, status, payload)
                        VALUES (:id, :job_id, :worker_id, :expires_at, :status, '{}')"""),
                    {
                        "id": attempt_id,
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "expires_at": completed_at,
                        "status": lease_status,
                    },
                )
                if audited:
                    connection.execute(
                        text("""INSERT INTO job_quarantine
                            (id, job_id, quarantined_at, original_state, reason, content_hash, payload)
                            VALUES (:id, :job_id, :now, 'completed', 'retention_test',
                                :content_hash, CAST(:payload AS jsonb))"""),
                        {
                            "id": f"retention-audit-{token}",
                            "job_id": job_id,
                            "now": now,
                            "content_hash": content_hash,
                            "payload": json.dumps(payload),
                        },
                    )

        assert _as_role(
            "trading_runtime", "SELECT public.prune_completed_public_market_batch_jobs()"
        ) == [(1,)]
        with engine.connect() as connection:
            remaining = {
                key
                for key, job_id in job_ids.items()
                if connection.execute(
                    text("SELECT 1 FROM job WHERE id=:id"), {"id": job_id}
                ).first()
            }
            assert remaining == set(job_ids) - {"expired"}
            assert (
                connection.execute(
                    text("SELECT 1 FROM job_attempt WHERE job_id=:id"),
                    {"id": job_ids["expired"]},
                ).first()
                is None
            )
            assert (
                connection.execute(
                    text("SELECT 1 FROM worker_lease WHERE job_id=:id"),
                    {"id": job_ids["expired"]},
                ).first()
                is None
            )

        with pytest.raises(DBAPIError):
            _as_role(
                "trading_runtime",
                "DELETE FROM job WHERE id=:id",
                {"id": job_ids["pending"]},
            )
        with pytest.raises(DBAPIError):
            _as_role(
                "trading_runtime",
                "DELETE FROM job_attempt WHERE job_id=:id",
                {"id": job_ids["legacy"]},
            )
    finally:
        with engine.begin() as connection:
            for job_id in job_ids.values():
                connection.execute(
                    text("DELETE FROM worker_lease WHERE job_id=:id"), {"id": job_id}
                )
                connection.execute(text("DELETE FROM job_attempt WHERE job_id=:id"), {"id": job_id})
                connection.execute(
                    text("DELETE FROM job_quarantine WHERE job_id=:id"), {"id": job_id}
                )
                connection.execute(
                    text("DELETE FROM emergency_action WHERE job_id=:id"), {"id": job_id}
                )
                connection.execute(
                    text("DELETE FROM heavy_compute_lease WHERE job_id=:id"), {"id": job_id}
                )
                connection.execute(text("DELETE FROM job WHERE id=:id"), {"id": job_id})
            connection.execute(text("DELETE FROM worker WHERE id=:id"), {"id": worker_id})
        engine.dispose()
