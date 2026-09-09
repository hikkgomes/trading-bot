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
def test_evaluator_completes_research_without_runtime_membership(monkeypatch):
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
        evaluator = create_engine(base.set(database=name, username=role, password=password))
        with evaluator.connect() as connection:
            check_connection_role(connection, "trading_evaluator")
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
                        submitted_at, dataset_snapshot_hashes, dataset_bundle_id, metadata
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
            assert report["research"]["funnel"]["candidates_generated"] == 1
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
