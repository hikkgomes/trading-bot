from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, insert, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from src.data.database import PlatformDatabase, schema_migration


def test_platform_migrations_record_immutable_revision_hashes(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.migrate()
    database.assert_migrated()
    with database.engine.connect() as connection:
        rows = connection.execute(schema_migration.select()).mappings().all()
    assert all(row["revision_hash"].startswith("sha256:") for row in rows)


def test_all_provider_upgrade_retains_counts_closed_epochs_and_exposure():
    from src.data.database import (
        dataset_bundle,
        experiment,
        holdout_claim,
        job,
        job_attempt,
        research_epoch,
        strategy_definition,
        strategy_version,
    )

    url = os.environ.get("TRADING_PLATFORM_DATABASE_URL", "")
    if not url.startswith("postgresql"):
        pytest.skip("requires isolated PostgreSQL migration rehearsal")
    base = make_url(url)
    name = "epoch_upgrade_" + uuid.uuid4().hex[:16]
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    database = PlatformDatabase(base.set(database=name).render_as_string(hide_password=False))
    with admin.connect() as connection:
        connection.execute(text(f"CREATE DATABASE {name}"))
    try:
        database.migrate(target="platform_v2_0035")
        at = "2026-09-07T00:00:00+00:00"
        with database.engine.begin() as connection:
            connection.execute(
                insert(job).values(
                    id="attempt-outcome-backfill-job",
                    name="reporting",
                    state="completed",
                    priority=0,
                    available_at=at,
                    attempts=1,
                    max_attempts=3,
                    terminal_reason=None,
                    producer_identity="migration-fixture",
                    content_hash="sha256:" + "8" * 64,
                    payload={},
                )
            )
            connection.execute(
                insert(job_attempt),
                [
                    {
                        "id": "attempt-outcome-backfill-long-label",
                        "job_id": "attempt-outcome-backfill-job",
                        "worker_id": "migration-fixture",
                        "started_at": at,
                        "completed_at": at,
                        "status": "completed",
                        "error": None,
                        "payload": {"outcome_category": "operator_review_" + "x" * 80},
                    },
                    {
                        "id": "attempt-outcome-backfill-failed-default",
                        "job_id": "attempt-outcome-backfill-job",
                        "worker_id": "migration-fixture",
                        "started_at": at,
                        "completed_at": at,
                        "status": "failed",
                        "error": "legacy failure",
                        "payload": {},
                    },
                    {
                        "id": "attempt-outcome-backfill-empty-override",
                        "job_id": "attempt-outcome-backfill-job",
                        "worker_id": "migration-fixture",
                        "started_at": at,
                        "completed_at": at,
                        "status": "failed",
                        "error": "empty outcome",
                        "payload": {"outcome_category": ""},
                    },
                    {
                        "id": "attempt-outcome-backfill-null-fallback",
                        "job_id": "attempt-outcome-backfill-job",
                        "worker_id": "migration-fixture",
                        "started_at": at,
                        "completed_at": at,
                        "status": "expired",
                        "error": "legacy expiry",
                        "payload": {"outcome_category": None},
                    },
                ],
            )
            for identity, providers in {
                "closed": ["bounded_hypothesis_generator", "agent", "catalogue"],
                "missing": ["agent"],
                "retained": ["catalogue"],
            }.items():
                connection.execute(
                    insert(dataset_bundle).values(
                        id=identity,
                        product_id="active_income",
                        created_at=at,
                        content_hash=identity,
                        payload={},
                    )
                )
                connection.execute(
                    insert(strategy_definition).values(
                        id=identity,
                        identity=identity,
                        product_id="active_income",
                        source_type="fixture",
                        source_hash=identity,
                        definition={},
                    )
                )
                connection.execute(
                    insert(strategy_version).values(
                        id=identity,
                        definition_id=identity,
                        version="fixture",
                        created_at=at,
                        payload={},
                    )
                )
                for provider in providers:
                    connection.execute(
                        insert(experiment).values(
                            id=identity + ":" + provider,
                            strategy_version_id=identity,
                            provider=provider,
                            state="infrastructure_exhausted",
                            submitted_at=at,
                            dataset_snapshot_hashes=[],
                            dataset_bundle_id=identity,
                            metadata={"protected_exposure": "retained"},
                        )
                    )
            connection.execute(
                insert(research_epoch),
                [
                    {
                        "id": "closed",
                        "product_id": "active_income",
                        "dataset_bundle_id": "closed",
                        "state": "closed_for_selection",
                        "round_cursor": 4,
                        "budget": 1,
                        "consumed": 1,
                        "created_at": at,
                        "payload": {"last_completion": "keep"},
                    },
                    {
                        "id": "retained",
                        "product_id": "active_income",
                        "dataset_bundle_id": "retained",
                        "state": "waiting_for_feedback",
                        "round_cursor": 9,
                        "budget": 12,
                        "consumed": 10,
                        "created_at": at,
                        "payload": {"last_completion": "keep"},
                    },
                ],
            )
            connection.execute(
                insert(holdout_claim).values(
                    id="exposed", created_at=at, payload={"consumed": True, "scope": "fixture"}
                )
            )
            before_trials = connection.execute(select(experiment).order_by(experiment.c.id)).all()
            before_exposure = connection.execute(select(holdout_claim)).all()
        database.migrate()
        database.assert_migrated()
        with database.engine.connect() as connection:
            backfilled_outcomes = dict(
                connection.execute(
                    select(job_attempt.c.id, job_attempt.c.outcome_category).where(
                        job_attempt.c.id.like("attempt-outcome-backfill-%")
                    )
                ).all()
            )
            assert backfilled_outcomes == {
                "attempt-outcome-backfill-long-label": "operator_review_" + "x" * 80,
                "attempt-outcome-backfill-failed-default": "blocked_infrastructure",
                "attempt-outcome-backfill-empty-override": "",
                "attempt-outcome-backfill-null-fallback": "blocked_infrastructure",
            }
            queue_options = connection.execute(
                text(
                    "SELECT relname, reloptions FROM pg_class "
                    "WHERE oid IN ('public.job'::regclass, 'public.job_attempt'::regclass)"
                )
            ).all()
            assert {row.relname for row in queue_options} == {"job", "job_attempt"}
            for row in queue_options:
                assert set(row.reloptions) >= {
                    "autovacuum_vacuum_scale_factor=0.02",
                    "autovacuum_analyze_scale_factor=0.02",
                }
            epochs = {
                row["dataset_bundle_id"]: row
                for row in connection.execute(select(research_epoch)).mappings()
            }
            assert (
                epochs["closed"]["consumed"],
                epochs["closed"]["budget"],
                epochs["closed"]["round_cursor"],
                epochs["closed"]["state"],
            ) == (3, 3, 4, "closed_for_selection")
            assert epochs["closed"]["payload"]["last_completion"] == "keep"
            assert epochs["missing"]["consumed"] == 1
            assert (
                epochs["retained"]["consumed"],
                epochs["retained"]["budget"],
                epochs["retained"]["round_cursor"],
            ) == (10, 12, 9)
            assert (
                connection.execute(select(experiment).order_by(experiment.c.id)).all()
                == before_trials
            )
            assert connection.execute(select(holdout_claim)).all() == before_exposure
        for change in ({"consumed": 2}, {"budget": 4}, {"state": "active"}):
            with pytest.raises(DBAPIError, match="cannot reopen"):
                with database.engine.begin() as connection:
                    connection.execute(
                        update(research_epoch)
                        .where(research_epoch.c.id == "closed")
                        .values(**change)
                    )
    finally:
        database.dispose()
        with admin.connect() as connection:
            connection.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
        admin.dispose()
