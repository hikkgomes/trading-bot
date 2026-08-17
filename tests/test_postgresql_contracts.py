"""Small real-PostgreSQL contract smoke used by CI when a service is present."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.exc import SQLAlchemyError

from src.data.database import PlatformDatabase, strategy_artefact
from src.services.scheduler import DatabaseJobQueue

DATABASE_URL = os.environ.get("TRADING_PLATFORM_DATABASE_URL")


@pytest.mark.skipif(not DATABASE_URL, reason="TRADING_PLATFORM_DATABASE_URL is not configured")
def test_postgresql_migrations_queue_and_append_only_evidence():
    database = PlatformDatabase(str(DATABASE_URL))
    database.migrate()
    database.assert_migrated()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="ci-worker",
        node_id="ci-node",
        role="ci",
        capabilities=("ci",),
        observed_at="2026-08-17T00:00:00+00:00",
    )
    queue.enqueue(
        job_id="ci-job",
        name="ci",
        payload={"identity": "ci"},
        available_at="2026-08-17T00:00:00+00:00",
    )
    claimed = queue.claim(
        worker_id="ci-worker",
        now="2026-08-17T00:00:01+00:00",
        lease_seconds=30,
        names=("ci",),
    )
    assert claimed is not None
    queue.complete(claimed, completed_at="2026-08-17T00:00:02+00:00")
    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_artefact).values(
                id="sha256:" + "a" * 64,
                created_at="2026-08-17T00:00:00+00:00",
                payload={"identity": "ci"},
            )
        )
    with pytest.raises(SQLAlchemyError):
        with database.engine.begin() as connection:
            connection.execute(
                update(strategy_artefact)
                .where(strategy_artefact.c.id == "sha256:" + "a" * 64)
                .values(payload={"identity": "mutated"})
            )
    with database.engine.connect() as connection:
        assert (
            connection.execute(select(strategy_artefact.c.id)).scalar_one() == "sha256:" + "a" * 64
        )
    database.dispose()
