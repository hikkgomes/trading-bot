"""Small real-PostgreSQL contract smoke used by CI when a service is present."""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.exc import SQLAlchemyError

from src.data.database import PlatformDatabase, strategy_artefact
from src.services.scheduler import DatabaseJobQueue

DATABASE_URL = os.environ.get("TRADING_PLATFORM_DATABASE_URL")


@pytest.mark.skipif(not DATABASE_URL, reason="TRADING_PLATFORM_DATABASE_URL is not configured")
def test_postgresql_migrations_queue_and_append_only_evidence():
    run_id = uuid.uuid4().hex
    worker_id = f"ci-worker-{run_id}"
    job_id = f"ci-job-{run_id}"
    artefact_id = "sha256:" + run_id * 2
    database = PlatformDatabase(str(DATABASE_URL))
    database.migrate()
    database.assert_migrated()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id=worker_id,
        node_id="ci-node",
        role="ci",
        capabilities=("ci",),
        observed_at="2026-08-17T00:00:00+00:00",
    )
    queue.enqueue(
        job_id=job_id,
        name="ci",
        payload={"identity": "ci"},
        available_at="2026-08-17T00:00:00+00:00",
    )
    claimed = queue.claim(
        worker_id=worker_id,
        now="2026-08-17T00:00:01+00:00",
        lease_seconds=30,
        names=("ci",),
    )
    assert claimed is not None
    queue.complete(claimed, completed_at="2026-08-17T00:00:02+00:00")
    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_artefact).values(
                id=artefact_id,
                created_at="2026-08-17T00:00:00+00:00",
                payload={"identity": "ci"},
            )
        )
    with pytest.raises(SQLAlchemyError):
        with database.engine.begin() as connection:
            connection.execute(
                update(strategy_artefact)
                .where(strategy_artefact.c.id == artefact_id)
                .values(payload={"identity": "mutated"})
            )
    with database.engine.connect() as connection:
        assert (
            connection.execute(
                select(strategy_artefact.c.id).where(strategy_artefact.c.id == artefact_id)
            ).scalar_one()
            == artefact_id
        )
    database.dispose()


@pytest.mark.skipif(not DATABASE_URL, reason="TRADING_PLATFORM_DATABASE_URL is not configured")
def test_postgresql_lease_recovery_renewal_and_stale_completion_are_fenced():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from src.data.database import job, job_attempt

    database = PlatformDatabase(str(DATABASE_URL))
    database.assert_migrated()
    queue = DatabaseJobQueue(database.engine)
    run_id = uuid.uuid4().hex
    job_name = f"fence-{run_id}"
    first_worker, second_worker = f"fence-a-{run_id}", f"fence-b-{run_id}"
    initial = "2026-08-17T00:00:00+00:00"
    expired = "2026-08-17T00:00:11+00:00"
    for worker_id in (first_worker, second_worker):
        queue.register_worker(
            worker_id=worker_id,
            node_id="ci-node",
            role="ci",
            capabilities=(job_name,),
            observed_at=initial,
        )
    identity = f"fence-renew-{run_id}"
    queue.enqueue(job_id=identity, name=job_name, payload={}, available_at=initial)
    original = queue.claim(worker_id=first_worker, now=initial, lease_seconds=10, names=(job_name,))
    assert original is not None and original.job_id == identity
    with ThreadPoolExecutor(max_workers=2) as pool:
        with database.engine.begin() as connection:
            connection.execute(select(job.c.id).where(job.c.id == identity).with_for_update()).one()
            renewal = pool.submit(queue.heartbeat, original, now=expired, lease_seconds=30)
            recovery = pool.submit(queue.recover_expired, now=expired)
            recovery.result(timeout=10)
            assert (
                connection.execute(select(job.c.state).where(job.c.id == identity)).scalar_one()
                == "running"
            )
        renewed = renewal.result(timeout=10)
    queue.recover_expired(now="2026-08-17T00:00:12+00:00")
    assert queue.state(identity) == "running"
    # A stalled queue writer must not block report renewal or failure recording.
    with database.engine.begin() as connection, ThreadPoolExecutor(max_workers=1) as pool:
        connection.execute(select(job.c.id).where(job.c.id == identity).with_for_update()).one()
        blocked_renewal = pool.submit(queue.heartbeat, renewed, now=expired, lease_seconds=30)
        with pytest.raises(SQLAlchemyError, match="lock timeout"):
            blocked_renewal.result(timeout=8)
        blocked_failure = pool.submit(
            queue.fail, renewed, completed_at=expired, error="report_failed", retry_at=expired
        )
        with pytest.raises(SQLAlchemyError, match="lock timeout"):
            blocked_failure.result(timeout=8)
    assert queue.state(identity) == "running"
    queue.defer(renewed, available_at="2026-08-17T00:00:13+00:00", reason="resource_wait")
    replacement = queue.claim(
        worker_id=first_worker, now="2026-08-17T00:00:13+00:00", lease_seconds=30, names=(job_name,)
    )
    assert replacement is not None and replacement.attempt == original.attempt
    assert replacement.lease_token != original.lease_token
    with pytest.raises(ValueError, match="no longer owns"):
        queue.complete(original, completed_at="2026-08-17T00:00:14+00:00")
    queue.complete(replacement, completed_at="2026-08-17T00:00:14+00:00")
    with database.engine.connect() as connection:
        statuses = (
            connection.execute(select(job_attempt.c.status).where(job_attempt.c.job_id == identity))
            .scalars()
            .all()
        )
    assert sorted(statuses) == ["completed", "deferred"]
    insertion = {
        "job_id": f"fence-insert-{run_id}",
        "name": f"{job_name}-insert",
        "payload": {},
        "available_at": initial,
    }
    with database.engine.begin() as connection, ThreadPoolExecutor(max_workers=1) as pool:
        assert queue.enqueue_if_absent(**insertion, connection=connection)
        duplicate = pool.submit(queue.enqueue_if_absent, **insertion)
        with pytest.raises(SQLAlchemyError, match="lock timeout"):
            duplicate.result(timeout=8)
    assert not queue.enqueue_if_absent(**insertion)
    for index in range(5):
        identity = f"fence-race-{run_id}-{index}"
        queue.enqueue(job_id=identity, name=job_name, payload={}, available_at=initial)
        original = queue.claim(
            worker_id=first_worker, now=initial, lease_seconds=10, names=(job_name,)
        )
        assert original is not None and original.job_id == identity
        start = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            completion = pool.submit(
                lambda start=start, original=original: (
                    start.wait(),
                    queue.complete(original, completed_at=expired),
                )[1]
            )
            reclamation = pool.submit(
                lambda start=start: (
                    start.wait(),
                    queue.claim(
                        worker_id=second_worker, now=expired, lease_seconds=30, names=(job_name,)
                    ),
                )[1]
            )
            replacement = reclamation.result(timeout=10)
            try:
                completion.result(timeout=10)
                completion_rejected = False
            except ValueError:
                completion_rejected = True
        if replacement is None:
            assert not completion_rejected and queue.state(identity) == "completed"
        else:
            assert completion_rejected and queue.state(identity) == "running"
            assert replacement.attempt == 2
            queue.complete(replacement, completed_at=expired)
    database.dispose()
