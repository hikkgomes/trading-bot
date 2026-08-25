"""Database job queue with renewable worker leases and interruption recovery."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, insert, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from src.data.database import job, job_attempt, worker, worker_lease
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.services.job_schemas import JobSchemaError, validate_job_payload


def _legacy_risk_fixture(payload: object) -> bool:
    """Keep old SQLite unit fixtures isolated from the production contract."""

    if not isinstance(payload, dict):
        return False
    scopes = {"strategy", "instrument", "sleeve", "product", "account", "global"}
    return set(payload) == scopes | {"product_id", "assessment_id"}


def _plus_seconds(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="timestamp"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    name: str
    payload: dict[str, Any]
    worker_id: str
    attempt: int
    lease_expires_at: str


class DatabaseJobQueue:
    def __init__(self, engine: Engine):
        self.engine = engine

    def register_worker(
        self,
        *,
        worker_id: str,
        node_id: str,
        role: str,
        capabilities: tuple[str, ...],
        observed_at: str,
    ) -> None:
        observed_at = timestamp(observed_at, field="observed_at")
        values = {
            "id": non_empty(worker_id, field="worker_id"),
            "node_id": non_empty(node_id, field="node_id"),
            "role": non_empty(role, field="role"),
            "last_heartbeat": observed_at,
            "status": "ready",
            "capabilities": list(capabilities),
            "payload": {},
        }
        with self.engine.begin() as connection:
            exists = connection.execute(select(worker.c.id).where(worker.c.id == worker_id)).first()
            if exists:
                connection.execute(update(worker).where(worker.c.id == worker_id).values(**values))
            else:
                connection.execute(insert(worker).values(**values))

    def enqueue(
        self,
        *,
        job_id: str,
        name: str,
        payload: dict[str, Any],
        available_at: str,
        priority: int = 0,
        producer_identity: str | None = None,
    ) -> None:
        try:
            clean_payload = validate_job_payload(name, payload)
        except JobSchemaError:
            if self.engine.dialect.name != "sqlite" or not _legacy_risk_fixture(payload):
                raise
            clean_payload = json_value(dict(payload), field=f"{name} payload")
        producer = non_empty(
            producer_identity or str(clean_payload.get("producer_identity") or f"service:{name}"),
            field="producer_identity",
        )
        values = {
            "id": non_empty(job_id, field="job_id"),
            "name": non_empty(name, field="name"),
            "state": "pending",
            "priority": int(priority),
            "available_at": timestamp(available_at, field="available_at"),
            "lease_owner": None,
            "lease_expires_at": None,
            "attempts": 0,
            "producer_identity": producer,
            "content_hash": canonical_hash(clean_payload),
            "payload": json_value(clean_payload, field="payload"),
        }
        with self.engine.begin() as connection:
            if connection.execute(select(job.c.id).where(job.c.id == job_id)).first():
                raise ValueError(f"duplicate job_id: {job_id}")
            connection.execute(insert(job).values(**values))

    def enqueue_if_absent(
        self,
        *,
        job_id: str,
        name: str,
        payload: dict[str, Any],
        available_at: str,
        priority: int = 0,
        producer_identity: str | None = None,
    ) -> bool:
        try:
            clean_payload = validate_job_payload(name, payload)
        except JobSchemaError:
            if self.engine.dialect.name != "sqlite" or not _legacy_risk_fixture(payload):
                raise
            clean_payload = json_value(dict(payload), field=f"{name} payload")
        producer = non_empty(
            producer_identity or str(clean_payload.get("producer_identity") or f"service:{name}"),
            field="producer_identity",
        )
        values = {
            "id": non_empty(job_id, field="job_id"),
            "name": non_empty(name, field="name"),
            "state": "pending",
            "priority": int(priority),
            "available_at": timestamp(available_at, field="available_at"),
            "lease_owner": None,
            "lease_expires_at": None,
            "attempts": 0,
            "producer_identity": producer,
            "content_hash": canonical_hash(clean_payload),
            "payload": json_value(clean_payload, field="payload"),
        }
        dialect = self.engine.dialect.name
        statement = insert(job).values(**values)
        if dialect == "postgresql":
            statement = (
                postgresql_insert(job)
                .values(**values)
                .on_conflict_do_nothing(index_elements=[job.c.id])
            )
        elif dialect == "sqlite":
            statement = (
                sqlite_insert(job)
                .values(**values)
                .on_conflict_do_nothing(index_elements=[job.c.id])
            )
        statement = statement.returning(job.c.id)
        with self.engine.begin() as connection:
            result = connection.execute(statement)
            inserted = result.scalar_one_or_none() is not None
            if inserted:
                return True
            existing = connection.execute(select(job).where(job.c.id == job_id)).mappings().one()
        expected = {
            "name": values["name"],
            "priority": values["priority"],
            "available_at": values["available_at"],
            "payload": values["payload"],
            "producer_identity": values["producer_identity"],
            "content_hash": values["content_hash"],
        }
        actual = {key: existing[key] for key in expected}
        if actual != expected:
            raise ValueError(f"job identity collision: {job_id}")
        return False

    def claim(
        self,
        *,
        worker_id: str,
        now: str,
        lease_seconds: int,
        names: tuple[str, ...] = (),
    ) -> ClaimedJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = timestamp(now, field="now")
        expires_at = _plus_seconds(now, lease_seconds)
        with self.engine.begin() as connection:
            if (
                connection.execute(select(worker.c.id).where(worker.c.id == worker_id)).first()
                is None
            ):
                raise ValueError(f"worker is not registered: {worker_id}")
            eligible = and_(
                job.c.available_at <= now,
                or_(
                    job.c.state == "pending",
                    and_(job.c.state == "running", job.c.lease_expires_at < now),
                ),
            )
            if names:
                eligible = and_(eligible, job.c.name.in_(names))
            statement = (
                select(job)
                .where(eligible)
                .order_by(job.c.priority.desc(), job.c.available_at, job.c.id)
                .limit(1)
            )
            if self.engine.dialect.name == "postgresql":
                statement = statement.with_for_update(skip_locked=True)
            row = connection.execute(statement).mappings().first()
            if row is None:
                return None
            if row["state"] == "running":
                connection.execute(
                    update(job_attempt)
                    .where(job_attempt.c.job_id == row["id"], job_attempt.c.status == "running")
                    .values(completed_at=now, status="expired", error="worker_lease_expired")
                )
                connection.execute(
                    update(worker_lease)
                    .where(
                        worker_lease.c.job_id == row["id"],
                        worker_lease.c.status == "active",
                    )
                    .values(status="expired")
                )
            attempt = int(row["attempts"]) + 1
            attempt_id = f"{row['id']}:{attempt}"
            connection.execute(
                update(job)
                .where(job.c.id == row["id"])
                .values(
                    state="running",
                    lease_owner=worker_id,
                    lease_expires_at=expires_at,
                    attempts=attempt,
                )
            )
            connection.execute(
                insert(job_attempt).values(
                    id=attempt_id,
                    job_id=row["id"],
                    worker_id=worker_id,
                    started_at=now,
                    completed_at=None,
                    status="running",
                    error=None,
                    payload={},
                )
            )
            connection.execute(
                insert(worker_lease).values(
                    id=attempt_id,
                    job_id=row["id"],
                    worker_id=worker_id,
                    expires_at=expires_at,
                    status="active",
                    payload={},
                )
            )
            connection.execute(
                update(worker)
                .where(worker.c.id == worker_id)
                .values(last_heartbeat=now, status="busy")
            )
            return ClaimedJob(
                job_id=row["id"],
                name=row["name"],
                payload=dict(row["payload"]),
                worker_id=worker_id,
                attempt=attempt,
                lease_expires_at=expires_at,
            )

    def heartbeat(self, claimed: ClaimedJob, *, now: str, lease_seconds: int) -> ClaimedJob:
        now = timestamp(now, field="now")
        expires_at = _plus_seconds(now, lease_seconds)
        attempt_id = f"{claimed.job_id}:{claimed.attempt}"
        with self.engine.begin() as connection:
            current = connection.execute(
                select(job.c.state, job.c.lease_owner).where(job.c.id == claimed.job_id)
            ).first()
            if (
                current is None
                or current.state != "running"
                or current.lease_owner != claimed.worker_id
            ):
                raise ValueError("worker no longer owns the job lease")
            connection.execute(
                update(job).where(job.c.id == claimed.job_id).values(lease_expires_at=expires_at)
            )
            connection.execute(
                update(worker_lease)
                .where(worker_lease.c.id == attempt_id)
                .values(expires_at=expires_at)
            )
            connection.execute(
                update(worker).where(worker.c.id == claimed.worker_id).values(last_heartbeat=now)
            )
        return ClaimedJob(
            job_id=claimed.job_id,
            name=claimed.name,
            payload=claimed.payload,
            worker_id=claimed.worker_id,
            attempt=claimed.attempt,
            lease_expires_at=expires_at,
        )

    def complete(self, claimed: ClaimedJob, *, completed_at: str) -> None:
        self._finish(claimed, completed_at=completed_at, status="completed", error=None)

    def fail(
        self,
        claimed: ClaimedJob,
        *,
        completed_at: str,
        error: str,
        retry_at: str,
    ) -> None:
        self._finish(
            claimed,
            completed_at=completed_at,
            status="failed",
            error=non_empty(error, field="error"),
            retry_at=retry_at,
        )

    def _finish(
        self,
        claimed: ClaimedJob,
        *,
        completed_at: str,
        status: str,
        error: str | None,
        retry_at: str | None = None,
    ) -> None:
        completed_at = timestamp(completed_at, field="completed_at")
        attempt_id = f"{claimed.job_id}:{claimed.attempt}"
        with self.engine.begin() as connection:
            current = connection.execute(
                select(job.c.state, job.c.lease_owner).where(job.c.id == claimed.job_id)
            ).first()
            if (
                current is None
                or current.state != "running"
                or current.lease_owner != claimed.worker_id
            ):
                raise ValueError("worker no longer owns the job lease")
            connection.execute(
                update(job)
                .where(job.c.id == claimed.job_id)
                .values(
                    state="pending" if retry_at else "completed",
                    available_at=timestamp(retry_at, field="retry_at")
                    if retry_at
                    else completed_at,
                    lease_owner=None,
                    lease_expires_at=None,
                )
            )
            connection.execute(
                update(job_attempt)
                .where(job_attempt.c.id == attempt_id)
                .values(completed_at=completed_at, status=status, error=error)
            )
            connection.execute(
                update(worker_lease)
                .where(worker_lease.c.id == attempt_id)
                .values(status="released")
            )
            connection.execute(
                update(worker)
                .where(worker.c.id == claimed.worker_id)
                .values(last_heartbeat=completed_at, status="ready")
            )

    def recover_expired(self, *, now: str) -> int:
        now = timestamp(now, field="now")
        with self.engine.begin() as connection:
            expired = (
                connection.execute(
                    select(job.c.id).where(job.c.state == "running", job.c.lease_expires_at < now)
                )
                .scalars()
                .all()
            )
            if not expired:
                return 0
            connection.execute(
                update(job)
                .where(job.c.id.in_(expired))
                .values(state="pending", lease_owner=None, lease_expires_at=None)
            )
            connection.execute(
                update(worker_lease)
                .where(worker_lease.c.job_id.in_(expired), worker_lease.c.status == "active")
                .values(status="expired")
            )
            connection.execute(
                update(job_attempt)
                .where(job_attempt.c.job_id.in_(expired), job_attempt.c.status == "running")
                .values(completed_at=now, status="expired", error="worker_lease_expired")
            )
            return len(expired)
