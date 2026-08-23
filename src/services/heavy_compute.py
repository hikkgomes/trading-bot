"""Database-backed singleton lease for bounded heavy computation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from src.data.database import heavy_compute_lease
from src.domain._codec import non_empty, timestamp


@dataclass(frozen=True)
class HeavyComputeLease:
    slot_id: str
    owner: str
    job_id: str
    expires_at: str


class HeavyComputeLeaseStore:
    SLOT_ID = "heavy-compute:0"

    def __init__(self, engine: Engine, *, slot_id: str = SLOT_ID):
        self.engine = engine
        self.slot_id = non_empty(slot_id, field="slot_id")

    def acquire(
        self, *, owner: str, job_id: str, now: str, lease_seconds: int
    ) -> HeavyComputeLease | None:
        if lease_seconds <= 0:
            raise ValueError("heavy-compute lease must be positive")
        now = timestamp(now, field="now")
        expires = _plus_seconds(now, lease_seconds)
        with self.engine.begin() as connection:
            statement = select(heavy_compute_lease).where(
                heavy_compute_lease.c.slot_id == self.slot_id
            )
            if connection.dialect.name == "postgresql":
                statement = statement.with_for_update()
            current = connection.execute(statement).mappings().first()
            if (
                current is not None
                and current["status"] == "active"
                and str(current["expires_at"]) > now
            ):
                return None
            values = {
                "slot_id": self.slot_id,
                "owner": non_empty(owner, field="owner"),
                "job_id": non_empty(job_id, field="job_id"),
                "acquired_at": now,
                "expires_at": expires,
                "status": "active",
            }
            if current is None:
                connection.execute(insert(heavy_compute_lease).values(**values))
            else:
                connection.execute(
                    update(heavy_compute_lease)
                    .where(heavy_compute_lease.c.slot_id == self.slot_id)
                    .values(**values)
                )
        return HeavyComputeLease(self.slot_id, values["owner"], values["job_id"], expires)

    def renew(self, lease: HeavyComputeLease, *, now: str, lease_seconds: int) -> HeavyComputeLease:
        now = timestamp(now, field="now")
        expires = _plus_seconds(now, lease_seconds)
        with self.engine.begin() as connection:
            result = connection.execute(
                update(heavy_compute_lease)
                .where(
                    heavy_compute_lease.c.slot_id == lease.slot_id,
                    heavy_compute_lease.c.owner == lease.owner,
                    heavy_compute_lease.c.job_id == lease.job_id,
                    heavy_compute_lease.c.status == "active",
                )
                .values(expires_at=expires)
            )
            if result.rowcount != 1:
                raise RuntimeError("heavy-compute lease is no longer owned")
        return HeavyComputeLease(lease.slot_id, lease.owner, lease.job_id, expires)

    def release(self, lease: HeavyComputeLease) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(heavy_compute_lease)
                .where(
                    heavy_compute_lease.c.slot_id == lease.slot_id,
                    heavy_compute_lease.c.owner == lease.owner,
                    heavy_compute_lease.c.job_id == lease.job_id,
                )
                .values(status="released")
            )


def _plus_seconds(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
