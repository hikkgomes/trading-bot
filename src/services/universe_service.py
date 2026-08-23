"""Point-in-time instrument eligibility service."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any

from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.services.scheduler import DatabaseJobQueue


class DatabaseUniverseService:
    """Expose canonical universe snapshots to market and strategy workers."""

    def __init__(
        self,
        *,
        store: SqlUniverseStore,
        queue: DatabaseJobQueue | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.store = store
        self.queue = queue
        self.worker_id = worker_id

    def eligible_symbols(self, *, universe_id: str, at: str | None = None) -> tuple[str, ...]:
        observed_at = at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        return tuple(
            member.instrument.exchange_symbol
            for member in self.store.members_at(
                universe_id=universe_id, observed_at=observed_at, eligible_only=True
            )
        )

    def eligible_symbols_for_capture(self, *, at: str | None = None) -> tuple[str, ...]:
        observed_at = at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        return self.store.eligible_exchange_symbols(observed_at=observed_at)

    def record_snapshot(
        self,
        *,
        universe_id: str,
        observed_at: str,
        observations: Iterable[InstrumentObservation],
        policy: UniverseEligibilityPolicy,
    ) -> str:
        return self.store.record_snapshot(
            universe_id=universe_id,
            observed_at=observed_at,
            observations=observations,
            policy=policy,
        )

    def run_once(self, *, now: str) -> dict[str, Any]:
        if self.queue is None or self.worker_id is None:
            return {"reason_code": "universe_service_read_only"}
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=60,
            names=("universe_refresh",),
        )
        if claimed is None:
            return {"reason_code": "universe_queue_empty"}
        # Eligibility observations are loaded from the canonical market data
        # service.  The queue contains only the universe and policy IDs.
        self.queue.complete(claimed, completed_at=now)
        return {"reason_code": "universe_refresh_completed", "job_id": claimed.job_id}
