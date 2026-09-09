"""Deliver committed financial follow-ups after crashes without replaying fills."""

import datetime as dt
from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from src.data.database import financial_application, financial_outbox
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.services.financial_application import (
    FinancialApplication,
    FinancialScope,
    financial_followup_job_name,
)
from src.services.scheduler import DatabaseJobQueue


class FinancialFollowupWorker:
    def __init__(self, queue: DatabaseJobQueue, worker_id: str, on_fill: Callable[..., Any] | None):
        self.queue, self.worker_id, self.on_fill = queue, worker_id, on_fill
        self.job_name = financial_followup_job_name(queue.claim_scope)

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id, now=now, lease_seconds=60, names=(self.job_name,)
        )
        if claimed is None:
            return {"reason_code": "financial_followup_queue_empty"}
        try:
            result = self._deliver(str(claimed.payload["application_id"]), now)
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=str(exc),
                retry_at=(dt.datetime.fromisoformat(now) + dt.timedelta(seconds=5)).isoformat(),
            )
            return {"reason_code": "financial_followup_failed", "error": str(exc)}
        self.queue.complete(claimed, completed_at=now)
        return result

    def _deliver(self, identity: str, now: str) -> dict[str, Any]:
        with self.queue.engine.connect() as connection:
            row = (
                connection.execute(
                    select(financial_application.c.payload, financial_outbox.c.processed_at)
                    .join(financial_outbox, financial_outbox.c.id == financial_application.c.id)
                    .where(financial_application.c.id == identity)
                )
                .mappings()
                .one()
            )
        if row["processed_at"] is not None:
            return {"reason_code": "financial_followup_already_delivered"}
        payload = row["payload"]
        scope = FinancialScope(**payload["scope"])
        order = OrderManager(SqlOrderStore(self.queue.engine)).get(payload["order_id"])
        position = PositionManager(SqlPositionStore(self.queue.engine)).get(
            order.portfolio_id, order.instrument_id
        )
        if self.on_fill is None and ":futures:" in order.instrument_id and position.quantity:
            raise RuntimeError("native protection handler is unavailable for committed exposure")
        if self.on_fill is not None:
            self.on_fill(scope.product_id, order, position.quantity, now)
        FinancialApplication(self.queue.engine, scope).acknowledge(identity, at=now)
        return {"reason_code": "financial_followup_delivered", "application_id": identity}
