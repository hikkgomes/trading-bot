"""Artefact-bound forward-paper observations from canonical platform state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import select

from src.data.database import alpha_forecast, position, target_position
from src.domain._codec import timestamp
from src.research.canonical import (
    SqlActiveStrategyAssignmentRepository,
    SqlForwardEvidenceRepository,
    SqlStrategyArtefactRepository,
)
from src.services.scheduler import DatabaseJobQueue


class DatabaseForwardObservationWorker:
    """Record paper evidence only after the exact frozen artefact exists."""

    def __init__(
        self,
        *,
        engine,
        queue: DatabaseJobQueue,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> None:
        self.engine = engine
        self.queue = queue
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.assignments = SqlActiveStrategyAssignmentRepository(engine)
        self.artefacts = SqlStrategyArtefactRepository(engine)
        self.evidence = SqlForwardEvidenceRepository(engine)

    def run_once(self, *, now: str) -> dict[str, Any]:
        now = timestamp(now, field="now")
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("forward_paper_observation",),
        )
        if claimed is None:
            return {"reason_code": "forward_observation_queue_empty"}
        try:
            payload = claimed.payload
            required = {
                "assignment_id",
                "strategy_version_id",
                "product_id",
                "instrument_id",
                "artefact_hash",
                "evaluation_time",
            }
            if set(payload) != required:
                raise ValueError("forward observation command has an invalid field set")
            evaluation_time = timestamp(str(payload["evaluation_time"]), field="evaluation_time")
            assignment = self.assignments.by_id(str(payload["assignment_id"]))
            if (
                assignment is None
                or assignment.get("active") is not True
                or assignment.get("execution_mode") != "paper"
                or assignment.get("strategy_version_id") != payload["strategy_version_id"]
                or assignment.get("product_id") != payload["product_id"]
                or assignment.get("artefact_hash") != payload["artefact_hash"]
                or assignment.get("instrument_id") != payload["instrument_id"]
            ):
                raise ValueError("forward observation assignment binding is invalid")
            artefact = self.artefacts.get(str(payload["artefact_hash"]))
            artefact_created_at = timestamp(
                str(artefact["created_at"]), field="artefact.created_at"
            )
            forecast = self._latest_payload(
                alpha_forecast,
                product_id=str(payload["product_id"]),
                strategy_version_id=str(payload["strategy_version_id"]),
                instrument_id=str(payload["instrument_id"]),
                after=artefact_created_at,
                at=evaluation_time,
            )
            if forecast is None:
                raise LookupError("no post-artefact paper forecast is available")
            forecast_id, observed_at, forecast_payload = forecast
            target = self._latest_payload(
                target_position,
                product_id=str(payload["product_id"]),
                after=artefact_created_at,
                at=evaluation_time,
            )
            current_position = self._latest_position(
                portfolio_id=str(assignment["portfolio_id"]),
                instrument_id=str(payload["instrument_id"]),
                at=evaluation_time,
            )
            observation_id = self.evidence.append(
                strategy_version_id=str(payload["strategy_version_id"]),
                product_id=str(payload["product_id"]),
                instrument_id=str(payload["instrument_id"]),
                observed_at=observed_at,
                artefact_hash=str(payload["artefact_hash"]),
                evaluation_time=evaluation_time,
                observation={
                    "schema": "platform.forward_paper_observation/v1",
                    "assignment_id": str(payload["assignment_id"]),
                    "forecast_id": forecast_id,
                    "forecast_content": forecast_payload,
                    "target_position_id": target[0] if target is not None else None,
                    "position": current_position,
                    "accepted": False,
                    "reason_code": "collecting_forward_evidence",
                },
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "forward_observation_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "forward_observation_recorded",
            "job_id": claimed.job_id,
            "observation_id": observation_id,
            "observed_at": observed_at,
        }

    def _latest_payload(
        self,
        table,
        *,
        product_id: str,
        after: str,
        at: str,
        strategy_version_id: str | None = None,
        instrument_id: str | None = None,
    ) -> tuple[str, str, dict[str, Any]] | None:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(table.c.id, table.c.created_at, table.c.payload)
                .where(table.c.created_at > after, table.c.created_at <= at)
                .order_by(table.c.created_at.desc(), table.c.id.desc())
            ).mappings()
            for row in rows:
                payload = row["payload"]
                if not isinstance(payload, Mapping) or payload.get("product_id") != product_id:
                    continue
                if (
                    strategy_version_id is not None
                    and payload.get("strategy_version_id") != strategy_version_id
                ):
                    continue
                if instrument_id is not None and payload.get("instrument_id") != instrument_id:
                    continue
                return str(row["id"]), str(row["created_at"]), dict(payload)
        return None

    def _latest_position(
        self, *, portfolio_id: str, instrument_id: str, at: str
    ) -> Mapping[str, Any] | None:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(position.c.payload)
                .where(position.c.created_at <= at)
                .order_by(position.c.created_at.desc(), position.c.id.desc())
            ).scalars()
            for payload in rows:
                if (
                    isinstance(payload, Mapping)
                    and payload.get("portfolio_id") == portfolio_id
                    and payload.get("instrument_id") == instrument_id
                ):
                    return dict(payload)
        return None


def _retry_at(value: str, seconds: int) -> str:
    import datetime as dt

    return (
        (dt.datetime.fromisoformat(value) + dt.timedelta(seconds=seconds))
        .replace(microsecond=0)
        .isoformat()
    )
