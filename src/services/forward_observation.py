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
from src.services.forward_metrics import ForwardEvidenceCollector
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
        self.metrics = ForwardEvidenceCollector(engine)

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
            if not set(payload).issubset(required | {"waiting_reason"}) or not required.issubset(
                payload
            ):
                raise ValueError("forward observation command has an invalid field set")
            evaluation_time = timestamp(str(payload["evaluation_time"]), field="evaluation_time")
            assignment = self.assignments.by_id(str(payload["assignment_id"]))
            if payload.get("waiting_reason") is not None:
                if (
                    assignment is None
                    or assignment.get("active") is not True
                    or assignment.get("execution_mode") != "paper"
                    or assignment.get("strategy_version_id") != payload["strategy_version_id"]
                    or assignment.get("product_id") != payload["product_id"]
                    or assignment.get("artefact_hash") != payload["artefact_hash"]
                    or assignment.get("instrument_id") is not None
                ):
                    raise ValueError("forward waiting assignment binding is invalid")
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": "forward_waiting_for_universe_data",
                    "job_id": claimed.job_id,
                    "waiting_reason": str(payload["waiting_reason"]),
                }
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
            previous_observed_at = self.metrics.latest_observed_at(
                strategy_version_id=str(payload["strategy_version_id"]),
                product_id=str(payload["product_id"]),
                instrument_id=str(payload["instrument_id"]),
                artefact_hash=str(payload["artefact_hash"]),
            )
            forecast_fact = {**forecast_payload, "forecast_id": forecast_id}
            target_fact = (
                {**target[2], "target_position_id": target[0]} if target is not None else None
            )
            metrics = self.metrics.collect(
                assignment=assignment,
                product_id=str(payload["product_id"]),
                instrument_id=str(payload["instrument_id"]),
                artefact_created_at=artefact_created_at,
                evaluation_time=evaluation_time,
                forecast=forecast_fact,
                target=target_fact,
                previous_observed_at=previous_observed_at,
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
                    "facts": metrics.to_payload(forecast=forecast_fact, target=target_fact),
                    "reason_code": "forward_evidence_observed",
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


class DatabaseForwardSummaryWorker:
    """Aggregate immutable forward facts into one promotion input."""

    def __init__(
        self,
        *,
        engine,
        queue: DatabaseJobQueue,
        worker_id: str,
        minimum_days_by_product: Mapping[str, int] | None = None,
        minimum_decisions: int = 1,
        policies_by_product: Mapping[str, Mapping[str, Any]] | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.engine = engine
        self.queue = queue
        self.worker_id = worker_id
        self.minimum_days_by_product = {
            str(key): int(value) for key, value in (minimum_days_by_product or {}).items()
        }
        self.minimum_decisions = minimum_decisions
        self.policies_by_product = {
            str(key): dict(value) for key, value in (policies_by_product or {}).items()
        }
        self.lease_seconds = lease_seconds
        self.evidence = SqlForwardEvidenceRepository(engine)

    def run_once(self, *, now: str) -> dict[str, Any]:
        now = timestamp(now, field="now")
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("forward_paper_summary",),
        )
        if claimed is None:
            return {"reason_code": "forward_summary_queue_empty"}
        try:
            required = {
                "strategy_version_id",
                "product_id",
                "artefact_hash",
                "evaluation_time",
            }
            if set(claimed.payload) != required:
                raise ValueError("forward summary command has an invalid field set")
            product_id = str(claimed.payload["product_id"])
            evaluated_at = timestamp(
                str(claimed.payload["evaluation_time"]), field="evaluation_time"
            )
            summary_id, summary = self.evidence.build_summary(
                strategy_version_id=str(claimed.payload["strategy_version_id"]),
                product_id=product_id,
                artefact_hash=str(claimed.payload["artefact_hash"]),
                observed_at=evaluated_at,
            )
            policy = self.policies_by_product.get(product_id, {})
            decision_id, accepted, reason_code = self.evidence.decide_summary(
                summary_id,
                decided_at=evaluated_at,
                minimum_days=int(
                    policy.get(
                        "required_forward_evidence_days",
                        self.minimum_days_by_product.get(product_id, 0),
                    )
                ),
                minimum_decisions=int(
                    policy.get("minimum_forward_independent_decisions", self.minimum_decisions)
                ),
                minimum_net_pnl=float(policy.get("minimum_forward_net_pnl", 0.0)),
                maximum_drawdown=float(policy.get("maximum_drawdown", 1.0)),
                maximum_data_gaps=int(policy.get("maximum_forward_data_gaps", 0)),
                minimum_effective_trades=int(policy.get("minimum_forward_effective_trades", 0)),
                minimum_fill_rate=float(policy.get("minimum_forward_fill_rate", 0.0)),
                maximum_slippage=float(policy.get("maximum_forward_slippage", 1.0)),
                minimum_data_uptime=float(policy.get("minimum_forward_data_uptime", 0.0)),
                maximum_rejected_orders=int(policy.get("maximum_forward_rejected_orders", 0)),
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "forward_summary_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "forward_summary_decided",
            "job_id": claimed.job_id,
            "summary_id": summary_id,
            "decision_id": decision_id,
            "accepted": accepted,
            "reason_code_detail": reason_code,
            "independent_decisions": summary.independent_decisions,
        }
