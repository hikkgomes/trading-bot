"""Deterministic closed-event strategy evaluation service."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import select

from src.data.database import strategy_artefact
from src.data.feature_store import SqlFeatureStore
from src.domain._codec import canonical_hash, timestamp
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.research.canonical import SqlActiveStrategyAssignmentRepository
from src.services.artefact_dispatcher import ArtefactDispatcher
from src.services.job_schemas import validate_job_payload
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.scheduler import DatabaseJobQueue


class DatabaseStrategyEvaluator:
    """Load immutable assignment and as-of features, then persist one forecast."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        feature_store: SqlFeatureStore,
        portfolio: SqlPortfolioRepository,
        assignments: SqlActiveStrategyAssignmentRepository,
        engine_version: str = "strategy-evaluator/v1",
        forecast_fn: Callable[[Mapping[str, float], Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        artefact_dispatcher: ArtefactDispatcher | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.feature_store = feature_store
        self.portfolio = portfolio
        self.assignments = assignments
        self.engine_version = engine_version
        self.forecast_fn = (
            forecast_fn or (artefact_dispatcher or ArtefactDispatcher.default()).evaluate
        )
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("strategy_evaluation",),
        )
        if claimed is None:
            return {"reason_code": "strategy_evaluation_queue_empty"}
        try:
            payload = validate_job_payload("strategy_evaluation", claimed.payload)
            product_id = payload["product_id"]
            assignment = self.assignments.by_id(str(payload["assignment_id"]))
            if (
                assignment is None
                or not assignment["active"]
                or assignment["product_id"] != product_id
            ):
                raise ValueError(f"no active strategy assignment for {product_id}")
            if assignment["id"] != payload["assignment_id"]:
                raise ValueError("strategy assignment changed after feature publication")
            assignment_payload = assignment.get("payload")
            if (
                isinstance(assignment_payload, Mapping)
                and assignment_payload.get("diagnostic") is True
            ):
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": "diagnostic_strategy_evaluation_skipped",
                    "job_id": claimed.job_id,
                }
            at = timestamp(payload["evaluated_at"], field="evaluated_at")
            values = self.feature_store.by_ids(payload["feature_ids"])
            if any(
                value.instrument_id != payload["instrument_id"]
                or value.feature_set_version != payload["feature_set_version"]
                or value.availability_time > at
                for value in values
            ):
                raise ValueError("feature batch is not available for the requested event")
            feature_ids = tuple(value.feature_id for value in values)
            expected_features = tuple(payload["feature_ids"])
            if feature_ids != expected_features:
                raise ValueError("feature batch is not the exact immutable input requested")
            features = {value.feature_name: value.value for value in values}
            artefact_hash = str(assignment["artefact_hash"])
            with self.portfolio.engine.connect() as connection:
                artefact = connection.execute(
                    select(strategy_artefact.c.payload).where(
                        strategy_artefact.c.id == artefact_hash
                    )
                ).scalar_one_or_none()
            if not isinstance(artefact, Mapping):
                raise ValueError("active strategy artefact is missing")
            forecast_values = dict(self.forecast_fn(features, artefact))
            valid_from = at
            valid_until = (
                (
                    dt.datetime.fromisoformat(at)
                    + dt.timedelta(seconds=int(payload["horizon_seconds"]))
                )
                .replace(microsecond=0)
                .isoformat()
            )
            forecast = AlphaForecast(
                strategy_version_id=str(assignment["strategy_version_id"]),
                product_id=product_id,
                instrument_id=payload["instrument_id"],
                direction=ForecastDirection(str(forecast_values["direction"])),
                score=float(forecast_values["score"]),
                expected_return=float(forecast_values["expected_return"]),
                confidence=float(forecast_values["confidence"]),
                horizon_seconds=int(payload["horizon_seconds"]),
                valid_from=valid_from,
                valid_until=valid_until,
                target_volatility=float(forecast_values["target_volatility"]),
                maximum_position=float(forecast_values["maximum_position"]),
                metadata={
                    "market_event_id": payload["event_id"],
                    "feature_ids": list(feature_ids),
                    "artefact_hash": artefact_hash,
                    "engine_version": self.engine_version,
                    "assignment_id": assignment["id"],
                    "execution_receipt": forecast_values.get("execution_receipt", {}),
                },
            )
            forecast_id = self.portfolio.save_forecast(forecast)
            portfolio_job_id = canonical_hash(
                {
                    "forecast_id": forecast_id,
                    "event_id": payload["event_id"],
                    "product_id": product_id,
                }
            )
            target_payload = {
                "event_id": payload["event_id"],
                "product_id": product_id,
                "forecast_id": forecast_id,
                "evaluated_at": at,
                "producer_identity": self.worker_id,
                **(
                    {"market_data_snapshot_id": payload["market_data_snapshot_id"]}
                    if "market_data_snapshot_id" in payload
                    else {}
                ),
            }
            target_payload["content_hash"] = canonical_hash(target_payload)
            self.queue.enqueue_if_absent(
                job_id=f"portfolio-target:{portfolio_job_id.removeprefix('sha256:')}",
                name="portfolio_target_build",
                payload=target_payload,
                available_at=at,
                priority=15,
                producer_identity=self.worker_id,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=now,
            )
            return {
                "reason_code": "strategy_evaluation_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "alpha_forecast_persisted",
            "job_id": claimed.job_id,
            "forecast_id": forecast_id,
            "portfolio_job_id": f"portfolio-target:{portfolio_job_id.removeprefix('sha256:')}",
        }


def _strict_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = frozenset(
        {
            "event_id",
            "product_id",
            "instrument_id",
            "assignment_id",
            "feature_ids",
            "feature_set_version",
            "evaluated_at",
            "horizon_seconds",
            "producer_identity",
        }
    )
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("strategy evaluation payload contains unknown fields")
    required = allowed - {"producer_identity"}
    missing = sorted(field for field in required if field not in payload)
    if missing:
        raise ValueError("strategy evaluation payload is missing: " + ", ".join(missing))
    if not isinstance(payload["feature_ids"], list | tuple) or not payload["feature_ids"]:
        raise ValueError("strategy evaluation requires feature IDs")
    return dict(payload)
