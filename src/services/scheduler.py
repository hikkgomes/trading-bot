"""Database job queue with renewable worker leases and interruption recovery."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, delete, func, insert, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from src.data.database import (
    dataset_snapshot,
    evidence_invalidation,
    experiment,
    forward_paper_observation,
    job,
    job_attempt,
    job_quarantine,
    paper_eligibility_event,
    platform_schedule,
    strategy_artefact,
    universe_member,
    universe_snapshot,
    worker,
    worker_lease,
)
from src.data.transactions import connection_scope
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.research.accounting import FUTURES_ACCOUNTING_RISK_CONTRACT
from src.research.canonical import SqlActiveStrategyAssignmentRepository
from src.research.data_plan import product_data_plan
from src.research.datasets import _load_snapshot_manifests
from src.research.outcomes import OutcomeCategory, TrialDisposition
from src.research.rounds import ResearchRoundCoordinator
from src.research.scheduling import research_inputs
from src.services.alerting import AlertSeverity, SqlAlertService, configured_alert_service
from src.services.job_schemas import JobSchemaError, validate_job_payload


def _legacy_risk_fixture(payload: object) -> bool:
    """Keep old SQLite unit fixtures isolated from the production contract."""

    if not isinstance(payload, dict):
        return False
    scopes = {"strategy", "instrument", "sleeve", "product", "account", "global"}
    return set(payload) == scopes | {"product_id", "assessment_id"}


def _diagnostic_candidate(
    candidate_metadata: Mapping[str, Any], definition_metadata: Mapping[str, Any]
) -> bool:
    return (
        candidate_metadata.get("diagnostic") is True
        or definition_metadata.get("diagnostic") is True
    )


def _non_promotable_artefact(metadata: Mapping[str, Any]) -> bool:
    return metadata.get("diagnostic") is True or metadata.get("promotable") is False


def _plus_seconds(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="timestamp"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()


def immutable_work_id(
    name: str, payload: Mapping[str, Any], *, volatile: set[str] | None = None
) -> str:
    """Return the stable identity for one immutable unit of work.

    Scheduling time is deliberately excluded by callers for research work.
    Observation work keeps its observation timestamp because each observation is
    a separate immutable fact.
    """

    ignored = volatile or set()
    identity = {str(key): value for key, value in payload.items() if key not in ignored}
    return f"work:{name}:{canonical_hash(identity).removeprefix('sha256:')}"


def _queue_payload_identity(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(payload)
    if name == "market_event_batch_write":
        # Queue retention is bookkeeping and must not change a replayed batch's identity.
        identity.pop("queue_retention_policy", None)
    return identity


def _queue_values_match(
    name: str, actual: Mapping[str, Any], expected: Mapping[str, Any], *, attempts: int
) -> bool:
    if attempts > 0:
        # Operator recovery can extend a used job's retry budget without changing its work.
        actual = {key: value for key, value in actual.items() if key != "max_attempts"}
        expected = {key: value for key, value in expected.items() if key != "max_attempts"}
    if name == "market_event_batch_write":
        actual = {**actual, "payload": _queue_payload_identity(name, actual["payload"])}
        expected = {**expected, "payload": _queue_payload_identity(name, expected["payload"])}
    return actual == expected


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    name: str
    payload: dict[str, Any]
    worker_id: str
    attempt: int
    lease_expires_at: str
    lease_token: str


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    interval_seconds: int
    priority: int = 0


AUTONOMOUS_SCHEDULES = (
    ScheduleSpec("universe_refresh", 300),
    ScheduleSpec("register_strategy_catalogue", 900),
    ScheduleSpec("candidate_generation", 900),
    ScheduleSpec("candidate_evaluation", 900),
    ScheduleSpec("bounded_backtest", 900),
    ScheduleSpec("event_replay", 1800),
    ScheduleSpec("ml_research", 3600),
    ScheduleSpec("forward_paper_observation", 300),
    ScheduleSpec("forward_paper_summary", 900),
    ScheduleSpec("promotion_evaluation", 900),
    ScheduleSpec("live_account_backfill", 300),
    ScheduleSpec("reporting", 300),
    ScheduleSpec("agent_review", 3600),
    ScheduleSpec("maintenance", 3600),
    ScheduleSpec("historical_bootstrap", 3600),
)


class PlatformScheduler:
    """Persist and enqueue the platform's recurring work without cron."""

    def __init__(
        self,
        *,
        engine: Engine,
        products: Mapping[str, Mapping[str, Any]],
        node_id: str,
        bootstrap: Any | None = None,
        research_configuration: Mapping[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.products = {str(key): dict(value) for key, value in products.items()}
        self.node_id = node_id
        self.bootstrap = bootstrap
        self.research_configuration = research_configuration or {}
        self.queue = DatabaseJobQueue(engine)

    def run_once(self, *, now: str) -> dict[str, Any]:
        now = timestamp(now, field="now")
        if self.bootstrap is not None:
            self.bootstrap.ensure(now=now)
        self.queue.register_worker(
            worker_id=f"{self.node_id}:platform-scheduler",
            node_id=self.node_id,
            role="platform-scheduler",
            capabilities=tuple(spec.name for spec in AUTONOMOUS_SCHEDULES),
            observed_at=now,
        )
        enqueued: list[str] = []
        for spec in AUTONOMOUS_SCHEDULES:
            schedule_id = f"platform:{spec.name}"
            queue_name = self._queue_name(spec.name)
            due_at = now
            with self.engine.begin() as connection:
                row = (
                    connection.execute(
                        select(platform_schedule).where(platform_schedule.c.id == schedule_id)
                    )
                    .mappings()
                    .first()
                )
                if row is not None and str(row["next_run_at"]) > now:
                    continue
                if row is not None:
                    due_at = timestamp(str(row["next_run_at"]), field="next_run_at")
            scheduled_job_ids: list[str] = []
            if queue_name is not None:
                products = (
                    (("global", {}),)
                    if spec.name == "reporting"
                    else tuple(sorted(self.products.items()))
                )
                for product_id, product in products:
                    for job_id, job_name, payload in self._jobs(
                        spec.name, product_id, product, due_at, due_at
                    ):
                        scheduled_job_ids.append(job_id)
                        if self.queue.enqueue_if_absent(
                            job_id=job_id,
                            name=job_name,
                            payload=payload,
                            available_at=due_at,
                            priority=spec.priority,
                            producer_identity=f"platform-scheduler:{self.node_id}",
                        ):
                            enqueued.append(job_id)
            next_run = _plus_seconds(now, spec.interval_seconds)
            values = {
                "id": schedule_id,
                "job_name": spec.name,
                "interval_seconds": spec.interval_seconds,
                "next_run_at": next_run,
                "last_run_at": now,
                "last_job_id": (
                    scheduled_job_ids[-1]
                    if scheduled_job_ids
                    else (str(row["last_job_id"]) if row and row["last_job_id"] else None)
                ),
                "state": "scheduled" if scheduled_job_ids else "waiting_for_inputs",
                "payload": {
                    "schema": "platform.schedule/v1",
                    "products": sorted(self.products),
                    "producer_identity": f"platform-scheduler:{self.node_id}",
                },
                "created_at": now,
                "updated_at": now,
            }
            with self.engine.begin() as connection:
                if row is None:
                    connection.execute(insert(platform_schedule).values(**values))
                else:
                    connection.execute(
                        update(platform_schedule)
                        .where(platform_schedule.c.id == schedule_id)
                        .values(
                            next_run_at=next_run,
                            last_run_at=now,
                            last_job_id=values["last_job_id"],
                            state="scheduled" if scheduled_job_ids else "waiting_for_inputs",
                            updated_at=now,
                        )
                    )
        maintenance = self._run_maintenance(now)
        return {
            "reason_code": "platform_scheduler_completed",
            "schedules": len(AUTONOMOUS_SCHEDULES),
            "jobs_enqueued": len(enqueued),
            "job_ids": enqueued,
            "maintenance": maintenance,
        }

    @staticmethod
    def _queue_name(schedule_name: str) -> str | None:
        return {
            "universe_refresh": "universe_refresh",
            "register_strategy_catalogue": "register_strategy_catalogue",
            "candidate_generation": "register_strategy_catalogue",
            "candidate_evaluation": "evaluate_candidate",
            "bounded_backtest": "bounded_backtest",
            "event_replay": "event_replay",
            "ml_research": "train_ml_experiment",
            "forward_paper_observation": "forward_paper_observation",
            "forward_paper_summary": "forward_paper_summary",
            "promotion_evaluation": "promotion_evaluation",
            "live_account_backfill": "live_account_backfill",
            "reporting": "reporting",
            "agent_review": "agent_review",
            "maintenance": "maintenance",
            "historical_bootstrap": "historical_bootstrap",
        }.get(schedule_name)

    def _jobs(
        self,
        schedule_name: str,
        product_id: str,
        product: Mapping[str, Any],
        now: str,
        due_at: str,
    ) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        if schedule_name in {"agent_review", "maintenance", "historical_bootstrap"}:
            return self._single_job(schedule_name, product_id, product, now, due_at)
        if schedule_name == "live_account_backfill" and product.get("execution_mode") != "live":
            return ()
        if schedule_name == "candidate_generation":
            return self._candidate_generation_jobs(product_id, product, due_at, now)
        if schedule_name in {"bounded_backtest", "event_replay", "ml_research"}:
            return self._research_jobs(schedule_name, product_id, now, due_at)
        if schedule_name == "universe_refresh" and self._pending_universe_refresh(product_id):
            return ()
        if schedule_name == "forward_paper_observation":
            return self._forward_jobs(product_id, now, due_at)
        if schedule_name == "forward_paper_summary":
            return self._summary_jobs(product_id, now, due_at)
        if schedule_name == "promotion_evaluation":
            return self._promotion_jobs(product_id, now, due_at)
        if schedule_name != "candidate_evaluation":
            return self._single_job(schedule_name, product_id, product, now, due_at)
        return self._candidate_evaluation_jobs(product_id, now, due_at)

    def _single_job(
        self,
        schedule_name: str,
        product_id: str,
        product: Mapping[str, Any],
        now: str,
        due_at: str,
    ) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        if schedule_name == "reporting":
            if self._pending_payloads("reporting"):
                return ()
            return (
                (
                    f"scheduled:reporting:global:{due_at}",
                    "reporting",
                    {"producer_identity": "platform-scheduler", "scheduled_at": due_at},
                ),
            )
        if schedule_name == "maintenance" and product_id != min(self.products):
            return ()
        if schedule_name == "historical_bootstrap":
            inputs = self._payload("register_strategy_catalogue", product_id, product, now)
            symbols = inputs["instrument_universe"]
            if not symbols:
                return ()
            plan = product_data_plan(product, self.research_configuration, symbols)
            payload = {
                **self._payload(schedule_name, product_id, product, now),
                "minimum_history_days": plan.minimum_history_days,
                "symbols": symbols,
                "timeframe": plan.timeframe,
                "data_plan": plan.to_payload(),
                "history_as_of": inputs["research_as_of"],
            }
            payload["available_at"] = "1970-01-01T00:00:00+00:00"
            payload["content_hash"] = canonical_hash(payload)
            return (
                (
                    immutable_work_id(
                        "historical_bootstrap",
                        {
                            "product_id": product_id,
                            "market_type": payload["market_type"],
                            "symbols": symbols,
                            "minimum_history_days": payload["minimum_history_days"],
                            "timeframe": payload["timeframe"],
                            "data_plan_id": plan.plan_id,
                            "history_as_of": inputs["research_as_of"],
                        },
                    ),
                    "historical_bootstrap",
                    payload,
                ),
            )
        if schedule_name == "register_strategy_catalogue":
            payload = self._payload(schedule_name, product_id, product, now)
            if not payload["instrument_universe"]:
                return ()
            payload["available_at"] = "1970-01-01T00:00:00+00:00"
            if payload.get("dataset_bundle_id"):
                payload = self._claim_research_round(
                    {**payload, "producer": "catalogue", "generation_budget": 12}, now
                )
                if payload is None:
                    return ()
            return (
                (
                    immutable_work_id(
                        "register_strategy_catalogue",
                        payload,
                        volatile={"available_at", "catalogue_submitted_at"},
                    ),
                    schedule_name,
                    payload,
                ),
            )
        return (
            (
                f"scheduled:{schedule_name}:{product_id}:{due_at}",
                schedule_name,
                self._payload(schedule_name, product_id, product, now),
            ),
        )

    def _candidate_generation_jobs(
        self,
        product_id: str,
        product: Mapping[str, Any],
        due_at: str,
        now: str,
    ) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        catalogue = self._payload("register_strategy_catalogue", product_id, product, now)
        payload = self._hypothesis_generation_payload(product_id, product, catalogue)
        if not payload.get("dataset_bundle_id"):
            return ()
        payload = self._claim_research_round({**payload, "producer": "adaptive"}, now)
        if payload is None:
            return ()
        return (
            (
                immutable_work_id("generate_hypotheses", payload),
                "generate_hypotheses",
                payload,
            ),
        )

    def _claim_research_round(self, payload, now):
        limits = self.research_configuration.get("search_budgets", {})
        return ResearchRoundCoordinator(self.engine).claim(
            payload,
            now=now,
            epoch_budget=int(limits.get("maximum_candidates_per_epoch", 100)),
            daily_budget=int(limits.get("maximum_candidates_per_day", 100)),
        )

    def _candidate_evaluation_jobs(
        self, product_id: str, now: str, due_at: str
    ) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        from src.research.store import SqlResearchStore

        jobs: list[tuple[str, str, dict[str, Any]]] = []
        product_candidates = tuple(
            candidate
            for candidate in SqlResearchStore(self.engine).load_candidates()
            if candidate.definition.product == product_id
        )
        diagnostic_ids = {
            candidate.candidate_id
            for candidate in product_candidates
            if _diagnostic_candidate(candidate.metadata, candidate.definition.metadata)
        }
        if diagnostic_ids:
            self._quarantine_diagnostic_evaluation_jobs(diagnostic_ids, now)
        candidates = tuple(
            candidate
            for candidate in product_candidates
            if candidate.candidate_id not in diagnostic_ids and candidate.dataset_snapshot_hashes
        )
        for candidate in candidates:
            with self.engine.connect() as connection:
                state = connection.execute(
                    select(experiment.c.state).where(experiment.c.id == candidate.candidate_id)
                ).scalar_one()
            state_value = str(state)
            requested_stage = {
                "queued": "screening",
                "screening": "development",
                "screening_deferred": "screening",
                "development": "robustness",
                "development_deferred": "development",
                "robustness": "protected",
                "robustness_deferred": "robustness",
                "protected_deferred": "protected",
            }.get(state_value)
            if requested_stage is None and state_value.startswith("waiting_for_dataset:"):
                requested_stage = state_value.removeprefix("waiting_for_dataset:")
            if requested_stage is None:
                continue
            if self._candidate_stage_dead_letter(candidate.candidate_id, requested_stage):
                self._mark_candidate_dead_letter(candidate.candidate_id, requested_stage, now)
                continue
            expected_role = {
                "screening": "screening",
                "development": "development",
                "robustness": "robustness",
                "protected": "protected_holdout",
            }[requested_stage]
            plan_snapshot_ids = (
                candidate.dataset_plan.snapshot_ids_for_stage(requested_stage)
                if candidate.dataset_plan is not None
                else candidate.dataset_snapshot_hashes
            )
            snapshot_id, snapshot_payload = self._candidate_snapshot(
                plan_snapshot_ids,
                role=expected_role,
            )
            if snapshot_id is None or snapshot_payload is None:
                self._mark_dataset_waiting(candidate.candidate_id, requested_stage, now)
                continue
            if self._candidate_stage_job_pending(candidate.candidate_id, requested_stage):
                continue
            if state_value.startswith("waiting_for_dataset:"):
                self._clear_dataset_waiting(
                    candidate.candidate_id,
                    requested_stage,
                    {
                        "screening": "queued",
                        "development": "screening",
                        "robustness": "development",
                        "protected": "robustness",
                    }[requested_stage],
                )
            jobs.append(
                (
                    immutable_work_id(
                        "candidate_evaluation",
                        self._research_request(
                            candidate_id=candidate.candidate_id,
                            snapshot_id=snapshot_id,
                            snapshot_payload=snapshot_payload,
                            requested_stage=requested_stage,
                            evaluated_at=now,
                        ),
                        volatile={"evaluated_at", "content_hash"},
                    ),
                    "evaluate_candidate",
                    self._research_request(
                        candidate_id=candidate.candidate_id,
                        snapshot_id=snapshot_id,
                        snapshot_payload=snapshot_payload,
                        requested_stage=requested_stage,
                        evaluated_at=now,
                    ),
                )
            )
        return tuple(jobs)

    def _quarantine_diagnostic_evaluation_jobs(self, candidate_ids: set[str], now: str) -> None:
        reason = "non_promotable_diagnostic_excluded_from_autonomous_evaluation"
        with self.engine.begin() as connection:
            statement = (
                select(job)
                .where(
                    job.c.name == "evaluate_candidate",
                    job.c.state == "pending",
                    job.c.producer_identity.like("platform-scheduler:%"),
                    job.c.payload["candidate_id"].as_string().in_(candidate_ids),
                )
                .order_by(job.c.id)
            )
            if self.engine.dialect.name == "postgresql":
                statement = statement.with_for_update(skip_locked=True)
            pending = connection.execute(statement).mappings()
            for row in pending:
                payload = row["payload"]
                if (
                    not isinstance(payload, Mapping)
                    or str(payload.get("candidate_id") or "") not in candidate_ids
                ):
                    continue
                job_id = str(row["id"])
                transitioned = connection.execute(
                    update(job)
                    .where(job.c.id == job_id, job.c.state == "pending")
                    .values(
                        state="quarantined",
                        lease_owner=None,
                        lease_expires_at=None,
                        terminal_reason=reason,
                    )
                )
                if transitioned.rowcount != 1:
                    continue
                audit = {
                    "job_id": job_id,
                    "name": str(row["name"]),
                    "state": str(row["state"]),
                    "content_hash": str(row["content_hash"]),
                    "payload": payload,
                    "reason": reason,
                    "quarantined_at": now,
                }
                prior = connection.execute(
                    select(job_quarantine.c.job_id).where(job_quarantine.c.job_id == job_id)
                ).first()
                if prior is None:
                    connection.execute(
                        insert(job_quarantine).values(
                            id=canonical_hash(audit),
                            job_id=job_id,
                            quarantined_at=now,
                            original_state=str(row["state"]),
                            reason=reason,
                            content_hash=str(row["content_hash"]),
                            payload=json_value(audit, field="quarantined job"),
                        )
                    )

    def _mark_dataset_waiting(self, candidate_id: str, stage: str, now: str) -> None:
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    select(experiment.c.state, experiment.c.metadata).where(
                        experiment.c.id == candidate_id
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return
            metadata = dict(row["metadata"] or {})
            waiting = {
                "stage": stage,
                "reason_code": "missing_canonical_stage_dataset",
                "observed_at": now,
            }
            if (
                row["state"] == f"waiting_for_dataset:{stage}"
                and metadata.get("dataset_waiting") == waiting
            ):
                return
            metadata["dataset_waiting"] = waiting
            connection.execute(
                update(experiment)
                .where(experiment.c.id == candidate_id)
                .values(
                    state=f"waiting_for_dataset:{stage}",
                    metadata=json_value(metadata, field="metadata"),
                )
            )

    def _clear_dataset_waiting(self, candidate_id: str, stage: str, state: str) -> None:
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    select(experiment.c.metadata).where(experiment.c.id == candidate_id)
                )
                .mappings()
                .first()
            )
            if row is None:
                return
            metadata = dict(row["metadata"] or {})
            metadata.pop("dataset_waiting", None)
            connection.execute(
                update(experiment)
                .where(experiment.c.id == candidate_id)
                .values(state=state, metadata=json_value(metadata, field="metadata"))
            )

    def _candidate_stage_job_pending(self, candidate_id: str, stage: str) -> bool:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(job.c.state, job.c.payload).where(job.c.name == "evaluate_candidate")
            ).mappings()
        return any(
            str(row["state"]) in {"pending", "running"}
            and isinstance(row["payload"], Mapping)
            and row["payload"].get("candidate_id") == candidate_id
            and row["payload"].get("requested_stage") == stage
            for row in rows
        )

    def _candidate_stage_dead_letter(self, candidate_id: str, stage: str) -> bool:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(job.c.payload).where(
                    job.c.name == "evaluate_candidate",
                    job.c.state == "dead_letter",
                )
            ).scalars()
            return any(
                isinstance(payload, Mapping)
                and payload.get("candidate_id") == candidate_id
                and payload.get("requested_stage") == stage
                for payload in rows
            )

    def _mark_candidate_dead_letter(self, candidate_id: str, stage: str, now: str) -> None:
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    select(experiment.c.state, experiment.c.metadata).where(
                        experiment.c.id == candidate_id
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return
            metadata = dict(row["metadata"] or {})
            blocked = {
                "stage": stage,
                "reason_code": "evaluation_job_dead_letter",
                "observed_at": now,
            }
            if (
                row["state"] == TrialDisposition.INFRASTRUCTURE_EXHAUSTED.value
                and metadata.get("blocked_reason") == blocked
            ):
                return
            metadata["blocked_reason"] = blocked
            connection.execute(
                update(experiment)
                .where(experiment.c.id == candidate_id)
                .values(
                    state=TrialDisposition.INFRASTRUCTURE_EXHAUSTED.value,
                    metadata=json_value(metadata, field="metadata"),
                )
            )

    def _pending_universe_refresh(self, product_id: str) -> bool:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(job.c.state, job.c.payload).where(job.c.name == "universe_refresh")
            ).mappings()
            return any(
                str(row["state"]) in {"pending", "running"}
                and isinstance(row["payload"], Mapping)
                and str(row["payload"].get("product_id")) == product_id
                for row in rows
            )

    def _research_jobs(
        self, schedule_name: str, product_id: str, now: str, due_at: str
    ) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        from src.research.store import SqlResearchStore

        required_payload, job_name = {
            "bounded_backtest": ("bar_steps", "bounded_backtest"),
            "event_replay": ("event_replay", "event_replay"),
            "ml_research": ("ml_rows", "train_ml_experiment"),
        }[schedule_name]
        jobs: list[tuple[str, str, dict[str, Any]]] = []
        for candidate in SqlResearchStore(self.engine).load_candidates():
            if candidate.definition.product != product_id:
                continue
            is_ml = candidate.definition.source_type.value == "machine_learning"
            if is_ml != (schedule_name == "ml_research"):
                continue
            snapshot_id, payload = self._candidate_snapshot(
                candidate.dataset_snapshot_hashes,
                required_payload=required_payload,
            )
            if snapshot_id is None or payload is None:
                continue
            role = str(payload.get("role") or "screening")
            stage = {
                "screening": "screening",
                "development": "development",
                "robustness": "robustness",
            }.get(role)
            if stage is None:
                continue
            jobs.append(
                (
                    immutable_work_id(
                        job_name,
                        self._research_request(
                            candidate_id=candidate.candidate_id,
                            snapshot_id=snapshot_id,
                            snapshot_payload=payload,
                            requested_stage=stage,
                            evaluated_at=now,
                        ),
                        volatile={"evaluated_at", "content_hash"},
                    ),
                    job_name,
                    self._research_request(
                        candidate_id=candidate.candidate_id,
                        snapshot_id=snapshot_id,
                        snapshot_payload=payload,
                        requested_stage=stage,
                        evaluated_at=now,
                    ),
                )
            )
        return tuple(jobs)

    @staticmethod
    def _hypothesis_generation_payload(
        product_id: str,
        product: Mapping[str, Any],
        catalogue: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "product_id": product_id,
            "instrument_universe": list(catalogue["instrument_universe"]),
            "dataset_snapshot_hashes": list(catalogue["dataset_snapshot_hashes"]),
            "dataset_bundle_id": catalogue.get("dataset_bundle_id"),
            "universe_snapshot_id": catalogue.get("universe_snapshot_id"),
            "submitted_at": (
                str(catalogue["catalogue_submitted_at"])
                if catalogue.get("dataset_snapshot_hashes")
                else "1970-01-01T00:00:00+00:00"
            ),
            "generation_budget": int(product.get("generation_budget", 6)),
            "universe_id": str(catalogue.get("universe_id") or ""),
            "research_as_of": catalogue.get("research_as_of"),
            "market_type": "spot" if product_id == "btc_accumulation" else "futures",
            "dataset_timeframe": str(product.get("dataset_timeframe") or "1m"),
            "data_plan": catalogue.get("data_plan"),
            "available_at": ("1970-01-01T00:00:00+00:00"),
        }
        return {key: value for key, value in payload.items() if value is not None}

    def _candidate_snapshot(
        self,
        snapshot_ids: tuple[str, ...],
        *,
        role: str | None = None,
        required_payload: str | None = None,
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        ordered_ids = tuple(dict.fromkeys(snapshot_ids))
        with self.engine.connect() as connection:
            manifests = _load_snapshot_manifests(connection, ordered_ids)
            eligible_ids = tuple(
                snapshot_id
                for snapshot_id in ordered_ids
                if snapshot_id in manifests
                and (
                    role is None or str(manifests[snapshot_id].get("role") or "unspecified") == role
                )
            )
            if not eligible_ids:
                return None, None
            if required_payload is None:
                selected_id = eligible_ids[0]
                return selected_id, manifests[selected_id]

            payload = dataset_snapshot.c.payload
            if connection.dialect.name == "postgresql":
                encoding = payload["payload_encoding"].as_string()
                inline_data = payload["payload"]
                compressed_keys = inline_data["payload_keys"]
                has_compressed_key = and_(
                    func.jsonb_typeof(compressed_keys) == "array",
                    func.jsonb_exists(compressed_keys, required_payload),
                )
                has_inline_key = and_(
                    func.jsonb_typeof(inline_data) == "object",
                    func.jsonb_exists(inline_data, required_payload),
                )
                has_required_payload = or_(
                    and_(encoding == "canonical-json-gzip/v1", has_compressed_key),
                    and_(
                        func.jsonb_exists(payload, "payload_encoding").is_(False),
                        has_inline_key,
                    ),
                )
            else:
                inline_data = (
                    func.json_each(payload, "$.payload")
                    .table_valued("key")
                    .alias("inline_payload_keys")
                )
                compressed_keys = (
                    func.json_each(payload, "$.payload.payload_keys")
                    .table_valued("value")
                    .alias("compressed_payload_keys")
                )
                has_inline_key = (
                    select(1)
                    .select_from(inline_data)
                    .where(inline_data.c.key == required_payload)
                    .exists()
                )
                has_compressed_key = and_(
                    func.json_type(payload, "$.payload.payload_keys") == "array",
                    select(1)
                    .select_from(compressed_keys)
                    .where(compressed_keys.c.value == required_payload)
                    .exists(),
                )
                has_required_payload = or_(
                    and_(
                        func.json_extract(payload, "$.payload_encoding")
                        == "canonical-json-gzip/v1",
                        has_compressed_key,
                    ),
                    and_(
                        func.json_type(payload, "$.payload_encoding").is_(None),
                        has_inline_key,
                    ),
                )
            available = set(
                connection.execute(
                    select(dataset_snapshot.c.id).where(
                        dataset_snapshot.c.id.in_(eligible_ids), has_required_payload
                    )
                ).scalars()
            )
            capable_id = next((item for item in eligible_ids if item in available), None)
            return (capable_id, manifests[capable_id]) if capable_id is not None else (None, None)

    @staticmethod
    def _research_request(
        *,
        candidate_id: str,
        snapshot_id: str,
        snapshot_payload: Mapping[str, Any],
        requested_stage: str,
        evaluated_at: str,
    ) -> dict[str, Any]:
        role = str(snapshot_payload.get("role") or "screening")
        request = {
            "candidate_id": candidate_id,
            "dataset_snapshot_ids": [snapshot_id],
            "feature_manifest_id": str(snapshot_payload.get("feature_manifest_id") or ""),
            "cost_model_id": str(snapshot_payload.get("cost_model_id") or ""),
            "parameter_set_id": str(snapshot_payload.get("parameter_set_id") or ""),
            "evaluator_version": (
                "platform-scheduler/v1:" + FUTURES_ACCOUNTING_RISK_CONTRACT
                if snapshot_payload.get("product_id") == "active_income"
                else "platform-scheduler/v1"
            ),
            "requested_stage": requested_stage,
            "evaluated_at": str(snapshot_payload.get("availability_timestamp") or evaluated_at),
            "producer_identity": "platform-scheduler",
            "dataset_roles": {snapshot_id: role},
        }
        request["content_hash"] = canonical_hash(request)
        return request

    def _promotion_jobs(
        self, product_id: str, now: str, due_at: str
    ) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(strategy_artefact.c.payload)).scalars()
            artefacts = [dict(payload) for payload in rows if isinstance(payload, Mapping)]
        jobs: list[tuple[str, str, dict[str, Any]]] = []
        active_assignments = SqlActiveStrategyAssignmentRepository(self.engine).active_assignments(
            product_id, at=now
        )
        pending_strategy_ids = {
            str(payload.get("strategy_version_id"))
            for payload in self._pending_payloads("promotion_evaluation")
            if payload.get("product_id") in {None, product_id}
        }
        for artefact in artefacts:
            metadata = artefact.get("metadata")
            if (
                artefact.get("product_id") != product_id
                or isinstance(metadata, Mapping)
                and _non_promotable_artefact(metadata)
            ):
                continue
            strategy_version_id = str(artefact.get("strategy_version_id") or "")
            if not strategy_version_id:
                continue
            if strategy_version_id in pending_strategy_ids:
                continue
            if any(
                row["strategy_version_id"] == strategy_version_id
                and row["lifecycle_state"] in {"live_canary", "live"}
                for row in active_assignments
            ):
                continue
            latest_summary = self._latest_forward_summary(
                strategy_version_id=strategy_version_id,
                product_id=product_id,
                artefact_hash=str(artefact.get("artefact_hash") or ""),
                at=due_at,
            )
            # Promotion is evidence-driven. Do not create a timestamped error
            # job for an artefact which has not produced a forward summary.
            if latest_summary is None:
                entry = self._paper_entry_job(artefact, active_assignments, now)
                if entry is not None:
                    jobs.append(entry)
                continue
            summary_id = str(latest_summary["id"])
            evaluated_at = str(latest_summary["created_at"])
            jobs.append(
                (
                    immutable_work_id(
                        "promotion_evaluation",
                        {
                            "strategy_version_id": strategy_version_id,
                            "requested_transition": "forward_paper",
                            "forward_summary_id": summary_id,
                        },
                    ),
                    "promotion_evaluation",
                    {
                        "strategy_version_id": strategy_version_id,
                        "requested_transition": "forward_paper",
                        "requested_capital": 0.0,
                        "evaluated_at": evaluated_at,
                        "forward_summary_id": summary_id,
                        "artefact_hash": str(artefact["artefact_hash"]),
                    },
                )
            )
        return tuple(jobs)

    def _paper_entry_job(self, artefact, assignments, now: str):
        identity = str(artefact.get("artefact_hash") or "")
        job_id = f"paper-entry:{identity}"
        if self.queue.state(job_id) is not None:
            return None
        if any(row["artefact_hash"] == identity for row in assignments):
            return None
        with self.engine.connect() as connection:
            eligibility = (
                connection.execute(
                    select(paper_eligibility_event).where(
                        paper_eligibility_event.c.id == identity,
                        paper_eligibility_event.c.created_at <= now,
                    )
                )
                .mappings()
                .first()
            )
        if eligibility is None:
            return None
        return (
            job_id,
            "promotion_evaluation",
            {
                "strategy_version_id": artefact["strategy_version_id"],
                "artefact_hash": identity,
                "requested_transition": "forward_paper",
                "requested_capital": 0.0,
                "evaluated_at": now,
            },
        )

    def _latest_forward_summary(
        self,
        *,
        strategy_version_id: str,
        product_id: str,
        artefact_hash: str,
        at: str,
    ) -> Mapping[str, Any] | None:
        if not artefact_hash:
            return None
        from src.data.database import forward_paper_summary

        with self.engine.connect() as connection:
            return (
                connection.execute(
                    select(
                        forward_paper_summary.c.id,
                        forward_paper_summary.c.created_at,
                    )
                    .where(
                        forward_paper_summary.c.strategy_version_id == strategy_version_id,
                        forward_paper_summary.c.product_id == product_id,
                        forward_paper_summary.c.artefact_hash == artefact_hash,
                        forward_paper_summary.c.created_at <= at,
                    )
                    .order_by(
                        forward_paper_summary.c.created_at.desc(),
                        forward_paper_summary.c.id.desc(),
                    )
                    .limit(1)
                )
                .mappings()
                .first()
            )

    def _forward_jobs(
        self, product_id: str, now: str, due_at: str
    ) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        assignments = tuple(
            assignment
            for assignment in SqlActiveStrategyAssignmentRepository(self.engine).active_assignments(
                product_id, at=now
            )
            if assignment["execution_mode"] == "paper"
        )
        invalidated = self._invalidated_artefacts(
            {str(assignment["artefact_hash"]) for assignment in assignments}
        )
        jobs: list[tuple[str, str, dict[str, Any]]] = []
        pending = {
            (
                str(payload.get("assignment_id")),
                str(payload.get("instrument_id")),
            )
            for payload in self._pending_payloads("forward_paper_observation")
        }
        for assignment in assignments:
            if str(assignment["artefact_hash"]) in invalidated:
                continue
            payload = assignment.get("payload")
            if isinstance(payload, Mapping) and payload.get("promotable") is False:
                continue
            assignment_id = str(assignment["id"])
            instrument_ids = self._forward_instrument_ids(assignment, product_id, now)
            if not instrument_ids:
                instrument_ids = ("",)
            for instrument_id in instrument_ids:
                if (assignment_id, instrument_id) in pending:
                    continue
                jobs.append(
                    (
                        immutable_work_id(
                            "forward_paper_observation",
                            {
                                "assignment_id": assignment_id,
                                "instrument_id": instrument_id,
                                "due_at": due_at,
                            },
                        ),
                        "forward_paper_observation",
                        {
                            "assignment_id": assignment_id,
                            "strategy_version_id": str(assignment["strategy_version_id"]),
                            "product_id": product_id,
                            "instrument_id": instrument_id,
                            "artefact_hash": str(assignment["artefact_hash"]),
                            "evaluation_time": now,
                            **(
                                {"waiting_reason": "universe_members_unavailable"}
                                if not instrument_id
                                else {}
                            ),
                        },
                    )
                )
        return tuple(jobs)

    def _invalidated_artefacts(self, artefact_hashes: set[str]) -> set[str]:
        if not artefact_hashes:
            return set()
        invalidation_ids = {f"artefact:{identity}" for identity in artefact_hashes}
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(evidence_invalidation.c.id).where(
                    evidence_invalidation.c.id.in_(invalidation_ids)
                )
            ).scalars()
        return {str(identity).removeprefix("artefact:") for identity in rows}

    def _summary_jobs(
        self, product_id: str, now: str, due_at: str
    ) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        pending = {
            (
                str(payload.get("strategy_version_id")),
                str(payload.get("artefact_hash")),
                str(payload.get("evaluation_time")),
            )
            for payload in self._pending_payloads("forward_paper_summary")
        }
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    forward_paper_observation.c.strategy_version_id,
                    forward_paper_observation.c.product_id,
                    forward_paper_observation.c.artefact_hash,
                    func.max(forward_paper_observation.c.observed_at).label("latest_observed_at"),
                )
                .where(
                    forward_paper_observation.c.product_id == product_id,
                    forward_paper_observation.c.observed_at <= now,
                )
                .group_by(
                    forward_paper_observation.c.strategy_version_id,
                    forward_paper_observation.c.product_id,
                    forward_paper_observation.c.artefact_hash,
                )
            ).mappings()
            groups = tuple(
                sorted(
                    {
                        (
                            str(row["strategy_version_id"]),
                            str(row["product_id"]),
                            str(row["artefact_hash"]),
                            str(row["latest_observed_at"]),
                        )
                        for row in rows
                    }
                )
            )
        invalidated = self._invalidated_artefacts({group[2] for group in groups})
        return tuple(
            (
                immutable_work_id(
                    "forward_paper_summary",
                    {
                        "strategy_version_id": strategy_version_id,
                        "product_id": group_product_id,
                        "artefact_hash": artefact_hash,
                        "evaluation_time": evaluation_time,
                    },
                ),
                "forward_paper_summary",
                {
                    "strategy_version_id": strategy_version_id,
                    "product_id": group_product_id,
                    "artefact_hash": artefact_hash,
                    "evaluation_time": evaluation_time,
                },
            )
            for strategy_version_id, group_product_id, artefact_hash, evaluation_time in groups
            if (strategy_version_id, artefact_hash, evaluation_time) not in pending
            and artefact_hash not in invalidated
        )

    def _pending_payloads(self, name: str) -> tuple[Mapping[str, Any], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(job.c.payload).where(
                    job.c.name == name,
                    job.c.state.in_(("pending", "running")),
                )
            ).scalars()
        return tuple(payload for payload in rows if isinstance(payload, Mapping))

    def _forward_instrument_ids(
        self, assignment: Mapping[str, Any], product_id: str, now: str
    ) -> tuple[str, ...]:
        direct = str(assignment.get("instrument_id") or "")
        if direct:
            return (direct,)
        payload = assignment.get("payload")
        if isinstance(payload, Mapping):
            configured = payload.get("instrument_ids")
            if isinstance(configured, list | tuple):
                values = tuple(sorted({str(item) for item in configured if str(item)}))
                if values:
                    return values
        universe_id = str(assignment.get("universe_id") or "")
        configured_product = self.products.get(product_id, {})
        if universe_id.startswith("product:") or not universe_id:
            universe_id = str(configured_product.get("universe_id") or universe_id)
        if not universe_id:
            return ()
        with self.engine.connect() as connection:
            snapshot_id = connection.execute(
                select(universe_snapshot.c.id)
                .where(
                    universe_snapshot.c.universe_id == universe_id,
                    universe_snapshot.c.observed_at <= now,
                )
                .order_by(universe_snapshot.c.observed_at.desc(), universe_snapshot.c.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            if snapshot_id is None:
                return ()
            rows = connection.execute(
                select(universe_member.c.instrument_id)
                .where(
                    universe_member.c.snapshot_id == snapshot_id,
                    universe_member.c.eligible.is_(True),
                )
                .order_by(universe_member.c.instrument_id)
            ).scalars()
        return tuple(str(item) for item in rows)

    def _run_maintenance(self, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=f"{self.node_id}:platform-scheduler",
            now=now,
            lease_seconds=60,
            names=("maintenance",),
        )
        if claimed is None:
            return {"reason_code": "maintenance_queue_empty"}
        recovered = self.queue.recover_expired(now=now)
        self.queue.complete(claimed, completed_at=now)
        delivery = self.queue.alerts.deliver_pending(now=now)
        return {
            "reason_code": "platform_maintenance_completed",
            "job_id": claimed.job_id,
            "expired_jobs_recovered": recovered,
            "alert_delivery": {
                "delivered": list(delivery.delivered),
                "failed": list(delivery.failed),
            },
        }

    def _payload(
        self,
        schedule_name: str,
        product_id: str,
        product: Mapping[str, Any],
        now: str,
    ) -> dict[str, Any]:
        base = {
            "schedule_name": schedule_name,
            "product_id": product_id,
            "universe_id": str(product.get("universe_id", "")),
            "available_at": now,
            "producer_identity": "platform-scheduler",
            "market_type": "spot" if product_id == "btc_accumulation" else "futures",
        }
        if schedule_name == "universe_refresh":
            return {
                **base,
                "policy": {},
                "maximum_symbols": int(product.get("research_max_symbols", 100)),
            }
        if schedule_name == "register_strategy_catalogue":
            return {
                **base,
                **research_inputs(
                    self.engine,
                    {**product, "product_id": product_id},
                    self.research_configuration,
                    now,
                ),
                "dataset_timeframe": str(product.get("dataset_timeframe") or "1m"),
            }
        if schedule_name == "agent_review":
            return {
                **base,
                "review_scope": "development_and_robustness",
                "reason_code": "scheduled_research_review",
            }
        return base


class DatabaseJobQueue:
    def __init__(
        self,
        engine: Engine,
        *,
        alerts: SqlAlertService | None = None,
        claim_scope: str | None = None,
    ):
        self.engine = engine
        self.alerts = alerts or configured_alert_service(engine)
        self.claim_scope = (
            non_empty(claim_scope, field="claim_scope") if claim_scope is not None else None
        )

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
        with self._transaction() as connection:
            exists = connection.execute(select(worker.c.id).where(worker.c.id == worker_id)).first()
            if exists:
                connection.execute(update(worker).where(worker.c.id == worker_id).values(**values))
            else:
                connection.execute(insert(worker).values(**values))

    @contextmanager
    def _transaction(self):
        # Queue locks must not outlive worker leases or block service shutdown.
        with self.engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                connection.exec_driver_sql("SET LOCAL lock_timeout = '2s'")
                connection.exec_driver_sql("SET LOCAL statement_timeout = '10s'")
            yield connection

    def _submission_values(
        self,
        *,
        job_id: str,
        name: str,
        payload: dict[str, Any],
        available_at: str,
        priority: int,
        producer_identity: str | None,
        max_attempts: int,
    ) -> dict[str, Any]:
        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        try:
            clean_payload = validate_job_payload(name, payload)
        except JobSchemaError:
            if self.engine.dialect.name != "sqlite" or not _legacy_risk_fixture(payload):
                raise
            clean_payload = json_value(dict(payload), field=f"{name} payload")
        producer = self._producer_identity(
            name=name, payload=clean_payload, producer_identity=producer_identity
        )
        return {
            "id": non_empty(job_id, field="job_id"),
            "name": non_empty(name, field="name"),
            "state": "pending",
            "priority": int(priority),
            "available_at": timestamp(available_at, field="available_at"),
            "lease_owner": None,
            "lease_expires_at": None,
            "attempts": 0,
            "max_attempts": max_attempts,
            "terminal_reason": None,
            "producer_identity": producer,
            "content_hash": canonical_hash(_queue_payload_identity(name, clean_payload)),
            "payload": json_value(clean_payload, field="payload"),
        }

    def enqueue(
        self,
        *,
        job_id: str,
        name: str,
        payload: dict[str, Any],
        available_at: str,
        priority: int = 0,
        producer_identity: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        values = self._submission_values(
            job_id=job_id,
            name=name,
            payload=payload,
            available_at=available_at,
            priority=priority,
            producer_identity=producer_identity,
            max_attempts=max_attempts,
        )
        with self._transaction() as connection:
            restricted = _restricted_job_insert(connection, values)
            if restricted is not None:
                if not restricted:
                    raise ValueError(f"duplicate job_id: {job_id}")
                return
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
        max_attempts: int = 3,
        connection=None,
    ) -> bool:
        values = self._submission_values(
            job_id=job_id,
            name=name,
            payload=payload,
            available_at=available_at,
            priority=priority,
            producer_identity=producer_identity,
            max_attempts=max_attempts,
        )
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
        transaction = (
            self._transaction()
            if connection is None
            else connection_scope(self.engine, connection, write=True)
        )
        with transaction as connection:
            restricted = _restricted_job_insert(connection, values)
            if restricted is not None:
                return restricted
            result = connection.execute(statement)
            inserted = result.scalar_one_or_none() is not None
            if inserted:
                return True
            existing = connection.execute(select(job).where(job.c.id == job_id)).mappings().one()
        expected = {
            "name": values["name"],
            "priority": values["priority"],
            "payload": values["payload"],
            "producer_identity": values["producer_identity"],
            "content_hash": values["content_hash"],
            "max_attempts": values["max_attempts"],
        }
        if (
            existing["state"] == "pending"
            and existing["attempts"] == 0
            and not job_id.startswith("work:")
        ):
            expected["available_at"] = values["available_at"]
        actual = {key: existing[key] for key in expected}
        if not _queue_values_match(name, actual, expected, attempts=int(existing["attempts"])):
            raise ValueError(f"job identity collision: {job_id}")
        return False

    def state(self, job_id: str) -> str | None:
        """Read one durable job state without claiming or mutating it."""

        with self.engine.connect() as connection:
            value = connection.execute(
                select(job.c.state).where(job.c.id == job_id)
            ).scalar_one_or_none()
        return None if value is None else str(value)

    def prune_completed_public_market_batches(self) -> int:
        """Prune at most 1,000 expired tagged market batch jobs on PostgreSQL."""

        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("market batch queue retention requires PostgreSQL")
        with self._transaction() as connection:
            return int(
                connection.execute(
                    text("SELECT public.prune_completed_public_market_batch_jobs()")
                ).scalar_one()
            )

    def requeue(
        self,
        job_id: str,
        *,
        available_at: str,
        reason: str,
        additional_attempts: int = 3,
    ) -> bool:
        """Re-admit one dead-letter job after an operator-safe recovery event."""

        if isinstance(additional_attempts, bool) or additional_attempts < 1:
            raise ValueError("additional_attempts must be positive")
        available_at = timestamp(available_at, field="available_at")
        non_empty(reason, field="requeue reason")
        with self._transaction() as connection:
            current = connection.execute(
                select(job.c.state, job.c.attempts, job.c.max_attempts).where(job.c.id == job_id)
            ).first()
            if current is None or current.state != "dead_letter":
                return False
            next_max_attempts = max(
                int(current.max_attempts), int(current.attempts) + additional_attempts
            )
            connection.execute(
                update(job)
                .where(job.c.id == job_id, job.c.state == "dead_letter")
                .values(
                    state="pending",
                    available_at=available_at,
                    max_attempts=next_max_attempts,
                    lease_owner=None,
                    lease_expires_at=None,
                    terminal_reason=None,
                )
            )
        return True

    def claim(
        self,
        *,
        worker_id: str,
        now: str,
        lease_seconds: int,
        names: tuple[str, ...] = (),
        producer_identity: str | None = None,
    ) -> ClaimedJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = timestamp(now, field="now")
        expires_at = _plus_seconds(now, lease_seconds)
        with self._transaction() as connection:
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
            claim_producer = producer_identity or self.claim_scope
            if claim_producer is not None:
                eligible = and_(
                    eligible,
                    job.c.producer_identity == non_empty(claim_producer, field="producer_identity"),
                )
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
            if self._expire_for_reclaim(connection, row, now):
                terminal_row = row
            else:
                attempt = int(row["attempts"]) + 1
                attempt_id = f"{row['id']}:{attempt}"
                lease_token = str(uuid4())
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
                        payload={"lease_token": lease_token},
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
                    lease_token=lease_token,
                )

        self._emit_terminal_failure(
            job_id=str(terminal_row["id"]),
            job_name=str(terminal_row["name"]),
            attempt=int(terminal_row["attempts"]),
            error="worker_lease_expired",
            observed_at=now,
        )
        return None

    @staticmethod
    def _expire_for_reclaim(connection, row, now: str) -> bool:
        if row["state"] != "running":
            return False
        connection.execute(
            update(job_attempt)
            .where(job_attempt.c.job_id == row["id"], job_attempt.c.status == "running")
            .values(completed_at=now, status="expired", error="worker_lease_expired")
        )
        connection.execute(
            update(worker_lease)
            .where(worker_lease.c.job_id == row["id"], worker_lease.c.status == "active")
            .values(status="expired")
        )
        terminal = int(row["attempts"]) >= int(row["max_attempts"])
        if terminal:
            connection.execute(
                update(job)
                .where(job.c.id == row["id"])
                .values(
                    state="dead_letter",
                    lease_owner=None,
                    lease_expires_at=None,
                    terminal_reason="worker_lease_expired",
                )
            )
        return terminal

    def _producer_identity(
        self,
        *,
        name: str,
        payload: Mapping[str, Any],
        producer_identity: str | None,
    ) -> str:
        if self.claim_scope is not None:
            return self.claim_scope
        return non_empty(
            producer_identity or str(payload.get("producer_identity") or f"service:{name}"),
            field="producer_identity",
        )

    @staticmethod
    def _owned_job(connection, claimed: ClaimedJob):
        statement = select(
            job.c.state, job.c.lease_owner, job.c.attempts, job.c.max_attempts
        ).where(job.c.id == claimed.job_id)
        if connection.dialect.name == "postgresql":
            statement = statement.with_for_update()
        current = connection.execute(statement).first()
        if (
            current is None
            or current.state != "running"
            or current.lease_owner != claimed.worker_id
            or current.attempts != claimed.attempt
        ):
            raise ValueError("worker no longer owns the job lease")
        payload = connection.execute(
            select(worker_lease.c.payload).where(
                worker_lease.c.id == f"{claimed.job_id}:{claimed.attempt}",
                worker_lease.c.status == "active",
            )
        ).scalar_one_or_none()
        if not isinstance(payload, Mapping) or payload.get("lease_token") != claimed.lease_token:
            raise ValueError("worker no longer owns the job lease")
        return current

    def heartbeat(self, claimed: ClaimedJob, *, now: str, lease_seconds: int) -> ClaimedJob:
        now = timestamp(now, field="now")
        expires_at = _plus_seconds(now, lease_seconds)
        attempt_id = f"{claimed.job_id}:{claimed.attempt}"
        with self._transaction() as connection:
            self._owned_job(connection, claimed)
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
            lease_token=claimed.lease_token,
        )

    def complete(
        self, claimed: ClaimedJob, *, completed_at: str, outcome: Mapping[str, Any] | None = None
    ) -> None:
        self._finish(
            claimed, completed_at=completed_at, status="completed", error=None, outcome=outcome
        )

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

    def defer(
        self,
        claimed: ClaimedJob,
        *,
        available_at: str,
        reason: str,
        outcome_category: OutcomeCategory = OutcomeCategory.BLOCKED_INFRASTRUCTURE,
    ) -> None:
        """Return a claimed job to the queue without consuming a retry.

        Resource admission is not work failure. Preserve the admission receipt
        under a separate identity so the next execution can reuse its retry.
        """

        available_at = timestamp(available_at, field="available_at")
        reason = non_empty(reason, field="defer reason")
        attempt_id = f"{claimed.job_id}:{claimed.attempt}"
        with self._transaction() as connection:
            current = self._owned_job(connection, claimed)
            connection.execute(
                update(job)
                .where(job.c.id == claimed.job_id)
                .values(
                    state="pending",
                    available_at=available_at,
                    lease_owner=None,
                    lease_expires_at=None,
                    attempts=max(0, int(current.attempts) - 1),
                    terminal_reason=None,
                )
            )
            connection.execute(
                update(job_attempt)
                .where(job_attempt.c.id == attempt_id)
                .values(
                    id=f"deferred:{uuid4()}",
                    status="deferred",
                    completed_at=timestamp(dt.datetime.now(dt.UTC), field="deferred_at"),
                    error=reason,
                    payload={
                        "outcome_category": outcome_category.value,
                        "available_at": available_at,
                        "retry_consumed": False,
                    },
                )
            )
            connection.execute(delete(worker_lease).where(worker_lease.c.id == attempt_id))
            connection.execute(
                update(worker).where(worker.c.id == claimed.worker_id).values(status="ready")
            )

    def _finish(
        self,
        claimed: ClaimedJob,
        *,
        completed_at: str,
        status: str,
        error: str | None,
        retry_at: str | None = None,
        outcome: Mapping[str, Any] | None = None,
    ) -> None:
        completed_at = timestamp(completed_at, field="completed_at")
        attempt_id = f"{claimed.job_id}:{claimed.attempt}"
        terminal_failure = False
        with self._transaction() as connection:
            current = self._owned_job(connection, claimed)
            terminal = int(current.attempts) >= int(current.max_attempts)
            terminal_failure = bool(retry_at and terminal and error)
            next_state = "dead_letter" if terminal else "pending"
            connection.execute(
                update(job)
                .where(job.c.id == claimed.job_id)
                .values(
                    state=next_state if retry_at else "completed",
                    available_at=timestamp(retry_at, field="retry_at")
                    if retry_at
                    else completed_at,
                    lease_owner=None,
                    lease_expires_at=None,
                    terminal_reason=error if terminal else None,
                )
            )
            connection.execute(
                update(job_attempt)
                .where(job_attempt.c.id == attempt_id)
                .values(
                    completed_at=completed_at,
                    status=status,
                    error=error,
                    payload=json_value(dict(outcome or {}), field="attempt outcome"),
                )
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
        if terminal_failure:
            self._emit_terminal_failure(
                job_id=claimed.job_id,
                job_name=claimed.name,
                attempt=claimed.attempt,
                error=error or "terminal job failure",
                observed_at=completed_at,
            )

    def _emit_terminal_failure(
        self,
        *,
        job_id: str,
        job_name: str,
        attempt: int,
        error: str,
        observed_at: str,
    ) -> None:
        try:
            self.alerts.emit(
                event_type="queue_terminal_failure",
                severity=AlertSeverity.CRITICAL,
                dedupe_key=f"queue:{job_id}:terminal",
                target=job_name,
                message="queue job reached its terminal retry limit",
                emitted_at=observed_at,
                payload={
                    "job_id": job_id,
                    "job_name": job_name,
                    "attempt": attempt,
                    "error": error[:500],
                },
                cooldown_seconds=0,
            )
        except Exception:
            pass

    def recover_expired(self, *, now: str) -> int:
        now = timestamp(now, field="now")
        terminal_failures: list[dict[str, Any]] = []
        with self._transaction() as connection:
            statement = (
                select(job.c.id, job.c.attempts, job.c.max_attempts)
                .where(job.c.state == "running", job.c.lease_expires_at < now)
                .order_by(job.c.id)
            )
            if connection.dialect.name == "postgresql":
                statement = statement.with_for_update(skip_locked=True)
            expired = connection.execute(statement).mappings().all()
            if not expired:
                return 0
            for row in expired:
                terminal = int(row["attempts"]) >= int(row["max_attempts"])
                if terminal:
                    terminal_failures.append(
                        {
                            "job_id": str(row["id"]),
                            "attempt": int(row["attempts"]),
                        }
                    )
                connection.execute(
                    update(job)
                    .where(job.c.id == row["id"])
                    .values(
                        state="dead_letter" if terminal else "pending",
                        lease_owner=None,
                        lease_expires_at=None,
                        terminal_reason="worker_lease_expired" if terminal else None,
                    )
                )
            connection.execute(
                update(worker_lease)
                .where(
                    worker_lease.c.job_id.in_([row["id"] for row in expired]),
                    worker_lease.c.status == "active",
                )
                .values(status="expired")
            )
            connection.execute(
                update(job_attempt)
                .where(
                    job_attempt.c.job_id.in_([row["id"] for row in expired]),
                    job_attempt.c.status == "running",
                )
                .values(completed_at=now, status="expired", error="worker_lease_expired")
            )
        for item in terminal_failures:
            with self.engine.connect() as connection:
                name = connection.execute(
                    select(job.c.name).where(job.c.id == item["job_id"])
                ).scalar_one()
            self._emit_terminal_failure(
                job_id=item["job_id"],
                job_name=str(name),
                attempt=item["attempt"],
                error="worker lease expired at retry limit",
                observed_at=now,
            )
        return len(expired)


def _restricted_job_insert(connection, values) -> bool | None:
    if connection.dialect.name != "postgresql":
        return None
    if connection.execute(
        text("SELECT has_table_privilege(current_user, 'job', 'INSERT')")
    ).scalar_one():
        return None
    return bool(
        connection.execute(
            text("SELECT submit_typed_research_job(CAST(:request AS jsonb))"),
            {"request": json.dumps(values)},
        ).scalar_one()
    )
