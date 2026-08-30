"""Database job queue with renewable worker leases and interruption recovery."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, insert, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from src.data.database import (
    dataset_bundle,
    dataset_snapshot,
    experiment,
    job,
    job_attempt,
    platform_schedule,
    strategy_artefact,
    universe_snapshot,
    worker,
    worker_lease,
)
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.research.canonical import SqlActiveStrategyAssignmentRepository
from src.research.store import SqlResearchStore
from src.research.theses import StrategyThesisFactory
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
    ScheduleSpec("promotion_evaluation", 900),
    ScheduleSpec("reporting", 900),
    ScheduleSpec("maintenance", 3600),
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
    ) -> None:
        self.engine = engine
        self.products = {str(key): dict(value) for key, value in products.items()}
        self.node_id = node_id
        self.bootstrap = bootstrap
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
                for product_id, product in sorted(self.products.items()):
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
            "promotion_evaluation": "promotion_evaluation",
            "reporting": "reporting",
            "maintenance": "maintenance",
        }.get(schedule_name)

    def _jobs(
        self,
        schedule_name: str,
        product_id: str,
        product: Mapping[str, Any],
        now: str,
        due_at: str,
    ) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        if schedule_name in {"reporting", "maintenance"}:
            if product_id != min(self.products):
                return ()
            return (
                (
                    f"scheduled:{schedule_name}:{product_id}:{due_at}",
                    schedule_name,
                    self._payload(schedule_name, product_id, product, now),
                ),
            )
        if schedule_name == "candidate_generation":
            catalogue = self._payload("register_strategy_catalogue", product_id, product, now)
            return (
                (
                    f"scheduled:{schedule_name}:{product_id}:catalogue:{due_at}",
                    "register_strategy_catalogue",
                    catalogue,
                ),
                (
                    f"scheduled:{schedule_name}:{product_id}:ml:{due_at}",
                    "register_ml_candidate",
                    self._ml_candidate_payload(product_id, product, catalogue),
                ),
            )
        if schedule_name in {"bounded_backtest", "event_replay", "ml_research"}:
            return self._research_jobs(schedule_name, product_id, now, due_at)
        if schedule_name == "universe_refresh" and self._pending_universe_refresh(product_id):
            return ()
        if schedule_name == "forward_paper_observation":
            return self._forward_jobs(product_id, now, due_at)
        if schedule_name == "promotion_evaluation":
            return self._promotion_jobs(product_id, now, due_at)
        if schedule_name != "candidate_evaluation":
            return (
                (
                    f"scheduled:{schedule_name}:{product_id}:{due_at}",
                    schedule_name,
                    self._payload(schedule_name, product_id, product, now),
                ),
            )
        jobs: list[tuple[str, str, dict[str, Any]]] = []
        for candidate in SqlResearchStore(self.engine).load_candidates():
            if candidate.definition.product != product_id or not candidate.dataset_snapshot_hashes:
                continue
            with self.engine.connect() as connection:
                state = connection.execute(
                    select(experiment.c.state).where(experiment.c.id == candidate.candidate_id)
                ).scalar_one()
            requested_stage = {
                "queued": "screening",
                "screening": "development",
                "development": "robustness",
                "robustness": "protected",
            }.get(str(state))
            if requested_stage is None:
                continue
            expected_role = {
                "screening": "screening",
                "development": "development",
                "robustness": "robustness",
                "protected": "protected_holdout",
            }[requested_stage]
            snapshot_id, snapshot_payload = self._candidate_snapshot(
                candidate.dataset_snapshot_hashes,
                role=expected_role,
            )
            if snapshot_id is None or snapshot_payload is None:
                continue
            jobs.append(
                (
                    f"scheduled:candidate_evaluation:{candidate.candidate_id}:{requested_stage}:{due_at}",
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
                    f"scheduled:{schedule_name}:{candidate.candidate_id}:{due_at}",
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

    def _ml_candidate_payload(
        self,
        product_id: str,
        product: Mapping[str, Any],
        catalogue: Mapping[str, Any],
    ) -> dict[str, Any]:
        universe = tuple(str(item) for item in catalogue["instrument_universe"])
        thesis = StrategyThesisFactory.default().build(
            name="frozen_logistic_model",
            family="machine_learning",
            product=product_id,
            instrument_universe=universe,
        )
        return {
            "identity": f"platform-scheduler:{product_id}:ml-classifier",
            "version": "diagnostic-v1",
            "family": "machine_learning",
            "product": product_id,
            "thesis_id": thesis.thesis_id,
            "lineage_id": canonical_hash(
                {
                    "thesis_id": thesis.thesis_id,
                    "provider": "platform-scheduler",
                    "product_id": product_id,
                }
            ),
            "provider": "platform-scheduler",
            "source_type": "machine_learning",
            "source_payload": {
                "kind": "ml_classifier",
                "model_name": "logistic_regression",
            },
            "dataset_snapshot_hashes": list(catalogue["dataset_snapshot_hashes"]),
            "submitted_at": str(catalogue["catalogue_submitted_at"]),
            "universe": {"symbols": list(universe)},
            "data_requirements": {"canonical_ml_rows": True},
            "feature_graph": {"required_nodes": ["return_1", "range_fraction"]},
            "position_model": {"kind": "volatility_scaled"},
            "execution_preferences": {"policy": "paper_only"},
            "risk_policy": {"id": str(product["risk_policy_id"])},
            "validation_policy": {"promotable": False},
            "metadata": {"diagnostic": True, "promotable": False},
        }

    def _candidate_snapshot(
        self,
        snapshot_ids: tuple[str, ...],
        *,
        role: str | None = None,
        required_payload: str | None = None,
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(dataset_snapshot.c.id, dataset_snapshot.c.payload).where(
                    dataset_snapshot.c.id.in_(snapshot_ids)
                )
            ).mappings()
            for row in rows:
                payload = row["payload"]
                if not isinstance(payload, Mapping):
                    continue
                if role is not None and str(payload.get("role") or "unspecified") != role:
                    continue
                data = payload.get("payload")
                if required_payload is not None and (
                    not isinstance(data, Mapping) or required_payload not in data
                ):
                    continue
                return str(row["id"]), payload
        return None, None

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
            "evaluator_version": "platform-scheduler/v1",
            "requested_stage": requested_stage,
            "evaluated_at": evaluated_at,
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
        for artefact in artefacts:
            metadata = artefact.get("metadata")
            if (
                artefact.get("product_id") != product_id
                or isinstance(metadata, Mapping)
                and metadata.get("promotable") is False
            ):
                continue
            strategy_version_id = str(artefact.get("strategy_version_id") or "")
            if not strategy_version_id:
                continue
            jobs.append(
                (
                    f"scheduled:promotion_evaluation:{strategy_version_id}:{due_at}",
                    "promotion_evaluation",
                    {
                        "strategy_version_id": strategy_version_id,
                        "requested_transition": "forward_paper",
                        "requested_capital": 0.0,
                        "evaluated_at": now,
                    },
                )
            )
        return tuple(jobs)

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
        jobs: list[tuple[str, str, dict[str, Any]]] = []
        for assignment in assignments:
            payload = assignment.get("payload")
            if isinstance(payload, Mapping) and payload.get("promotable") is False:
                continue
            instrument_id = str(assignment.get("instrument_id") or "")
            if not instrument_id:
                continue
            assignment_id = str(assignment["id"])
            jobs.append(
                (
                    f"scheduled:forward_paper_observation:{assignment_id}:{due_at}",
                    "forward_paper_observation",
                    {
                        "assignment_id": assignment_id,
                        "strategy_version_id": str(assignment["strategy_version_id"]),
                        "product_id": product_id,
                        "instrument_id": instrument_id,
                        "artefact_hash": str(assignment["artefact_hash"]),
                        "evaluation_time": now,
                    },
                )
            )
        return tuple(jobs)

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
        return {
            "reason_code": "platform_maintenance_completed",
            "job_id": claimed.job_id,
            "expired_jobs_recovered": recovered,
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
                "maximum_symbols": 100,
            }
        if schedule_name == "register_strategy_catalogue":
            with self.engine.connect() as connection:
                row = connection.execute(
                    select(dataset_bundle.c.id, dataset_bundle.c.created_at, dataset_bundle.c.payload)
                    .where(dataset_bundle.c.product_id == product_id)
                    .order_by(dataset_bundle.c.created_at.desc(), dataset_bundle.c.id.desc())
                    .limit(1)
                ).mappings().first()
            bundle_payload = row["payload"] if row is not None else None
            stage_ids = (
                bundle_payload.get("stage_snapshot_ids")
                if isinstance(bundle_payload, Mapping)
                else None
            )
            snapshot_ids = (
                list(dict.fromkeys(str(value) for value in stage_ids.values()))
                if isinstance(stage_ids, Mapping)
                else []
            )
            return {
                **base,
                "instrument_universe": self._universe_symbols(product, now),
                "dataset_snapshot_hashes": snapshot_ids,
                "dataset_bundle_id": str(row["id"]) if row is not None else None,
                "universe_snapshot_id": (
                    str(bundle_payload["universe_snapshot_id"])
                    if isinstance(bundle_payload, Mapping)
                    and bundle_payload.get("universe_snapshot_id")
                    else None
                ),
                "catalogue_submitted_at": (str(row["created_at"]) if row is not None else now),
            }
        return base

    def _universe_symbols(self, product: Mapping[str, Any], now: str) -> list[str]:
        universe_id = str(product.get("universe_id") or "")
        with self.engine.connect() as connection:
            row = connection.execute(
                select(universe_snapshot.c.payload)
                .where(
                    universe_snapshot.c.universe_id == universe_id,
                    universe_snapshot.c.observed_at <= now,
                )
                .order_by(universe_snapshot.c.observed_at.desc())
                .limit(1)
            ).scalar_one_or_none()
        if isinstance(row, Mapping) and isinstance(row.get("observations"), list):
            symbols = [
                str(item["instrument"]["exchange_symbol"])
                for item in row["observations"]
                if isinstance(item, Mapping)
                and item.get("reason_code") == "eligible"
                and isinstance(item.get("instrument"), Mapping)
                and item["instrument"].get("exchange_symbol")
            ]
            if symbols:
                return sorted(set(symbols))
        return [str(product.get("exchange_symbol") or "BTCUSDT")]


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
        max_attempts: int = 3,
    ) -> None:
        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
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
            "max_attempts": max_attempts,
            "terminal_reason": None,
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
        max_attempts: int = 3,
    ) -> bool:
        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
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
            "max_attempts": max_attempts,
            "terminal_reason": None,
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
            "payload": values["payload"],
            "producer_identity": values["producer_identity"],
            "content_hash": values["content_hash"],
            "max_attempts": values["max_attempts"],
        }
        if existing["state"] == "pending" and existing["attempts"] == 0:
            expected["available_at"] = values["available_at"]
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
                select(
                    job.c.state,
                    job.c.lease_owner,
                    job.c.attempts,
                    job.c.max_attempts,
                ).where(job.c.id == claimed.job_id)
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
                select(
                    job.c.state,
                    job.c.lease_owner,
                    job.c.attempts,
                    job.c.max_attempts,
                ).where(job.c.id == claimed.job_id)
            ).first()
            if (
                current is None
                or current.state != "running"
                or current.lease_owner != claimed.worker_id
            ):
                raise ValueError("worker no longer owns the job lease")
            terminal = int(current.attempts) >= int(current.max_attempts)
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
            expired = connection.execute(
                select(job.c.id, job.c.attempts, job.c.max_attempts).where(
                    job.c.state == "running", job.c.lease_expires_at < now
                )
            ).mappings().all()
            if not expired:
                return 0
            for row in expired:
                terminal = int(row["attempts"]) >= int(row["max_attempts"])
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
            return len(expired)
