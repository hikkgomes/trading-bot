"""Durable, budgeted development rounds closed before protected selection."""

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any

from sqlalchemy import column, insert, select, table, text, update
from sqlalchemy.engine import Connection

from src.data.database import (
    dataset_bundle,
    experiment,
    generation_feedback,
    research_epoch,
    research_round,
    thesis_trial,
    validation_stage,
)
from src.domain._codec import canonical_hash
from src.research.economic_series import ECONOMIC_ENGINE_VERSION

_ADAPTIVE_CANDIDATE = table(
    "adaptive_candidate", *(column(item.name, item.type) for item in experiment.columns)
)
CATALOGUE_BATCH_SIZE = 12
CATALOGUE_EPOCH_SHARE = 0.8


def _candidate_source(connection: Connection):
    return _ADAPTIVE_CANDIDATE if connection.dialect.name == "postgresql" else experiment


class EpochState(StrEnum):
    ACTIVE = "active"
    WAITING_FOR_FEEDBACK = "waiting_for_feedback"
    WAITING_FOR_BUDGET = "waiting_for_budget"
    EXHAUSTED = "exhausted"
    CLOSED_FOR_SELECTION = "closed_for_selection"


def lock_epoch(connection: Connection, product_id: str) -> None:
    if connection.dialect.name == "postgresql":
        connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:scope))"),
            {"scope": f"selection-cohort:{product_id}"},
        )


def assert_epoch_open(connection: Connection, bundle_id: str | None) -> None:
    if bundle_id is None:
        return
    state = connection.execute(
        select(research_epoch.c.state).where(research_epoch.c.dataset_bundle_id == bundle_id)
    ).scalar_one_or_none()
    if state in {EpochState.EXHAUSTED, EpochState.CLOSED_FOR_SELECTION}:
        raise ValueError("research epoch is closed to new adaptive candidates")


def assert_epoch_closed(connection: Connection, bundle_id: str | None) -> None:
    state = connection.execute(
        select(research_epoch.c.state).where(research_epoch.c.dataset_bundle_id == bundle_id)
    ).scalar_one_or_none()
    if state not in {EpochState.EXHAUSTED, EpochState.CLOSED_FOR_SELECTION}:
        from src.research.cohorts import CohortPending

        raise CohortPending("development research epoch must close before protected freeze")


class ResearchRoundCoordinator:
    def __init__(self, engine):
        self.engine = engine

    def close_finite(self, bundle_id: str, *, product_id: str, candidate_ids, at: str) -> str:
        """Seal an explicitly declared finite campaign before protected selection."""
        declared = sorted(set(candidate_ids))
        if not declared:
            raise ValueError("finite campaign requires its complete declared candidate set")
        with self.engine.begin() as connection:
            lock_epoch(connection, product_id)
            candidates = _candidate_source(connection)
            actual_product = connection.execute(
                select(dataset_bundle.c.product_id).where(dataset_bundle.c.id == bundle_id)
            ).scalar_one()
            actual = sorted(
                connection.execute(
                    select(candidates.c.id).where(candidates.c.dataset_bundle_id == bundle_id)
                ).scalars()
            )
            if actual_product != product_id or actual != declared:
                raise ValueError("finite campaign declaration differs from registered trials")
            identity = canonical_hash({"finite_campaign": bundle_id, "candidate_ids": declared})
            statement = select(research_epoch).where(
                research_epoch.c.dataset_bundle_id == bundle_id
            )
            if connection.dialect.name == "postgresql":
                statement = statement.with_for_update()
            existing = connection.execute(statement).mappings().first()
            if existing is not None:
                return _seal_existing_finite_epoch(
                    connection, dict(existing), declared=declared, declaration_id=identity
                )
            connection.execute(
                insert(research_epoch).values(
                    id=identity,
                    product_id=product_id,
                    dataset_bundle_id=bundle_id,
                    state=EpochState.CLOSED_FOR_SELECTION.value,
                    round_cursor=0,
                    budget=len(declared),
                    consumed=len(declared),
                    created_at=at,
                    payload={
                        "engine_version": ECONOMIC_ENGINE_VERSION,
                        "declared_candidate_ids": declared,
                        "closure_reason": "finite_campaign",
                        "last_completion": None,
                    },
                )
            )
            return identity

    def claim(
        self, payload: Mapping[str, Any], *, now: str, epoch_budget: int, daily_budget: int
    ) -> dict[str, Any] | None:
        bundle_id = payload.get("dataset_bundle_id")
        if not bundle_id:
            return None
        producer = str(payload.get("producer", "adaptive"))
        if producer not in {"catalogue", "adaptive"}:
            raise ValueError("research producer must be catalogue or adaptive")
        with self.engine.begin() as connection:
            lock_epoch(connection, str(payload["product_id"]))
            epoch = self._epoch(connection, payload, now, epoch_budget)
            if epoch["state"] in {EpochState.EXHAUSTED, EpochState.CLOSED_FOR_SELECTION}:
                return None
            latest = (
                connection.execute(
                    select(research_round)
                    .where(research_round.c.epoch_id == epoch["id"])
                    .order_by(research_round.c.round_number.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if latest and latest["state"] == "pending":
                request = dict(latest["payload"]["request"])
                return request if request.get("producer") == producer else None
            if producer == "adaptive" and not _development_complete(connection, str(bundle_id)):
                self._state(connection, epoch, EpochState.WAITING_FOR_FEEDBACK)
                return None
            return self._next_round(
                connection, epoch, payload, now, daily_budget, producer=producer
            )

    def _epoch(self, connection, payload, now, budget):
        row = (
            connection.execute(
                select(research_epoch).where(
                    research_epoch.c.dataset_bundle_id == payload["dataset_bundle_id"]
                )
            )
            .mappings()
            .first()
        )
        if row:
            return dict(row)
        # Existing trials retain their consumption and are never submitted again.
        candidates = _candidate_source(connection)
        existing = list(
            connection.execute(
                select(candidates.c.id, candidates.c.provider).where(
                    candidates.c.dataset_bundle_id == payload["dataset_bundle_id"],
                )
            ).mappings()
        )
        existing_ids = [str(item["id"]) for item in existing]
        previous = connection.execute(
            select(research_epoch.c.payload)
            .where(research_epoch.c.product_id == payload["product_id"])
            .order_by(research_epoch.c.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        previous = dict(previous or {})
        catalogue_closed = bool(previous.get("catalogue_closed", False))
        catalogue_cursor = 0 if catalogue_closed else int(previous.get("catalogue_cursor", 0))
        catalogue_consumed = sum(
            item["provider"] == "registered_strategy_catalogue" for item in existing
        )
        identity = canonical_hash(
            {
                "product_id": payload["product_id"],
                "bundle_id": payload["dataset_bundle_id"],
                "engine_version": ECONOMIC_ENGINE_VERSION,
            }
        )
        row = {
            "id": identity,
            "product_id": payload["product_id"],
            "dataset_bundle_id": payload["dataset_bundle_id"],
            "state": EpochState.EXHAUSTED.value
            if len(existing_ids) >= budget
            else EpochState.ACTIVE.value,
            "round_cursor": 0,
            "budget": max(budget, len(existing_ids)),
            "consumed": len(existing_ids),
            "created_at": now,
            "payload": {
                "engine_version": ECONOMIC_ENGINE_VERSION,
                "adopted_candidate_ids": existing_ids,
                "catalogue_consumed": catalogue_consumed,
                "catalogue_cursor": catalogue_cursor,
                "catalogue_closed": False,
                "data_plan": payload.get("data_plan"),
                "last_completion": None,
            },
        }
        connection.execute(insert(research_epoch).values(**row))
        return row

    def _next_round(self, connection, epoch, payload, now, daily_budget, *, producer):
        remaining = epoch["budget"] - epoch["consumed"]
        if remaining <= 0:
            self._state(connection, epoch, EpochState.CLOSED_FOR_SELECTION)
            return None
        used_today = len(
            connection.execute(
                select(thesis_trial.c.id).where(
                    thesis_trial.c.claimed_at >= now[:10] + "T00:00:00+00:00",
                    thesis_trial.c.claimed_at <= now,
                )
            ).all()
        )
        daily_remaining = max(0, daily_budget - used_today)
        request_budget = int(payload.get("generation_budget", CATALOGUE_BATCH_SIZE))
        limits = [request_budget, remaining, daily_remaining]
        catalogue_used = int((epoch["payload"] or {}).get("catalogue_consumed", 0))
        catalogue_closed = bool((epoch["payload"] or {}).get("catalogue_closed", False))
        catalogue_cursor = int((epoch["payload"] or {}).get("catalogue_cursor", 0))
        if producer == "catalogue":
            catalogue_limit = int(epoch["budget"] * CATALOGUE_EPOCH_SHARE)
            limits.extend([CATALOGUE_BATCH_SIZE, max(0, catalogue_limit - catalogue_used)])
            if catalogue_closed:
                limits.append(0)
        budget = max(0, min(limits))
        if budget == 0:
            self._state(connection, epoch, EpochState.WAITING_FOR_BUDGET)
            return None
        cursor = epoch["round_cursor"] + 1
        feedback = list(
            connection.execute(
                select(generation_feedback.c.id)
                .where(generation_feedback.c.created_at <= now)
                .order_by(generation_feedback.c.id)
            ).scalars()
        )
        development = list(
            connection.execute(
                select(validation_stage.c.id)
                .where(
                    validation_stage.c.stage == "development",
                    validation_stage.c.evaluated_at <= now,
                    validation_stage.c.engine_version == ECONOMIC_ENGINE_VERSION,
                )
                .order_by(validation_stage.c.id)
            ).scalars()
        )
        request = {
            **payload,
            "epoch_id": epoch["id"],
            "round_number": cursor,
            "generation_budget": budget,
            "producer": producer,
            "catalogue_cursor": catalogue_cursor,
            "catalogue_closed": catalogue_closed,
            "feedback_watermark": canonical_hash(
                {"feedback": feedback, "development": development}
            ),
            "feedback_ids": feedback,
            "development_stage_ids": development,
            "feedback_at": now,
            "submitted_at": now,
        }
        round_id = canonical_hash({"epoch_id": epoch["id"], "round_number": cursor})
        request["round_id"] = round_id
        connection.execute(
            insert(research_round).values(
                id=round_id,
                epoch_id=epoch["id"],
                round_number=cursor,
                state="pending",
                created_at=now,
                payload={"request": request},
            )
        )
        connection.execute(
            update(research_epoch)
            .where(research_epoch.c.id == epoch["id"])
            .values(
                state=EpochState.ACTIVE.value,
                round_cursor=cursor,
            )
        )
        return request

    def complete(
        self, round_id: str, *, produce: Callable[[Connection, Mapping[str, Any]], dict[str, Any]]
    ) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = (
                connection.execute(select(research_round).where(research_round.c.id == round_id))
                .mappings()
                .one()
            )
            request = row["payload"]["request"]
            lock_epoch(connection, request["product_id"])
            row = (
                connection.execute(select(research_round).where(research_round.c.id == round_id))
                .mappings()
                .one()
            )
            if row["state"] == "completed":
                return dict(row["payload"]["result"])
            assert_epoch_open(connection, request["dataset_bundle_id"])
            candidates = _candidate_source(connection)
            before = set(
                connection.execute(
                    select(candidates.c.id).where(
                        candidates.c.dataset_bundle_id == request["dataset_bundle_id"]
                    )
                ).scalars()
            )
            result = produce(connection, request)
            if result.get("_defer"):
                return result
            return self._complete(connection, row, result, before)

    def _complete(self, connection, row, result, before):
        request = row["payload"]["request"]
        ids = tuple(dict.fromkeys(result["candidate_ids"]))
        if len(ids) > request["generation_budget"]:
            raise ValueError("research round exceeded its declared trial budget")
        candidates = _candidate_source(connection)
        registered = set(
            connection.execute(
                select(candidates.c.id).where(
                    candidates.c.id.in_(ids),
                    candidates.c.dataset_bundle_id == request["dataset_bundle_id"],
                )
            ).scalars()
        )
        if registered != set(ids):
            raise ValueError("research round contains unregistered candidates")
        epoch = (
            connection.execute(select(research_epoch).where(research_epoch.c.id == row["epoch_id"]))
            .mappings()
            .one()
        )
        fresh = sorted(set(ids) - before)
        consumed = epoch["consumed"]
        state = (
            EpochState.CLOSED_FOR_SELECTION
            if consumed >= epoch["budget"]
            else EpochState.WAITING_FOR_FEEDBACK
        )
        epoch_payload = dict(epoch["payload"] or {})
        request_producer = request.get("producer")
        if request_producer == "catalogue":
            next_cursor = int(result.get("next_catalogue_cursor", request["catalogue_cursor"]))
            catalogue_closed = bool(result.get("catalogue_closed", False))
            if next_cursor < int(request["catalogue_cursor"]):
                raise ValueError("catalogue cursor cannot move backwards")
            epoch_payload.update(
                {
                    "catalogue_cursor": next_cursor,
                    "catalogue_closed": catalogue_closed,
                }
            )
            result = {
                **result,
                "next_catalogue_cursor": next_cursor,
                "catalogue_closed": catalogue_closed,
            }
        result = {
            **result,
            "epoch_id": epoch["id"],
            "round_number": row["round_number"],
            "next_action": state.value,
            "new_candidate_ids": fresh,
            "reused_candidate_ids": sorted(set(ids) & before),
        }
        connection.execute(
            update(research_round)
            .where(research_round.c.id == row["id"])
            .values(
                state="completed",
                payload={**row["payload"], "result": result},
            )
        )
        connection.execute(
            update(research_epoch)
            .where(research_epoch.c.id == epoch["id"])
            .values(
                consumed=consumed,
                state=state.value,
                payload={
                    **epoch_payload,
                    "last_completion": row["id"],
                    "feedback_watermark": request["feedback_watermark"],
                },
            )
        )
        return result

    @staticmethod
    def _state(connection, epoch, state):
        connection.execute(
            update(research_epoch)
            .where(research_epoch.c.id == epoch["id"])
            .values(state=state.value)
        )


def _development_complete(connection, bundle_id: str) -> bool:
    candidates = _candidate_source(connection)
    trials = list(
        connection.execute(
            select(candidates.c.id, candidates.c.state).where(
                candidates.c.dataset_bundle_id == bundle_id
            )
        ).mappings()
    )
    stages = list(
        connection.execute(
            select(
                validation_stage.c.experiment_id,
                validation_stage.c.stage,
                validation_stage.c.accepted,
            ).where(
                validation_stage.c.experiment_id.in_([trial["id"] for trial in trials]),
                validation_stage.c.stage.in_(["screening", "development"]),
                validation_stage.c.engine_version == ECONOMIC_ENGINE_VERSION,
            )
        ).mappings()
    )
    completed = {
        row["experiment_id"]
        for row in stages
        if row["stage"] == "development" or not row["accepted"]
    }
    return all(trial["id"] in completed or trial["state"] == "legacy_import" for trial in trials)


def _seal_existing_finite_epoch(connection, epoch, *, declared, declaration_id: str) -> str:
    payload = dict(epoch["payload"] or {})
    if _same_finite_declaration(epoch, payload, declared, declaration_id):
        return str(epoch["id"])
    if not _finite_epoch_is_pristine(connection, epoch, payload, declared):
        raise ValueError("finite campaign cannot replace an existing adaptive epoch")
    payload.update(
        {
            "declared_candidate_ids": declared,
            "closure_reason": "finite_campaign",
            "finite_campaign_id": declaration_id,
        }
    )
    connection.execute(
        update(research_epoch)
        .where(research_epoch.c.id == epoch["id"])
        .values(state=EpochState.CLOSED_FOR_SELECTION.value, payload=payload)
    )
    return str(epoch["id"])


def _same_finite_declaration(epoch, payload, declared, declaration_id: str) -> bool:
    return (
        payload.get("closure_reason") == "finite_campaign"
        and payload.get("declared_candidate_ids") == declared
        and (payload.get("finite_campaign_id") == declaration_id or epoch["id"] == declaration_id)
    )


def _finite_epoch_is_pristine(connection, epoch, payload, declared) -> bool:
    untouched = (
        _has_admission_cursor(payload)
        and epoch["round_cursor"] == 0
        and epoch["state"]
        not in {EpochState.EXHAUSTED.value, EpochState.CLOSED_FOR_SELECTION.value}
        and not _has_research_round(connection, epoch["id"])
    )
    return untouched and _finite_trials_conserved(connection, epoch, declared)


def _has_admission_cursor(payload) -> bool:
    return all(
        key in payload for key in ("catalogue_cursor", "catalogue_closed", "catalogue_consumed")
    )


def _has_research_round(connection, epoch_id: str) -> bool:
    return (
        connection.execute(
            select(research_round.c.id).where(research_round.c.epoch_id == epoch_id).limit(1)
        ).first()
        is not None
    )


def _finite_trials_conserved(connection, epoch, declared) -> bool:
    trials = sorted(
        connection.execute(
            select(thesis_trial.c.candidate_id)
            .where(thesis_trial.c.candidate_id.in_(declared))
            .order_by(thesis_trial.c.candidate_id)
        ).scalars()
    )
    return trials == declared and epoch["consumed"] == len(declared)
