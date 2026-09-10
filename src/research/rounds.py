"""Durable, budgeted development rounds closed before protected selection."""

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any

from sqlalchemy import insert, select, text, update
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
from src.research.outcomes import terminal_trial_disposition


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
            actual_product = connection.execute(
                select(dataset_bundle.c.product_id).where(dataset_bundle.c.id == bundle_id)
            ).scalar_one()
            actual = sorted(
                connection.execute(
                    select(experiment.c.id).where(experiment.c.dataset_bundle_id == bundle_id)
                ).scalars()
            )
            if actual_product != product_id or actual != declared:
                raise ValueError("finite campaign declaration differs from registered trials")
            identity = canonical_hash({"finite_campaign": bundle_id, "candidate_ids": declared})
            existing = connection.execute(
                select(research_epoch.c.id).where(research_epoch.c.dataset_bundle_id == bundle_id)
            ).scalar_one_or_none()
            if existing is not None:
                if existing != identity:
                    raise ValueError("finite campaign cannot replace an existing adaptive epoch")
                return identity
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
                return dict(latest["payload"]["request"])
            if not _development_complete(connection, str(bundle_id)):
                self._state(connection, epoch, EpochState.WAITING_FOR_FEEDBACK)
                return None
            return self._next_round(connection, epoch, payload, now, daily_budget)

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
        existing = list(
            connection.execute(
                select(experiment.c.id).where(
                    experiment.c.dataset_bundle_id == payload["dataset_bundle_id"],
                )
            ).scalars()
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
            if len(existing) >= budget
            else EpochState.ACTIVE.value,
            "round_cursor": 0,
            "budget": max(budget, len(existing)),
            "consumed": len(existing),
            "created_at": now,
            "payload": {
                "engine_version": ECONOMIC_ENGINE_VERSION,
                "adopted_candidate_ids": existing,
                "data_plan": payload.get("data_plan"),
                "last_completion": None,
            },
        }
        connection.execute(insert(research_epoch).values(**row))
        return row

    def _next_round(self, connection, epoch, payload, now, daily_budget):
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
        budget = min(
            int(payload["generation_budget"]), remaining, max(0, daily_budget - used_today)
        )
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
            before = set(
                connection.execute(
                    select(experiment.c.id).where(
                        experiment.c.dataset_bundle_id == request["dataset_bundle_id"]
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
        registered = set(
            connection.execute(
                select(experiment.c.id).where(
                    experiment.c.id.in_(ids),
                    experiment.c.dataset_bundle_id == request["dataset_bundle_id"],
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
        consumed = epoch["consumed"] + len(fresh)
        state = (
            EpochState.EXHAUSTED
            if not fresh
            else (
                EpochState.CLOSED_FOR_SELECTION
                if consumed >= epoch["budget"]
                else EpochState.WAITING_FOR_FEEDBACK
            )
        )
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
                    **epoch["payload"],
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
    trials = list(
        connection.execute(
            select(experiment.c.id, experiment.c.state, experiment.c.metadata).where(
                experiment.c.dataset_bundle_id == bundle_id
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
    return all(
        trial["id"] in completed
        or terminal_trial_disposition(trial["state"], trial["metadata"]) is not None
        for trial in trials
    )
