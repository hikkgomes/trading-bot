"""Stable financial ownership across paper assignment lifecycle revisions."""

from sqlalchemy import select

from src.data.database import active_strategy_assignment
from src.domain._codec import timestamp


def financial_assignment_ids(engine, assignment_id: str, *, at: str) -> tuple[str, ...]:
    at = timestamp(at, field="assignment ownership time")
    with engine.connect() as connection:
        assignment = (
            connection.execute(
                select(active_strategy_assignment).where(
                    active_strategy_assignment.c.id == assignment_id,
                    active_strategy_assignment.c.assigned_at <= at,
                )
            )
            .mappings()
            .one_or_none()
        )
        if assignment is None:
            return ()
        fields = (
            "product_id",
            "portfolio_id",
            "sleeve_id",
            "strategy_version_id",
            "assignment_scope_id",
            "artefact_hash",
            "execution_mode",
        )
        return tuple(
            connection.execute(
                select(active_strategy_assignment.c.id)
                .where(
                    *(active_strategy_assignment.c[field] == assignment[field] for field in fields),
                    active_strategy_assignment.c.assigned_at <= at,
                )
                .order_by(active_strategy_assignment.c.assigned_at, active_strategy_assignment.c.id)
            ).scalars()
        )
