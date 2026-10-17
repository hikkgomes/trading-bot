"""Aggregate queue history in SQL without loading every job payload into memory."""

from sqlalchemy import func, literal_column, select

from src.data.database import job, job_attempt
from src.domain._codec import to_primitive

RESOURCE_DEFERRAL_DETAIL_LIMIT = 100


def queue_progress(engine, *, now):
    with engine.connect() as connection:
        states = dict(
            connection.execute(select(job.c.state, func.count()).group_by(job.c.state)).all()
        )
        eligible = (job.c.state == "pending") & (job.c.available_at <= now)
        oldest = (
            connection.execute(
                select(job.c.id.label("job_id"), job.c.available_at)
                .where(eligible)
                .order_by(job.c.available_at, job.c.id)
                .limit(1)
            )
            .mappings()
            .first()
        )
        failed = connection.execute(
            select(func.count())
            .select_from(job_attempt)
            .where(job_attempt.c.status.in_(("failed", "expired")))
        ).scalar_one()
        terminal = (
            connection.execute(
                select(job.c.id.label("job_id"), job.c.name, job.c.attempts, job.c.terminal_reason)
                .where(job.c.state == "dead_letter")
                .order_by(job.c.available_at.desc())
                .limit(100)
            )
            .mappings()
            .all()
        )
        return {
            "history_retention": {
                "completed_public_market_batch_days": 7,
                "cleanup_batch_limit": 1_000,
                "cleanup_interval_minutes": 5,
                "other_job_and_attempt_history": "retained",
                "aggregate_scope": (
                    "queue and attempt totals exclude expired tagged public market batch receipts"
                ),
            },
            **{
                f"jobs_{label}": states.get(state, 0)
                for label, state in (
                    ("waiting", "pending"),
                    ("running", "running"),
                    ("completed", "completed"),
                    ("dead_letter", "dead_letter"),
                )
            },
            "jobs_by_state": states,
            "jobs_failed_attempts": failed,
            "jobs_terminal_failures": to_primitive([dict(row) for row in terminal]),
            "terminal_failure_detail_limit": 100,
            "eligible_work_count": connection.execute(
                select(func.count()).select_from(job).where(eligible)
            ).scalar_one(),
            "oldest_eligible_work": to_primitive(dict(oldest)) if oldest else None,
            "scheduled_versus_started_jobs": _scheduled_progress(connection),
            **_attempt_progress(connection),
        }


def _attempt_progress(connection):
    outcome = job_attempt.c.outcome_category
    outcomes = dict(connection.execute(select(outcome, func.count()).group_by(outcome)).all())
    reason = func.coalesce(job_attempt.c.error, "unspecified")
    deferral_groups = (
        select(reason.label("reason"), func.count().label("attempt_count"))
        .where(job_attempt.c.status == "deferred", outcome == "blocked_infrastructure")
        .group_by(reason)
        .subquery()
    )
    ranked_deferrals = select(
        deferral_groups.c.reason,
        deferral_groups.c.attempt_count,
        func.sum(deferral_groups.c.attempt_count).over().label("total_attempts"),
        func.count().over().label("reason_count"),
        func.row_number()
        .over(
            order_by=(
                deferral_groups.c.attempt_count.desc(),
                deferral_groups.c.reason,
            )
        )
        .label("rank"),
    ).subquery()
    rows = connection.execute(
        select(
            ranked_deferrals.c.reason,
            ranked_deferrals.c.attempt_count,
            ranked_deferrals.c.total_attempts,
            ranked_deferrals.c.reason_count,
        )
        .where(ranked_deferrals.c.rank <= RESOURCE_DEFERRAL_DETAIL_LIMIT)
        .order_by(ranked_deferrals.c.rank)
    ).all()
    deferrals = {row.reason: int(row.attempt_count) for row in rows}
    deferral_count = int(rows[0].total_attempts) if rows else 0
    deferral_reason_count = int(rows[0].reason_count) if rows else 0
    displayed_deferral_count = sum(deferrals.values())
    return {
        "attempt_count": sum(outcomes.values()),
        "attempt_outcomes": outcomes,
        "resource_deferrals": deferrals,
        "resource_deferral_count": deferral_count,
        "resource_deferral_detail_limit": RESOURCE_DEFERRAL_DETAIL_LIMIT,
        "resource_deferral_omitted_count": deferral_count - displayed_deferral_count,
        "resource_deferral_omitted_reason_count": deferral_reason_count - len(deferrals),
    }


def _scheduled_progress(connection):
    job_suffix = func.substr(job.c.id, 11)
    scheduled_family = (
        func.split_part(job.c.id, ":", 2)
        if connection.dialect.name == "postgresql"
        else func.substr(job_suffix, 1, func.instr(job_suffix, ":") - 1)
    )
    scheduled_counts = (
        select(scheduled_family.label("schedule"), func.count().label("scheduled"))
        .where(literal_column("job.id LIKE 'scheduled:%'"))
        .group_by(scheduled_family)
        .cte("scheduled_counts")
    )
    attempted_scheduled_jobs = (
        select(job_attempt.c.job_id.label("job_id"))
        .where(literal_column("job_attempt.job_id LIKE 'scheduled:%'"))
        .distinct()
        .cte("attempted_scheduled_jobs")
    )
    attempt_suffix = func.substr(attempted_scheduled_jobs.c.job_id, 11)
    attempt_family = (
        func.split_part(attempted_scheduled_jobs.c.job_id, ":", 2)
        if connection.dialect.name == "postgresql"
        else func.substr(attempt_suffix, 1, func.instr(attempt_suffix, ":") - 1)
    )
    started_counts = (
        select(attempt_family.label("schedule"), func.count().label("started"))
        .group_by(attempt_family)
        .cte("started_counts")
    )
    rows = (
        connection.execute(
            select(
                scheduled_counts.c.schedule,
                scheduled_counts.c.scheduled,
                func.coalesce(started_counts.c.started, 0).label("started"),
            )
            .select_from(
                scheduled_counts.outerjoin(
                    started_counts, started_counts.c.schedule == scheduled_counts.c.schedule
                )
            )
            .order_by(scheduled_counts.c.schedule)
        )
        .mappings()
        .all()
    )
    grouped = {row["schedule"]: {key: row[key] for key in ("scheduled", "started")} for row in rows}
    scheduled_count = sum(row["scheduled"] for row in rows)
    started_count = sum(row["started"] for row in rows)
    return {
        "scheduled": scheduled_count,
        "started": started_count,
        "not_started": scheduled_count - started_count,
        "by_schedule": dict(sorted(grouped.items())),
    }


def active_candidate_jobs(engine):
    candidate = job.c.payload["candidate_id"].as_string()
    with engine.connect() as connection:
        return set(
            connection.execute(
                select(candidate)
                .where(job.c.state.in_(("pending", "running")), candidate.is_not(None))
                .distinct()
            ).scalars()
        )
