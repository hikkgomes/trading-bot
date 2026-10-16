"""Aggregate queue history in SQL without loading every job payload into memory."""

from sqlalchemy import case, exists, func, select

from src.data.database import job, job_attempt
from src.domain._codec import to_primitive


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
    outcome = func.coalesce(
        job_attempt.c.payload["outcome_category"].as_string(),
        case(
            (job_attempt.c.status.in_(("failed", "expired")), "blocked_infrastructure"),
            else_="unclassified",
        ),
    )
    outcomes = dict(connection.execute(select(outcome, func.count()).group_by(outcome)).all())
    reason = func.coalesce(job_attempt.c.error, "unspecified")
    deferrals = dict(
        connection.execute(
            select(reason, func.count())
            .where(job_attempt.c.status == "deferred", outcome == "blocked_infrastructure")
            .group_by(reason)
        ).all()
    )
    return {
        "attempt_count": sum(outcomes.values()),
        "attempt_outcomes": outcomes,
        "resource_deferrals": deferrals,
    }


def _scheduled_progress(connection):
    suffix = func.substr(job.c.id, 11)
    schedule = (
        func.split_part(job.c.id, ":", 2)
        if connection.dialect.name == "postgresql"
        else func.substr(suffix, 1, func.instr(suffix, ":") - 1)
    )
    started = exists(select(job_attempt.c.id).where(job_attempt.c.job_id == job.c.id))
    rows = (
        connection.execute(
            select(
                schedule.label("schedule"),
                func.count().label("scheduled"),
                func.sum(case((started, 1), else_=0)).label("started"),
            )
            .where(job.c.id.startswith("scheduled:"))
            .group_by(schedule)
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
