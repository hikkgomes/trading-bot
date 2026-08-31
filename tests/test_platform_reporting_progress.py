from __future__ import annotations

from sqlalchemy import insert

from src.data.database import (
    PlatformDatabase,
    account_snapshot,
    job,
    job_attempt,
    platform_schedule,
    promotion_event,
    reconciliation_event,
    risk_snapshot,
)
from src.observability.reports import DatabasePlatformReport
from src.services.scheduler import DatabaseJobQueue

NOW = "2026-08-31T10:00:00+00:00"


def test_report_counts_real_queue_states_and_terminal_attempts(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'report.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="worker",
        node_id="node",
        role="report-test",
        capabilities=("research",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="poison",
        name="research",
        payload={"candidate": "poison"},
        available_at=NOW,
        max_attempts=1,
    )
    claimed = queue.claim(worker_id="worker", now=NOW, lease_seconds=10)
    assert claimed is not None
    queue.fail(
        claimed,
        completed_at="2026-08-31T10:00:01+00:00",
        error="terminal failure",
        retry_at="2026-08-31T10:00:02+00:00",
    )
    report = DatabasePlatformReport(database.engine).build()
    funnel = report["research"]["funnel"]
    assert funnel["jobs_waiting"] == 0
    assert funnel["jobs_dead_letter"] == 1
    assert funnel["jobs_failed_attempts"] == 1
    assert funnel["jobs_terminal_failures"][0]["job_id"] == "poison"
    database.dispose()


def test_report_counts_only_accepted_state_advances_as_promotions(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'promotion-report.sqlite3'}")
    database.create_schema()
    with database.engine.begin() as connection:
        connection.execute(
            insert(promotion_event).values(
                id="rejected",
                created_at=NOW,
                payload={
                    "strategy_version_id": "strategy-1",
                    "prior_state": "registered",
                    "next_state": "registered",
                    "accepted": False,
                },
            )
        )
        connection.execute(
            insert(promotion_event).values(
                id="advanced",
                created_at="2026-08-31T10:00:01+00:00",
                payload={
                    "strategy_version_id": "strategy-1",
                    "prior_state": "registered",
                    "next_state": "forward_paper",
                    "accepted": True,
                },
            )
        )
    report = DatabasePlatformReport(database.engine).build()
    assert report["research"]["funnel"]["strategy_promotions"] == 1
    database.dispose()


def test_report_exposes_funnel_and_operational_sli_state(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'sli-report.sqlite3'}")
    database.create_schema()
    with database.engine.begin() as connection:
        connection.execute(
            insert(platform_schedule).values(
                id="platform:research",
                job_name="research",
                interval_seconds=60,
                next_run_at=NOW,
                last_run_at=NOW,
                last_job_id="scheduled:research:1",
                state="scheduled",
                payload={},
                created_at=NOW,
                updated_at=NOW,
            )
        )
        connection.execute(
            insert(job).values(
                id="scheduled:research:1",
                name="research",
                state="completed",
                priority=0,
                available_at=NOW,
                attempts=1,
                max_attempts=3,
                terminal_reason=None,
                producer_identity="platform-scheduler:test",
                content_hash="sha256:" + "1" * 64,
                payload={"candidate_id": "candidate-1"},
            )
        )
        connection.execute(
            insert(job_attempt).values(
                id="scheduled:research:1:1",
                job_id="scheduled:research:1",
                worker_id="worker",
                started_at=NOW,
                completed_at=NOW,
                status="completed",
                error=None,
                payload={},
            )
        )
        connection.execute(
            insert(account_snapshot).values(
                id="account-snapshot",
                account_id="account-1",
                observed_at="2026-08-31T09:00:00+00:00",
                source="user_stream_delta",
                content_hash="sha256:" + "2" * 64,
                payload={"account_state_known": False},
            )
        )
        connection.execute(
            insert(risk_snapshot).values(
                id="risk-snapshot",
                created_at="2026-08-31T09:00:00+00:00",
                payload={
                    "kind": "market_data_input",
                    "product_id": "active_income",
                    "instrument_id": "BTCUSDT",
                    "availability_time": "2026-08-31T09:00:00+00:00",
                },
            )
        )
        connection.execute(
            insert(reconciliation_event).values(
                id="recovery-plan",
                created_at=NOW,
                payload={
                    "record_type": "recovery_plan",
                    "plan": {"requires_operator_review": True, "actions": []},
                },
            )
        )

    report = DatabasePlatformReport(
        database.engine,
        account_stale_after_seconds=60,
        market_data_stale_after_seconds=60,
    ).build(now="2026-08-31T10:00:00+00:00")
    funnel = report["research"]["funnel"]
    slis = report["operations"]["slis"]
    assert funnel["scheduled_versus_started_jobs"] == {
        "scheduled": 1,
        "started": 1,
        "not_started": 0,
        "by_schedule": {"research": {"scheduled": 1, "started": 1}},
    }
    assert slis["unresolved_recovery_count"] == 1
    assert slis["stale_account_authority"]["count"] == 1
    assert slis["stale_market_data"]["count"] == 1
    database.dispose()
