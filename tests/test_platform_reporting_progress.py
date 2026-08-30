from __future__ import annotations

from sqlalchemy import insert

from src.data.database import PlatformDatabase, promotion_event
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
