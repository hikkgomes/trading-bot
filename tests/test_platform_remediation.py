from __future__ import annotations

from sqlalchemy import select

from src.data.database import PlatformDatabase, job
from src.research.evaluation import EvidencePolicy, EvidenceStatus
from src.research.executors import _cross_symbol_stability, _portfolio_overlap
from src.services.scheduler import DatabaseJobQueue

NOW = "2026-08-30T10:00:00+00:00"


def test_job_retries_end_in_a_durable_dead_letter_state(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'queue.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="worker",
        node_id="node",
        role="research-worker",
        capabilities=("research",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="bounded",
        name="research",
        payload={"candidate": "bounded"},
        available_at=NOW,
        max_attempts=2,
    )

    first = queue.claim(worker_id="worker", now=NOW, lease_seconds=10)
    assert first is not None
    queue.fail(
        first,
        completed_at="2026-08-30T10:00:01+00:00",
        error="first failure",
        retry_at="2026-08-30T10:00:02+00:00",
    )
    second = queue.claim(worker_id="worker", now="2026-08-30T10:00:02+00:00", lease_seconds=10)
    assert second is not None and second.attempt == 2
    queue.fail(
        second,
        completed_at="2026-08-30T10:00:03+00:00",
        error="terminal failure",
        retry_at="2026-08-30T10:00:04+00:00",
    )

    with database.engine.connect() as connection:
        row = connection.execute(
            select(job.c.state, job.c.attempts, job.c.max_attempts, job.c.terminal_reason).where(
                job.c.id == "bounded"
            )
        ).one()
    assert row.state == "dead_letter"
    assert row.attempts == row.max_attempts == 2
    assert row.terminal_reason == "terminal failure"
    assert (
        queue.claim(worker_id="worker", now="2026-08-30T10:00:10+00:00", lease_seconds=10) is None
    )


def test_evidence_policy_preserves_applicability_and_thesis_scoped_controls() -> None:
    policy = EvidencePolicy()
    input_hash = "sha256:" + "a" * 64
    evidence = {
        "parameter_stability": {"status": "not_applicable", "passed": True},
        "cross_symbol_stability": {"status": "not_applicable", "passed": True},
        "portfolio_overlap": {"status": "not_applicable", "passed": True},
        "negative_control_results": {
            "placebo_event_times": {
                "passed": True,
                "observations": 10,
                "input_hash": input_hash,
            }
        },
    }
    statuses = policy.statuses(
        "development",
        evidence,
        ("placebo_event_times",),
    )
    assert statuses["parameter_stability"] is EvidenceStatus.NOT_APPLICABLE
    assert statuses["cross_symbol_stability"] is EvidenceStatus.NOT_APPLICABLE
    assert statuses["portfolio_overlap"] is EvidenceStatus.NOT_APPLICABLE
    robustness_statuses = policy.statuses(
        "robustness",
        evidence,
        ("placebo_event_times",),
    )
    assert robustness_statuses["negative_control_results"] is EvidenceStatus.PASS


def test_single_symbol_and_empty_portfolio_overlap_are_not_applicable() -> None:
    cross_symbol = _cross_symbol_stability(
        {"instrument_scope": ("BTCUSDT",)},
        [0.001, -0.0005],
    )
    overlap = _portfolio_overlap({}, [0.001, -0.0005])

    assert cross_symbol["status"] == "not_applicable"
    assert cross_symbol["passed"] is True
    assert overlap["status"] == "not_applicable"
    assert overlap["passed"] is True
