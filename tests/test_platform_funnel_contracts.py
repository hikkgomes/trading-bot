import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import insert, select

from src.data.database import (
    PlatformDatabase,
    dataset_snapshot,
    holdout_outcome,
    job_attempt,
    legacy_research_archive,
    strategy_version,
    validation_stage,
)
from src.domain._codec import canonical_hash
from src.observability.reports import DatabasePlatformReport
from src.observability.research_progress import current_stages, measured_progress
from src.research.canonical import CanonicalEvidenceError, SqlHoldoutRepository
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.sqlite_import import import_sqlite_memory
from src.services.scheduler import DatabaseJobQueue

NOW = "2026-09-07T00:00:00+00:00"
LATER = "2026-09-07T00:01:00+00:00"


def test_registration_replay_keeps_original_evidence_without_recharging_trial(tmp_path):
    from test_platform_candidate_lifecycle import _bundle, _candidate

    from src.data.database import thesis_trial
    from src.research.coordinator import ResearchCoordinator
    from src.research.store import SqlResearchStore

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'registration.sqlite3'}")
    database.create_schema()
    candidate = _candidate(database, _bundle(database))
    repeated = replace(
        candidate,
        submitted_at=LATER,
        definition=replace(
            candidate.definition,
            version="new-display-label",
            metadata={
                **candidate.definition.metadata,
                "source_provenance": {"deployment_hash": canonical_hash("new-release")},
            },
        ),
    )
    assert repeated.candidate_id == candidate.candidate_id
    store = SqlResearchStore(database.engine)
    assert ResearchCoordinator(store).submit(repeated) == candidate.candidate_id
    store.save_candidate(repeated)
    assert store.get_candidate(candidate.candidate_id) == candidate
    with database.engine.connect() as connection:
        assert (
            len(
                connection.execute(
                    select(thesis_trial).where(
                        thesis_trial.c.candidate_id == candidate.candidate_id,
                    )
                ).all()
            )
            == 1
        )
    database.dispose()


def test_release_detects_untracked_runtime_source(tmp_path, monkeypatch):
    from src.services import release

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    source = tmp_path / "src" / "services"
    source.mkdir(parents=True)
    monkeypatch.setattr(release, "__file__", str(source / "release.py"))
    check_output = subprocess.check_output

    def git_output(command, **kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return "fixture-revision\n"
        return check_output(command, **kwargs)

    monkeypatch.setattr(release.subprocess, "check_output", git_output)
    release.observed_release.cache_clear()
    try:
        assert release.observed_release()["source_modified"] is False
        before = release.observed_release()["source_hash"]
        (source / "new_worker.py").write_text("value = 1\n")
        release.observed_release.cache_clear()
        assert release.observed_release()["source_modified"] is True
        assert release.observed_release()["source_hash"] != before
    finally:
        release.observed_release.cache_clear()


def test_previous_engine_and_invalidated_stages_do_not_count_as_current_progress():
    runs = [
        {
            "payload": {
                "run_name": "canonical:screening",
                "receipt": {
                    "candidate_id": "candidate",
                    "dataset_snapshot_ids": ["dataset"],
                    "input_hash": "inputs",
                    "executor_version": version,
                },
            }
        }
        for version in ("old-engine", ECONOMIC_ENGINE_VERSION, ECONOMIC_ENGINE_VERSION)
    ]
    progress = measured_progress(runs, [])
    assert progress["unique_measured_evaluations"] == 1
    assert progress["duplicate_measured_evaluations"] == 1
    assert progress["historical_measured_evaluations"] == 1
    stages = [
        {"id": "old", "engine_version": "old-engine"},
        {"id": "invalid", "engine_version": ECONOMIC_ENGINE_VERSION},
        {"id": "current", "engine_version": ECONOMIC_ENGINE_VERSION},
    ]
    assert current_stages(
        stages, [{"payload": {"kind": "validation_stage", "source_id": "invalid"}}]
    ) == [stages[-1]]


def test_legacy_import_preserves_original_cohort_and_exposure_on_replay(tmp_path):
    source = tmp_path / "legacy.sqlite3"
    fixture = Path(__file__).parent / "fixtures/legacy_exposure.sql"
    with sqlite3.connect(source) as connection:
        connection.executescript(fixture.read_text())
        behavior_hash = connection.execute("SELECT behavior_hash FROM strategies").fetchone()[0]
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'canonical.sqlite3'}")
    first = import_sqlite_memory(source, database)
    second = import_sqlite_memory(source, database)
    assert first.destination_hash == second.destination_hash
    with database.engine.connect() as connection:
        assert connection.execute(select(holdout_outcome)).first() is None
        assert connection.execute(select(validation_stage)).first() is None
        archived = connection.execute(
            select(legacy_research_archive.c.payload).where(
                legacy_research_archive.c.id == f"sqlite-exposure:{first.source_hash}"
            )
        ).scalar_one()
    assert archived["qualification_authority"] is False
    assert len(archived["source_tables"]["holdout_claim_scopes"]) == 1
    assert len(archived["source_tables"]["holdout_cohorts"]) == 1
    assert len(archived["source_tables"]["protected_intervals"]) == 1
    assert archived["strategy_dispositions"][0]["holdout_exposed_at"] is not None
    snapshot_id = canonical_hash({"corrected_engine": "same_exposed_dates"})
    with database.engine.begin() as connection:
        connection.execute(
            insert(dataset_snapshot).values(
                id=snapshot_id,
                created_at=NOW,
                payload={
                    "product_id": "active_income",
                    "instrument_scope": ["binance:futures:BTCUSDT:USDT"],
                    "interval": {
                        "start": "2025-06-10T00:00:00+00:00",
                        "end": "2025-06-20T00:00:00+00:00",
                    },
                },
            )
        )
        connection.execute(
            insert(strategy_version).values(
                id="new-version",
                definition_id=behavior_hash,
                version="new",
                created_at=NOW,
                payload={},
            )
        )
    with pytest.raises(CanonicalEvidenceError, match="legacy exposed interval"):
        SqlHoldoutRepository(database.engine).claim(
            strategy_version_id="new-version",
            data_snapshot_id=snapshot_id,
            cohort_id="new-cohort",
            source_hashes={"engine": canonical_hash("corrected-engine")},
            claimed_at=NOW,
        )
    database.dispose()


def test_resource_wait_survives_restart_without_spending_retry(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'funnel.sqlite3'}"
    database = PlatformDatabase(url)
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="research",
        node_id="test",
        role="research",
        capabilities=("research",),
        observed_at=NOW,
    )
    queue.enqueue(job_id="trial", name="research", payload={}, available_at=NOW, max_attempts=1)
    claimed = queue.claim(worker_id="research", now=NOW, lease_seconds=10)
    assert claimed is not None
    queue.defer(claimed, available_at=LATER, reason="heavy_compute_slot_unavailable")
    database.dispose()

    database = PlatformDatabase(url)
    queue = DatabaseJobQueue(database.engine)
    claimed = queue.claim(worker_id="research", now=LATER, lease_seconds=10)
    assert claimed is not None and claimed.attempt == 1
    queue.complete(claimed, completed_at=LATER)
    with database.engine.connect() as connection:
        statuses = connection.execute(select(job_attempt.c.status)).scalars().all()
    assert sorted(statuses) == ["completed", "deferred"]
    funnel = DatabasePlatformReport(database.engine).build(now=LATER)["research"]["funnel"]
    assert funnel["resource_deferrals"] == {"heavy_compute_slot_unavailable": 1}
    assert funnel["jobs_failed_attempts"] == 0
    assert funnel["jobs_completed"] == 1
    assert funnel["unique_measured_evaluations"] == 0
    assert funnel["candidates_evaluated"] == 0
    database.dispose()
