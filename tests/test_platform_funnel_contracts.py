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
    deferred_claim = claimed
    database.dispose()

    database = PlatformDatabase(url)
    queue = DatabaseJobQueue(database.engine)
    claimed = queue.claim(worker_id="research", now=LATER, lease_seconds=10)
    assert claimed is not None and claimed.attempt == 1
    assert claimed.lease_token != deferred_claim.lease_token
    with pytest.raises(ValueError, match="no longer owns"):
        queue.complete(deferred_claim, completed_at=LATER)
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


def test_financial_contract_revalidation_has_distinct_immutable_measurement(tmp_path):
    from test_platform_candidate_lifecycle import _bundle, _candidate

    from src.data.database import evidence_invalidation, experiment_run
    from src.observability.research_progress import evaluation_identity
    from src.research.accounting import FUTURES_ACCOUNTING_RISK_CONTRACT
    from src.research.canonical import SqlValidationRepository
    from src.research.executors import execution_receipt

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'revalidation.sqlite3'}")
    database.create_schema()
    candidate = _candidate(database, _bundle(database))
    repository = SqlValidationRepository(database.engine)
    receipts = [
        execution_receipt(
            candidate=candidate,
            dataset_snapshot_ids=candidate.dataset_snapshot_hashes,
            executor_version=ECONOMIC_ENGINE_VERSION,
            financial_risk_contract=contract,
        )
        for contract in (None, FUTURES_ACCOUNTING_RISK_CONTRACT)
    ]
    identities = []
    for index, receipt in enumerate(receipts):
        run = {"candidate_id": candidate.candidate_id, "run_name": "screening", "receipt": receipt}
        with database.engine.begin() as connection:
            connection.execute(
                insert(experiment_run).values(id=str(index), created_at=NOW, payload=run)
            )
        stage_id = repository.append_stage(
            experiment_id=candidate.candidate_id,
            stage="screening",
            source_run_id=str(index),
            evaluated_at=NOW,
            accepted=True,
            reason_code=None,
            evidence={
                "evaluation_context": {
                    "financial_risk_contract": receipt.get("financial_risk_contract")
                }
            },
        )
        identities.append(evaluation_identity(run))
        if index == 0:
            with database.engine.begin() as connection:
                connection.execute(
                    insert(evidence_invalidation).values(
                        id="stage:" + stage_id,
                        created_at=NOW,
                        payload={"kind": "validation_stage", "source_id": stage_id},
                    )
                )
    with database.engine.connect() as connection:
        stored = connection.execute(select(validation_stage.c.evaluation_identity)).scalars().all()
    assert len(stored) == 2 and set(stored) == set(identities)
    assert len(set(identities)) == 2
    assert receipts[0]["input_hash"] != receipts[1]["input_hash"]
    assert len(repository.stages(candidate.candidate_id)) == 1
    database.dispose()


@pytest.mark.parametrize(
    "seal_contract,receipt_contract",
    [(None, None), ("current", None), (None, "current"), ("current", "current")],
)
def test_paper_eligibility_requires_contract_inside_sealed_holdout(
    tmp_path, seal_contract, receipt_contract
):
    from test_platform_candidate_lifecycle import _bundle, _candidate

    from src.data.database import experiment_run, holdout_claim, paper_eligibility_event
    from src.research.accounting import FUTURES_ACCOUNTING_RISK_CONTRACT
    from src.research.canonical import SqlValidationRepository
    from src.research.paper_eligibility import publish_paper_eligibility

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'paper-contract.sqlite3'}")
    database.create_schema()
    bundle = _bundle(database)
    candidate = _candidate(database, bundle)
    contract = FUTURES_ACCOUNTING_RISK_CONTRACT
    stage_ids = []
    for stage in ("screening", "development", "robustness", "protected"):
        with database.engine.begin() as connection:
            connection.execute(
                insert(experiment_run).values(
                    id=stage,
                    created_at=NOW,
                    payload={
                        "candidate_id": candidate.candidate_id,
                        "receipt": {
                            "executor_version": ECONOMIC_ENGINE_VERSION,
                            "dataset_snapshot_ids": [stage],
                            "financial_risk_contract": contract,
                        },
                    },
                )
            )
        stage_ids.append(
            SqlValidationRepository(database.engine).append_stage(
                experiment_id=candidate.candidate_id,
                stage=stage,
                source_run_id=stage,
                evaluated_at=NOW,
                accepted=True,
                reason_code=None,
                evidence={"evaluation_context": {"financial_risk_contract": contract}},
            )
        )
    sealed = {
        "financial_risk_contract": contract if seal_contract else None,
        "execution_receipt": {"financial_risk_contract": contract if receipt_contract else None},
    }
    artefact = {
        "artefact_hash": "paper",
        "product_id": "active_income",
        "created_at": LATER,
        "engine_version": ECONOMIC_ENGINE_VERSION,
        "strategy_version_id": candidate.definition.strategy_version_id,
        "dataset_snapshot_hashes": [bundle.stage_snapshot_ids["protected_holdout"]],
        "metadata": {"promotable": True, "candidate_id": candidate.candidate_id},
        "authoritative_evidence": {"validation_stage_ids": stage_ids, "holdout_claim_id": "claim"},
    }
    with database.engine.begin() as connection:
        connection.execute(
            insert(holdout_claim).values(
                id="claim",
                created_at=NOW,
                payload={
                    "strategy_version_id": candidate.definition.strategy_version_id,
                    "data_snapshot_id": bundle.stage_snapshot_ids["protected_holdout"],
                },
            )
        )
        connection.execute(
            insert(holdout_outcome).values(
                id="outcome",
                holdout_claim_id="claim",
                evaluated_at=NOW,
                accepted=True,
                outcome_hash=canonical_hash(sealed),
                payload=sealed,
            )
        )
        if seal_contract and receipt_contract:
            publish_paper_eligibility(connection, artefact, created_at=LATER)
            assert connection.execute(select(paper_eligibility_event.c.id)).scalar_one() == "paper"
        else:
            with pytest.raises(ValueError, match="financial feasibility"):
                publish_paper_eligibility(connection, artefact, created_at=LATER)
            assert connection.execute(select(paper_eligibility_event.c.id)).first() is None
    database.dispose()
