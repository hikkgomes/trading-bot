from __future__ import annotations

from sqlalchemy import insert

from src.data.database import (
    PlatformDatabase,
    account_snapshot,
    instrument,
    job,
    job_attempt,
    platform_schedule,
    promotion_event,
    reconciliation_event,
    risk_snapshot,
    universe,
    universe_member,
    universe_snapshot,
)
from src.observability.reports import DatabasePlatformReport
from src.services.alerting import SqlAlertService
from src.services.report_worker import DatabaseReportWorker
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


def test_report_exposes_latest_universe_attrition(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'universe-report.sqlite3'}")
    database.create_schema()
    with database.engine.begin() as connection:
        connection.execute(
            insert(universe).values(
                id="universe-report",
                created_at=NOW,
                payload={"market": "futures"},
            )
        )
        connection.execute(
            insert(instrument).values(
                id="instrument-report",
                venue="binance",
                market_type="futures",
                exchange_symbol="ETHUSDT",
                base_asset="ETH",
                quote_asset="USDT",
                settlement_asset="USDT",
                payload={},
            )
        )
        connection.execute(
            insert(universe_snapshot).values(
                id="universe-snapshot-report",
                universe_id="universe-report",
                observed_at=NOW,
                content_hash="sha256:" + "a" * 64,
                payload={},
            )
        )
        connection.execute(
            insert(universe_snapshot).values(
                id="universe-snapshot-report-old",
                universe_id="universe-report",
                observed_at="2026-08-30T10:00:00+00:00",
                content_hash="sha256:" + "b" * 64,
                payload={},
            )
        )
        connection.execute(
            insert(universe_member).values(
                id="universe-member-report",
                snapshot_id="universe-snapshot-report",
                instrument_id="instrument-report",
                eligible=False,
                reason_code="low_volume",
                payload={},
            )
        )
        connection.execute(
            insert(universe_member).values(
                id="universe-member-report-old-a",
                snapshot_id="universe-snapshot-report-old",
                instrument_id="instrument-report",
                eligible=True,
                reason_code=None,
                payload={},
            )
        )
        connection.execute(
            insert(universe_member).values(
                id="universe-member-report-old-b",
                snapshot_id="universe-snapshot-report-old",
                instrument_id="instrument-report",
                eligible=False,
                reason_code="old_exclusion",
                payload={},
            )
        )

    attrition = DatabasePlatformReport(database.engine).build()["research"]["universe_attrition"]

    assert attrition["snapshot_count"] == 2
    latest = attrition["latest_by_universe"]["universe-report"]
    assert latest["total_members"] == 1
    assert latest["eligible_members"] == 0
    assert latest["excluded_members"] == 1
    assert latest["exclusion_reasons"] == {"low_volume": 1}
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


def test_report_only_alerts_on_latest_missing_risk_state(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-report.sqlite3'}")
    database.create_schema()
    with database.engine.begin() as connection:
        connection.execute(
            risk_snapshot.insert().values(
                id="risk-old",
                created_at="2026-08-31T09:00:00+00:00",
                payload={
                    "kind": "canonical_portfolio_risk_state",
                    "product_id": "active_income",
                    "risk_data_available": False,
                    "risk_data_missing": ["beta:ETHUSDT"],
                },
            )
        )
        connection.execute(
            risk_snapshot.insert().values(
                id="risk-new",
                created_at=NOW,
                payload={
                    "kind": "canonical_portfolio_risk_state",
                    "product_id": "active_income",
                    "risk_data_available": True,
                    "risk_data_missing": [],
                },
            )
        )

    assert DatabasePlatformReport(database.engine)._missing_risk_data()["count"] == 0
    database.dispose()


def test_report_worker_alerts_on_funnel_and_live_safety_slis(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'sli-alerts.sqlite3'}")
    database.create_schema()
    alerts = SqlAlertService(database.engine, default_cooldown_seconds=0)
    worker = DatabaseReportWorker(
        engine=database.engine,
        root=tmp_path / "reports",
        alerts=alerts,
        minimum_valid_screenings_before_progress=10,
    )
    worker.report.build = lambda **_kwargs: {
        "schema": "platform.operator_report/v1",
        "research": {
            "funnel": {
                "candidates_generated": 1,
                "candidates_compiled": 10,
                "candidate_age_by_state": {"queued": {"count": 1, "oldest_seconds": 120}},
                "missing_stage_dataset_count": 1,
                "missing_stage_datasets": {"development": 1},
                "candidates_without_job_or_reason": ["candidate-1"],
                "scheduled_versus_started_jobs": {"not_started": 1},
            }
        },
        "operations": {
            "slis": {
                "unresolved_recovery_count": 1,
                "unresolved_recovery": {"count": 1},
                "stale_account_authority": {"count": 1},
                "stale_market_data": {"count": 1},
                "missing_risk_data": {"count": 1, "snapshots": []},
                "execution_authority_conflicts": {"conflict": True},
            }
        },
        "trading": {},
        "products": {},
    }

    result = worker.run_once(now=NOW)

    assert result["reason_code"] == "operator_report_written"
    assert {record.event_type for record in alerts.events()} == {
        "candidate_funnel_stalled",
        "research_dataset_missing",
        "research_candidate_without_job",
        "research_schedule_not_started",
        "live_recovery_required",
        "account_authority_stale",
        "market_data_stale",
        "risk_state_data_missing",
        "execution_authority_conflict",
    }
    assert len(result["alert_ids"]) == 9
    database.dispose()


def test_scheduler_process_report_keeps_diagnostics_separate_and_replaces_latest(
    tmp_path, monkeypatch
) -> None:
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy import select
    from test_platform_candidate_lifecycle import _bundle, _candidate

    from src.data.database import experiment, experiment_run, strategy_definition, strategy_version
    from src.domain._codec import canonical_hash
    from src.research.economic_series import ECONOMIC_ENGINE_VERSION
    from src.services.process import _build_trading_work
    from src.services.scheduler import PlatformScheduler
    from src.services.supervisor import _service_work

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'report-pipeline.sqlite3'}")
    database.create_schema()
    now = NOW
    bundle = _bundle(database)
    regular_candidate = _candidate(database, bundle)
    diagnostic_definition = canonical_hash("definition-diagnostic-report-definition")
    diagnostic_version = canonical_hash("definition-diagnostic-report-version")
    with database.engine.begin() as connection:
        candidate_metadata = dict(
            connection.execute(
                select(experiment.c.metadata).where(
                    experiment.c.id == regular_candidate.candidate_id
                )
            ).scalar_one()
        )
        candidate_metadata.update(
            {"promotable": False, "blocked_reason": "waiting for fixture data"}
        )
        connection.execute(
            experiment.update()
            .where(experiment.c.id == regular_candidate.candidate_id)
            .values(metadata=candidate_metadata)
        )
    with database.engine.begin() as connection:
        connection.execute(
            strategy_definition.insert().values(
                id=diagnostic_definition,
                identity="definition-diagnostic-report",
                product_id="active_income",
                source_type="generated_dsl",
                source_hash=diagnostic_definition,
                definition={"metadata": {"diagnostic": True}},
            )
        )
        connection.execute(
            strategy_version.insert().values(
                id=diagnostic_version,
                definition_id=diagnostic_definition,
                version="1",
                created_at=now,
                payload={},
            )
        )
        for experiment_id, version_id, metadata, dataset_hashes, bundle_id in (
            (
                "experiment-diagnostic-candidate",
                regular_candidate.definition.strategy_version_id,
                {"diagnostic": True, "promotable": False},
                list(regular_candidate.dataset_snapshot_hashes),
                bundle.bundle_id,
            ),
            ("definition-diagnostic-candidate", diagnostic_version, {}, [], None),
        ):
            connection.execute(
                experiment.insert().values(
                    id=experiment_id,
                    strategy_version_id=version_id,
                    provider="fixture",
                    state="queued",
                    submitted_at=now,
                    dataset_snapshot_hashes=dataset_hashes,
                    dataset_bundle_id=bundle_id,
                    metadata=metadata,
                )
            )
        for run_id, candidate_id in (
            ("regular-run", regular_candidate.candidate_id),
            ("diagnostic-run", "experiment-diagnostic-candidate"),
        ):
            connection.execute(
                experiment_run.insert().values(
                    id=run_id,
                    created_at=now,
                    payload={
                        "candidate_id": candidate_id,
                        "run_name": "screening",
                        "evidence": {},
                        "metrics": {"net_return": 0.01},
                        "receipt": {
                            "candidate_id": candidate_id,
                            "dataset_snapshot_ids": ["sha256:" + "c" * 64],
                            "executor_version": ECONOMIC_ENGINE_VERSION,
                            "input_hash": "sha256:" + "d" * 64,
                        },
                    },
                )
            )
    parquet_root = tmp_path / "data"
    parquet_root.mkdir()
    (tmp_path / "runtime").mkdir()
    reports_root = tmp_path / "runtime" / "reports"
    reports_root.mkdir(mode=0o755)
    config = SimpleNamespace(
        paths={
            "parquet": str(parquet_root),
            "artefacts": str(tmp_path / "data" / "artefacts"),
            "backups": str(tmp_path / "missing-backups"),
        },
        alerting={"minimum_valid_screenings_before_progress": 10},
        backup={"maximum_age_seconds": 172_800, "minimum_free_bytes": 0},
    )
    alerts = SqlAlertService(database.engine, default_cooldown_seconds=0)
    scheduler = PlatformScheduler(engine=database.engine, products={}, node_id="fixture")
    scheduler.run_once(now=now)
    scheduler.run_once(now="2026-08-31T10:05:00+00:00")
    with database.engine.connect() as connection:
        reporting_jobs = connection.execute(
            select(job.c.id, job.c.state).where(job.c.name == "reporting")
        ).all()
    assert len(reporting_jobs) == 1
    assert reporting_jobs[0].id.startswith("scheduled:reporting:global:")
    work = _build_trading_work(
        database=database,
        config=config,
        node_id="fixture",
        configuration={},
        alerts=alerts,
        selected_services=("report-worker",),
    )
    queued_result = work["report-worker"]()
    assert queued_result["reason_code"] == "operator_report_written"
    latest_path = Path(queued_result["path"])
    assert latest_path.name == "latest.json"
    assert latest_path.parent.stat().st_mode & 0o777 == 0o700
    assert latest_path.stat().st_mode & 0o777 == 0o600
    report = json.loads(latest_path.read_text())
    research = report["research"]
    assert research["diagnostic_candidate_count"] == 2
    assert set(research["diagnostic_candidate_ids"]) == {
        "experiment-diagnostic-candidate",
        "definition-diagnostic-candidate",
    }
    assert [item["id"] for item in research["candidate_queue"]] == [regular_candidate.candidate_id]
    funnel = research["funnel"]
    assert funnel["candidates_generated"] == 1
    assert funnel["diagnostic_candidates_excluded"] == 2
    assert funnel["unique_measured_evaluations"] == 1
    assert funnel["diagnostic_run_count"] == 1
    assert funnel["candidates_without_job_or_reason"] == []
    assert alerts.events()[0].event_type == "backup_failure_or_stale"

    from src.services import supervisor

    monkeypatch.setattr(supervisor, "utc_now", lambda: now)
    direct_args = SimpleNamespace(service="report-worker", once=True, node="fixture")
    direct_work = _service_work(
        direct_args,
        {
            "database": database,
            "config": config,
            "split": {},
            "runtime": SimpleNamespace(),
            "alerts": alerts,
            "control_plane": SimpleNamespace(),
        },
    )
    direct_result = direct_work()
    assert direct_result["reason_code"] == "operator_report_written"
    assert direct_result["path"] == queued_result["path"]
    assert list(latest_path.parent.iterdir()) == [latest_path]
    for terminal_state in (
        "screening_rejected",
        "infrastructure_exhausted",
        "unsupported",
        "protected",
    ):
        with database.engine.begin() as connection:
            connection.execute(
                experiment.update()
                .where(experiment.c.id == regular_candidate.candidate_id)
                .values(state=terminal_state, metadata={})
            )
        direct_work()
        terminal_report = json.loads(latest_path.read_text())["research"]
        assert terminal_report["candidate_queue"] == []
        assert terminal_report["funnel"]["candidates_without_job_or_reason"] == []
    database.dispose()


def test_report_storage_inventory_is_bounded_and_metrics_name_its_scope(tmp_path) -> None:
    from src.observability.metrics import DatabaseMetricsProvider
    from src.observability.reports import _storage_health

    data_root = tmp_path / "data"
    data_root.mkdir()
    spool_root = tmp_path / "events" / "spool"
    pending = spool_root / "pending"
    pending.mkdir(parents=True)
    for index in range(260):
        (pending / f"segment-{index:03d}.jsonl").write_text("{}\n")

    storage = _storage_health(data_root, spool_root)
    inventory = storage["spool_inventory"]
    assert inventory["scope"] == "bounded_inventory_sample"
    assert inventory["entry_limit_per_directory"] == 256
    assert inventory["files_observed"] == 256
    assert inventory["truncated"] is True
    rendered = DatabaseMetricsProvider(lambda: {"operations": {"storage_health": storage}}).render()
    assert (
        'platform_storage_filesystem_inode_stats_available{scope="backing_filesystem"}' in rendered
    )
    assert 'platform_storage_spool_files_observed{scope="bounded_inventory_sample"} 256' in rendered
    assert (
        'platform_storage_spool_inventory_truncated{scope="bounded_inventory_sample"} 1' in rendered
    )
