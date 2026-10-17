from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import insert, select

from src.accounting.ledger import LedgerEntry
from src.data.database import (
    PlatformDatabase,
    account_snapshot,
    accounting_entry,
    alert,
    alpha_forecast,
    decision_trace,
    experiment,
    experiment_run,
    holdout_claim,
    holdout_outcome,
    instrument,
    job,
    job_attempt,
    nav_snapshot,
    platform_schedule,
    promotion_event,
    reconciliation_event,
    risk_snapshot,
    strategy_definition,
    strategy_version,
    universe,
    universe_member,
    universe_snapshot,
    validation_result,
    validation_stage,
)
from src.domain._codec import canonical_hash
from src.observability.decision_trace import DecisionTrace, DecisionTraceStage
from src.observability.reports import DatabasePlatformReport
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.services.alerting import SqlAlertService
from src.services.report_worker import DatabaseReportWorker
from src.services.scheduler import DatabaseJobQueue

NOW = "2026-08-31T10:00:00+00:00"


def test_report_counts_real_queue_states_and_terminal_attempts(tmp_path) -> None:
    from unittest.mock import Mock

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
    for index in range(105):
        queue.enqueue(
            job_id=f"deferred-{index:03d}",
            name="report-deferral-test",
            payload={},
            available_at=NOW,
        )
        deferred = queue.claim(
            worker_id="worker",
            now=NOW,
            lease_seconds=10,
            names=("report-deferral-test",),
        )
        assert deferred is not None
        queue.defer(
            deferred,
            available_at="2030-01-01T00:00:00+00:00",
            reason=f"temporary failure {index:03d}",
        )
    report_builder = DatabasePlatformReport(database.engine)
    queue_progress = Mock(wraps=report_builder._queue_progress)
    report_builder._queue_progress = queue_progress
    report = report_builder.build(now=NOW)
    funnel = report["research"]["funnel"]
    assert funnel["jobs_waiting"] == 105
    assert funnel["jobs_dead_letter"] == 1
    assert funnel["jobs_failed_attempts"] == 1
    assert funnel["jobs_terminal_failures"][0]["job_id"] == "poison"
    assert queue_progress.call_count == 1
    progress = report["operations"]["queue_progress"]
    assert progress["attempt_count"] == 106
    assert progress["resource_deferral_count"] == 105
    assert progress["resource_deferral_detail_limit"] == 100
    assert len(progress["resource_deferrals"]) == 100
    assert sum(progress["resource_deferrals"].values()) == 100
    assert progress["resource_deferral_omitted_count"] == 5
    assert progress["resource_deferral_omitted_reason_count"] == 5
    assert funnel["resource_deferrals"] == progress["resource_deferrals"]
    assert progress["scheduled_versus_started_jobs"] == {
        "scheduled": 0,
        "started": 0,
        "not_started": 0,
        "by_schedule": {},
    }
    database.dispose()


def test_report_caps_details_and_keeps_full_research_and_financial_totals(tmp_path) -> None:
    from src.observability.reports import REPORT_SNAPSHOT_MAX_BYTES, read_persisted_platform_report

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'bounded-report.sqlite3'}")
    database.create_schema()
    count = 1001
    start = dt.datetime.fromisoformat(NOW)
    observed_at = [(start + dt.timedelta(seconds=index)).isoformat() for index in range(count)]
    experiments = []
    runs = []
    stages = []
    forecasts = []
    claims = []
    outcomes = []
    alerts = []
    active_income_entries = []
    btc_entries = []
    navs = []
    traces = []
    bulky_evidence = "x" * (REPORT_SNAPSHOT_MAX_BYTES + 1)
    previous_hash = "0" * 64
    fee = Decimal("0.000000001")
    for index, timestamp in enumerate(observed_at):
        candidate_id = f"candidate-{index:04d}"
        run_id = f"run-{index:04d}"
        experiments.append(
            {
                "id": candidate_id,
                "strategy_version_id": "report-version",
                "provider": "fixture",
                "state": "screening",
                "submitted_at": timestamp,
                "dataset_snapshot_hashes": [],
                "dataset_bundle_id": None,
                "metadata": {},
            }
        )
        runs.append({"id": run_id, "created_at": timestamp, "payload": {}})
        stage_payload = {
            "signal_frequency": 0.001,
            "candidate_correlation": 0.25,
        }
        if index == 0:
            stage_payload["evidence"] = {
                "signal_frequency": 0.001,
                "opaque_history": bulky_evidence,
            }
        elif 1 <= index <= 5:
            nested_values = {1: True, 2: False, 3: None, 4: "0.5"}
            stage_payload["evidence"] = (
                {"signal_frequency": nested_values[index]} if index in nested_values else {}
            )
        elif 6 <= index <= 9:
            fallback_values = {6: True, 7: False, 8: None, 9: "0.5"}
            stage_payload["signal_frequency"] = fallback_values[index]
        stages.append(
            {
                "id": f"stage-{index:04d}",
                "experiment_id": candidate_id,
                "stage": "screening",
                "source_run_id": run_id,
                "evaluated_at": timestamp,
                "state": "accepted",
                "accepted": True,
                "reason_code": None,
                "evidence_hash": f"sha256:{index:064x}",
                "evaluation_identity": None,
                "engine_version": ECONOMIC_ENGINE_VERSION,
                "outcome_category": "accepted",
                "payload": stage_payload,
            }
        )
        trace = DecisionTrace.start(
            event_id=f"blocked-{index:04d}",
            instrument_id="BTCUSDT",
            evaluated_at=timestamp,
        ).block(DecisionTraceStage.DATA_AVAILABLE, reason_code="market_data_stale")
        trace_payload = trace.to_dict()
        traces.append(
            {
                "id": canonical_hash(trace_payload),
                "event_id": trace.event_id,
                "instrument_id": trace.instrument_id,
                "evaluated_at": trace.evaluated_at,
                "first_blocked_stage": trace.first_blocked_stage,
                "payload": trace_payload,
            }
        )
        forecasts.append(
            {"id": f"forecast-{index:04d}", "created_at": timestamp, "payload": {"index": index}}
        )
        claim_id = f"claim-{index:04d}"
        claims.append({"id": claim_id, "created_at": timestamp, "payload": {"index": index}})
        outcomes.append(
            {
                "id": f"holdout-{index:04d}",
                "holdout_claim_id": claim_id,
                "evaluated_at": timestamp,
                "accepted": True,
                "outcome_hash": f"holdout-hash-{index:04d}",
                "payload": {"index": index},
            }
        )
        alerts.append(
            {
                "id": f"alert-{index:04d}",
                "created_at": timestamp,
                "payload": {"event_type": "notice"},
            }
        )
        active_income_entries.append(
            {
                "id": f"active-entry-{index:04d}",
                "product_id": "active_income",
                "sequence": index,
                "created_at": timestamp,
                "entry_hash": f"active-hash-{index:04d}",
                "payload": {
                    "product_id": "active_income",
                    "metadata": {"kind": "realised_pnl", "pnl_effect": "-0.000000001"},
                },
            }
        )
        btc_postings = (
            {"assets:btc": Decimal("1"), "equity:capital": Decimal("-1")}
            if index == 0
            else {"expense:fees": fee, "assets:btc": -fee}
        )
        btc_metadata = (
            {"kind": "capital", "pnl_effect": "0"}
            if index == 0
            else {"kind": "fee", "pnl_effect": str(-fee)}
        )
        entry = LedgerEntry(
            entry_id=f"btc-entry-{index:04d}",
            product_id="btc_accumulation",
            accounting_asset="BTC",
            occurred_at=timestamp,
            postings=btc_postings,
            metadata=btc_metadata,
            previous_hash=previous_hash,
        )
        previous_hash = entry.entry_hash
        btc_entries.append(
            {
                "id": entry.entry_id,
                "product_id": entry.product_id,
                "sequence": index,
                "created_at": timestamp,
                "entry_hash": entry.entry_hash,
                "payload": entry.to_dict(),
            }
        )
        navs.append(
            {
                "id": f"nav-{index:04d}",
                "created_at": timestamp,
                "payload": {
                    "product_id": "btc_accumulation",
                    "accounting_asset": "BTC",
                    "nav": 1.0 if index == 0 else 0.9,
                    "observed_at": timestamp,
                    "components": {
                        "btc_balance": 0.8,
                        "stablecoin_balance": 20_000,
                        "stablecoin_per_btc": 100_000 + index,
                        "regime": "trend",
                    },
                    "passive_benchmark_nav": 1.0,
                },
            }
        )

    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_definition).values(
                id="report-definition",
                identity="report-definition",
                product_id="active_income",
                source_type="fixture",
                source_hash="report-source",
                definition={},
            )
        )
        connection.execute(
            insert(strategy_version).values(
                id="report-version",
                definition_id="report-definition",
                version="1",
                created_at=NOW,
                payload={},
            )
        )
        connection.execute(insert(experiment), experiments)
        connection.execute(insert(experiment_run), runs)
        connection.execute(insert(validation_stage), stages)
        connection.execute(
            insert(validation_result).values(
                id="legacy-validation-result",
                experiment_id="candidate-0000",
                state="screening_rejected",
                accepted=False,
                reason_code="fixture_rejection",
                evidence={
                    "candidate_id": "candidate-0000",
                    "definition_hash": "sha256:" + "d" * 64,
                    "evaluated_at": observed_at[0],
                    "opaque_history": "legacy evidence detail",
                },
            )
        )
        connection.execute(insert(decision_trace), traces)
        connection.execute(insert(alpha_forecast), forecasts)
        connection.execute(insert(holdout_claim), claims)
        connection.execute(insert(holdout_outcome), outcomes)
        connection.execute(insert(alert), alerts)
        connection.execute(insert(accounting_entry), active_income_entries)
        connection.execute(insert(accounting_entry), btc_entries)
        connection.execute(insert(nav_snapshot), navs)

    report_root = tmp_path / "runtime" / "reports"
    assert (
        DatabaseReportWorker(engine=database.engine, root=report_root).run_once(
            now=observed_at[-1]
        )["reason_code"]
        == "operator_report_written"
    )
    latest_report = report_root / "latest.json"
    assert latest_report.stat().st_size < REPORT_SNAPSHOT_MAX_BYTES
    snapshot = read_persisted_platform_report(report_root, now=observed_at[-1])
    assert snapshot["snapshot_status"] == "fresh"
    report = snapshot["report"]
    research = report["research"]
    funnel = research["funnel"]
    active_income = report["products"]["active_income"]
    btc = report["products"]["btc_accumulation"]

    assert len(report["trading"]["alpha_forecasts"]) == 100
    assert report["trading"]["detail_counts"]["alpha_forecasts"] == count
    assert len(research["candidate_queue"]) == 100
    assert research["detail_counts"]["candidate_queue"] == count
    assert len(research["validation_stages"]) == 100
    assert research["detail_counts"]["validation_stages"] == count
    stage_detail = research["validation_stages"][0]
    assert stage_detail["id"] == "stage-0000"
    assert stage_detail["experiment_id"] == "candidate-0000"
    assert stage_detail["candidate_id"] == "candidate-0000"
    assert stage_detail["product_id"] == "active_income"
    assert stage_detail["dataset_snapshot_hashes"] == []
    assert stage_detail["source_run_id"] == "run-0000"
    assert stage_detail["evidence_hash"] == "sha256:" + "0" * 64
    assert stage_detail["evaluated_at"] == observed_at[0]
    assert stage_detail["accepted"] is True
    assert "payload" not in stage_detail
    assert len(research["validation_results"]) == 1
    result_detail = research["validation_results"][0]
    assert result_detail["candidate_id"] == "candidate-0000"
    assert result_detail["definition_hash"] == "sha256:" + "d" * 64
    assert "evidence" not in result_detail
    assert len(research["holdout_outcomes"]) == 100
    assert research["detail_counts"]["holdout_outcomes"] == count
    assert funnel["candidates_generated"] == count
    assert funnel["candidates_evaluated"] == count
    assert funnel["candidates_compiled"] == count
    assert funnel["candidate_age_by_state"]["screening"]["count"] == count
    assert funnel["signal_frequency_count"] == count - 9
    assert len(funnel["signal_frequency_distribution"]) == 100
    assert funnel["candidate_correlation_count"] == count
    assert funnel["protected_holdout_count"] == count
    assert len(report["operations"]["alerts"]) == 100
    assert report["operations"]["detail_counts"]["alerts"] == count
    assert report["operations"]["decision_funnel_blocked"] == {
        "data_available:market_data_stale": count
    }
    assert active_income["accounting_entry_count"] == count
    assert active_income["realised_pnl"] == pytest.approx(-0.000001001)
    assert btc["nav_snapshot_count"] == count
    assert btc["accounting_entry_count"] == count
    assert btc["btc_performance"]["btc_nav"] == pytest.approx(0.9)
    assert btc["btc_performance"]["btc_drawdown_fraction"] == pytest.approx(0.1)
    assert btc["btc_performance"]["fees_paid_btc"] == "0.000001000"
    database.dispose()


def test_queued_report_failure_is_recorded_and_retried_after_delay(tmp_path) -> None:
    from unittest.mock import Mock

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'report-failure.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="reporter",
        node_id="node",
        role="report-worker",
        capabilities=("reporting",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="report-failure",
        name="reporting",
        payload={},
        available_at=NOW,
        max_attempts=2,
    )
    worker = DatabaseReportWorker(
        engine=database.engine,
        root=tmp_path / "reports",
        queue=queue,
        worker_id="reporter",
    )

    worker.report.build = Mock(side_effect=RuntimeError("database temporarily unavailable"))
    result = worker.run_once(now=NOW)
    with database.engine.connect() as connection:
        saved_job = connection.execute(
            select(job.c.state, job.c.available_at, job.c.attempts).where(
                job.c.id == "report-failure"
            )
        ).one()
        saved_attempt = connection.execute(
            select(job_attempt.c.status, job_attempt.c.completed_at, job_attempt.c.error).where(
                job_attempt.c.job_id == "report-failure"
            )
        ).one()

    completed_at = dt.datetime.fromisoformat(result["completed_at"])
    available_at = dt.datetime.fromisoformat(saved_job.available_at)
    assert result["reason_code"] == "operator_report_failed"
    assert saved_attempt.status == "failed"
    assert "RuntimeError" in saved_attempt.error
    assert saved_attempt.completed_at == result["completed_at"]
    assert saved_job.state == "pending"
    assert saved_job.attempts == 1
    assert (available_at - completed_at).total_seconds() == 300
    assert completed_at > dt.datetime.fromisoformat(NOW)
    database.dispose()


def test_report_build_deadline_cancels_and_resets_connection_scope(tmp_path, monkeypatch) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'report-deadline.sqlite3'}")
    database.create_schema()
    report = DatabasePlatformReport(database.engine)
    monkeypatch.setattr("src.observability.reports.REPORT_BUILD_BUDGET_SECONDS", -1)

    with pytest.raises(TimeoutError, match="120-second build budget"):
        report.build(now=NOW)

    monkeypatch.setattr("src.observability.reports.REPORT_BUILD_BUDGET_SECONDS", 120)
    assert report.build(now=NOW)["schema"] == "platform.operator_report/v1"
    database.dispose()


def test_queued_report_renews_lease_and_stops_before_publish_when_lost(
    tmp_path, monkeypatch
) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'report-lease.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="reporter",
        node_id="node",
        role="report-worker",
        capabilities=("reporting",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="report-renewal",
        name="reporting",
        payload={},
        available_at=NOW,
    )
    worker = DatabaseReportWorker(
        engine=database.engine,
        root=tmp_path / "reports",
        queue=queue,
        worker_id="reporter",
    )
    heartbeat = queue.heartbeat
    heartbeat_times = []

    def count_heartbeats(claimed, **kwargs):
        heartbeat_times.append(kwargs["now"])
        return heartbeat(claimed, **kwargs)

    monkeypatch.setattr(queue, "heartbeat", count_heartbeats)
    completed = worker.run_once(now=NOW)
    assert completed["reason_code"] == "operator_report_written"
    assert len(heartbeat_times) > 1

    queue.enqueue(
        job_id="report-lost-lease",
        name="reporting",
        payload={},
        available_at=NOW,
    )
    lost_worker = DatabaseReportWorker(
        engine=database.engine,
        root=tmp_path / "lost-reports",
        queue=queue,
        worker_id="reporter",
    )

    def reject_heartbeat(_claimed, **_kwargs):
        raise RuntimeError("report lease was lost")

    monkeypatch.setattr(queue, "heartbeat", reject_heartbeat)
    cancelled = lost_worker.run_once(now=NOW)

    assert cancelled["reason_code"] == "operator_report_failed"
    assert cancelled["error_type"] == "RuntimeError"
    assert not (tmp_path / "lost-reports" / "latest.json").exists()
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
            insert(job),
            [
                {
                    "id": "scheduled:research:1",
                    "name": "research",
                    "state": "completed",
                    "priority": 0,
                    "available_at": NOW,
                    "attempts": 1,
                    "max_attempts": 3,
                    "terminal_reason": None,
                    "producer_identity": "platform-scheduler:test",
                    "content_hash": "sha256:" + "1" * 64,
                    "payload": {"candidate_id": "candidate-1"},
                },
                {
                    "id": "scheduled:research:2",
                    "name": "research",
                    "state": "pending",
                    "priority": 0,
                    "available_at": NOW,
                    "attempts": 0,
                    "max_attempts": 3,
                    "terminal_reason": None,
                    "producer_identity": "platform-scheduler:test",
                    "content_hash": "sha256:" + "2" * 64,
                    "payload": {"candidate_id": "candidate-2"},
                },
                {
                    "id": "scheduled:risk:1",
                    "name": "risk",
                    "state": "pending",
                    "priority": 0,
                    "available_at": NOW,
                    "attempts": 0,
                    "max_attempts": 3,
                    "terminal_reason": None,
                    "producer_identity": "platform-scheduler:test",
                    "content_hash": "sha256:" + "3" * 64,
                    "payload": {},
                },
            ],
        )
        connection.execute(
            insert(job_attempt),
            [
                {
                    "id": "scheduled:research:1:1",
                    "job_id": "scheduled:research:1",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "completed",
                    "error": None,
                    "payload": {},
                },
                {
                    "id": "scheduled:research:2:1",
                    "job_id": "scheduled:research:2",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "deferred",
                    "error": "temporary database load",
                    "payload": {"outcome_category": "blocked_infrastructure"},
                },
                {
                    "id": "scheduled:research:2:2",
                    "job_id": "scheduled:research:2",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "deferred",
                    "error": "temporary database load",
                    "payload": {"outcome_category": "blocked_infrastructure"},
                },
                {
                    "id": "attempt-null-category",
                    "job_id": "scheduled:research:2",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "deferred",
                    "error": "legacy null category",
                    "payload": {"outcome_category": None},
                },
                {
                    "id": "attempt-empty-category",
                    "job_id": "scheduled:research:2",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "deferred",
                    "error": "legacy empty category",
                    "payload": {"outcome_category": ""},
                },
                {
                    "id": "attempt-unknown-category",
                    "job_id": "scheduled:research:1",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "completed",
                    "error": None,
                    "payload": {"outcome_category": "operator_review_" + "x" * 80},
                },
                {
                    "id": "attempt-expired-default",
                    "job_id": "scheduled:research:1",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "expired",
                    "error": "lease expired",
                    "payload": {},
                },
                {
                    "id": "attempt-failed-default",
                    "job_id": "scheduled:research:1",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "failed",
                    "error": "legacy failure",
                    "payload": {},
                },
                {
                    "id": "attempt-failed-explicit-category",
                    "job_id": "scheduled:research:1",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "failed",
                    "error": "explicit result",
                    "payload": {"outcome_category": "accepted"},
                },
                {
                    "id": "attempt-unclassified-default",
                    "job_id": "scheduled:research:1",
                    "worker_id": "worker",
                    "started_at": NOW,
                    "completed_at": NOW,
                    "status": "completed",
                    "error": None,
                    "payload": {},
                },
            ],
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
            insert(account_snapshot),
            [
                {
                    "id": "account-1-latest",
                    "account_id": "account-1",
                    "observed_at": "2026-08-31T09:59:00+00:00",
                    "source": "user_stream_reconciliation",
                    "content_hash": "sha256:" + "4" * 64,
                    "payload": {"account_state_known": True},
                },
                {
                    "id": "account-2-tie-a",
                    "account_id": "account-2",
                    "observed_at": "2026-08-31T09:59:00+00:00",
                    "source": "user_stream_reconciliation",
                    "content_hash": "sha256:" + "5" * 64,
                    "payload": {"account_state_known": True},
                },
                {
                    "id": "account-2-tie-z",
                    "account_id": "account-2",
                    "observed_at": "2026-08-31T09:59:00+00:00",
                    "source": "user_stream_delta",
                    "content_hash": "sha256:" + "6" * 64,
                    "payload": {"account_state_known": False},
                },
                {
                    "id": "account-3-latest",
                    "account_id": "account-3",
                    "observed_at": "2026-08-31T09:59:00+00:00",
                    "source": "user_stream_delta",
                    "content_hash": "sha256:" + "7" * 64,
                    "payload": {"account_state_known": False},
                },
            ],
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
        "scheduled": 3,
        "started": 2,
        "not_started": 1,
        "by_schedule": {
            "research": {"scheduled": 2, "started": 2},
            "risk": {"scheduled": 1, "started": 0},
        },
    }
    assert slis["unresolved_recovery_count"] == 1
    assert slis["stale_account_authority"]["count"] == 2
    assert [item["snapshot_id"] for item in slis["stale_account_authority"]["accounts"]] == [
        "account-2-tie-z",
        "account-3-latest",
    ]
    assert slis["stale_market_data"]["count"] == 1
    assert report["operations"]["queue_progress"]["attempt_count"] == 10
    assert report["operations"]["queue_progress"]["attempt_outcomes"] == {
        "": 1,
        "accepted": 1,
        "blocked_infrastructure": 4,
        "operator_review_" + "x" * 80: 1,
        "unclassified": 3,
    }
    assert report["operations"]["queue_progress"]["resource_deferral_count"] == 2
    assert report["operations"]["queue_progress"]["resource_deferrals"] == {
        "temporary database load": 2,
    }
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


def test_control_cli_status_uses_the_report_worker_root(tmp_path, monkeypatch, capsys) -> None:
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from src.observability.reports import DatabasePlatformReport
    from src.services import control_cli
    from src.services.runtime import utc_now

    database_path = tmp_path / "control-cli.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path}"
    database = PlatformDatabase(database_url)
    database.create_schema()
    data_root = tmp_path / "data"
    parquet_path = data_root / "parquet"
    report_root = data_root / "runtime" / "reports"
    assert (
        DatabaseReportWorker(engine=database.engine, root=report_root).run_once(now=utc_now())[
            "reason_code"
        ]
        == "operator_report_written"
    )
    database.dispose()
    config = SimpleNamespace(
        paths={"parquet": str(parquet_path)},
        database_url=lambda: database_url,
    )
    monkeypatch.setattr(control_cli, "load_platform_config", lambda _path: config)
    monkeypatch.setattr(control_cli, "load_split_configuration", lambda _root: {})
    build = Mock(side_effect=AssertionError("CLI status must read the published report"))
    monkeypatch.setattr(DatabasePlatformReport, "build", build)

    assert (
        control_cli.run(
            SimpleNamespace(config=tmp_path / "config" / "platform.json", command="status")
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["report_snapshot"]["snapshot_status"] == "fresh"
    assert response["schema"] == "platform.control_status/v1"
    build.assert_not_called()


def test_persisted_report_reader_rejects_bad_and_oversized_snapshots(tmp_path, monkeypatch) -> None:
    import json
    import os
    import threading

    from src.observability import reports

    root = tmp_path / "runtime" / "reports"
    root.mkdir(parents=True)
    latest = root / "latest.json"
    valid = {
        "schema": "platform.operator_report/v1",
        "generated_at": NOW,
        "trading": {},
        "research": {},
        "products": {},
        "operations": {},
    }
    valid_json = json.dumps(valid)
    latest.write_text(valid_json)
    fresh = reports.read_persisted_platform_report(root, now=NOW)
    assert fresh["snapshot_status"] == "fresh"
    assert fresh["fresh"] is True
    assert fresh["age_seconds"] == 0
    stale = reports.read_persisted_platform_report(root, now="2026-08-31T10:15:01+00:00")
    assert stale["snapshot_status"] == "stale"
    assert stale["fresh"] is False
    assert stale["age_seconds"] == 901

    latest.write_text("not json")
    assert reports.read_persisted_platform_report(root, now=NOW)["snapshot_status"] == "invalid"
    latest.write_text(json.dumps({**valid, "schema": "unexpected"}))
    assert reports.read_persisted_platform_report(root, now=NOW)["snapshot_status"] == "invalid"
    latest.write_text(json.dumps({**valid, "generated_at": "invalid"}))
    assert reports.read_persisted_platform_report(root, now=NOW)["snapshot_status"] == "invalid"
    for nonfinite in ("NaN", "Infinity", "-Infinity"):
        latest.write_text(
            valid_json.replace('"trading": {}', f'"trading": {{"value": {nonfinite}}}')
        )
        assert reports.read_persisted_platform_report(root, now=NOW)["snapshot_status"] == "invalid"
    latest.write_text(
        valid_json.replace('"trading": {}', '"trading": ' + "[" * 2_000 + "0" + "]" * 2_000)
    )
    assert reports.read_persisted_platform_report(root, now=NOW)["snapshot_status"] == "invalid"
    latest.write_text(
        valid_json.replace('"trading": {}', '"trading": {"value": ' + "9" * 5_000 + "}")
    )
    assert reports.read_persisted_platform_report(root, now=NOW)["snapshot_status"] == "invalid"

    latest.unlink()
    latest.symlink_to(tmp_path / "missing-target.json")
    assert reports.read_persisted_platform_report(root, now=NOW)["snapshot_status"] == "invalid"
    latest.unlink()
    os.mkfifo(latest)
    fifo_result = []
    fifo_reader = threading.Thread(
        target=lambda: fifo_result.append(reports.read_persisted_platform_report(root, now=NOW)),
        daemon=True,
    )
    fifo_reader.start()
    fifo_reader.join(timeout=1)
    assert not fifo_reader.is_alive()
    assert fifo_result[0]["snapshot_status"] == "invalid"
    latest.unlink()
    monkeypatch.setattr(reports, "REPORT_SNAPSHOT_MAX_BYTES", 128)
    latest.write_bytes(b" " * 129)
    assert reports.read_persisted_platform_report(root, now=NOW)["snapshot_status"] == "invalid"
    latest.unlink()
    assert reports.read_persisted_platform_report(root, now=NOW)["snapshot_status"] == "missing"


def test_control_and_metrics_read_published_report_without_rebuilding(
    tmp_path, monkeypatch
) -> None:
    import json
    from unittest.mock import Mock

    from src.observability.metrics import DatabaseMetricsProvider
    from src.observability.reports import (
        DatabasePlatformReport,
        read_persisted_platform_report,
    )
    from src.services.control_api import DatabaseControlPlane
    from src.services.health import DatabaseHeartbeatStore

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'persisted-report.sqlite3'}")
    database.create_schema()
    report_root = tmp_path / "runtime" / "reports"
    assert (
        DatabaseReportWorker(engine=database.engine, root=report_root).run_once(now=NOW)[
            "reason_code"
        ]
        == "operator_report_written"
    )
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    heartbeat_store.record(
        service_name="control-api",
        node_id="fixture",
        observed_at=NOW,
        healthy=True,
        payload={"reason_code": "serving"},
    )
    control = DatabaseControlPlane(database.engine, heartbeat_store, report_root=report_root)
    control.set_paused(
        target="active_income",
        paused=True,
        reason_code="operator_review",
        requested_by="fixture",
        changed_at=NOW,
    )
    build = Mock(side_effect=AssertionError("status and metrics must use the published report"))
    monkeypatch.setattr(DatabasePlatformReport, "build", build)

    status = control.status(now=NOW)
    assert status["report_snapshot"]["snapshot_status"] == "fresh"
    assert status["controls"][0]["target"] == "active_income"
    assert status["operations"]["heartbeats"] == [
        {
            "service_name": "control-api",
            "node_id": "fixture",
            "observed_at": NOW,
            "healthy": True,
            "payload": {"reason_code": "serving"},
        }
    ]
    response = control.report(now=NOW)
    assert response["report_snapshot"]["fresh"] is True
    assert response["report"]["schema"] == "platform.operator_report/v1"
    stale_status = control.status(now="2026-08-31T10:15:01+00:00")
    assert stale_status["report_snapshot"]["snapshot_status"] == "stale"
    assert stale_status["report_snapshot"]["fresh"] is False
    assert (
        control.report(now="2026-08-31T10:15:01+00:00")["report_snapshot"]["snapshot_status"]
        == "stale"
    )
    missing_control = DatabaseControlPlane(database.engine, heartbeat_store)
    assert missing_control.report(now=NOW)["report"] is None
    assert missing_control.status(now=NOW)["report_snapshot"]["snapshot_status"] == "missing"

    fresh_metrics = DatabaseMetricsProvider(
        lambda: read_persisted_platform_report(report_root, now=NOW)
    ).render()
    assert "platform_report_available 1" in fresh_metrics
    assert "platform_report_fresh 1" in fresh_metrics
    assert "platform_report_age_seconds 0" in fresh_metrics
    malformed_report = dict(response["report"])
    malformed_report["research"] = {"funnel": []}
    (report_root / "latest.json").write_text(json.dumps(malformed_report))
    malformed_metrics = DatabaseMetricsProvider(
        lambda: read_persisted_platform_report(report_root, now=NOW)
    ).render()
    assert "platform_report_available 0" in malformed_metrics
    assert "platform_report_fresh 0" in malformed_metrics
    assert 'platform_report_snapshot_status{status="invalid"} 1' in malformed_metrics
    assert "platform_candidates_generated" not in malformed_metrics
    (report_root / "latest.json").write_text(json.dumps(response["report"]))
    stale_metrics = DatabaseMetricsProvider(
        lambda: read_persisted_platform_report(report_root, now="2026-08-31T10:15:01+00:00")
    ).render()
    assert "platform_report_available 1" in stale_metrics
    assert "platform_report_fresh 0" in stale_metrics
    assert "platform_report_age_seconds 901" in stale_metrics
    missing_metrics = DatabaseMetricsProvider(
        lambda: read_persisted_platform_report(None, now=NOW)
    ).render()
    assert "platform_report_available 0" in missing_metrics
    assert "platform_report_fresh 0" in missing_metrics
    assert 'platform_report_snapshot_status{status="missing"} 1' in missing_metrics
    build.assert_not_called()
    database.dispose()


def test_postgres_market_input_scope_walk_matches_window_reference() -> None:
    import os
    import uuid

    from sqlalchemy import create_engine, func, text, union_all
    from sqlalchemy.engine import make_url

    from src.data.database import market_input_buffer

    url = os.environ.get("TRADING_PLATFORM_DATABASE_URL", "")
    if not url.startswith("postgresql"):
        pytest.skip("requires isolated PostgreSQL report-query validation")

    base = make_url(url)
    name = "market_scope_" + uuid.uuid4().hex[:16]
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    database = None
    with admin.connect() as connection:
        connection.execute(text(f"CREATE DATABASE {name}"))
    try:
        database = PlatformDatabase(base.set(database=name).render_as_string(hide_password=False))
        database.migrate()
        with database.engine.begin() as connection:
            connection.execute(
                insert(risk_snapshot),
                [
                    {
                        "id": "null-null-old",
                        "created_at": "2026-08-31T09:00:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": None,
                            "instrument_id": None,
                        },
                    },
                    {
                        "id": "null-null-tie-a",
                        "created_at": "2026-08-31T09:30:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": None,
                            "instrument_id": None,
                        },
                    },
                    {
                        "id": "null-null-tie-z",
                        "created_at": "2026-08-31T09:30:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": None,
                            "instrument_id": None,
                        },
                    },
                    {
                        "id": "null-empty",
                        "created_at": "2026-08-31T09:30:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": None,
                            "instrument_id": "",
                        },
                    },
                    {
                        "id": "empty-null",
                        "created_at": "2026-08-31T09:30:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": "",
                            "instrument_id": None,
                        },
                    },
                    {
                        "id": "empty-empty",
                        "created_at": "2026-08-31T09:30:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": "",
                            "instrument_id": "",
                        },
                    },
                    {
                        "id": "normal-old",
                        "created_at": "2026-08-31T09:00:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": "active_income",
                            "instrument_id": "BTCUSDT",
                        },
                    },
                    {
                        "id": "normal-new",
                        "created_at": "2026-08-31T09:30:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": "active_income",
                            "instrument_id": "BTCUSDT",
                        },
                    },
                    {
                        "id": "shared-id",
                        "created_at": "2026-08-31T09:15:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": "active_income",
                            "instrument_id": "ETHUSDT",
                            "source_event_type": "trade",
                        },
                    },
                ],
            )
            connection.execute(
                insert(market_input_buffer),
                [
                    {
                        "id": "buffer-newest",
                        "product_id": "active_income",
                        "instrument_id": "BTCUSDT",
                        "event_type": "trade",
                        "created_at": "2026-08-31T09:45:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": "active_income",
                            "instrument_id": "BTCUSDT",
                            "source_event_type": "trade",
                        },
                    },
                    {
                        "id": "shared-id",
                        "product_id": "active_income",
                        "instrument_id": "ETHUSDT",
                        "event_type": "trade",
                        "created_at": "2026-08-31T09:16:00+00:00",
                        "payload": {
                            "kind": "market_data_input",
                            "product_id": "active_income",
                            "instrument_id": "ETHUSDT",
                            "source_event_type": "trade",
                        },
                    },
                ],
            )

        immutable = select(
            risk_snapshot.c.id, risk_snapshot.c.created_at, risk_snapshot.c.payload
        ).where(risk_snapshot.c.payload["kind"].as_string() == "market_data_input")
        buffered = select(
            market_input_buffer.c.id,
            market_input_buffer.c.created_at,
            market_input_buffer.c.payload,
        ).where(market_input_buffer.c.payload["kind"].as_string() == "market_data_input")
        sources = union_all(immutable, buffered).subquery()
        reference_ranked = select(
            sources.c.id,
            sources.c.created_at,
            sources.c.payload,
            func.row_number()
            .over(
                partition_by=(
                    sources.c.payload["product_id"].as_string(),
                    sources.c.payload["instrument_id"].as_string(),
                ),
                order_by=(sources.c.created_at.desc(), sources.c.id.desc()),
            )
            .label("recency"),
        ).subquery()
        reference = select(
            reference_ranked.c.id,
            reference_ranked.c.created_at,
            reference_ranked.c.payload,
        ).where(reference_ranked.c.recency == 1)
        with database.engine.connect() as connection:
            expected = {
                row["id"]: (row["created_at"], row["payload"])
                for row in connection.execute(reference).mappings()
            }
            assert (
                connection.execute(
                    text("SELECT to_regclass('ix_risk_snapshot_market_scope_latest')")
                ).scalar_one()
                == "ix_risk_snapshot_market_scope_latest"
            )

        actual_rows = list(
            DatabasePlatformReport(database.engine)._latest_risk_rows(
                "market_data_input", per_instrument=True
            )
        )
        actual = {row["id"]: (row["created_at"], row["payload"]) for row in actual_rows}
        assert actual == expected
        assert "null-null-tie-z" in actual
        assert "null-empty" in actual
        assert "empty-null" in actual
        assert "empty-empty" in actual
        assert "buffer-newest" in actual
        assert [row["id"] for row in actual_rows].count("shared-id") == 1
        assert actual["shared-id"][1]["source_event_type"] == "trade"
    finally:
        if database is not None:
            database.dispose()
        with admin.connect() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {name}"))
        admin.dispose()
