"""Database-backed control report for trading, research, products, and operations."""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import and_, case, exists, func, or_, select
from sqlalchemy.engine import Connection, Engine

from src.accounting.ledger import LedgerEntry
from src.accounting.nav import NavSnapshot
from src.data.database import (
    account_snapshot,
    accounting_entry,
    active_strategy_assignment,
    agent_action,
    alert,
    alpha_forecast,
    balance_snapshot,
    dataset_bundle,
    dataset_snapshot,
    evidence_invalidation,
    exchange_order,
    experiment,
    experiment_run,
    fill,
    forward_evidence,
    forward_paper_decision,
    forward_paper_observation,
    forward_paper_summary,
    holdout_claim,
    holdout_outcome,
    job,
    market_input_sources,
    nav_snapshot,
    platform_schedule,
    position,
    production_preflight,
    promotion_event,
    protective_stop,
    reconciliation_event,
    research_epoch,
    research_round,
    research_thesis,
    risk_snapshot,
    service_heartbeat,
    strategy_approval,
    strategy_artefact,
    strategy_definition,
    strategy_identity,
    strategy_lineage,
    strategy_version,
    target_position,
    thesis_trial,
    universe_member,
    universe_snapshot,
    validation_result,
    validation_stage,
    worker,
)
from src.data.database import (
    decision_trace as decision_trace_table,
)
from src.domain._codec import canonical_hash, timestamp, to_primitive
from src.observability.decision_trace import DecisionTraceStage
from src.observability.queue_progress import queue_progress
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.outcomes import terminal_trial_disposition

REPORT_DETAIL_LIMIT = 1000
REPORT_BUILD_BUDGET_SECONDS = 120
REPORT_STATEMENT_TIMEOUT_MILLISECONDS = 20_000
REPORT_LOCK_TIMEOUT_MILLISECONDS = 5_000
_DATASET_PLAN_FIELDS = {
    "screening": "screening_snapshot_ids",
    "development": "development_snapshot_ids",
    "robustness": "robustness_snapshot_ids",
    "protected_holdout": "protected_holdout_snapshot_id",
}


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _waiting_dataset_role(state: str) -> str | None:
    if state.startswith("waiting_for_dataset:"):
        return state.removeprefix("waiting_for_dataset:")
    return {
        "queued": "screening",
        "screening": "development",
        "development": "robustness",
        "robustness": "protected_holdout",
    }.get(state)


def _empty_product() -> dict[str, Any]:
    return {
        "fees": 0.0,
        "funding": 0.0,
        "slippage": 0.0,
        "realised_pnl": 0.0,
        "nav": None,
        "peak_nav": None,
        "drawdown": None,
        "accounting_entry_count": 0,
        "nav_snapshot_count": 0,
        "attribution": {"strategy": {}, "symbol": {}, "sleeve": {}, "product": {}},
    }


def _reported_cost(metric: str, metadata: Mapping[str, Any], effect: float) -> float:
    if metric == "slippage" and metadata.get("included_in_fill_price") is True:
        return float(metadata["cost"])
    return -effect


class DatabasePlatformReport:
    def __init__(
        self,
        engine: Engine,
        *,
        account_stale_after_seconds: int = 60,
        market_data_stale_after_seconds: int = 5,
        storage_root: Path | None = None,
        spool_root: Path | None = None,
    ):
        if account_stale_after_seconds <= 0 or market_data_stale_after_seconds <= 0:
            raise ValueError("report staleness thresholds must be positive")
        self.engine = engine
        self.account_stale_after_seconds = account_stale_after_seconds
        self.market_data_stale_after_seconds = market_data_stale_after_seconds
        self.storage_root = storage_root.resolve() if storage_root is not None else None
        self.spool_root = spool_root.resolve() if spool_root is not None else None
        self._active_connection: Connection | None = None
        self._build_deadline: float | None = None
        self._progress_callback: Callable[[], Any] | None = None

    def build(
        self,
        *,
        now: str | None = None,
        on_progress: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        report_time = timestamp(now or dt.datetime.now(dt.UTC), field="report time")
        if self._active_connection is not None:
            raise RuntimeError("a platform report build is already active")
        self._build_deadline = time.monotonic() + REPORT_BUILD_BUDGET_SECONDS
        self._progress_callback = on_progress
        try:
            with self._read_connection():
                report = {
                    "schema": "platform.operator_report/v1",
                    "trading": self._trading(),
                    "research": self._research(now=report_time),
                    "products": self._products(),
                    "operations": self._operations(now=report_time),
                }
                self._check_deadline()
                return report
        finally:
            self._active_connection = None
            self._build_deadline = None
            self._progress_callback = None

    @contextmanager
    def _read_connection(self) -> Iterator[Connection]:
        if self._active_connection is not None:
            yield self._active_connection
            return
        with self.engine.connect() as connection, connection.begin():
            if connection.dialect.name == "postgresql":
                connection.exec_driver_sql(
                    f"SET LOCAL lock_timeout = '{REPORT_LOCK_TIMEOUT_MILLISECONDS}ms'"
                )
                connection.exec_driver_sql(
                    f"SET LOCAL statement_timeout = '{REPORT_STATEMENT_TIMEOUT_MILLISECONDS}ms'"
                )
            self._active_connection = connection
            try:
                yield connection
            finally:
                self._active_connection = None

    def _check_deadline(self) -> float | None:
        if self._build_deadline is None:
            return None
        remaining = self._build_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("platform report exceeded its 120-second build budget")
        return remaining

    def _before_query(self, connection: Connection) -> None:
        remaining = self._check_deadline()
        if self._progress_callback is not None:
            self._progress_callback()
            remaining = self._check_deadline()
        if connection.dialect.name == "postgresql":
            milliseconds = REPORT_STATEMENT_TIMEOUT_MILLISECONDS
            if remaining is not None:
                milliseconds = max(1, min(milliseconds, int(remaining * 1000)))
            connection.exec_driver_sql(f"SET LOCAL statement_timeout = '{milliseconds}ms'")

    def _execute(self, connection: Connection, statement):
        self._before_query(connection)
        return connection.execute(statement)

    def _queue_progress(self, *, now: str) -> dict[str, Any]:
        report = self

        class ReportConnection:
            def __init__(self, connection: Connection):
                self.connection = connection
                self.dialect = connection.dialect

            def execute(self, statement):
                return report._execute(self.connection, statement)

        class ReportEngine:
            @contextmanager
            def connect(self):
                with report._read_connection() as connection:
                    yield ReportConnection(connection)

        return queue_progress(ReportEngine(), now=now)

    def _progress_checkpoint(self) -> None:
        self._check_deadline()
        if self._progress_callback is not None:
            self._progress_callback()
            self._check_deadline()

    def _detail_payloads(
        self, table, *, order_by=None, where=None
    ) -> tuple[list[dict[str, Any]], int]:
        statement = select(table.c.payload)
        if where is not None:
            statement = statement.where(where)
        statement = statement.order_by(order_by if order_by is not None else table.c.id).limit(
            REPORT_DETAIL_LIMIT
        )
        with self._read_connection() as connection:
            total = self._count_where(table, where) if where is not None else self._count(table)
            rows = self._execute(connection, statement).scalars().all()
            self._check_deadline()
        return [dict(payload) for payload in rows], total

    def _detail_rows(self, table, *, order_by=None, where=None) -> tuple[list[dict[str, Any]], int]:
        statement = select(table)
        if where is not None:
            statement = statement.where(where)
        statement = statement.order_by(order_by if order_by is not None else table.c.id).limit(
            REPORT_DETAIL_LIMIT
        )
        with self._read_connection() as connection:
            total = self._count_where(table, where) if where is not None else self._count(table)
            rows = self._execute(connection, statement).mappings().all()
            self._check_deadline()
        return [to_primitive(dict(row)) for row in rows], total

    def _iter_rows(self, statement):
        statement = statement.execution_options(stream_results=True, yield_per=REPORT_DETAIL_LIMIT)
        with self._read_connection() as connection:
            result = self._execute(connection, statement)
            for index, row in enumerate(result.mappings(), start=1):
                yield row
                if index % REPORT_DETAIL_LIMIT == 0:
                    self._progress_checkpoint()

    def _count_where(self, table, *criteria) -> int:
        statement = select(func.count()).select_from(table)
        if criteria:
            statement = statement.where(*criteria)
        with self._read_connection() as connection:
            result = self._execute(connection, statement).scalar_one()
            self._check_deadline()
            return int(result)

    def _trading(self) -> dict[str, Any]:
        positions, position_count = self._detail_payloads(
            position, order_by=position.c.created_at.desc()
        )
        targets, target_count = self._detail_payloads(
            target_position, order_by=target_position.c.created_at.desc()
        )
        forecasts, forecast_count = self._detail_payloads(
            alpha_forecast, order_by=alpha_forecast.c.created_at.desc()
        )
        stops, stop_count = self._detail_payloads(
            protective_stop, order_by=protective_stop.c.created_at.desc()
        )
        assignments, assignment_count = self._detail_rows(
            active_strategy_assignment,
            order_by=active_strategy_assignment.c.assigned_at.desc(),
        )
        balances, balance_count = self._detail_payloads(
            balance_snapshot, order_by=balance_snapshot.c.created_at.desc()
        )
        return {
            "detail_limit": REPORT_DETAIL_LIMIT,
            "detail_counts": {
                "positions": position_count,
                "target_positions": target_count,
                "alpha_forecasts": forecast_count,
                "stops": stop_count,
                "active_strategy_assignments": assignment_count,
                "account_balances": balance_count,
            },
            "positions": positions,
            "target_positions": targets,
            "alpha_forecasts": forecasts,
            "stops": stops,
            "active_strategy_assignments": assignments,
            "account_balances": balances,
            "order_events": self._count(exchange_order),
            "fills": self._count(fill),
        }

    def _research(self, *, now: str) -> dict[str, Any]:
        experiments, experiment_count = self._detail_rows(
            experiment, order_by=experiment.c.submitted_at.desc()
        )
        diagnostic_ids, diagnostic_candidate_count = self._diagnostic_candidate_ids()
        candidate_queue, active_candidate_count = self._candidate_queue()
        results, result_count = self._detail_rows(validation_result)
        validation_stages, stage_count = self._detail_rows(validation_stage)
        holdout_claims, holdout_claim_count = self._detail_rows(holdout_claim)
        holdout_outcomes, holdout_outcome_count = self._detail_rows(holdout_outcome)
        forward_details = {
            "forward_evidence": self._detail_rows(forward_evidence),
            "forward_paper_observations": self._detail_rows(forward_paper_observation),
            "forward_paper_summaries": self._detail_rows(forward_paper_summary),
            "forward_paper_decisions": self._detail_rows(forward_paper_decision),
        }
        assignments, assignment_count = self._detail_rows(
            active_strategy_assignment,
            order_by=active_strategy_assignment.c.assigned_at.desc(),
        )
        identities, identity_count = self._detail_rows(strategy_identity)
        lineages, lineage_count = self._detail_rows(strategy_lineage)
        theses, thesis_count = self._detail_rows(research_thesis)
        trials, trial_count = self._detail_rows(thesis_trial)
        artefacts, artefact_count = self._detail_rows(strategy_artefact)
        approvals, approval_count = self._detail_rows(strategy_approval)
        preflights, preflight_count = self._detail_rows(production_preflight)
        promotions, promotion_count = self._detail_rows(promotion_event)
        bundles, bundle_count = self._detail_rows(dataset_bundle)
        actions, action_count = self._detail_payloads(
            agent_action, order_by=agent_action.c.created_at.desc()
        )
        report = {
            "detail_limit": REPORT_DETAIL_LIMIT,
            "candidate_queue": candidate_queue,
            "experiments": experiments,
            "diagnostic_candidate_count": diagnostic_candidate_count,
            "diagnostic_candidate_ids": diagnostic_ids,
            "validation_results": results,
            "validation_stages": validation_stages,
            "holdout_claims": holdout_claims,
            "holdout_outcomes": holdout_outcomes,
            **{key: value[0] for key, value in forward_details.items()},
            "active_strategy_assignments": assignments,
            "strategy_identities": identities,
            "strategy_lineage": lineages,
            "research_theses": theses,
            "thesis_trials": trials,
            "universe_attrition": self._universe_attrition(),
            "strategy_artefacts": artefacts,
            "strategy_approvals": approvals,
            "production_preflights": preflights,
            "promotion_events": promotions,
            "dataset_bundles": bundles,
            "rejection_reasons": [
                item.get("reason_code") for item in results if item.get("reason_code")
            ],
            "agent_activity": actions,
            "detail_counts": {
                "candidate_queue": active_candidate_count,
                "experiments": experiment_count,
                "diagnostic_candidate_ids": diagnostic_candidate_count,
                "validation_results": result_count,
                "validation_stages": stage_count,
                "holdout_claims": holdout_claim_count,
                "holdout_outcomes": holdout_outcome_count,
                **{key: value[1] for key, value in forward_details.items()},
                "active_strategy_assignments": assignment_count,
                "strategy_identities": identity_count,
                "strategy_lineage": lineage_count,
                "research_theses": thesis_count,
                "thesis_trials": trial_count,
                "strategy_artefacts": artefact_count,
                "strategy_approvals": approval_count,
                "production_preflights": preflight_count,
                "promotion_events": promotion_count,
                "dataset_bundles": bundle_count,
                "rejection_reasons": self._count_where(
                    validation_result,
                    validation_result.c.reason_code.is_not(None),
                    validation_result.c.reason_code != "",
                ),
                "agent_activity": action_count,
            },
        }
        report["funnel"] = self._research_funnel(
            report,
            now=now,
            diagnostic_candidate_count=diagnostic_candidate_count,
        )
        return report

    def _diagnostic_experiments(self):
        diagnostic_versions = (
            select(strategy_version.c.id)
            .select_from(
                strategy_version.join(
                    strategy_definition,
                    strategy_version.c.definition_id == strategy_definition.c.id,
                )
            )
            .where(
                strategy_definition.c.definition["metadata"]["diagnostic"].as_boolean().is_(True)
            )
        )
        return (
            select(experiment.c.id)
            .where(
                or_(
                    experiment.c.metadata["diagnostic"].as_boolean().is_(True),
                    experiment.c.strategy_version_id.in_(diagnostic_versions),
                )
            )
            .subquery()
        )

    def _diagnostic_candidate_ids(self) -> tuple[list[str], int]:
        diagnostic_ids = self._diagnostic_experiments()
        total = self._count_select(select(diagnostic_ids.c.id))
        statement = (
            select(diagnostic_ids.c.id).order_by(diagnostic_ids.c.id).limit(REPORT_DETAIL_LIMIT)
        )
        with self._read_connection() as connection:
            rows = self._execute(connection, statement).scalars().all()
            self._check_deadline()
        return [str(row) for row in rows], total

    def _candidate_queue(self) -> tuple[list[dict[str, Any]], int]:
        diagnostic_ids = self._diagnostic_experiments()
        state = experiment.c.state
        deferred = func.coalesce(
            experiment.c.metadata["evaluation_deferred"]["outcome_category"].as_string(),
            "",
        )
        inactive = {
            "completed",
            "rejected",
            "retired",
            "protected",
            "forward",
            "forward_paper",
            "live_ready",
            "live_canary",
            "live",
            "suspended",
        }
        active = and_(
            state.not_in(inactive),
            state.not_like("%\\_rejected", escape="\\"),
            state.not_like("blocked\\_%", escape="\\"),
            ~or_(
                state.in_(("infrastructure_exhausted", "unsupported")),
                deferred.in_(("blocked_data", "inconclusive", "unsupported")),
            ),
            ~exists(select(diagnostic_ids.c.id).where(diagnostic_ids.c.id == experiment.c.id)),
        )
        statement = (
            select(experiment)
            .where(active)
            .order_by(experiment.c.submitted_at.desc(), experiment.c.id)
            .limit(REPORT_DETAIL_LIMIT)
        )
        total = self._count_where(experiment, active)
        with self._read_connection() as connection:
            rows = self._execute(connection, statement).mappings().all()
            self._check_deadline()
        return [to_primitive(dict(row)) for row in rows], total

    def _research_funnel(
        self,
        research: dict[str, Any],
        *,
        now: str,
        diagnostic_candidate_count: int,
    ) -> dict[str, Any]:
        diagnostic_ids = self._diagnostic_experiments()
        invalidated = exists(
            select(evidence_invalidation.c.id).where(
                evidence_invalidation.c.payload["kind"].as_string() == "validation_stage",
                evidence_invalidation.c.payload["source_id"].as_string() == validation_stage.c.id,
            )
        )
        current = (
            select(validation_stage)
            .where(validation_stage.c.engine_version == ECONOMIC_ENGINE_VERSION, ~invalidated)
            .subquery()
        )
        production_stages = (
            select(current)
            .where(
                ~exists(
                    select(diagnostic_ids.c.id).where(
                        diagnostic_ids.c.id == current.c.experiment_id
                    )
                )
            )
            .subquery()
        )
        current_count = self._count_select(select(current.c.id))
        diagnostic_stage_count = self._count_select(
            select(current.c.id).where(
                exists(
                    select(diagnostic_ids.c.id).where(
                        diagnostic_ids.c.id == current.c.experiment_id
                    )
                )
            )
        )
        rejected = production_stages.c.accepted.is_(False)
        reason = case(
            (
                or_(
                    production_stages.c.reason_code.is_(None),
                    production_stages.c.reason_code == "",
                ),
                "unknown",
            ),
            else_=production_stages.c.reason_code,
        )
        rejection_by_stage = dict(
            self._grouped_counts(
                select(production_stages.c.stage, func.count())
                .where(rejected)
                .group_by(production_stages.c.stage)
            )
        )
        rejection_reasons = dict(
            self._grouped_counts(select(reason, func.count()).where(rejected).group_by(reason))
        )
        blocked_ranked = (
            select(
                production_stages.c.experiment_id,
                production_stages.c.stage,
                func.row_number()
                .over(
                    partition_by=production_stages.c.experiment_id,
                    order_by=(production_stages.c.evaluated_at, production_stages.c.id),
                )
                .label("rank"),
            )
            .where(rejected)
            .subquery()
        )
        first_blocked = dict(
            self._grouped_counts(
                select(blocked_ranked.c.stage, func.count())
                .where(blocked_ranked.c.rank == 1)
                .group_by(blocked_ranked.c.stage)
            )
        )
        candidates_generated = self._count(experiment) - diagnostic_candidate_count
        evaluated_count = self._count_select(
            select(func.distinct(production_stages.c.experiment_id))
        )
        candidates_compiled = self._sum_case(
            production_stages,
            and_(
                production_stages.c.stage == "screening",
                production_stages.c.accepted.is_(True),
            ),
        )
        never_evaluated = self._never_evaluated_candidates(diagnostic_ids, production_stages)
        candidate_summary = self._candidate_summary(diagnostic_ids, now=now)
        missing_stage_datasets = candidate_summary["missing_stage_datasets"]
        candidate_correlations, candidate_correlation_count = self._detail_values(
            select(production_stages.c.payload["candidate_correlation"].as_json())
            .where(production_stages.c.payload["candidate_correlation"].as_string().is_not(None))
            .order_by(production_stages.c.evaluated_at, production_stages.c.id)
        )
        signal_frequencies, signal_frequency_count = self._signal_frequencies(production_stages)
        identity_count = self._count(strategy_identity)
        duplicate_count = self._count_where(
            strategy_identity, strategy_identity.c.is_duplicate.is_(True)
        )
        near_duplicate_count = sum(
            bool((row["metadata"] or {}).get("near_duplicate_of"))
            for row in self._iter_rows(select(strategy_identity.c.metadata))
            if isinstance(row["metadata"], Mapping)
        )
        feature_families = self._json_group_counts(
            strategy_definition.c.definition["family"].as_string(),
            strategy_definition,
        )
        thesis_families = self._json_group_counts(
            research_thesis.c.payload["mechanism_category"].as_string(),
            research_thesis,
        )
        measured = self._measured_run_progress(diagnostic_ids)
        epoch_rows, epoch_count = self._detail_rows(research_epoch)
        epochs = [
            {
                key: row[key]
                for key in (
                    "id",
                    "product_id",
                    "dataset_bundle_id",
                    "state",
                    "round_cursor",
                    "budget",
                    "consumed",
                )
            }
            for row in epoch_rows
        ]
        schedule_states = {
            str(row["job_name"]): str(row["state"])
            for row in self._iter_rows(
                select(platform_schedule.c.job_name, platform_schedule.c.state)
            )
        }
        promotions = sum(
            row["accepted"] is True and str(row["prior_state"]) != str(row["next_state"])
            for row in self._iter_rows(
                select(
                    promotion_event.c.payload["accepted"].as_boolean().label("accepted"),
                    promotion_event.c.payload["prior_state"].as_string().label("prior_state"),
                    promotion_event.c.payload["next_state"].as_string().label("next_state"),
                )
            )
        )
        active_forward_count = self._count_where(
            active_strategy_assignment,
            active_strategy_assignment.c.active.is_(True),
            active_strategy_assignment.c.execution_mode == "paper",
            active_strategy_assignment.c.lifecycle_state == "forward_paper",
        )
        return {
            **measured,
            **self._queue_progress(now=now),
            "research_rounds_completed": self._count_where(
                research_round, research_round.c.state == "completed"
            ),
            "research_epochs": epochs,
            "research_epoch_count": epoch_count,
            "theses_generated": self._count(research_thesis),
            "candidates_generated": candidates_generated,
            "diagnostic_candidates_excluded": diagnostic_candidate_count,
            "diagnostic_run_count": self._diagnostic_run_count(diagnostic_ids),
            "diagnostic_validation_stage_count": diagnostic_stage_count,
            "candidates_compiled": candidates_compiled,
            "candidates_evaluated": evaluated_count,
            "historical_validation_stage_count": self._count(validation_stage) - current_count,
            "candidates_rejected_by_stage": rejection_by_stage,
            "candidates_deferred_by_stage": candidate_summary["deferred_by_stage"],
            "top_rejection_reasons": dict(
                sorted(rejection_reasons.items(), key=lambda item: (-item[1], item[0]))[:10]
            ),
            "signal_frequency_distribution": sorted(signal_frequencies),
            "signal_frequency_count": signal_frequency_count,
            "feature_family_concentration": feature_families,
            "thesis_family_coverage": thesis_families,
            "universe_attrition": research["universe_attrition"],
            "candidate_correlation": candidate_correlations,
            "candidate_correlation_count": candidate_correlation_count,
            "exact_duplicate_rate": duplicate_count / identity_count if identity_count else 0.0,
            "near_duplicate_rate": near_duplicate_count / identity_count if identity_count else 0.0,
            "cumulative_trial_count": self._count(thesis_trial),
            "protected_holdout_count": self._count(holdout_outcome),
            "forward_paper_count": self._count(forward_paper_observation),
            "active_forward_count": active_forward_count,
            "strategy_promotions": promotions,
            "first_blocked_stage": first_blocked,
            "candidates_never_evaluated": never_evaluated[0],
            "candidates_never_evaluated_count": never_evaluated[1],
            "candidate_age_by_state": candidate_summary["age_by_state"],
            "missing_stage_dataset_count": sum(missing_stage_datasets.values()),
            "missing_stage_datasets": missing_stage_datasets,
            "schedule_states": schedule_states,
            "candidates_without_job_or_reason": candidate_summary["without_job"],
            "candidates_without_job_or_reason_count": candidate_summary["without_job_count"],
            "detail_limit": REPORT_DETAIL_LIMIT,
        }

    def _count_select(self, statement) -> int:
        with self._read_connection() as connection:
            value = self._execute(
                connection,
                select(func.count()).select_from(statement.subquery()),
            ).scalar_one()
            self._check_deadline()
            return int(value)

    def _grouped_counts(self, statement):
        return [tuple(row.values()) for row in self._iter_rows(statement)]

    def _sum_case(self, table, condition) -> int:
        statement = select(func.coalesce(func.sum(case((condition, 1), else_=0)), 0)).select_from(
            table
        )
        with self._read_connection() as connection:
            value = self._execute(connection, statement).scalar_one()
            self._check_deadline()
            return int(value)

    def _json_group_counts(self, value, table) -> dict[str, int]:
        group = case((or_(value.is_(None), value == ""), "unknown"), else_=value)
        return dict(
            self._grouped_counts(select(group, func.count()).select_from(table).group_by(group))
        )

    def _detail_values(self, statement) -> tuple[list[Any], int]:
        total = self._count_select(statement.order_by(None))
        limited = statement.limit(REPORT_DETAIL_LIMIT)
        with self._read_connection() as connection:
            rows = self._execute(connection, limited).scalars().all()
            self._check_deadline()
        return list(rows), total

    def _signal_frequencies(self, stages) -> tuple[list[float], int]:
        values: list[float] = []
        count = 0
        statement = select(
            stages.c.payload["evidence"].as_json().label("evidence"),
            stages.c.payload["signal_frequency"].as_json().label("fallback_frequency"),
        ).order_by(stages.c.evaluated_at, stages.c.id)
        for row in self._iter_rows(statement):
            evidence = row["evidence"]
            value = (
                evidence.get("signal_frequency")
                if isinstance(evidence, Mapping)
                else row["fallback_frequency"]
            )
            if isinstance(value, int | float) and not isinstance(value, bool):
                count += 1
                if len(values) < REPORT_DETAIL_LIMIT:
                    values.append(float(value))
        return values, count

    def _never_evaluated_candidates(
        self, diagnostic_ids, production_stages
    ) -> tuple[list[str], int]:
        has_stage = exists(
            select(production_stages.c.id).where(
                production_stages.c.experiment_id == experiment.c.id
            )
        )
        is_production = ~exists(
            select(diagnostic_ids.c.id).where(diagnostic_ids.c.id == experiment.c.id)
        )
        details = (
            select(experiment.c.id)
            .where(is_production, ~has_stage)
            .order_by(experiment.c.id)
            .limit(REPORT_DETAIL_LIMIT)
        )
        count = self._count_where(experiment, is_production, ~has_stage)
        with self._read_connection() as connection:
            ids = [str(value) for value in self._execute(connection, details).scalars()]
            self._check_deadline()
        return ids, count

    def _diagnostic_run_count(self, diagnostic_ids) -> int:
        run_candidate = func.coalesce(
            func.nullif(experiment_run.c.payload["candidate_id"].as_string(), ""),
            experiment_run.c.payload["receipt"]["candidate_id"].as_string(),
        )
        is_diagnostic = exists(
            select(diagnostic_ids.c.id).where(diagnostic_ids.c.id == run_candidate)
        )
        return self._count_where(experiment_run, is_diagnostic)

    def _measured_run_progress(self, diagnostic_ids) -> dict[str, Any]:
        run_candidate = func.coalesce(
            func.nullif(experiment_run.c.payload["candidate_id"].as_string(), ""),
            experiment_run.c.payload["receipt"]["candidate_id"].as_string(),
        )
        is_diagnostic = exists(
            select(diagnostic_ids.c.id).where(diagnostic_ids.c.id == run_candidate)
        )
        current: set[str] = set()
        measured: set[str] = set()
        current_count = 0
        without_receipt = 0
        statement = select(
            experiment_run.c.payload["run_name"].as_string().label("run_name"),
            experiment_run.c.payload["receipt"]["candidate_id"].as_string().label("candidate_id"),
            experiment_run.c.payload["receipt"]["dataset_snapshot_ids"].as_json().label("datasets"),
            experiment_run.c.payload["receipt"]["executor_version"].as_string().label("executor"),
            experiment_run.c.payload["receipt"]["input_hash"].as_string().label("input_hash"),
        ).where(~is_diagnostic)
        for row in self._iter_rows(statement):
            candidate_id = row["candidate_id"]
            datasets = row["datasets"]
            executor = row["executor"]
            input_hash = row["input_hash"]
            if not all((candidate_id, datasets, executor, input_hash)):
                without_receipt += 1
                continue
            identity = canonical_hash(
                {
                    "stage": row["run_name"],
                    "candidate_id": candidate_id,
                    "datasets": sorted(datasets),
                    "engine_version": executor,
                }
            )
            measured.add(identity)
            if executor == ECONOMIC_ENGINE_VERSION:
                current_count += 1
                current.add(identity)
        return {
            "economic_engine_version": ECONOMIC_ENGINE_VERSION,
            "unique_measured_evaluations": len(current),
            "duplicate_measured_evaluations": current_count - len(current),
            "historical_measured_evaluations": len(measured - current),
            "runs_without_execution_receipts": without_receipt,
            "attempt_count": 0,
            "resource_deferrals": {},
            "attempt_outcomes": {},
        }

    def _candidate_summary(self, diagnostic_ids, *, now: str) -> dict[str, Any]:
        is_production = ~exists(
            select(diagnostic_ids.c.id).where(diagnostic_ids.c.id == experiment.c.id)
        )
        active_candidate_jobs = (
            select(job.c.payload["candidate_id"].as_string().label("candidate_id"))
            .where(
                job.c.state.in_(("pending", "running")),
                job.c.payload["candidate_id"].as_string().is_not(None),
            )
            .distinct()
            .subquery()
        )
        ages: dict[str, list[float | int]] = {}
        deferred: dict[str, int] = {}
        missing: dict[str, int] = {}
        without_job: list[str] = []
        without_job_count = 0
        batch: list[dict[str, Any]] = []
        statement = (
            select(
                experiment.c.id,
                experiment.c.state,
                experiment.c.submitted_at,
                experiment.c.dataset_bundle_id,
                experiment.c.metadata["dataset_waiting"].as_json().label("dataset_waiting"),
                experiment.c.metadata["blocked_reason"].as_json().label("blocked_reason"),
                experiment.c.metadata["evaluation_deferred"].as_json().label("evaluation_deferred"),
                experiment.c.metadata["_dataset_plan"].as_json().label("dataset_plan"),
                active_candidate_jobs.c.candidate_id.is_not(None).label("has_active_job"),
            )
            .select_from(
                experiment.outerjoin(
                    active_candidate_jobs,
                    active_candidate_jobs.c.candidate_id == experiment.c.id,
                )
            )
            .where(is_production)
            .order_by(experiment.c.id)
        )
        for row in self._iter_rows(statement):
            candidate = dict(row)
            candidate["metadata"] = {
                "dataset_waiting": candidate.pop("dataset_waiting"),
                "blocked_reason": candidate.pop("blocked_reason"),
                "evaluation_deferred": candidate.pop("evaluation_deferred") or {},
                "_dataset_plan": candidate.pop("dataset_plan"),
            }
            state = str(candidate.get("state") or "unknown")
            age = _age_seconds(str(candidate["submitted_at"]), now)
            bucket = ages.setdefault(state, [0, 0.0, 0.0])
            bucket[0] = int(bucket[0]) + 1
            bucket[1] = float(bucket[1]) + age
            bucket[2] = max(float(bucket[2]), age)
            if state.endswith("_deferred"):
                stage = state.removesuffix("_deferred")
                deferred[stage] = deferred.get(stage, 0) + 1
            metadata = candidate.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            if (
                _candidate_is_active(candidate)
                and not candidate["has_active_job"]
                and not (
                    metadata.get("dataset_waiting")
                    or metadata.get("blocked_reason")
                    or metadata.get("evaluation_deferred")
                )
                and not state.startswith("waiting_")
            ):
                without_job_count += 1
                if len(without_job) < REPORT_DETAIL_LIMIT:
                    without_job.append(str(candidate["id"]))
            if _waiting_dataset_role(state):
                batch.append(candidate)
                if len(batch) == REPORT_DETAIL_LIMIT:
                    self._count_missing_stage_datasets(batch, missing)
                    batch.clear()
        if batch:
            self._count_missing_stage_datasets(batch, missing)
        return {
            "age_by_state": {
                state: {
                    "count": int(values[0]),
                    "oldest_seconds": float(values[2]),
                    "average_seconds": float(values[1]) / int(values[0]),
                }
                for state, values in sorted(ages.items())
            },
            "deferred_by_stage": dict(sorted(deferred.items())),
            "missing_stage_datasets": dict(sorted(missing.items())),
            "without_job": without_job,
            "without_job_count": without_job_count,
        }

    def _count_missing_stage_datasets(self, candidates, missing: dict[str, int]) -> None:
        direct, bundle_ids = self._waiting_dataset_requirements(candidates)
        bundles = self._dataset_bundle_stages(bundle_ids)
        resolved, snapshot_ids = self._resolve_candidate_stage_ids(direct, bundles)
        existing = self._existing_dataset_snapshots(snapshot_ids)
        for _candidate_id, role, values in resolved:
            if not values or any(item is None or str(item) not in existing for item in values):
                missing[role] = missing.get(role, 0) + 1

    def _waiting_dataset_requirements(self, candidates):
        bundle_ids: set[str] = set()
        direct: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            state = str(candidate.get("state") or "")
            role = _waiting_dataset_role(state)
            if role is None:
                continue
            metadata = candidate.get("metadata")
            plan = metadata.get("_dataset_plan") if isinstance(metadata, Mapping) else None
            stage_ids: dict[str, Any] = {}
            field = _DATASET_PLAN_FIELDS.get(role)
            if isinstance(plan, dict) and field is not None and field in plan:
                stage_ids = {role: plan[field]}
            bundle_id = str(candidate.get("dataset_bundle_id") or "")
            if not stage_ids and bundle_id:
                bundle_ids.add(bundle_id)
            direct[str(candidate["id"])] = {
                "role": role,
                "stage_ids": stage_ids,
                "bundle_id": bundle_id,
            }
        return direct, bundle_ids

    def _dataset_bundle_stages(self, bundle_ids: set[str]) -> dict[str, Any]:
        bundles: dict[str, Any] = {}
        for chunk in _chunks(sorted(bundle_ids), 500):
            statement = select(
                dataset_bundle.c.id.label("bundle_id"),
                dataset_bundle.c.payload["stage_snapshot_ids"].as_json().label("stage_ids"),
            ).where(dataset_bundle.c.id.in_(chunk))
            bundles.update(
                {str(row["bundle_id"]): row["stage_ids"] for row in self._iter_rows(statement)}
            )
        return bundles

    def _resolve_candidate_stage_ids(self, direct, bundles):
        snapshot_ids: set[str] = set()
        resolved: list[tuple[str, str, list[Any]]] = []
        for candidate_id, detail in direct.items():
            stage_ids = detail["stage_ids"]
            if not stage_ids:
                stage_ids = bundles.get(detail["bundle_id"], {})
                stage_ids = stage_ids if isinstance(stage_ids, Mapping) else {}
            value = stage_ids.get(detail["role"])
            values = value if isinstance(value, list | tuple) else [value]
            resolved.append((candidate_id, detail["role"], values))
            snapshot_ids.update(str(item) for item in values if item is not None)
        return resolved, snapshot_ids

    def _existing_dataset_snapshots(self, snapshot_ids: set[str]) -> set[str]:
        existing: set[str] = set()
        for chunk in _chunks(sorted(snapshot_ids), 500):
            existing.update(
                str(row["snapshot_id"])
                for row in self._iter_rows(
                    select(dataset_snapshot.c.id.label("snapshot_id")).where(
                        dataset_snapshot.c.id.in_(chunk)
                    )
                )
            )
        return existing

    def _products(self) -> dict[str, Any]:
        products: dict[str, dict[str, Any]] = {}
        for row in self._iter_rows(
            select(accounting_entry.c.payload).order_by(accounting_entry.c.created_at)
        ):
            entry = row["payload"]
            product_id = str(entry.get("product_id") or "unknown")
            product = products.setdefault(product_id, _empty_product())
            product["accounting_entry_count"] += 1
            metadata = entry.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            kind = str(metadata.get("kind") or "")
            effect = float(metadata.get("pnl_effect") or 0.0)
            metric = "fees" if kind == "fee" else kind
            if metric in product:
                product[metric] += (
                    effect
                    if metric in {"funding", "realised_pnl"}
                    else _reported_cost(metric, metadata, effect)
                )
            for dimension in product["attribution"]:
                key = str(metadata.get(dimension) or "unattributed")
                current = product["attribution"][dimension].get(key, 0.0)
                product["attribution"][dimension][key] = current + effect

        for row in self._iter_rows(
            select(nav_snapshot.c.payload).order_by(nav_snapshot.c.created_at)
        ):
            snapshot = row["payload"]
            product_id = str(snapshot.get("product_id") or "unknown")
            product = products.setdefault(product_id, _empty_product())
            value = float(snapshot["nav"])
            product["nav_snapshot_count"] += 1
            product["nav"] = value
            product["peak_nav"] = (
                value if product["peak_nav"] is None else max(product["peak_nav"], value)
            )
            product["latest_nav_snapshot"] = snapshot

        for product_id, product in products.items():
            if product["nav"] is not None:
                peak = float(product["peak_nav"])
                product["drawdown"] = (
                    0.0 if peak <= 0 else max(0.0, (peak - float(product["nav"])) / peak)
                )
            if product_id == "btc_accumulation" and product["nav_snapshot_count"]:
                product["btc_performance"] = self._btc_performance()
        return products

    def _btc_performance(self) -> dict[str, Any]:
        previous: NavSnapshot | None = None
        latest: NavSnapshot | None = None
        peak = 0.0
        seconds_total = 0.0
        seconds_outside = 0.0
        missed = 0.0
        regime: dict[str, float] = {}
        statement = (
            select(nav_snapshot.c.payload)
            .where(nav_snapshot.c.payload["product_id"].as_string() == "btc_accumulation")
            .order_by(
                nav_snapshot.c.payload["observed_at"].as_string(),
                nav_snapshot.c.created_at,
                nav_snapshot.c.id,
            )
        )
        for row in self._iter_rows(statement):
            current = NavSnapshot(**row["payload"])
            if (
                current.product_id != "btc_accumulation"
                or current.accounting_asset.upper() != "BTC"
            ):
                raise ValueError("BTC performance received another product or accounting asset")
            if current.nav < 0:
                raise ValueError("BTC NAV cannot be negative")
            peak = max(peak, current.nav)
            if previous is not None:
                start = dt.datetime.fromisoformat(previous.observed_at)
                end = dt.datetime.fromisoformat(current.observed_at)
                seconds = max(0.0, (end - start).total_seconds())
                seconds_total += seconds
                previous_btc = float(previous.components.get("btc_balance", 0.0))
                previous_fraction = 0.0 if previous.nav <= 0 else previous_btc / previous.nav
                if previous_fraction < 1.0 - 1e-12:
                    seconds_outside += seconds
                prior_price = float(previous.components.get("stablecoin_per_btc", 0.0))
                current_price = float(current.components.get("stablecoin_per_btc", 0.0))
                stablecoin = float(previous.components.get("stablecoin_balance", 0.0))
                if prior_price > 0 and current_price > prior_price and stablecoin > 0:
                    missed += stablecoin / prior_price - stablecoin / current_price
                name = str(previous.components.get("regime") or "unclassified")
                regime[name] = regime.get(name, 0.0) + float(current.nav - previous.nav)
            previous = current
            latest = current
        if latest is None:
            raise ValueError("BTC performance requires NAV snapshots")

        fees = self._btc_fees_paid()
        btc_balance = float(latest.components.get("btc_balance", 0.0))
        price = float(latest.components.get("stablecoin_per_btc", 0.0))
        stablecoin = float(latest.components.get("stablecoin_balance", 0.0))
        stablecoin_btc = stablecoin / price if price > 0 else 0.0
        return {
            "btc_balance": btc_balance,
            "btc_nav": float(latest.nav),
            "btc_vs_passive_hold": (
                0.0
                if latest.passive_benchmark_nav is None
                else float(latest.nav - latest.passive_benchmark_nav)
            ),
            "time_outside_btc_fraction": (
                seconds_outside / seconds_total if seconds_total > 0 else 0.0
            ),
            "stablecoin_exposure_fraction": stablecoin_btc / latest.nav if latest.nav > 0 else 0.0,
            "missed_btc_appreciation": missed,
            "btc_drawdown_fraction": 0.0 if peak <= 0 else max(0.0, (peak - latest.nav) / peak),
            "fees_paid_btc": str(fees),
            "performance_by_regime": regime,
        }

    def _btc_fees_paid(self) -> Decimal:
        previous_hash = "0" * 64
        fees = Decimal("0")
        statement = (
            select(accounting_entry.c.payload)
            .where(accounting_entry.c.product_id == "btc_accumulation")
            .order_by(accounting_entry.c.sequence)
        )
        for row in self._iter_rows(statement):
            values = dict(row["payload"])
            saved_hash = values.pop("entry_hash")
            values["postings"] = {key: Decimal(value) for key, value in values["postings"].items()}
            entry = LedgerEntry(**values)
            if (
                entry.product_id != "btc_accumulation"
                or entry.accounting_asset != "BTC"
                or entry.previous_hash != previous_hash
                or entry.entry_hash != saved_hash
            ):
                raise ValueError("SQL ledger hash chain or identity is invalid")
            fees += entry.postings.get("expense:fees", Decimal("0"))
            previous_hash = entry.entry_hash
        return fees

    def _universe_attrition(self) -> dict[str, Any]:
        ranked = select(
            universe_snapshot.c.id,
            universe_snapshot.c.universe_id,
            universe_snapshot.c.observed_at,
            func.row_number()
            .over(
                partition_by=universe_snapshot.c.universe_id,
                order_by=(
                    universe_snapshot.c.observed_at.desc(),
                    universe_snapshot.c.id.desc(),
                ),
            )
            .label("recency"),
        ).subquery()
        by_snapshot: dict[str, dict[str, Any]] = {}
        with self._read_connection() as connection:
            snapshot_count = int(
                self._execute(
                    connection,
                    select(func.count()).select_from(universe_snapshot),
                ).scalar_one()
            )
            snapshot_rows = select(ranked.c.id, ranked.c.universe_id, ranked.c.observed_at).where(
                ranked.c.recency == 1
            )
            for row in self._iter_rows(snapshot_rows):
                by_snapshot[str(row["id"])] = {
                    "universe_id": row["universe_id"],
                    "observed_at": row["observed_at"],
                    "total_members": 0,
                    "eligible_members": 0,
                    "excluded_members": 0,
                    "exclusion_reasons": {},
                }

            member_rows = (
                select(
                    universe_member.c.snapshot_id,
                    universe_member.c.reason_code,
                    func.count(universe_member.c.id).label("total_members"),
                    func.sum(case((universe_member.c.eligible.is_(True), 1), else_=0)).label(
                        "eligible_members"
                    ),
                    func.sum(case((universe_member.c.eligible.is_(True), 0), else_=1)).label(
                        "excluded_members"
                    ),
                )
                .select_from(
                    universe_member.join(ranked, ranked.c.id == universe_member.c.snapshot_id)
                )
                .where(ranked.c.recency == 1)
                .group_by(universe_member.c.snapshot_id, universe_member.c.reason_code)
            )
            for row in self._iter_rows(member_rows):
                summary = by_snapshot[str(row["snapshot_id"])]
                total = int(row["total_members"] or 0)
                eligible = int(row["eligible_members"] or 0)
                excluded = int(row["excluded_members"] or 0)
                summary["total_members"] += total
                summary["eligible_members"] += eligible
                summary["excluded_members"] += excluded
                if excluded:
                    reason = str(row["reason_code"] or "unknown")
                    reasons = summary["exclusion_reasons"]
                    reasons[reason] = int(reasons.get(reason, 0)) + excluded

        latest = {str(summary["universe_id"]): summary for summary in by_snapshot.values()}
        return {
            "snapshot_count": snapshot_count,
            "latest_by_universe": {key: latest[key] for key in sorted(latest) if key},
        }

    def _operations(self, *, now: str) -> dict[str, Any]:
        blocked = self._decision_funnel_blocked()
        active_job_states = job.c.state.in_(("pending", "running", "dead_letter"))
        job_count = self._count_where(job, active_job_states)
        with self._read_connection() as connection:
            job_rows = (
                self._execute(
                    connection,
                    select(job)
                    .where(active_job_states)
                    .order_by(job.c.available_at)
                    .limit(REPORT_DETAIL_LIMIT),
                )
                .mappings()
                .all()
            )
            self._check_deadline()
        jobs = [to_primitive(dict(row)) for row in job_rows]
        account_authority = self._stale_account_authority(now)
        market_data = self._stale_market_data(now)
        missing_risk_data = self._missing_risk_data()
        recovery = self._unresolved_recovery()
        workers, worker_count = self._detail_rows(worker, order_by=worker.c.last_heartbeat.desc())
        alerts, alert_count = self._detail_payloads(alert, order_by=alert.c.created_at.desc())
        latest_heartbeats: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self._iter_rows(
            select(service_heartbeat).order_by(
                service_heartbeat.c.observed_at.desc(), service_heartbeat.c.id.desc()
            )
        ):
            key = (str(row["node_id"]), str(row["service_name"]))
            latest_heartbeats.setdefault(
                key,
                {
                    "service_name": row["service_name"],
                    "node_id": row["node_id"],
                    "observed_at": row["observed_at"],
                    "healthy": bool(row["healthy"]),
                    "payload": dict(row["payload"]),
                },
            )
        worker_roles = {
            str(row["role"] or "") for row in self._iter_rows(select(worker.c.role).distinct())
        }
        authority_conflicts = _execution_authority_conflicts(
            worker_roles,
            self._count_where(
                active_strategy_assignment,
                active_strategy_assignment.c.active.is_(True),
                active_strategy_assignment.c.execution_mode == "live",
            ),
        )
        operations = {
            "detail_limit": REPORT_DETAIL_LIMIT,
            "detail_counts": {
                "job_queue": job_count,
                "workers": worker_count,
                "alerts": alert_count,
            },
            "alert_delivery_failure_count": self._count_where(
                alert, alert.c.payload["event_type"].as_string() == "delivery_failed"
            ),
            "job_queue": jobs,
            "job_queue_detail_limit": 1000,
            "queue_progress": self._queue_progress(now=now),
            "workers": workers,
            "heartbeats": [latest_heartbeats[key] for key in sorted(latest_heartbeats)],
            "alerts": alerts,
            "decision_funnel_blocked": blocked,
            "slis": {
                "unresolved_recovery_count": recovery["count"],
                "unresolved_recovery": recovery,
                "stale_account_authority": account_authority,
                "stale_market_data": market_data,
                "missing_risk_data": missing_risk_data,
                "execution_authority_conflicts": authority_conflicts,
            },
        }
        if self.storage_root is not None:
            operations["storage_health"] = _storage_health(
                self.storage_root,
                self.spool_root or self.storage_root.parent / "runtime" / "events" / "spool",
            )
        return operations

    def _decision_funnel_blocked(self) -> dict[str, int]:
        stage = decision_trace_table.c.first_blocked_stage
        reason = case(
            *(
                (
                    stage == item.value,
                    decision_trace_table.c.payload["stages"][item.value]["reason_code"].as_string(),
                )
                for item in DecisionTraceStage
            ),
            else_="unknown",
        )
        grouped = self._grouped_counts(
            select(stage, reason, func.count()).where(stage.is_not(None)).group_by(stage, reason)
        )
        return {
            f"{stage_name}:{reason_code or 'unknown'}": int(count)
            for stage_name, reason_code, count in grouped
        }

    def _stale_account_authority(self, now: str) -> dict[str, Any]:
        ranked = select(
            account_snapshot.c.id,
            account_snapshot.c.account_id,
            account_snapshot.c.observed_at,
            account_snapshot.c.source,
            account_snapshot.c.payload,
            func.row_number()
            .over(
                partition_by=account_snapshot.c.account_id,
                order_by=(account_snapshot.c.observed_at.desc(), account_snapshot.c.id.desc()),
            )
            .label("recency"),
        ).subquery()
        stale: list[dict[str, Any]] = []
        stale_count = 0
        statement = select(ranked).where(ranked.c.recency == 1).order_by(ranked.c.account_id)
        for row in self._iter_rows(statement):
            account_id = str(row.get("account_id") or "")
            if not account_id:
                continue
            age = _age_seconds(str(row["observed_at"]), now)
            raw_payload = row.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
            if (
                payload.get("account_state_known") is not True
                or age > self.account_stale_after_seconds
            ):
                stale_count += 1
                if len(stale) < REPORT_DETAIL_LIMIT:
                    stale.append(
                        {
                            "account_id": account_id,
                            "snapshot_id": row["id"],
                            "observed_at": row["observed_at"],
                            "age_seconds": age,
                            "account_state_known": payload.get("account_state_known") is True,
                            "source": row.get("source"),
                        }
                    )
        return {
            "count": stale_count,
            "threshold_seconds": self.account_stale_after_seconds,
            "accounts": stale,
            "detail_limit": REPORT_DETAIL_LIMIT,
        }

    def _stale_market_data(self, now: str) -> dict[str, Any]:
        stale: list[dict[str, Any]] = []
        stale_count = 0
        for row in self._latest_risk_rows("market_data_input", per_instrument=True):
            raw_payload = row.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
            if payload.get("kind") != "market_data_input":
                continue
            product_id = str(payload.get("product_id") or "")
            instrument_id = str(payload.get("instrument_id") or "")
            if (product_id, instrument_id) == ("", ""):
                continue
            observed_at = str(payload.get("availability_time") or row["created_at"])
            age = _age_seconds(observed_at, now)
            if age > self.market_data_stale_after_seconds:
                stale_count += 1
                if len(stale) < REPORT_DETAIL_LIMIT:
                    stale.append(
                        {
                            "product_id": product_id,
                            "instrument_id": instrument_id,
                            "snapshot_id": row["id"],
                            "observed_at": observed_at,
                            "age_seconds": age,
                        }
                    )
        return {
            "count": stale_count,
            "threshold_seconds": self.market_data_stale_after_seconds,
            "instruments": stale,
            "detail_limit": REPORT_DETAIL_LIMIT,
        }

    def _unresolved_recovery(self) -> dict[str, Any]:
        resolutions = (
            select(
                reconciliation_event.c.payload["recovery_plan_id"]
                .as_string()
                .label("recovery_plan_id")
            )
            .where(
                reconciliation_event.c.payload["record_type"].as_string() == "recovery_resolution"
            )
            .subquery()
        )
        unresolved_plan = and_(
            reconciliation_event.c.payload["record_type"].as_string() == "recovery_plan",
            ~exists(
                select(resolutions.c.recovery_plan_id).where(
                    resolutions.c.recovery_plan_id == reconciliation_event.c.id
                )
            ),
        )
        plan_count = self._count_where(reconciliation_event, unresolved_plan)
        plan_rows, _ = self._detail_rows(
            reconciliation_event,
            order_by=reconciliation_event.c.created_at.desc(),
            where=unresolved_plan,
        )
        plans = [
            {"plan_id": row["id"], **dict(row["payload"].get("plan") or {})} for row in plan_rows
        ]
        recovery_orders, recovery_order_count = self._latest_status_details(
            exchange_order, key="order_id", status="recovery_required"
        )
        failed_stops, failed_stop_count = self._latest_status_details(
            protective_stop, key="stop_id", status="confirmation_failed"
        )
        return {
            "count": plan_count + recovery_order_count + failed_stop_count,
            "plans": plans,
            "orders": recovery_orders,
            "protective_stops": failed_stops,
            "detail_limit": REPORT_DETAIL_LIMIT,
            "detail_counts": {
                "plans": plan_count,
                "orders": recovery_order_count,
                "protective_stops": failed_stop_count,
            },
        }

    def _missing_risk_data(self) -> dict[str, Any]:
        missing: list[dict[str, Any]] = []
        missing_count = 0
        for row in self._latest_risk_rows("canonical_portfolio_risk_state"):
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if payload.get("kind") != "canonical_portfolio_risk_state":
                continue
            product_id = str(payload.get("product_id") or "")
            if not product_id:
                continue
            values = payload.get("risk_data_missing")
            if payload.get("risk_data_available") is False or values:
                missing_count += 1
                if len(missing) < REPORT_DETAIL_LIMIT:
                    missing.append(
                        {
                            "snapshot_id": row["id"],
                            "product_id": payload.get("product_id"),
                            "observed_at": payload.get("observed_at", row.get("created_at")),
                            "risk_data_available": payload.get("risk_data_available"),
                            "risk_data_missing": list(values)
                            if isinstance(values, list | tuple)
                            else [],
                        }
                    )
        return {
            "count": missing_count,
            "snapshots": missing,
            "detail_limit": REPORT_DETAIL_LIMIT,
        }

    def _latest_risk_rows(self, kind, *, per_instrument=False):
        source = market_input_sources if kind == "market_data_input" else risk_snapshot
        groups = [source.c.payload["product_id"].as_string()]
        if per_instrument:
            groups.append(source.c.payload["instrument_id"].as_string())
        ranked = (
            select(
                source.c.id,
                func.row_number()
                .over(
                    partition_by=groups,
                    order_by=(source.c.created_at.desc(), source.c.id.desc()),
                )
                .label("recency"),
            )
            .where(source.c.payload["kind"].as_string() == kind)
            .subquery()
        )
        return self._iter_rows(
            select(source).join(ranked, ranked.c.id == source.c.id).where(ranked.c.recency == 1)
        )

    def _latest_status_details(self, table, *, key: str, status: str):
        ranked = select(
            table.c.id,
            getattr(table.c, key),
            table.c.status,
            table.c.sequence,
            func.row_number()
            .over(
                partition_by=getattr(table.c, key),
                order_by=(table.c.sequence.desc(), table.c.id.desc()),
            )
            .label("recency"),
        ).subquery()
        latest = select(
            ranked.c.id,
            getattr(ranked.c, key),
            ranked.c.status,
        ).where(ranked.c.recency == 1, ranked.c.status == status)
        count = self._count_select(latest)
        rows = latest.order_by(getattr(ranked.c, key)).limit(REPORT_DETAIL_LIMIT)
        details = [
            {key: row[key], "status": row["status"], "event_id": row["id"]}
            for row in self._iter_rows(rows)
        ]
        return details, count

    def _count(self, table) -> int:
        with self._read_connection() as connection:
            value = self._execute(connection, select(func.count()).select_from(table)).scalar_one()
            self._check_deadline()
            return int(value)


def _age_seconds(observed_at: str, now: str) -> float:
    current = dt.datetime.fromisoformat(timestamp(now, field="now"))
    observed = dt.datetime.fromisoformat(timestamp(observed_at, field="observed_at"))
    return max(0.0, (current - observed).total_seconds())


def _storage_health(data_root: Path, spool_root: Path) -> dict[str, Any]:
    return {
        "filesystem": _filesystem_inode_health(data_root),
        "spool_inventory": _spool_inventory(
            spool_root,
            spool_root.parent / "feature-spool",
            stale_cutoff=dt.datetime.now(dt.UTC).timestamp() - 7 * 86_400,
        ),
    }


def _filesystem_inode_health(path: Path) -> dict[str, Any]:
    try:
        stats = os.statvfs(path)
    except OSError as exc:
        return {
            "path": str(path),
            "scope": "backing_filesystem",
            "available": False,
            "reason": f"statvfs_failed:{type(exc).__name__}",
        }
    available = stats.f_files > 0
    return {
        "path": str(path),
        "scope": "backing_filesystem",
        "available": available,
        "total_inodes": int(stats.f_files) if available else None,
        "free_inodes": int(stats.f_ffree) if available else None,
        "available_inodes": int(stats.f_favail) if available else None,
    }


def _spool_inventory(
    spool_root: Path,
    feature_spool: Path,
    *,
    stale_cutoff: float,
    entry_limit_per_directory: int = 256,
) -> dict[str, Any]:
    states = ("pending", "claimed", "committed", "deleted")
    counts = dict.fromkeys(states, 0)
    observed_bytes = 0
    stale_files = 0
    unreadable_directories = 0
    truncated = False
    for state in states:
        for root in (spool_root, feature_spool):
            try:
                entries = os.scandir(root / state)
            except FileNotFoundError:
                continue
            except OSError:
                unreadable_directories += 1
                continue
            with entries:
                for index, entry in enumerate(entries):
                    if index >= entry_limit_per_directory:
                        truncated = True
                        break
                    if not entry.name.startswith("segment-") or not entry.name.endswith(".jsonl"):
                        continue
                    try:
                        if entry.is_file(follow_symlinks=False):
                            details = entry.stat(follow_symlinks=False)
                        else:
                            continue
                    except OSError:
                        unreadable_directories += 1
                        continue
                    counts[state] += 1
                    observed_bytes += details.st_size
                    stale_files += int(details.st_mtime < stale_cutoff)
    return {
        "scope": "bounded_inventory_sample",
        "entry_limit_per_directory": entry_limit_per_directory,
        "files_observed": sum(counts.values()),
        "files_observed_by_state": counts,
        "bytes_observed": observed_bytes,
        "stale_files_over_7_days": stale_files,
        "unreadable_entries": unreadable_directories,
        "truncated": truncated,
    }


def _candidate_is_active(row: Mapping[str, Any]) -> bool:
    state = str(row.get("state") or "queued")
    metadata = row.get("metadata")
    disposition = terminal_trial_disposition(
        state, metadata if isinstance(metadata, Mapping) else None
    )
    return (
        disposition is None
        and state
        not in {
            "completed",
            "rejected",
            "retired",
            "protected",
            "forward",
            "forward_paper",
            "live_ready",
            "live_canary",
            "live",
            "suspended",
        }
        and not state.endswith("_rejected")
        and not state.startswith("blocked_")
    )


def _execution_authority_conflicts(roles: set[str], active_live_assignments: int) -> dict[str, Any]:
    old_services = sorted(role for role in roles if role in {"autopilot", "legacy-execution"})
    new_services = sorted(role for role in roles if role in {"execution-engine", "live-execution"})
    return {
        "conflict": bool(old_services and new_services),
        "old_services": old_services,
        "new_services": new_services,
        "active_live_assignments": active_live_assignments,
    }
