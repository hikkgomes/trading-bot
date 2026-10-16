"""Database-backed control report for trading, research, products, and operations."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.engine import Engine

from src.accounting.btc_performance import build_btc_performance_report
from src.accounting.ledger import Ledger, SqlLedgerStore
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
from src.domain._codec import timestamp, to_primitive
from src.observability.decision_trace import SqlDecisionTraceStore
from src.observability.queue_progress import active_candidate_jobs, queue_progress
from src.observability.research_progress import (
    current_stages,
    epoch_progress,
    measured_progress,
)
from src.research.outcomes import terminal_trial_disposition
from src.services.health import DatabaseHeartbeatStore


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

    def build(self, *, now: str | None = None) -> dict[str, Any]:
        report_time = timestamp(now or dt.datetime.now(dt.UTC), field="report time")
        return {
            "schema": "platform.operator_report/v1",
            "trading": self._trading(),
            "research": self._research(now=report_time),
            "products": self._products(),
            "operations": self._operations(now=report_time),
        }

    def _payloads(self, table, *, order_by=None) -> list[dict[str, Any]]:
        statement = select(table.c.payload)
        if order_by is not None:
            statement = statement.order_by(order_by)
        with self.engine.connect() as connection:
            return [dict(item) for item in connection.execute(statement).scalars()]

    def _rows(self, table, *, order_by=None) -> list[dict[str, Any]]:
        statement = select(table)
        if order_by is not None:
            statement = statement.order_by(order_by)
        with self.engine.connect() as connection:
            return [to_primitive(dict(item)) for item in connection.execute(statement).mappings()]

    def _trading(self) -> dict[str, Any]:
        positions = self._payloads(position, order_by=position.c.created_at.desc())
        targets = self._payloads(target_position, order_by=target_position.c.created_at.desc())
        forecasts = self._payloads(alpha_forecast, order_by=alpha_forecast.c.created_at.desc())
        stops = self._payloads(protective_stop, order_by=protective_stop.c.created_at.desc())
        return {
            "positions": positions,
            "target_positions": targets,
            "alpha_forecasts": forecasts,
            "stops": stops,
            "active_strategy_assignments": self._rows(
                active_strategy_assignment,
                order_by=active_strategy_assignment.c.assigned_at.desc(),
            ),
            "account_balances": self._payloads(
                balance_snapshot, order_by=balance_snapshot.c.created_at.desc()
            ),
            "order_events": self._count(exchange_order),
            "fills": self._count(fill),
        }

    def _research(self, *, now: str) -> dict[str, Any]:
        experiments = self._rows(experiment, order_by=experiment.c.submitted_at.desc())
        diagnostic_ids = self._diagnostic_candidate_ids(experiments)
        production_candidates = [row for row in experiments if str(row["id"]) not in diagnostic_ids]
        results = self._rows(validation_result)
        report = {
            "candidate_queue": [
                item for item in production_candidates if _candidate_is_active(item)
            ],
            "experiments": experiments,
            "diagnostic_candidate_count": len(diagnostic_ids),
            "diagnostic_candidate_ids": sorted(diagnostic_ids),
            "validation_results": results,
            "validation_stages": self._rows(validation_stage),
            "holdout_claims": self._rows(holdout_claim),
            "holdout_outcomes": self._rows(holdout_outcome),
            "forward_evidence": self._rows(forward_evidence),
            "forward_paper_observations": self._rows(forward_paper_observation),
            "forward_paper_summaries": self._rows(forward_paper_summary),
            "forward_paper_decisions": self._rows(forward_paper_decision),
            "active_strategy_assignments": self._rows(
                active_strategy_assignment,
                order_by=active_strategy_assignment.c.assigned_at.desc(),
            ),
            "strategy_identities": self._rows(strategy_identity),
            "strategy_lineage": self._rows(strategy_lineage),
            "research_theses": self._rows(research_thesis),
            "thesis_trials": self._rows(thesis_trial),
            "universe_attrition": self._universe_attrition(),
            "strategy_artefacts": self._rows(strategy_artefact),
            "strategy_approvals": self._rows(strategy_approval),
            "production_preflights": self._rows(production_preflight),
            "promotion_events": self._rows(promotion_event),
            "dataset_bundles": self._rows(dataset_bundle),
            "rejection_reasons": [
                item.get("reason_code") for item in results if item.get("reason_code")
            ],
            "agent_activity": self._payloads(
                agent_action, order_by=agent_action.c.created_at.desc()
            ),
        }
        report["funnel"] = self._research_funnel(
            production_candidates,
            results,
            report,
            now=now,
            diagnostic_candidate_ids=diagnostic_ids,
        )
        return report

    def _diagnostic_candidate_ids(self, experiments: list[dict[str, Any]]) -> set[str]:
        diagnostic_ids = {
            str(row["id"])
            for row in experiments
            if isinstance(row.get("metadata"), Mapping)
            and row["metadata"].get("diagnostic") is True
        }
        versions = {str(row["strategy_version_id"]) for row in experiments}
        if not versions:
            return diagnostic_ids
        statement = (
            select(strategy_version.c.id, strategy_definition.c.definition)
            .select_from(
                strategy_version.join(
                    strategy_definition,
                    strategy_version.c.definition_id == strategy_definition.c.id,
                )
            )
            .where(strategy_version.c.id.in_(versions))
        )
        with self.engine.connect() as connection:
            definitions = connection.execute(statement).all()
        diagnostic_versions = set()
        for version_id, raw_definition in definitions:
            if not isinstance(raw_definition, Mapping):
                continue
            metadata = raw_definition.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("diagnostic") is True:
                diagnostic_versions.add(str(version_id))
        return diagnostic_ids | {
            str(row["id"])
            for row in experiments
            if str(row["strategy_version_id"]) in diagnostic_versions
        }

    def _research_funnel(
        self,
        experiments: list[dict[str, Any]],
        results: list[dict[str, Any]],
        research: dict[str, Any],
        *,
        now: str,
        diagnostic_candidate_ids: set[str],
    ) -> dict[str, Any]:
        all_stages = current_stages(
            research["validation_stages"], self._rows(evidence_invalidation)
        )
        stages = [
            row for row in all_stages if str(row["experiment_id"]) not in diagnostic_candidate_ids
        ]
        diagnostic_stage_count = len(all_stages) - len(stages)
        identities = list(research["strategy_identities"])
        schedules = self._rows(platform_schedule)
        rejection_by_stage: dict[str, int] = {}
        rejection_reasons: dict[str, int] = {}
        first_blocked: dict[str, int] = {}
        evaluated_ids: set[str] = set()
        signal_frequencies: list[float] = []
        for row in stages:
            candidate_id = str(row["experiment_id"])
            evaluated_ids.add(candidate_id)
            raw_payload = row.get("payload")
            payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
            raw_evidence = payload.get("evidence")
            evidence: dict[str, Any] = raw_evidence if isinstance(raw_evidence, dict) else payload
            frequency = evidence.get("signal_frequency")
            if isinstance(frequency, int | float) and not isinstance(frequency, bool):
                signal_frequencies.append(float(frequency))
            if not row["accepted"]:
                stage = str(row["stage"])
                reason = str(row.get("reason_code") or "unknown")
                rejection_by_stage[stage] = rejection_by_stage.get(stage, 0) + 1
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        by_candidate: dict[str, list[dict[str, Any]]] = {}
        for row in stages:
            by_candidate.setdefault(str(row["experiment_id"]), []).append(row)
        for candidate_stages in by_candidate.values():
            ordered = sorted(candidate_stages, key=lambda item: str(item["evaluated_at"]))
            blocked = next((item for item in ordered if not item["accepted"]), None)
            if blocked is not None:
                key = str(blocked["stage"])
                first_blocked[key] = first_blocked.get(key, 0) + 1
        candidate_ages = _candidate_age_by_state(experiments, now=now)
        deferred_by_stage = _deferred_candidates_by_stage(experiments)
        missing_stage_datasets = _missing_stage_datasets(
            experiments,
            bundles=research["dataset_bundles"],
            snapshots=self._rows(dataset_snapshot),
        )
        candidate_job_ids = active_candidate_jobs(self.engine)
        candidates_without_job_or_reason = _candidates_without_job_or_reason(
            experiments,
            candidate_job_ids=candidate_job_ids,
        )
        active_forward_count = sum(
            row.get("active") is True
            and row.get("execution_mode") == "paper"
            and row.get("lifecycle_state") == "forward_paper"
            for row in research["active_strategy_assignments"]
        )
        definitions = self._rows(strategy_definition)
        feature_families: dict[str, int] = {}
        thesis_families: dict[str, int] = {}
        for row in definitions:
            raw_definition = row.get("definition")
            definition: dict[str, Any] = raw_definition if isinstance(raw_definition, dict) else {}
            family = str(definition.get("family") or "unknown")
            feature_families[family] = feature_families.get(family, 0) + 1
        for row in research["research_theses"]:
            raw_payload = row.get("payload")
            payload = raw_payload if isinstance(raw_payload, dict) else {}
            family = str(payload.get("mechanism_category") or "unknown")
            thesis_families[family] = thesis_families.get(family, 0) + 1
        duplicate_count = sum(bool(row.get("is_duplicate")) for row in identities)
        near_duplicate_count = sum(
            bool((row.get("metadata") or {}).get("near_duplicate_of"))
            for row in identities
            if isinstance(row.get("metadata"), dict)
        )
        runs = self._rows(experiment_run)
        diagnostic_run_ids = _diagnostic_run_ids(runs, diagnostic_candidate_ids)
        measured_runs = [row for row in runs if str(row["id"]) not in diagnostic_run_ids]
        return {
            **measured_progress(measured_runs, ()),
            **queue_progress(self.engine, now=now),
            **epoch_progress(self._rows(research_epoch), self._rows(research_round)),
            "theses_generated": self._count(research_thesis),
            "candidates_generated": len(experiments),
            "diagnostic_candidates_excluded": len(diagnostic_candidate_ids),
            "diagnostic_run_count": len(diagnostic_run_ids),
            "diagnostic_validation_stage_count": diagnostic_stage_count,
            "candidates_compiled": sum(
                row["stage"] == "screening" and row["accepted"] for row in stages
            ),
            "candidates_evaluated": len(evaluated_ids),
            "historical_validation_stage_count": len(research["validation_stages"])
            - len(all_stages),
            "candidates_rejected_by_stage": rejection_by_stage,
            "candidates_deferred_by_stage": deferred_by_stage,
            "top_rejection_reasons": dict(
                sorted(rejection_reasons.items(), key=lambda item: (-item[1], item[0]))[:10]
            ),
            "signal_frequency_distribution": sorted(signal_frequencies),
            "feature_family_concentration": feature_families,
            "thesis_family_coverage": thesis_families,
            "universe_attrition": research["universe_attrition"],
            "candidate_correlation": [
                row.get("payload", {}).get("candidate_correlation")
                for row in stages
                if isinstance(row.get("payload"), dict)
                and row.get("payload", {}).get("candidate_correlation") is not None
            ],
            "exact_duplicate_rate": duplicate_count / len(identities) if identities else 0.0,
            "near_duplicate_rate": near_duplicate_count / len(identities) if identities else 0.0,
            "cumulative_trial_count": self._count(thesis_trial),
            "protected_holdout_count": len(research["holdout_outcomes"]),
            "forward_paper_count": len(research["forward_paper_observations"]),
            "active_forward_count": active_forward_count,
            "strategy_promotions": sum(
                _promotion_advanced(row.get("payload")) for row in research["promotion_events"]
            ),
            "first_blocked_stage": first_blocked,
            "candidates_never_evaluated": sorted(
                str(row["id"]) for row in experiments if str(row["id"]) not in evaluated_ids
            ),
            "candidate_age_by_state": candidate_ages,
            "missing_stage_dataset_count": sum(missing_stage_datasets.values()),
            "missing_stage_datasets": missing_stage_datasets,
            "schedule_states": {str(row["job_name"]): str(row["state"]) for row in schedules},
            "candidates_without_job_or_reason": candidates_without_job_or_reason,
        }

    def _products(self) -> dict[str, Any]:
        entries = self._payloads(accounting_entry, order_by=accounting_entry.c.created_at)
        navs = self._payloads(nav_snapshot, order_by=nav_snapshot.c.created_at)
        products: dict[str, dict[str, Any]] = {}
        for entry in entries:
            product_id = str(entry.get("product_id") or "unknown")
            product = products.setdefault(
                product_id,
                {
                    "fees": 0.0,
                    "funding": 0.0,
                    "slippage": 0.0,
                    "realised_pnl": 0.0,
                    "nav": None,
                    "peak_nav": None,
                    "drawdown": None,
                    "attribution": {
                        "strategy": {},
                        "symbol": {},
                        "sleeve": {},
                        "product": {},
                    },
                },
            )
            raw_metadata = entry.get("metadata")
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
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
        nav_by_product: dict[str, list[dict[str, Any]]] = {}
        for snapshot in navs:
            nav_by_product.setdefault(str(snapshot.get("product_id") or "unknown"), []).append(
                snapshot
            )
        for product_id, snapshots in nav_by_product.items():
            product = products.setdefault(
                product_id,
                {
                    "fees": 0.0,
                    "funding": 0.0,
                    "slippage": 0.0,
                    "realised_pnl": 0.0,
                    "attribution": {
                        "strategy": {},
                        "symbol": {},
                        "sleeve": {},
                        "product": {},
                    },
                },
            )
            values = [float(item["nav"]) for item in snapshots]
            latest = values[-1]
            peak = max(values)
            product.update(
                {
                    "nav": latest,
                    "peak_nav": peak,
                    "drawdown": 0.0 if peak <= 0 else max(0.0, (peak - latest) / peak),
                    "latest_nav_snapshot": snapshots[-1],
                }
            )
            if product_id == "btc_accumulation":
                btc = build_btc_performance_report(
                    tuple(NavSnapshot(**item) for item in snapshots),
                    ledger=Ledger(
                        product_id="btc_accumulation",
                        accounting_asset="BTC",
                        store=SqlLedgerStore(self.engine, product_id="btc_accumulation"),
                    ),
                )
                product["btc_performance"] = {
                    **btc.__dict__,
                    "fees_paid_btc": str(btc.fees_paid_btc),
                }
        return products

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
        with self.engine.connect() as connection:
            snapshot_count = int(
                connection.execute(select(func.count()).select_from(universe_snapshot)).scalar_one()
            )
            snapshots = (
                connection.execute(
                    select(ranked.c.id, ranked.c.universe_id, ranked.c.observed_at).where(
                        ranked.c.recency == 1
                    )
                )
                .mappings()
                .all()
            )
            latest_ids = [str(row["id"]) for row in snapshots]
            member_rows = []
            if latest_ids:
                member_rows = (
                    connection.execute(
                        select(
                            universe_member.c.snapshot_id,
                            universe_member.c.reason_code,
                            func.count(universe_member.c.id).label("total_members"),
                            func.sum(
                                case((universe_member.c.eligible.is_(True), 1), else_=0)
                            ).label("eligible_members"),
                            func.sum(
                                case((universe_member.c.eligible.is_(True), 0), else_=1)
                            ).label("excluded_members"),
                        )
                        .where(universe_member.c.snapshot_id.in_(latest_ids))
                        .group_by(universe_member.c.snapshot_id, universe_member.c.reason_code)
                    )
                    .mappings()
                    .all()
                )
        by_snapshot = {
            str(row["id"]): {
                "universe_id": row["universe_id"],
                "observed_at": row["observed_at"],
                "total_members": 0,
                "eligible_members": 0,
                "excluded_members": 0,
                "exclusion_reasons": {},
            }
            for row in snapshots
        }
        for row in member_rows:
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
        latest = {str(row["universe_id"]): by_snapshot[str(row["id"])] for row in snapshots}
        return {
            "snapshot_count": snapshot_count,
            "latest_by_universe": {key: latest[key] for key in sorted(latest) if key},
        }

    def _operations(self, *, now: str) -> dict[str, Any]:
        traces = SqlDecisionTraceStore(self.engine).read()
        blocked: dict[str, int] = {}
        for _identity, trace in traces:
            if trace.first_blocked_stage:
                detail = trace.stages[trace.first_blocked_stage]
                key = f"{trace.first_blocked_stage}:{detail.get('reason_code', 'unknown')}"
                blocked[key] = blocked.get(key, 0) + 1
        with self.engine.connect() as connection:
            jobs = [
                to_primitive(dict(row))
                for row in connection.execute(
                    select(job)
                    .where(job.c.state.in_(("pending", "running", "dead_letter")))
                    .order_by(job.c.available_at)
                    .limit(1000)
                ).mappings()
            ]
        account_authority = self._stale_account_authority(now)
        market_data = self._stale_market_data(now)
        missing_risk_data = self._missing_risk_data()
        recovery = self._unresolved_recovery()
        authority_conflicts = _execution_authority_conflicts(
            self._rows(worker), self._rows(active_strategy_assignment)
        )
        operations = {
            "job_queue": jobs,
            "job_queue_detail_limit": 1000,
            "queue_progress": queue_progress(self.engine, now=now),
            "workers": self._rows(worker, order_by=worker.c.last_heartbeat.desc()),
            "heartbeats": [
                heartbeat.__dict__ for heartbeat in DatabaseHeartbeatStore(self.engine).latest()
            ],
            "alerts": self._payloads(alert, order_by=alert.c.created_at.desc()),
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

    def _stale_account_authority(self, now: str) -> dict[str, Any]:
        rows = self._rows(account_snapshot, order_by=account_snapshot.c.observed_at.desc())
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            account_id = str(row.get("account_id") or "")
            if account_id and account_id not in latest:
                latest[account_id] = row
        stale: list[dict[str, Any]] = []
        for account_id, row in sorted(latest.items()):
            age = _age_seconds(str(row["observed_at"]), now)
            raw_payload = row.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
            if (
                payload.get("account_state_known") is not True
                or age > self.account_stale_after_seconds
            ):
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
            "count": len(stale),
            "threshold_seconds": self.account_stale_after_seconds,
            "accounts": stale,
        }

    def _stale_market_data(self, now: str) -> dict[str, Any]:
        rows = self._latest_risk_rows("market_data_input", per_instrument=True)
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            raw_payload = row.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
            if payload.get("kind") != "market_data_input":
                continue
            key = (str(payload.get("product_id") or ""), str(payload.get("instrument_id") or ""))
            if key != ("", "") and key not in latest:
                latest[key] = row
        stale: list[dict[str, Any]] = []
        for (product_id, instrument_id), row in sorted(latest.items()):
            payload = row["payload"]
            observed_at = str(payload.get("availability_time") or row["created_at"])
            age = _age_seconds(observed_at, now)
            if age > self.market_data_stale_after_seconds:
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
            "count": len(stale),
            "threshold_seconds": self.market_data_stale_after_seconds,
            "instruments": stale,
        }

    def _unresolved_recovery(self) -> dict[str, Any]:
        plans = []
        resolved_plan_ids: set[str] = set()
        for row in self._rows(reconciliation_event):
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("record_type") == "recovery_resolution":
                plan_id = str(payload.get("recovery_plan_id") or "")
                if plan_id:
                    resolved_plan_ids.add(plan_id)
            elif (
                payload.get("record_type") == "recovery_plan"
                and str(row["id"]) not in resolved_plan_ids
            ):
                plans.append({"plan_id": row["id"], **dict(payload.get("plan") or {})})
        plans = [plan for plan in plans if plan["plan_id"] not in resolved_plan_ids]
        recovery_orders = _latest_statuses(
            self._rows(exchange_order), key="order_id", status="recovery_required"
        )
        failed_stops = _latest_statuses(
            self._rows(protective_stop), key="stop_id", status="confirmation_failed"
        )
        return {
            "count": len(plans) + len(recovery_orders) + len(failed_stops),
            "plans": plans,
            "orders": recovery_orders,
            "protective_stops": failed_stops,
        }

    def _missing_risk_data(self) -> dict[str, Any]:
        missing: list[dict[str, Any]] = []
        latest: dict[str, dict[str, Any]] = {}
        for row in self._latest_risk_rows("canonical_portfolio_risk_state"):
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if payload.get("kind") != "canonical_portfolio_risk_state":
                continue
            product_id = str(payload.get("product_id") or "")
            if product_id and product_id not in latest:
                latest[product_id] = row
        for row in latest.values():
            payload = row["payload"]
            values = payload.get("risk_data_missing")
            if payload.get("risk_data_available") is False or values:
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
        return {"count": len(missing), "snapshots": missing}

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
        with self.engine.connect() as connection:
            return (
                connection.execute(
                    select(source)
                    .join(ranked, ranked.c.id == source.c.id)
                    .where(ranked.c.recency == 1)
                )
                .mappings()
                .all()
            )

    def _count(self, table) -> int:
        with self.engine.connect() as connection:
            return int(connection.execute(select(func.count()).select_from(table)).scalar_one())


def _promotion_advanced(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    return int(
        payload.get("accepted") is True
        and str(payload.get("prior_state")) != str(payload.get("next_state"))
    )


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


def _candidate_age_by_state(
    experiments: list[dict[str, Any]], *, now: str
) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = {}
    for row in experiments:
        state = str(row.get("state") or "unknown")
        values.setdefault(state, []).append(_age_seconds(str(row["submitted_at"]), now))
    return {
        state: {
            "count": len(ages),
            "oldest_seconds": max(ages),
            "average_seconds": sum(ages) / len(ages),
        }
        for state, ages in sorted(values.items())
    }


def _deferred_candidates_by_stage(experiments: list[dict[str, Any]]) -> dict[str, int]:
    deferred: dict[str, int] = {}
    for row in experiments:
        state = str(row.get("state") or "")
        if state.endswith("_deferred"):
            stage = state.removesuffix("_deferred")
            deferred[stage] = deferred.get(stage, 0) + 1
    return dict(sorted(deferred.items()))


def _missing_stage_datasets(
    experiments: list[dict[str, Any]],
    *,
    bundles: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> dict[str, int]:
    bundle_by_id = {str(row["id"]): row for row in bundles}
    snapshot_ids = {str(row["id"]) for row in snapshots}
    missing: dict[str, int] = {}
    required_roles = {
        "queued": "screening",
        "screening": "development",
        "development": "robustness",
        "robustness": "protected_holdout",
    }
    for candidate in experiments:
        state = str(candidate.get("state") or "")
        role = (
            state.removeprefix("waiting_for_dataset:")
            if state.startswith("waiting_for_dataset:")
            else required_roles.get(state, "")
        )
        if not role:
            continue
        stage_ids: Mapping[str, Any] = {}
        metadata = candidate.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("_dataset_plan"), dict):
            plan = metadata["_dataset_plan"]
            plan_fields = {
                "screening": "screening_snapshot_ids",
                "development": "development_snapshot_ids",
                "robustness": "robustness_snapshot_ids",
                "protected_holdout": "protected_holdout_snapshot_id",
            }
            field = plan_fields.get(role)
            if field is not None:
                value = plan.get(field)
                stage_ids = {role: value} if field.endswith("snapshot_id") else {role: value}
        if not stage_ids:
            bundle = bundle_by_id.get(str(candidate.get("dataset_bundle_id") or ""))
            payload = bundle.get("payload") if isinstance(bundle, dict) else None
            stage_ids = payload.get("stage_snapshot_ids", {}) if isinstance(payload, dict) else {}
        value = stage_ids.get(role)
        values = value if isinstance(value, list | tuple) else [value]
        if not values or any(item is None or str(item) not in snapshot_ids for item in values):
            missing[role] = missing.get(role, 0) + 1
    return dict(sorted(missing.items()))


def _candidates_without_job_or_reason(
    experiments: list[dict[str, Any]], *, candidate_job_ids: set[str]
) -> list[str]:
    missing: list[str] = []
    for row in experiments:
        candidate_id = str(row["id"])
        metadata = row.get("metadata")
        has_reason = isinstance(metadata, dict) and bool(
            metadata.get("dataset_waiting")
            or metadata.get("blocked_reason")
            or metadata.get("evaluation_deferred")
        )
        if (
            _candidate_is_active(row)
            and candidate_id not in candidate_job_ids
            and not has_reason
            and not str(row.get("state") or "").startswith("waiting_")
        ):
            missing.append(candidate_id)
    return sorted(missing)


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


def _diagnostic_run_ids(runs: list[dict[str, Any]], diagnostic_candidate_ids: set[str]) -> set[str]:
    result: set[str] = set()
    for row in runs:
        raw_payload = row.get("payload")
        if not isinstance(raw_payload, Mapping):
            continue
        candidate_id = str(raw_payload.get("candidate_id") or "")
        receipt = raw_payload.get("receipt")
        if not candidate_id and isinstance(receipt, Mapping):
            candidate_id = str(receipt.get("candidate_id") or "")
        if candidate_id in diagnostic_candidate_ids:
            result.add(str(row["id"]))
    return result


def _latest_statuses(rows: list[dict[str, Any]], *, key: str, status: str) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or row.get("id") or "")
        previous = latest.get(identity)
        if previous is None or int(row.get("sequence") or 0) >= int(previous.get("sequence") or 0):
            latest[identity] = row
    return [
        {key: identity, "status": row.get("status"), "event_id": row.get("id")}
        for identity, row in sorted(latest.items())
        if str(row.get("status")) == status
    ]


def _execution_authority_conflicts(
    workers: list[dict[str, Any]], assignments: list[dict[str, Any]]
) -> dict[str, Any]:
    roles = {str(row.get("role") or "") for row in workers}
    old_services = sorted(role for role in roles if role in {"autopilot", "legacy-execution"})
    new_services = sorted(role for role in roles if role in {"execution-engine", "live-execution"})
    active_live = sum(
        row.get("active") is True and row.get("execution_mode") == "live" for row in assignments
    )
    return {
        "conflict": bool(old_services and new_services),
        "old_services": old_services,
        "new_services": new_services,
        "active_live_assignments": active_live,
    }
