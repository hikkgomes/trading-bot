"""Database-backed control report for trading, research, products, and operations."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from src.accounting.btc_performance import build_btc_performance_report
from src.accounting.ledger import Ledger, SqlLedgerStore
from src.accounting.nav import NavSnapshot
from src.data.database import (
    accounting_entry,
    active_strategy_assignment,
    agent_action,
    alert,
    alpha_forecast,
    balance_snapshot,
    exchange_order,
    experiment,
    fill,
    forward_evidence,
    forward_paper_observation,
    holdout_claim,
    holdout_outcome,
    job,
    nav_snapshot,
    position,
    production_preflight,
    promotion_event,
    protective_stop,
    strategy_approval,
    strategy_artefact,
    strategy_identity,
    strategy_lineage,
    target_position,
    validation_result,
    validation_stage,
    worker,
)
from src.domain._codec import to_primitive
from src.observability.decision_trace import SqlDecisionTraceStore
from src.services.health import DatabaseHeartbeatStore


class DatabasePlatformReport:
    def __init__(self, engine: Engine):
        self.engine = engine

    def build(self) -> dict[str, Any]:
        return {
            "schema": "platform.operator_report/v1",
            "trading": self._trading(),
            "research": self._research(),
            "products": self._products(),
            "operations": self._operations(),
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

    def _research(self) -> dict[str, Any]:
        experiments = self._rows(experiment, order_by=experiment.c.submitted_at.desc())
        results = self._rows(validation_result)
        return {
            "candidate_queue": [
                item for item in experiments if item.get("state", "queued") == "queued"
            ],
            "experiments": experiments,
            "validation_results": results,
            "validation_stages": self._rows(validation_stage),
            "holdout_claims": self._rows(holdout_claim),
            "holdout_outcomes": self._rows(holdout_outcome),
            "forward_evidence": self._rows(forward_evidence),
            "forward_paper_observations": self._rows(forward_paper_observation),
            "strategy_identities": self._rows(strategy_identity),
            "strategy_lineage": self._rows(strategy_lineage),
            "strategy_artefacts": self._rows(strategy_artefact),
            "strategy_approvals": self._rows(strategy_approval),
            "production_preflights": self._rows(production_preflight),
            "promotion_events": self._rows(promotion_event),
            "rejection_reasons": [
                item.get("reason_code") for item in results if item.get("reason_code")
            ],
            "agent_activity": self._payloads(
                agent_action, order_by=agent_action.c.created_at.desc()
            ),
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
                product[metric] += effect if metric in {"funding", "realised_pnl"} else -effect
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

    def _operations(self) -> dict[str, Any]:
        traces = SqlDecisionTraceStore(self.engine).read()
        blocked: dict[str, int] = {}
        for _identity, trace in traces:
            if trace.first_blocked_stage:
                detail = trace.stages[trace.first_blocked_stage]
                key = f"{trace.first_blocked_stage}:{detail.get('reason_code', 'unknown')}"
                blocked[key] = blocked.get(key, 0) + 1
        jobs = self._rows(job, order_by=job.c.available_at)
        return {
            "job_queue": jobs,
            "workers": self._rows(worker, order_by=worker.c.last_heartbeat.desc()),
            "heartbeats": [
                heartbeat.__dict__ for heartbeat in DatabaseHeartbeatStore(self.engine).latest()
            ],
            "alerts": self._payloads(alert, order_by=alert.c.created_at.desc()),
            "decision_funnel_blocked": blocked,
        }

    def _count(self, table) -> int:
        with self.engine.connect() as connection:
            return int(connection.execute(select(func.count()).select_from(table)).scalar_one())
