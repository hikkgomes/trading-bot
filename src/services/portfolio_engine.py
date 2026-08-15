"""Leased product coordination and target-position optimisation services."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.domain._codec import canonical_hash, timestamp, to_primitive
from src.domain.portfolios import TargetPosition
from src.execution.position_manager import PositionManager
from src.observability.decision_trace import (
    DecisionTrace,
    DecisionTraceStage,
    SqlDecisionTraceStore,
)
from src.products.active_income import ActiveIncomePortfolio
from src.products.btc_accumulation import BtcAllocationPolicy, target_btc_allocation
from src.risk.engine import SqlRiskDecisionStore
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.scheduler import DatabaseJobQueue


class DatabaseProductCoordinator:
    """Move product triggers into the portfolio queue without doing execution work."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("active_income_cycle", "btc_accumulation_cycle"),
        )
        if claimed is None:
            return {"reason_code": "product_cycle_queue_empty"}
        try:
            name = (
                "active_income_portfolio"
                if claimed.name == "active_income_cycle"
                else "btc_accumulation_portfolio"
            )
            identity = canonical_hash(
                {"source_job_id": claimed.job_id, "name": name, "payload": claimed.payload}
            )
            job_id = f"portfolio:{identity.removeprefix('sha256:')}"
            self.queue.enqueue_if_absent(
                job_id=job_id,
                name=name,
                payload={**claimed.payload, "source_job_id": claimed.job_id},
                available_at=now,
                priority=10,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "product_coordination_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "portfolio_cycle_enqueued",
            "job_id": claimed.job_id,
            "portfolio_job_id": job_id,
        }


class DatabasePortfolioWorker:
    """Build durable targets, then hand them to the independent execution service."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        repository: SqlPortfolioRepository,
        positions: PositionManager,
        active_income: ActiveIncomePortfolio,
        risk_store: SqlRiskDecisionStore,
        trace_store: SqlDecisionTraceStore,
        execution_modes: Mapping[str, str],
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.repository = repository
        self.positions = positions
        self.active_income = active_income
        self.risk_store = risk_store
        self.trace_store = trace_store
        self.execution_modes = dict(execution_modes)
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("active_income_portfolio", "btc_accumulation_portfolio"),
        )
        if claimed is None:
            return {"reason_code": "portfolio_queue_empty"}
        try:
            self.positions.reload()
            product_id = (
                "active_income" if claimed.name == "active_income_portfolio" else "btc_accumulation"
            )
            payload = claimed.payload
            evaluated_at = timestamp(str(payload["evaluated_at"]), field="evaluated_at")
            assessment = self.risk_store.assessment(str(payload["risk_assessment_id"]))
            if assessment.aggregate.input_snapshot.get("product_id") != product_id:
                raise ValueError("risk assessment belongs to another product")
            targets, prices, reconciled_positions = self._targets(
                product_id=product_id,
                payload=payload,
                evaluated_at=evaluated_at,
            )
            event_id = str(payload["event_id"])
            target_ids = self.repository.save_targets(
                event_id=event_id,
                targets=targets,
                created_at=evaluated_at,
            )
            if not targets:
                reason = (
                    "no_actionable_forecast"
                    if not self.repository.active_forecasts(product_id=product_id, at=evaluated_at)
                    else "portfolio_no_target"
                )
                instrument_id = str(payload.get("instrument_id") or sorted(prices)[0])
                trace = (
                    DecisionTrace.start(event_id=event_id, instrument_id=instrument_id)
                    .pass_stage(DecisionTraceStage.DATA_AVAILABLE)
                    .pass_stage(DecisionTraceStage.FEATURE_AVAILABLE)
                    .pass_stage(DecisionTraceStage.STRATEGY_EVALUATED)
                    .pass_stage(DecisionTraceStage.REGIME_PASSED)
                    .pass_stage(DecisionTraceStage.SETUP_PASSED)
                    .pass_stage(DecisionTraceStage.TRIGGER_PASSED)
                )
                trace = (
                    trace.block(DecisionTraceStage.SIGNAL_PRODUCED, reason_code=reason)
                    if reason == "no_actionable_forecast"
                    else trace.pass_stage(DecisionTraceStage.SIGNAL_PRODUCED).block(
                        DecisionTraceStage.PORTFOLIO_ACCEPTED,
                        reason_code=reason,
                    )
                )
                self.trace_store.append(trace)
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": reason,
                    "job_id": claimed.job_id,
                    "targets": 0,
                    "first_blocked_stage": trace.first_blocked_stage,
                }
            execution_payload = {
                "product_id": product_id,
                "event_id": event_id,
                "evaluated_at": evaluated_at,
                "risk_assessment_id": assessment.aggregate.decision_id,
                "execution_mode": self.execution_modes[product_id],
                "prices": prices,
                "reconciled_positions": reconciled_positions,
                "targets": [to_primitive(item) for item in targets],
            }
            identity = canonical_hash(execution_payload)
            execution_job_id = f"execution:{identity.removeprefix('sha256:')}"
            self.queue.enqueue_if_absent(
                job_id=execution_job_id,
                name="execute_targets",
                payload=execution_payload,
                available_at=evaluated_at,
                priority=20,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "portfolio_cycle_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "execution_cycle_enqueued",
            "job_id": claimed.job_id,
            "execution_job_id": execution_job_id,
            "targets": len(target_ids),
        }

    def _targets(
        self,
        *,
        product_id: str,
        payload: Mapping[str, Any],
        evaluated_at: str,
    ) -> tuple[tuple[TargetPosition, ...], dict[str, float], dict[str, float]]:
        forecasts = self.repository.active_forecasts(product_id=product_id, at=evaluated_at)
        if product_id == "active_income":
            prices = {str(key): float(value) for key, value in payload["prices"].items()}
            equity = float(payload["equity"])
            portfolio = ActiveIncomePortfolio(
                replace(self.active_income.constraints, equity=equity)
            )
            current = self.positions.current_quantities(portfolio.constraints.portfolio_id)
            targets = portfolio.target_positions(
                forecasts,
                prices=prices,
                valid_until=str(payload["valid_until"]),
                correlations=payload.get("correlations"),
                beta_by_instrument=payload.get("beta_by_instrument"),
                observed_volatility=payload.get("observed_volatility"),
                liquidity_fraction_caps=payload.get("liquidity_fraction_caps"),
                funding_rates=payload.get("funding_rates"),
                current_quantities=current,
                sleeve_budgets=payload.get("sleeve_budgets"),
                cluster_by_instrument=payload.get("cluster_by_instrument"),
                cluster_fraction_caps=payload.get("cluster_fraction_caps"),
                product_drawdown_fraction=float(payload.get("product_drawdown_fraction", 0.0)),
                available_margin_fraction=float(payload.get("available_margin_fraction", 1.0)),
            )
            return targets, prices, {}
        instrument_id = str(payload["instrument_id"])
        price = float(payload["stablecoin_per_btc"])
        btc_balance = float(payload["btc_balance"])
        stablecoin_balance = float(payload["stablecoin_balance"])
        if btc_balance < 0 or stablecoin_balance < 0 or price <= 0:
            raise ValueError("BTC balances must be non-negative and price positive")
        allocation = target_btc_allocation(forecasts, policy=BtcAllocationPolicy())
        btc_nav = btc_balance + stablecoin_balance / price
        target_quantity = btc_nav * allocation.target_btc_fraction
        target = TargetPosition(
            portfolio_id="btc-accumulation-portfolio",
            instrument_id=instrument_id,
            target_quantity=target_quantity,
            target_notional=target_quantity * price,
            target_fraction=allocation.target_btc_fraction,
            strategy_contributions=allocation.contributions
            or {"btc_allocation:core": allocation.core_btc_fraction},
            risk_budget=BtcAllocationPolicy().max_tactical_fraction,
            valid_until=str(payload["valid_until"]),
            metadata={
                "sleeve": "btc_tactical",
                "btc_nav_before_costs": btc_nav,
                "stablecoin_balance": stablecoin_balance,
                "stablecoin_per_btc": price,
            },
        )
        return (target,), {instrument_id: price}, {instrument_id: btc_balance}


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
