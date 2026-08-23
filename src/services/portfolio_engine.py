"""Leased product coordination and target-position optimisation services."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from sqlalchemy import select

from src.data.database import balance_snapshot
from src.domain._codec import canonical_hash, timestamp, to_primitive
from src.domain.forecasts import ForecastDirection
from src.domain.portfolios import TargetPosition
from src.execution.position_manager import PositionManager
from src.observability.decision_trace import (
    DecisionTrace,
    DecisionTraceStage,
    SqlDecisionTraceStore,
)
from src.products.active_income import ActiveIncomePortfolio
from src.products.btc_accumulation import BtcAllocationPolicy, target_btc_allocation
from src.risk.engine import SqlRiskDecisionStore, SqlRiskSnapshotStore
from src.services.job_schemas import build_content_hash, validate_job_payload
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.scheduler import DatabaseJobQueue


@dataclass(frozen=True)
class TargetRiskReferences:
    """Canonical IDs produced after a preliminary target has been persisted."""

    target_position_snapshot_id: str
    account_snapshot_id: str
    positions_snapshot_id: str
    balances_snapshot_id: str
    market_data_snapshot_id: str
    risk_policy_ids: tuple[str, ...]


class DatabasePortfolioTargetBuilder:
    """Build one preliminary target from immutable forecast and account inputs."""

    def __init__(
        self,
        *,
        repository: SqlPortfolioRepository,
        snapshot_store: SqlRiskSnapshotStore,
        positions: PositionManager,
        product_configuration: Mapping[str, Mapping[str, Any]],
        account_configuration: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.repository = repository
        self.snapshot_store = snapshot_store
        self.positions = positions
        self.product_configuration = {
            str(key): dict(value) for key, value in product_configuration.items()
        }
        self.account_configuration = {
            str(key): dict(value) for key, value in account_configuration.items()
        }

    def __call__(self, payload: Mapping[str, Any]) -> TargetRiskReferences:
        product_id = str(payload["product_id"])
        product = self.product_configuration[product_id]
        forecast = self.repository.forecast(str(payload["forecast_id"]))
        if forecast.product_id != product_id:
            raise ValueError("forecast product differs from target product")
        if "market_data_snapshot_id" not in payload:
            raise ValueError("target build requires a canonical market-data snapshot")
        market_input = self.snapshot_store.get(str(payload["market_data_snapshot_id"]))
        raw_values = market_input.get("values")
        if not isinstance(raw_values, Mapping):
            raise ValueError("market-data snapshot has no canonical values")
        price = float(raw_values.get("close", 0.0))
        if price <= 0:
            raise ValueError("market-data snapshot close must be positive")

        account_id = str(product["account_id"])
        account = self._latest_account_snapshot(account_id)
        raw_balances = account.get("balances")
        if not isinstance(raw_balances, Mapping):
            raise ValueError("account snapshot has no balances")
        balances = {str(key): float(value) for key, value in raw_balances.items()}
        if product_id == "btc_accumulation":
            equity = balances.get("BTC", 0.0) + balances.get("USDT", 0.0) / price
        else:
            equity = balances.get("USDT", 0.0)
        if equity <= 0:
            raise ValueError("account snapshot has no positive product equity")

        self.positions.reload()
        portfolio_id = str(product["portfolio_id"])
        current_positions = self.positions.current_quantities(portfolio_id)
        sign = {
            ForecastDirection.LONG: 1.0,
            ForecastDirection.SHORT: -1.0,
            ForecastDirection.FLAT: 0.0,
        }[forecast.direction]
        target_fraction = sign * forecast.maximum_position
        target_quantity = target_fraction * equity / price
        target = TargetPosition(
            portfolio_id=portfolio_id,
            instrument_id=forecast.instrument_id,
            target_quantity=target_quantity,
            target_notional=target_fraction * equity,
            target_fraction=target_fraction,
            strategy_contributions={forecast.strategy_version_id: forecast.signed_strength},
            risk_budget=forecast.maximum_position,
            valid_until=forecast.valid_until,
            metadata={
                "event_id": str(payload["event_id"]),
                "forecast_id": str(payload["forecast_id"]),
                "market_data_snapshot_id": str(payload["market_data_snapshot_id"]),
                "account_id": account_id,
            },
        )
        target_payload = to_primitive(target)
        target_ids = self.repository.save_targets(
            event_id=str(payload["event_id"]),
            targets=(target,),
            created_at=str(payload["evaluated_at"]),
        )
        target_fraction_abs = abs(target_fraction)
        product_max_net = 1.0 if product_id == "btc_accumulation" else 0.5
        target_scopes = {
            "strategy": {
                "inputs": {
                    "position_fraction": target_fraction,
                    "turnover_fraction": abs(
                        target_quantity - current_positions.get(forecast.instrument_id, 0.0)
                    )
                    * price
                    / equity,
                    "trades_today": 0,
                    "expected_slippage_bps": 0.0,
                    "expected_funding_cost_fraction": 0.0,
                },
                "limits": {
                    "max_position_fraction": 0.25,
                    "max_turnover_fraction": 1.0,
                    "max_trades_per_day": 100,
                    "max_slippage_bps": 25.0,
                    "max_funding_cost_fraction": 0.05,
                },
                "decision_id": canonical_hash({"scope": "strategy", "target_ids": target_ids}),
            },
            "instrument": {
                "inputs": {
                    "position_notional": target.target_notional,
                    "order_notional": abs(target.target_notional),
                    "visible_depth_fraction": 0.0,
                    "spread_bps": 0.0,
                    "volatility": forecast.target_volatility,
                    "concentration_fraction": target_fraction,
                },
                "limits": {
                    "max_position_notional": max(abs(target.target_notional) * 2.0, 1.0),
                    "max_order_notional": max(abs(target.target_notional) * 2.0, 1.0),
                    "max_visible_depth_fraction": 1.0,
                    "max_spread_bps": 25.0,
                    "max_volatility": 1.0,
                    "max_concentration_fraction": 0.2,
                },
                "decision_id": canonical_hash({"scope": "instrument", "target_ids": target_ids}),
            },
            "sleeve": {
                "inputs": {
                    "capital_fraction": target_fraction_abs,
                    "drawdown_fraction": 0.0,
                    "maximum_correlation": 0.0,
                    "beta": 0.0,
                    "turnover_fraction": target_fraction_abs,
                },
                "limits": {
                    "max_capital_fraction": 0.5,
                    "max_drawdown_fraction": 0.2,
                    "max_correlation": 0.8,
                    "max_abs_beta": 1.0,
                    "max_turnover_fraction": 1.0,
                },
                "decision_id": canonical_hash({"scope": "sleeve", "target_ids": target_ids}),
            },
            "product": {
                "inputs": {
                    "gross_fraction": target_fraction_abs,
                    "net_fraction": target_fraction,
                    "drawdown_fraction": 0.0,
                    "margin_fraction": 0.0,
                    "daily_pnl_fraction": 0.0,
                },
                "limits": {
                    "max_gross_fraction": float(product.get("maximum_gross", 1.5)),
                    "max_net_fraction": float(product.get("maximum_net", product_max_net)),
                    "max_drawdown_fraction": 0.2,
                    "max_margin_fraction": float(product.get("maximum_margin", 0.5)),
                    "max_daily_loss_fraction": 0.02,
                },
                "decision_id": canonical_hash({"scope": "product", "target_ids": target_ids}),
            },
        }
        target_snapshot_id = self.snapshot_store.save(
            {
                "kind": "target_position",
                "product_id": product_id,
                "event_id": str(payload["event_id"]),
                "target_position_ids": list(target_ids),
                "targets": [target_payload],
                "prices": {forecast.instrument_id: price},
                "reconciled_positions": {
                    forecast.instrument_id: current_positions.get(forecast.instrument_id, 0.0)
                },
                "scopes": target_scopes,
            },
            created_at=str(payload["evaluated_at"]),
        )
        account_snapshot_id = self.snapshot_store.save(
            {
                "scope": "account",
                "product_id": product_id,
                "inputs": {
                    "used_margin_fraction": 0.0,
                    "liquidation_buffer_fraction": 1.0,
                    "unknown_positions": {},
                },
                "limits": {
                    "max_used_margin_fraction": float(product.get("maximum_margin", 0.5)),
                    "min_liquidation_buffer_fraction": 0.0,
                    "reject_unknown_exposure": True,
                },
                "decision_id": canonical_hash({"scope": "account", "target_ids": target_ids}),
                "account_id": account_id,
            },
            created_at=str(payload["evaluated_at"]),
        )
        positions_snapshot_id = self.snapshot_store.save(
            {
                "kind": "positions",
                "product_id": product_id,
                "portfolio_id": portfolio_id,
                "positions": current_positions,
            },
            created_at=str(payload["evaluated_at"]),
        )
        balances_snapshot_id = self.snapshot_store.save(
            {
                "kind": "balances",
                "product_id": product_id,
                "account_id": account_id,
                "balances": balances,
            },
            created_at=str(payload["evaluated_at"]),
        )
        market_data_snapshot_id = self.snapshot_store.save(
            {
                "scope": "global",
                "product_id": product_id,
                "inputs": {
                    "drawdown_fraction": 0.0,
                    "exchange_connected": True,
                    "data_age_seconds": 0.0,
                    "clock_skew_seconds": 0.0,
                    "database_healthy": True,
                    "execution_drift": False,
                    "model_drift": False,
                },
                "limits": {
                    "max_drawdown_fraction": 0.2,
                    "max_data_age_seconds": 5.0,
                    "max_clock_skew_seconds": 1.0,
                },
                "decision_id": canonical_hash({"scope": "global", "target_ids": target_ids}),
                "source_snapshot_id": str(payload["market_data_snapshot_id"]),
                "values": dict(raw_values),
            },
            created_at=str(payload["evaluated_at"]),
        )
        return TargetRiskReferences(
            target_position_snapshot_id=target_snapshot_id,
            account_snapshot_id=account_snapshot_id,
            positions_snapshot_id=positions_snapshot_id,
            balances_snapshot_id=balances_snapshot_id,
            market_data_snapshot_id=market_data_snapshot_id,
            risk_policy_ids=(str(product["risk_policy_id"]), account_id),
        )

    def _latest_account_snapshot(self, account_id: str) -> Mapping[str, Any]:
        with self.repository.engine.connect() as connection:
            rows = connection.execute(
                select(balance_snapshot.c.payload, balance_snapshot.c.created_at).order_by(
                    balance_snapshot.c.created_at.desc(), balance_snapshot.c.id.desc()
                )
            ).mappings()
            for row in rows:
                payload = row["payload"]
                if isinstance(payload, Mapping) and str(payload.get("account_id")) == account_id:
                    return payload
        raise ValueError(f"no canonical balance snapshot exists for account {account_id}")


class DatabasePortfolioTargetWorker:
    """Turn an immutable forecast into a target, then submit snapshot IDs to risk."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        build_target: Callable[[Mapping[str, Any]], TargetRiskReferences],
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.build_target = build_target
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("portfolio_target_build",),
        )
        if claimed is None:
            return {"reason_code": "portfolio_target_queue_empty"}
        try:
            payload = validate_job_payload("portfolio_target_build", claimed.payload)
            references = self.build_target(payload)
            risk_payload = {
                "assessment_id": canonical_hash(
                    {
                        "event_id": payload["event_id"],
                        "product_id": payload["product_id"],
                        "forecast_id": payload["forecast_id"],
                        "target_position_snapshot_id": references.target_position_snapshot_id,
                    }
                ),
                "product_id": payload["product_id"],
                "event_id": payload["event_id"],
                "target_position_snapshot_id": references.target_position_snapshot_id,
                "account_snapshot_id": references.account_snapshot_id,
                "positions_snapshot_id": references.positions_snapshot_id,
                "balances_snapshot_id": references.balances_snapshot_id,
                "market_data_snapshot_id": references.market_data_snapshot_id,
                "risk_policy_ids": list(references.risk_policy_ids),
                "evaluated_at": payload["evaluated_at"],
                "producer_identity": self.worker_id,
            }
            risk_payload["content_hash"] = build_content_hash(risk_payload)
            risk_job_id = f"risk:{risk_payload['assessment_id'].removeprefix('sha256:')}"
            self.queue.enqueue_if_absent(
                job_id=risk_job_id,
                name="risk_assessment",
                payload=risk_payload,
                available_at=str(payload["evaluated_at"]),
                priority=18,
                producer_identity=self.worker_id,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "portfolio_target_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "risk_assessment_enqueued",
            "job_id": claimed.job_id,
            "risk_job_id": risk_job_id,
        }


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
