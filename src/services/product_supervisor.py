"""End-to-end active-income paper product supervision."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from src.domain.forecasts import AlphaForecast
from src.domain.orders import Fill, OrderIntent
from src.domain.portfolios import TargetPosition
from src.observability.decision_trace import DecisionTrace, DecisionTraceStage
from src.products.active_income import ActiveIncomePortfolio
from src.products.btc_accumulation import (
    BtcAllocationPolicy,
    BtcAllocationTarget,
    assert_btc_spot_instrument,
    btc_step_aside_metadata,
    target_btc_allocation,
)
from src.risk.engine import REQUIRED_RISK_SCOPES, HierarchicalRiskAssessment
from src.services.execution_service import ExecutionService


@dataclass(frozen=True)
class ProductCycleResult:
    event_id: str
    targets: tuple[TargetPosition, ...]
    orders: tuple[OrderIntent, ...]
    fills: tuple[Fill, ...]
    traces: tuple[DecisionTrace, ...]
    accepted: bool
    first_blocked_stage: str | None


class ActiveIncomeProductSupervisor:
    """Convert forecasts to risk-approved targets and durable paper fills."""

    def __init__(
        self, *, portfolio: ActiveIncomePortfolio, execution_service: ExecutionService
    ) -> None:
        self.portfolio = portfolio
        self.execution_service = execution_service

    def process_forecasts(
        self,
        *,
        event_id: str,
        event_instrument_id: str,
        forecasts: Iterable[AlphaForecast],
        prices: Mapping[str, float],
        valid_until: str,
        risk_assessment: HierarchicalRiskAssessment,
        correlations: Mapping[str, Mapping[str, float]] | None = None,
        beta_by_instrument: Mapping[str, float] | None = None,
        observed_volatility: Mapping[str, float] | None = None,
        liquidity_fraction_caps: Mapping[str, float] | None = None,
        funding_rates: Mapping[str, float] | None = None,
        sleeve_budgets: Mapping[str, float] | None = None,
        cluster_by_instrument: Mapping[str, str] | None = None,
        cluster_fraction_caps: Mapping[str, float] | None = None,
        product_drawdown_fraction: float = 0.0,
        available_margin_fraction: float = 1.0,
        equity: float | None = None,
    ) -> ProductCycleResult:
        if tuple(item.scope for item in risk_assessment.decisions) != REQUIRED_RISK_SCOPES:
            raise ValueError("product execution requires all six risk scopes")
        if risk_assessment.aggregate.input_snapshot.get("product_id") != "active_income":
            raise ValueError("risk assessment belongs to another product")
        materialised = tuple(forecasts)
        if any(item.product_id != "active_income" for item in materialised):
            raise ValueError("active-income supervisor received a forecast for another product")
        portfolio = self.portfolio
        if equity is not None:
            if equity <= 0:
                raise ValueError("active-income equity must be positive")
            portfolio = ActiveIncomePortfolio(replace(self.portfolio.constraints, equity=equity))
        portfolio_id = portfolio.constraints.portfolio_id
        current = self.execution_service.positions.current_quantities(portfolio_id)
        targets = portfolio.target_positions(
            materialised,
            prices=prices,
            valid_until=valid_until,
            correlations=correlations,
            beta_by_instrument=beta_by_instrument,
            observed_volatility=observed_volatility,
            liquidity_fraction_caps=liquidity_fraction_caps,
            funding_rates=funding_rates,
            current_quantities=current,
            sleeve_budgets=sleeve_budgets,
            cluster_by_instrument=cluster_by_instrument,
            cluster_fraction_caps=cluster_fraction_caps,
            product_drawdown_fraction=product_drawdown_fraction,
            available_margin_fraction=available_margin_fraction,
        )
        if not targets:
            trace = self._no_target_trace(
                event_id=event_id,
                instrument_id=event_instrument_id,
                forecasts_present=bool(materialised),
                portfolio_reason=(
                    "product_drawdown_limit"
                    if product_drawdown_fraction > portfolio.constraints.max_drawdown_fraction
                    else "portfolio_no_target"
                ),
            )
            if self.execution_service.trace_store is not None:
                self.execution_service.trace_store.append(trace)
            return ProductCycleResult(
                event_id=event_id,
                targets=(),
                orders=(),
                fills=(),
                traces=(trace,),
                accepted=False,
                first_blocked_stage=trace.first_blocked_stage,
            )
        orders, fills, traces = self.execution_service.execute_targets(
            portfolio_id=portfolio_id,
            targets=targets,
            risk_decision=risk_assessment.aggregate,
            event_id=event_id,
        )
        first_blocked = next(
            (trace.first_blocked_stage for trace in traces if trace.first_blocked_stage), None
        )
        return ProductCycleResult(
            event_id=event_id,
            targets=targets,
            orders=orders,
            fills=fills,
            traces=traces,
            accepted=first_blocked is None,
            first_blocked_stage=first_blocked,
        )

    @staticmethod
    def _no_target_trace(
        *,
        event_id: str,
        instrument_id: str,
        forecasts_present: bool,
        portfolio_reason: str,
    ) -> DecisionTrace:
        trace = (
            DecisionTrace.start(event_id=event_id, instrument_id=instrument_id)
            .pass_stage(DecisionTraceStage.DATA_AVAILABLE)
            .pass_stage(DecisionTraceStage.FEATURE_AVAILABLE)
            .pass_stage(DecisionTraceStage.STRATEGY_EVALUATED)
            .pass_stage(DecisionTraceStage.REGIME_PASSED)
            .pass_stage(DecisionTraceStage.SETUP_PASSED)
            .pass_stage(DecisionTraceStage.TRIGGER_PASSED)
        )
        if not forecasts_present:
            return trace.block(
                DecisionTraceStage.SIGNAL_PRODUCED,
                reason_code="no_actionable_forecast",
            )
        return trace.pass_stage(DecisionTraceStage.SIGNAL_PRODUCED).block(
            DecisionTraceStage.PORTFOLIO_ACCEPTED,
            reason_code=portfolio_reason,
        )


@dataclass(frozen=True)
class BtcProductCycleResult:
    event_id: str
    allocation: BtcAllocationTarget
    target: TargetPosition
    orders: tuple[OrderIntent, ...]
    fills: tuple[Fill, ...]
    traces: tuple[DecisionTrace, ...]
    btc_nav_before_costs: float


class BtcAccumulationProductSupervisor:
    """Convert BTC allocation forecasts into spot target-position orders."""

    def __init__(
        self,
        *,
        execution_service: ExecutionService,
        policy: BtcAllocationPolicy = BtcAllocationPolicy(),
        portfolio_id: str = "btc_accumulation",
    ) -> None:
        self.execution_service = execution_service
        self.policy = policy
        if not portfolio_id.strip():
            raise ValueError("portfolio_id cannot be empty")
        self.portfolio_id = portfolio_id.strip()

    def process_forecasts(
        self,
        *,
        event_id: str,
        instrument_id: str,
        forecasts: Iterable[AlphaForecast],
        btc_balance: float,
        stablecoin_balance: float,
        stablecoin_per_btc: float,
        valid_until: str,
        risk_assessment: HierarchicalRiskAssessment,
    ) -> BtcProductCycleResult:
        instrument_id = assert_btc_spot_instrument(instrument_id)
        if tuple(item.scope for item in risk_assessment.decisions) != REQUIRED_RISK_SCOPES:
            raise ValueError("product execution requires all six risk scopes")
        if risk_assessment.aggregate.input_snapshot.get("product_id") != "btc_accumulation":
            raise ValueError("risk assessment belongs to another product")
        if btc_balance < 0 or stablecoin_balance < 0 or stablecoin_per_btc <= 0:
            raise ValueError("BTC and stablecoin balances must be non-negative and price positive")
        materialised = tuple(forecasts)
        if any(item.product_id != "btc_accumulation" for item in materialised):
            raise ValueError("BTC supervisor received a forecast for another product")
        btc_nav = btc_balance + stablecoin_balance / stablecoin_per_btc
        allocation = target_btc_allocation(materialised, policy=self.policy)
        target_quantity = btc_nav * allocation.target_btc_fraction
        cycle_metadata = btc_step_aside_metadata(
            instrument_id=instrument_id,
            current_btc=btc_balance,
            target_btc=target_quantity,
            price=stablecoin_per_btc,
            stablecoin_balance=stablecoin_balance,
            state_id=event_id,
        )
        self.execution_service.positions.reconcile_position(
            portfolio_id=self.portfolio_id,
            instrument_id=instrument_id,
            quantity=btc_balance,
            average_entry_price=stablecoin_per_btc,
            updated_at=valid_until,
        )
        target = TargetPosition(
            portfolio_id=self.portfolio_id,
            instrument_id=instrument_id,
            target_quantity=target_quantity,
            target_notional=target_quantity * stablecoin_per_btc,
            target_fraction=allocation.target_btc_fraction,
            strategy_contributions=allocation.contributions
            or {"btc_allocation:core": allocation.core_btc_fraction},
            risk_budget=self.policy.max_tactical_fraction,
            valid_until=valid_until,
            metadata={
                "sleeve": "btc_tactical",
                "btc_nav_before_costs": btc_nav,
                "stablecoin_balance": stablecoin_balance,
                "stablecoin_per_btc": stablecoin_per_btc,
                **cycle_metadata,
            },
        )
        orders, fills, traces = self.execution_service.execute_targets(
            portfolio_id=self.portfolio_id,
            targets=(target,),
            risk_decision=risk_assessment.aggregate,
            event_id=event_id,
        )
        return BtcProductCycleResult(
            event_id=event_id,
            allocation=allocation,
            target=target,
            orders=orders,
            fills=fills,
            traces=traces,
            btc_nav_before_costs=btc_nav,
        )
