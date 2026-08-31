"""Portfolio-target execution service shared by paper and live adapters."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from decimal import Decimal
from typing import Protocol

from src.accounting.ledger import Ledger
from src.domain._codec import timestamp
from src.domain.orders import Fill, OrderIntent
from src.domain.portfolios import TargetPosition
from src.domain.positions import Position
from src.domain.risk import RiskDecision
from src.execution.order_manager import OrderManager
from src.execution.order_planner import plan_orders
from src.execution.position_manager import PositionManager
from src.observability.decision_trace import (
    DecisionTrace,
    DecisionTraceStage,
)


class ExecutionVenue(Protocol):
    order_manager: OrderManager

    def submit(self, intent: OrderIntent) -> Fill: ...


class DecisionTraceStore(Protocol):
    def append(self, trace: DecisionTrace) -> str: ...


class ExecutionService:
    """Translate accepted targets into durable paper orders.

    It deliberately requires a previously persisted :class:`RiskDecision`.
    Research code can create forecasts and targets but cannot submit orders.
    """

    def __init__(
        self,
        *,
        paper_exchange: ExecutionVenue,
        positions: PositionManager,
        ledger: Ledger | None = None,
        trace_store: DecisionTraceStore | None = None,
    ) -> None:
        self.paper_exchange = paper_exchange
        self.positions = positions
        self.ledger = ledger
        self.trace_store = trace_store
        if not positions.all() and paper_exchange.order_manager.all():
            positions.recover_from_orders(paper_exchange.order_manager)

    def execute_targets(
        self,
        *,
        portfolio_id: str,
        targets: Iterable[TargetPosition],
        risk_decision: RiskDecision,
        event_id: str | None = None,
        decided_at: str | None = None,
    ) -> tuple[tuple[OrderIntent, ...], tuple[Fill, ...], tuple[DecisionTrace, ...]]:
        if risk_decision.scope not in {"account", "global", "portfolio"}:
            raise ValueError("execution requires a portfolio, account, or global risk decision")
        materialised = tuple(targets)
        if decided_at is None:
            now_value = dt.datetime.now(dt.UTC).replace(microsecond=0)
            # Paper replay commonly consumes historical targets.  Use their
            # own decision window when the host clock is later, while live
            # callers always provide the actual submission timestamp.
            if materialised:
                latest_window = min(
                    dt.datetime.fromisoformat(target.valid_until) for target in materialised
                )
                if latest_window <= now_value:
                    now_value = latest_window - dt.timedelta(seconds=1)
            decided_at = now_value.isoformat()
        now = timestamp(decided_at, field="decided_at")
        traces: list[DecisionTrace] = []
        if not risk_decision.accepted:
            for target in materialised:
                traces.append(
                    DecisionTrace.start(
                        event_id=f"target:{target.instrument_id}:{now}",
                        instrument_id=target.instrument_id,
                    )
                    .pass_stage(DecisionTraceStage.DATA_AVAILABLE)
                    .pass_stage(DecisionTraceStage.FEATURE_AVAILABLE)
                    .pass_stage(DecisionTraceStage.STRATEGY_EVALUATED)
                    .pass_stage(DecisionTraceStage.REGIME_PASSED)
                    .pass_stage(DecisionTraceStage.SETUP_PASSED)
                    .pass_stage(DecisionTraceStage.TRIGGER_PASSED)
                    .pass_stage(DecisionTraceStage.SIGNAL_PRODUCED)
                    .pass_stage(DecisionTraceStage.PORTFOLIO_ACCEPTED)
                    .block(
                        DecisionTraceStage.RISK_ACCEPTED,
                        reason_code=risk_decision.reason_code or "risk_rejected",
                    )
                )
            return self._result((), (), traces)
        instrument_ids = [target.instrument_id for target in materialised]
        if len(instrument_ids) != len(set(instrument_ids)):
            raise ValueError("targets must contain at most one position per instrument")
        fills: list[Fill] = []
        all_orders: list[OrderIntent] = []
        for target in materialised:
            orders = plan_orders(
                (target,),
                current_quantities=self.positions.current_quantities(portfolio_id),
                decided_at=now,
            )
            if not orders:
                traces.append(
                    self._accepted_trace(
                        target.instrument_id, now, risk_decision, event_id=event_id
                    ).block(
                        DecisionTraceStage.ORDER_PLANNED,
                        reason_code="target_already_satisfied",
                    )
                )
                continue
            for order in orders:
                all_orders.append(order)
                previous_position = self.positions.get(portfolio_id, order.instrument_id)
                fill = self.paper_exchange.submit(order)
                fills.append(fill)
                self.record_execution_costs(order, fill, previous_position=previous_position)
                updated_order = self.paper_exchange.order_manager.get(order.order_id)
                position = self.positions.get(portfolio_id, order.instrument_id)
                trace = (
                    self._accepted_trace(
                        order.instrument_id,
                        now,
                        risk_decision,
                        event_id=f"{event_id}:{order.order_id}" if event_id else None,
                    )
                    .pass_stage(DecisionTraceStage.ORDER_PLANNED)
                    .pass_stage(DecisionTraceStage.ORDER_SUBMITTED)
                )
                if updated_order.remaining_quantity > 1e-12:
                    traces.append(
                        trace.block(
                            DecisionTraceStage.ORDER_FILLED,
                            reason_code="partial_fill_pending",
                            fill_id=fill.fill_id,
                            filled_quantity=updated_order.filled_quantity,
                            remaining_quantity=updated_order.remaining_quantity,
                        )
                    )
                    continue
                trace = trace.pass_stage(DecisionTraceStage.ORDER_FILLED, fill_id=fill.fill_id)
                stage = (
                    DecisionTraceStage.POSITION_CLOSED
                    if position.quantity == 0
                    else DecisionTraceStage.POSITION_OPENED
                )
                if stage is DecisionTraceStage.POSITION_CLOSED:
                    trace = trace.pass_stage(DecisionTraceStage.POSITION_OPENED)
                traces.append(trace.pass_stage(stage, quantity=position.quantity))
        return self._result(all_orders, fills, traces)

    def _result(
        self,
        orders: Iterable[OrderIntent],
        fills: Iterable[Fill],
        traces: Iterable[DecisionTrace],
    ) -> tuple[tuple[OrderIntent, ...], tuple[Fill, ...], tuple[DecisionTrace, ...]]:
        materialised_traces = tuple(traces)
        if self.trace_store is not None:
            for trace in materialised_traces:
                self.trace_store.append(trace)
        return tuple(orders), tuple(fills), materialised_traces

    def record_execution_costs(
        self,
        order: OrderIntent,
        fill: Fill,
        *,
        previous_position: Position,
    ) -> None:
        if self.ledger is None:
            return
        if fill.fee_asset not in {None, self.ledger.accounting_asset}:
            raise ValueError(
                f"fee conversion to {self.ledger.accounting_asset} is required for {fill.fee_asset}"
            )
        target_metadata = order.metadata.get("target_metadata")
        target_metadata = target_metadata if isinstance(target_metadata, dict) else {}
        contributions = sorted(order.strategy_contributions)
        attribution = {
            "product": self.ledger.product_id,
            "symbol": order.instrument_id,
            "strategy": contributions[0] if len(contributions) == 1 else "ensemble",
            "sleeve": target_metadata.get("sleeve") or "unassigned",
        }
        self.ledger.record_fee(
            entry_id=f"{fill.fill_id}:fee",
            amount=Decimal(str(fill.fee)),
            occurred_at=fill.occurred_at,
            attribution=attribution,
        )
        slippage = Decimal(str(fill.metadata.get("slippage_cost") or 0))
        if slippage:
            self.ledger.record_slippage(
                entry_id=f"{fill.fill_id}:slippage",
                amount=slippage,
                occurred_at=fill.occurred_at,
                attribution=attribution,
            )
        fill_sign = 1.0 if fill.side.value == "buy" else -1.0
        opposing = previous_position.quantity != 0 and (previous_position.quantity > 0) != (
            fill_sign > 0
        )
        if opposing:
            closing_quantity = min(abs(previous_position.quantity), fill.quantity)
            direction = Decimal("1") if previous_position.quantity > 0 else Decimal("-1")
            reference_price = float(fill.metadata.get("reference_price") or fill.price)
            reference_entry_price = float(
                previous_position.metadata.get("reference_entry_price")
                or previous_position.average_entry_price
            )
            quote_pnl = (
                Decimal(str(reference_price - reference_entry_price))
                * Decimal(str(closing_quantity))
                * direction
            )
            realised_pnl = (
                quote_pnl / Decimal(str(fill.price))
                if self.ledger.accounting_asset == "BTC"
                else quote_pnl
            )
            if realised_pnl:
                self.ledger.record_realised_pnl(
                    entry_id=f"{fill.fill_id}:realised_pnl",
                    amount=realised_pnl,
                    occurred_at=fill.occurred_at,
                    attribution=attribution,
                )

    @staticmethod
    def _accepted_trace(
        instrument_id: str,
        now: str,
        risk_decision: RiskDecision,
        *,
        event_id: str | None = None,
    ) -> DecisionTrace:
        return (
            DecisionTrace.start(
                event_id=event_id or f"target:{instrument_id}:{now}", instrument_id=instrument_id
            )
            .pass_stage(DecisionTraceStage.DATA_AVAILABLE)
            .pass_stage(DecisionTraceStage.FEATURE_AVAILABLE)
            .pass_stage(DecisionTraceStage.STRATEGY_EVALUATED)
            .pass_stage(DecisionTraceStage.REGIME_PASSED)
            .pass_stage(DecisionTraceStage.SETUP_PASSED)
            .pass_stage(DecisionTraceStage.TRIGGER_PASSED)
            .pass_stage(DecisionTraceStage.SIGNAL_PRODUCED)
            .pass_stage(DecisionTraceStage.PORTFOLIO_ACCEPTED)
            .pass_stage(DecisionTraceStage.RISK_ACCEPTED, decision_id=risk_decision.decision_id)
        )
