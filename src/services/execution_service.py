"""Portfolio-target execution service shared by paper and live adapters."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from decimal import Decimal
from typing import Protocol

from src.accounting.fees import FeeConversionError, convert_fee
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
        now = self._decision_time(materialised, decided_at)
        traces: list[DecisionTrace] = []
        if not risk_decision.accepted:
            traces.extend(self._risk_rejection_traces(materialised, now, risk_decision))
            return self._result((), (), traces)
        self._assert_unique_instruments(materialised)
        fills: list[Fill] = []
        all_orders: list[OrderIntent] = []
        for target in materialised:
            orders = self._target_orders(target, portfolio_id=portfolio_id, decided_at=now)
            if not orders:
                traces.append(self._already_satisfied_trace(target, now, risk_decision, event_id))
                continue
            for order in orders:
                fill, trace = self._execute_order(order, now, risk_decision, event_id)
                all_orders.append(order)
                fills.append(fill)
                traces.append(trace)
        return self._result(all_orders, fills, traces)

    @staticmethod
    def _decision_time(targets: tuple[TargetPosition, ...], decided_at: str | None) -> str:
        if decided_at is not None:
            return timestamp(decided_at, field="decided_at")
        now_value = dt.datetime.now(dt.UTC).replace(microsecond=0)
        if targets:
            latest_window = min(dt.datetime.fromisoformat(target.valid_until) for target in targets)
            if latest_window <= now_value:
                now_value = latest_window - dt.timedelta(seconds=1)
        return timestamp(now_value.isoformat(), field="decided_at")

    @staticmethod
    def _assert_unique_instruments(targets: tuple[TargetPosition, ...]) -> None:
        instrument_ids = [target.instrument_id for target in targets]
        if len(instrument_ids) != len(set(instrument_ids)):
            raise ValueError("targets must contain at most one position per instrument")

    def _risk_rejection_traces(
        self,
        targets: tuple[TargetPosition, ...],
        now: str,
        risk_decision: RiskDecision,
    ) -> tuple[DecisionTrace, ...]:
        return tuple(
            self._portfolio_trace(target.instrument_id, now).block(
                DecisionTraceStage.RISK_ACCEPTED,
                reason_code=risk_decision.reason_code or "risk_rejected",
            )
            for target in targets
        )

    @staticmethod
    def _portfolio_trace(instrument_id: str, now: str) -> DecisionTrace:
        return (
            DecisionTrace.start(
                event_id=f"target:{instrument_id}:{now}", instrument_id=instrument_id
            )
            .pass_stage(DecisionTraceStage.DATA_AVAILABLE)
            .pass_stage(DecisionTraceStage.FEATURE_AVAILABLE)
            .pass_stage(DecisionTraceStage.STRATEGY_EVALUATED)
            .pass_stage(DecisionTraceStage.REGIME_PASSED)
            .pass_stage(DecisionTraceStage.SETUP_PASSED)
            .pass_stage(DecisionTraceStage.TRIGGER_PASSED)
            .pass_stage(DecisionTraceStage.SIGNAL_PRODUCED)
            .pass_stage(DecisionTraceStage.PORTFOLIO_ACCEPTED)
        )

    def _target_orders(
        self, target: TargetPosition, *, portfolio_id: str, decided_at: str
    ) -> tuple[OrderIntent, ...]:
        return plan_orders(
            (target,),
            current_quantities=self.positions.current_quantities(portfolio_id),
            decided_at=decided_at,
        )

    def _already_satisfied_trace(
        self,
        target: TargetPosition,
        now: str,
        risk_decision: RiskDecision,
        event_id: str | None,
    ) -> DecisionTrace:
        return self._accepted_trace(
            target.instrument_id, now, risk_decision, event_id=event_id
        ).block(
            DecisionTraceStage.ORDER_PLANNED,
            reason_code="target_already_satisfied",
        )

    def _execute_order(
        self,
        order: OrderIntent,
        now: str,
        risk_decision: RiskDecision,
        event_id: str | None,
    ) -> tuple[Fill, DecisionTrace]:
        previous_position = self.positions.get(order.portfolio_id, order.instrument_id)
        fill = self.paper_exchange.submit(order)
        self.record_execution_costs(order, fill, previous_position=previous_position)
        updated_order = self.paper_exchange.order_manager.get(order.order_id)
        position = self.positions.get(order.portfolio_id, order.instrument_id)
        trace = self._order_trace(order, now, risk_decision, event_id)
        if updated_order.remaining_quantity > 1e-12:
            return fill, trace.block(
                DecisionTraceStage.ORDER_FILLED,
                reason_code="partial_fill_pending",
                fill_id=fill.fill_id,
                filled_quantity=updated_order.filled_quantity,
                remaining_quantity=updated_order.remaining_quantity,
            )
        trace = trace.pass_stage(DecisionTraceStage.ORDER_FILLED, fill_id=fill.fill_id)
        if position.quantity == 0:
            trace = trace.pass_stage(DecisionTraceStage.POSITION_OPENED)
            return fill, trace.pass_stage(
                DecisionTraceStage.POSITION_CLOSED, quantity=position.quantity
            )
        return fill, trace.pass_stage(
            DecisionTraceStage.POSITION_OPENED, quantity=position.quantity
        )

    def _order_trace(
        self,
        order: OrderIntent,
        now: str,
        risk_decision: RiskDecision,
        event_id: str | None,
    ) -> DecisionTrace:
        return (
            self._accepted_trace(
                order.instrument_id,
                now,
                risk_decision,
                event_id=f"{event_id}:{order.order_id}" if event_id else None,
            )
            .pass_stage(DecisionTraceStage.ORDER_PLANNED)
            .pass_stage(DecisionTraceStage.ORDER_SUBMITTED)
        )

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
        target_metadata = order.metadata.get("target_metadata")
        target_metadata = target_metadata if isinstance(target_metadata, dict) else {}
        fee_metadata = {
            **dict(target_metadata),
            **dict(order.metadata),
            **dict(fill.metadata),
        }
        try:
            fee_conversion = convert_fee(
                amount=fill.fee,
                fee_asset=fill.fee_asset,
                accounting_asset=self.ledger.accounting_asset,
                trade_price=fill.price,
                base_asset=_instrument_asset(order.instrument_id, "base"),
                quote_asset=_instrument_asset(order.instrument_id, "quote"),
                metadata=fee_metadata,
            )
        except FeeConversionError as exc:
            raise ValueError(str(exc)) from exc
        contributions = sorted(order.strategy_contributions)
        attribution = {
            "product": self.ledger.product_id,
            "symbol": order.instrument_id,
            "instrument_id": order.instrument_id,
            "order_id": order.order_id,
            "strategy": contributions[0] if len(contributions) == 1 else "ensemble",
            "strategy_version_id": contributions[0] if len(contributions) == 1 else None,
            "sleeve": target_metadata.get("sleeve") or "unassigned",
            "assignment_id": target_metadata.get("assignment_id"),
            **fee_conversion.to_payload(),
        }
        self.ledger.record_fee(
            entry_id=f"{fill.fill_id}:fee",
            amount=Decimal(str(fee_conversion.accounting_amount)),
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


def _instrument_asset(instrument_id: str, asset: str) -> str | None:
    symbol = str(instrument_id).rsplit(":", 1)[-1].upper()
    for quote in ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[: -len(quote)] if asset == "base" else quote
    return None
