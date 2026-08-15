"""Independent durable order-planning and paper-execution queue workers."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.accounting.ledger import Ledger
from src.data.database import reconciliation_event
from src.domain._codec import canonical_hash, timestamp
from src.domain.market_events import MarketEvent, MarketEventType
from src.domain.orders import Fill, OrderIntent, OrderSide, OrderStatus
from src.domain.portfolios import TargetPosition
from src.execution.order_groups import (
    OrderGroupManager,
    OrderGroupStatus,
    plan_order_group,
)
from src.execution.order_manager import OrderManager
from src.execution.order_planner import plan_orders
from src.execution.paper_exchange import PaperExchange
from src.execution.position_manager import PositionManager
from src.observability.decision_trace import (
    DecisionTrace,
    DecisionTraceStage,
    SqlDecisionTraceStore,
)
from src.risk.engine import SqlRiskDecisionStore
from src.services.execution_service import ExecutionService
from src.services.scheduler import DatabaseJobQueue


class DatabaseExecutionWorker:
    """Persist target deltas before handing paper orders to the venue worker."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        order_manager: OrderManager,
        positions: PositionManager,
        risk_store: SqlRiskDecisionStore,
        trace_store: SqlDecisionTraceStore,
        product_execution: Mapping[str, Mapping[str, Any]],
        order_groups: OrderGroupManager | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.order_manager = order_manager
        self.positions = positions
        self.risk_store = risk_store
        self.trace_store = trace_store
        self.order_groups = order_groups
        self.product_execution = {
            product_id: dict(configuration)
            for product_id, configuration in product_execution.items()
        }
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("execute_targets",),
        )
        if claimed is None:
            return {"reason_code": "execution_queue_empty"}
        try:
            payload = claimed.payload
            self.order_manager.reload()
            self.positions.reload()
            product_id = str(payload["product_id"])
            configuration = self.product_execution[product_id]
            mode = str(payload["execution_mode"])
            if mode != configuration["execution_mode"]:
                raise ValueError("execution job mode differs from product configuration")
            assessment = self.risk_store.assessment(str(payload["risk_assessment_id"]))
            if assessment.aggregate.input_snapshot.get("product_id") != product_id:
                raise ValueError("execution risk assessment belongs to another product")
            targets = tuple(TargetPosition(**dict(item)) for item in payload["targets"])
            if not assessment.accepted:
                for target in targets:
                    self.trace_store.append(
                        _risk_rejected_trace(
                            event_id=str(payload["event_id"]),
                            target=target,
                            reason_code=assessment.aggregate.reason_code or "risk_rejected",
                        )
                    )
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": assessment.aggregate.reason_code or "risk_rejected",
                    "job_id": claimed.job_id,
                    "orders": 0,
                    "first_blocked_stage": DecisionTraceStage.RISK_ACCEPTED.value,
                }
            for instrument_id, quantity in payload.get("reconciled_positions", {}).items():
                self.positions.reconcile_position(
                    portfolio_id=next(
                        target.portfolio_id
                        for target in targets
                        if target.instrument_id == instrument_id
                    ),
                    instrument_id=str(instrument_id),
                    quantity=float(quantity),
                    average_entry_price=float(payload["prices"][instrument_id]),
                    updated_at=str(payload["evaluated_at"]),
                )
            current = self.positions.current_quantities(targets[0].portfolio_id)
            orders = self._plan_orders(
                targets=targets,
                current=current,
                decided_at=str(payload["evaluated_at"]),
                prices={str(key): float(value) for key, value in payload["prices"].items()},
            )
            if mode not in {"paper", "live"}:
                raise ValueError(f"unsupported execution mode: {mode}")
            venue_jobs: list[str] = []
            for order in orders:
                existing = {item.order_id: item for item in self.order_manager.all()}.get(
                    order.order_id
                )
                if existing is None:
                    self.order_manager.create(order)
                    self.order_manager.persist_for_submission(order.order_id)
                elif not _same_order_identity(existing, order):
                    raise ValueError(f"order identity collision: {order.order_id}")
                order_payload = {
                    "order_id": order.order_id,
                    "product_id": product_id,
                    "event_id": str(payload["event_id"]),
                    "price": float(payload["prices"][order.instrument_id]),
                    "execution_costs": configuration["execution_costs"],
                    "accounting_asset": configuration["base_accounting_asset"],
                    "fee_in_base": product_id == "btc_accumulation",
                    "order_group_id": order.group_id,
                }
                identity = canonical_hash(order_payload)
                job_prefix = "paper-order" if mode == "paper" else "live-order"
                job_name = "paper_order_submit" if mode == "paper" else "live_order_submit"
                job_id = f"{job_prefix}:{identity.removeprefix('sha256:')}"
                self.queue.enqueue_if_absent(
                    job_id=job_id,
                    name=job_name,
                    payload=order_payload,
                    available_at=str(payload["evaluated_at"]),
                    priority=self._order_priority(order),
                )
                venue_jobs.append(job_id)
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "execution_planning_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": f"{mode}_orders_enqueued" if orders else "target_already_satisfied",
            "job_id": claimed.job_id,
            "orders": len(orders),
            "venue_job_ids": venue_jobs,
            **({"paper_job_ids": venue_jobs} if mode == "paper" else {}),
        }

    def _plan_orders(
        self,
        *,
        targets: tuple[TargetPosition, ...],
        current: Mapping[str, float],
        decided_at: str,
        prices: Mapping[str, float],
    ) -> tuple[OrderIntent, ...]:
        grouped: dict[str, list[TargetPosition]] = {}
        standalone: list[TargetPosition] = []
        for target in targets:
            group_key = str(target.metadata.get("order_group_key") or "").strip()
            if group_key:
                grouped.setdefault(group_key, []).append(target)
            else:
                standalone.append(target)
        orders = list(
            plan_orders(
                standalone,
                current_quantities=current,
                decided_at=decided_at,
                prices=prices,
            )
        )
        if grouped and self.order_groups is None:
            raise ValueError("multi-leg targets require a durable order-group manager")
        if self.order_groups is not None:
            self.order_groups.reload()
        for group_key, group_targets in sorted(grouped.items()):
            if len(group_targets) < 2:
                raise ValueError(f"order group {group_key} has fewer than two target legs")
            recovery_policies = {
                str(target.metadata.get("recovery_policy") or "unwind") for target in group_targets
            }
            if len(recovery_policies) != 1:
                raise ValueError(f"order group {group_key} has conflicting recovery policies")
            plan = plan_order_group(
                group_targets,
                current_quantities=current,
                decided_at=decided_at,
                recovery_policy=recovery_policies.pop(),
            )
            assert self.order_groups is not None
            try:
                existing = self.order_groups.get(plan.group.group_id)
            except KeyError:
                self.order_groups.create(plan.group)
            else:
                if existing != plan.group:
                    raise ValueError(f"order-group identity collision: {plan.group.group_id}")
            orders.extend(plan.orders)
        return tuple(orders)

    def _order_priority(self, order: OrderIntent) -> int:
        if order.group_id is None or self.order_groups is None:
            return 30
        group = self.order_groups.get(order.group_id)
        return 31 if order.order_id == group.primary_order_id else 30


class DatabaseLiveExecutionWorker:
    """Submit authorised durable intents and never retry an ambiguous side effect."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        order_manager: OrderManager,
        positions: PositionManager,
        ledgers: Mapping[str, Ledger],
        trace_store: SqlDecisionTraceStore,
        venues: Mapping[str, Any],
        authorise: Callable[[Mapping[str, Any], OrderIntent], None],
        order_groups: OrderGroupManager | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.order_manager = order_manager
        self.positions = positions
        self.ledgers = dict(ledgers)
        self.trace_store = trace_store
        self.venues = dict(venues)
        self.authorise = authorise
        self.order_groups = order_groups
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("live_order_submit",),
        )
        if claimed is None:
            return {"reason_code": "live_order_queue_empty"}
        payload = claimed.payload
        try:
            self.order_manager.reload()
            self.positions.reload()
            order = self.order_manager.get(str(payload["order_id"]))
            if order.status is OrderStatus.FILLED:
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": "live_order_already_filled",
                    "job_id": claimed.job_id,
                    "order_id": order.order_id,
                }
            if order.status is not OrderStatus.PERSISTED:
                raise ValueError(
                    f"live order is not in the durable pre-submission state: {order.status.value}"
                )
            self.authorise(payload, order)
            product_id = str(payload["product_id"])
            venue = self.venues[product_id]
            if venue.order_manager is not self.order_manager:
                raise ValueError("live venue must share the durable order manager")
            previous_position = self.positions.get(order.portfolio_id, order.instrument_id)
            _before_group_submission(self.order_groups, order)
            fill = venue.submit(order)
            recorder = ExecutionService(
                paper_exchange=venue,
                positions=self.positions,
                ledger=self.ledgers[product_id],
                trace_store=self.trace_store,
            )
            recorder.record_execution_costs(order, fill, previous_position=previous_position)
            updated = self.order_manager.get(order.order_id)
            position = self.positions.get(order.portfolio_id, order.instrument_id)
            self.trace_store.append(
                _filled_trace(
                    event_id=str(payload["event_id"]),
                    order_id=order.order_id,
                    instrument_id=order.instrument_id,
                    fill_id=fill.fill_id,
                    partial=updated.status is OrderStatus.PARTIALLY_FILLED,
                    position_quantity=position.quantity,
                )
            )
            _after_group_fill(self.order_groups, self.order_manager, updated)
        except Exception as exc:
            uncertain = False
            try:
                self.order_manager.reload()
                current = self.order_manager.get(str(payload.get("order_id") or ""))
                uncertain = current.status in {
                    OrderStatus.SUBMITTED,
                    OrderStatus.ACKNOWLEDGED,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.CANCEL_PENDING,
                    OrderStatus.RECOVERY_REQUIRED,
                }
                if uncertain and current.status is not OrderStatus.RECOVERY_REQUIRED:
                    self.order_manager.recovery_required(current.order_id)
            except (KeyError, ValueError):
                pass
            if uncertain:
                _mark_group_recovery(self.order_groups, current if "current" in locals() else None)
                recovery_payload = {
                    "order_id": str(payload.get("order_id") or ""),
                    "product_id": str(payload.get("product_id") or ""),
                    "reason_code": "ambiguous_live_submission",
                    "error_type": type(exc).__name__,
                }
                recovery_job_id = "live-recovery:" + canonical_hash(recovery_payload).removeprefix(
                    "sha256:"
                )
                self.queue.enqueue_if_absent(
                    job_id=recovery_job_id,
                    name="live_order_recovery",
                    payload=recovery_payload,
                    available_at=now,
                    priority=100,
                )
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": "live_order_recovery_enqueued",
                    "job_id": claimed.job_id,
                    "recovery_job_id": recovery_job_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "live_order_rejected_before_submission",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": (
                "live_order_partially_filled"
                if updated.status is OrderStatus.PARTIALLY_FILLED
                else "live_order_filled"
            ),
            "job_id": claimed.job_id,
            "order_id": order.order_id,
            "fill_id": fill.fill_id,
            "remaining_quantity": updated.remaining_quantity,
        }


class DatabasePaperExecutionWorker:
    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        order_manager: OrderManager,
        positions: PositionManager,
        ledgers: Mapping[str, Ledger],
        trace_store: SqlDecisionTraceStore,
        order_groups: OrderGroupManager | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.order_manager = order_manager
        self.positions = positions
        self.ledgers = dict(ledgers)
        self.trace_store = trace_store
        self.order_groups = order_groups
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("paper_order_submit", "paper_order_continue"),
        )
        if claimed is None:
            return {"reason_code": "paper_order_queue_empty"}
        try:
            payload = claimed.payload
            self.order_manager.reload()
            self.positions.reload()
            order = self.order_manager.get(str(payload["order_id"]))
            if order.status is OrderStatus.FILLED:
                fills = self.order_manager.fills_for(order.order_id)
                fill_id = fills[-1].fill_id if fills else None
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": "paper_order_already_filled",
                    "job_id": claimed.job_id,
                    "order_id": order.order_id,
                    "fill_id": fill_id,
                }
            previous_position = self.positions.get(order.portfolio_id, order.instrument_id)
            costs = dict(payload["execution_costs"])
            venue = PaperExchange(
                order_manager=self.order_manager,
                position_manager=self.positions,
                price_source=lambda _instrument_id: float(payload["price"]),
                fee_bps=float(costs["fee_bps"]),
                slippage_bps=float(costs["slippage_bps"]),
                fill_fraction=float(payload.get("fill_fraction", 1.0)),
                fee_asset=str(payload["accounting_asset"]),
                fee_in_base=bool(payload["fee_in_base"]),
            )
            _before_group_submission(self.order_groups, order)
            fill = (
                venue.fill_remaining(order.order_id, fill_fraction=1.0)
                if order.status is OrderStatus.PARTIALLY_FILLED
                else venue.submit(order)
            )
            recorder = ExecutionService(
                paper_exchange=venue,
                positions=self.positions,
                ledger=self.ledgers[str(payload["product_id"])],
                trace_store=self.trace_store,
            )
            recorder.record_execution_costs(order, fill, previous_position=previous_position)
            updated = self.order_manager.get(order.order_id)
            position = self.positions.get(order.portfolio_id, order.instrument_id)
            trace = _filled_trace(
                event_id=str(payload["event_id"]),
                order_id=order.order_id,
                instrument_id=order.instrument_id,
                fill_id=fill.fill_id,
                partial=updated.status is OrderStatus.PARTIALLY_FILLED,
                position_quantity=position.quantity,
            )
            self.trace_store.append(trace)
            _after_group_fill(self.order_groups, self.order_manager, updated)
            continuation_job_id = None
            if updated.status is OrderStatus.PARTIALLY_FILLED:
                continuation_payload = {**payload, "fill_fraction": 1.0}
                continuation_job_id = "paper-order-continue:" + canonical_hash(
                    {
                        "order_id": order.order_id,
                        "filled_quantity": updated.filled_quantity,
                    }
                ).removeprefix("sha256:")
                self.queue.enqueue_if_absent(
                    job_id=continuation_job_id,
                    name="paper_order_continue",
                    payload=continuation_payload,
                    available_at=now,
                    priority=32,
                )
        except Exception as exc:
            try:
                current = self.order_manager.get(str(claimed.payload.get("order_id") or ""))
                if current.status in {
                    OrderStatus.PERSISTED,
                    OrderStatus.SUBMITTED,
                    OrderStatus.ACKNOWLEDGED,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.CANCEL_PENDING,
                }:
                    self.order_manager.recovery_required(current.order_id)
            except (KeyError, ValueError):
                pass
            _mark_group_recovery(self.order_groups, current if "current" in locals() else None)
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "paper_order_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": (
                "paper_order_partially_filled"
                if updated.status is OrderStatus.PARTIALLY_FILLED
                else "paper_order_filled"
            ),
            "job_id": claimed.job_id,
            "order_id": order.order_id,
            "fill_id": fill.fill_id,
            "remaining_quantity": updated.remaining_quantity,
            "continuation_job_id": continuation_job_id,
        }


class DatabaseUserStreamWorker:
    """Persist account events and route balance updates to accounting."""

    def __init__(
        self,
        *,
        engine: Engine,
        queue: DatabaseJobQueue,
        worker_id: str,
        order_manager: OrderManager | None = None,
        positions: PositionManager | None = None,
        ledgers: Mapping[str, Ledger] | None = None,
        trace_store: SqlDecisionTraceStore | None = None,
        account_products: Mapping[str, str] | None = None,
        order_groups: OrderGroupManager | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.engine = engine
        self.queue = queue
        self.worker_id = worker_id
        self.order_manager = order_manager
        self.positions = positions
        self.ledgers = dict(ledgers or {})
        self.trace_store = trace_store
        self.account_products = dict(account_products or {})
        self.order_groups = order_groups
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("user_stream_event",),
        )
        if claimed is None:
            return {"reason_code": "user_stream_queue_empty"}
        try:
            event = MarketEvent(**dict(claimed.payload["event"]))
            record = {
                "account_id": str(claimed.payload["account_id"]),
                "market": str(claimed.payload["market"]),
                "event": claimed.payload["event"],
            }
            identity = canonical_hash(record)
            with self.engine.begin() as connection:
                existing = connection.execute(
                    select(reconciliation_event.c.payload).where(
                        reconciliation_event.c.id == identity
                    )
                ).scalar_one_or_none()
                if existing is None:
                    connection.execute(
                        insert(reconciliation_event).values(
                            id=identity,
                            created_at=event.receive_timestamp,
                            payload=record,
                        )
                    )
                elif dict(existing) != record:
                    raise ValueError("user-stream event identity collision")
            balances = _balance_update(event)
            accounting_job_id = None
            if balances:
                accounting_payload = {
                    "kind": "balance",
                    "account_id": record["account_id"],
                    "observed_at": event.receive_timestamp,
                    "balances": balances,
                }
                accounting_job_id = "accounting:" + canonical_hash(accounting_payload).removeprefix(
                    "sha256:"
                )
                self.queue.enqueue_if_absent(
                    job_id=accounting_job_id,
                    name="accounting_event",
                    payload=accounting_payload,
                    available_at=event.receive_timestamp,
                    priority=20,
                )
            order_result = self._apply_order_event(
                event=event,
                account_id=record["account_id"],
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "user_stream_event_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "user_stream_event_recorded",
            "job_id": claimed.job_id,
            "event_id": identity,
            "event_type": event.event_type.value,
            "accounting_job_id": accounting_job_id,
            "order_result": order_result,
        }

    def _apply_order_event(self, *, event: MarketEvent, account_id: str) -> dict[str, Any] | None:
        if event.event_type not in {MarketEventType.ORDER_UPDATE, MarketEventType.FILL_UPDATE}:
            return None
        if self.order_manager is None or self.positions is None or self.trace_store is None:
            return {"reason_code": "order_state_handler_not_configured"}
        raw_data = event.payload.get("data")
        if not isinstance(raw_data, Mapping):
            raise ValueError("order update has no data object")
        raw_order = raw_data.get("o")
        values = raw_order if isinstance(raw_order, Mapping) else raw_data
        client_order_id = str(values.get("c") or values.get("C") or "")
        if not client_order_id:
            raise ValueError("order update has no client order ID")
        self.order_manager.reload()
        self.positions.reload()
        matches = tuple(
            order for order in self.order_manager.all() if order.order_id[:36] == client_order_id
        )
        if len(matches) != 1:
            recovery_payload = {
                "account_id": account_id,
                "client_order_id": client_order_id,
                "reason_code": "unknown_or_ambiguous_exchange_order",
                "event_id": event.event_id,
            }
            job_id = "live-recovery:" + canonical_hash(recovery_payload).removeprefix("sha256:")
            self.queue.enqueue_if_absent(
                job_id=job_id,
                name="live_order_recovery",
                payload=recovery_payload,
                available_at=event.receive_timestamp,
                priority=100,
            )
            return {"reason_code": recovery_payload["reason_code"], "recovery_job_id": job_id}
        order = matches[0]
        if order.status is OrderStatus.FILLED:
            return {"reason_code": "exchange_order_already_filled", "order_id": order.order_id}
        if event.event_type is MarketEventType.FILL_UPDATE:
            return self._apply_fill_event(
                event=event,
                account_id=account_id,
                order=order,
                values=values,
            )
        return self._apply_status_event(order=order, values=values)

    def _apply_fill_event(
        self,
        *,
        event: MarketEvent,
        account_id: str,
        order: OrderIntent,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        quantity = float(values.get("l", 0.0))
        price = float(values.get("L", 0.0))
        fee = float(values.get("n", 0.0) or 0.0)
        fee_asset = str(values.get("N") or "").upper() or None
        trade_id = str(values.get("t") or values.get("T") or event.sequence)
        if quantity <= 0 or price <= 0 or fee < 0:
            raise ValueError("fill update has invalid quantity, price, or fee")
        fill_id = canonical_hash(
            {
                "venue": "binance",
                "account_id": account_id,
                "instrument_id": order.instrument_id,
                "trade_id": trade_id,
            }
        )
        if any(fill.fill_id == fill_id for fill in self.order_manager.fills_for(order.order_id)):
            return {"reason_code": "exchange_fill_already_recorded", "fill_id": fill_id}
        product_id = self.account_products.get(account_id)
        ledger = self.ledgers.get(product_id) if product_id is not None else None
        if fee and ledger is not None and fee_asset != ledger.accounting_asset:
            if order.status is not OrderStatus.RECOVERY_REQUIRED:
                order = self.order_manager.recovery_required(order.order_id)
            recovery_payload = {
                "account_id": account_id,
                "order_id": order.order_id,
                "product_id": product_id,
                "event_id": event.event_id,
                "fill_id": fill_id,
                "fee": fee,
                "fee_asset": fee_asset,
                "accounting_asset": ledger.accounting_asset,
                "reason_code": "fee_conversion_required",
            }
            recovery_job_id = "live-recovery:" + canonical_hash(recovery_payload).removeprefix(
                "sha256:"
            )
            self.queue.enqueue_if_absent(
                job_id=recovery_job_id,
                name="live_order_recovery",
                payload=recovery_payload,
                available_at=event.receive_timestamp,
                priority=100,
            )
            return {
                "reason_code": "fee_conversion_required",
                "order_id": order.order_id,
                "fill_id": fill_id,
                "recovery_job_id": recovery_job_id,
            }
        if order.status is OrderStatus.PERSISTED:
            self.order_manager.submitted(order.order_id)
            self.order_manager.acknowledged(order.order_id)
        elif order.status is OrderStatus.SUBMITTED:
            self.order_manager.acknowledged(order.order_id)
        previous_position = self.positions.get(order.portfolio_id, order.instrument_id)
        fill = Fill(
            fill_id=fill_id,
            order_id=order.order_id,
            instrument_id=order.instrument_id,
            side=OrderSide(str(values.get("S") or "").lower()),
            quantity=quantity,
            price=price,
            fee=fee,
            occurred_at=event.exchange_timestamp,
            fee_asset=fee_asset,
            metadata={"reference_price": price, "slippage_cost": 0.0, "user_stream": True},
        )
        updated = self.order_manager.apply_fill(fill)
        position = self.positions.apply_fill(
            order.portfolio_id,
            fill,
            contributions=dict(order.strategy_contributions),
        )
        if product_id is not None and product_id in self.ledgers:
            recorder = ExecutionService(
                paper_exchange=_RecordedVenue(self.order_manager),
                positions=self.positions,
                ledger=self.ledgers[product_id],
                trace_store=self.trace_store,
            )
            recorder.record_execution_costs(order, fill, previous_position=previous_position)
        self.trace_store.append(
            _filled_trace(
                event_id=event.event_id,
                order_id=order.order_id,
                instrument_id=order.instrument_id,
                fill_id=fill.fill_id,
                partial=updated.status is OrderStatus.PARTIALLY_FILLED,
                position_quantity=position.quantity,
            )
        )
        _after_group_fill(self.order_groups, self.order_manager, updated)
        return {
            "reason_code": (
                "exchange_order_partially_filled"
                if updated.status is OrderStatus.PARTIALLY_FILLED
                else "exchange_order_filled"
            ),
            "order_id": order.order_id,
            "fill_id": fill.fill_id,
            "remaining_quantity": updated.remaining_quantity,
        }

    def _apply_status_event(
        self, *, order: OrderIntent, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        status = str(values.get("X") or values.get("x") or "").upper()
        if order.status is OrderStatus.PERSISTED:
            self.order_manager.submitted(order.order_id)
            order = self.order_manager.acknowledged(order.order_id)
        elif order.status is OrderStatus.SUBMITTED:
            order = self.order_manager.acknowledged(order.order_id)
        if status == "NEW":
            pass
        elif status in {"CANCELED", "CANCELLED"}:
            if order.status in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}:
                self.order_manager.request_cancel(order.order_id)
            order = self.order_manager.cancelled(order.order_id)
            _mark_group_recovery(self.order_groups, order)
        elif status == "REJECTED":
            order = self.order_manager.transition(order.order_id, OrderStatus.REJECTED)
        elif status == "EXPIRED":
            order = self.order_manager.transition(order.order_id, OrderStatus.EXPIRED)
        elif status == "FILLED" and order.status is not OrderStatus.FILLED:
            order = self.order_manager.recovery_required(order.order_id)
        elif status not in {"PARTIALLY_FILLED", "FILLED"}:
            raise ValueError(f"unsupported exchange order status: {status}")
        return {
            "reason_code": f"exchange_order_{order.status.value}",
            "order_id": order.order_id,
        }


class _RecordedVenue:
    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager

    def submit(self, _intent: OrderIntent) -> Fill:
        raise RuntimeError("recorded user-stream venues cannot submit orders")


def _before_group_submission(
    manager: OrderGroupManager | None,
    order: OrderIntent,
) -> None:
    if manager is None or order.group_id is None:
        return
    manager.reload()
    group = manager.get(order.group_id)
    if order.order_id == group.primary_order_id:
        if group.status is OrderGroupStatus.PLANNED:
            manager.transition(group.group_id, OrderGroupStatus.PRIMARY_SUBMITTED)
        elif group.status not in {
            OrderGroupStatus.PRIMARY_SUBMITTED,
            OrderGroupStatus.PRIMARY_PARTIAL,
        }:
            raise ValueError(f"primary leg cannot submit from group state {group.status.value}")
        return
    if order.order_id not in group.hedge_order_ids:
        raise ValueError(f"order {order.order_id} is not a member of group {group.group_id}")
    if group.status in {
        OrderGroupStatus.PRIMARY_SUBMITTED,
        OrderGroupStatus.PRIMARY_PARTIAL,
    }:
        manager.transition(group.group_id, OrderGroupStatus.HEDGE_SUBMITTED)
    elif group.status is not OrderGroupStatus.HEDGE_SUBMITTED:
        raise ValueError(f"hedge leg cannot submit from group state {group.status.value}")


def _after_group_fill(
    manager: OrderGroupManager | None,
    orders: OrderManager,
    order: OrderIntent,
) -> None:
    if manager is None or order.group_id is None:
        return
    manager.reload()
    group = manager.get(order.group_id)
    if order.order_id == group.primary_order_id:
        if (
            order.status is OrderStatus.PARTIALLY_FILLED
            and group.status is OrderGroupStatus.PRIMARY_SUBMITTED
        ):
            manager.transition(group.group_id, OrderGroupStatus.PRIMARY_PARTIAL)
        return
    if order.order_id not in group.hedge_order_ids:
        raise ValueError(f"order {order.order_id} is not a member of group {group.group_id}")
    if order.status is not OrderStatus.FILLED:
        return
    orders.reload()
    if all(orders.get(order_id).status is OrderStatus.FILLED for order_id in group.hedge_order_ids):
        if group.status is not OrderGroupStatus.HEDGE_SUBMITTED:
            raise ValueError(f"hedges filled from invalid group state {group.status.value}")
        manager.transition(group.group_id, OrderGroupStatus.HEDGED)
        manager.transition(group.group_id, OrderGroupStatus.ACTIVE)


def _mark_group_recovery(
    manager: OrderGroupManager | None,
    order: OrderIntent | None,
) -> None:
    if manager is None or order is None or order.group_id is None:
        return
    manager.reload()
    group = manager.get(order.group_id)
    if group.status not in {OrderGroupStatus.RECOVERY, OrderGroupStatus.FLAT}:
        manager.transition(group.group_id, OrderGroupStatus.RECOVERY)


def _balance_update(event: MarketEvent) -> dict[str, float]:
    if event.event_type is not MarketEventType.ACCOUNT_BALANCE:
        return {}
    raw_data = event.payload.get("data")
    if not isinstance(raw_data, Mapping):
        raise ValueError("account-balance event has no data object")
    account_update = raw_data.get("a")
    if isinstance(account_update, Mapping):
        rows = account_update.get("B")
    else:
        rows = raw_data.get("B")
    balances: dict[str, float] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("account-balance row must be an object")
            asset = str(row.get("a") or "").upper()
            if not asset:
                raise ValueError("account-balance row has no asset")
            free = float(row.get("f", row.get("wb", 0.0)))
            locked = float(row.get("l", 0.0))
            balances[asset] = free + locked
    return balances


def _risk_rejected_trace(
    *, event_id: str, target: TargetPosition, reason_code: str
) -> DecisionTrace:
    return (
        DecisionTrace.start(event_id=event_id, instrument_id=target.instrument_id)
        .pass_stage(DecisionTraceStage.DATA_AVAILABLE)
        .pass_stage(DecisionTraceStage.FEATURE_AVAILABLE)
        .pass_stage(DecisionTraceStage.STRATEGY_EVALUATED)
        .pass_stage(DecisionTraceStage.REGIME_PASSED)
        .pass_stage(DecisionTraceStage.SETUP_PASSED)
        .pass_stage(DecisionTraceStage.TRIGGER_PASSED)
        .pass_stage(DecisionTraceStage.SIGNAL_PRODUCED)
        .pass_stage(DecisionTraceStage.PORTFOLIO_ACCEPTED)
        .block(DecisionTraceStage.RISK_ACCEPTED, reason_code=reason_code)
    )


def _same_order_identity(existing, planned) -> bool:
    immutable_fields = (
        "order_id",
        "portfolio_id",
        "instrument_id",
        "side",
        "quantity",
        "order_type",
        "created_at",
        "limit_price",
        "reduce_only",
        "group_id",
        "strategy_contributions",
        "metadata",
    )
    return all(getattr(existing, name) == getattr(planned, name) for name in immutable_fields)


def _filled_trace(
    *,
    event_id: str,
    order_id: str,
    instrument_id: str,
    fill_id: str,
    partial: bool,
    position_quantity: float,
) -> DecisionTrace:
    trace = (
        DecisionTrace.start(event_id=f"{event_id}:{order_id}", instrument_id=instrument_id)
        .pass_stage(DecisionTraceStage.DATA_AVAILABLE)
        .pass_stage(DecisionTraceStage.FEATURE_AVAILABLE)
        .pass_stage(DecisionTraceStage.STRATEGY_EVALUATED)
        .pass_stage(DecisionTraceStage.REGIME_PASSED)
        .pass_stage(DecisionTraceStage.SETUP_PASSED)
        .pass_stage(DecisionTraceStage.TRIGGER_PASSED)
        .pass_stage(DecisionTraceStage.SIGNAL_PRODUCED)
        .pass_stage(DecisionTraceStage.PORTFOLIO_ACCEPTED)
        .pass_stage(DecisionTraceStage.RISK_ACCEPTED)
        .pass_stage(DecisionTraceStage.ORDER_PLANNED)
        .pass_stage(DecisionTraceStage.ORDER_SUBMITTED)
    )
    if partial:
        return trace.block(
            DecisionTraceStage.ORDER_FILLED,
            reason_code="partial_fill_pending",
            fill_id=fill_id,
        )
    trace = trace.pass_stage(DecisionTraceStage.ORDER_FILLED, fill_id=fill_id)
    if position_quantity == 0:
        trace = trace.pass_stage(DecisionTraceStage.POSITION_OPENED)
        return trace.pass_stage(DecisionTraceStage.POSITION_CLOSED, quantity=position_quantity)
    return trace.pass_stage(DecisionTraceStage.POSITION_OPENED, quantity=position_quantity)


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
