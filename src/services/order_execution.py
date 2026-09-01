"""Independent durable order-planning and paper-execution queue workers."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.accounting.ledger import Ledger
from src.data.database import account_snapshot, reconciliation_event
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
from src.products.btc_accumulation import assert_btc_spot_instrument
from src.risk.engine import SqlRiskDecisionStore, SqlRiskSnapshotStore
from src.services.execution_service import ExecutionService
from src.services.scheduler import DatabaseJobQueue


def _split_order_groups(
    targets: tuple[TargetPosition, ...],
) -> tuple[dict[str, list[TargetPosition]], list[TargetPosition]]:
    grouped: dict[str, list[TargetPosition]] = {}
    standalone: list[TargetPosition] = []
    for target in targets:
        group_key = str(target.metadata.get("order_group_key") or "").strip()
        if group_key:
            grouped.setdefault(group_key, []).append(target)
        else:
            standalone.append(target)
    return grouped, standalone


def _validate_fill_values(quantity: float, price: float, fee: float) -> None:
    if quantity <= 0 or price <= 0 or fee < 0:
        raise ValueError("fill update has invalid quantity, price, or fee")


def _acknowledge_order(manager: OrderManager, order: OrderIntent, event_at: str) -> OrderIntent:
    if order.status is OrderStatus.PERSISTED:
        manager.submitted(order.order_id)
        return manager.acknowledged(order.order_id, event_at=event_at)
    if order.status is OrderStatus.SUBMITTED:
        return manager.acknowledged(order.order_id, event_at=event_at)
    return order


def _apply_exchange_status(
    manager: OrderManager,
    order: OrderIntent,
    status: str,
    *,
    order_groups: OrderGroupManager | None,
) -> OrderIntent:
    if status in {"NEW", "PARTIALLY_FILLED"}:
        return order
    if status in {"CANCELED", "CANCELLED"}:
        if order.status in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}:
            manager.request_cancel(order.order_id)
        cancelled = manager.cancelled(order.order_id)
        _mark_group_recovery(order_groups, cancelled)
        return cancelled
    if status == "REJECTED":
        return manager.transition(order.order_id, OrderStatus.REJECTED)
    if status == "EXPIRED":
        return manager.transition(order.order_id, OrderStatus.EXPIRED)
    if status == "FILLED":
        return (
            order
            if order.status is OrderStatus.FILLED
            else manager.recovery_required(order.order_id)
        )
    raise ValueError(f"unsupported exchange order status: {status}")


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
        snapshot_store: SqlRiskSnapshotStore | None = None,
        control_plane: Any | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.order_manager = order_manager
        self.positions = positions
        self.risk_store = risk_store
        self.trace_store = trace_store
        self.order_groups = order_groups
        self.snapshot_store = snapshot_store
        self.control_plane = control_plane
        self.product_execution = {
            product_id: dict(configuration)
            for product_id, configuration in product_execution.items()
        }
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        now = timestamp(now, field="now")
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("execute_targets",),
        )
        if claimed is None:
            return {"reason_code": "execution_queue_empty"}
        try:
            result = self._process_execution_job(claimed.payload, now)
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
        return {**result, "job_id": claimed.job_id}

    def _process_execution_job(self, payload: Mapping[str, Any], now: str) -> dict[str, Any]:
        self.order_manager.reload()
        self.positions.reload()
        product_id, mode, configuration, assessment, inputs, targets = self._execution_inputs(
            payload
        )
        if not assessment.accepted:
            self._record_risk_rejections(
                targets,
                event_id=str(payload["event_id"]),
                reason_code=assessment.aggregate.reason_code or "risk_rejected",
            )
            return {
                "reason_code": assessment.aggregate.reason_code or "risk_rejected",
                "orders": 0,
                "first_blocked_stage": DecisionTraceStage.RISK_ACCEPTED.value,
            }
        self._reconcile_positions(inputs, targets, evaluated_at=str(payload["evaluated_at"]))
        current = self.positions.current_quantities(targets[0].portfolio_id)
        prices = {str(key): float(value) for key, value in inputs["prices"].items()}
        orders = self._plan_orders(
            targets=targets,
            current=current,
            decided_at=str(payload["evaluated_at"]),
            prices=prices,
        )
        if product_id == "btc_accumulation" and "balances" in inputs:
            _validate_btc_spot_orders(
                orders,
                current=current,
                balances=inputs["balances"],
                prices=prices,
                execution_costs=configuration["execution_costs"],
            )
        venue_jobs = [
            job_id
            for order in orders
            if (
                job_id := self._enqueue_order(
                    order, payload, product_id, mode, configuration, prices
                )
            )
        ]
        return {
            "reason_code": f"{mode}_orders_enqueued" if venue_jobs else "target_already_satisfied",
            "orders": len(venue_jobs),
            "venue_job_ids": venue_jobs,
            **({"paper_job_ids": venue_jobs} if mode == "paper" else {}),
        }

    def _execution_inputs(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, str, Mapping[str, Any], Any, Mapping[str, Any], tuple[TargetPosition, ...]]:
        product_id = str(payload["product_id"])
        configuration = self.product_execution[product_id]
        mode = str(payload["execution_mode"])
        if mode != configuration["execution_mode"]:
            raise ValueError("execution job mode differs from product configuration")
        if mode not in {"paper", "live"}:
            raise ValueError(f"unsupported execution mode: {mode}")
        assessment = self.risk_store.assessment(str(payload["risk_assessment_id"]))
        if assessment.aggregate.input_snapshot.get("product_id") != product_id:
            raise ValueError("execution risk assessment belongs to another product")
        inputs = self._canonical_inputs(payload)
        targets = tuple(TargetPosition(**dict(item)) for item in inputs["targets"])
        if not targets:
            raise ValueError("execution job has no targets")
        return product_id, mode, configuration, assessment, inputs, targets

    def _record_risk_rejections(
        self, targets: tuple[TargetPosition, ...], *, event_id: str, reason_code: str
    ) -> None:
        for target in targets:
            self.trace_store.append(
                _risk_rejected_trace(event_id=event_id, target=target, reason_code=reason_code)
            )

    def _reconcile_positions(
        self,
        inputs: Mapping[str, Any],
        targets: tuple[TargetPosition, ...],
        *,
        evaluated_at: str,
    ) -> None:
        for instrument_id, quantity in inputs["reconciled_positions"].items():
            target = next((item for item in targets if item.instrument_id == instrument_id), None)
            if target is None:
                raise ValueError(f"reconciled position is outside target scope: {instrument_id}")
            self.positions.reconcile_position(
                portfolio_id=target.portfolio_id,
                instrument_id=str(instrument_id),
                quantity=float(quantity),
                average_entry_price=float(inputs["prices"][instrument_id]),
                updated_at=evaluated_at,
            )

    def _enqueue_order(
        self,
        order: OrderIntent,
        payload: Mapping[str, Any],
        product_id: str,
        mode: str,
        configuration: Mapping[str, Any],
        prices: Mapping[str, float],
    ) -> str | None:
        if self._blocks_new_risk(product_id, order):
            self._record_control_rejection(order, payload, prices)
            return None
        existing = {item.order_id: item for item in self.order_manager.all()}.get(order.order_id)
        if existing is None:
            self.order_manager.create(order)
            self.order_manager.persist_for_submission(order.order_id)
        elif not _same_order_identity(existing, order):
            raise ValueError(f"order identity collision: {order.order_id}")
        order_payload = self._order_payload(order, payload, product_id, configuration, prices)
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
        return job_id

    def _record_control_rejection(
        self,
        order: OrderIntent,
        payload: Mapping[str, Any],
        prices: Mapping[str, float],
    ) -> None:
        target_metadata = order.metadata.get("target_metadata")
        if order.valid_until is None:
            raise ValueError("control rejection requires an order expiry")
        target = TargetPosition(
            portfolio_id=order.portfolio_id,
            instrument_id=order.instrument_id,
            target_quantity=order.quantity,
            target_notional=order.quantity * float(prices[order.instrument_id]),
            target_fraction=0.0,
            strategy_contributions=order.strategy_contributions,
            risk_budget=float(order.metadata.get("risk_budget") or 0.0),
            valid_until=order.valid_until,
            metadata=dict(target_metadata) if isinstance(target_metadata, Mapping) else {},
        )
        self.trace_store.append(
            _risk_rejected_trace(
                event_id=str(payload["event_id"]),
                target=target,
                reason_code="control_plane_blocks_new_risk",
            )
        )

    @staticmethod
    def _order_payload(
        order: OrderIntent,
        payload: Mapping[str, Any],
        product_id: str,
        configuration: Mapping[str, Any],
        prices: Mapping[str, float],
    ) -> dict[str, Any]:
        target_metadata = order.metadata.get("target_metadata")
        assignment_id = (
            target_metadata.get("assignment_id") if isinstance(target_metadata, Mapping) else None
        )
        result = {
            "order_id": order.order_id,
            "product_id": product_id,
            "event_id": str(payload["event_id"]),
            "price": float(prices[order.instrument_id]),
            "execution_costs": configuration["execution_costs"],
            "accounting_asset": configuration["base_accounting_asset"],
            "fee_in_base": product_id == "btc_accumulation",
            "order_group_id": order.group_id,
            "strategy_version_ids": sorted(order.strategy_contributions),
            "strategy_version_id": (
                next(iter(order.strategy_contributions))
                if len(order.strategy_contributions) == 1
                else None
            ),
            "assignment_id": assignment_id,
        }
        if "fill_fraction" in configuration:
            result["fill_fraction"] = float(configuration["fill_fraction"])
        return result

    def _blocks_new_risk(self, product_id: str, order: OrderIntent) -> bool:
        if self.control_plane is None or order.reduce_only:
            return False
        strategy_ids = tuple(sorted(order.strategy_contributions))
        strategy_id = strategy_ids[0] if len(strategy_ids) == 1 else None
        return bool(
            self.control_plane.blocks_new_risk(
                product_id=product_id,
                strategy_id=strategy_id,
            )
        )

    def _canonical_inputs(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        snapshot_id = payload.get("target_position_snapshot_id")
        if snapshot_id is None:
            return {
                "targets": payload["targets"],
                "prices": payload["prices"],
                "reconciled_positions": payload.get("reconciled_positions", {}),
            }
        if self.snapshot_store is None:
            raise ValueError("canonical execution input store is not configured")
        snapshot = self.snapshot_store.get(str(snapshot_id))
        if not isinstance(snapshot.get("targets"), list):
            raise ValueError("target snapshot has no targets")
        if not isinstance(snapshot.get("prices"), Mapping):
            raise ValueError("target snapshot has no prices")
        reconciled_positions = snapshot.get("reconciled_positions", {})
        if not isinstance(reconciled_positions, Mapping):
            raise ValueError("target snapshot has invalid reconciled positions")
        return {
            "targets": snapshot["targets"],
            "prices": snapshot["prices"],
            "balances": snapshot.get("balances", {}),
            "reconciled_positions": reconciled_positions,
        }

    def _plan_orders(
        self,
        *,
        targets: tuple[TargetPosition, ...],
        current: Mapping[str, float],
        decided_at: str,
        prices: Mapping[str, float],
    ) -> tuple[OrderIntent, ...]:
        grouped, standalone = _split_order_groups(targets)
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
            orders.extend(
                self._plan_group(
                    group_key,
                    group_targets,
                    current=current,
                    decided_at=decided_at,
                    prices=prices,
                )
            )
        return tuple(orders)

    def _plan_group(
        self,
        group_key: str,
        targets: list[TargetPosition],
        *,
        current: Mapping[str, float],
        decided_at: str,
        prices: Mapping[str, float],
    ) -> tuple[OrderIntent, ...]:
        if len(targets) < 2:
            raise ValueError(f"order group {group_key} has fewer than two target legs")
        policies = {str(target.metadata.get("recovery_policy") or "unwind") for target in targets}
        if len(policies) != 1:
            raise ValueError(f"order group {group_key} has conflicting recovery policies")
        plan = plan_order_group(
            targets,
            current_quantities=current,
            decided_at=decided_at,
            recovery_policy=policies.pop(),
            prices=prices,
        )
        assert self.order_groups is not None
        try:
            existing = self.order_groups.get(plan.group.group_id)
        except KeyError:
            self.order_groups.create(plan.group)
        else:
            if existing != plan.group:
                raise ValueError(f"order-group identity collision: {plan.group.group_id}")
        return plan.orders

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
        job_name: str = "live_order_submit",
        prepare_protective_stop: Callable[[str, OrderIntent, str], object] | None = None,
        control_plane: Any | None = None,
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
        self.job_name = job_name
        self.prepare_protective_stop = prepare_protective_stop
        self.control_plane = control_plane

    def run_once(self, *, now: str) -> dict[str, Any]:
        now = timestamp(now, field="now")
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=(self.job_name,),
        )
        if claimed is None:
            return {"reason_code": "live_order_queue_empty"}
        payload = claimed.payload
        try:
            result = self._submit_order(payload, now)
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
        return {**result, "job_id": claimed.job_id}

    def _submit_order(self, payload: Mapping[str, Any], now: str) -> dict[str, Any]:
        self.order_manager.reload()
        self.positions.reload()
        order = self.order_manager.get(str(payload["order_id"]))
        early = self._early_order_result(order, now)
        if early is not None:
            return early
        self._assert_order_dependencies(order)
        product_id = str(payload["product_id"])
        self._assert_control_allows(product_id, order, payload)
        self.authorise({**payload, "authorisation_at": now}, order)
        acknowledgement = self._submit_to_venue(product_id, order, now)
        updated = self.order_manager.get(order.order_id)
        return {
            "reason_code": "live_order_acknowledged",
            "order_id": order.order_id,
            "exchange_order_id": acknowledgement.exchange_order_id,
            "client_order_id": acknowledgement.client_order_id,
            "remaining_quantity": updated.remaining_quantity,
        }

    def _early_order_result(self, order: OrderIntent, now: str) -> dict[str, Any] | None:
        if order.status is OrderStatus.FILLED:
            return {"reason_code": "live_order_already_filled", "order_id": order.order_id}
        if order.status is not OrderStatus.PERSISTED:
            raise ValueError(
                f"live order is not in the durable pre-submission state: {order.status.value}"
            )
        if order.valid_until is None:
            raise ValueError("live order has no expiry")
        if now < order.valid_until:
            return None
        self.order_manager.transition(order.order_id, OrderStatus.EXPIRED, event_at=now)
        return {
            "reason_code": "live_order_expired",
            "order_id": order.order_id,
            "valid_until": order.valid_until,
        }

    def _assert_order_dependencies(self, order: OrderIntent) -> None:
        if order.depends_on_order_id is None:
            return
        dependency = self.order_manager.get(order.depends_on_order_id)
        if dependency.status is not OrderStatus.FILLED:
            raise ValueError("dependent opening order is blocked until close fill")
        reconciled = self.positions.get(order.portfolio_id, order.instrument_id)
        if abs(reconciled.quantity) > 1e-12:
            raise ValueError("dependent opening order is blocked until position is flat")

    def _assert_control_allows(
        self, product_id: str, order: OrderIntent, payload: Mapping[str, Any]
    ) -> None:
        if (
            self.control_plane is not None
            and not order.reduce_only
            and self.control_plane.blocks_new_risk(
                product_id=product_id,
                strategy_id=str(payload.get("strategy_version_id") or "") or None,
            )
        ):
            raise PermissionError("control plane blocks new live risk")

    def _submit_to_venue(self, product_id: str, order: OrderIntent, now: str) -> Any:
        venue = self.venues[product_id]
        if venue.order_manager is not self.order_manager:
            raise ValueError("live venue must share the durable order manager")
        if not order.reduce_only and self.prepare_protective_stop is not None:
            self.prepare_protective_stop(product_id, order, now)
        _before_group_submission(self.order_groups, order)
        return venue.submit(order)


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
            if now >= order.valid_until and not order.is_terminal:
                self.order_manager.transition(
                    order.order_id,
                    OrderStatus.EXPIRED,
                    event_at=now,
                )
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": "paper_order_expired",
                    "job_id": claimed.job_id,
                    "order_id": order.order_id,
                    "valid_until": order.valid_until,
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
        job_name: str = "user_stream_event",
        accounting_job_name: str = "accounting_event",
        accounting_job_prefix: str = "accounting",
        on_live_fill: Callable[[str, OrderIntent, float, str], object] | None = None,
        on_algo_update: Callable[[str, MarketEvent], object] | None = None,
        fee_converter: Callable[[str, str, str, float, float], Mapping[str, Any]] | None = None,
        on_order_status: Callable[[str, OrderIntent, str, str], object] | None = None,
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
        self.job_name = job_name
        self.accounting_job_name = accounting_job_name
        self.accounting_job_prefix = accounting_job_prefix
        self.on_live_fill = on_live_fill
        self.on_algo_update = on_algo_update
        self.fee_converter = fee_converter
        self.on_order_status = on_order_status

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=(self.job_name,),
        )
        if claimed is None:
            return {"reason_code": "user_stream_queue_empty"}
        try:
            event = MarketEvent(**dict(claimed.payload["event"]))
            product_id = self.account_products.get(str(claimed.payload["account_id"]))
            record = {
                "account_id": str(claimed.payload["account_id"]),
                "market": str(claimed.payload["market"]),
                "event": claimed.payload["event"],
                **({"product_id": product_id} if product_id is not None else {}),
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
            if product_id is not None:
                _mark_account_authority_unknown(
                    engine=self.engine,
                    account_id=record["account_id"],
                    product_id=product_id,
                    observed_at=event.receive_timestamp,
                    event_id=event.event_id,
                )
            balances = _balance_update(event)
            accounting_job_id = None
            if balances:
                accounting_payload = {
                    "kind": "balance",
                    "account_id": record["account_id"],
                    "observed_at": event.receive_timestamp,
                    "balances": balances,
                    "account_state_known": False,
                    "account_state_authority": "user_stream_delta",
                    **({"product_id": product_id} if product_id is not None else {}),
                }
                accounting_job_id = (
                    self.accounting_job_prefix
                    + ":"
                    + canonical_hash(accounting_payload).removeprefix("sha256:")
                )
                self.queue.enqueue_if_absent(
                    job_id=accounting_job_id,
                    name=self.accounting_job_name,
                    payload=accounting_payload,
                    available_at=event.receive_timestamp,
                    priority=20,
                )
            order_result = self._apply_order_event(
                event=event,
                account_id=record["account_id"],
                product_id=product_id,
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

    def _apply_order_event(
        self, *, event: MarketEvent, account_id: str, product_id: str | None
    ) -> dict[str, Any] | None:
        if event.event_type is MarketEventType.ALGO_UPDATE:
            if self.on_algo_update is None or product_id is None:
                return {"reason_code": "protective_algo_handler_not_configured"}
            result = self.on_algo_update(product_id, event)
            return result if isinstance(result, dict) else {"reason_code": "algo_update_applied"}
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
            order
            for order in self.order_manager.all()
            if order.metadata.get("client_order_id") == client_order_id
            or order.order_id[:36] == client_order_id
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
        if event.event_type is MarketEventType.FILL_UPDATE:
            return self._apply_fill_event(
                event=event,
                account_id=account_id,
                order=order,
                values=values,
                product_id=product_id,
            )
        if order.status is OrderStatus.FILLED:
            return {"reason_code": "exchange_order_already_filled", "order_id": order.order_id}
        result = self._apply_status_event(order=order, values=values, event=event)
        self._notify_order_status(product_id, order.order_id, event.receive_timestamp)
        return result

    def _notify_order_status(self, product_id: str | None, order_id: str, at: str) -> None:
        if self.on_order_status is None or product_id is None or self.order_manager is None:
            return
        current = self.order_manager.get(order_id)
        self.on_order_status(product_id, current, current.status.value, at)

    def _apply_fill_event(
        self,
        *,
        event: MarketEvent,
        account_id: str,
        order: OrderIntent,
        values: Mapping[str, Any],
        product_id: str | None,
    ) -> dict[str, Any]:
        if self.order_manager is None or self.positions is None or self.trace_store is None:
            raise RuntimeError("user-stream worker requires durable execution stores")
        order_manager = self.order_manager
        positions = self.positions
        trace_store = self.trace_store
        quantity = float(values.get("l", 0.0))
        price = float(values.get("L", 0.0))
        fee = float(values.get("n", 0.0) or 0.0)
        fee_asset = str(values.get("N") or "").upper() or None
        trade_id = str(values.get("t") or values.get("T") or event.sequence)
        _validate_fill_values(quantity, price, fee)
        fill_id = canonical_hash(
            {
                "venue": "binance",
                "instrument_id": order.instrument_id,
                "trade_id": trade_id,
            }
        )
        if any(fill.fill_id == fill_id for fill in order_manager.fills_for(order.order_id)):
            return self._duplicate_fill_result(order, fill_id, product_id, event.receive_timestamp)
        product_id = self.account_products.get(account_id)
        ledger = self.ledgers.get(product_id) if product_id is not None else None
        if fee and ledger is not None and fee_asset != ledger.accounting_asset:
            try:
                conversion = self._fee_conversion_metadata(
                    product_id=product_id,
                    instrument_id=order.instrument_id,
                    fee_asset=fee_asset,
                    fee=fee,
                    price=price,
                )
            except Exception:
                return self._fee_conversion_recovery(
                    order=order,
                    account_id=account_id,
                    product_id=product_id,
                    event=event,
                    fill_id=fill_id,
                    fee=fee,
                    fee_asset=fee_asset,
                    accounting_asset=ledger.accounting_asset,
                )
        else:
            conversion = {}
        if order.status is OrderStatus.PERSISTED:
            order_manager.submitted(order.order_id)
            order_manager.acknowledged(order.order_id, event_at=event.exchange_timestamp)
        elif order.status is OrderStatus.SUBMITTED:
            order_manager.acknowledged(order.order_id, event_at=event.exchange_timestamp)
        previous_position = positions.get(order.portfolio_id, order.instrument_id)
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
            metadata={
                "reference_price": price,
                "slippage_cost": 0.0,
                "user_stream": True,
                "exchange_order_id": str(values.get("i") or "") or None,
                "trade_id": trade_id,
                **conversion,
            },
        )
        updated = order_manager.apply_fill(fill)
        position = positions.apply_fill(
            order.portfolio_id,
            fill,
            contributions=dict(order.strategy_contributions),
        )
        if product_id is not None and product_id in self.ledgers:
            recorder = ExecutionService(
                paper_exchange=_RecordedVenue(order_manager),
                positions=positions,
                ledger=self.ledgers[product_id],
                trace_store=trace_store,
            )
            recorder.record_execution_costs(order, fill, previous_position=previous_position)
        trace_store.append(
            _filled_trace(
                event_id=event.event_id,
                order_id=order.order_id,
                instrument_id=order.instrument_id,
                fill_id=fill.fill_id,
                partial=updated.status is OrderStatus.PARTIALLY_FILLED,
                position_quantity=position.quantity,
            )
        )
        _after_group_fill(self.order_groups, order_manager, updated)
        if self.on_live_fill is not None and product_id is not None:
            self.on_live_fill(product_id, order, position.quantity, event.receive_timestamp)
        return {
            "reason_code": (
                "exchange_order_partially_filled"
                if updated.status is OrderStatus.PARTIALLY_FILLED
                else "exchange_order_filled"
            ),
            "order_id": order.order_id,
            "fill_id": fill.fill_id,
            "position_quantity": position.quantity,
            "remaining_quantity": updated.remaining_quantity,
        }

    def _fee_conversion_metadata(
        self,
        *,
        product_id: str | None,
        instrument_id: str,
        fee_asset: str | None,
        fee: float,
        price: float,
    ) -> Mapping[str, Any]:
        if self.fee_converter is None or product_id is None:
            raise ValueError("fee conversion callback is unavailable")
        result = self.fee_converter(
            product_id,
            instrument_id,
            str(fee_asset or ""),
            fee,
            price,
        )
        if not isinstance(result, Mapping):
            raise ValueError("fee converter returned an invalid conversion record")
        return dict(result)

    def _duplicate_fill_result(
        self,
        order: OrderIntent,
        fill_id: str,
        product_id: str | None,
        received_at: str,
    ) -> dict[str, Any]:
        if self.positions is None:
            raise RuntimeError("user-stream worker requires a durable position store")
        position = self.positions.get(order.portfolio_id, order.instrument_id)
        if self.on_live_fill is not None and product_id is not None:
            self.on_live_fill(product_id, order, position.quantity, received_at)
        return {
            "reason_code": "exchange_fill_already_recorded",
            "fill_id": fill_id,
            "position_quantity": position.quantity,
        }

    def _fee_conversion_recovery(
        self,
        *,
        order: OrderIntent,
        account_id: str,
        product_id: str | None,
        event: MarketEvent,
        fill_id: str,
        fee: float,
        fee_asset: str | None,
        accounting_asset: str,
    ) -> dict[str, Any]:
        if self.order_manager is None:
            raise RuntimeError("user-stream worker requires a durable order store")
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
            "accounting_asset": accounting_asset,
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

    def _apply_status_event(
        self, *, order: OrderIntent, values: Mapping[str, Any], event: MarketEvent
    ) -> dict[str, Any]:
        if self.order_manager is None:
            raise RuntimeError("user-stream worker requires a durable order store")
        order_manager = self.order_manager
        status = str(values.get("X") or values.get("x") or "").upper()
        order = _acknowledge_order(order_manager, order, event.exchange_timestamp)
        order = _apply_exchange_status(
            order_manager,
            order,
            status,
            order_groups=self.order_groups,
        )
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


def _mark_account_authority_unknown(
    *, engine: Engine, account_id: str, product_id: str, observed_at: str, event_id: str
) -> None:
    """Invalidate the last REST snapshot until a fresh reconciliation completes."""

    payload: dict[str, Any] = {
        "account_id": account_id,
        "product_id": product_id,
        "balances": {},
        "free_balances": {},
        "positions": {},
        "regular_orders": [],
        "conditional_orders": [],
        "used_margin": None,
        "maintenance_margin": None,
        "used_margin_fraction": None,
        "liquidation_buffer_fraction": None,
        "account_mode": "unknown",
        "unknown_exposure": {"account_state": "reconciliation_required"},
        "account_state_known": False,
        "account_state_authority": "user_stream_delta",
        "event_id": event_id,
        "observed_at": observed_at,
    }
    snapshot_id = canonical_hash(
        {"account_id": account_id, "product_id": product_id, "payload": payload}
    )
    with engine.begin() as connection:
        existing = connection.execute(
            select(account_snapshot.c.payload).where(account_snapshot.c.id == snapshot_id)
        ).scalar_one_or_none()
        if existing is None:
            connection.execute(
                insert(account_snapshot).values(
                    id=snapshot_id,
                    account_id=account_id,
                    observed_at=observed_at,
                    source="user_stream_delta",
                    content_hash=canonical_hash(payload),
                    payload=payload,
                )
            )
        elif dict(existing) != payload:
            raise ValueError("account authority invalidation identity collision")


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
        "valid_until",
        "limit_price",
        "reduce_only",
        "group_id",
        "strategy_contributions",
        "metadata",
    )
    return all(getattr(existing, name) == getattr(planned, name) for name in immutable_fields)


def _validate_btc_spot_orders(
    orders: tuple[OrderIntent, ...],
    *,
    current: Mapping[str, float],
    balances: Mapping[str, Any],
    prices: Mapping[str, float],
    execution_costs: Mapping[str, Any],
) -> None:
    """Keep BTC spot orders inside owned inventory and same-cycle quote proceeds."""

    _validate_btc_spot_identity(orders, balances)
    if not orders:
        return
    instrument_id = orders[0].instrument_id
    owned_btc = float(current.get(instrument_id, 0.0))
    _validate_owned_btc(owned_btc)
    sell_quantity = sum(order.quantity for order in orders if order.side is OrderSide.SELL)
    price = float(prices[instrument_id])
    if sell_quantity > owned_btc + max(1e-12, owned_btc * 1e-9):
        raise ValueError("BTC spot sell exceeds the reconciled owned BTC position")
    fee_bps, slippage_bps = _btc_spot_costs(execution_costs)
    quote_balance = _btc_quote_balance(balances)
    _assert_btc_quote_capacity(
        orders,
        price,
        fee_bps,
        slippage_bps,
        quote_balance,
        sell_quantity,
    )


def _validate_btc_spot_identity(
    orders: tuple[OrderIntent, ...], balances: Mapping[str, Any]
) -> None:
    if not isinstance(balances, Mapping):
        raise ValueError("BTC spot execution requires canonical account balances")
    for order in orders:
        try:
            assert_btc_spot_instrument(order.instrument_id)
        except ValueError as exc:
            raise ValueError("BTC accumulation orders must use BTCUSDT spot") from exc


def _validate_owned_btc(value: float) -> None:
    if not math.isfinite(value) or value < -1e-12:
        raise ValueError("BTC spot position is invalid")


def _btc_spot_costs(execution_costs: Mapping[str, Any]) -> tuple[float, float]:
    try:
        fee_bps = float(execution_costs["fee_bps"])
        slippage_bps = float(execution_costs["slippage_bps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BTC spot execution costs are invalid") from exc
    if not all(math.isfinite(value) and value >= 0.0 for value in (fee_bps, slippage_bps)):
        raise ValueError("BTC spot execution costs are invalid")
    return fee_bps, slippage_bps


def _btc_quote_balance(balances: Mapping[str, Any]) -> float:
    quote_asset = next(
        (str(key).upper() for key in balances if str(key).upper() in {"USDT", "USDC", "BUSD"}),
        "USDT",
    )
    quote_balance = float(balances.get(quote_asset, 0.0))
    if not math.isfinite(quote_balance) or quote_balance < -1e-12:
        raise ValueError("BTC spot quote balance is invalid")
    return quote_balance


def _assert_btc_quote_capacity(
    orders: tuple[OrderIntent, ...],
    price: float,
    fee_bps: float,
    slippage_bps: float,
    quote_balance: float,
    sell_quantity: float,
) -> None:
    if not math.isfinite(price) or price <= 0:
        raise ValueError("BTC spot execution price is invalid")
    buy_cost = sum(
        order.quantity
        * price
        * (1.0 + slippage_bps / 10_000.0)
        * (1.0 + fee_bps / 10_000.0)
        for order in orders
        if order.side is OrderSide.BUY
    )
    sell_proceeds = (
        sell_quantity
        * price
        * max(0.0, 1.0 - slippage_bps / 10_000.0)
        * max(0.0, 1.0 - fee_bps / 10_000.0)
    )
    if buy_cost > quote_balance + sell_proceeds + max(1e-12, quote_balance * 1e-9):
        raise ValueError("BTC spot buys exceed quote balance and same-cycle sell proceeds")


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
