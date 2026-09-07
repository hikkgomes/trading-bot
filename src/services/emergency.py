"""Idempotent emergency reduction and flatten workers."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from sqlalchemy import insert, select, update

from src.data.database import emergency_action
from src.domain._codec import canonical_hash, json_value, timestamp
from src.domain.orders import OrderIntent, OrderSide, OrderStatus, OrderType
from src.execution.live_exchange import bounded_client_order_id
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.services.alerting import AlertSeverity, SqlAlertService
from src.services.scheduler import DatabaseJobQueue


class DatabaseEmergencyFlattenWorker:
    """Cancel new-risk orders, then submit deterministic reduce-only orders."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        order_manager: OrderManager,
        positions: PositionManager,
        venues: Mapping[str, Any],
        products: Mapping[str, Mapping[str, Any]],
        lease_seconds: int = 60,
        alerts: SqlAlertService | None = None,
        reconcile_product: Callable[[str], Any] | None = None,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.order_manager = order_manager
        self.positions = positions
        self.venues = dict(venues)
        self.products = {str(key): dict(value) for key, value in products.items()}
        self.lease_seconds = lease_seconds
        self.alerts = alerts
        self.reconcile_product = reconcile_product

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("emergency_flatten", "emergency_reduction", "cancel_entry_orders"),
        )
        if claimed is None:
            return {"reason_code": "emergency_queue_empty"}
        return self._run_claimed(claimed, now=now)

    def _run_claimed(self, claimed, *, now: str) -> dict[str, Any]:
        previous_action = self._action(claimed.job_id)
        previous_payload: Mapping[str, Any] = (
            dict(previous_action["payload"])
            if isinstance(previous_action, Mapping)
            and isinstance(previous_action.get("payload"), Mapping)
            else {}
        )
        target = str(claimed.payload.get("target") or "global")
        state_product_id = str(claimed.payload.get("product_id") or "") or (
            target.removeprefix("product:") if target.startswith("product:") else "global"
        )
        self._transition(
            job_id=claimed.job_id,
            product_id=state_product_id,
            state="requested",
            now=now,
            payload={"target": target, "reason_code": claimed.payload.get("reason_code")},
        )
        try:
            name, cancelled, flattened, action_context, reconciled, reconciliation_detail = (
                self._execute_action(
                    claimed,
                    now=now,
                    target=target,
                    state_product_id=state_product_id,
                    previous_payload=previous_payload,
                )
            )
        except Exception as exc:
            return self._fail_action(
                claimed,
                now=now,
                product_id=state_product_id,
                error=exc,
            )
        if not reconciled:
            return self._defer_action(
                claimed,
                now=now,
                reconciliation_detail=reconciliation_detail,
                cancelled=cancelled,
                flattened=flattened,
            )
        return self._complete_action(
            claimed,
            now=now,
            state_product_id=state_product_id,
            name=name,
            action_context=action_context,
            cancelled=cancelled,
            flattened=flattened,
            reconciliation_detail=reconciliation_detail,
        )

    def _execute_action(
        self,
        claimed,
        *,
        now: str,
        target: str,
        state_product_id: str,
        previous_payload: Mapping[str, Any],
    ) -> tuple[str, int, list[dict[str, Any]], Mapping[str, Any], bool, dict[str, Any]]:
        self.order_manager.reload()
        self.positions.reload()
        name = claimed.name
        product_ids = self._product_ids(target, claimed.payload)
        self._transition(
            job_id=claimed.job_id,
            product_id=state_product_id,
            state="entries_blocked",
            now=now,
            payload={"product_ids": list(product_ids)},
        )
        exchange_states = self._refresh_exchange_state(product_ids)
        baseline_exchange_states = (
            _exchange_states_from_payload(previous_payload.get("baseline_exchange_states"))
            or exchange_states
        )
        self._transition(
            job_id=claimed.job_id,
            product_id=state_product_id,
            state="exchange_state_refreshed",
            now=now,
            payload={
                "products": sorted(exchange_states),
                "exchange_states": exchange_states,
                "baseline_exchange_states": baseline_exchange_states,
            },
        )
        cancelled = self._cancel_entries(product_ids, now, exchange_states=exchange_states)
        reduction_targets = _reduction_targets_from_payload(
            previous_payload.get("reduction_targets")
        )
        if name != "cancel_entry_orders":
            for product_id in product_ids:
                reduction_targets.setdefault(
                    product_id,
                    self._reduction_targets(
                        portfolio_id=str(self.products[product_id].get("portfolio_id") or ""),
                        payload=claimed.payload,
                        product_id=product_id,
                        exchange_state=exchange_states.get(product_id),
                    ),
                )
        action_context = {
            "reduction_targets": _serialise_reduction_targets(reduction_targets),
            "baseline_exchange_states": baseline_exchange_states,
        }
        self._transition(
            job_id=claimed.job_id,
            product_id=state_product_id,
            state="close_planned",
            now=now,
            payload={"cancelled_entries": cancelled, **action_context},
        )
        flattened: list[dict[str, Any]] = []
        if name != "cancel_entry_orders":
            for product_id in product_ids:
                flattened.extend(
                    self._reduce_product(
                        product_id,
                        claimed.payload,
                        now,
                        exchange_state=exchange_states.get(product_id),
                        planned_targets=reduction_targets.get(product_id, ()),
                    )
                )
        if flattened:
            self._transition(
                job_id=claimed.job_id,
                product_id=state_product_id,
                state="close_submitted",
                now=now,
                payload={**action_context, "orders": flattened},
            )
            if _all_reductions_acknowledged(flattened):
                self._transition(
                    job_id=claimed.job_id,
                    product_id=state_product_id,
                    state="close_acknowledged",
                    now=now,
                    payload={**action_context, "orders": flattened},
                )
        post_exchange_states = self._refresh_exchange_state(product_ids)
        self._sync_local_positions(post_exchange_states)
        reconciled, reconciliation_detail = self._reconcile_action_state(
            name=name,
            product_ids=product_ids,
            exchange_states=post_exchange_states or exchange_states,
            reduction_targets=reduction_targets,
            baseline_exchange_states=baseline_exchange_states,
        )
        unresolved_orders = self._unresolved_reduction_orders(flattened)
        if unresolved_orders:
            reconciled = False
            reconciliation_detail = {
                **reconciliation_detail,
                "unresolved_emergency_orders": unresolved_orders,
            }
        if reconciled:
            self._transition(
                job_id=claimed.job_id,
                product_id=state_product_id,
                state=("reconciled_orders" if name == "cancel_entry_orders" else "fills_observed"),
                now=now,
                payload={**action_context, "reconciliation": reconciliation_detail},
            )
        return name, cancelled, flattened, action_context, reconciled, reconciliation_detail

    def _reconcile_action_state(
        self,
        *,
        name: str,
        product_ids: tuple[str, ...],
        exchange_states: Mapping[str, Mapping[str, Any]],
        reduction_targets: Mapping[str, tuple[tuple[str, float], ...]],
        baseline_exchange_states: Mapping[str, Mapping[str, Any]],
    ) -> tuple[bool, dict[str, Any]]:
        if name == "cancel_entry_orders":
            return self._reconcile_orders_only(
                product_ids,
                exchange_states=exchange_states,
            )
        return self._reconcile_products(
            product_ids,
            exchange_states=exchange_states,
            require_flat=name != "emergency_reduction",
            reduction_targets=reduction_targets,
            before_exchange_states=baseline_exchange_states,
        )

    def _unresolved_reduction_orders(self, results: Sequence[Mapping[str, Any]]) -> list[str]:
        unresolved: list[str] = []
        for result in results:
            order_id = str(result.get("order_id") or "")
            if not order_id:
                continue
            try:
                status = self.order_manager.get(order_id).status
            except KeyError:
                unresolved.append(order_id)
                continue
            if status in {
                OrderStatus.PERSISTED,
                OrderStatus.SUBMITTED,
                OrderStatus.ACKNOWLEDGED,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.CANCEL_PENDING,
                OrderStatus.RECOVERY_REQUIRED,
            }:
                unresolved.append(order_id)
        return sorted(set(unresolved))

    def _action(self, job_id: str) -> Mapping[str, Any] | None:
        with self.queue.engine.connect() as connection:
            row = (
                connection.execute(
                    select(emergency_action.c.state, emergency_action.c.payload).where(
                        emergency_action.c.job_id == job_id
                    )
                )
                .mappings()
                .first()
            )
        return None if row is None else dict(row)

    def _fail_action(
        self, claimed, *, now: str, product_id: str, error: Exception
    ) -> dict[str, Any]:
        current_action = self._action(claimed.job_id)
        current_payload: Mapping[str, Any] = (
            dict(current_action["payload"])
            if isinstance(current_action, Mapping)
            and isinstance(current_action.get("payload"), Mapping)
            else {}
        )
        preserved = {
            key: current_payload[key]
            for key in ("reduction_targets", "baseline_exchange_states", "orders")
            if key in current_payload
        }
        self._transition(
            job_id=claimed.job_id,
            product_id=product_id,
            state="recovery_required",
            now=now,
            payload={
                **preserved,
                "error_type": type(error).__name__,
                "error": str(error)[:500],
            },
        )
        self._emit_alert(
            event_type="emergency_action_failed",
            dedupe_key=f"emergency:{claimed.job_id}:failed",
            target=str(claimed.payload.get("target") or "global"),
            message=f"emergency action failed: {type(error).__name__}",
            emitted_at=now,
            payload={"job_id": claimed.job_id, "error_type": type(error).__name__},
        )
        self.queue.fail(
            claimed,
            completed_at=now,
            error=f"{type(error).__name__}: {error}",
            retry_at=_retry_at(now, self.lease_seconds),
        )
        return {
            "reason_code": "emergency_action_failed",
            "job_id": claimed.job_id,
            "error_type": type(error).__name__,
            "error": str(error),
        }

    def _defer_action(
        self,
        claimed,
        *,
        now: str,
        reconciliation_detail: Mapping[str, Any],
        cancelled: int,
        flattened: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        self.queue.defer(
            claimed,
            available_at=_retry_at(now, min(self.lease_seconds, 30)),
            reason="emergency_reconciliation_pending",
        )
        self._emit_alert(
            event_type="emergency_reconciliation_pending",
            dedupe_key=f"emergency:{claimed.job_id}:pending",
            target=str(claimed.payload.get("target") or "global"),
            message="emergency action is waiting for authenticated reconciliation",
            emitted_at=now,
            payload={
                "job_id": claimed.job_id,
                "cancelled_entries": cancelled,
                "flattened_orders": len(flattened),
                "reconciliation": reconciliation_detail,
            },
        )
        return {
            "reason_code": "emergency_reconciliation_pending",
            "job_id": claimed.job_id,
            "cancelled_entries": cancelled,
            "flattened": flattened,
            "reconciliation": reconciliation_detail,
        }

    def _complete_action(
        self,
        claimed,
        *,
        now: str,
        state_product_id: str,
        name: str,
        action_context: Mapping[str, Any],
        cancelled: int,
        flattened: Sequence[Mapping[str, Any]],
        reconciliation_detail: Mapping[str, Any],
    ) -> dict[str, Any]:
        final_state = (
            "reconciled_orders"
            if name == "cancel_entry_orders"
            else "reconciled_flat"
            if name == "emergency_flatten"
            else "reconciled_reduced"
        )
        payload = {**action_context, "reconciliation": reconciliation_detail}
        self._transition(
            job_id=claimed.job_id,
            product_id=state_product_id,
            state=final_state,
            now=now,
            payload=payload,
        )
        self._transition(
            job_id=claimed.job_id,
            product_id=state_product_id,
            state="completed",
            now=now,
            payload=payload,
        )
        self.queue.complete(claimed, completed_at=now)
        self._emit_alert(
            event_type="emergency_action_completed",
            dedupe_key=f"emergency:{claimed.job_id}:completed",
            target=str(claimed.payload.get("target") or "global"),
            message="emergency action completed",
            emitted_at=now,
            payload={
                "job_id": claimed.job_id,
                "cancelled_entries": cancelled,
                "flattened_orders": len(flattened),
                "reconciliation": reconciliation_detail,
            },
        )
        return {
            "reason_code": "emergency_action_completed",
            "job_id": claimed.job_id,
            "cancelled_entries": cancelled,
            "flattened": flattened,
            "reconciliation": reconciliation_detail,
        }

    def _transition(
        self,
        *,
        job_id: str,
        product_id: str,
        state: str,
        now: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Persist the emergency state before the next external side effect."""

        now = timestamp(now, field="emergency state timestamp")
        state_payload = json_value(
            {
                "schema": "platform.emergency_action/v1",
                "job_id": job_id,
                "product_id": product_id,
                "state": state,
                **dict(payload),
            },
            field="emergency state payload",
        )
        content_hash = canonical_hash(state_payload)
        identity = canonical_hash({"job_id": job_id})
        with self.queue.engine.begin() as connection:
            existing = connection.execute(
                select(emergency_action.c.attempts).where(emergency_action.c.job_id == job_id)
            ).scalar_one_or_none()
            attempts = int(existing or 0) + (1 if state == "requested" else 0)
            values = {
                "id": identity,
                "job_id": job_id,
                "product_id": product_id,
                "state": state,
                "attempts": attempts,
                "updated_at": now,
                "content_hash": content_hash,
                "payload": state_payload,
            }
            if existing is None:
                connection.execute(insert(emergency_action).values(**values))
            else:
                connection.execute(
                    update(emergency_action)
                    .where(emergency_action.c.job_id == job_id)
                    .values(**{key: value for key, value in values.items() if key != "id"})
                )

    def _refresh_exchange_state(self, product_ids: tuple[str, ...]) -> dict[str, Mapping[str, Any]]:
        """Refresh authenticated state before deciding what must be reduced."""

        states: dict[str, Mapping[str, Any]] = {}
        for product_id in product_ids:
            venue = self.venues.get(product_id)
            broker = getattr(venue, "broker", None)
            reader = getattr(broker, "account_snapshot", None)
            if not callable(reader):
                continue
            symbols = tuple(
                str(instrument.exchange_symbol)
                for instrument in getattr(venue, "instruments", {}).values()
            )
            state = reader(expected_symbols=symbols)
            if not isinstance(state, Mapping):
                raise RuntimeError(
                    f"authenticated emergency account state is invalid for {product_id}"
                )
            states[product_id] = dict(state)
        return states

    def _reconcile_products(
        self,
        product_ids: tuple[str, ...],
        *,
        exchange_states: Mapping[str, Mapping[str, Any]],
        require_flat: bool = True,
        reduction_targets: Mapping[str, tuple[tuple[str, float], ...]] | None = None,
        before_exchange_states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        details: dict[str, Any] = {}
        all_matched = True
        for product_id in product_ids:
            detail: dict[str, Any]
            state = exchange_states.get(product_id)
            exchange_ready, exchange_detail = self._flat_exchange_state(
                state, require_flat=require_flat
            )
            if not require_flat and exchange_ready:
                exchange_ready = self._reduction_exchange_state(
                    before=(before_exchange_states or exchange_states).get(product_id),
                    after=state,
                    targets=(reduction_targets or {}).get(product_id, ()),
                )
                exchange_detail = {
                    **exchange_detail,
                    "reduction_reconciled": exchange_ready,
                }
            if self.reconcile_product is not None:
                result = self.reconcile_product(product_id)
                matched = bool(getattr(result, "matched", False)) and exchange_ready
                detail = {
                    "matched": matched,
                    "recovery_required": bool(getattr(result, "recovery_required", not matched)),
                    **exchange_detail,
                }
            elif state is not None:
                matched = exchange_ready
                detail = {"matched": matched, **exchange_detail}
            else:
                matched = False
                detail = {
                    "matched": False,
                    "authenticated": False,
                    "reason_code": "authenticated_account_state_unavailable",
                }
            details[product_id] = detail
            all_matched = all_matched and matched
        return all_matched, details

    @staticmethod
    def _reduction_exchange_state(
        *,
        before: Mapping[str, Any] | None,
        after: Mapping[str, Any] | None,
        targets: tuple[tuple[str, float], ...],
    ) -> bool:
        if before is None or after is None or not targets:
            return False
        before_positions = before.get("positions")
        after_positions = after.get("positions")
        if not isinstance(before_positions, Mapping) or not isinstance(after_positions, Mapping):
            return False
        for instrument_id, requested in targets:
            try:
                before_quantity = float(before_positions.get(instrument_id, 0.0))
                after_quantity = float(after_positions.get(instrument_id, 0.0))
                requested_quantity = abs(float(requested))
            except (TypeError, ValueError):
                return False
            if not all(
                math.isfinite(value)
                for value in (before_quantity, after_quantity, requested_quantity)
            ):
                return False
            expected_maximum = max(0.0, abs(before_quantity) - requested_quantity)
            if abs(after_quantity) > expected_maximum + max(1e-12, abs(before_quantity) * 1e-9):
                return False
        return True

    def _reconcile_orders_only(
        self,
        product_ids: tuple[str, ...],
        *,
        exchange_states: Mapping[str, Mapping[str, Any]],
    ) -> tuple[bool, dict[str, Any]]:
        details: dict[str, Any] = {}
        complete = True
        for product_id in product_ids:
            state = exchange_states.get(product_id)
            if state is None:
                details[product_id] = {
                    "matched": False,
                    "authenticated": False,
                    "reason_code": "authenticated_account_state_unavailable",
                }
                complete = False
                continue
            authenticated = (
                state.get("account_state_known") is True
                and state.get("account_state_authority") == "authenticated_rest"
            )
            regular = state.get("regular_orders")
            conditional = state.get("conditional_orders")
            matched = authenticated and isinstance(regular, list | tuple) and not regular
            matched = matched and isinstance(conditional, list | tuple) and not conditional
            details[product_id] = {
                "matched": matched,
                "authenticated": authenticated,
                "regular_orders": len(regular) if isinstance(regular, list | tuple) else None,
                "conditional_orders": (
                    len(conditional) if isinstance(conditional, list | tuple) else None
                ),
            }
            complete = complete and matched
        return complete, details

    @staticmethod
    def _flat_exchange_state(
        state: Mapping[str, Any] | None,
        *,
        require_flat: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        if state is None:
            return False, {
                "authenticated": False,
                "reason_code": "authenticated_account_state_unavailable",
            }
        positions = state.get("positions")
        orders = state.get("regular_orders")
        conditionals = state.get("conditional_orders")
        authenticated = (
            state.get("account_state_known") is True
            and state.get("account_state_authority") == "authenticated_rest"
        )
        if isinstance(positions, Mapping):
            valid_positions = all(math.isfinite(float(value)) for value in positions.values())
            flat = valid_positions and all(
                abs(float(value)) <= 1e-12 for value in positions.values()
            )
        else:
            valid_positions = False
            flat = False
        no_open_orders = isinstance(orders, list | tuple) and not orders
        no_conditionals = isinstance(conditionals, list | tuple) and not conditionals
        no_unknown = state.get("unknown_exposure") == {}
        ready = (
            authenticated and valid_positions and no_open_orders and no_conditionals and no_unknown
        )
        ready = ready and (flat if require_flat else True)
        return ready, {
            "exchange_flat": flat,
            "exchange_ready": ready,
            "authenticated": authenticated,
            "exchange_positions": dict(positions) if isinstance(positions, Mapping) else None,
            "regular_orders": len(orders) if isinstance(orders, list | tuple) else None,
            "conditional_orders": (
                len(conditionals) if isinstance(conditionals, list | tuple) else None
            ),
            "unknown_exposure": dict(state.get("unknown_exposure") or {})
            if isinstance(state.get("unknown_exposure"), Mapping)
            else state.get("unknown_exposure"),
        }

    def _product_ids(self, target: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
        explicit = str(payload.get("product_id") or "").strip()
        if explicit:
            if explicit not in self.products:
                raise ValueError(f"emergency product is not configured: {explicit}")
            return (explicit,)
        if target in {"", "global"}:
            return tuple(sorted(self.venues))
        if target.startswith("product:"):
            product_id = target.removeprefix("product:")
            if product_id not in self.products:
                raise ValueError(f"emergency product is not configured: {product_id}")
            return (product_id,)
        if target in self.products:
            return (target,)
        if target.startswith("strategy:"):
            return tuple(sorted(self.venues))
        raise ValueError(f"emergency target is unsupported: {target}")

    def _cancel_entries(
        self,
        product_ids: tuple[str, ...],
        now: str,
        *,
        exchange_states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> int:
        portfolios = {
            str(self.products[product_id].get("portfolio_id") or "") for product_id in product_ids
        }
        cancelled = self._cancel_exchange_entries(product_ids, exchange_states or {})
        for order in self.order_manager.all():
            if (
                order.portfolio_id not in portfolios
                or order.reduce_only
                or order.is_terminal
                or order.status is OrderStatus.RECOVERY_REQUIRED
            ):
                continue
            product_id = next(
                product_id
                for product_id in product_ids
                if str(self.products[product_id].get("portfolio_id")) == order.portfolio_id
            )
            venue = self.venues.get(product_id)
            if venue is None:
                raise ValueError(f"no emergency venue for {product_id}")
            if order.status in {OrderStatus.CREATED, OrderStatus.VALIDATED, OrderStatus.PERSISTED}:
                terminal = (
                    OrderStatus.REJECTED
                    if order.status in {OrderStatus.CREATED, OrderStatus.VALIDATED}
                    else OrderStatus.EXPIRED
                )
                self.order_manager.transition(order.order_id, terminal, event_at=now)
                cancelled += 1
                continue
            if not hasattr(venue, "cancel"):
                self.order_manager.recovery_required(order.order_id)
                continue
            venue.cancel(order)
            cancelled += 1
        return cancelled

    def _cancel_exchange_entries(
        self,
        product_ids: tuple[str, ...],
        exchange_states: Mapping[str, Mapping[str, Any]],
    ) -> int:
        """Cancel every authenticated regular and conditional open order."""

        cancelled = 0
        for product_id in product_ids:
            state = exchange_states.get(product_id)
            if state is None:
                continue
            venue = self.venues.get(product_id)
            if venue is None:
                raise ValueError(f"no emergency venue for {product_id}")
            broker = getattr(venue, "broker", None)
            if broker is None:
                raise RuntimeError(f"emergency venue has no broker for {product_id}")
            for conditional in (False, True):
                raw_orders = state.get("conditional_orders" if conditional else "regular_orders")
                if not isinstance(raw_orders, list | tuple):
                    raise RuntimeError("authenticated emergency order inventory is invalid")
                for raw in raw_orders:
                    # An emergency close already in flight is part of the
                    # flatten operation, not an entry to cancel.
                    if bool(_order_value(raw, "reduce_only")):
                        continue
                    exchange_id, client_id = self._cancel_exchange_order(
                        broker, raw, conditional=conditional
                    )
                    cancelled += 1
        return cancelled

    def _cancel_exchange_order(
        self, broker: Any, raw: Any, *, conditional: bool
    ) -> tuple[str, str]:
        symbol = str(_order_value(raw, "symbol") or "")
        exchange_id = str(_order_value(raw, "order_id") or "")
        client_id = str(_order_value(raw, "client_id") or "")
        if not symbol or not exchange_id and not client_id:
            raise RuntimeError("authenticated emergency order identity is incomplete")
        if conditional and callable(getattr(broker, "cancel_protective_stop", None)):
            result = broker.cancel_protective_stop(
                symbol=symbol,
                order_id=exchange_id or None,
                client_id=client_id,
            )
        else:
            cancel = getattr(broker, "cancel_order", None)
            if not callable(cancel):
                raise RuntimeError("emergency broker cannot cancel open orders")
            result = cancel(
                symbol=symbol,
                exchange_order_id=exchange_id,
                client_order_id=client_id,
            )
        status = str(getattr(result, "status", "canceled")).casefold()
        if status in {"open", "new", "accepted", "partially_filled"}:
            raise RuntimeError("exchange order remains open after emergency cancellation")
        self._mark_local_order_cancelled(exchange_id=exchange_id, client_id=client_id)
        return exchange_id, client_id

    def _mark_local_order_cancelled(self, *, exchange_id: str, client_id: str) -> None:
        for order in self.order_manager.all():
            if order.is_terminal:
                continue
            if exchange_id not in {
                str(order.metadata.get("exchange_order_id") or ""),
                str(order.metadata.get("client_order_id") or ""),
            } and client_id not in {
                str(order.metadata.get("exchange_order_id") or ""),
                str(order.metadata.get("client_order_id") or ""),
            }:
                continue
            try:
                if order.status in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}:
                    self.order_manager.request_cancel(order.order_id)
                    self.order_manager.cancelled(order.order_id)
            except ValueError:
                self.order_manager.recovery_required(order.order_id)

    def _reduce_product(
        self,
        product_id: str,
        payload: Mapping[str, Any],
        now: str,
        *,
        exchange_state: Mapping[str, Any] | None = None,
        planned_targets: tuple[tuple[str, float], ...] | None = None,
    ) -> list[dict[str, Any]]:
        product = self.products[product_id]
        portfolio_id = str(product.get("portfolio_id") or "")
        if not portfolio_id:
            raise ValueError(f"product {product_id} has no portfolio_id")
        venue = self.venues.get(product_id)
        if venue is None:
            raise ValueError(f"no emergency venue for {product_id}")
        results: list[dict[str, Any]] = []
        reduction_targets = (
            planned_targets
            if planned_targets is not None
            else self._reduction_targets(
                portfolio_id=portfolio_id,
                payload=payload,
                product_id=product_id,
                exchange_state=exchange_state,
            )
        )
        for instrument_id, position_quantity in reduction_targets:
            quantity = self._reduction_quantity(product_id, instrument_id, position_quantity)
            if quantity <= 1e-12:
                continue
            side = OrderSide.SELL if position_quantity > 0 else OrderSide.BUY
            reference_price = _reference_price(venue, instrument_id)
            control_id = str(payload.get("control_id") or payload.get("stop_id") or "emergency")
            unsigned = {
                "control_id": control_id,
                "product_id": product_id,
                "instrument_id": instrument_id,
                "quantity": quantity,
                "side": side.value,
            }
            order_id = "emergency:" + canonical_hash(unsigned).removeprefix("sha256:")[:40]
            existing = next(
                (order for order in self.order_manager.all() if order.order_id == order_id),
                None,
            )
            if existing is not None:
                result = self._resume_existing_reduction(
                    product_id=product_id,
                    instrument_id=instrument_id,
                    order=existing,
                    venue=venue,
                    now=now,
                )
                results.append(result)
                continue
            intent = OrderIntent(
                order_id=order_id,
                portfolio_id=portfolio_id,
                instrument_id=instrument_id,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                created_at=timestamp(now, field="emergency order.created_at"),
                reduce_only=True,
                strategy_contributions={"emergency": 1.0},
                metadata={
                    "control_id": control_id,
                    "reason_code": str(payload.get("reason_code") or "emergency_flatten"),
                    "reference_price": reference_price,
                    "emergency": True,
                },
            )
            self.order_manager.create(intent)
            self.order_manager.persist_for_submission(intent.order_id)
            try:
                acknowledgement = venue.submit(intent)
            except Exception as exc:
                self.order_manager.recovery_required(intent.order_id)
                self._enqueue_order_recovery(product_id=product_id, order=intent, now=now)
                results.append(
                    {
                        "product_id": product_id,
                        "instrument_id": instrument_id,
                        "order_id": order_id,
                        "status": OrderStatus.RECOVERY_REQUIRED.value,
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            results.append(
                {
                    "product_id": product_id,
                    "instrument_id": instrument_id,
                    "order_id": order_id,
                    "status": str(getattr(acknowledgement, "status", "submitted")),
                }
            )
        return results

    def _resume_existing_reduction(
        self,
        *,
        product_id: str,
        instrument_id: str,
        order: OrderIntent,
        venue: Any,
        now: str,
    ) -> dict[str, Any]:
        """Resume a durable reduction without repeating an uncertain submission."""

        if order.status is OrderStatus.PERSISTED:
            try:
                acknowledgement = venue.submit(order)
            except Exception as exc:
                self.order_manager.recovery_required(order.order_id)
                self._enqueue_order_recovery(product_id=product_id, order=order, now=now)
                return {
                    "product_id": product_id,
                    "instrument_id": instrument_id,
                    "order_id": order.order_id,
                    "status": OrderStatus.RECOVERY_REQUIRED.value,
                    "error_type": type(exc).__name__,
                }
            return {
                "product_id": product_id,
                "instrument_id": instrument_id,
                "order_id": order.order_id,
                "status": str(getattr(acknowledgement, "status", "acknowledged")),
            }
        if order.status in {
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.RECOVERY_REQUIRED,
            OrderStatus.CANCEL_PENDING,
        }:
            try:
                state = self._query_ambiguous_order(venue, instrument_id, order)
            except Exception:
                state = None
            if state is not None:
                status = str(getattr(state, "status", "")).casefold()
                if status in {"open", "new", "accepted", "partially_filled"}:
                    self._enqueue_order_recovery(product_id=product_id, order=order, now=now)
                elif status in {"filled", "closed", "reconciled"}:
                    self._enqueue_order_recovery(product_id=product_id, order=order, now=now)
                return {
                    "product_id": product_id,
                    "instrument_id": instrument_id,
                    "order_id": order.order_id,
                    "status": status or order.status.value,
                }
            self._enqueue_order_recovery(product_id=product_id, order=order, now=now)
            return {
                "product_id": product_id,
                "instrument_id": instrument_id,
                "order_id": order.order_id,
                "status": OrderStatus.RECOVERY_REQUIRED.value,
            }
        return {
            "product_id": product_id,
            "instrument_id": instrument_id,
            "order_id": order.order_id,
            "status": order.status.value,
        }

    @staticmethod
    def _query_ambiguous_order(venue: Any, instrument_id: str, order: Any) -> Any | None:
        broker = getattr(venue, "broker", None)
        query = getattr(broker, "query_order", None)
        if not callable(query):
            return None
        instrument = getattr(venue, "instruments", {}).get(instrument_id)
        symbol = str(getattr(instrument, "exchange_symbol", ""))
        return query(
            symbol=symbol,
            exchange_order_id=str(order.metadata.get("exchange_order_id") or ""),
            client_order_id=bounded_client_order_id(order),
        )

    def _enqueue_order_recovery(self, *, product_id: str, order: Any, now: str) -> str:
        payload = {
            "product_id": product_id,
            "order_id": order.order_id,
            "client_order_id": bounded_client_order_id(order),
            "recovery_kind": "emergency_ambiguous_submission",
            "reason_code": "emergency_order_recovery",
        }
        job_id = "work:live-recovery:" + canonical_hash(payload).removeprefix("sha256:")
        self.queue.enqueue_if_absent(
            job_id=job_id,
            name="live_order_recovery",
            payload=payload,
            available_at=now,
            priority=250,
            producer_identity="emergency-control",
        )
        return job_id

    def _reduction_targets(
        self,
        *,
        portfolio_id: str,
        payload: Mapping[str, Any],
        product_id: str,
        exchange_state: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, float], ...]:
        instrument_id = str(payload.get("instrument_id") or "").strip()
        raw_quantity = payload.get("position_quantity")
        if instrument_id and raw_quantity is not None:
            quantity = float(raw_quantity)
            if not math.isfinite(quantity):
                raise ValueError("emergency position quantity must be finite")
            return ((instrument_id, quantity),) if abs(quantity) > 1e-12 else ()
        targets = {
            position.instrument_id: position.quantity
            for position in self.positions.all()
            if position.portfolio_id == portfolio_id and abs(position.quantity) > 1e-12
        }
        if exchange_state is not None:
            raw_positions = exchange_state.get("positions")
            if not isinstance(raw_positions, Mapping):
                raise RuntimeError("authenticated emergency positions are invalid")
            venue = self.venues[product_id]
            for raw_instrument_id, raw_quantity in raw_positions.items():
                quantity = float(raw_quantity)
                if not math.isfinite(quantity):
                    raise RuntimeError("authenticated emergency position is not finite")
                if abs(quantity) <= 1e-12:
                    continue
                instrument_id = str(raw_instrument_id)
                if instrument_id not in venue.instruments:
                    instrument_id = next(
                        (
                            identity
                            for identity, item in venue.instruments.items()
                            if item.exchange_symbol == instrument_id
                        ),
                        "",
                    )
                if not instrument_id:
                    raise RuntimeError(
                        f"exchange-only emergency position is not an approved instrument: {raw_instrument_id}"
                    )
                targets[instrument_id] = quantity
        return tuple(sorted(targets.items()))

    def _reduction_quantity(self, product_id: str, instrument_id: str, quantity: float) -> float:
        del product_id
        return abs(quantity)

    def _sync_local_positions(self, exchange_states: Mapping[str, Mapping[str, Any]]) -> None:
        """Adopt the authenticated post-action inventory before declaring flat."""

        for product_id, state in exchange_states.items():
            venue = self.venues.get(product_id)
            product = self.products.get(product_id)
            if venue is None or product is None or not isinstance(state.get("positions"), Mapping):
                continue
            raw_positions = state["positions"]
            for instrument_id, instrument in venue.instruments.items():
                quantity = raw_positions.get(
                    instrument_id, raw_positions.get(instrument.exchange_symbol, 0.0)
                )
                try:
                    quantity = float(quantity)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(quantity):
                    continue
                self.positions.reconcile_position(
                    portfolio_id=str(product.get("portfolio_id") or ""),
                    instrument_id=instrument_id,
                    quantity=quantity,
                    average_entry_price=0.0,
                    updated_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
                )

    def _emit_alert(
        self,
        *,
        event_type: str,
        dedupe_key: str,
        target: str,
        message: str,
        emitted_at: str,
        payload: Mapping[str, Any],
    ) -> None:
        if self.alerts is None:
            return
        try:
            self.alerts.emit(
                event_type=event_type,
                severity=AlertSeverity.CRITICAL,
                dedupe_key=dedupe_key,
                target=target,
                message=message,
                emitted_at=emitted_at,
                payload=payload,
                cooldown_seconds=0,
            )
        except Exception:
            pass


def _reference_price(venue: Any, instrument_id: str) -> float:
    instrument = venue.instruments.get(instrument_id)
    if instrument is None:
        raise ValueError(f"emergency instrument is not mapped: {instrument_id}")
    price = float(venue.broker.get_price(instrument.exchange_symbol))
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"emergency reference price is invalid for {instrument_id}")
    return price


def _serialise_reduction_targets(
    targets: Mapping[str, tuple[tuple[str, float], ...]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(product_id): [
            {"instrument_id": str(instrument_id), "position_quantity": float(quantity)}
            for instrument_id, quantity in values
        ]
        for product_id, values in sorted(targets.items())
    }


def _all_reductions_acknowledged(results: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        result.get("status") not in {OrderStatus.RECOVERY_REQUIRED.value, "recovery_required"}
        for result in results
    )


def _reduction_targets_from_payload(
    raw: Any,
) -> dict[str, tuple[tuple[str, float], ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("emergency reduction targets are invalid")
    result: dict[str, tuple[tuple[str, float], ...]] = {}
    for raw_product_id, raw_targets in raw.items():
        if not isinstance(raw_targets, list | tuple):
            raise ValueError("emergency reduction target list is invalid")
        values: list[tuple[str, float]] = []
        for raw_target in raw_targets:
            if not isinstance(raw_target, Mapping):
                raise ValueError("emergency reduction target is invalid")
            instrument_id = str(raw_target.get("instrument_id") or "").strip()
            raw_quantity = raw_target.get("position_quantity")
            if raw_quantity is None:
                raise ValueError("emergency reduction target quantity is invalid")
            try:
                quantity = float(raw_quantity)
            except (TypeError, ValueError) as exc:
                raise ValueError("emergency reduction target quantity is invalid") from exc
            if not instrument_id or not math.isfinite(quantity):
                raise ValueError("emergency reduction target is invalid")
            values.append((instrument_id, quantity))
        result[str(raw_product_id)] = tuple(values)
    return result


def _exchange_states_from_payload(raw: Any) -> dict[str, Mapping[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("emergency exchange state is invalid")
    result: dict[str, Mapping[str, Any]] = {}
    for product_id, state in raw.items():
        if not isinstance(state, Mapping):
            raise ValueError("emergency exchange state is invalid")
        result[str(product_id)] = dict(state)
    return result


def _order_value(order: Any, name: str) -> Any:
    if isinstance(order, Mapping):
        return order.get(name)
    return getattr(order, name, None)


def _bounded_fraction(value: object) -> float:
    result = float(cast(Any, value))
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("btc_core_fraction must be in [0, 1]")
    return result


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
