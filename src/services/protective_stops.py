"""Durable native-stop lifecycle for live product execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from src.domain._codec import canonical_hash, timestamp
from src.domain.market_events import MarketEvent
from src.domain.orders import OrderIntent, OrderSide, ProtectionKind
from src.execution.broker import ProtectiveOrder, ProtectiveOrderStatus
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.stops import ProtectiveStop, StopManager, StopStatus
from src.services.alerting import AlertSeverity, SqlAlertService
from src.services.protective_orders import adopt_protective_child
from src.services.scheduler import DatabaseJobQueue


class ProtectiveStopError(RuntimeError):
    """A live position is not safely protected by a confirmed native stop."""


class LiveProtectiveStopService:
    """Coordinate write-ahead stop intents and exchange-native confirmation."""

    def __init__(
        self,
        *,
        stop_manager: StopManager,
        venues: Mapping[str, Any],
        products: Mapping[str, Mapping[str, Any]],
        accounts: Mapping[str, Mapping[str, Any]],
        queue: DatabaseJobQueue | None = None,
        order_manager: OrderManager | None = None,
        positions: PositionManager | None = None,
        alerts: SqlAlertService | None = None,
    ) -> None:
        self.stop_manager = stop_manager
        self.venues = dict(venues)
        self.products = {str(key): dict(value) for key, value in products.items()}
        self.accounts = {str(key): dict(value) for key, value in accounts.items()}
        self.queue = queue
        self.order_manager = order_manager
        self.positions = positions
        self.alerts = alerts

    def prepare_entry(self, product_id: str, order: OrderIntent, at: str) -> ProtectiveStop | None:
        """Persist every declared bracket leg before submitting the entry."""
        if order.reduce_only or not _is_futures(self.products[product_id], self.accounts):
            return None
        self._assert_existing_protection(order)
        supports = getattr(self.venues[product_id].broker, "supports_native_protective_stops", None)
        if not callable(supports) or not supports():
            raise ProtectiveStopError("live futures entry requires native protective stops")
        plans = [
            self._prepare_leg(product_id, order, at, kind, price)
            for kind, price in _protection_prices(order).items()
        ]
        return next(plan for plan in plans if plan.kind is ProtectionKind.STOP)

    def _prepare_leg(self, product_id, order, at, kind, trigger_price):
        _validate_trigger(order.side, trigger_price, _reference_price(order), kind=kind)
        existing = tuple(
            stop for stop in self.stop_manager.for_entry_order(order.order_id) if stop.kind is kind
        )
        if len(existing) > 1:
            raise ProtectiveStopError("entry order has duplicate protection of the same type")
        if existing:
            stop = existing[0]
            if (
                abs(stop.trigger_price - trigger_price) > 1e-12
                or stop.entry_quantity != order.quantity
            ):
                stop = self.stop_manager.reprice_unsubmitted(
                    stop.stop_id, trigger_price, quantity=order.quantity
                )
            if (
                stop.execution_mode != "live"
                or stop.instrument_id != order.instrument_id
                or stop.exit_side is not _exit_side(order.side)
                or abs(stop.trigger_price - trigger_price) > 1e-12
                or stop.entry_quantity != order.quantity
            ):
                raise ProtectiveStopError("durable protective stop does not match entry intent")
            return stop
        identity = order.order_id + ":" + kind.value
        return self.stop_manager.create(
            ProtectiveStop(
                stop_id=_stop_id(product_id, identity),
                portfolio_id=order.portfolio_id,
                instrument_id=order.instrument_id,
                exit_side=_exit_side(order.side),
                quantity=order.quantity,
                trigger_price=trigger_price,
                created_at=timestamp(at, field="stop.created_at"),
                entry_order_id=order.order_id,
                entry_quantity=order.quantity,
                native_client_id=_native_client_id(product_id, identity),
                kind=kind,
            )
        )

    def _assert_existing_protection(self, order: OrderIntent) -> None:
        stops = self._position_stops(order)
        if any(
            stop.status in {StopStatus.CONFIRMATION_FAILED, StopStatus.REPLACE_PENDING}
            or stop.previous_native_order_id
            for stop in stops
        ):
            raise ProtectiveStopError("unresolved position protection blocks new entry risk")
        if self.positions is None:
            return
        self.positions.reload()
        quantity = self.positions.get(order.portfolio_id, order.instrument_id).quantity
        side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
        coverage = sum(
            stop.protected_quantity
            for stop in stops
            if stop.status is StopStatus.PROTECTED
            and stop.exit_side is side
            and stop.kind is ProtectionKind.STOP
        )
        if coverage + 1e-12 < abs(quantity):
            raise ProtectiveStopError("existing position lacks confirmed protective coverage")

    def on_fill(
        self, product_id: str, order: OrderIntent, position_quantity: float, at: str
    ) -> ProtectiveStop | None:
        """Confirm both bracket legs for the actually filled exposure."""
        if order.reduce_only:
            return self._after_reduce(product_id, order, position_quantity, at)
        plans = sorted(
            self.stop_manager.for_entry_order(order.order_id),
            key=lambda plan: plan.kind is not ProtectionKind.STOP,
        )
        expected = set(_protection_prices(order))
        if (
            len(plans) != len(expected)
            or {plan.kind for plan in plans} != expected
            or any(plan.execution_mode != "live" for plan in plans)
        ):
            return self._fail_unprotected(
                product_id,
                order,
                position_quantity,
                at,
                "filled live entry has no unique durable protective stop/bracket",
            )
        if abs(position_quantity) <= 1e-12:
            return None
        primary = None
        for plan in plans:
            updated = self._cover_entry(product_id, order, plan, position_quantity, at)
            if plan.kind is ProtectionKind.STOP:
                primary = updated
        return primary

    def _cover_entry(self, product_id, order, stop, position_quantity, at):
        if stop.status in {StopStatus.TRIGGERED, StopStatus.RECONCILED}:
            self._enqueue_reduction(
                product_id=product_id,
                stop=stop,
                position_quantity=position_quantity,
                at=at,
                reason_code="late_entry_after_protective_trigger",
            )
            return stop
        if stop.exit_side is not _exit_side(order.side):
            return self._fail_unprotected(
                product_id,
                order,
                position_quantity,
                at,
                "protective stop side does not reduce the filled position",
                stop=stop,
            )
        covered = sum(
            other.protected_quantity
            for other in self._position_stops(order)
            if other.stop_id != stop.stop_id
            and other.exit_side is stop.exit_side
            and other.kind is stop.kind
        )
        desired = max(0.0, abs(position_quantity) - covered)
        if stop.kind is ProtectionKind.TAKE_PROFIT:
            desired = min(desired, order.filled_quantity)
        if desired <= 1e-12:
            return self._after_reduce(product_id, order, position_quantity, at)
        return self._ensure_quantity(product_id, stop, desired, position_quantity, at)

    def _ensure_quantity(
        self,
        product_id: str,
        stop: ProtectiveStop,
        desired_quantity: float,
        position_quantity: float,
        at: str,
    ) -> ProtectiveStop:
        if (
            stop.status is StopStatus.PROTECTED
            and abs(stop.protected_quantity - desired_quantity) <= 1e-12
        ):
            return self._retire_previous(product_id, stop, position_quantity, at)
        if abs(stop.quantity - desired_quantity) > 1e-12:
            stop = self.stop_manager.resize(stop.stop_id, quantity=desired_quantity)
        return self._place(product_id, stop, position_quantity, at)

    def _position_stops(self, order: OrderIntent) -> tuple[ProtectiveStop, ...]:
        return tuple(
            stop
            for stop in self.stop_manager.active()
            if stop.execution_mode == "live"
            and stop.portfolio_id == order.portfolio_id
            and stop.instrument_id == order.instrument_id
        )

    def _retire_previous(
        self,
        product_id: str,
        stop: ProtectiveStop,
        position_quantity: float,
        at: str,
    ) -> ProtectiveStop:
        if not stop.previous_native_order_id:
            return stop
        venue = self.venues[product_id]
        try:
            current_quantity = float(
                venue.broker.get_position(_exchange_symbol(venue, stop.instrument_id)).qty
            )
            if (
                not math.isfinite(current_quantity)
                or abs(current_quantity - position_quantity) > 1e-12
            ):
                raise ProtectiveStopError("exchange position changed during protection replacement")
            cancelled = venue.broker.cancel_protective_stop(
                symbol=_exchange_symbol(venue, stop.instrument_id),
                order_id=stop.previous_native_order_id,
                client_id=str(stop.previous_native_client_id or ""),
            )
            if ProtectiveOrderStatus(cancelled.status) is ProtectiveOrderStatus.TRIGGERED:
                outcome = self._record_trigger(product_id, stop, cancelled, at)
                if outcome["reason_code"] == "protective_child_reconciled":
                    return self.stop_manager.replacement_complete(stop.stop_id)
            if ProtectiveOrderStatus(cancelled.status) is not ProtectiveOrderStatus.CANCELED:
                raise ProtectiveStopError("previous stop was not cancelled; reconcile trigger race")
            return self.stop_manager.replacement_complete(stop.stop_id)
        except Exception as exc:
            return self._fail_unprotected(
                product_id,
                None,
                position_quantity,
                at,
                f"previous protection requires reconciliation: {exc}",
                stop=stop,
            )

    def reconcile(self, product_id: str, at: str) -> tuple[dict[str, Any], ...]:
        """Re-read every confirmed native stop and queue unsafe exposure recovery."""

        venue = self.venues[product_id]
        results: list[dict[str, Any]] = []
        for stop in self._reconcilable_stops(product_id):
            position_quantity = self._position_quantity(stop)
            if stop.status is StopStatus.TRIGGERED:
                results.append(self._reconcile_triggered(product_id, stop, at))
                continue
            if abs(position_quantity) <= 1e-12:
                if stop.status is StopStatus.ACTIVE and not stop.native_order_id:
                    results.append({"stop_id": stop.stop_id, "status": "awaiting_entry_fill"})
                    continue
                self._cancel_flat_stops(product_id, (stop,), at)
                status = self.stop_manager.get(stop.stop_id).status
                results.append(
                    {
                        "stop_id": stop.stop_id,
                        "status": "cancelled_flat"
                        if status is StopStatus.CANCELLED
                        else "orphan_cancellation_failed",
                    }
                )
                continue
            if stop.submission_attempted and not stop.native_order_id:
                try:
                    recovered = self._place(product_id, stop, position_quantity, at)
                    results.append({"stop_id": stop.stop_id, "status": recovered.status.value})
                except ProtectiveStopError:
                    results.append({"stop_id": stop.stop_id, "status": "recovery_required"})
                continue
            if not stop.native_order_id or not stop.native_client_id:
                self.stop_manager.mark_failure(
                    stop.stop_id, reason="live position has no confirmed native protective stop"
                )
                self._enqueue_reduction(
                    product_id=product_id,
                    stop=stop,
                    position_quantity=position_quantity,
                    at=at,
                    reason_code="protective_stop_missing_after_reconciliation",
                )
                results.append({"stop_id": stop.stop_id, "status": "recovery_required"})
                continue
            try:
                native = venue.broker.get_protective_stop(
                    symbol=_exchange_symbol(venue, stop.instrument_id),
                    order_id=stop.native_order_id,
                    client_id=stop.native_client_id,
                )
                _validate_native(
                    native,
                    stop,
                    symbol=_exchange_symbol(venue, stop.instrument_id),
                )
                results.append(self._apply_native_status(product_id, stop, native, at))
            except Exception as exc:
                self.stop_manager.mark_failure(
                    stop.stop_id, reason=f"stop reconciliation failed: {exc}"
                )
                self._enqueue_reduction(
                    product_id=product_id,
                    stop=stop,
                    position_quantity=self._position_quantity(stop),
                    at=at,
                    reason_code="protective_stop_reconciliation_failed",
                )
                results.append({"stop_id": stop.stop_id, "status": "reconciliation_failed"})
        return tuple(results)

    def _reconcilable_stops(self, product_id: str) -> tuple[ProtectiveStop, ...]:
        return tuple(
            stop
            for stop in self.stop_manager.all()
            if stop.execution_mode == "live"
            and stop.portfolio_id == _portfolio_id(self.products[product_id])
            and stop.status not in {StopStatus.CANCELLED, StopStatus.RECONCILED}
        )

    def _reconcile_triggered(
        self, product_id: str, stop: ProtectiveStop, at: str
    ) -> dict[str, Any]:
        venue = self.venues[product_id]
        native = venue.broker.get_protective_stop(
            symbol=_exchange_symbol(venue, stop.instrument_id),
            order_id=stop.native_order_id,
            client_id=str(stop.native_client_id or ""),
        )
        _validate_native(native, stop, symbol=_exchange_symbol(venue, stop.instrument_id))
        return self._record_trigger(product_id, stop, native, at)

    def on_algo_update(self, product_id: str, event: MarketEvent) -> dict[str, Any]:
        """Apply a Binance conditional-order update to its durable stop."""

        values = _algo_values(event)
        native_order_id = _first_text(values, "i", "orderId", "aid", "algoId")
        native_client_id = _first_text(values, "c", "C", "clientOrderId", "clientAlgoId", "caid")
        matches = tuple(
            stop
            for stop in self.stop_manager.all()
            if stop.execution_mode == "live"
            and stop.portfolio_id == _portfolio_id(self.products[product_id])
            and stop.instrument_id == event.instrument_id
            and _matches_native(stop, native_order_id, native_client_id)
        )
        if len(matches) != 1:
            recovery_payload = {
                "product_id": product_id,
                "reason_code": "unknown_protective_algo_update",
                "event_id": event.event_id,
                "native_order_id": native_order_id,
                "native_client_id": native_client_id,
            }
            self._enqueue_recovery(recovery_payload, event.receive_timestamp)
            return {"reason_code": recovery_payload["reason_code"]}
        stop = matches[0]
        status = _protective_status(values)
        if status is ProtectiveOrderStatus.TRIGGERED:
            venue = self.venues[product_id]
            native = venue.broker.get_protective_stop(
                symbol=_exchange_symbol(venue, stop.instrument_id),
                order_id=native_order_id or None,
                client_id=native_client_id,
            )
            plan = self.stop_manager.native_plan(native.order_id) or stop
            _validate_native(native, plan, symbol=_exchange_symbol(venue, stop.instrument_id))
            return self._record_trigger(product_id, stop, native, event.receive_timestamp)
        if status in {
            ProtectiveOrderStatus.CANCELED,
            ProtectiveOrderStatus.EXPIRED,
            ProtectiveOrderStatus.REJECTED,
        }:
            self.stop_manager.mark_failure(
                stop.stop_id, reason=f"native protective stop {status.value}"
            )
            self._enqueue_reduction(
                product_id=product_id,
                stop=stop,
                position_quantity=self._position_quantity(stop),
                at=event.receive_timestamp,
                reason_code=f"protective_stop_{status.value}",
            )
            return {"reason_code": f"protective_stop_{status.value}", "stop_id": stop.stop_id}
        return {"reason_code": "protective_stop_open", "stop_id": stop.stop_id}

    def _record_trigger(
        self,
        product_id: str,
        stop: ProtectiveStop,
        native: ProtectiveOrder,
        at: str,
    ) -> dict[str, Any]:
        child = None
        pending_entry = self._pending_entry(stop)
        if native.child_order_id:
            if self.order_manager is None:
                raise ProtectiveStopError("triggered child has no durable order manager")
            child = adopt_protective_child(
                self.order_manager,
                stop,
                native,
                self.venues[product_id].broker,
                at,
            )
        previous = native.order_id == stop.previous_native_order_id
        if previous and (child is None or not child.is_terminal):
            self.stop_manager.mark_failure(
                stop.stop_id, reason="previous stop triggered during replacement"
            )
        elif not previous and stop.status is not StopStatus.TRIGGERED:
            self.stop_manager.mark_protected(
                stop.stop_id,
                native_order_id=native.order_id,
                native_client_id=native.client_id,
                protected_quantity=stop.quantity,
            )
            self.stop_manager.triggered(stop.stop_id, triggered_at=at)
        if child is not None and child.is_terminal:
            self._complete_child(product_id, stop, previous, at)
            if pending_entry is None:
                return {
                    "reason_code": "protective_child_reconciled",
                    "stop_id": stop.stop_id,
                    "child_order_id": child.order_id,
                }
        self._enqueue_recovery(
            {
                "product_id": product_id,
                "stop_id": stop.stop_id,
                "reason_code": "protective_stop_triggered",
                "recovery_kind": "protective_child",
                "protective_child_order_id": child.order_id if child else None,
                "protective_entry_order_id": pending_entry,
                "observed_at": at,
            },
            at,
        )
        return {
            "reason_code": "protective_stop_triggered",
            "stop_id": stop.stop_id,
            "child_order_id": child.order_id if child else None,
        }

    def _pending_entry(self, stop):
        if self.order_manager is None or not stop.entry_order_id:
            return None
        self.order_manager.reload()
        order = self.order_manager.get(stop.entry_order_id)
        return order.order_id if not order.is_terminal else None

    def _complete_child(
        self, product_id: str, stop: ProtectiveStop, previous: bool, at: str
    ) -> None:
        if previous:
            return
        self.stop_manager.reconciled(stop.stop_id)
        remaining = self._position_quantity(stop)
        if self.order_manager is not None and stop.entry_order_id:
            self._after_reduce(
                product_id, self.order_manager.get(stop.entry_order_id), remaining, at
            )
        side = OrderSide.SELL if remaining > 0 else OrderSide.BUY
        covered = sum(
            item.protected_quantity
            for item in self.stop_manager.active()
            if item.execution_mode == "live"
            and item.portfolio_id == stop.portfolio_id
            and item.instrument_id == stop.instrument_id
            and item.exit_side is side
            and item.status is StopStatus.PROTECTED
            and item.kind is ProtectionKind.STOP
        )
        uncovered = max(0.0, abs(remaining) - covered)
        self._enqueue_reduction(
            product_id=product_id,
            stop=stop,
            position_quantity=math.copysign(uncovered, remaining),
            at=at,
            reason_code="protective_child_terminal_residual",
        )

    def _place(
        self, product_id: str, stop: ProtectiveStop, position_quantity: float, at: str
    ) -> ProtectiveStop:
        venue = self.venues[product_id]
        try:
            native = self._submit_or_recover(venue, stop)
            _validate_native(
                native,
                stop,
                symbol=_exchange_symbol(venue, stop.instrument_id),
            )
            if ProtectiveOrderStatus(native.status) is ProtectiveOrderStatus.TRIGGERED:
                self._record_trigger(product_id, stop, native, at)
                raise ProtectiveStopError("protective stop triggered during placement")
            updated = self.stop_manager.mark_protected(
                stop.stop_id,
                native_order_id=native.order_id,
                native_client_id=native.client_id,
                protected_quantity=stop.quantity,
            )
            return self._retire_previous(product_id, updated, position_quantity, at)
        except ProtectiveStopError as exc:
            if self.stop_manager.get(stop.stop_id).status is not StopStatus.TRIGGERED:
                self._fail_unprotected(
                    product_id,
                    None,
                    position_quantity,
                    at,
                    str(exc),
                    stop=stop,
                )
            raise
        except Exception as exc:
            return self._fail_unprotected(
                product_id,
                None,
                position_quantity,
                at,
                f"native protective stop placement failed: {exc}",
                stop=stop,
            )

    def _submit_or_recover(self, venue: Any, stop: ProtectiveStop) -> ProtectiveOrder:
        symbol = _exchange_symbol(venue, stop.instrument_id)
        if stop.submission_attempted or stop.native_order_id:
            return venue.broker.get_protective_stop(
                symbol=symbol,
                order_id=stop.native_order_id,
                client_id=str(stop.native_client_id or ""),
            )
        self.stop_manager.submission_started(stop.stop_id)
        return venue.broker.place_protective_stop(
            symbol=symbol,
            side=stop.exit_side,
            qty=stop.quantity,
            trigger_price=stop.trigger_price,
            client_id=str(stop.native_client_id or ""),
            kind=stop.kind,
        )

    def _after_reduce(
        self, product_id: str, order: OrderIntent, position_quantity: float, at: str
    ) -> ProtectiveStop | None:
        stops = self._position_stops(order)
        if not stops:
            return None
        if abs(position_quantity) <= 1e-12:
            self._cancel_flat_stops(product_id, stops, at)
            return None
        primary = None
        for kind in ProtectionKind:
            matching = tuple(stop for stop in stops if stop.kind is kind)
            result = self._reduce_protection_kind(
                product_id, order, position_quantity, at, matching, kind
            )
            if kind is ProtectionKind.STOP:
                primary = result
        return primary

    def _reduce_protection_kind(self, product_id, order, position_quantity, at, stops, kind):
        remaining = abs(position_quantity)
        result = None
        for stop in stops:
            if stop.status is StopStatus.ACTIVE and not stop.native_order_id:
                continue
            desired = min(stop.quantity, remaining)
            if desired <= 1e-12:
                try:
                    self._cancel_stop(product_id, stop)
                except Exception as exc:
                    return self._fail_unprotected(
                        product_id,
                        order,
                        position_quantity,
                        at,
                        f"redundant protection cancellation failed: {exc}",
                        stop=stop,
                    )
                continue
            result = self._ensure_quantity(product_id, stop, desired, position_quantity, at)
            remaining -= desired
        if remaining > 1e-12 and kind is ProtectionKind.STOP:
            return self._fail_unprotected(
                product_id,
                order,
                position_quantity,
                at,
                "position exceeds confirmed stop coverage",
                stop=stops[0],
            )
        return result

    def _cancel_flat_stops(
        self, product_id: str, stops: tuple[ProtectiveStop, ...], at: str
    ) -> None:
        for stop in stops:
            try:
                self._cancel_stop(product_id, stop)
            except Exception as exc:
                self.stop_manager.mark_failure(
                    stop.stop_id, reason=f"flat stop cancellation failed: {exc}"
                )
                self._enqueue_recovery(
                    {
                        "product_id": product_id,
                        "stop_id": stop.stop_id,
                        "reason_code": "protective_stop_cancel_failed",
                    },
                    at,
                )
                self._emit_alert(
                    event_type="protective_stop_orphaned",
                    dedupe_key=f"protective-stop:{product_id}:{stop.stop_id}:flat",
                    target=product_id,
                    message="flat position has unresolved native protection",
                    emitted_at=at,
                    payload={"stop_id": stop.stop_id},
                )

    def _cancel_stop(self, product_id: str, stop: ProtectiveStop) -> None:
        venue = self.venues[product_id]
        native_order_id = stop.native_order_id
        if stop.submission_attempted and not native_order_id:
            native = venue.broker.get_protective_stop(
                symbol=_exchange_symbol(venue, stop.instrument_id),
                order_id=None,
                client_id=str(stop.native_client_id or ""),
            )
            native_order_id = native.order_id
        for order_id, client_id in (
            (native_order_id, stop.native_client_id),
            (stop.previous_native_order_id, stop.previous_native_client_id),
        ):
            if order_id:
                native = venue.broker.cancel_protective_stop(
                    symbol=_exchange_symbol(venue, stop.instrument_id),
                    order_id=order_id,
                    client_id=str(client_id or ""),
                )
                if ProtectiveOrderStatus(native.status) is not ProtectiveOrderStatus.CANCELED:
                    raise ProtectiveStopError("stop cancellation requires trigger reconciliation")
        self.stop_manager.cancel(stop.stop_id)

    def on_order_status(
        self, product_id: str, order: OrderIntent, status: str, at: str
    ) -> ProtectiveStop | None:
        """Clean up or resize protection when an entry ends without a fill."""

        if order.reduce_only or status not in {
            "cancelled",
            "rejected",
            "expired",
            "reconciled",
        }:
            return None
        stops = self.stop_manager.for_entry_order(order.order_id)
        if not stops:
            return None
        stop = stops[0]
        position_quantity = self._position_quantity(stop)
        if abs(position_quantity) > 1e-12:
            return self.on_fill(product_id, order, position_quantity, at)
        self._cancel_flat_stops(product_id, stops, at)
        return self.stop_manager.get(stop.stop_id)

    def _fail_unprotected(
        self,
        product_id: str,
        order: OrderIntent | None,
        position_quantity: float,
        at: str,
        reason: str,
        *,
        stop: ProtectiveStop | None = None,
    ) -> ProtectiveStop:
        if stop is None:
            if order is None:
                raise ProtectiveStopError(reason)
            matches = self.stop_manager.for_entry_order(order.order_id)
            if not matches:
                self._enqueue_recovery(
                    {
                        "product_id": product_id,
                        "order_id": order.order_id,
                        "reason_code": "filled_entry_missing_protection",
                    },
                    at,
                )
                raise ProtectiveStopError(reason)
            stop = matches[0]
        self.stop_manager.mark_failure(stop.stop_id, reason=reason)
        self._emit_alert(
            event_type="protective_stop_failed",
            dedupe_key=f"protective-stop:{product_id}:{stop.stop_id}:{reason}",
            target=product_id,
            message=reason,
            emitted_at=at,
            payload={
                "stop_id": stop.stop_id,
                "instrument_id": stop.instrument_id,
                "position_quantity": position_quantity,
            },
        )
        self._enqueue_reduction(
            product_id=product_id,
            stop=stop,
            position_quantity=position_quantity,
            at=at,
            reason_code="protective_stop_missing_or_failed",
        )
        raise ProtectiveStopError(reason)

    def _apply_native_status(
        self, product_id: str, stop: ProtectiveStop, native: ProtectiveOrder, at: str
    ) -> dict[str, Any]:
        status = ProtectiveOrderStatus(native.status)
        if status is ProtectiveOrderStatus.OPEN:
            if stop.status is not StopStatus.PROTECTED:
                stop = self.stop_manager.mark_protected(
                    stop.stop_id,
                    native_order_id=native.order_id,
                    native_client_id=native.client_id,
                    protected_quantity=native.qty,
                )
            self._retire_previous(product_id, stop, self._position_quantity(stop), at)
            return {"stop_id": stop.stop_id, "status": "open"}
        if status is ProtectiveOrderStatus.TRIGGERED:
            self._record_trigger(product_id, stop, native, at)
            return {"stop_id": stop.stop_id, "status": "triggered"}
        self.stop_manager.mark_failure(
            stop.stop_id, reason=f"native protective stop {status.value}"
        )
        self._enqueue_reduction(
            product_id=product_id,
            stop=stop,
            position_quantity=self._position_quantity(stop),
            at=at,
            reason_code=f"protective_stop_{status.value}",
        )
        return {"stop_id": stop.stop_id, "status": status.value}

    def _position_quantity(self, stop: ProtectiveStop) -> float:
        if self.positions is None:
            return stop.protected_quantity
        self.positions.reload()
        return self.positions.get(stop.portfolio_id, stop.instrument_id).quantity

    def _enqueue_reduction(
        self,
        *,
        product_id: str,
        stop: ProtectiveStop,
        position_quantity: float,
        at: str,
        reason_code: str,
    ) -> None:
        if self.queue is None or abs(position_quantity) <= 1e-12:
            return
        unsigned = {
            "product_id": product_id,
            "portfolio_id": stop.portfolio_id,
            "instrument_id": stop.instrument_id,
            "stop_id": stop.stop_id,
            "position_quantity": position_quantity,
            "reason_code": reason_code,
            "evaluated_at": timestamp(at, field="emergency_reduction.evaluated_at"),
            "producer_identity": "protective-stop-service",
        }
        payload = {**unsigned, "content_hash": canonical_hash(unsigned)}
        job_id = "emergency-reduction:" + canonical_hash(unsigned).removeprefix("sha256:")
        self.queue.enqueue_if_absent(
            job_id=job_id,
            name="emergency_reduction",
            payload=payload,
            available_at=payload["evaluated_at"],
            priority=200,
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

    def _enqueue_recovery(self, payload: Mapping[str, Any], at: str) -> None:
        if self.queue is None:
            return
        identity = canonical_hash(dict(payload)).removeprefix("sha256:")
        self.queue.enqueue_if_absent(
            job_id="live-recovery:" + identity,
            name="live_order_recovery",
            payload={**dict(payload), "available_at": at},
            available_at=at,
            priority=200,
        )


def _is_futures(product: Mapping[str, Any], accounts: Mapping[str, Mapping[str, Any]]) -> bool:
    account = accounts.get(str(product.get("account_id") or ""), {})
    return str(account.get("market") or "") != "spot"


def _portfolio_id(product: Mapping[str, Any]) -> str:
    portfolio_id = str(product.get("portfolio_id") or "")
    if not portfolio_id:
        raise ProtectiveStopError("live product has no portfolio_id")
    return portfolio_id


def _exchange_symbol(venue: Any, instrument_id: str) -> str:
    instrument = venue.instruments.get(instrument_id)
    if instrument is None:
        raise ProtectiveStopError(f"live instrument is not mapped by its venue: {instrument_id}")
    return str(instrument.exchange_symbol)


def _stop_id(product_id: str, entry_order_id: str) -> str:
    return (
        "stop:"
        + canonical_hash({"product_id": product_id, "entry_order_id": entry_order_id}).removeprefix(
            "sha256:"
        )[:32]
    )


def _native_client_id(product_id: str, entry_order_id: str) -> str:
    return (
        "s"
        + canonical_hash({"product_id": product_id, "entry_order_id": entry_order_id}).removeprefix(
            "sha256:"
        )[:35]
    )


def _reference_price(order: OrderIntent) -> float:
    raw = order.metadata.get("entry_reference", {}).get(
        "price", order.metadata.get("reference_price")
    )
    if raw is None and isinstance(order.metadata.get("target_metadata"), Mapping):
        raw = order.metadata["target_metadata"].get("reference_price")
    if raw is None:
        raw = order.limit_price
    try:
        value = float(cast(Any, raw))
    except (TypeError, ValueError) as exc:
        raise ProtectiveStopError("live entry has no positive reference price") from exc
    if not math.isfinite(value) or value <= 0:
        raise ProtectiveStopError("live entry has no positive reference price")
    return value


def _trigger_price(order: OrderIntent) -> float:
    target_metadata = order.metadata.get("target_metadata")
    raw = order.metadata.get("protective_stop_price")
    if raw is None and isinstance(target_metadata, Mapping):
        raw = target_metadata.get("protective_stop_price")
    try:
        value = float(cast(Any, raw))
    except (TypeError, ValueError) as exc:
        raise ProtectiveStopError("live futures entry has no protective_stop_price") from exc
    if not math.isfinite(value) or value <= 0:
        raise ProtectiveStopError("live futures entry has no positive protective_stop_price")
    return value


def _protection_prices(order: OrderIntent) -> dict[ProtectionKind, float]:
    metadata = order.metadata.get("target_metadata", order.metadata)
    rules = metadata.get("position_rules") or {}
    if rules.get("execution_policy"):
        from src.strategies.position_state import PositionRules

        declared = PositionRules(**rules)
        sign = 1 if order.side is OrderSide.BUY else -1
        stop = _reference_price(order) * (1 - sign * declared.stop_loss)
    else:
        stop = _trigger_price(order)
    prices = {ProtectionKind.STOP: stop}
    profit = rules.get("take_profit")
    if profit is not None:
        if not math.isfinite(float(profit)) or not 0 < float(profit) < 1:
            raise ProtectiveStopError("declared take-profit fraction is invalid")
        sign = 1 if order.side is OrderSide.BUY else -1
        prices[ProtectionKind.TAKE_PROFIT] = _reference_price(order) * (1 + sign * float(profit))
    return prices


def _validate_trigger(
    side: OrderSide,
    trigger_price: float,
    reference_price: float,
    *,
    kind: ProtectionKind = ProtectionKind.STOP,
) -> None:
    valid = (
        trigger_price < reference_price
        if side is OrderSide.BUY
        else trigger_price > reference_price
    )
    if kind is ProtectionKind.TAKE_PROFIT:
        valid = (
            trigger_price > reference_price
            if side is OrderSide.BUY
            else trigger_price < reference_price
        )
    if not valid:
        raise ProtectiveStopError(
            "protective stop/target trigger is on the wrong side of the entry"
        )


def _exit_side(side: OrderSide) -> OrderSide:
    return OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY


def _validate_native(native: ProtectiveOrder, stop: ProtectiveStop, *, symbol: str) -> None:
    status = ProtectiveOrderStatus(native.status)
    if native.kind is not stop.kind or native.working_type != "CONTRACT_PRICE":
        raise ProtectiveStopError("native protection type or reference does not match its policy")
    if status not in {ProtectiveOrderStatus.OPEN, ProtectiveOrderStatus.TRIGGERED}:
        raise ProtectiveStopError(f"native protective stop returned unsafe status {status.value}")
    if native.order_id.strip() == "" or native.client_id != stop.native_client_id:
        raise ProtectiveStopError("native protective stop identity does not match the durable stop")
    if native.symbol != symbol:
        raise ProtectiveStopError("native protective stop symbol does not match the durable stop")
    if native.side.value != stop.exit_side.value or abs(native.qty - stop.quantity) > 1e-12:
        raise ProtectiveStopError("native protective stop side or quantity does not match")
    if abs(native.trigger_price - stop.trigger_price) > 1e-12:
        raise ProtectiveStopError("native protective stop trigger does not match")


def _algo_values(event: MarketEvent) -> Mapping[str, Any]:
    raw = event.payload.get("data")
    if not isinstance(raw, Mapping):
        return {}
    nested = raw.get("o")
    if isinstance(nested, Mapping):
        return {**dict(raw), **dict(nested)}
    return raw


def _first_text(values: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if value is not None and not isinstance(value, bool) and str(value).strip():
            return str(value).strip()
    return ""


def _protective_status(values: Mapping[str, Any]) -> ProtectiveOrderStatus:
    raw = _first_text(values, "algoStatus", "status", "X", "x").lower()
    if raw in {"new", "open", "accepted", "pending"}:
        return ProtectiveOrderStatus.OPEN
    if raw in {"triggering", "triggered", "finished", "filled", "executed", "closed"}:
        return ProtectiveOrderStatus.TRIGGERED
    if raw in {"canceled", "cancelled"}:
        return ProtectiveOrderStatus.CANCELED
    if raw == "expired":
        return ProtectiveOrderStatus.EXPIRED
    if raw == "rejected":
        return ProtectiveOrderStatus.REJECTED
    raise ProtectiveStopError(f"unsupported protective algo status: {raw or '<missing>'}")


def _matches_native(stop: ProtectiveStop, order_id: str, client_id: str) -> bool:
    identities = (
        (stop.native_order_id, stop.native_client_id),
        (stop.previous_native_order_id, stop.previous_native_client_id),
    )
    return any(
        (
            not order_id
            or native_id == order_id
            or (native_id is None and client_id == native_client)
        )
        and (not client_id or native_client == client_id)
        and bool(order_id or client_id)
        for native_id, native_client in identities
    )
