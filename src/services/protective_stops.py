"""Durable native-stop lifecycle for live product execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from src.domain._codec import canonical_hash, timestamp
from src.domain.market_events import MarketEvent
from src.domain.orders import OrderIntent, OrderSide
from src.execution.broker import ProtectiveOrder, ProtectiveOrderStatus
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.stops import ProtectiveStop, StopManager, StopStatus
from src.services.alerting import AlertSeverity, SqlAlertService
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
        """Persist the stop intent before the entry can reach the exchange."""

        if order.reduce_only:
            return None
        product = self.products[product_id]
        venue = self.venues[product_id]
        broker = venue.broker
        if not _is_futures(product, self.accounts):
            return None
        supports_stops = getattr(broker, "supports_native_protective_stops", None)
        if not callable(supports_stops) or not supports_stops():
            raise ProtectiveStopError("live futures entry requires native protective stops")
        trigger_price = _trigger_price(order)
        reference_price = _reference_price(order)
        _validate_trigger(order.side, trigger_price, reference_price)
        existing = self.stop_manager.for_entry_order(order.order_id)
        if len(existing) > 1:
            raise ProtectiveStopError("entry order has multiple durable protective stops")
        client_id = _native_client_id(product_id, order.order_id)
        if existing:
            stop = existing[0]
            if (
                stop.instrument_id != order.instrument_id
                or stop.exit_side is not _exit_side(order.side)
                or abs(stop.trigger_price - trigger_price) > 1e-12
                or abs(stop.quantity - order.quantity) > 1e-12
            ):
                raise ProtectiveStopError("durable protective stop does not match entry intent")
            return stop
        stop = ProtectiveStop(
            stop_id=_stop_id(product_id, order.order_id),
            portfolio_id=order.portfolio_id,
            instrument_id=order.instrument_id,
            exit_side=_exit_side(order.side),
            quantity=order.quantity,
            trigger_price=trigger_price,
            created_at=timestamp(at, field="stop.created_at"),
            entry_order_id=order.order_id,
            native_client_id=client_id,
        )
        return self.stop_manager.create(stop)

    def on_fill(
        self,
        product_id: str,
        order: OrderIntent,
        position_quantity: float,
        at: str,
    ) -> ProtectiveStop | None:
        """Confirm, resize, or cancel protection after an authoritative fill."""

        if order.reduce_only:
            return self._after_reduce(product_id, order, position_quantity, at)
        stops = self.stop_manager.for_entry_order(order.order_id)
        if len(stops) != 1:
            return self._fail_unprotected(
                product_id,
                order,
                position_quantity,
                at,
                "filled live entry has no unique durable protective stop",
            )
        if abs(position_quantity) <= 1e-12:
            return None
        stop = stops[0]
        if stop.exit_side is not _exit_side(order.side):
            return self._fail_unprotected(
                product_id,
                order,
                position_quantity,
                at,
                "protective stop side does not reduce the filled position",
                stop=stop,
            )
        desired_quantity = abs(position_quantity)
        venue = self.venues[product_id]
        broker = venue.broker
        if (
            stop.status is StopStatus.PROTECTED
            and abs(stop.protected_quantity - desired_quantity) <= 1e-12
        ):
            return stop
        if stop.native_order_id:
            try:
                cancelled = broker.cancel_protective_stop(
                    symbol=_exchange_symbol(venue, stop.instrument_id),
                    order_id=stop.native_order_id,
                    client_id=str(stop.native_client_id or ""),
                )
                if ProtectiveOrderStatus(cancelled.status) is ProtectiveOrderStatus.OPEN:
                    raise ProtectiveStopError("native protective stop remained open after resize")
            except Exception as exc:
                return self._fail_unprotected(
                    product_id,
                    order,
                    position_quantity,
                    at,
                    f"could not cancel protective stop for resize: {exc}",
                    stop=stop,
                )
        if abs(stop.quantity - desired_quantity) > 1e-12 or stop.native_order_id:
            stop = self.stop_manager.resize(stop.stop_id, quantity=desired_quantity)
        return self._place(product_id, stop, position_quantity, at)

    def reconcile(self, product_id: str, at: str) -> tuple[dict[str, Any], ...]:
        """Re-read every confirmed native stop and queue unsafe exposure recovery."""

        venue = self.venues[product_id]
        results: list[dict[str, Any]] = []
        for stop in self.stop_manager.active():
            if stop.portfolio_id != _portfolio_id(self.products[product_id]):
                continue
            if not stop.native_order_id or not stop.native_client_id:
                results.append({"stop_id": stop.stop_id, "status": stop.status.value})
                continue
            try:
                native = venue.broker.get_protective_stop(
                    symbol=_exchange_symbol(venue, stop.instrument_id),
                    order_id=stop.native_order_id,
                    client_id=stop.native_client_id,
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

    def on_algo_update(self, product_id: str, event: MarketEvent) -> dict[str, Any]:
        """Apply a Binance conditional-order update to its durable stop."""

        values = _algo_values(event)
        native_order_id = _first_text(values, "i", "orderId", "aid", "algoId")
        native_client_id = _first_text(values, "c", "C", "clientOrderId", "clientAlgoId", "caid")
        matches = tuple(
            stop
            for stop in self.stop_manager.active()
            if (native_order_id and stop.native_order_id == native_order_id)
            or (native_client_id and stop.native_client_id == native_client_id)
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
            if stop.status is not StopStatus.TRIGGERED:
                stop = self.stop_manager.triggered(
                    stop.stop_id, triggered_at=event.exchange_timestamp
                )
            return {"reason_code": "protective_stop_triggered", "stop_id": stop.stop_id}
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

    def _place(
        self, product_id: str, stop: ProtectiveStop, position_quantity: float, at: str
    ) -> ProtectiveStop:
        venue = self.venues[product_id]
        try:
            native = venue.broker.place_protective_stop(
                symbol=_exchange_symbol(venue, stop.instrument_id),
                side=stop.exit_side,
                qty=stop.quantity,
                trigger_price=stop.trigger_price,
                client_id=str(stop.native_client_id or _native_client_id(product_id, stop.stop_id)),
            )
            _validate_native(
                native,
                stop,
                symbol=_exchange_symbol(venue, stop.instrument_id),
            )
            if ProtectiveOrderStatus(native.status) is ProtectiveOrderStatus.TRIGGERED:
                updated = self.stop_manager.triggered(stop.stop_id, triggered_at=at)
                self._enqueue_reduction(
                    product_id=product_id,
                    stop=updated,
                    position_quantity=position_quantity,
                    at=at,
                    reason_code="protective_stop_triggered_on_placement",
                )
                raise ProtectiveStopError("protective stop triggered during placement")
            return self.stop_manager.mark_protected(
                stop.stop_id,
                native_order_id=native.order_id,
                native_client_id=native.client_id,
                protected_quantity=stop.quantity,
            )
        except ProtectiveStopError as exc:
            if stop.status is not StopStatus.TRIGGERED:
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

    def _after_reduce(
        self, product_id: str, order: OrderIntent, position_quantity: float, at: str
    ) -> ProtectiveStop | None:
        stops = tuple(
            stop
            for stop in self.stop_manager.active()
            if stop.portfolio_id == order.portfolio_id and stop.instrument_id == order.instrument_id
        )
        if not stops:
            return None
        if abs(position_quantity) <= 1e-12:
            for stop in stops:
                if stop.native_order_id:
                    try:
                        native = self.venues[product_id].broker.cancel_protective_stop(
                            symbol=_exchange_symbol(self.venues[product_id], stop.instrument_id),
                            order_id=stop.native_order_id,
                            client_id=str(stop.native_client_id or ""),
                        )
                        if ProtectiveOrderStatus(native.status) is ProtectiveOrderStatus.OPEN:
                            raise ProtectiveStopError(
                                "protective stop remained open after flat close"
                            )
                    except Exception as exc:
                        self.stop_manager.mark_failure(
                            stop.stop_id, reason=f"stop cancellation after flat close failed: {exc}"
                        )
                        self._enqueue_reduction(
                            product_id=product_id,
                            stop=stop,
                            position_quantity=0.0,
                            at=at,
                            reason_code="protective_stop_cancel_failed",
                        )
                        continue
                self.stop_manager.cancel(stop.stop_id)
            return None
        for stop in stops:
            if (
                stop.status is StopStatus.PROTECTED
                and abs(stop.protected_quantity - abs(position_quantity)) > 1e-12
            ):
                self.stop_manager.resize(stop.stop_id, quantity=abs(position_quantity))
                return self._place(
                    product_id, self.stop_manager.get(stop.stop_id), position_quantity, at
                )
        return stops[0]

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
            stop = self.stop_manager.for_entry_order(order.order_id)[0]
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
                self.stop_manager.mark_protected(
                    stop.stop_id,
                    native_order_id=native.order_id,
                    native_client_id=native.client_id,
                    protected_quantity=native.qty,
                )
            return {"stop_id": stop.stop_id, "status": "open"}
        if status is ProtectiveOrderStatus.TRIGGERED:
            if stop.status is not StopStatus.TRIGGERED:
                self.stop_manager.triggered(stop.stop_id, triggered_at=at)
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
    raw = order.metadata.get("reference_price")
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


def _validate_trigger(side: OrderSide, trigger_price: float, reference_price: float) -> None:
    valid = (
        trigger_price < reference_price
        if side is OrderSide.BUY
        else trigger_price > reference_price
    )
    if not valid:
        raise ProtectiveStopError("protective stop trigger is on the wrong side of the entry")


def _exit_side(side: OrderSide) -> OrderSide:
    return OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY


def _validate_native(native: ProtectiveOrder, stop: ProtectiveStop, *, symbol: str) -> None:
    status = ProtectiveOrderStatus(native.status)
    if status not in {ProtectiveOrderStatus.OPEN, ProtectiveOrderStatus.TRIGGERED}:
        raise ProtectiveStopError(f"native protective stop returned unsafe status {status.value}")
    if native.order_id.strip() == "" or native.client_id != stop.native_client_id:
        raise ProtectiveStopError("native protective stop identity does not match the durable stop")
    if native.symbol != symbol:
        raise ProtectiveStopError("native protective stop symbol does not match the durable stop")
    if native.side is not stop.exit_side or abs(native.qty - stop.quantity) > 1e-12:
        raise ProtectiveStopError("native protective stop side or quantity does not match")
    if abs(native.trigger_price - stop.trigger_price) > 1e-12:
        raise ProtectiveStopError("native protective stop trigger does not match")


def _algo_values(event: MarketEvent) -> Mapping[str, Any]:
    raw = event.payload.get("data")
    if not isinstance(raw, Mapping):
        return {}
    nested = raw.get("a")
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
    if raw in {"new", "open", "accepted", "triggering", "pending"}:
        return ProtectiveOrderStatus.OPEN
    if raw in {"triggered", "filled", "executed", "closed"}:
        return ProtectiveOrderStatus.TRIGGERED
    if raw in {"canceled", "cancelled"}:
        return ProtectiveOrderStatus.CANCELED
    if raw == "expired":
        return ProtectiveOrderStatus.EXPIRED
    if raw == "rejected":
        return ProtectiveOrderStatus.REJECTED
    raise ProtectiveStopError(f"unsupported protective algo status: {raw or '<missing>'}")
