"""Idempotent emergency reduction and flatten workers."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from typing import Any, cast

from src.domain._codec import canonical_hash, timestamp
from src.domain.orders import OrderIntent, OrderSide, OrderStatus, OrderType
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
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.order_manager = order_manager
        self.positions = positions
        self.venues = dict(venues)
        self.products = {str(key): dict(value) for key, value in products.items()}
        self.lease_seconds = lease_seconds
        self.alerts = alerts

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("emergency_flatten", "emergency_reduction", "cancel_entry_orders"),
        )
        if claimed is None:
            return {"reason_code": "emergency_queue_empty"}
        try:
            self.order_manager.reload()
            self.positions.reload()
            name = claimed.name
            target = str(claimed.payload.get("target") or "")
            product_ids = self._product_ids(target, claimed.payload)
            cancelled = self._cancel_entries(product_ids, now)
            flattened = []
            if name != "cancel_entry_orders":
                for product_id in product_ids:
                    flattened.extend(self._reduce_product(product_id, claimed.payload, now))
        except Exception as exc:
            self._emit_alert(
                event_type="emergency_action_failed",
                dedupe_key=f"emergency:{claimed.job_id}:failed",
                target=str(claimed.payload.get("target") or "global"),
                message=f"emergency action failed: {type(exc).__name__}",
                emitted_at=now,
                payload={"job_id": claimed.job_id, "error_type": type(exc).__name__},
            )
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "emergency_action_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
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
            },
        )
        return {
            "reason_code": "emergency_action_completed",
            "job_id": claimed.job_id,
            "cancelled_entries": cancelled,
            "flattened": flattened,
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

    def _cancel_entries(self, product_ids: tuple[str, ...], now: str) -> int:
        portfolios = {
            str(self.products[product_id].get("portfolio_id") or "") for product_id in product_ids
        }
        cancelled = 0
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

    def _reduce_product(
        self, product_id: str, payload: Mapping[str, Any], now: str
    ) -> list[dict[str, Any]]:
        product = self.products[product_id]
        portfolio_id = str(product.get("portfolio_id") or "")
        if not portfolio_id:
            raise ValueError(f"product {product_id} has no portfolio_id")
        venue = self.venues.get(product_id)
        if venue is None:
            raise ValueError(f"no emergency venue for {product_id}")
        results: list[dict[str, Any]] = []
        reduction_targets = self._reduction_targets(
            portfolio_id=portfolio_id,
            payload=payload,
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
                results.append(
                    {
                        "product_id": product_id,
                        "instrument_id": instrument_id,
                        "order_id": order_id,
                        "status": existing.status.value,
                    }
                )
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
            venue.submit(intent)
            results.append(
                {
                    "product_id": product_id,
                    "instrument_id": instrument_id,
                    "order_id": order_id,
                    "status": "submitted",
                }
            )
        return results

    def _reduction_targets(
        self, *, portfolio_id: str, payload: Mapping[str, Any]
    ) -> tuple[tuple[str, float], ...]:
        instrument_id = str(payload.get("instrument_id") or "").strip()
        raw_quantity = payload.get("position_quantity")
        if instrument_id and raw_quantity is not None:
            quantity = float(raw_quantity)
            if not math.isfinite(quantity):
                raise ValueError("emergency position quantity must be finite")
            return ((instrument_id, quantity),) if abs(quantity) > 1e-12 else ()
        return tuple(
            (position.instrument_id, position.quantity)
            for position in self.positions.all()
            if position.portfolio_id == portfolio_id and abs(position.quantity) > 1e-12
        )

    def _reduction_quantity(self, product_id: str, instrument_id: str, quantity: float) -> float:
        if product_id != "btc_accumulation":
            return abs(quantity)
        product = self.products[product_id]
        minimum_fraction = product.get("btc_minimum_fraction")
        if minimum_fraction is None:
            core_fraction = _bounded_fraction(product.get("btc_core_fraction", 1.0))
            tactical_fraction = _bounded_fraction(product.get("btc_max_tactical_fraction", 0.0))
            minimum_fraction = max(0.0, core_fraction - tactical_fraction)
        return abs(quantity) * (1.0 - _bounded_fraction(minimum_fraction))

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


def _bounded_fraction(value: object) -> float:
    result = float(cast(Any, value))
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("btc_core_fraction must be in [0, 1]")
    return result


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
