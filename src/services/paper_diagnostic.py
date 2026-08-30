"""Run the bootstrap diagnostic paper round trip through durable queues."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

from src.domain._codec import canonical_hash, timestamp
from src.domain.orders import OrderIntent, OrderSide, OrderStatus, OrderType
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.services.scheduler import DatabaseJobQueue


class DatabaseDiagnosticPaperWorker:
    """Turn each bootstrap diagnostic assignment into an open/close round trip."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        order_manager: OrderManager,
        positions: PositionManager,
        products: Mapping[str, Mapping[str, Any]],
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.order_manager = order_manager
        self.positions = positions
        self.products = {str(key): dict(value) for key, value in products.items()}
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        now = timestamp(now, field="now")
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("diagnostic_paper_open", "diagnostic_paper_close"),
        )
        if claimed is None:
            return {"reason_code": "diagnostic_paper_queue_empty"}
        try:
            payload = claimed.payload
            product_id = str(payload["product_id"])
            product = self.products[product_id]
            if str(product.get("execution_mode")) != "paper":
                raise ValueError("diagnostic paper jobs require a paper product")
            if payload.get("diagnostic") is not True:
                raise ValueError("diagnostic paper jobs require diagnostic=true")
            result = (
                self._open(payload, product, now)
                if claimed.name == "diagnostic_paper_open"
                else self._close(payload, product, now)
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=self._retry_at(now),
            )
            return {
                "reason_code": "diagnostic_paper_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if result.get("pending"):
            self.queue.fail(
                claimed,
                completed_at=now,
                error=str(result["pending"]),
                retry_at=self._retry_at(now),
            )
            return {key: value for key, value in result.items() if key != "pending"}
        self.queue.complete(claimed, completed_at=now)
        return result

    def _open(
        self, payload: Mapping[str, Any], product: Mapping[str, Any], now: str
    ) -> dict[str, Any]:
        order_id = str(payload["open_order_id"])
        order = self._ensure_order(
            payload=payload,
            product=product,
            order_id=order_id,
            side=OrderSide.BUY,
            quantity=float(payload["quantity"]),
            reduce_only=False,
            phase="open",
            created_at=now,
        )
        self._enqueue_paper_order(payload, product, order, phase="open", now=now)
        close_payload = {**dict(payload), "open_order_id": order_id}
        self.queue.enqueue_if_absent(
            job_id=f"diagnostic-paper:close:{payload['product_id']}",
            name="diagnostic_paper_close",
            payload=close_payload,
            available_at=now,
            priority=99,
            producer_identity="platform-bootstrap:diagnostic-paper",
        )
        return {
            "reason_code": "diagnostic_paper_open_enqueued",
            "job_id": f"diagnostic-paper:open:{payload['product_id']}",
            "order_id": order.order_id,
        }

    def _close(
        self, payload: Mapping[str, Any], product: Mapping[str, Any], now: str
    ) -> dict[str, Any]:
        self.order_manager.reload()
        opening = self.order_manager.get(str(payload["open_order_id"]))
        if opening.status is not OrderStatus.FILLED:
            return {
                "reason_code": "diagnostic_paper_close_pending",
                "job_id": f"diagnostic-paper:close:{payload['product_id']}",
                "pending": "diagnostic opening order is not filled",
            }
        self.positions.reload()
        position = self.positions.get(str(product["portfolio_id"]), str(payload["instrument_id"]))
        if position.quantity == 0:
            return {
                "reason_code": "diagnostic_paper_round_trip_completed",
                "job_id": f"diagnostic-paper:close:{payload['product_id']}",
                "order_id": str(payload["close_order_id"]),
                "position_quantity": 0.0,
            }
        if position.quantity < 0:
            raise ValueError("diagnostic opening order produced a short position")
        order_id = str(payload["close_order_id"])
        existing = self._existing_order(order_id)
        quantity = (
            existing.quantity
            if existing is not None
            else self._close_quantity(position.quantity, product)
        )
        order = self._ensure_order(
            payload=payload,
            product=product,
            order_id=order_id,
            side=OrderSide.SELL,
            quantity=quantity,
            reduce_only=True,
            phase="close",
            created_at=now,
            depends_on_order_id=str(payload["open_order_id"]),
        )
        if order.status is OrderStatus.FILLED:
            return {
                "reason_code": "diagnostic_paper_round_trip_completed",
                "job_id": f"diagnostic-paper:close:{payload['product_id']}",
                "order_id": order.order_id,
                "position_quantity": position.quantity,
            }
        self._enqueue_paper_order(payload, product, order, phase="close", now=now)
        return {
            "reason_code": "diagnostic_paper_close_pending",
            "job_id": f"diagnostic-paper:close:{payload['product_id']}",
            "order_id": order.order_id,
            "pending": "diagnostic closing order is not filled",
        }

    def _ensure_order(
        self,
        *,
        payload: Mapping[str, Any],
        product: Mapping[str, Any],
        order_id: str,
        side: OrderSide,
        quantity: float,
        reduce_only: bool,
        phase: str,
        created_at: str,
        depends_on_order_id: str | None = None,
    ) -> OrderIntent:
        self.order_manager.reload()
        metadata = {
            "diagnostic": True,
            "phase": phase,
            "assignment_id": str(payload["assignment_id"]),
            "artefact_hash": str(payload["artefact_hash"]),
        }
        planned = OrderIntent(
            order_id=order_id,
            portfolio_id=str(product["portfolio_id"]),
            instrument_id=str(payload["instrument_id"]),
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            created_at=created_at,
            reduce_only=reduce_only,
            depends_on_order_id=depends_on_order_id,
            strategy_contributions={str(payload["strategy_version_id"]): 1.0},
            metadata=metadata,
        )
        existing = self._existing_order(order_id)
        if existing is None:
            self.order_manager.create(planned)
            self.order_manager.persist_for_submission(order_id)
            return self.order_manager.get(order_id)
        immutable = (
            "portfolio_id",
            "instrument_id",
            "side",
            "quantity",
            "order_type",
            "reduce_only",
            "depends_on_order_id",
            "strategy_contributions",
            "metadata",
        )
        if any(getattr(existing, key) != getattr(planned, key) for key in immutable):
            raise ValueError(f"diagnostic order identity collision: {order_id}")
        if existing.status is OrderStatus.CREATED:
            self.order_manager.persist_for_submission(order_id)
        return self.order_manager.get(order_id)

    def _enqueue_paper_order(
        self,
        payload: Mapping[str, Any],
        product: Mapping[str, Any],
        order: OrderIntent,
        *,
        phase: str,
        now: str,
    ) -> None:
        order_payload = {
            "order_id": order.order_id,
            "product_id": str(payload["product_id"]),
            "event_id": f"{payload['event_id']}:{phase}",
            "price": float(payload["price"]),
            "execution_costs": dict(product["execution_costs"]),
            "accounting_asset": str(product["base_accounting_asset"]),
            "fee_in_base": str(payload["product_id"]) == "btc_accumulation",
        }
        self.queue.enqueue_if_absent(
            job_id="diagnostic-paper-order:"
            + canonical_hash(order_payload).removeprefix("sha256:"),
            name="paper_order_submit",
            payload=order_payload,
            available_at=now,
            priority=98,
            producer_identity="platform-bootstrap:diagnostic-paper",
        )

    def _existing_order(self, order_id: str) -> OrderIntent | None:
        try:
            return self.order_manager.get(order_id)
        except KeyError:
            return None

    @staticmethod
    def _close_quantity(position_quantity: float, product: Mapping[str, Any]) -> float:
        if str(product.get("product_id")) == "btc_accumulation":
            fee_bps = float(dict(product["execution_costs"])["fee_bps"])
            return position_quantity / (1.0 + fee_bps / 10_000.0)
        return position_quantity

    def _retry_at(self, now: str) -> str:
        parsed = dt.datetime.fromisoformat(now)
        return (
            (parsed + dt.timedelta(seconds=self.lease_seconds)).replace(microsecond=0).isoformat()
        )
