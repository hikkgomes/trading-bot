"""Production-contract paper exchange with deterministic partial fills."""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Callable

from src.domain.orders import Fill, OrderIntent, OrderStatus
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager

PriceSource = Callable[[str], float]


class PaperExchange:
    """Submit the same :class:`OrderIntent` used by live execution.

    ``fill_fraction`` is intentionally configurable so recovery and partial-fill
    paths can be tested without an exchange connection.
    """

    def __init__(
        self,
        *,
        order_manager: OrderManager,
        position_manager: PositionManager,
        price_source: PriceSource,
        fee_bps: float = 5.0,
        slippage_bps: float = 0.0,
        fill_fraction: float = 1.0,
        fee_asset: str | None = None,
        fee_in_base: bool = False,
    ) -> None:
        if not 0 < fill_fraction <= 1:
            raise ValueError("fill_fraction must be in (0, 1]")
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.price_source = price_source
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.fill_fraction = float(fill_fraction)
        self.fee_asset = fee_asset.upper() if fee_asset else None
        if fee_in_base and self.fee_asset is None:
            raise ValueError("fee_in_base requires fee_asset")
        self.fee_in_base = fee_in_base

    def submit(self, intent: OrderIntent) -> Fill:
        existing_ids = {item.order_id for item in self.order_manager.all()}
        if intent.order_id not in existing_ids:
            self.order_manager.create(intent)
        else:
            current = self.order_manager.get(intent.order_id)
            if current != intent:
                raise ValueError(f"order identity collision: {intent.order_id}")
        self.order_manager.persist_for_submission(intent.order_id)
        self.order_manager.submitted(intent.order_id)
        return self._fill(intent.order_id, fill_fraction=self.fill_fraction)

    def fill_remaining(self, order_id: str, *, fill_fraction: float = 1.0) -> Fill:
        """Apply another deterministic fill to an acknowledged partial order."""
        if not 0 < fill_fraction <= 1:
            raise ValueError("fill_fraction must be in (0, 1]")
        current = self.order_manager.get(order_id)
        if current.status not in {
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.CANCEL_PENDING,
        }:
            raise ValueError(f"order {order_id} cannot receive another fill")
        return self._fill(order_id, fill_fraction=fill_fraction)

    def cancel_remaining(self, order_id: str) -> OrderIntent:
        current = self.order_manager.get(order_id)
        if current.status not in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}:
            raise ValueError(f"order {order_id} cannot be cancelled")
        self.order_manager.request_cancel(order_id)
        return self.order_manager.cancelled(order_id)

    def _fill(self, order_id: str, *, fill_fraction: float) -> Fill:
        intent = self.order_manager.get(order_id)
        price = float(self.price_source(intent.instrument_id))
        if price <= 0:
            raise ValueError("paper price must be positive")
        sign = 1 if intent.side.value == "buy" else -1
        fill_price = price * (1 + sign * self.slippage_bps / 10_000)
        quantity = intent.remaining_quantity * fill_fraction
        fee = round(
            quantity * (1.0 if self.fee_in_base else fill_price) * self.fee_bps / 10_000,
            12,
        )
        slippage_quote = round(abs(fill_price - price) * quantity, 12)
        now = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        fill = Fill(
            fill_id="fill_"
            + hashlib.sha256(
                (f"{intent.order_id}|{intent.filled_quantity:.12f}|{quantity:.12f}|{now}").encode()
            ).hexdigest()[:24],
            order_id=intent.order_id,
            instrument_id=intent.instrument_id,
            side=intent.side,
            quantity=quantity,
            price=fill_price,
            fee=fee,
            occurred_at=now,
            fee_asset=self.fee_asset,
            metadata={
                "reference_price": price,
                "slippage_cost": slippage_quote / price if self.fee_in_base else slippage_quote,
                "base_fee_quantity": fee if self.fee_in_base else 0.0,
                "simulated": True,
            },
        )
        updated = self.order_manager.apply_fill(fill)
        self.position_manager.apply_fill(
            updated.portfolio_id,
            fill,
            contributions=dict(updated.strategy_contributions),
        )
        return fill
