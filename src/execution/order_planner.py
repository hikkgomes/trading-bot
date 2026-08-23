"""Convert target-position deltas into venue-neutral order intents."""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable, Mapping

from src.domain.orders import OrderIntent, OrderSide, OrderType
from src.domain.portfolios import TargetPosition


def _order_id(
    *,
    portfolio_id: str,
    instrument_id: str,
    side: OrderSide,
    quantity: float,
    reduce_only: bool,
    phase: str,
    decided_at: str,
) -> str:
    material = (
        f"{portfolio_id}|{instrument_id}|{side.value}|{quantity:.12f}|"
        f"{int(reduce_only)}|{phase}|{decided_at}"
    )
    return "ord_" + hashlib.sha256(material.encode()).hexdigest()[:24]


def _intent(
    *,
    target: TargetPosition,
    side: OrderSide,
    quantity: float,
    reduce_only: bool,
    phase: str,
    current_quantity: float,
    decided_at: str,
    order_type: OrderType,
    limit_price: float | None,
    depends_on_order_id: str | None = None,
) -> OrderIntent:
    return OrderIntent(
        order_id=_order_id(
            portfolio_id=target.portfolio_id,
            instrument_id=target.instrument_id,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
            phase=phase,
            decided_at=decided_at,
        ),
        portfolio_id=target.portfolio_id,
        instrument_id=target.instrument_id,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        created_at=decided_at,
        reduce_only=reduce_only,
        depends_on_order_id=depends_on_order_id,
        strategy_contributions=target.strategy_contributions,
        metadata={
            "phase": phase,
            "target_quantity": target.target_quantity,
            "current_quantity": current_quantity,
            "target_notional": target.target_notional,
            "target_fraction": target.target_fraction,
            "risk_budget": target.risk_budget,
            "target_metadata": dict(target.metadata),
        },
    )


def plan_orders(
    targets: Iterable[TargetPosition],
    *,
    current_quantities: Mapping[str, float],
    decided_at: str | None = None,
    order_type: OrderType = OrderType.MARKET,
    minimum_quantities: Mapping[str, float] | None = None,
    minimum_notionals: Mapping[str, float] | None = None,
    prices: Mapping[str, float] | None = None,
    limit_prices: Mapping[str, float] | None = None,
) -> tuple[OrderIntent, ...]:
    """Produce executable order intents for target deltas.

    A position reversal is deliberately split into a reduce-only close and a
    separate non-reduce opening order. Exchanges cannot open the new side with
    a reduce-only order.
    """
    decided_at = decided_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    minimum_quantities = minimum_quantities or {}
    minimum_notionals = minimum_notionals or {}
    prices = prices or {}
    limit_prices = limit_prices or {}
    intents: list[OrderIntent] = []
    for target in sorted(targets, key=lambda item: (item.portfolio_id, item.instrument_id)):
        current = float(current_quantities.get(target.instrument_id, 0.0))
        desired = float(target.target_quantity)
        delta = desired - current
        minimum = max(0.0, float(minimum_quantities.get(target.instrument_id, 0.0)))
        if abs(delta) < max(1e-12, minimum):
            continue

        price = float(prices.get(target.instrument_id, 0.0))
        minimum_notional = max(0.0, float(minimum_notionals.get(target.instrument_id, 0.0)))
        if minimum_notional and price <= 0:
            raise ValueError(
                f"price is required to enforce minimum notional for {target.instrument_id}"
            )
        if minimum_notional and abs(delta) * price < minimum_notional:
            continue
        limit_price = limit_prices.get(target.instrument_id)
        if order_type is OrderType.LIMIT and limit_price is None:
            raise ValueError(f"limit price is required for {target.instrument_id}")

        reverses = current != 0 and desired != 0 and (current > 0) != (desired > 0)
        legs = (
            (
                (
                    OrderSide.SELL if current > 0 else OrderSide.BUY,
                    abs(current),
                    True,
                    "close_for_reversal",
                ),
                (
                    OrderSide.BUY if desired > 0 else OrderSide.SELL,
                    abs(desired),
                    False,
                    "open_after_reversal",
                ),
            )
            if reverses
            else (
                (
                    OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    abs(delta),
                    current != 0 and abs(desired) < abs(current),
                    "rebalance",
                ),
            )
        )
        reversal_close_id: str | None = None
        for side, quantity, reduce_only, phase in legs:
            intent = _intent(
                target=target,
                side=side,
                quantity=quantity,
                reduce_only=reduce_only,
                phase=phase,
                current_quantity=current,
                decided_at=decided_at,
                order_type=order_type,
                limit_price=float(limit_price) if limit_price is not None else None,
                depends_on_order_id=(reversal_close_id if phase == "open_after_reversal" else None),
            )
            intents.append(intent)
            if phase == "close_for_reversal":
                reversal_close_id = intent.order_id
    return tuple(intents)
