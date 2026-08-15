"""Compare local and exchange inventory without requiring a flat account."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ReconciliationResult:
    matched: bool
    unknown_positions: dict[str, float]
    quantity_mismatches: dict[str, dict[str, float]]
    unknown_orders: tuple[str, ...]
    missing_exchange_orders: tuple[str, ...]
    recovery_required: bool


def reconcile_account(
    *,
    local_positions: Mapping[str, float],
    exchange_positions: Mapping[str, float],
    local_open_order_ids: set[str],
    exchange_open_order_ids: set[str],
    tolerance: float = 1e-12,
) -> ReconciliationResult:
    unknown_positions = {
        symbol: quantity
        for symbol, quantity in exchange_positions.items()
        if symbol not in local_positions and abs(quantity) > tolerance
    }
    mismatches: dict[str, dict[str, float]] = {}
    for symbol in sorted(set(local_positions) | set(exchange_positions)):
        local = float(local_positions.get(symbol, 0.0))
        exchange = float(exchange_positions.get(symbol, 0.0))
        if abs(local - exchange) > tolerance:
            mismatches[symbol] = {"local_quantity": local, "exchange_quantity": exchange}
    unknown_orders = tuple(sorted(exchange_open_order_ids - local_open_order_ids))
    missing_exchange_orders = tuple(sorted(local_open_order_ids - exchange_open_order_ids))
    recovery_required = bool(
        unknown_positions or mismatches or unknown_orders or missing_exchange_orders
    )
    return ReconciliationResult(
        matched=not recovery_required,
        unknown_positions=unknown_positions,
        quantity_mismatches=mismatches,
        unknown_orders=unknown_orders,
        missing_exchange_orders=missing_exchange_orders,
        recovery_required=recovery_required,
    )
