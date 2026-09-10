"""Net explicit book decisions without treating a missing forecast as an exit."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, localcontext
from typing import Any


@dataclass(frozen=True)
class NetBookPlan:
    desired_positions: Mapping[str, Mapping[str, float]]
    internal_transfers: tuple[dict[str, Any], ...]
    external_changes: Mapping[str, float]
    contributions: Mapping[str, tuple[dict[str, Any], ...]]


def net_book_changes(targets, current, prices) -> NetBookPlan:
    desired: dict[str, dict[str, float]] = {}
    changes: dict[str, dict[str, float]] = {}
    for book_id in sorted(set(current) | set(targets)):
        held = current.get(book_id, {})
        requested = targets.get(book_id)
        selected = held if requested is None else requested
        desired[book_id] = {
            symbol: _quantity(selected.get(symbol, 0))
            for symbol in sorted(set(held) | set(selected))
        }
        for symbol, quantity in desired[book_id].items():
            delta = quantity - _quantity(held.get(symbol, 0))
            if abs(delta) > 1e-12:
                changes.setdefault(symbol, {})[book_id] = delta
    transfers, external, contributions = [], {}, {}
    for symbol, deltas in sorted(changes.items()):
        price = _quantity(prices[symbol])
        if price <= 0:
            raise ValueError("book netting requires a positive execution reference")
        matched, remaining = _net_instrument(symbol, deltas, price)
        transfers.extend(matched)
        quantity = sum(item["signed_quantity"] for item in remaining)
        if abs(quantity) > 1e-12:
            external[symbol] = quantity
            contributions[symbol] = tuple(remaining)
    return NetBookPlan(desired, tuple(transfers), external, contributions)


def _net_instrument(symbol, deltas, price):
    buys = [[book, value] for book, value in sorted(deltas.items()) if value > 0]
    sells = [[book, -value] for book, value in sorted(deltas.items()) if value < 0]
    transfers = []
    buy = sell = 0
    while buy < len(buys) and sell < len(sells):
        quantity = min(buys[buy][1], sells[sell][1])
        transfers.append(
            {
                "kind": "internal_netting",
                "instrument_id": symbol,
                "from_book": sells[sell][0],
                "to_book": buys[buy][0],
                "quantity": quantity,
                "price": price,
            }
        )
        buys[buy][1] -= quantity
        sells[sell][1] -= quantity
        buy += buys[buy][1] <= 1e-12
        sell += sells[sell][1] <= 1e-12
    remaining = [
        {"book_id": book, "signed_quantity": sign * quantity}
        for items, sign in ((buys, 1), (sells, -1))
        for book, quantity in items
        if quantity > 1e-12
    ]
    return transfers, remaining


def allocate_book_fill(signed_quantity: float, fee: float, contributions):
    total = _decimal_quantity(signed_quantity)
    commission = _decimal_quantity(fee)
    if not contributions or total == 0 or commission < 0:
        raise ValueError("fill allocation requires contributions, quantity and a non-negative fee")
    if len({item["book_id"] for item in contributions}) != len(contributions):
        raise ValueError("a fill must have one allocation per book")
    weights = [_decimal_quantity(item["signed_quantity"]) for item in contributions]
    if any(weight * total <= 0 for weight in weights):
        raise ValueError(
            "external allocations must match the fill side; opposing intent requires internal netting"
        )
    exponent = min(-18, int(total.as_tuple().exponent), int(commission.as_tuple().exponent))
    quantum = Decimal(1).scaleb(exponent)
    with localcontext() as context:
        context.prec = max(38, max(total.adjusted(), commission.adjusted()) - exponent + 4)
        denominator = sum(abs(weight) for weight in weights)
        allocated = fees = Decimal(0)
        result = []
        for index, (item, weight) in enumerate(zip(contributions, weights, strict=True)):
            last = index == len(contributions) - 1
            quantity = (
                total - allocated
                if last
                else (total * abs(weight) / denominator).quantize(quantum, rounding=ROUND_DOWN)
            )
            amount = (
                commission - fees
                if last
                else (commission * abs(weight) / denominator).quantize(quantum, rounding=ROUND_DOWN)
            )
            result.append({**item, "signed_quantity": quantity, "fee": amount})
            allocated += quantity
            fees += amount
    return tuple(result)


def protective_book_metadata(metadata, *, exit_sign):
    """A protective child closes the entry owners, retaining their relative weights."""
    if exit_sign not in {-1, 1}:
        raise ValueError("protective allocation requires an explicit exit direction")
    contributions = metadata.get("book_contributions")
    if not contributions:
        return dict(metadata)
    return {
        **metadata,
        "book_contributions": [
            {**row, "signed_quantity": exit_sign * abs(_quantity(row["signed_quantity"]))}
            for row in contributions
        ],
    }


def _decimal_quantity(value):
    if isinstance(value, bool):
        raise ValueError("book allocation amount must be numeric")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("book allocation amount must be finite")
    return result


def _quantity(value) -> float:
    if isinstance(value, bool):
        raise ValueError("book quantity must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("book quantity must be finite")
    return result
