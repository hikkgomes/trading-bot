"""Flat-to-flat execution campaigns with unresolved inventory kept explicit."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from src.domain._codec import canonical_hash
from src.research.historical.imports import HistoricalOrder


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    symbol: str
    side: str
    started_at: str
    ended_at: str
    remaining_quantity: Decimal
    minimum_inventory: Decimal
    maximum_inventory: Decimal
    recorded_gross: Decimal
    fees: Decimal
    gross_cashflow: Decimal
    reconciled: bool
    closed: bool
    source_rows: tuple[str, ...]


@dataclass
class _Lane:
    quantity: Decimal = Decimal(0)
    minimum: Decimal = Decimal(0)
    maximum: Decimal = Decimal(0)
    orders: list[HistoricalOrder] = field(default_factory=list)


@dataclass(frozen=True)
class Reconstruction:
    campaigns: tuple[Campaign, ...]
    inventory_events: tuple[dict, ...]
    ambiguous_timestamps: tuple[str, ...]


def reconstruct_campaigns(
    orders: Sequence[HistoricalOrder], *, tolerance: Decimal = Decimal("0.003")
) -> Reconstruction:
    lanes: dict[tuple[str, str], _Lane] = {}
    campaigns, events = [], []
    time_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    ordered = sorted(
        (order for order in orders if order.market == "swap"),
        key=lambda order: (order.occurred_at, order.source.sheet, -order.source.row_number),
    )
    for order in ordered:
        key = (order.symbol, order.position_side)
        lane = lanes.setdefault(key, _Lane())
        _append_order(lane, order)
        time_counts[(*key, order.occurred_at.isoformat())] += 1
        events.append(
            {
                "timestamp": order.occurred_at.isoformat(),
                "symbol": order.symbol,
                "side": order.position_side,
                "quantity": lane.quantity,
                "source_row": order.source.row_hash,
                "unresolved_inventory": lane.quantity < 0,
            }
        )
        if lane.quantity == 0:
            campaigns.append(_campaign(lane, tolerance))
            lanes[key] = _Lane()
    campaigns.extend(_campaign(lane, tolerance) for lane in lanes.values() if lane.orders)
    return Reconstruction(
        tuple(campaigns),
        tuple(events),
        tuple(": ".join(key) for key, count in time_counts.items() if count > 1),
    )


def _append_order(lane: _Lane, order: HistoricalOrder) -> None:
    if order.action not in {"open", "close"} or order.position_side not in {"long", "short"}:
        raise ValueError("swap campaigns require explicit open/close and long/short fields")
    if order.fee_currency != "USDT":
        raise ValueError("campaign fees require an explicit USDT conversion")
    lane.orders.append(order)
    lane.quantity += order.quantity if order.action == "open" else -order.quantity
    lane.minimum = min(lane.minimum, lane.quantity)
    lane.maximum = max(lane.maximum, lane.quantity)


def _campaign(lane: _Lane, tolerance: Decimal) -> Campaign:
    first, last = lane.orders[0], lane.orders[-1]
    gross = sum((order.recorded_pnl for order in lane.orders), Decimal(0))
    fees = sum((order.fee for order in lane.orders), Decimal(0))
    cashflow = sum((_cashflow(order) for order in lane.orders), Decimal(0))
    rows = tuple(order.source.row_hash for order in lane.orders)
    balanced = lane.quantity == 0 and lane.minimum >= 0
    return Campaign(
        canonical_hash({"rows": rows}),
        first.symbol,
        first.position_side,
        first.occurred_at.isoformat(),
        last.occurred_at.isoformat(),
        lane.quantity,
        lane.minimum,
        lane.maximum,
        gross,
        fees,
        cashflow,
        balanced and abs(cashflow - gross) <= tolerance,
        lane.quantity == 0,
        rows,
    )


def _cashflow(order: HistoricalOrder) -> Decimal:
    position_sign = 1 if order.position_side == "long" else -1
    action_sign = -1 if order.action == "open" else 1
    return order.quantity * order.price * position_sign * action_sign


def hedged_exposure(reconstruction: Reconstruction) -> tuple[dict, ...]:
    inventory: dict[tuple[str, str], Decimal] = {}
    result = []
    for event in reconstruction.inventory_events:
        symbol, side = event["symbol"], event["side"]
        inventory[(symbol, side)] = event["quantity"]
        long = inventory.get((symbol, "long"), Decimal(0))
        short = inventory.get((symbol, "short"), Decimal(0))
        if long > 0 and short > 0:
            result.append(
                {
                    "timestamp": event["timestamp"],
                    "symbol": symbol,
                    "long_quantity": long,
                    "short_quantity": short,
                    "net_quantity": long - short,
                    "gross_quantity": long + short,
                }
            )
    return tuple(result)
