"""Position-management tactics: DCA ladders, scaled exits, stink bids.

These turn the blueprint's *execution* advice into broker-agnostic plans (lists
of price/quantity legs) that a runner can place through any ``Broker``:

* **DCA ladder** — split a quote budget into several buys across a support zone
  ("dollar cost average into your position near the range lows").
* **Scaled exit** — sell 50% / 30% / 20% into and above range-high resistance
  ("sell into FOMO wicks"), which raises your average exit vs dumping all at once.
* **Stink bids** — far-below resting limit buys to catch capitulation wicks;
  if unfilled they cost nothing and your DCA still builds the position.

Pure functions + small dataclasses, so they are trivially unit-tested and reused
by ``run_bot`` or the live executor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.execution.broker import Order, OrderSide, OrderType


@dataclass
class PlanLeg:
    """One leg of an execution plan."""

    side: OrderSide
    price: float          # limit price for this leg
    qty: float            # base-asset quantity
    fraction: float       # share of the budget / position this leg represents
    note: str = ""

    def to_order(self, symbol: str, reduce_only: bool = False) -> Order:
        return Order(
            symbol=symbol, side=self.side, qty=self.qty, type=OrderType.LIMIT,
            price=self.price, reduce_only=reduce_only,
        )


def _normalize(weights: Sequence[float]) -> list[float]:
    if len(weights) == 0:
        raise ValueError("weights must not be empty.")
    clean = [_positive_float("weight", weight) for weight in weights]
    total = float(sum(clean))
    if total <= 0:
        raise ValueError("weights must sum to a positive number.")
    return [w / total for w in clean]


def _positive_float(name: str, value: float) -> float:
    try:
        clean = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(clean) or clean <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return clean


def _non_negative_float(name: str, value: float) -> float:
    try:
        clean = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(clean) or clean < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return clean


def dca_buy_plan(
    quote_budget: float, low: float, high: float, levels: int = 4, lower_heavy: bool = True
) -> list[PlanLeg]:
    """Split ``quote_budget`` into ``levels`` buys evenly priced across [low, high].

    With ``lower_heavy`` more budget is allocated to the cheaper levels (a simple
    linear weighting), so the average entry sits below the midpoint of the zone.
    """
    quote_budget = _positive_float("quote_budget", quote_budget)
    low = _positive_float("low", low)
    high = _positive_float("high", high)
    if not isinstance(levels, int) or levels < 1 or high < low:
        raise ValueError("Need levels>=1 and 0 < low <= high.")
    prices = [high] if levels == 1 else [high - (high - low) * i / (levels - 1) for i in range(levels)]
    raw = [1.0 + i if lower_heavy else 1.0 for i in range(levels)]  # heavier toward lower prices
    weights = _normalize(raw)
    legs: list[PlanLeg] = []
    for price, w in zip(prices, weights):
        quote = quote_budget * w
        legs.append(PlanLeg(OrderSide.BUY, price, quote / price, w, note="dca"))
    return legs


def scaled_exit_plan(
    qty: float,
    range_high: float,
    fractions: Sequence[float] = (0.5, 0.3, 0.2),
    offsets: Sequence[float] = (0.0, 0.02, 0.05),
) -> list[PlanLeg]:
    """Ladder out of a long: sell ``fractions`` of ``qty`` at ``range_high`` and
    ``offsets`` above it (the 50/30/20 "into FOMO wicks" pattern)."""
    qty = _positive_float("qty", qty)
    range_high = _positive_float("range_high", range_high)
    if len(fractions) == 0:
        raise ValueError("fractions must not be empty.")
    if len(fractions) != len(offsets):
        raise ValueError("fractions and offsets must be the same length.")
    clean_fractions = [_positive_float("fraction", fraction) for fraction in fractions]
    clean_offsets = [_non_negative_float("offset", offset) for offset in offsets]
    if abs(sum(clean_fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1.0.")
    legs: list[PlanLeg] = []
    for frac, off in zip(clean_fractions, clean_offsets):
        price = range_high * (1.0 + off)
        legs.append(PlanLeg(OrderSide.SELL, price, qty * frac, frac, note="scaled_exit"))
    return legs


def stink_bid_plan(
    quote_budget: float,
    ref_price: float,
    depths: Sequence[float] = (0.10, 0.20, 0.35),
    weights: Sequence[float] | None = None,
) -> list[PlanLeg]:
    """Resting limit buys at ``depths`` below ``ref_price`` to catch capitulation.

    ``weights`` allocates the budget across depths (defaults to equal). Deeper
    levels get a bigger discount; if never filled they simply expire.
    """
    quote_budget = _positive_float("quote_budget", quote_budget)
    ref_price = _positive_float("ref_price", ref_price)
    if len(depths) == 0:
        raise ValueError("Provide at least one depth.")
    clean_depths = [_non_negative_float("depth", depth) for depth in depths]
    if weights is not None and len(weights) != len(clean_depths):
        raise ValueError("weights and depths must be the same length.")
    w = _normalize(weights if weights is not None else [1.0] * len(depths))
    legs: list[PlanLeg] = []
    for depth, weight in zip(clean_depths, w):
        if not 0.0 < depth < 1.0:
            raise ValueError("each depth must be in (0, 1).")
        price = ref_price * (1.0 - depth)
        quote = quote_budget * weight
        legs.append(PlanLeg(OrderSide.BUY, price, quote / price, weight, note=f"stink_bid:-{depth:.0%}"))
    return legs
