"""Self-financing spot rebalancing with quote-currency execution costs."""

import math


def target_trade_quantity(
    *,
    base: float,
    quote: float,
    mark: float,
    fraction: float,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    values = (base, quote, mark, fraction, fee_rate, slippage_rate)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("spot allocation inputs must be finite and non-negative")
    if mark <= 0 or fraction > 1 or fee_rate >= 1 or slippage_rate >= 1:
        raise ValueError("spot allocation price or fraction is invalid")
    gap = fraction * (base * mark + quote) - base * mark
    if gap >= 0:
        execution_cost = mark * (1 + slippage_rate) * (1 + fee_rate)
    else:
        execution_cost = mark * (1 - slippage_rate) * (1 - fee_rate)
    return gap / (mark * (1 - fraction) + fraction * execution_cost)
