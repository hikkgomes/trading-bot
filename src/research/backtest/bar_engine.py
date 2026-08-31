"""Small deterministic multi-symbol target-position bar simulator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from src.domain._codec import timestamp


@dataclass(frozen=True)
class BarStep:
    timestamp: str
    prices: Mapping[str, float]
    target_fractions: Mapping[str, float]
    funding_rates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", timestamp(self.timestamp, field="timestamp"))
        if not self.prices:
            raise ValueError("bar step needs prices")
        if any(float(price) <= 0 for price in self.prices.values()):
            raise ValueError("bar prices must be positive")
        if any(not -1 <= float(value) <= 1 for value in self.target_fractions.values()):
            raise ValueError("target fractions must be in [-1, 1]")


@dataclass(frozen=True)
class BarSimulationResult:
    equity_curve: tuple[tuple[str, float], ...]
    quantities: Mapping[str, float]
    fees_paid: float
    funding_paid: float
    slippage_paid: float = 0.0


class BarPortfolioEngine:
    """Rebalance simultaneous positions from target fractions at each bar."""

    def __init__(
        self, *, initial_equity: float, fee_bps: float = 5.0, slippage_bps: float = 0.0
    ) -> None:
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if fee_bps < 0 or slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be non-negative")
        self.initial_equity = float(initial_equity)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)

    def simulate(self, steps: tuple[BarStep, ...]) -> BarSimulationResult:
        if not steps:
            return BarSimulationResult((), {}, 0.0, 0.0)
        cash = self.initial_equity
        quantities: dict[str, float] = {}
        fees_paid = funding_paid = slippage_paid = 0.0
        curve: list[tuple[str, float]] = []
        for step in steps:
            equity_before = cash + sum(
                quantity * float(step.prices[symbol])
                for symbol, quantity in quantities.items()
                if symbol in step.prices
            )
            for symbol, target_fraction in step.target_fractions.items():
                price = float(step.prices[symbol])
                target_quantity = equity_before * float(target_fraction) / price
                current_quantity = quantities.get(symbol, 0.0)
                delta = target_quantity - current_quantity
                fee = abs(delta * price) * self.fee_bps / 10_000
                slippage = abs(delta * price) * self.slippage_bps / 10_000
                cash -= delta * price + fee + slippage
                fees_paid += fee
                slippage_paid += slippage
                quantities[symbol] = target_quantity
            for symbol, quantity in quantities.items():
                rate = float(step.funding_rates.get(symbol, 0.0))
                funding = quantity * float(step.prices[symbol]) * rate
                cash -= funding
                funding_paid += funding
            equity = cash + sum(
                quantity * float(step.prices[symbol]) for symbol, quantity in quantities.items()
            )
            curve.append((step.timestamp, equity))
        return BarSimulationResult(tuple(curve), quantities, fees_paid, funding_paid, slippage_paid)
