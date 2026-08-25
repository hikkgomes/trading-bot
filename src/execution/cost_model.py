"""Simple transparent execution cost estimates for portfolio allocation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCostModel:
    fee_bps: float
    half_spread_bps: float = 0.0
    impact_bps_per_daily_volume_fraction: float = 0.0

    def estimate_bps(self, *, notional: float, daily_quote_volume: float) -> float:
        if notional < 0 or daily_quote_volume < 0:
            raise ValueError("notional and daily_quote_volume must be non-negative")
        impact = (
            0.0
            if daily_quote_volume == 0
            else self.impact_bps_per_daily_volume_fraction * notional / daily_quote_volume
        )
        return self.fee_bps + self.half_spread_bps + impact

    def estimate_cost(self, *, notional: float, daily_quote_volume: float) -> float:
        return (
            notional
            * self.estimate_bps(notional=notional, daily_quote_volume=daily_quote_volume)
            / 10_000
        )
