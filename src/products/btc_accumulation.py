"""BTC-denominated tactical allocation model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.domain.forecasts import AlphaForecast, ForecastDirection


@dataclass(frozen=True)
class BtcAllocationPolicy:
    """Spot BTC policy with a neutral 100% BTC default.

    ``max_tactical_fraction`` is a bounded reduction from the neutral BTC
    allocation. A stablecoin reserve is therefore explicit and never an
    implicit consequence of a missing forecast.
    """

    core_btc_fraction: float = 1.0
    max_tactical_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.core_btc_fraction <= 1:
            raise ValueError("core_btc_fraction must be in [0, 1]")
        if not 0 <= self.max_tactical_fraction <= 1:
            raise ValueError("max_tactical_fraction must be in [0, 1]")
        if self.core_btc_fraction == 0 and self.max_tactical_fraction > 0:
            raise ValueError("a tactical BTC sleeve needs a positive neutral allocation")


@dataclass(frozen=True)
class BtcAllocationTarget:
    target_btc_fraction: float
    core_btc_fraction: float
    tactical_btc_fraction: float
    stablecoin_fraction: float
    contributions: dict[str, float]


def target_btc_allocation(
    forecasts: Iterable[AlphaForecast], *, policy: BtcAllocationPolicy = BtcAllocationPolicy()
) -> BtcAllocationTarget:
    """Merge strategy sleeves into a BTC fraction in [0, 1]."""
    contributions: dict[str, float] = {}
    for forecast in forecasts:
        if forecast.direction is ForecastDirection.FLAT:
            contribution = 0.0
        else:
            sign = 1.0 if forecast.direction is ForecastDirection.LONG else -1.0
            contribution = sign * forecast.score * forecast.confidence * forecast.maximum_position
        contributions[forecast.strategy_version_id] = contribution
    signal = sum(contributions.values())
    tactical = max(-policy.max_tactical_fraction, min(policy.max_tactical_fraction, signal))
    minimum_target = max(0.0, policy.core_btc_fraction - policy.max_tactical_fraction)
    target = max(minimum_target, min(1.0, policy.core_btc_fraction + tactical))
    return BtcAllocationTarget(
        target_btc_fraction=target,
        core_btc_fraction=policy.core_btc_fraction,
        tactical_btc_fraction=tactical,
        stablecoin_fraction=1.0 - target,
        contributions=contributions,
    )
