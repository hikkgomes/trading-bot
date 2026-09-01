"""BTC-denominated tactical allocation model."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from src.domain._codec import canonical_hash
from src.domain.forecasts import AlphaForecast, ForecastDirection

BTC_SPOT_INSTRUMENT_ID = "binance:spot:BTCUSDT"


def assert_btc_spot_instrument(instrument_id: str) -> str:
    """Validate the sole instrument allowed by the BTC accumulation product."""

    value = str(instrument_id).strip()
    if value != BTC_SPOT_INSTRUMENT_ID:
        raise ValueError("BTC accumulation requires binance:spot:BTCUSDT")
    return value


def btc_step_aside_metadata(
    *,
    instrument_id: str,
    current_btc: float,
    target_btc: float,
    price: float,
    stablecoin_balance: float,
    state_id: str,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict[str, object]:
    """Create restart-safe metadata for one bounded BTC tactical cycle."""

    assert_btc_spot_instrument(instrument_id)
    values = (current_btc, target_btc, price, stablecoin_balance, fee_bps, slippage_bps)
    if (
        any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values)
        or price <= 0.0
    ):
        raise ValueError("BTC step-aside values must be non-negative with a positive price")
    sell_quantity = max(0.0, current_btc - target_btc)
    sell_price = price * max(0.0, 1.0 - slippage_bps / 10_000.0)
    quote_proceeds = sell_quantity * sell_price * max(0.0, 1.0 - fee_bps / 10_000.0)
    budget = stablecoin_balance + quote_proceeds
    state = (
        "step_aside"
        if sell_quantity > 0.0
        else "rebuy"
        if target_btc > current_btc
        else "core_hold"
    )
    lot_payload = {
        "instrument_id": instrument_id,
        "state_id": str(state_id),
        "current_btc": current_btc,
        "target_btc": target_btc,
        "price": price,
        "sell_quantity_btc": sell_quantity,
    }
    return {
        "btc_cycle_state": state,
        "btc_step_aside_lot_id": canonical_hash(lot_payload) if sell_quantity > 0.0 else None,
        "btc_step_aside_sold_quantity": sell_quantity,
        "btc_step_aside_quote_proceeds": quote_proceeds,
        "btc_quote_reinvest_budget": budget,
        "btc_quote_budget_source": "owned_balance_plus_step_aside_proceeds",
        "btc_step_aside_state_hash": canonical_hash(lot_payload),
    }


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

    @property
    def minimum_btc_fraction(self) -> float:
        """Lowest BTC fraction reachable by the bounded tactical sleeve."""

        return max(0.0, self.core_btc_fraction - self.max_tactical_fraction)


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
    minimum_target = policy.minimum_btc_fraction
    target = max(minimum_target, min(1.0, policy.core_btc_fraction + tactical))
    return BtcAllocationTarget(
        target_btc_fraction=target,
        core_btc_fraction=policy.core_btc_fraction,
        tactical_btc_fraction=tactical,
        stablecoin_fraction=1.0 - target,
        contributions=contributions,
    )
