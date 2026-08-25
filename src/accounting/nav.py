"""Product NAV calculations in BTC or USDT accounting units."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class NavSnapshot:
    product_id: str
    accounting_asset: str
    nav: float
    observed_at: str
    components: Mapping[str, float]
    passive_benchmark_nav: float | None = None


def btc_nav(
    *,
    btc_balance: float,
    stablecoin_balance: float,
    stablecoin_per_btc: float,
) -> float:
    if btc_balance < 0 or stablecoin_balance < 0 or stablecoin_per_btc <= 0:
        raise ValueError("BTC NAV inputs must be non-negative with a positive conversion price")
    return btc_balance + stablecoin_balance / stablecoin_per_btc


def usdt_nav(
    *,
    cash_balance: float,
    positions: Mapping[str, tuple[float, float, float]],
) -> float:
    """Return cash plus unrealised PnL from quantity, entry price, and mark."""
    nav = float(cash_balance)
    for quantity, entry_price, mark_price in positions.values():
        if entry_price <= 0 or mark_price <= 0:
            raise ValueError("position entry and mark prices must be positive")
        nav += float(quantity) * (float(mark_price) - float(entry_price))
    return nav
