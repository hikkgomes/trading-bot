"""Product NAV calculations in BTC or USDT accounting units."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.domain._codec import finite, json_value, non_empty, timestamp


@dataclass(frozen=True)
class NavSnapshot:
    product_id: str
    accounting_asset: str
    nav: float
    observed_at: str
    components: Mapping[str, float]
    passive_benchmark_nav: float | None = None
    portfolio_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "product_id", non_empty(self.product_id, field="product_id"))
        object.__setattr__(
            self,
            "accounting_asset",
            non_empty(self.accounting_asset, field="accounting_asset").upper(),
        )
        object.__setattr__(self, "observed_at", timestamp(self.observed_at, field="observed_at"))
        object.__setattr__(self, "nav", finite(self.nav, field="nav", minimum=0.0))
        if not isinstance(self.components, Mapping):
            raise ValueError("NAV components must be an object")
        object.__setattr__(
            self,
            "components",
            json_value(dict(self.components), field="NAV components"),
        )
        if self.passive_benchmark_nav is not None:
            object.__setattr__(
                self,
                "passive_benchmark_nav",
                finite(self.passive_benchmark_nav, field="passive_benchmark_nav", minimum=0.0),
            )
        if self.portfolio_id is not None:
            object.__setattr__(
                self, "portfolio_id", non_empty(self.portfolio_id, field="portfolio_id")
            )


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
    if nav < 0:
        raise ValueError("cash balance must be non-negative")
    for quantity, entry_price, mark_price in positions.values():
        if entry_price <= 0 or mark_price <= 0:
            raise ValueError("position entry and mark prices must be positive")
        nav += float(quantity) * (float(mark_price) - float(entry_price))
    if nav < 0:
        raise ValueError("USDT NAV cannot be negative")
    return nav
