"""Venue-normalised instrument identities and trading constraints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.domain._codec import finite, non_empty


class MarketType(StrEnum):
    SPOT = "spot"
    FUTURES = "futures"


def canonical_instrument_id(
    exchange_symbol: str,
    *,
    market_type: MarketType | str,
    venue: str = "binance",
    settlement_asset: str | None = None,
) -> str:
    """Build the durable platform identity for an exchange symbol."""

    symbol = non_empty(exchange_symbol, field="exchange_symbol").upper()
    market = MarketType(market_type)
    venue_name = non_empty(venue, field="venue").lower()
    settlement = (
        non_empty(settlement_asset, field="settlement_asset").upper()
        if settlement_asset is not None
        else None
    )
    if market is MarketType.SPOT and settlement is not None:
        raise ValueError("spot instruments cannot have a settlement asset")
    suffix = f":{settlement}" if settlement else ""
    return f"{venue_name}:{market.value}:{symbol}{suffix}"


@dataclass(frozen=True)
class Instrument:
    venue: str
    market_type: MarketType
    base_asset: str
    quote_asset: str
    settlement_asset: str | None
    exchange_symbol: str
    price_precision: int
    quantity_precision: int
    minimum_quantity: float
    minimum_notional: float
    contract_size: float = 1.0
    status: str = "trading"

    def __post_init__(self) -> None:
        object.__setattr__(self, "venue", non_empty(self.venue, field="venue").lower())
        object.__setattr__(
            self, "base_asset", non_empty(self.base_asset, field="base_asset").upper()
        )
        object.__setattr__(
            self, "quote_asset", non_empty(self.quote_asset, field="quote_asset").upper()
        )
        if self.settlement_asset is not None:
            object.__setattr__(
                self,
                "settlement_asset",
                non_empty(self.settlement_asset, field="settlement_asset").upper(),
            )
        if self.market_type is MarketType.SPOT and self.settlement_asset is not None:
            raise ValueError("spot instruments cannot have a settlement_asset")
        object.__setattr__(
            self,
            "exchange_symbol",
            non_empty(self.exchange_symbol, field="exchange_symbol").upper(),
        )
        object.__setattr__(self, "status", non_empty(self.status, field="status").lower())
        for field in ("price_precision", "quantity_precision"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        for field in ("minimum_quantity", "minimum_notional", "contract_size"):
            object.__setattr__(self, field, finite(getattr(self, field), field=field, minimum=0.0))
        if self.contract_size == 0:
            raise ValueError("contract_size must be positive")

    @property
    def instrument_id(self) -> str:
        settlement = f":{self.settlement_asset}" if self.settlement_asset else ""
        return f"{self.venue}:{self.market_type.value}:{self.exchange_symbol}{settlement}"

    @property
    def is_tradable(self) -> bool:
        return self.status == "trading"
