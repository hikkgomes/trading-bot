"""Deterministic conversion of exchange fees into the product accounting asset."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class FeeConversionError(ValueError):
    """A fee cannot be valued in the configured accounting asset."""


@dataclass(frozen=True)
class FeeConversion:
    fee_asset: str
    fee_amount: float
    accounting_asset: str
    accounting_amount: float
    conversion_rate: float
    source: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "fee_asset": self.fee_asset,
            "fee_amount": self.fee_amount,
            "accounting_asset": self.accounting_asset,
            "accounting_amount": self.accounting_amount,
            "fee_conversion_rate": self.conversion_rate,
            "fee_conversion_source": self.source,
        }


def convert_fee(
    *,
    amount: float,
    fee_asset: str | None,
    accounting_asset: str,
    trade_price: float,
    base_asset: str | None = None,
    quote_asset: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FeeConversion:
    fee_amount = _number(amount, field="fee amount", minimum=0.0)
    accounting = _asset(accounting_asset, field="accounting asset")
    asset = _asset(fee_asset or accounting, field="fee asset")
    price = _number(trade_price, field="trade price", minimum=0.0)
    if fee_amount == 0.0:
        return FeeConversion(asset, fee_amount, accounting, 0.0, 1.0, "zero_fee")
    if price <= 0.0:
        raise FeeConversionError("trade price must be positive for fee conversion")
    if asset == accounting:
        return FeeConversion(asset, fee_amount, accounting, fee_amount, 1.0, "same_asset")

    details = metadata if isinstance(metadata, Mapping) else {}
    explicit = _explicit_conversion(
        asset=asset,
        amount=fee_amount,
        accounting=accounting,
        details=details,
    )
    if explicit is not None:
        return explicit
    market = _market_conversion(
        asset=asset,
        amount=fee_amount,
        accounting=accounting,
        price=price,
        base_asset=base_asset,
        quote_asset=quote_asset,
    )
    if market is not None:
        return market
    raise FeeConversionError(f"fee asset {asset} needs a deterministic conversion to {accounting}")


def _explicit_conversion(
    *,
    asset: str,
    amount: float,
    accounting: str,
    details: Mapping[str, Any],
) -> FeeConversion | None:
    explicit_amount = _optional_number(details, "fee_in_accounting_asset")
    if explicit_amount is not None:
        if explicit_amount < 0.0:
            raise FeeConversionError("fee_in_accounting_asset must be non-negative")
        rate = explicit_amount / amount
        return FeeConversion(asset, amount, accounting, explicit_amount, rate, "explicit_amount")
    for field in ("fee_conversion_rate", "fee_conversion_price", "fee_asset_price"):
        explicit_rate = _optional_number(details, field)
        if explicit_rate is None:
            continue
        if explicit_rate <= 0.0:
            raise FeeConversionError("fee conversion rate must be positive")
        return FeeConversion(
            asset,
            amount,
            accounting,
            amount * explicit_rate,
            explicit_rate,
            "explicit_rate",
        )
    return None


def _market_conversion(
    *,
    asset: str,
    amount: float,
    accounting: str,
    price: float,
    base_asset: str | None,
    quote_asset: str | None,
) -> FeeConversion | None:
    base = _asset(base_asset, field="base asset", optional=True)
    quote = _asset(quote_asset, field="quote asset", optional=True)
    if base and quote and asset == quote and accounting == base:
        rate = 1.0 / price
        return FeeConversion(asset, amount, accounting, amount * rate, rate, "quote_to_base")
    if base and quote and asset == base and accounting == quote:
        return FeeConversion(asset, amount, accounting, amount * price, price, "base_to_quote")
    if asset in {"USDT", "USDC", "BUSD"} and accounting in {"USDT", "USDC", "BUSD"}:
        return FeeConversion(asset, amount, accounting, amount, 1.0, "stablecoin_pair")
    return None


def _asset(value: object, *, field: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    result = str(value or "").strip().upper()
    if not result:
        if optional:
            return None
        raise FeeConversionError(f"{field} is required")
    return result


def _number(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise FeeConversionError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FeeConversionError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        bound = f" and at least {minimum:g}" if minimum is not None else ""
        raise FeeConversionError(f"{field} must be finite{bound}")
    return result


def _optional_number(metadata: Mapping[str, Any], field: str) -> float | None:
    if metadata.get(field) is None:
        return None
    return _number(metadata[field], field=field)
