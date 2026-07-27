"""Product-specific exchange routing constraints for live execution."""

from __future__ import annotations

import re

from src.autopilot.config import ProductConfig
from src.execution.config import ExchangeConfig

ACTIVE_INCOME_FUTURES_EXCHANGES = {"binanceusdm"}
BTC_ACCUMULATION_SPOT_EXCHANGES = {"binance"}
ACTIVE_INCOME_MAX_FUTURES_LEVERAGE = 1
QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")


def split_symbol(symbol: str) -> tuple[str, str, str | None]:
    raw = re.sub(r"\s+", "", str(symbol or "").upper())
    if not raw:
        return "", "", None
    settlement = None
    pair = raw
    if ":" in pair:
        pair, settlement = pair.split(":", 1)
    if "/" in pair:
        base, quote = pair.split("/", 1)
        return base, quote, settlement
    compact = re.sub(r"[^A-Z0-9]", "", pair)
    for quote in QUOTE_ASSETS:
        if compact.endswith(quote) and len(compact) > len(quote):
            return compact[: -len(quote)], quote, settlement
    return compact, "", settlement


def validate_product_symbol_policy(product: ProductConfig) -> list[str]:
    errors: list[str] = []
    base, quote, settlement = split_symbol(product.symbol)
    if product.objective == "btc_accumulation":
        if (base, quote) != ("BTC", "USDT"):
            errors.append(
                f"{product.name}: BTC accumulation symbol must be BTC/USDT; got {product.symbol!r}."
            )
        if settlement is not None:
            errors.append(
                f"{product.name}: BTC accumulation spot symbol must not include a settlement asset."
            )
    if product.objective == "active_income":
        if not base or quote != "USDT":
            errors.append(
                f"{product.name}: active income symbol must be a USDT pair; got {product.symbol!r}."
            )
        if settlement is not None and settlement != "USDT":
            errors.append(
                f"{product.name}: active income futures settlement must be USDT; got {settlement!r}."
            )
    return errors


def validate_exchange_policy(product: ProductConfig, cfg: ExchangeConfig) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_product_symbol_policy(product))
    exchange = str(cfg.exchange).lower()
    quote_asset = str(cfg.quote_asset).upper()
    if product.objective == "active_income":
        if cfg.market_type != "futures":
            errors.append(f"{product.name}: active income live execution must use futures.")
        if exchange not in ACTIVE_INCOME_FUTURES_EXCHANGES:
            allowed = ", ".join(sorted(ACTIVE_INCOME_FUTURES_EXCHANGES))
            errors.append(
                f"{product.name}: active income live futures must use Binance USDT futures "
                f"(FUTURES_EXCHANGE={allowed}); got {cfg.exchange!r}."
            )
        if quote_asset != "USDT":
            errors.append(
                f"{product.name}: active income quote asset must be USDT; got {cfg.quote_asset!r}."
            )
        if getattr(cfg, "max_futures_leverage", None) != ACTIVE_INCOME_MAX_FUTURES_LEVERAGE:
            errors.append(
                f"{product.name}: active income futures must use "
                f"MAX_FUTURES_LEVERAGE={ACTIVE_INCOME_MAX_FUTURES_LEVERAGE}."
            )
        if str(getattr(cfg, "futures_margin_mode", "")).lower() != "isolated":
            errors.append(f"{product.name}: active income futures must use isolated margin.")
    if product.objective == "btc_accumulation":
        if cfg.market_type != "spot":
            errors.append(f"{product.name}: BTC accumulation live execution must use spot.")
        if exchange not in BTC_ACCUMULATION_SPOT_EXCHANGES:
            allowed = ", ".join(sorted(BTC_ACCUMULATION_SPOT_EXCHANGES))
            errors.append(
                f"{product.name}: BTC accumulation live spot must use Binance spot "
                f"(SPOT_EXCHANGE={allowed}); got {cfg.exchange!r}."
            )
        if quote_asset != "USDT":
            errors.append(
                f"{product.name}: BTC accumulation trades BTC/USDT, so QUOTE_ASSET must be USDT."
            )
    return errors
