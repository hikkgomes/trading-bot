"""PnL attribution views over immutable ledger entries."""

from __future__ import annotations

from decimal import Decimal

from src.accounting.ledger import Ledger


def attribution_cube(ledger: Ledger) -> dict[str, dict[str, Decimal]]:
    return {
        dimension: ledger.attribution(dimension)
        for dimension in ("strategy", "symbol", "sleeve", "product", "regime")
    }
