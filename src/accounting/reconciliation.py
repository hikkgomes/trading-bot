"""Reconcile execution costs and product accounting evidence."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.accounting.ledger import Ledger
from src.domain.orders import Fill


@dataclass(frozen=True)
class AccountingReconciliation:
    matched: bool
    fee_difference: Decimal
    slippage_difference: Decimal
    funding_difference: Decimal
    reason_code: str


def reconcile_accounting(
    *,
    ledger: Ledger,
    fills: tuple[Fill, ...],
    expected_funding: Decimal = Decimal("0"),
    tolerance: Decimal = Decimal("0.00000001"),
) -> AccountingReconciliation:
    expected_fees = sum((Decimal(str(fill.fee)) for fill in fills), Decimal("0"))
    expected_slippage = sum(
        (Decimal(str(fill.metadata.get("slippage_cost") or 0)) for fill in fills),
        Decimal("0"),
    )
    ledger_fees = sum(
        (entry.postings.get("expense:fees", Decimal("0")) for entry in ledger.entries),
        Decimal("0"),
    )
    ledger_slippage = sum(
        (entry.postings.get("expense:slippage", Decimal("0")) for entry in ledger.entries),
        Decimal("0"),
    )
    ledger_funding = sum(
        (
            Decimal(str(entry.metadata.get("pnl_effect") or 0))
            for entry in ledger.entries
            if entry.metadata.get("kind") == "funding"
        ),
        Decimal("0"),
    )
    fee_difference = ledger_fees - expected_fees
    slippage_difference = ledger_slippage - expected_slippage
    funding_difference = ledger_funding - expected_funding
    matched = all(
        abs(value) <= tolerance
        for value in (fee_difference, slippage_difference, funding_difference)
    )
    return AccountingReconciliation(
        matched=matched,
        fee_difference=fee_difference,
        slippage_difference=slippage_difference,
        funding_difference=funding_difference,
        reason_code="accounting_reconciled" if matched else "accounting_mismatch",
    )
