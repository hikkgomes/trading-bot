"""Reconcile exchange income with fills instead of posting the same PnL twice."""

import datetime as dt
from decimal import Decimal

from sqlalchemy import select

from src.accounting.ledger import Ledger
from src.data.database import financial_application
from src.domain._codec import canonical_hash, to_primitive
from src.execution.broker import BrokerIncome
from src.services.financial_application import FinancialScope


def income_window(
    ledger: Ledger, scope: FinancialScope, created_at: list[str], at: str
) -> tuple[float, float]:
    end = dt.datetime.fromisoformat(at).timestamp()
    cursors = [
        float(entry.metadata["through"])
        for entry in ledger.entries
        if entry.metadata.get("kind") == "income_recovery_cursor"
        and entry.metadata.get("account_id") == scope.account_id
        and entry.metadata.get("environment") == scope.environment
    ]
    initial = min(
        (dt.datetime.fromisoformat(value).timestamp() for value in created_at), default=end
    )
    through = max(cursors, default=initial)
    if through > end:
        raise ValueError("income recovery clock precedes its durable cursor")
    return max(0.0, through - 60), min(end, through + 86400)


def checkpoint_income(ledger: Ledger, scope: FinancialScope, through: float, at: str) -> None:
    metadata = {
        "kind": "income_recovery_cursor",
        "account_id": scope.account_id,
        "environment": scope.environment,
        "through": through,
        "pnl_effect": "0",
    }
    identity = canonical_hash(metadata)
    if any(entry.entry_id == identity for entry in ledger.entries):
        return
    ledger.append(
        entry_id=identity,
        occurred_at=at,
        postings={"memo:income_cursor": Decimal(0), "memo:offset": Decimal(0)},
        metadata=metadata,
    )


def reconcile_income(engine, ledger: Ledger, scope: FinancialScope, income: BrokerIncome) -> bool:
    source = to_primitive(income)
    identity = canonical_hash(
        {
            "account": scope.account_id,
            "environment": scope.environment,
            "venue": scope.venue,
            "income_type": income.income_type.casefold(),
            "income_id": income.income_id,
            "symbol": income.symbol,
            "asset": income.asset,
        }
    )
    existing = next((entry for entry in ledger.entries if entry.entry_id == identity), None)
    if existing is not None:
        if existing.metadata.get("exchange_income") != source:
            raise ValueError("exchange income identity changed its financial values")
        return False
    if any(entry.entry_id == f"rest:{income.income_id}" for entry in ledger.entries):
        raise ValueError("legacy REST income requires reconciliation before new recovery")
    metadata = {
        "exchange_income": source,
        "account_id": scope.account_id,
        "environment": scope.environment,
        "income_receipt_id": identity,
    }
    at = dt.datetime.fromtimestamp(income.occurred_at, dt.UTC).isoformat()
    kind = income.income_type.casefold()
    if kind == "funding_fee":
        if income.asset != ledger.accounting_asset:
            raise ValueError("funding income must use the accounting asset")
        ledger.record_funding(
            entry_id=identity,
            amount=Decimal(str(income.amount)),
            occurred_at=at,
            attribution=metadata,
        )
    else:
        fill_id = _match_fill(engine, ledger, scope, income)
        ledger.append(
            entry_id=identity,
            occurred_at=at,
            postings={"memo:exchange_income": Decimal(0), "memo:offset": Decimal(0)},
            metadata={
                **metadata,
                "kind": "income_reconciliation",
                "pnl_effect": "0",
                "financial_fill_id": fill_id,
            },
        )
    return True


def _match_fill(engine, ledger: Ledger, scope: FinancialScope, income: BrokerIncome) -> str:
    if not income.trade_id:
        raise ValueError("fee or realised income requires an exact exchange trade identity")
    with engine.connect() as connection:
        rows = (
            connection.execute(
                select(financial_application.c.payload).where(
                    financial_application.c.account_id == scope.account_id,
                    financial_application.c.environment == scope.environment,
                    financial_application.c.venue == scope.venue,
                    financial_application.c.exchange_trade_id == income.trade_id,
                )
            )
            .scalars()
            .all()
        )
    rows = [
        row
        for row in rows
        if row["scope"]["product_id"] == scope.product_id
        and row["fill"]["instrument_id"].split(":")[2] == income.symbol
    ]
    if len(rows) != 1:
        raise ValueError("exchange income has no unique reconciled financial fill")
    fill = rows[0]["fill"]
    if income.income_type.casefold() in {"commission", "commission_fee"}:
        if fill.get("fee_asset") != income.asset:
            raise ValueError("income fee currency differs from its fill")
        expected = -Decimal(str(fill["fee"]))
    else:
        if income.asset != ledger.accounting_asset:
            raise ValueError("realised income currency differs from the ledger")
        expected = sum(
            (
                entry.postings.get("assets:cash", Decimal(0))
                for entry in ledger.entries
                if entry.entry_id.startswith(fill["fill_id"] + ":realised_pnl")
            ),
            Decimal(0),
        )
    if abs(expected - Decimal(str(income.amount))) > Decimal("0.00000001"):
        raise ValueError("exchange income does not reconcile to its financial fill")
    return str(fill["fill_id"])
