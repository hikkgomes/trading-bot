"""Durable strategy ownership, separate from the account's exchange authority."""

from collections.abc import Mapping
from typing import Any

from sqlalchemy import insert, select

from src.accounting.paper_account import PaperAccountLedger
from src.data.database import account, book_event, strategy_book
from src.domain._codec import canonical_hash


def book_identity(artefact_hash: str, execution_mode: str, sleeve_id: str) -> str:
    return canonical_hash(
        {
            "artefact_hash": artefact_hash,
            "execution_mode": execution_mode,
            "sleeve_id": sleeve_id,
            "schema": "strategy_book/v1",
        }
    )


def assignment_book_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = book_identity(
        str(record["artefact_hash"]), str(record["execution_mode"]), str(record["sleeve_id"])
    )
    supplied = record["payload"].get("book_id")
    if supplied is not None and supplied != identity:
        raise ValueError("assignment book belongs to a different artefact or execution mode")
    return {
        **record["payload"],
        "book_id": identity,
        "position_portfolio_id": identity,
        **({"account_id": identity} if record["execution_mode"] == "paper" else {}),
    }


def ensure_assignment_book(connection, record, artefact):
    identity = record["payload"]["book_id"]
    existing = connection.execute(
        select(strategy_book.c.payload).where(strategy_book.c.id == identity)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    parent_account = str(artefact.get("account_id") or "")
    payload = {
        "schema": "strategy_book/v1",
        "book_id": identity,
        "product_id": record["product_id"],
        "strategy_version_id": record["strategy_version_id"],
        "artefact_hash": record["artefact_hash"],
        "execution_mode": record["execution_mode"],
        "sleeve_id": record["sleeve_id"],
        "execution_account_id": parent_account,
        "account_id": identity if record["execution_mode"] == "paper" else parent_account,
        "product_portfolio_id": record["portfolio_id"],
        "position_portfolio_id": identity,
        "created_at": record["assigned_at"],
        "instrument_ids": list(artefact["supported_instruments"]),
        "initial_capital_limit": record["capital_limit"],
        "capital_currency": record["payload"].get("capital_limit_currency", "USDT"),
    }
    connection.execute(
        insert(strategy_book).values(
            id=identity,
            product_id=record["product_id"],
            artefact_hash=record["artefact_hash"],
            strategy_version_id=record["strategy_version_id"],
            execution_mode=record["execution_mode"],
            sleeve_id=record["sleeve_id"],
            account_id=payload["account_id"],
            created_at=record["assigned_at"],
            payload=payload,
        )
    )
    template = connection.execute(
        select(account.c.payload).where(account.c.id == parent_account)
    ).scalar_one_or_none()
    if record["execution_mode"] == "paper" and template and template.get("paper_starting_balances"):
        seed_paper_book(
            connection,
            payload,
            balances=template["paper_starting_balances"],
            positions=template.get("paper_starting_positions", {}),
            accounting_asset="BTC" if template.get("market") == "spot" else "USDT",
        )
    return payload


def seed_paper_book(connection, book, *, balances, positions, accounting_asset):
    identity = "book-capital:" + book["book_id"]
    payload = {
        "kind": "paper_initial_capital",
        "book_id": book["book_id"],
        "account_id": book["account_id"],
        "accounting_asset": accounting_asset,
        "starting_balances": dict(balances),
        "starting_positions": dict(positions),
    }
    previous = connection.execute(
        select(book_event.c.payload).where(book_event.c.id == identity)
    ).scalar_one_or_none()
    if previous is None:
        connection.execute(
            insert(book_event).values(id=identity, created_at=book["created_at"], payload=payload)
        )
    elif previous != payload:
        raise ValueError("paper book initial resources are immutable across qualification windows")


def load_book(engine, book_id: str, *, connection=None):
    from src.data.transactions import connection_scope

    with connection_scope(engine, connection, write=False) as opened:
        return opened.execute(
            select(strategy_book.c.payload).where(strategy_book.c.id == book_id)
        ).scalar_one()


def paper_book_account(engine, book_id: str, *, connection=None) -> PaperAccountLedger:
    from src.data.transactions import connection_scope

    with connection_scope(engine, connection, write=False) as opened:
        book = load_book(engine, book_id, connection=opened)
        if book["execution_mode"] != "paper":
            raise ValueError("a live book cannot use simulated funds")
        capital = opened.execute(
            select(book_event.c.payload).where(book_event.c.id == "book-capital:" + book_id)
        ).scalar_one_or_none()
    if capital is None:
        raise ValueError("paper book has no declared initial resources")
    return PaperAccountLedger(
        engine=engine,
        account_id=book["account_id"],
        product_id=book["product_id"],
        accounting_asset=capital["accounting_asset"],
        starting_balances=capital["starting_balances"],
        starting_positions=capital["starting_positions"],
        book_id=book_id,
        connection=connection,
    )


def paper_account_for_order(engine, order, templates, *, product_id: str, connection=None):
    metadata = order.metadata.get("target_metadata", order.metadata)
    if metadata.get("book_id"):
        return paper_book_account(engine, str(metadata["book_id"]), connection=connection)
    return templates.get(product_id)


def position_scope(assignment: Mapping[str, Any]) -> str:
    return str(
        assignment.get("payload", {}).get("position_portfolio_id") or assignment["portfolio_id"]
    )


def active_paper_book_views(engine, products, accounts, *, at: str):
    from src.research.canonical import SqlActiveStrategyAssignmentRepository

    seen = set()
    for product_id, product in products.items():
        for assignment in SqlActiveStrategyAssignmentRepository(engine).active_assignments(
            product_id, at=at
        ):
            identity = assignment["payload"].get("book_id")
            if assignment["execution_mode"] != "paper" or not identity or identity in seen:
                continue
            seen.add(identity)
            yield paper_book_view(engine, assignment, product, accounts)


def paper_book_view(engine, assignment, product, accounts):
    identity = assignment["payload"]["book_id"]
    book = load_book(engine, identity)
    ledger = paper_book_account(engine, identity)
    template = accounts[book["execution_account_id"]]
    scoped_product = {
        **product,
        "book_id": identity,
        "portfolio_id": identity,
        "account_id": book["account_id"],
        "execution_mode": "paper",
        "instrument_ids": book["instrument_ids"],
    }
    scoped_account = {
        **template,
        "account_id": book["account_id"],
        "environment": "paper",
        "paper_starting_balances": ledger.starting_balances,
        "paper_starting_positions": ledger.starting_positions,
    }
    return book, scoped_product, scoped_account


def initialise_paper_book(engine, connection, book, assignment_id):
    from src.domain.positions import Position, PositionStatus
    from src.execution.position_manager import SqlPositionStore

    ledger = paper_book_account(engine, book["book_id"], connection=connection)
    existing = ledger.state()
    state = ledger.ensure_seed(observed_at=book["created_at"])
    if existing is None:
        store = SqlPositionStore(engine, connection=connection)
        for instrument_id, quantity in state.positions.items():
            store.save(
                Position(
                    portfolio_id=book["book_id"],
                    instrument_id=instrument_id,
                    quantity=quantity,
                    average_entry_price=0.0,
                    status=PositionStatus.OPEN,
                    updated_at=book["created_at"],
                    strategy_contributions={book["strategy_version_id"]: 1.0},
                    metadata={
                        "book_id": book["book_id"],
                        "assignment_id": assignment_id,
                        "source": "paper_capital_seed",
                    },
                )
            )
    return ledger


def book_balances(engine, book_id: str, *, at: str | None = None, connection=None):
    return book_balance_snapshot(engine, book_id, at=at, connection=connection)[0]


def book_financial_events(engine, book_id, *, start=None, at=None, connection=None):
    from src.data.transactions import connection_scope

    with connection_scope(engine, connection, write=False) as opened:
        return tuple(
            opened.execute(
                select(book_event.c.id, book_event.c.created_at, book_event.c.payload)
                .where(
                    book_event.c.payload["book_id"].as_string() == book_id,
                    *([book_event.c.created_at > start] if start is not None else []),
                    *([book_event.c.created_at <= at] if at is not None else []),
                )
                .order_by(book_event.c.created_at, book_event.c.id)
            ).mappings()
        )


def book_event_pnl(payload):
    from decimal import Decimal

    return (
        Decimal(str(payload.get("realised_pnl", 0)))
        + Decimal(str(payload.get("amount", 0)))
        - Decimal(str(payload.get("fee_accounting_amount", 0)))
    )


def book_foreign_fee_cost(engine, book_id, *, at=None, connection=None):
    """Value fees without inventing a conversion in the raw asset ledger."""
    return sum(
        float(row["payload"]["fee_accounting_amount"])
        for row in book_financial_events(engine, book_id, at=at, connection=connection)
        if row["payload"].get("accounting_asset") == "USDT"
        and row["payload"].get("fee_asset") not in (None, "USDT")
    )


def book_balance_snapshot(engine, book_id: str, *, at: str | None = None, connection=None):
    from src.data.transactions import connection_scope

    with connection_scope(engine, connection, write=False) as opened:
        rows = list(
            opened.execute(
                select(book_event.c.created_at, book_event.c.payload)
                .where(
                    book_event.c.payload["book_id"].as_string() == book_id,
                    *([book_event.c.created_at <= at] if at is not None else []),
                )
                .order_by(book_event.c.created_at, book_event.c.id)
            ).all()
        )
    balances: dict[str, float] = {}
    for _created_at, row in rows:
        if row.get("kind") == "paper_initial_capital":
            balances.update(
                {asset: float(value) for asset, value in row["starting_balances"].items()}
            )
        for asset, amount in row.get("balance_changes", {}).items():
            balances[asset] = balances.get(asset, 0.0) + float(amount)
    return balances, rows[-1][0] if rows else None
