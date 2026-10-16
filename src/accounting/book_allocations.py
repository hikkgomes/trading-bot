"""Conserve external fills and costs while projecting each strategy's own book."""

import datetime as dt
from dataclasses import replace
from decimal import ROUND_DOWN, Decimal, localcontext

from sqlalchemy import insert, or_, select

from src.accounting.execution import realised_fill_pnl
from src.accounting.fees import convert_fee, instrument_asset
from src.data.database import (
    book_event,
    book_fill_allocation,
    financial_application,
    funding_entry,
    job,
    strategy_book,
)
from src.data.database import fill as fill_table
from src.domain._codec import canonical_hash, timestamp
from src.execution.stops import live_protected_instruments
from src.portfolio.book_plans import allocate_book_fill
from src.portfolio.books import load_book


def record_book_allocations(engine, connection, scope, order, fill, positions, previous):
    metadata = order.metadata.get("target_metadata", order.metadata)
    contributions = metadata.get("book_contributions") or order.metadata.get("book_contributions")
    sign = 1 if fill.side.value == "buy" else -1
    if not contributions and metadata.get("book_id"):
        contributions = [
            {
                "book_id": metadata["book_id"],
                "assignment_id": metadata.get("assignment_id"),
                "signed_quantity": sign * order.quantity,
            }
        ]
    if not contributions:
        return
    allocations = allocate_book_fill(sign * fill.quantity, fill.fee, contributions)
    for row in allocations:
        book = load_book(engine, row["book_id"], connection=connection)
        _assert_book_scope(book, scope, order, row)
        identity = canonical_hash({"fill_id": fill.fill_id, "book_id": book["book_id"]})
        allocated = replace(
            fill,
            fill_id=identity,
            quantity=float(abs(row["signed_quantity"])),
            fee=float(row["fee"]),
            metadata={
                **fill.metadata,
                "book_id": book["book_id"],
                "source_fill_id": fill.fill_id,
                "base_fee_quantity": float(row["fee"])
                if fill.metadata.get("base_fee_quantity")
                else 0,
            },
        )
        before = (
            previous
            if order.portfolio_id == book["position_portfolio_id"]
            else positions.get(book["position_portfolio_id"], fill.instrument_id)
        )
        if order.portfolio_id != book["position_portfolio_id"]:
            positions.apply_fill(
                book["position_portfolio_id"],
                allocated,
                contributions={book["strategy_version_id"]: 1.0},
                assignment_id=row.get("assignment_id"),
            )
        fee_metadata = {**metadata, **order.metadata, **fill.metadata}
        if fill.fee and "fee_in_accounting_asset" in fee_metadata:
            fee_metadata["fee_in_accounting_asset"] = (
                float(fee_metadata["fee_in_accounting_asset"]) * allocated.fee / fill.fee
            )
        converted = convert_fee(
            amount=allocated.fee,
            fee_asset=allocated.fee_asset,
            accounting_asset=scope.accounting_asset,
            trade_price=fill.price,
            base_asset=instrument_asset(fill.instrument_id, "base"),
            quote_asset=instrument_asset(fill.instrument_id, "quote"),
            metadata=fee_metadata,
        )
        payload = {
            "kind": "external_fill_allocation",
            "book_id": book["book_id"],
            "fill_id": fill.fill_id,
            "instrument_id": fill.instrument_id,
            "assignment_id": row.get("assignment_id"),
            "signed_quantity": str(row["signed_quantity"]),
            "price": fill.price,
            "fee": str(row["fee"]),
            "fee_asset": fill.fee_asset or scope.accounting_asset,
            "fee_accounting_amount": str(converted.accounting_amount),
            "realised_pnl": str(realised_fill_pnl(before, allocated, scope.accounting_asset)),
            "accounting_asset": scope.accounting_asset,
            "environment": scope.environment,
        }
        payload["balance_changes"] = _balance_changes(allocated, before, scope.accounting_asset)
        connection.execute(
            insert(book_fill_allocation).values(
                id=identity,
                fill_id=fill.fill_id,
                book_id=book["book_id"],
                signed_quantity=row["signed_quantity"],
                fee=row["fee"],
                fee_asset=payload["fee_asset"],
                created_at=fill.occurred_at,
                payload=payload,
            )
        )
        connection.execute(
            insert(book_event).values(
                id="book-fill:" + identity, created_at=fill.occurred_at, payload=payload
            )
        )
    if sum(row["signed_quantity"] for row in allocations) != Decimal(
        str(sign * fill.quantity)
    ) or sum(row["fee"] for row in allocations) != Decimal(str(fill.fee)):
        raise ValueError("book fill allocation does not conserve the external financial event")


def _assert_book_scope(book, scope, order, contribution):
    expected = "paper" if scope.environment == "paper" else "live"
    if (
        book["product_id"] != scope.product_id
        or book["account_id"] != scope.account_id
        or book["execution_mode"] != expected
    ):
        raise ValueError("fill allocation crosses a book account or environment")
    if (
        contribution.get("strategy_version_id", book["strategy_version_id"])
        != book["strategy_version_id"]
    ):
        raise ValueError("fill allocation strategy does not own the book")
    if order.instrument_id not in book["instrument_ids"]:
        raise ValueError("fill allocation instrument is outside the book's declared vector")


def record_book_funding(connection, scope, income, receipt_id, at):
    """Attribute actual exchange income to inventory held before settlement."""
    symbol = f"{scope.venue}:futures:{income.symbol}:USDT"
    books = tuple(
        connection.execute(
            select(strategy_book.c.payload).where(
                strategy_book.c.account_id == scope.account_id,
                strategy_book.c.product_id == scope.product_id,
                strategy_book.c.execution_mode == "live",
                strategy_book.c.created_at < at,
            )
        ).scalars()
    )
    books = tuple(book for book in books if symbol in book["instrument_ids"])
    if not books:
        return
    quantities = _book_settlement_quantities(
        connection, [book["book_id"] for book in books], symbol, at
    )
    actual = _account_settlement_quantity(connection, scope, symbol, at)
    native = (
        _native_zero_net_funding(connection, scope, symbol, at)
        if not actual and any(quantities.values())
        else None
    )
    with localcontext() as context:
        context.prec = _funding_precision(
            [*quantities.values(), actual, Decimal(str(income.amount))]
        )
        if sum(quantities.values()) != actual:
            raise ValueError("funding inventory has unallocated account exposure")
        allocated = _funding_amounts(Decimal(str(income.amount)), quantities, native=native)
        for book_id, amount in allocated.items():
            payload = {
                "kind": "funding_allocation",
                "book_id": book_id,
                "instrument_id": symbol,
                "income_receipt_id": receipt_id,
                "account_id": scope.account_id,
                "environment": scope.environment,
                "accounting_asset": income.asset,
                "quantity_at_settlement": str(quantities[book_id]),
                "amount": str(amount),
                "balance_changes": {income.asset: str(amount)},
                "native_settlement": native,
                "exchange_income": True,
            }
            _store_live_funding(connection, payload, at)


def _store_live_funding(connection, payload, at):
    keys = ("book_id", "instrument_id", "account_id", "environment")
    identity = "book-funding:" + canonical_hash(
        {**{key: payload[key] for key in keys}, "settlement_at": at}
    )
    previous = connection.execute(
        select(book_event.c.payload).where(book_event.c.id == identity)
    ).scalar_one_or_none()
    if previous is not None:
        financial = (*keys, "quantity_at_settlement", "amount", "accounting_asset")
        if any(previous[key] != payload[key] for key in financial):
            raise ValueError("native funding receipt conflicts with allocated settlement")
        return 0
    connection.execute(insert(book_event).values(id=identity, created_at=at, payload=payload))
    return 1


def _funding_precision(values):
    if any(not value.is_finite() for value in values):
        raise ValueError("funding amounts and inventory must be finite")
    return max(
        38,
        max((value.adjusted() for value in values), default=0)
        - min((-18, *(int(value.as_tuple().exponent) for value in values)))
        + len(str(len(values)))
        + 8,
    )


def _funding_amounts(amount, quantities, *, native=None):
    net = sum(quantities.values())
    held = {book: quantity for book, quantity in sorted(quantities.items()) if quantity}
    if not net:
        if amount or (held and native is None):
            raise ValueError("zero-net book funding requires native settlement rate and mark")
        if not held:
            return {}
    unit = amount / net if net else -Decimal(native["funding_rate"]) * Decimal(native["mark_price"])
    quantum = Decimal(1).scaleb(min(-18, int(amount.as_tuple().exponent)))
    allocated, result = Decimal(0), {}
    for index, (book, quantity) in enumerate(held.items()):
        value = (
            amount - allocated
            if index == len(held) - 1
            else (unit * quantity).quantize(quantum, rounding=ROUND_DOWN)
        )
        result[book] = value
        allocated += value
    if sum(result.values()) != amount:
        raise ValueError("book funding does not conserve actual exchange income")
    return result


def _native_zero_net_funding(connection, scope, symbol, at):
    rows = tuple(
        connection.execute(
            select(funding_entry.c.id, funding_entry.c.payload).where(
                funding_entry.c.payload["product_id"].as_string() == scope.product_id,
                funding_entry.c.payload["instrument_id"].as_string() == symbol,
                funding_entry.c.payload["kind"].as_string() == "historical_funding",
                funding_entry.c.payload["source"].as_string() == "binance_public_funding_rate",
            )
        ).mappings()
    )
    matches = [
        row
        for row in rows
        if row["payload"].get("source_event_id")
        and dt.datetime.fromisoformat(row["payload"]["funding_time"])
        == dt.datetime.fromisoformat(at)
    ]
    values = {
        (Decimal(str(row["payload"]["funding_rate"])), Decimal(str(row["payload"]["mark_price"])))
        for row in matches
    }
    if len(values) != 1:
        raise ValueError("zero-net funding has no unique native settlement rate and mark")
    rate, mark = values.pop()
    if not rate.is_finite() or not mark.is_finite() or mark <= 0:
        raise ValueError("native funding rate or mark is invalid")
    return {
        "funding_rate": str(rate),
        "mark_price": str(mark),
        "source_record_ids": sorted(row["id"] for row in matches),
    }


class PaperFundingPending(RuntimeError):
    """Earlier simulated executions must settle before charging funding."""


def settle_public_paper_funding(engine, record_id, *, now):
    from src.services.financial_application import FinancialApplication, FinancialScope

    with engine.begin() as connection:
        native = connection.execute(
            select(funding_entry.c.payload).where(funding_entry.c.id == record_id)
        ).scalar_one()
        if (
            native.get("kind") != "historical_funding"
            or not native.get("source_event_id")
            or native.get("source") != "binance_public_funding_rate"
        ):
            raise ValueError("paper funding requires a canonical final native settlement")
        at = timestamp(native["funding_time"], field="funding settlement", preserve_subseconds=True)
        if at > timestamp(now, field="funding processing time", preserve_subseconds=True):
            raise ValueError("paper funding cannot consume a future settlement")
        books = tuple(
            connection.execute(
                select(strategy_book.c.payload).where(
                    strategy_book.c.product_id == native["product_id"],
                    strategy_book.c.execution_mode == "paper",
                    strategy_book.c.created_at < at,
                )
            ).scalars()
        )
        applied = 0
        for book in books:
            if native["instrument_id"] not in book["instrument_ids"]:
                continue
            scope = FinancialScope(
                native["product_id"], book["account_id"], "paper", "binance", "USDT"
            )
            FinancialApplication(engine, scope)._lock(connection)
            applied += _settle_paper_book(engine, connection, book, native, record_id, at)
        applied += _settle_zero_net_live_books(engine, connection, native, at)
        return applied


def _settle_zero_net_live_books(engine, connection, native, at):
    from src.data.database import account
    from src.services.financial_application import FinancialApplication, FinancialScope

    books = tuple(
        connection.execute(
            select(strategy_book.c.payload).where(
                strategy_book.c.product_id == native["product_id"],
                strategy_book.c.execution_mode == "live",
                strategy_book.c.created_at < at,
            )
        ).scalars()
    )
    books = tuple(book for book in books if native["instrument_id"] in book["instrument_ids"])
    applied = 0
    for account_id in sorted({book["account_id"] for book in books}):
        configuration = connection.execute(
            select(account.c.payload).where(account.c.id == account_id)
        ).scalar_one()
        scope = FinancialScope(
            native["product_id"],
            account_id,
            configuration["environment"],
            configuration["venue"],
            "USDT",
        )
        FinancialApplication(engine, scope)._lock(connection)
        selected = [book for book in books if book["account_id"] == account_id]
        applied += _settle_zero_net_account(engine, connection, scope, selected, native, at)
    return applied


def _settle_zero_net_account(engine, connection, scope, books, native, at):
    from src.execution.order_manager import OrderManager, SqlOrderStore

    symbol = native["instrument_id"]
    portfolios = {book["product_portfolio_id"] for book in books}
    orders = OrderManager(SqlOrderStore(engine, connection=connection))
    if any(
        order.portfolio_id in portfolios and not order.is_terminal and order.created_at < at
        for order in orders.all()
    ):
        raise PaperFundingPending(
            "account orders before the internal funding settlement remain unresolved"
        )
    if _account_settlement_quantity(connection, scope, symbol, at):
        return 0  # Net exposure is charged only from an authenticated income receipt.
    quantities = _book_settlement_quantities(
        connection, [book["book_id"] for book in books], symbol, at
    )
    if sum(quantities.values()):
        raise ValueError("zero-net funding has unallocated account inventory")
    settlement = _native_zero_net_funding(connection, scope, symbol, at)
    amounts = _funding_amounts(Decimal(0), quantities, native=settlement)
    return sum(
        _store_live_funding(
            connection,
            {
                "kind": "funding_allocation",
                "book_id": book_id,
                "instrument_id": symbol,
                "account_id": scope.account_id,
                "environment": scope.environment,
                "accounting_asset": "USDT",
                "quantity_at_settlement": str(quantities[book_id]),
                "amount": str(amount),
                "balance_changes": {"USDT": str(amount)},
                "native_settlement": settlement,
                "source_event_id": native["source_event_id"],
                "exchange_income": False,
            },
            at,
        )
        for book_id, amount in amounts.items()
    )


def _settle_paper_book(engine, connection, book, native, record_id, at):
    from src.accounting.ledger import Ledger, SqlLedgerStore
    from src.execution.order_manager import OrderManager, SqlOrderStore
    from src.portfolio.books import paper_book_account

    _assert_paper_pipeline_settled(connection, book, native["instrument_id"], at)
    orders = OrderManager(SqlOrderStore(engine, connection=connection))
    if any(
        order.portfolio_id == book["position_portfolio_id"]
        and not order.is_terminal
        and order.created_at < at
        for order in orders.all()
    ):
        raise PaperFundingPending("paper orders before settlement are still pending")
    account = paper_book_account(engine, book["book_id"], connection=connection)
    symbol = native["instrument_id"]
    quantity = _book_settlement_quantities(connection, [book["book_id"]], symbol, at)[
        book["book_id"]
    ]
    quantity += Decimal(str(account.starting_positions.get(symbol, 0)))
    rate, mark = Decimal(str(native["funding_rate"])), Decimal(str(native["mark_price"]))
    if not rate.is_finite() or not mark.is_finite() or mark <= 0:
        raise ValueError("paper funding requires a finite final rate and positive mark")
    with localcontext() as context:
        context.prec = _funding_precision([quantity, rate, mark])
        amount = -quantity * rate * mark
    identity = "paper-funding:" + canonical_hash(
        {"book_id": book["book_id"], "source_event_id": native["source_event_id"]}
    )
    payload = {
        "kind": "funding_allocation",
        "book_id": book["book_id"],
        "instrument_id": symbol,
        "account_id": book["account_id"],
        "environment": "paper",
        "accounting_asset": "USDT",
        "quantity_at_settlement": str(quantity),
        "amount": str(amount),
        "balance_changes": {"USDT": str(amount)},
        "source_event_id": native["source_event_id"],
        "funding_rate": str(rate),
        "mark_price": str(mark),
        "exchange_income": False,
    }
    previous = connection.execute(
        select(book_event.c.payload).where(book_event.c.id == identity)
    ).scalar_one_or_none()
    if previous is not None:
        if previous != payload:
            raise ValueError("paper funding identity changed its native facts or settled inventory")
        return 0
    Ledger(
        product_id=book["product_id"],
        accounting_asset="USDT",
        store=SqlLedgerStore(engine, product_id=book["product_id"], connection=connection),
    ).record_funding(
        entry_id=identity,
        amount=amount,
        occurred_at=at,
        attribution={
            **payload,
            "kind": "funding",
            "strategy_version_id": book["strategy_version_id"],
            "source_record_id": record_id,
        },
    )
    connection.execute(insert(book_event).values(id=identity, created_at=at, payload=payload))
    return 1


def _assert_paper_pipeline_settled(connection, book, symbol, at):
    pending = connection.execute(
        select(job.c.id)
        .where(
            job.c.state.in_(["pending", "running"]),
            job.c.name.in_(
                [
                    "live_feature_calculation",
                    "strategy_evaluation",
                    "portfolio_target_build",
                    "risk_assessment",
                    "execute_targets",
                ]
            ),
            or_(
                job.c.payload["product_id"].as_string() == book["product_id"],
                job.c.payload["instrument_id"].as_string() == symbol,
            ),
            or_(
                job.c.payload["evaluated_at"].as_string() < at,
                job.c.payload["availability_time"].as_string() < at,
            ),
        )
        .limit(1)
    ).first()
    if pending is not None:
        raise PaperFundingPending("paper decision work before settlement is still pending")


def _book_settlement_quantities(connection, books, symbol, at):
    quantities = dict.fromkeys(books, Decimal(0))
    rows = connection.execute(
        select(book_fill_allocation.c.book_id, book_fill_allocation.c.signed_quantity)
        .join(fill_table, fill_table.c.id == book_fill_allocation.c.fill_id)
        .where(
            book_fill_allocation.c.book_id.in_(books),
            book_fill_allocation.c.created_at < at,
            fill_table.c.payload["instrument_id"].as_string() == symbol,
        )
    ).all()
    transfers = tuple(
        connection.execute(
            select(book_event.c.payload).where(
                book_event.c.payload["book_id"].as_string().in_(books),
                book_event.c.created_at < at,
                book_event.c.payload["kind"].as_string() == "internal_netting",
                book_event.c.payload["instrument_id"].as_string() == symbol,
            )
        ).scalars()
    )
    changes = [(book, Decimal(str(value))) for book, value in rows]
    changes.extend((row["book_id"], Decimal(str(row["signed_quantity"]))) for row in transfers)
    with localcontext() as context:
        context.prec = _funding_precision([value for _book, value in changes])
        for book, quantity in changes:
            quantities[book] += quantity
    return quantities


def _account_settlement_quantity(connection, scope, symbol, at):
    rows = tuple(
        connection.execute(
            select(financial_application.c.payload).where(
                financial_application.c.account_id == scope.account_id,
                financial_application.c.environment == scope.environment,
                financial_application.c.instrument_id == symbol,
                financial_application.c.created_at < at,
            )
        ).scalars()
    )
    quantities = [
        Decimal(str(row["fill"]["quantity"])) * (1 if row["fill"]["side"] == "buy" else -1)
        for row in rows
    ]
    with localcontext() as context:
        context.prec = _funding_precision(quantities)
        return sum(quantities, Decimal(0))


def _balance_changes(fill, previous, accounting_asset):
    quote = instrument_asset(fill.instrument_id, "quote")
    fee_asset = fill.fee_asset or accounting_asset
    if ":spot:" in fill.instrument_id:
        base = instrument_asset(fill.instrument_id, "base")
        signed = fill.quantity if fill.side.value == "buy" else -fill.quantity
        changes = {base: signed, quote: -signed * fill.price}
    else:
        changes = {quote: float(realised_fill_pnl(previous, fill, "USDT"))}
    changes[fee_asset] = changes.get(fee_asset, 0.0) - fill.fee
    return changes


def apply_internal_transfers(engine, connection, plan, positions):
    from src.domain.orders import Fill, OrderSide

    for index, transfer in enumerate(plan["internal_transfers"]):
        source_book = load_book(engine, transfer["from_book"], connection=connection)
        if transfer["instrument_id"] in live_protected_instruments(
            engine, source_book["product_portfolio_id"], connection=connection
        ):
            raise ValueError("native protection ownership requires closing before book transfer")
        for key, side in (("from_book", OrderSide.SELL), ("to_book", OrderSide.BUY)):
            book = load_book(engine, transfer[key], connection=connection)
            identity = canonical_hash(
                {"plan": plan["plan_id"], "transfer": index, "book_id": book["book_id"]}
            )
            before = positions.get(book["position_portfolio_id"], transfer["instrument_id"])
            logical = Fill(
                fill_id=identity,
                order_id=plan["plan_id"],
                instrument_id=transfer["instrument_id"],
                side=side,
                quantity=transfer["quantity"],
                price=transfer["price"],
                fee=0,
                fee_asset="USDT",
                occurred_at=plan["decision_at"],
                metadata={
                    "source": "internal_netting",
                    "book_id": book["book_id"],
                    "reference_price": transfer["price"],
                },
            )
            owner = plan["books"][book["book_id"]]
            positions.apply_fill(
                book["position_portfolio_id"],
                logical,
                contributions={book["strategy_version_id"]: 1.0},
                assignment_id=owner["assignment_id"],
            )
            asset = "BTC" if book["product_id"] == "btc_accumulation" else "USDT"
            payload = {
                "kind": "internal_netting",
                "book_id": book["book_id"],
                "plan_id": plan["plan_id"],
                "instrument_id": transfer["instrument_id"],
                "signed_quantity": transfer["quantity"] * (1 if side is OrderSide.BUY else -1),
                "price": transfer["price"],
                "realised_pnl": str(realised_fill_pnl(before, logical, asset)),
                "balance_changes": _balance_changes(logical, before, asset),
                "exchange_fill": False,
            }
            connection.execute(
                insert(book_event).values(
                    id=identity, created_at=plan["decision_at"], payload=payload
                )
            )
