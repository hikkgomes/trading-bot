import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from sqlalchemy import func, select

from src.accounting.ledger import SqlLedgerStore
from src.data.database import (
    PlatformDatabase,
    accounting_entry,
    fill,
    financial_application,
    financial_outbox,
    position,
)
from src.domain.orders import Fill, OrderIntent, OrderSide, OrderStatus, OrderType
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.services.financial_application import FinancialApplication, FinancialScope

NOW = "2026-09-07T00:00:00+00:00"


def _setup(tmp_path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'financial.sqlite3'}")
    database.create_schema()
    manager = OrderManager(SqlOrderStore(database.engine))
    order = OrderIntent(
        "order",
        "portfolio",
        "binance:futures:BTCUSDT:USDT",
        OrderSide.BUY,
        2,
        OrderType.MARKET,
        created_at=NOW,
        metadata={
            "reference_price": 100,
            "target_metadata": {"account_id": "account", "execution_environment": "paper"},
        },
    )
    manager.create(order)
    execution = Fill(
        "transport-id",
        order.order_id,
        order.instrument_id,
        order.side,
        1,
        100,
        1,
        NOW,
        fee_asset="USDT",
        metadata={"trade_id": "trade-1"},
    )
    scope = FinancialScope("active_income", "account", "paper", "binance", "USDT")
    return database, manager, execution, scope


@pytest.mark.parametrize("boundary", ["order_and_fill", "position", "accounting", "outbox"])
def test_failure_at_each_projection_boundary_rolls_back_every_effect(tmp_path, boundary):
    database, manager, execution, scope = _setup(tmp_path)

    def fail(stage):
        if stage == boundary:
            raise RuntimeError("injected process failure")

    with pytest.raises(RuntimeError, match="injected"):
        FinancialApplication(database.engine, scope, fault=fail).apply_fill(execution)
    manager.reload()
    assert manager.get("order").filled_quantity == 0
    assert manager.get("order").status is OrderStatus.CREATED
    with database.engine.connect() as connection:
        for table in (fill, position, accounting_entry, financial_application, financial_outbox):
            assert connection.execute(select(func.count()).select_from(table)).scalar_one() == 0
    recovered = FinancialApplication(database.engine, scope).apply_fill(execution)
    assert recovered.position.quantity == 1
    database.dispose()


def test_replay_and_duplicate_transport_produce_one_economic_effect(tmp_path):
    database, manager, execution, scope = _setup(tmp_path)
    first = FinancialApplication(database.engine, scope).apply_fill(execution)
    duplicate = FinancialApplication(database.engine, scope).apply_fill(
        replace(execution, fill_id="other-transport")
    )
    assert first.applied and not duplicate.applied
    assert duplicate.position.quantity == 1
    assert duplicate.order.filled_quantity == 1
    entries = SqlLedgerStore(database.engine, product_id="active_income").read()
    assert sum(entry.postings.get("expense:fees", 0) for entry in entries) == 1
    assert (
        PositionManager(SqlPositionStore(database.engine))
        .get("portfolio", execution.instrument_id)
        .quantity
        == 1
    )
    database.dispose()


def test_fill_position_and_ledger_keep_subsecond_financial_ordering(tmp_path):
    database, _, execution, scope = _setup(tmp_path)
    instant = "2026-09-07T00:00:00.013000+00:00"
    execution = replace(execution, occurred_at=instant)
    applied = FinancialApplication(database.engine, scope).apply_fill(execution)
    assert applied.fill.occurred_at == applied.position.updated_at == instant
    with database.engine.connect() as connection:
        assert (
            connection.execute(
                select(fill.c.id).where(fill.c.created_at <= "2026-09-07T00:00:00.010000+00:00")
            ).first()
            is None
        )
        assert connection.execute(select(fill.c.created_at)).scalar_one() == instant
        assert connection.execute(select(position.c.created_at)).scalar_one() == instant
    assert {
        entry.occurred_at
        for entry in SqlLedgerStore(database.engine, product_id="active_income").read()
    } == {instant}
    database.dispose()


def test_late_partial_fill_keeps_cancelled_remainder_closed(tmp_path):
    database, manager, execution, scope = _setup(tmp_path)
    manager.persist_for_submission("order")
    manager.submitted("order")
    manager.acknowledged("order")
    manager.request_cancel("order")
    manager.cancelled("order")
    result = FinancialApplication(database.engine, scope).apply_fill(execution)
    assert result.order.status is OrderStatus.CANCELLED
    assert result.order.remaining_quantity == 1
    assert result.position.quantity == 1
    database.dispose()


@pytest.mark.parametrize("price,shortfall", [(102, 2), (98, -2)])
def test_transport_cannot_replace_persisted_execution_reference(tmp_path, price, shortfall):
    database, _, execution, scope = _setup(tmp_path)
    execution = replace(
        execution,
        price=price,
        metadata={"trade_id": "trade-1", "reference_price": price, "slippage_cost": 0},
    )
    result = FinancialApplication(database.engine, scope).apply_fill(execution)
    assert result.fill.metadata["reference_price"] == 100
    assert result.fill.metadata["slippage_cost"] == shortfall
    assert result.position.average_entry_price == price
    entries = SqlLedgerStore(database.engine, product_id="active_income").read()
    memo = next(entry for entry in entries if entry.metadata.get("kind") == "slippage")
    assert float(memo.metadata["cost"]) == shortfall
    assert all(value == 0 for value in memo.postings.values())
    database.dispose()


def test_missing_order_reference_is_unknown_not_zero_shortfall(tmp_path):
    database, manager, execution, scope = _setup(tmp_path)
    order = replace(manager.get("order"), order_id="no-reference", metadata={})
    manager.create(order)
    result = FinancialApplication(database.engine, scope).apply_fill(
        replace(execution, order_id=order.order_id, metadata={"trade_id": "trade-2"})
    )
    assert result.fill.metadata["reference_price"] is None
    assert result.fill.metadata["slippage_cost"] is None
    assert result.fill.metadata["reference_source"] == "unavailable"
    database.dispose()


def test_wrong_account_cannot_apply_a_fill_to_another_accounts_order(tmp_path):
    database, manager, execution, scope = _setup(tmp_path)
    with pytest.raises(ValueError, match="account"):
        FinancialApplication(database.engine, replace(scope, account_id="other")).apply_fill(
            execution
        )
    assert manager.get("order").filled_quantity == 0
    database.dispose()


@pytest.mark.skipif(
    not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith("postgresql"),
    reason="requires isolated PostgreSQL",
)
def test_two_postgres_workers_apply_the_same_fill_once():
    database = PlatformDatabase(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    database.assert_migrated()
    token = uuid.uuid4().hex
    account, portfolio, order_id = f"account-{token}", f"portfolio-{token}", f"order-{token}"
    manager = OrderManager(SqlOrderStore(database.engine))
    order = OrderIntent(
        order_id,
        portfolio,
        "binance:futures:BTCUSDT:USDT",
        OrderSide.BUY,
        2,
        OrderType.MARKET,
        created_at=NOW,
        metadata={"target_metadata": {"account_id": account, "execution_environment": "paper"}},
    )
    manager.create(order)
    execution = Fill(
        "transport",
        order_id,
        order.instrument_id,
        order.side,
        1,
        100,
        1,
        NOW,
        fee_asset="USDT",
        metadata={"trade_id": "1"},
    )
    scope = FinancialScope("active_income", account, "paper", "binance", "USDT")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(FinancialApplication(database.engine, scope).apply_fill, execution)
            for _ in range(2)
        ]
        results = [future.result(timeout=30) for future in futures]
    assert sorted(result.applied for result in results) == [False, True]
    assert (
        PositionManager(SqlPositionStore(database.engine))
        .get(portfolio, order.instrument_id)
        .quantity
        == 1
    )
    manager.reload()
    assert manager.get(order_id).filled_quantity == 1
    with database.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count())
                .select_from(financial_application)
                .where(financial_application.c.account_id == account)
            ).scalar_one()
            == 1
        )
    database.dispose()


@pytest.mark.parametrize("postgres", [False, True])
def test_partial_fill_book_allocation_is_atomic_and_conserves_fees(tmp_path, postgres):
    from decimal import Decimal

    from sqlalchemy import insert

    from src.data.database import book_event, book_fill_allocation

    if postgres and not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith(
        "postgresql"
    ):
        pytest.skip("requires isolated PostgreSQL")
    database = PlatformDatabase(
        os.environ["TRADING_PLATFORM_DATABASE_URL"]
        if postgres
        else f"sqlite+pysqlite:///{tmp_path / 'books.sqlite3'}"
    )
    if postgres:
        database.assert_migrated()
    else:
        database.create_schema()
    token = uuid.uuid4().hex
    account = f"allocation-{token}"
    books = _allocation_books(database, account, token)
    manager = OrderManager(SqlOrderStore(database.engine))
    order = OrderIntent(
        f"book-order-{token}",
        account,
        "binance:futures:BTCUSDT:USDT",
        OrderSide.BUY,
        3,
        OrderType.MARKET,
        created_at=NOW,
        metadata={
            "target_metadata": {
                "account_id": account,
                "execution_environment": "testnet",
                "book_contributions": [
                    {"book_id": book, "signed_quantity": index + 1}
                    for index, book in enumerate(books)
                ],
            }
        },
    )
    manager.create(order)
    manager.persist_for_submission(order.order_id)
    manager.submitted(order.order_id)
    manager.acknowledged(order.order_id)
    execution = Fill(
        "transport",
        order.order_id,
        order.instrument_id,
        OrderSide.BUY,
        1,
        100,
        0.01,
        NOW,
        fee_asset="USDT",
        metadata={"trade_id": "1"},
    )
    scope = FinancialScope("active_income", account, "testnet", "binance", "USDT")

    def fail(stage):
        if stage == "book_allocations":
            raise RuntimeError("injected allocation failure")

    with pytest.raises(RuntimeError, match="injected allocation"):
        FinancialApplication(database.engine, scope, fault=fail).apply_fill(execution)
    with database.engine.connect() as connection:
        assert (
            connection.execute(select(fill.c.id).where(fill.c.order_id == order.order_id)).first()
            is None
        )
        assert (
            connection.execute(
                select(book_fill_allocation.c.id).where(book_fill_allocation.c.book_id.in_(books))
            ).first()
            is None
        )
    application = FinancialApplication(database.engine, scope)
    first = application.apply_fill(execution)
    assert not application.apply_fill(execution).applied
    second = application.apply_fill(
        replace(execution, quantity=2, fee=0.02, metadata={"trade_id": "2"})
    )
    assert second.order.filled_quantity == second.position.quantity == 3
    with database.engine.connect() as connection:
        rows = (
            connection.execute(
                select(book_fill_allocation).where(book_fill_allocation.c.book_id.in_(books))
            )
            .mappings()
            .all()
        )
        events = (
            connection.execute(
                select(book_event.c.payload).where(
                    book_event.c.payload["book_id"].as_string().in_(books)
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == len(events) == 4
    for result in (first, second):
        allocated = [row for row in rows if row["fill_id"] == result.fill.fill_id]
        assert sum(row["signed_quantity"] for row in allocated) == Decimal(
            str(result.fill.quantity)
        )
        assert sum(row["fee"] for row in allocated) == Decimal(str(result.fill.fee))
    positions = PositionManager(SqlPositionStore(database.engine))
    assert [positions.get(book, order.instrument_id).quantity for book in books] == pytest.approx(
        [1, 2]
    )
    assert sum(event["balance_changes"]["USDT"] for event in events) == pytest.approx(-0.03)
    if postgres:
        from sqlalchemy.exc import DBAPIError

        with pytest.raises(DBAPIError, match="conserve"):
            with database.engine.begin() as connection:
                connection.execute(
                    insert(fill).values(
                        id=f"unallocated-{token}",
                        order_id=order.order_id,
                        created_at=NOW,
                        payload={"quantity": 1, "side": "buy", "fee": 0, "fee_asset": "USDT"},
                    )
                )
    from src.portfolio.account_plans import prepare_book_flatten
    from src.portfolio.books import book_balances, book_foreign_fee_cost
    from src.services.forward_metrics import ForwardEvidenceCollector

    closing = OrderIntent(
        f"book-flatten-{token}",
        account,
        order.instrument_id,
        OrderSide.SELL,
        3,
        OrderType.MARKET,
        "2026-09-07T00:00:01+00:00",
        reduce_only=True,
        metadata={"recovery": True, "reference_price": 101},
    )
    owned = prepare_book_flatten(database.engine, scope, closing)
    assert prepare_book_flatten(database.engine, scope, closing) == owned
    assert [
        row["signed_quantity"] for row in owned.metadata["target_metadata"]["book_contributions"]
    ] == [-1, -2]
    manager.reload()
    manager.persist_for_submission(owned.order_id)
    closing_fill = replace(
        execution,
        order_id=owned.order_id,
        fill_id="close",
        side=OrderSide.SELL,
        quantity=3,
        price=101,
        fee=0.003,
        fee_asset="BNB",
        occurred_at=owned.created_at,
        metadata={"trade_id": "3", "fee_conversion_rate": 200},
    )
    assert application.apply_fill(closing_fill).applied
    assert not application.apply_fill(closing_fill).applied
    collector = ForwardEvidenceCollector(database.engine)
    for index, book in enumerate(books):
        balance = book_balances(database.engine, book)
        assert balance["USDT"] == pytest.approx(0.99 * (index + 1))
        assert balance["BNB"] == pytest.approx(-0.001 * (index + 1))
        assert book_foreign_fee_cost(database.engine, book) == pytest.approx(0.2 * (index + 1))
        evidence = collector._ledger_rows(
            product_id="active_income",
            start="2026-09-06T23:59:59+00:00",
            at=owned.created_at,
            book_id=book,
            instrument_id=order.instrument_id,
        )
        assert collector._ledger_pnl(evidence) == pytest.approx(0.79 * (index + 1))
    database.dispose()


def _allocation_books(database, account, token, *, product="active_income", mode="live"):
    from sqlalchemy import insert

    from src.data.database import (
        strategy_artefact,
        strategy_book,
        strategy_definition,
        strategy_version,
    )

    books = []
    with database.engine.begin() as connection:
        for index in range(2):
            identity = f"book-{token}-{index}"
            books.append(identity)
            connection.execute(
                insert(strategy_definition).values(
                    id=identity,
                    identity=identity,
                    product_id=product,
                    source_type="fixture",
                    source_hash=identity,
                    definition={},
                )
            )
            connection.execute(
                insert(strategy_version).values(
                    id=identity,
                    definition_id=identity,
                    version="fixture",
                    created_at=NOW,
                    payload={},
                )
            )
            connection.execute(
                insert(strategy_artefact).values(id=identity, created_at=NOW, payload={})
            )
            book = {
                "book_id": identity,
                "product_id": product,
                "artefact_hash": identity,
                "strategy_version_id": identity,
                "execution_mode": mode,
                "sleeve_id": "test",
                "account_id": identity if mode == "paper" else account,
                "execution_account_id": account,
                "product_portfolio_id": account,
                "created_at": NOW,
                "position_portfolio_id": identity,
                "instrument_ids": [
                    "binance:spot:BTCUSDT"
                    if product == "btc_accumulation"
                    else "binance:futures:BTCUSDT:USDT"
                ],
            }
            connection.execute(
                insert(strategy_book).values(
                    id=identity,
                    product_id=product,
                    artefact_hash=identity,
                    strategy_version_id=identity,
                    execution_mode=mode,
                    sleeve_id="test",
                    account_id=book["account_id"],
                    created_at=NOW,
                    payload=book,
                )
            )
    return books


@pytest.mark.parametrize("postgres", [False, True])
def test_account_plan_authority_rollback_and_restart_preserve_books(
    tmp_path, monkeypatch, postgres
):
    from sqlalchemy import insert

    from src.data.database import account_portfolio_plan, book_event, job
    from src.domain._codec import canonical_hash, to_primitive
    from src.domain.portfolios import TargetPosition
    from src.portfolio.account_plans import _capital_proposals
    from src.portfolio.book_plans import net_book_changes
    from src.services.order_execution import DatabaseExecutionWorker
    from src.services.scheduler import DatabaseJobQueue

    url = os.environ.get("TRADING_PLATFORM_DATABASE_URL", "")
    if postgres and not url.startswith("postgresql"):
        pytest.skip("requires isolated PostgreSQL")
    database = PlatformDatabase(url if postgres else f"sqlite+pysqlite:///{tmp_path / 'plan.db'}")
    database.assert_migrated() if postgres else database.create_schema()
    token = uuid.uuid4().hex
    account = "plan-" + token
    books = _allocation_books(database, account, token)
    symbol = "binance:futures:BTCUSDT:USDT"
    assignments = [
        {
            "id": book,
            "product_id": "active_income",
            "capital_limit": 100,
            "payload": {"book_id": book},
        }
        for book in books
    ]
    with database.engine.begin() as connection:
        capital = _capital_proposals(
            database.engine, connection, assignments, {"balances": {"USDT": 200}}, NOW
        )
        assert (
            connection.execute(
                select(book_event.c.id).where(
                    book_event.c.payload["book_id"].as_string().in_(books)
                )
            ).first()
            is None
        )
        with pytest.raises(ValueError, match="settled assets"):
            _capital_proposals(
                database.engine,
                connection,
                assignments,
                {"balances": {"USDT": 199}, "marked_equity": 1000},
                NOW,
            )
    owners = {
        book: {
            "assignment_id": book,
            "strategy_version_id": book,
            "book_id": book,
            "sleeve_id": "test",
        }
        for book in books
    }
    net = net_book_changes({books[0]: {symbol: 1}, books[1]: {symbol: -0.25}}, {}, {symbol: 100})
    identity = canonical_hash({"fixture_plan": token})
    target = TargetPosition(
        portfolio_id=account,
        instrument_id=symbol,
        target_quantity=0.75,
        target_notional=75,
        target_fraction=0.375,
        strategy_contributions={books[0]: 1},
        risk_budget=100,
        valid_until="2026-09-07T00:01:00+00:00",
        metadata={
            "account_plan_id": identity,
            "book_contributions": [
                {**owners[row["book_id"]], **row} for row in net.contributions[symbol]
            ],
            "book_authorities": list(owners.values()),
        },
    )
    plan = {
        "plan_id": identity,
        "product_id": "active_income",
        "account_id": account,
        "decision_at": NOW,
        "books": owners,
        "initial_capital": capital,
        "book_positions_before": dict.fromkeys(books, {}),
        "book_balances_before": dict.fromkeys(books, {}),
        "book_targets": net.desired_positions,
        "prices": {symbol: 100},
        "internal_transfers": list(net.internal_transfers),
        "targets": to_primitive((target,)),
    }
    with database.engine.begin() as connection:
        connection.execute(
            insert(account_portfolio_plan).values(
                id=identity,
                product_id="active_income",
                account_id=account,
                decision_at=NOW,
                payload=plan,
            )
        )
    queue = DatabaseJobQueue(database.engine, claim_scope="plan-" + token)
    checked = []

    def authority(candidate, **_kwargs):
        for book in candidate["books"]:
            checked.append(book)
            if book == books[1]:
                raise PermissionError("second contributor revoked")

    worker = DatabaseExecutionWorker(
        queue=queue,
        worker_id="fixture",
        order_manager=OrderManager(SqlOrderStore(database.engine)),
        positions=PositionManager(SqlPositionStore(database.engine)),
        risk_store=None,
        trace_store=None,
        product_execution={},
        plan_authority=authority,
    )
    inputs = {"prices": {symbol: 100}, "reconciled_positions": {symbol: 0}}
    payload = {"evaluated_at": NOW, "event_id": token}
    configuration = {"base_accounting_asset": "USDT", "execution_costs": {}}

    def execute():
        return worker._process_account_plan(
            identity, payload, "active_income", "live", configuration, inputs, (target,), NOW
        )

    with pytest.raises(PermissionError, match="second contributor"):
        execute()
    assert checked == books
    worker.plan_authority = lambda candidate, **_kwargs: checked.extend(candidate["books"])
    enqueue = queue.enqueue_if_absent

    def crash(**_kwargs):
        raise RuntimeError("crash after internal allocation")

    monkeypatch.setattr(queue, "enqueue_if_absent", crash)
    with pytest.raises(RuntimeError, match="after internal allocation"):
        execute()
    assert not PositionManager(SqlPositionStore(database.engine)).current_quantities(books[0])
    with database.engine.connect() as connection:
        assert (
            connection.execute(
                select(book_event.c.id).where(
                    book_event.c.payload["book_id"].as_string().in_(books)
                )
            ).first()
            is None
        )
    monkeypatch.setattr(queue, "enqueue_if_absent", enqueue)
    result = execute()
    assert result["orders"] == 1
    assert execute() == result
    positions = PositionManager(SqlPositionStore(database.engine))
    assert [positions.get(book, symbol).quantity for book in books] == [0.25, -0.25]
    assert positions.get(account, symbol).quantity == 0
    with database.engine.connect() as connection:
        assert (
            len(
                connection.execute(
                    select(job.c.id).where(job.c.id.in_(result["venue_job_ids"]))
                ).all()
            )
            == 1
        )
    database.dispose()


@pytest.mark.parametrize("postgres", [False, True])
def test_delayed_funding_uses_settlement_inventory_and_rolls_back_allocations(
    tmp_path, monkeypatch, postgres
):
    import datetime as dt
    from decimal import Decimal

    from src.accounting.ledger import Ledger
    from src.data.database import book_event
    from src.execution.broker import BrokerIncome
    from src.services.income_recovery import reconcile_income

    url = os.environ.get("TRADING_PLATFORM_DATABASE_URL", "")
    if postgres and not url.startswith("postgresql"):
        pytest.skip("requires isolated PostgreSQL")
    database = PlatformDatabase(
        url if postgres else f"sqlite+pysqlite:///{tmp_path / 'funding.db'}"
    )
    database.assert_migrated() if postgres else database.create_schema()
    token = uuid.uuid4().hex
    account = "funding-" + token
    books = _allocation_books(database, account, token)
    symbol = "binance:futures:BTCUSDT:USDT"
    scope = FinancialScope("active_income", account, "testnet", "binance", "USDT")
    start = dt.datetime.fromisoformat(NOW)
    orders = OrderManager(SqlOrderStore(database.engine))
    for offset, side, sign in ((0, OrderSide.BUY, 1), (2, OrderSide.SELL, -1)):
        at = (start + dt.timedelta(seconds=offset)).isoformat()
        order = OrderIntent(
            f"funding-{token}-{offset}",
            account,
            symbol,
            side,
            3,
            OrderType.MARKET,
            at,
            reduce_only=sign < 0,
            metadata={
                "target_metadata": {
                    "account_id": account,
                    "execution_environment": "testnet",
                    "book_contributions": [
                        {"book_id": book, "signed_quantity": sign * (index + 1)}
                        for index, book in enumerate(books)
                    ],
                }
            },
        )
        orders.create(order)
        orders.persist_for_submission(order.order_id)
        FinancialApplication(database.engine, scope).apply_fill(
            Fill(
                f"funding-fill-{token}-{offset}",
                order.order_id,
                symbol,
                side,
                3,
                100,
                0,
                at,
                fee_asset="USDT",
                metadata={"trade_id": str(offset)},
            )
        )
    positions = PositionManager(SqlPositionStore(database.engine))
    assert all(positions.get(book, symbol).quantity == 0 for book in books)
    ledger = Ledger(
        product_id="active_income",
        accounting_asset="USDT",
        store=SqlLedgerStore(database.engine, product_id="active_income"),
    )
    before = ledger.entries
    income = BrokerIncome(
        "funding-" + token, "BTCUSDT", "FUNDING_FEE", -0.07, "USDT", start.timestamp() + 1
    )

    def fail(*_args):
        raise RuntimeError("allocation persistence interrupted")

    with monkeypatch.context() as patch:
        patch.setattr("src.accounting.book_allocations.record_book_funding", fail)
        with pytest.raises(RuntimeError, match="persistence interrupted"):
            reconcile_income(database.engine, ledger, scope, income)
    ledger.reload()
    assert ledger.entries == before
    assert reconcile_income(database.engine, ledger, scope, income)
    assert not reconcile_income(database.engine, ledger, scope, income)
    with database.engine.connect() as connection:
        events = tuple(
            connection.execute(
                select(book_event.c.payload).where(
                    book_event.c.payload["book_id"].as_string().in_(books),
                    book_event.c.payload["kind"].as_string() == "funding_allocation",
                )
            ).scalars()
        )
    assert len(events) == 2
    assert {row["book_id"]: Decimal(row["quantity_at_settlement"]) for row in events} == dict(
        zip(books, (Decimal(1), Decimal(2)), strict=True)
    )
    assert sum(Decimal(row["amount"]) for row in events) == Decimal("-0.07")
    assert sum(Decimal(row["balance_changes"]["USDT"]) for row in events) == Decimal("-0.07")
    from src.services.forward_metrics import ForwardEvidenceCollector

    collector = ForwardEvidenceCollector(database.engine)
    lower, upper = (
        (start - dt.timedelta(seconds=1)).isoformat(),
        (start + dt.timedelta(seconds=3)).isoformat(),
    )
    for index, book in enumerate(books):
        scoped_orders = collector._orders(
            portfolio_id=book, instrument_id=symbol, start=lower, at=upper
        )
        assert len(scoped_orders) == 2
        assert [row["payload"]["quantity"] for row in scoped_orders] == [index + 1] * 2
        scoped_fills = collector._fills(
            tuple(row["id"] for row in scoped_orders), start=lower, at=upper, book_id=book
        )
        assert [row["payload"]["quantity"] for row in scoped_fills] == [index + 1] * 2
        assert all(
            row["payload"]["metadata"]["source_fill_id"] == row["id"] for row in scoped_fills
        )
        entries = collector._ledger_rows(
            product_id="active_income", start=lower, at=upper, instrument_id=symbol, book_id=book
        )
        expected = next(Decimal(row["amount"]) for row in events if row["book_id"] == book)
        assert collector._ledger_pnl(entries) == pytest.approx(float(expected))
    database.dispose()


def test_opposing_books_require_final_native_funding_even_with_zero_account_income(tmp_path):
    import datetime as dt
    from decimal import Decimal

    from sqlalchemy import insert

    from src.accounting.book_allocations import (
        apply_internal_transfers,
        settle_public_paper_funding,
    )
    from src.accounting.ledger import Ledger
    from src.data.database import account, funding_entry
    from src.domain._codec import canonical_hash
    from src.execution.broker import BrokerIncome
    from src.portfolio.books import book_balances
    from src.services.income_recovery import reconcile_income

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'opposing-funding.db'}")
    database.create_schema()
    books = _allocation_books(database, "account", "opposing-funding")
    symbol = "binance:futures:BTCUSDT:USDT"
    at = "2026-09-07T00:00:01+00:00"
    scope = FinancialScope("active_income", "account", "testnet", "binance", "USDT")
    with database.engine.begin() as connection:
        apply_internal_transfers(
            database.engine,
            connection,
            {
                "plan_id": "opposed",
                "decision_at": NOW,
                "books": {book: {"assignment_id": book} for book in books},
                "internal_transfers": [
                    {
                        "from_book": books[1],
                        "to_book": books[0],
                        "instrument_id": symbol,
                        "quantity": 2,
                        "price": 100,
                    }
                ],
            },
            PositionManager(SqlPositionStore(database.engine, connection=connection)),
        )
    ledger = Ledger(
        product_id="active_income",
        accounting_asset="USDT",
        store=SqlLedgerStore(database.engine, product_id="active_income"),
    )
    income = BrokerIncome(
        "zero-net", "BTCUSDT", "FUNDING_FEE", 0, "USDT", dt.datetime.fromisoformat(at).timestamp()
    )
    with pytest.raises(ValueError, match="unique native settlement"):
        reconcile_income(database.engine, ledger, scope, income)
    ledger.reload()
    assert not ledger.entries
    native = {
        "product_id": "active_income",
        "instrument_id": symbol,
        "symbol": "BTCUSDT",
        "kind": "historical_funding",
        "source": "binance_public_funding_rate",
        "source_event_id": "fixture-native-settlement",
        "funding_time": at,
        "funding_rate": 0.001,
        "mark_price": 100,
    }
    with database.engine.begin() as connection:
        connection.execute(
            insert(account).values(
                id="account", created_at=NOW, payload={"environment": "testnet", "venue": "binance"}
            )
        )
        connection.execute(
            insert(funding_entry).values(id=canonical_hash(native), created_at=at, payload=native)
        )
    assert settle_public_paper_funding(database.engine, canonical_hash(native), now=at) == 2
    assert settle_public_paper_funding(database.engine, canonical_hash(native), now=at) == 0
    assert [book_balances(database.engine, book, at=at)["USDT"] for book in books] == [-0.2, 0.2]
    ledger.reload()
    assert not ledger.entries  # Internal transfers are not exchange income.
    assert reconcile_income(database.engine, ledger, scope, income)
    assert not reconcile_income(database.engine, ledger, scope, income)
    assert [book_balances(database.engine, book, at=at)["USDT"] for book in books] == [-0.2, 0.2]
    assert ledger.balances()["assets:cash"] == Decimal(0)
    database.dispose()


def test_public_funding_updates_isolated_paper_balances_and_forward_pnl(tmp_path):
    import datetime as dt

    from src.accounting.book_allocations import PaperFundingPending, settle_public_paper_funding
    from src.domain._codec import canonical_hash
    from src.portfolio.books import (
        initialise_paper_book,
        load_book,
        paper_book_account,
        seed_paper_book,
    )
    from src.services.accounting_service import AccountingService, DatabaseAccountingWorker
    from src.services.forward_metrics import ForwardEvidenceCollector
    from src.services.historical_bootstrap import HistoricalBootstrapService
    from src.services.scheduler import DatabaseJobQueue

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'paper-funding.db'}")
    database.create_schema()
    books = _allocation_books(database, "paper-template", "public-funding", mode="paper")
    with database.engine.begin() as connection:
        for book_id in books:
            book = load_book(database.engine, book_id, connection=connection)
            seed_paper_book(
                connection, book, balances={"USDT": 1000}, positions={}, accounting_asset="USDT"
            )
            initialise_paper_book(database.engine, connection, book, book_id)
    symbol = "binance:futures:BTCUSDT:USDT"
    start = dt.datetime.fromisoformat(NOW)
    orders = OrderManager(SqlOrderStore(database.engine))

    def create(book, quantity, side, name):
        order = OrderIntent(
            name,
            book,
            symbol,
            side,
            quantity,
            OrderType.MARKET,
            NOW,
            reduce_only=side is OrderSide.SELL,
            strategy_contributions={book: 1},
            metadata={
                "target_metadata": {
                    "book_id": book,
                    "account_id": book,
                    "assignment_id": book,
                    "execution_environment": "paper",
                }
            },
        )
        orders.create(order)
        orders.persist_for_submission(order.order_id)
        return order

    def execute(order, offset):
        at = (start + dt.timedelta(seconds=offset)).isoformat()
        application = FinancialApplication(
            database.engine,
            FinancialScope("active_income", order.portfolio_id, "paper", "binance", "USDT"),
        )
        result = application.apply_paper(
            order_id=order.order_id,
            instrument_id=symbol,
            request_id=order.order_id,
            request_hash=canonical_hash({"order": order.order_id}),
            build_fill=lambda current: Fill(
                "fill-" + current.order_id,
                current.order_id,
                symbol,
                current.side,
                current.quantity,
                100,
                0.01,
                at,
                fee_asset="USDT",
            ),
            account=paper_book_account(database.engine, order.portfolio_id),
            request_payload={},
        )
        application.acknowledge(result.application_id, at=at)

    for index, book in enumerate(books):
        execute(create(book, index + 1, OrderSide.BUY, "open-" + book), 0)
    closes = [
        create(book, index + 1, OrderSide.SELL, "close-" + book) for index, book in enumerate(books)
    ]
    execute(closes[0], 2)
    source = HistoricalBootstrapService(engine=database.engine, root=tmp_path / "market")
    native_at = (start + dt.timedelta(seconds=1)).isoformat()
    rows = source._normalise_funding_page(
        [
            {
                "symbol": "BTCUSDT",
                "fundingTime": int(start.timestamp() * 1000) + 1000,
                "fundingRate": "0.001",
                "markPrice": "100",
            }
        ],
        product_id="active_income",
        symbol="BTCUSDT",
    )
    source._record_funding_entries(rows)
    now = (dt.datetime.now(dt.UTC) + dt.timedelta(seconds=1)).isoformat()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="accounting-fixture",
        node_id="fixture",
        role="accounting",
        capabilities=("accounting_event",),
        observed_at=now,
    )
    worker = DatabaseAccountingWorker(
        queue=queue,
        worker_id="accounting-fixture",
        service=AccountingService(engine=database.engine),
    )
    assert worker.run_once(now=now)["reason_code"] == "paper_funding_pending"
    assert [
        paper_book_account(database.engine, book).state().balances["USDT"] for book in books
    ] == [999.98, 999.99]
    execute(closes[1], 2)
    later = (dt.datetime.fromisoformat(now) + dt.timedelta(seconds=61)).isoformat()
    outcome = worker.run_once(now=later)
    assert outcome["reason_code"] == "accounting_event_recorded", outcome
    source._record_funding_entries(rows)
    assert worker.run_once(now=later)["reason_code"] == "accounting_queue_empty"
    for index, book in enumerate(books):
        account = paper_book_account(database.engine, book)
        funding = (index + 1) * 0.1
        assert account.state().balances["USDT"] == pytest.approx(999.98 - funding)
        past = account.state(at=native_at)
        assert past.positions[symbol] == index + 1
        assert past.observed_at == native_at
        assert past.balances["USDT"] == pytest.approx(999.99 - funding)
        collector = ForwardEvidenceCollector(database.engine)
        entries = collector._ledger_rows(
            product_id="active_income",
            start=(start - dt.timedelta(seconds=1)).isoformat(),
            at=(start + dt.timedelta(seconds=3)).isoformat(),
            strategy_version_id=book,
            assignment_id="later-observation-window",
            instrument_id=symbol,
            book_id=book,
        )
        assert collector._ledger_pnl(entries) == pytest.approx(-0.02 - funding)
    queue.enqueue(
        job_id="earlier-feature-retry",
        name="live_feature_calculation",
        payload={"instrument_id": symbol, "availability_time": NOW},
        available_at=(dt.datetime.fromisoformat(later) + dt.timedelta(minutes=10)).isoformat(),
    )
    with pytest.raises(PaperFundingPending, match="decision work"):
        settle_public_paper_funding(database.engine, outcome["record_id"], now=later)
    database.dispose()
