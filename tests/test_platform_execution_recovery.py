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
