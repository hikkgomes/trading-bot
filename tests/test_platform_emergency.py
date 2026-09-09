from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest
from sqlalchemy import select

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.data.database import PlatformDatabase, job
from src.domain.orders import Fill, OrderSide
from src.execution.broker import BrokerFill, BrokerOrderAcknowledgement, BrokerOrderState
from src.execution.broker import OrderSide as BrokerSide
from src.execution.live_exchange import (
    BrokerExecutionVenue,
    SubmissionAcknowledgement,
    bounded_client_order_id,
)
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.services.emergency import DatabaseEmergencyFlattenWorker
from src.services.live_execution import ApprovedLiveExecution
from src.services.scheduler import DatabaseJobQueue

NOW = "2026-08-31T10:00:00+00:00"
INSTRUMENT = "binance:futures:BTCUSDT:USDT"


@dataclass
class _Instrument:
    exchange_symbol: str = "BTCUSDT"
    is_tradable: bool = True


class _Broker:
    def __init__(self):
        self.quantity = 0.0

    def get_price(self, symbol: str) -> float:
        assert symbol == "BTCUSDT"
        return 100.0

    def account_snapshot(self, *, expected_symbols):
        return {
            "positions": {INSTRUMENT: self.quantity} if self.quantity else {},
            "regular_orders": [],
            "conditional_orders": [],
            "unknown_exposure": {},
            "account_state_known": True,
            "account_state_authority": "authenticated_rest",
        }


class _Venue:
    def __init__(self) -> None:
        self.broker = _Broker()
        self.instruments = {INSTRUMENT: _Instrument()}
        self.submitted = []
        self.order_manager = None
        self.position_manager = None

    def submit(self, order):
        self.submitted.append(order)
        self.broker.quantity += order.quantity if order.side is OrderSide.BUY else -order.quantity
        assert self.order_manager is not None
        self.order_manager.submitted(order.order_id)
        fill = Fill(
            fill_id=f"fill:{order.order_id}",
            order_id=order.order_id,
            instrument_id=order.instrument_id,
            side=order.side,
            quantity=order.quantity,
            price=100.0,
            fee=0.0,
            occurred_at=NOW,
            metadata={"reference_price": 100.0},
        )
        updated = self.order_manager.apply_fill(fill)
        if self.position_manager is not None:
            self.position_manager.apply_fill(
                updated.portfolio_id,
                fill,
                contributions=dict(updated.strategy_contributions),
            )
        return SubmissionAcknowledgement(
            order_id=order.order_id,
            client_order_id=bounded_client_order_id(order),
            exchange_order_id=f"exchange:{order.order_id}",
            status="acknowledged",
        )


@pytest.mark.parametrize("apply_positions", [True, False])
def test_emergency_flatten_is_reduce_only_and_idempotent_after_restart(
    tmp_path, apply_positions
) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'emergency.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="emergency-1",
        node_id="test-node",
        role="emergency-control",
        capabilities=("emergency_flatten",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="flatten-1",
        name="emergency_flatten",
        payload={
            "control_id": "control-1",
            "target": "product:active_income",
            "reason_code": "manual_flatten",
        },
        available_at=NOW,
        priority=200,
    )
    orders = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    positions.reconcile_position(
        portfolio_id="active-income-portfolio",
        instrument_id=INSTRUMENT,
        quantity=-2.0,
        average_entry_price=100.0,
        updated_at=NOW,
    )
    venue = _Venue()
    venue.broker.quantity = -2.0
    worker = DatabaseEmergencyFlattenWorker(
        queue=queue,
        worker_id="emergency-1",
        order_manager=orders,
        positions=positions,
        venues={"active_income": venue},
        products={
            "active_income": {
                "portfolio_id": "active-income-portfolio",
            }
        },
    )
    venue.order_manager = orders
    venue.position_manager = positions if apply_positions else None
    first = worker.run_once(now=NOW)
    if not apply_positions:
        assert first["reason_code"] == "emergency_reconciliation_pending"
        assert positions.get("active-income-portfolio", INSTRUMENT).quantity == -2.0
        assert venue.broker.quantity == 0
        database.dispose()
        return
    assert first["reason_code"] == "emergency_action_completed"
    assert len(venue.submitted) == 1
    assert venue.submitted[0].side is OrderSide.BUY
    assert venue.submitted[0].reduce_only is True
    assert venue.submitted[0].metadata["control_id"] == "control-1"

    restarted = DatabaseEmergencyFlattenWorker(
        queue=queue,
        worker_id="emergency-2",
        order_manager=OrderManager(SqlOrderStore(database.engine)),
        positions=PositionManager(SqlPositionStore(database.engine)),
        venues={"active_income": venue},
        products={"active_income": {"portfolio_id": "active-income-portfolio"}},
    )
    queue.register_worker(
        worker_id="emergency-2",
        node_id="test-node",
        role="emergency-control",
        capabilities=("emergency_flatten",),
        observed_at=NOW,
    )
    assert restarted.run_once(now=NOW)["reason_code"] == "emergency_queue_empty"
    assert len(venue.submitted) == 1
    with database.engine.connect() as connection:
        assert (
            connection.execute(select(job.c.state).where(job.c.id == "flatten-1")).scalar_one()
            == "completed"
        )
    database.dispose()


def test_emergency_reduction_uses_authoritative_stop_quantity(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'emergency-target.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="emergency-1",
        node_id="test-node",
        role="emergency-control",
        capabilities=("emergency_flatten",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="reduction-1",
        name="emergency_flatten",
        payload={
            "control_id": "stop-control-1",
            "target": "product:active_income",
            "instrument_id": INSTRUMENT,
            "position_quantity": -0.25,
            "reason_code": "stop_confirmation_failed",
        },
        available_at=NOW,
        priority=200,
    )
    venue = _Venue()
    worker = DatabaseEmergencyFlattenWorker(
        queue=queue,
        worker_id="emergency-1",
        order_manager=OrderManager(SqlOrderStore(database.engine)),
        positions=PositionManager(SqlPositionStore(database.engine)),
        venues={"active_income": venue},
        products={"active_income": {"portfolio_id": "active-income-portfolio"}},
    )
    venue.order_manager = worker.order_manager
    venue.position_manager = worker.positions
    venue.broker.quantity = -0.25
    worker.positions.reconcile_position(
        portfolio_id="active-income-portfolio",
        instrument_id=INSTRUMENT,
        quantity=-0.25,
        average_entry_price=100,
        updated_at=NOW,
    )

    result = worker.run_once(now=NOW)

    assert result["reason_code"] == "emergency_action_completed"
    assert len(venue.submitted) == 1
    assert venue.submitted[0].instrument_id == INSTRUMENT
    assert venue.submitted[0].quantity == 0.25
    assert venue.submitted[0].side is OrderSide.BUY
    database.dispose()


def test_emergency_native_ack_requires_real_fills_and_books_close_pnl(tmp_path):
    class ReceiptBroker(_Broker):
        def submit_order(self, order):
            self.order = order
            self.quantity += order.qty
            return BrokerOrderAcknowledgement("native-close", order.client_id, "new")

        def query_order(self, **kwargs):
            return BrokerOrderState("native-close", self.order.client_id, "closed", 1, 95)

        def query_order_fills(self, **kwargs):
            return (
                BrokerFill(
                    "1",
                    "native-close",
                    self.order.client_id,
                    "BTCUSDT",
                    BrokerSide.BUY,
                    1,
                    95,
                    0.1,
                    1788163200,
                    "USDT",
                ),
            )

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'native-emergency.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="emergency",
        node_id="fixture",
        role="execution",
        capabilities=("emergency_flatten",),
        observed_at=NOW,
    )
    orders = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    positions.reconcile_position(
        portfolio_id="portfolio",
        instrument_id=INSTRUMENT,
        quantity=-1,
        average_entry_price=100,
        updated_at=NOW,
    )
    ledger = Ledger(
        product_id="active_income",
        accounting_asset="USDT",
        store=SqlLedgerStore(database.engine, product_id="active_income"),
    )
    broker = ReceiptBroker()
    broker.quantity = -1
    venue = BrokerExecutionVenue(
        order_manager=orders,
        position_manager=positions,
        broker=broker,
        instruments={INSTRUMENT: _Instrument()},
    )
    product = {
        "product_id": "active_income",
        "portfolio_id": "portfolio",
        "account_id": "account",
        "execution_mode": "paper",
    }
    approved = ApprovedLiveExecution(
        engine=database.engine,
        order_manager=orders,
        positions=positions,
        ledgers={"active_income": ledger},
        configuration={
            "products": {"products": [product]},
            "promotion": {"policies": []},
            "risk": {},
            "accounts": {"accounts": [{"account_id": "account", "environment": "testnet"}]},
        },
    )
    approved.venues = {"active_income": venue}
    worker = DatabaseEmergencyFlattenWorker(
        queue=queue,
        worker_id="emergency",
        order_manager=orders,
        positions=positions,
        venues=approved.venues,
        products={"active_income": product},
        recover_action=approved.recover_action,
    )
    queue.enqueue(
        job_id="emergency",
        name="emergency_flatten",
        available_at=NOW,
        payload={"product_id": "active_income", "control_id": "stop"},
    )
    result = worker.run_once(now=NOW)
    assert result["reason_code"] == "emergency_action_completed", result
    assert positions.get("portfolio", INSTRUMENT).quantity == 0
    assert ledger.balances()["assets:cash"] == Decimal("4.9")
    assert sum(len(orders.fills_for(order.order_id)) for order in orders.all()) == 1
    database.dispose()
