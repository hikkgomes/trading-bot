from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from src.data.database import PlatformDatabase, job
from src.domain.orders import OrderSide
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.services.emergency import DatabaseEmergencyFlattenWorker
from src.services.scheduler import DatabaseJobQueue

NOW = "2026-08-31T10:00:00+00:00"
INSTRUMENT = "binance:futures:BTCUSDT:USDT"


@dataclass
class _Instrument:
    exchange_symbol: str = "BTCUSDT"


class _Broker:
    def get_price(self, symbol: str) -> float:
        assert symbol == "BTCUSDT"
        return 100.0


class _Venue:
    def __init__(self) -> None:
        self.broker = _Broker()
        self.instruments = {INSTRUMENT: _Instrument()}
        self.submitted = []

    def submit(self, order):
        self.submitted.append(order)


def test_emergency_flatten_is_reduce_only_and_idempotent_after_restart(tmp_path) -> None:
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
    first = worker.run_once(now=NOW)
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
