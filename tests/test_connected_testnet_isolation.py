from __future__ import annotations

from types import SimpleNamespace

from src.domain.market_events import MarketEvent, MarketEventType
from src.services.accounting_service import DatabaseAccountingWorker
from src.services.market_gateway import DatabaseGatewaySink
from src.services.order_execution import DatabaseLiveExecutionWorker, DatabaseUserStreamWorker
from src.services.portfolio_engine import DatabasePortfolioTargetWorker
from src.services.portfolio_state import DatabasePortfolioStateWorker
from src.services.risk_service import DatabaseRiskWorker

NOW = "2026-08-29T12:00:00+00:00"


class EmptyClaimQueue:
    def __init__(self) -> None:
        self.claimed_names: list[tuple[str, ...]] = []
        self.enqueued: list[dict] = []

    def claim(self, **values):
        self.claimed_names.append(tuple(values["names"]))
        return None

    def enqueue_if_absent(self, **values):
        self.enqueued.append(values)
        return True


def test_connected_workers_claim_only_their_private_job_names() -> None:
    queue = EmptyClaimQueue()
    workers = (
        DatabasePortfolioStateWorker(
            queue=queue,
            worker_id="state",
            store=SimpleNamespace(),
            job_name="connected_state",
        ),
        DatabasePortfolioTargetWorker(
            queue=queue,
            worker_id="target",
            build_target=lambda payload: payload,
            job_name="connected_target",
        ),
        DatabaseRiskWorker(
            queue=queue,
            worker_id="risk",
            store=SimpleNamespace(),
            job_name="connected_risk",
        ),
        DatabaseLiveExecutionWorker(
            queue=queue,
            worker_id="live",
            order_manager=SimpleNamespace(),
            positions=SimpleNamespace(),
            ledgers={},
            trace_store=SimpleNamespace(),
            venues={},
            authorise=lambda payload, order: None,
            job_name="connected_live",
        ),
        DatabaseUserStreamWorker(
            engine=SimpleNamespace(),
            queue=queue,
            worker_id="events",
            job_name="connected_user_stream",
        ),
        DatabaseAccountingWorker(
            queue=queue,
            worker_id="accounting",
            service=SimpleNamespace(),
            job_name="connected_accounting",
        ),
    )

    for worker in workers:
        worker.run_once(now=NOW)

    assert queue.claimed_names == [
        ("connected_state",),
        ("connected_target",),
        ("connected_risk",),
        ("connected_live",),
        ("connected_user_stream",),
        ("connected_accounting",),
    ]


def test_connected_gateway_uses_private_user_stream_identity_and_name() -> None:
    queue = EmptyClaimQueue()
    sink = DatabaseGatewaySink(
        queue,
        user_stream_job_name="connected_user_stream",
        user_stream_job_prefix="connected-event",
    )
    event = MarketEvent(
        instrument_id="binance:futures:BTCUSDT:USDT",
        event_type=MarketEventType.ACCOUNT_BALANCE,
        exchange_timestamp=NOW,
        receive_timestamp=NOW,
        sequence=1,
        payload={"data": {"e": "ACCOUNT_UPDATE"}},
    )

    sink.write_user(event, account_id="testnet-futures", market="futures")

    assert queue.enqueued[0]["name"] == "connected_user_stream"
    assert queue.enqueued[0]["job_id"].startswith("connected-event:")


def test_user_stream_reconnect_queues_rest_reconciliation() -> None:
    queue = EmptyClaimQueue()
    sink = DatabaseGatewaySink(
        queue,
        user_stream_job_prefix="connected-event",
    )

    job_id = sink.mark_user_stream_recovery(
        account_id="testnet-futures",
        market="futures",
        observed_at=NOW,
        reason_code="user_stream_disconnect",
    )

    assert queue.enqueued == [
        {
            "job_id": job_id,
            "name": "live_order_recovery",
            "payload": {
                "account_id": "testnet-futures",
                "market": "futures",
                "recovery_kind": "user_stream_reconnect",
                "reason_code": "user_stream_disconnect",
                "observed_at": NOW,
            },
            "available_at": NOW,
            "priority": 100,
            "producer_identity": "connected-event:recovery",
        }
    ]
