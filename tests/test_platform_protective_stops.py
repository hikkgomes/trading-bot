from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from src.domain.market_events import MarketEvent
from src.domain.orders import OrderIntent, OrderSide, OrderStatus, OrderType
from src.execution.broker import ProtectiveOrder, ProtectiveOrderStatus
from src.execution.stops import ProtectiveStop, StopManager, StopStatus
from src.services.protective_stops import LiveProtectiveStopService, ProtectiveStopError

NOW = "2026-08-31T10:00:00+00:00"
INSTRUMENT = "binance:futures:BTCUSDT:USDT"


class _StopStore:
    def __init__(self) -> None:
        self.events: list[ProtectiveStop] = []

    def append(self, stop: ProtectiveStop) -> None:
        self.events.append(stop)

    def read(self) -> tuple[ProtectiveStop, ...]:
        return tuple(self.events)


@dataclass
class _Instrument:
    exchange_symbol: str = "BTCUSDT"


class _Broker:
    def __init__(self) -> None:
        self.placed: list[ProtectiveOrder] = []
        self.cancelled: list[str] = []

    def supports_native_protective_stops(self) -> bool:
        return True

    def place_protective_stop(self, *, symbol, side, qty, trigger_price, client_id):
        item = ProtectiveOrder(
            symbol=symbol,
            side=side,
            qty=qty,
            trigger_price=trigger_price,
            status=ProtectiveOrderStatus.OPEN,
            order_id=f"native-{len(self.placed) + 1}",
            client_id=client_id,
        )
        self.placed.append(item)
        return item

    def cancel_protective_stop(self, *, symbol, order_id, client_id):
        self.cancelled.append(order_id)
        return ProtectiveOrder(
            symbol=symbol,
            side=OrderSide.SELL,
            qty=1.0,
            trigger_price=95.0,
            status=ProtectiveOrderStatus.CANCELED,
            order_id=order_id,
            client_id=client_id,
        )


class _Venue:
    def __init__(self, broker: _Broker) -> None:
        self.broker = broker
        self.instruments = {INSTRUMENT: _Instrument()}


def _order(*, quantity: float = 1.0, reduce_only: bool = False) -> OrderIntent:
    return OrderIntent(
        order_id="entry-1" if not reduce_only else "close-1",
        portfolio_id="active-income-portfolio",
        instrument_id=INSTRUMENT,
        side=OrderSide.BUY if not reduce_only else OrderSide.SELL,
        quantity=quantity,
        order_type=OrderType.MARKET,
        created_at=NOW,
        status=OrderStatus.PERSISTED,
        reduce_only=reduce_only,
        metadata={
            "reference_price": 100.0,
            "protective_stop_price": 95.0,
        },
    )


def _service(*, broker: _Broker | None = None, manager: StopManager | None = None):
    broker = broker or _Broker()
    manager = manager or StopManager(_StopStore())
    service = LiveProtectiveStopService(
        stop_manager=manager,
        venues={"active_income": _Venue(broker)},
        products={
            "active_income": {
                "portfolio_id": "active-income-portfolio",
                "account_id": "futures-account",
            }
        },
        accounts={"futures-account": {"market": "usdt_futures"}},
    )
    return service, broker, manager


def test_entry_stop_is_durable_before_native_side_effect_and_confirmed_after_fill() -> None:
    service, broker, manager = _service()
    order = _order()
    stop = service.prepare_entry("active_income", order, NOW)
    assert stop is not None
    assert stop.status is StopStatus.ACTIVE
    assert broker.placed == []

    protected = service.on_fill("active_income", order, 1.0, NOW)
    assert protected is not None
    assert protected.status is StopStatus.PROTECTED
    assert protected.protected_quantity == pytest.approx(1.0)
    assert len(broker.placed) == 1
    assert manager.for_entry_order(order.order_id)[0].native_order_id == "native-1"


def test_partial_fill_resizes_and_replaces_native_stop() -> None:
    service, broker, _manager = _service()
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    service.on_fill("active_income", order, 1.0, NOW)

    resized = service.on_fill("active_income", order, 0.4, NOW)
    assert resized is not None
    assert resized.status is StopStatus.PROTECTED
    assert resized.quantity == pytest.approx(0.4)
    assert resized.protected_quantity == pytest.approx(0.4)
    assert broker.cancelled == ["native-1"]
    assert [item.qty for item in broker.placed] == [1.0, 0.4]


def test_futures_entry_without_trigger_fails_before_exchange_side_effect() -> None:
    service, broker, _manager = _service()
    order = replace(_order(), metadata={"reference_price": 100.0})
    with pytest.raises(ProtectiveStopError, match="protective_stop_price"):
        service.prepare_entry("active_income", order, NOW)
    assert broker.placed == []


def test_stop_failure_is_durable_and_latches_unprotected_state() -> None:
    broker = _Broker()

    def fail(**_kwargs):
        raise RuntimeError("exchange unavailable")

    broker.place_protective_stop = fail
    service, _broker, manager = _service(broker=broker)
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    with pytest.raises(ProtectiveStopError, match="exchange unavailable"):
        service.on_fill("active_income", order, 1.0, NOW)
    assert (
        manager.get(manager.for_entry_order(order.order_id)[0].stop_id).status
        is StopStatus.CONFIRMATION_FAILED
    )


def test_rejected_entry_cancels_its_unsubmitted_protective_stop() -> None:
    service, broker, manager = _service()
    order = _order()
    service.prepare_entry("active_income", order, NOW)

    cancelled = service.on_order_status(
        "active_income",
        replace(order, status=OrderStatus.REJECTED),
        OrderStatus.REJECTED.value,
        NOW,
    )

    assert cancelled is not None
    assert cancelled.status is StopStatus.CANCELLED
    assert manager.active() == ()
    assert broker.placed == []


def test_unknown_algo_update_is_not_silently_ignored() -> None:
    service, _broker, _manager = _service()
    event = MarketEvent(
        instrument_id=INSTRUMENT,
        event_type="algo_update",
        exchange_timestamp=NOW,
        receive_timestamp=NOW,
        sequence=1,
        payload={
            "event": "ALGO_UPDATE",
            "data": {"a": {"s": "BTCUSDT", "aid": "unknown", "algoStatus": "TRIGGERED"}},
        },
    )
    result = service.on_algo_update("active_income", event)
    assert result["reason_code"] == "unknown_protective_algo_update"
