"""Adopt exchange-created protective children into the shared order ledger."""

from typing import Any

from src.domain._codec import canonical_hash
from src.domain.orders import OrderIntent, OrderSide, OrderStatus, OrderType
from src.execution.broker import ProtectiveOrder
from src.execution.order_manager import OrderManager
from src.execution.stops import ProtectiveStop
from src.portfolio.book_plans import protective_book_metadata


def adopt_protective_child(
    orders: OrderManager,
    stop: ProtectiveStop,
    native: ProtectiveOrder,
    broker: Any,
    at: str,
) -> OrderIntent:
    if not native.child_order_id or not stop.entry_order_id:
        raise ValueError("triggered protection requires child and parent order identities")
    state = broker.query_order(
        symbol=native.symbol,
        exchange_order_id=native.child_order_id,
        client_order_id="",
    )
    if state.exchange_order_id != native.child_order_id or not state.client_order_id:
        raise ValueError("protective child readback identity does not match")
    orders.reload()
    parent = orders.get(stop.entry_order_id)
    identity = canonical_hash(
        {
            "portfolio_id": stop.portfolio_id,
            "instrument_id": stop.instrument_id,
            "algo_id": native.order_id,
            "child_order_id": native.child_order_id,
        }
    )
    existing = next((order for order in orders.all() if order.order_id == identity), None)
    if existing is None:
        existing = orders.create(
            OrderIntent(
                order_id=identity,
                portfolio_id=stop.portfolio_id,
                instrument_id=stop.instrument_id,
                side=OrderSide(native.side.value),
                quantity=native.qty,
                order_type=OrderType.MARKET,
                created_at=at,
                reduce_only=True,
                strategy_contributions=parent.strategy_contributions,
                metadata={
                    **dict(parent.metadata),
                    "target_metadata": protective_book_metadata(
                        parent.metadata.get("target_metadata", {}),
                        exit_sign=1 if native.side.value == "buy" else -1,
                    ),
                    "reference_price": native.trigger_price,
                    "client_order_id": state.client_order_id,
                    "exchange_order_id": native.child_order_id,
                    "protective_stop_id": stop.stop_id,
                    "native_algo_id": native.order_id,
                    "source": "exchange_protective_child",
                },
            )
        )
    _assert_same_child(existing, stop, native, state.client_order_id)
    return _finish_adoption(orders, existing)


def _assert_same_child(
    order: OrderIntent,
    stop: ProtectiveStop,
    native: ProtectiveOrder,
    client_id: str,
) -> None:
    if (
        order.portfolio_id != stop.portfolio_id
        or order.instrument_id != stop.instrument_id
        or order.side.value != native.side.value
        or order.quantity != native.qty
        or not order.reduce_only
        or order.metadata.get("client_order_id") != client_id
        or order.metadata.get("exchange_order_id") != native.child_order_id
        or order.metadata.get("native_algo_id") != native.order_id
    ):
        raise ValueError("protective child identity conflicts with the durable order")


def _finish_adoption(orders: OrderManager, order: OrderIntent) -> OrderIntent:
    if order.status in {OrderStatus.CREATED, OrderStatus.VALIDATED, OrderStatus.PERSISTED}:
        orders.persist_for_submission(order.order_id)
        order = orders.submitted(order.order_id)
    if order.status is OrderStatus.SUBMITTED:
        order = orders.bind_exchange_acknowledgement(
            order.order_id,
            exchange_order_id=str(order.metadata["exchange_order_id"]),
            client_order_id=str(order.metadata["client_order_id"]),
        )
    return order
