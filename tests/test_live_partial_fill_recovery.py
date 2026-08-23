from __future__ import annotations

from src.domain.orders import Fill, OrderIntent, OrderSide, OrderStatus, OrderType
from src.execution.order_manager import JsonlOrderStore, OrderManager


def test_multiple_partial_fills_replay_with_exchange_times(tmp_path) -> None:
    manager = OrderManager(JsonlOrderStore(tmp_path / "orders.jsonl"))
    intent = OrderIntent(
        order_id="order-1",
        portfolio_id="portfolio-1",
        instrument_id="instrument-1",
        side=OrderSide.BUY,
        quantity=2.0,
        order_type=OrderType.MARKET,
        created_at="2026-08-23T00:00:00+00:00",
    )
    manager.create(intent)
    manager.persist_for_submission(intent.order_id)
    manager.submitted(intent.order_id)
    for index, quantity in enumerate((0.75, 1.25), 1):
        manager.apply_fill(
            Fill(
                fill_id=f"fill-{index}",
                order_id=intent.order_id,
                instrument_id=intent.instrument_id,
                side=OrderSide.BUY,
                quantity=quantity,
                price=100.0 + index,
                fee=0.01,
                occurred_at=f"2026-08-23T00:00:0{index}+00:00",
            )
        )
    assert manager.get(intent.order_id).status is OrderStatus.FILLED
    assert len(OrderManager(JsonlOrderStore(tmp_path / "orders.jsonl")).all_fills()) == 2
