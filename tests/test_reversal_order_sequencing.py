from __future__ import annotations

from src.domain.orders import OrderSide
from src.domain.portfolios import TargetPosition
from src.execution.order_planner import plan_orders


def test_reversal_plan_places_reduce_only_close_before_open() -> None:
    orders = plan_orders(
        [
            TargetPosition(
                portfolio_id="portfolio-1",
                instrument_id="instrument-1",
                target_quantity=2.0,
                target_notional=200.0,
                target_fraction=1.0,
                strategy_contributions={"strategy-1": 1.0},
                risk_budget=0.1,
                valid_until="2026-08-23T00:05:00+00:00",
            )
        ],
        current_quantities={"instrument-1": -1.0},
        prices={"instrument-1": 100.0},
        decided_at="2026-08-23T00:00:00+00:00",
    )
    assert len(orders) == 2
    assert orders[0].side is OrderSide.BUY and orders[0].reduce_only
    assert orders[1].side is OrderSide.BUY and not orders[1].reduce_only
