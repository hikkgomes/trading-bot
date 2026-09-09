"""Convert shared position actions into fixed-quantity research executions."""

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

from src.research.funding import row_settlements
from src.research.outcomes import OutcomeCategory
from src.strategies.position_state import PositionAction


class PositionTimingUnavailable(ValueError):
    """OHLC data cannot order an intrabar exit against a funding settlement."""

    outcome_category = OutcomeCategory.BLOCKED_DATA


def managed_futures_events(
    rows: Sequence[Mapping[str, Any]],
    times: Sequence[str],
    actions: Sequence[PositionAction],
    *,
    notional: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[dict[str, Any], ...]:
    by_bar: dict[int, list[PositionAction]] = {}
    for action in actions:
        by_bar.setdefault(action.bar_index, []).append(action)
    quantity = 0.0
    events: list[dict[str, Any]] = []
    for index, (row, time) in enumerate(zip(rows, times, strict=True)):
        events.extend(row_settlements(row))
        symbol = str(row.get("instrument_id", row.get("symbol", "BTCUSDT")))
        for action in by_bar.get(index, ()):
            action_time = _action_time(action, row, time)
            target = action.direction * notional / action.price if action.kind == "entry" else 0.0
            delta = target - quantity
            price = action.price * (1 + slippage_rate if delta > 0 else 1 - slippage_rate)
            events.append(
                {
                    "type": "fill",
                    "timestamp": action_time,
                    "symbol": symbol,
                    "side": "buy" if delta > 0 else "sell",
                    "quantity": abs(delta),
                    "price": price,
                    "reference_price": action.price,
                    "fee": abs(delta) * price * fee_rate,
                    "fee_asset": "USDT",
                    "position_reason": action.reason,
                    "execution_time_resolution": "bar",
                }
            )
            quantity = target
        events.append(
            {"type": "mark", "timestamp": time, "symbol": symbol, "mark_price": row["close"]}
        )
    return tuple(events)


def _action_time(action: PositionAction, row: Mapping[str, Any], close: str) -> str:
    if row.get("open_time_ms") is not None:
        opened = dt.datetime.fromtimestamp(float(row["open_time_ms"]) / 1000, dt.UTC).isoformat()
    else:
        opened = str(row.get("open_timestamp", close))
    if action.kind == "entry":
        return opened
    if action.reason == "horizon":
        return close
    gap = action.direction * (float(row["open"]) - action.price)
    if (action.reason == "stop" and gap <= 0) or (action.reason == "take_profit" and gap >= 0):
        return opened
    if any(opened < event["timestamp"] < close for event in row_settlements(row)):
        raise PositionTimingUnavailable("intrabar exit and funding need finer execution data")
    return close
