"""Deterministic event-time simulator for short-horizon and multi-leg research."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from src.domain._codec import non_empty, timestamp


class SimulatedOrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class SimulatedOrderStatus(StrEnum):
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ReplayEvent:
    event_time: str
    receive_time: str
    instrument_id: str
    best_bid: float | None = None
    best_ask: float | None = None
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    traded_at_bid: float = 0.0
    traded_at_ask: float = 0.0
    mark_price: float | None = None
    funding_rate: float = 0.0
    liquidation_quantity: float = 0.0
    connected: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_time", timestamp(self.event_time, field="event_time"))
        object.__setattr__(self, "receive_time", timestamp(self.receive_time, field="receive_time"))
        object.__setattr__(
            self, "instrument_id", non_empty(self.instrument_id, field="instrument_id")
        )
        if self.best_bid is not None and self.best_bid <= 0:
            raise ValueError("best_bid must be positive")
        if self.best_ask is not None and self.best_ask <= 0:
            raise ValueError("best_ask must be positive")
        if (
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid > self.best_ask
        ):
            raise ValueError("best bid cannot exceed best ask")
        if (
            min(
                self.bid_depth,
                self.ask_depth,
                self.traded_at_bid,
                self.traded_at_ask,
            )
            < 0
        ):
            raise ValueError("event liquidity quantities cannot be negative")


@dataclass(frozen=True)
class SimulatedLimitOrder:
    order_id: str
    instrument_id: str
    side: SimulatedOrderSide
    quantity: float
    limit_price: float
    submitted_at: str
    expires_at: str
    cancel_requested_at: str | None = None
    queue_ahead_quantity: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity <= 0 or self.limit_price <= 0 or self.queue_ahead_quantity < 0:
            raise ValueError("order quantity, price, and queue estimate are invalid")
        object.__setattr__(self, "submitted_at", timestamp(self.submitted_at, field="submitted_at"))
        object.__setattr__(self, "expires_at", timestamp(self.expires_at, field="expires_at"))
        if self.expires_at <= self.submitted_at:
            raise ValueError("order expiry must follow submission")
        if self.cancel_requested_at is not None:
            object.__setattr__(
                self,
                "cancel_requested_at",
                timestamp(self.cancel_requested_at, field="cancel_requested_at"),
            )


@dataclass(frozen=True)
class SimulatedEventFill:
    order_id: str
    instrument_id: str
    quantity: float
    price: float
    occurred_at: str
    spread_cost: float
    market_impact: float
    adverse_selection: float


@dataclass(frozen=True)
class SimulatedOrderResult:
    order: SimulatedLimitOrder
    status: SimulatedOrderStatus
    fills: tuple[SimulatedEventFill, ...]
    remaining_quantity: float
    reason_code: str


@dataclass(frozen=True)
class EventSimulationResult:
    orders: tuple[SimulatedOrderResult, ...]
    positions: Mapping[str, float]
    funding_paid: float
    connection_gaps: tuple[str, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)


class EventReplayEngine:
    def __init__(
        self,
        *,
        cancel_latency_seconds: float = 0.25,
        impact_bps_per_depth_fraction: float = 5.0,
        adverse_selection_horizon_events: int = 1,
    ) -> None:
        if cancel_latency_seconds < 0 or impact_bps_per_depth_fraction < 0:
            raise ValueError("event simulation latency and impact must be non-negative")
        if adverse_selection_horizon_events < 1:
            raise ValueError("adverse-selection horizon must be positive")
        self.cancel_latency_seconds = cancel_latency_seconds
        self.impact_bps_per_depth_fraction = impact_bps_per_depth_fraction
        self.adverse_selection_horizon_events = adverse_selection_horizon_events

    def simulate(
        self,
        *,
        events: Iterable[ReplayEvent],
        orders: Iterable[SimulatedLimitOrder],
    ) -> EventSimulationResult:
        replay = tuple(events)
        if any(
            replay[index].receive_time < replay[index - 1].receive_time
            for index in range(1, len(replay))
        ):
            raise ValueError("event receive timestamps must be chronological")
        gaps = tuple(event.receive_time for event in replay if not event.connected)
        results: list[SimulatedOrderResult] = []
        for order in sorted(orders, key=lambda item: (item.submitted_at, item.order_id)):
            result = self._simulate_order(order, replay)
            results.append(result)
        signed_fills = sorted(
            (
                fill.occurred_at,
                fill.order_id,
                fill.instrument_id,
                (1.0 if result.order.side is SimulatedOrderSide.BUY else -1.0) * fill.quantity,
            )
            for result in results
            for fill in result.fills
        )
        positions: dict[str, float] = {}
        funding_paid = 0.0
        fill_index = 0
        for event in replay:
            while (
                fill_index < len(signed_fills) and signed_fills[fill_index][0] <= event.receive_time
            ):
                _, _, instrument_id, signed_quantity = signed_fills[fill_index]
                positions[instrument_id] = positions.get(instrument_id, 0.0) + signed_quantity
                fill_index += 1
            funding_paid += (
                positions.get(event.instrument_id, 0.0)
                * float(event.mark_price or 0.0)
                * event.funding_rate
            )
        fill_count = sum(len(result.fills) for result in results)
        total_filled = sum(fill.quantity for result in results for fill in result.fills)
        return EventSimulationResult(
            orders=tuple(results),
            positions=positions,
            funding_paid=funding_paid,
            connection_gaps=gaps,
            metrics={"fill_count": float(fill_count), "filled_quantity": total_filled},
        )

    def _simulate_order(
        self,
        order: SimulatedLimitOrder,
        events: tuple[ReplayEvent, ...],
    ) -> SimulatedOrderResult:
        remaining = order.quantity
        queue_ahead = order.queue_ahead_quantity
        fills: list[SimulatedEventFill] = []
        status = SimulatedOrderStatus.OPEN
        reason = "no_executable_event"
        for index, event in enumerate(events):
            if (
                event.instrument_id != order.instrument_id
                or event.receive_time < order.submitted_at
            ):
                continue
            if event.receive_time >= order.expires_at:
                status = SimulatedOrderStatus.EXPIRED
                reason = "order_expired"
                break
            if order.cancel_requested_at is not None:
                cancel_effective = _plus_seconds(
                    order.cancel_requested_at, self.cancel_latency_seconds
                )
                if event.receive_time >= cancel_effective:
                    status = SimulatedOrderStatus.CANCELLED
                    reason = "cancel_effective"
                    break
            if not event.connected:
                reason = "connection_gap"
                continue
            traded = (
                event.traded_at_bid if order.side is SimulatedOrderSide.BUY else event.traded_at_ask
            )
            touches = (
                event.best_ask is not None and order.limit_price >= event.best_ask
                if order.side is SimulatedOrderSide.BUY
                else event.best_bid is not None and order.limit_price <= event.best_bid
            )
            if not touches and traded <= queue_ahead:
                queue_ahead = max(0.0, queue_ahead - traded)
                continue
            executable = traded - queue_ahead if traded > queue_ahead else 0.0
            queue_ahead = max(0.0, queue_ahead - traded)
            if touches:
                visible_depth = (
                    event.ask_depth if order.side is SimulatedOrderSide.BUY else event.bid_depth
                )
                executable = max(executable, visible_depth)
            quantity = min(remaining, max(0.0, executable))
            if quantity <= 0:
                continue
            if event.best_ask is None or event.best_bid is None:
                raise ValueError("trade event has no executable top of book")
            touch_price = (
                float(event.best_ask)
                if order.side is SimulatedOrderSide.BUY
                else float(event.best_bid)
            )
            depth = max(
                event.ask_depth if order.side is SimulatedOrderSide.BUY else event.bid_depth,
                quantity,
            )
            impact_fraction = quantity / depth * self.impact_bps_per_depth_fraction / 10_000
            sign = 1.0 if order.side is SimulatedOrderSide.BUY else -1.0
            price = (
                min(order.limit_price, touch_price * (1 + impact_fraction))
                if sign > 0
                else max(order.limit_price, touch_price * (1 - impact_fraction))
            )
            mid = (
                (event.best_bid + event.best_ask) / 2
                if event.best_bid is not None and event.best_ask is not None
                else touch_price
            )
            future_mark = self._future_mark(events, index, order.instrument_id, fallback=mid)
            fills.append(
                SimulatedEventFill(
                    order_id=order.order_id,
                    instrument_id=order.instrument_id,
                    quantity=quantity,
                    price=price,
                    occurred_at=event.receive_time,
                    spread_cost=abs(price - mid) * quantity,
                    market_impact=abs(price - touch_price) * quantity,
                    adverse_selection=sign * (future_mark - price) * quantity,
                )
            )
            remaining -= quantity
            if remaining <= 1e-12:
                status = SimulatedOrderStatus.FILLED
                reason = "fully_filled"
                remaining = 0.0
                break
            status = SimulatedOrderStatus.PARTIAL
            reason = "partial_fill"
        return SimulatedOrderResult(order, status, tuple(fills), remaining, reason)

    def _future_mark(
        self,
        events: tuple[ReplayEvent, ...],
        start_index: int,
        instrument_id: str,
        *,
        fallback: float,
    ) -> float:
        seen = 0
        for event in events[start_index + 1 :]:
            if event.instrument_id != instrument_id or not event.connected:
                continue
            seen += 1
            if seen >= self.adverse_selection_horizon_events:
                if event.mark_price is not None:
                    return event.mark_price
                if event.best_bid is not None and event.best_ask is not None:
                    return (event.best_bid + event.best_ask) / 2
        return fallback


def _plus_seconds(value: str, seconds: float) -> str:
    import datetime as dt

    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).isoformat()
