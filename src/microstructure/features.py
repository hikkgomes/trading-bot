"""Causal order-book and trade-flow features from recorded Binance events."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


def _number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("market event contains a non-finite number")
    return number


@dataclass
class MicrostructureState:
    symbol: str
    max_trade_window: int = 2_000
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    signed_trades: deque[tuple[int, float]] = field(default_factory=deque)
    liquidations: deque[tuple[int, float]] = field(default_factory=deque)
    mark_price: float | None = None
    index_price: float | None = None
    funding_rate: float | None = None
    previous_spread_bps: float | None = None
    depth_ema: float | None = None
    added_depth: float = 0.0
    removed_depth: float = 0.0

    def __post_init__(self) -> None:
        self.signed_trades = deque(self.signed_trades, maxlen=self.max_trade_window)
        self.liquidations = deque(self.liquidations, maxlen=self.max_trade_window)

    def _replace_depth(self, side: dict[float, float], levels: list[Any]) -> None:
        before = sum(side.values())
        replacement: dict[float, float] = {}
        for raw in levels:
            if not isinstance(raw, list | tuple) or len(raw) < 2:
                raise ValueError("depth level must contain price and quantity")
            price, quantity = _number(raw[0]), _number(raw[1])
            if price <= 0 or quantity < 0:
                raise ValueError("depth price/quantity is invalid")
            if quantity:
                replacement[price] = quantity
        after = sum(replacement.values())
        self.added_depth += max(0.0, after - before)
        self.removed_depth += max(0.0, before - after)
        side.clear()
        side.update(replacement)

    def apply(self, event: dict[str, Any]) -> None:
        if event.get("symbol") not in {None, self.symbol}:
            return
        stream = str(event.get("stream") or "")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("event payload must be an object")
        if "depth" in stream:
            bids = payload.get("bids", payload.get("b"))
            asks = payload.get("asks", payload.get("a"))
            if isinstance(bids, list) and isinstance(asks, list):
                self._replace_depth(self.bids, bids)
                self._replace_depth(self.asks, asks)
        elif "bookTicker" in stream:
            bid, bid_qty = _number(payload["b"]), _number(payload["B"])
            ask, ask_qty = _number(payload["a"]), _number(payload["A"])
            self.bids = {bid: bid_qty}
            self.asks = {ask: ask_qty}
        elif "aggTrade" in stream or stream.endswith("@trade"):
            price = _number(payload["p"])
            quantity = _number(payload["q"])
            buyer_is_maker = bool(payload.get("m"))
            signed_quote = price * quantity * (-1.0 if buyer_is_maker else 1.0)
            event_time = int(event.get("event_time_ms") or 0)
            self.signed_trades.append((event_time, signed_quote))
        elif "forceOrder" in stream:
            order = payload.get("o")
            if isinstance(order, dict):
                price = _number(order.get("ap") or order.get("p"))
                quantity = _number(order.get("z") or order.get("q"))
                side = str(order.get("S") or "").upper()
                if side not in {"BUY", "SELL"}:
                    raise ValueError("liquidation side must be BUY or SELL")
                event_time = int(event.get("event_time_ms") or 0)
                self.liquidations.append(
                    (event_time, price * quantity * (1.0 if side == "BUY" else -1.0))
                )
        elif "markPrice" in stream:
            self.mark_price = _number(payload["p"])
            self.index_price = _number(payload["i"])
            self.funding_rate = _number(payload.get("r", 0.0))

    def _levels(self, side: str, depth: int) -> list[tuple[float, float]]:
        book = self.bids if side == "bid" else self.asks
        return sorted(book.items(), key=lambda item: item[0], reverse=side == "bid")[:depth]

    @staticmethod
    def _depth_slope(levels: list[tuple[float, float]], mid: float) -> float:
        distances = [abs(price - mid) / mid * 10_000 for price, _ in levels]
        cumulative = []
        running = 0.0
        for _, quantity in levels:
            running += quantity
            cumulative.append(running)
        if len(levels) == 1:
            return cumulative[0] / max(distances[0], 1e-12)
        x_mean = sum(distances) / len(distances)
        y_mean = sum(cumulative) / len(cumulative)
        variance = sum((value - x_mean) ** 2 for value in distances)
        if variance <= 0:
            return 0.0
        return (
            sum((x - x_mean) * (y - y_mean) for x, y in zip(distances, cumulative, strict=True))
            / variance
        )

    def snapshot(self, *, depth: int = 10, trade_window_ms: int = 60_000) -> dict[str, Any]:
        bids = self._levels("bid", depth)
        asks = self._levels("ask", depth)
        if not bids or not asks:
            return {"ok": False, "reason": "book_not_initialized", "symbol": self.symbol}
        best_bid, bid_quantity = bids[0]
        best_ask, ask_quantity = asks[0]
        if best_ask <= best_bid:
            return {"ok": False, "reason": "crossed_or_locked_book", "symbol": self.symbol}
        mid = (best_bid + best_ask) / 2
        spread_bps = (best_ask - best_bid) / mid * 10_000
        microprice = (best_ask * bid_quantity + best_bid * ask_quantity) / (
            bid_quantity + ask_quantity
        )
        bid_depth = sum(quantity for _, quantity in bids)
        ask_depth = sum(quantity for _, quantity in asks)
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth else 0.0
        weighted_bid = sum(quantity / (1 + index) for index, (_, quantity) in enumerate(bids))
        weighted_ask = sum(quantity / (1 + index) for index, (_, quantity) in enumerate(asks))
        weighted_total = weighted_bid + weighted_ask
        weighted_imbalance = (
            (weighted_bid - weighted_ask) / weighted_total if weighted_total else 0.0
        )
        latest_time = max(
            self.signed_trades[-1][0] if self.signed_trades else 0,
            self.liquidations[-1][0] if self.liquidations else 0,
        )
        cutoff = latest_time - trade_window_ms
        recent = [value for timestamp, value in self.signed_trades if timestamp >= cutoff]
        absolute_flow = sum(abs(value) for value in recent)
        cvd_quote = sum(recent)
        aggressor_imbalance = cvd_quote / absolute_flow if absolute_flow else 0.0
        recent_liquidations = [
            value for timestamp, value in self.liquidations if timestamp >= cutoff
        ]
        liquidation_notional = sum(abs(value) for value in recent_liquidations)
        liquidation_imbalance = (
            sum(recent_liquidations) / liquidation_notional if liquidation_notional else 0.0
        )
        spread_velocity = (
            spread_bps - self.previous_spread_bps if self.previous_spread_bps is not None else 0.0
        )
        self.previous_spread_bps = spread_bps
        self.depth_ema = (
            total_depth if self.depth_ema is None else 0.95 * self.depth_ema + 0.05 * total_depth
        )
        liquidity_ratio = total_depth / self.depth_ema if self.depth_ema else 1.0
        replenishment = self.added_depth / max(self.removed_depth, 1e-12)
        cancel_add_pressure = (self.added_depth - self.removed_depth) / max(
            self.added_depth + self.removed_depth, 1e-12
        )
        self.added_depth = 0.0
        self.removed_depth = 0.0
        return {
            "ok": True,
            "symbol": self.symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": spread_bps,
            "spread_velocity_bps": spread_velocity,
            "microprice": microprice,
            "microprice_dislocation_bps": (microprice - mid) / mid * 10_000,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "depth_imbalance": imbalance,
            "weighted_depth_imbalance": weighted_imbalance,
            "bid_depth_slope_quantity_per_bps": self._depth_slope(bids, mid),
            "ask_depth_slope_quantity_per_bps": self._depth_slope(asks, mid),
            "aggressor_imbalance": aggressor_imbalance,
            "short_cvd_quote": cvd_quote,
            "liquidation_notional": liquidation_notional,
            "liquidation_imbalance": liquidation_imbalance,
            "liquidity_vacuum_ratio": liquidity_ratio,
            "book_replenishment_ratio": replenishment,
            "cancel_add_pressure": cancel_add_pressure,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "basis_bps": (
                (self.mark_price - self.index_price) / self.index_price * 10_000
                if self.mark_price is not None and self.index_price
                else None
            ),
            "funding_rate": self.funding_rate,
        }

    def market_fill(self, *, side: str, quantity: float, fee_bps: float = 5.0) -> dict[str, Any]:
        if side not in {"buy", "sell"} or quantity <= 0:
            raise ValueError("market fill requires buy/sell and positive quantity")
        levels = self._levels("ask" if side == "buy" else "bid", len(self.asks or self.bids))
        remaining = quantity
        notional = 0.0
        filled = 0.0
        for price, available in levels:
            take = min(remaining, available)
            filled += take
            notional += take * price
            remaining -= take
            if remaining <= 1e-12:
                break
        average = notional / filled if filled else None
        mid = None
        if self.bids and self.asks:
            mid = (max(self.bids) + min(self.asks)) / 2
        impact_bps = (
            (average - mid) / mid * 10_000 * (1 if side == "buy" else -1)
            if average is not None and mid
            else None
        )
        return {
            "side": side,
            "requested_quantity": quantity,
            "filled_quantity": filled,
            "remaining_quantity": remaining,
            "partial_fill": remaining > 1e-12,
            "average_price": average,
            "notional": notional,
            "fee": notional * fee_bps / 10_000,
            "impact_bps": impact_bps,
        }


@dataclass
class RestingLimitOrder:
    """Deterministic queue-ahead approximation for passive replay orders."""

    side: str
    price: float
    quantity: float
    submitted_ns: int
    latency_ns: int = 0
    cancel_after_ns: int | None = None
    maker_fee_bps: float = 1.0
    market: str = "futures"
    queue_ahead: float | None = None
    filled_quantity: float = 0.0
    fill_notional: float = 0.0
    first_fill_ns: int | None = None
    last_fill_ns: int | None = None
    canceled_ns: int | None = None
    last_book_quantity: float | None = None

    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell"}:
            raise ValueError("limit order side must be buy or sell")
        for label, value in (("price", self.price), ("quantity", self.quantity)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"limit order {label} must be finite and positive")
        if self.latency_ns < 0 or (self.cancel_after_ns is not None and self.cancel_after_ns < 0):
            raise ValueError("limit order latency/cancel delay must be non-negative")
        if self.market not in {"spot", "futures"}:
            raise ValueError("limit order market must be spot or futures")

    @property
    def active_ns(self) -> int:
        return self.submitted_ns + self.latency_ns

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def terminal(self) -> bool:
        return self.remaining_quantity <= 1e-12 or self.canceled_ns is not None

    def _book_quantity(self, state: MicrostructureState) -> float:
        book = state.bids if self.side == "buy" else state.asks
        return max(0.0, float(book.get(self.price, 0.0)))

    def _activate(self, received_ns: int, state: MicrostructureState) -> None:
        if self.queue_ahead is not None or received_ns < self.active_ns:
            return
        if state.bids and state.asks:
            best_bid, best_ask = max(state.bids), min(state.asks)
            if (self.side == "buy" and self.price >= best_ask) or (
                self.side == "sell" and self.price <= best_bid
            ):
                raise ValueError("passive limit order would cross the replayed book")
        self.queue_ahead = self._book_quantity(state)
        self.last_book_quantity = self.queue_ahead

    def observe(self, event: dict[str, Any], state: MicrostructureState) -> None:
        received_ns = int(event["received_ns"])
        self._activate(received_ns, state)
        if self.queue_ahead is None or self.terminal:
            return
        if (
            self.cancel_after_ns is not None
            and received_ns >= self.active_ns + self.cancel_after_ns
        ):
            self.canceled_ns = received_ns
            return
        stream = str(event.get("stream") or "")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        if "depth" in stream or "bookTicker" in stream:
            current = self._book_quantity(state)
            if self.last_book_quantity is not None and current < self.last_book_quantity:
                self.queue_ahead = max(0.0, self.queue_ahead - (self.last_book_quantity - current))
            self.last_book_quantity = current
            return
        if "aggTrade" not in stream and not stream.endswith("@trade"):
            return
        trade_price = _number(payload["p"])
        trade_quantity = _number(payload["q"])
        buyer_is_maker = bool(payload.get("m"))
        executable = (self.side == "buy" and buyer_is_maker and trade_price <= self.price) or (
            self.side == "sell" and not buyer_is_maker and trade_price >= self.price
        )
        if not executable:
            return
        ahead_consumed = min(self.queue_ahead, trade_quantity)
        self.queue_ahead -= ahead_consumed
        available = trade_quantity - ahead_consumed
        fill = min(self.remaining_quantity, available)
        if fill <= 0:
            return
        self.filled_quantity += fill
        self.fill_notional += fill * self.price
        self.first_fill_ns = self.first_fill_ns or received_ns
        self.last_fill_ns = received_ns

    def result(
        self,
        state: MicrostructureState,
        *,
        finished_ns: int | None,
        funding_rate_per_8h: float | None = None,
    ) -> dict[str, Any]:
        average = self.fill_notional / self.filled_quantity if self.filled_quantity else None
        mid = None
        if state.bids and state.asks:
            mid = (max(state.bids) + min(state.asks)) / 2
        adverse_selection_bps = None
        if average is not None and mid:
            adverse_selection_bps = (
                (average - mid) / average * 10_000
                if self.side == "buy"
                else (mid - average) / average * 10_000
            )
        duration_hours = 0.0
        if finished_ns is not None and self.first_fill_ns is not None:
            duration_hours = max(0.0, (finished_ns - self.first_fill_ns) / 3_600_000_000_000)
        funding = 0.0
        if self.market == "futures" and funding_rate_per_8h is not None:
            funding = self.fill_notional * funding_rate_per_8h * duration_hours / 8.0
        return {
            "side": self.side,
            "price": self.price,
            "requested_quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "fill_ratio": self.filled_quantity / self.quantity,
            "partial_fill": 0 < self.filled_quantity < self.quantity,
            "average_price": average,
            "maker_fee": self.fill_notional * self.maker_fee_bps / 10_000,
            "funding_cost": funding,
            "queue_ahead_remaining": self.queue_ahead,
            "active_ns": self.active_ns,
            "first_fill_ns": self.first_fill_ns,
            "last_fill_ns": self.last_fill_ns,
            "canceled_ns": self.canceled_ns,
            "adverse_selection_bps": adverse_selection_bps,
        }
