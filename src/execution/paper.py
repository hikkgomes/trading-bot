"""In-memory paper broker: simulated market fills with fees + slippage.

Deterministic and dependency-free (the price source is injectable), so it is
usable both in tests and as the execution backend for development. Realised PnL
is booked into the balance when a position is reduced/closed; the fee/slippage
model mirrors the backtester's ``2 * (fee_bps + slippage_bps) / 1e4`` per round
trip (here applied per fill).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from src.execution.broker import Broker, Fill, Order, OrderSide, OrderType, Position

PriceSource = Callable[[str], float]


class PaperBroker(Broker):
    name = "paper"

    def __init__(
        self,
        price_source: PriceSource,
        starting_balance: float = 10_000.0,
        fee_bps: float = 4.0,
        slippage_bps: float = 2.0,
    ):
        self._price_source = price_source
        self._balance = _finite_non_negative("starting_balance", starting_balance)
        self.fee_bps = _finite_non_negative("fee_bps", fee_bps)
        self.slippage_bps = _finite_non_negative("slippage_bps", slippage_bps)
        if self.slippage_bps >= 10_000:
            raise ValueError("slippage_bps must be less than 10000.")
        self._positions: dict[str, Position] = {}
        self.fills: list[Fill] = []

    # -- market data --------------------------------------------------------
    def get_price(self, symbol: str) -> float:
        price = float(self._price_source(symbol))
        if not math.isfinite(price) or price <= 0:
            raise ValueError(
                f"Paper price for {symbol} must be finite and positive, got {price:g}."
            )
        return price

    def get_balance(self) -> float:
        return self._balance

    def equity(self) -> float:
        """Balance + unrealised PnL across open positions (mark-to-market)."""
        total = self._balance
        for sym, pos in self._positions.items():
            if not pos.is_flat:
                mark = self.get_price(sym)
                total += pos.qty * (mark - pos.avg_price)
        return total

    def get_position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol=symbol))

    # -- order handling -----------------------------------------------------
    def _fill_price(self, side: OrderSide, ref: float) -> float:
        slip = self.slippage_bps / 10_000.0
        return ref * (1 + slip) if side == OrderSide.BUY else ref * (1 - slip)

    def place_order(self, order: Order) -> Fill:
        self._validate_quantity(order)
        ref = self._reference_price(order)
        fill_price = self._validated_fill_price(order.side, ref)
        if self._enforces_reduce_only() and order.reduce_only:
            self._assert_reduce_only_order(order)
        fee = fill_price * order.qty * (self.fee_bps / 10_000.0)
        self._balance -= fee

        pos = self.get_position(order.symbol)
        signed = order.qty if order.side == OrderSide.BUY else -order.qty
        new_qty = pos.qty + signed

        # Realise PnL on the portion that reduces/closes the existing position.
        self._realise_pnl(pos, signed, fill_price)
        avg = self._average_entry(pos, new_qty, fill_price, order.qty)

        self._positions[order.symbol] = Position(symbol=order.symbol, qty=new_qty, avg_price=avg)
        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fee=fee,
            exchange_order_id=f"paper-{len(self.fills) + 1}",
            client_order_id=order.client_id,
        )
        self.fills.append(fill)
        return fill

    @staticmethod
    def _validate_quantity(order: Order) -> None:
        if not math.isfinite(float(order.qty)) or order.qty <= 0:
            raise ValueError("Order qty must be positive.")

    def _reference_price(self, order: Order) -> float:
        if order.type == OrderType.LIMIT:
            if order.price is None:
                raise ValueError("Limit order price is required.")
            reference = float(order.price)
        else:
            reference = self.get_price(order.symbol)
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError(f"Order price must be finite and positive, got {reference:g}.")
        return reference

    def _validated_fill_price(self, side: OrderSide, reference: float) -> float:
        fill_price = self._fill_price(side, reference)
        if not math.isfinite(fill_price) or fill_price <= 0:
            raise ValueError(f"Fill price must be finite and positive, got {fill_price:g}.")
        return fill_price

    def _realise_pnl(self, position: Position, signed: float, fill_price: float) -> None:
        if position.qty == 0 or (position.qty > 0) == (signed > 0):
            return
        closing = min(abs(signed), abs(position.qty))
        direction = 1 if position.qty > 0 else -1
        self._balance += direction * closing * (fill_price - position.avg_price)

    @staticmethod
    def _average_entry(
        position: Position, new_quantity: float, fill_price: float, order_quantity: float
    ) -> float:
        if new_quantity == 0:
            return 0.0
        if (position.qty >= 0) == (new_quantity >= 0) and abs(new_quantity) > abs(position.qty):
            return (position.avg_price * abs(position.qty) + fill_price * order_quantity) / abs(
                new_quantity
            )
        if (position.qty > 0) != (new_quantity > 0):
            return fill_price
        return position.avg_price

    def _enforces_reduce_only(self) -> bool:
        return getattr(getattr(self, "config", None), "market_type", "futures") == "futures"

    def _assert_reduce_only_order(self, order: Order) -> None:
        pos = self.get_position(order.symbol)
        if pos.is_flat:
            raise ValueError("Reduce-only paper order requires an open position.")
        if pos.qty > 0 and order.side != OrderSide.SELL:
            raise ValueError("Reduce-only paper order side must reduce the current long position.")
        if pos.qty < 0 and order.side != OrderSide.BUY:
            raise ValueError("Reduce-only paper order side must reduce the current short position.")
        if order.qty > abs(pos.qty) + 1e-12:
            raise ValueError(
                f"Reduce-only paper order quantity {order.qty:g} exceeds open position {abs(pos.qty):g}."
            )


def _finite_non_negative(name: str, value: float) -> float:
    try:
        clean = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(clean) or clean < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return clean


def binance_mark_price(symbol: str = "BTCUSDT", market: str = "futures") -> float:
    """Public mark price (no API key). Lazy ``requests`` import.

    market: "futures" (USDM) or "spot".
    """
    import requests

    base = (
        "https://fapi.binance.com/fapi/v1"
        if market == "futures"
        else "https://api.binance.com/api/v3"
    )
    path = "/premiumIndex" if market == "futures" else "/ticker/price"
    resp = requests.get(f"{base}{path}", params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return float(data["markPrice"] if market == "futures" else data["price"])
