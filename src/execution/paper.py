"""In-memory paper broker: simulated market fills with fees + slippage.

Deterministic and dependency-free (the price source is injectable), so it is
usable both in tests and as the execution backend for development. Realised PnL
is booked into the balance when a position is reduced/closed; the fee/slippage
model mirrors the backtester's ``2 * (fee_bps + slippage_bps) / 1e4`` per round
trip (here applied per fill).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

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
        self._balance = float(starting_balance)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self._positions: Dict[str, Position] = {}
        self.fills: List[Fill] = []

    # -- market data --------------------------------------------------------
    def get_price(self, symbol: str) -> float:
        return float(self._price_source(symbol))

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
        if order.qty <= 0:
            raise ValueError("Order qty must be positive.")
        ref = order.price if (order.type == OrderType.LIMIT and order.price) else self.get_price(order.symbol)
        fill_price = self._fill_price(order.side, ref)
        fee = fill_price * order.qty * (self.fee_bps / 10_000.0)
        self._balance -= fee

        pos = self.get_position(order.symbol)
        signed = order.qty if order.side == OrderSide.BUY else -order.qty
        new_qty = pos.qty + signed

        # Realise PnL on the portion that reduces/closes the existing position.
        if pos.qty != 0 and (pos.qty > 0) != (signed > 0):
            closing = min(abs(signed), abs(pos.qty))
            direction = 1 if pos.qty > 0 else -1
            self._balance += direction * closing * (fill_price - pos.avg_price)

        if new_qty == 0:
            avg = 0.0
        elif (pos.qty >= 0) == (new_qty >= 0) and abs(new_qty) > abs(pos.qty):
            # Adding to the position -> weighted-average entry.
            avg = (pos.avg_price * abs(pos.qty) + fill_price * order.qty) / abs(new_qty)
        elif (pos.qty > 0) != (new_qty > 0) and new_qty != 0:
            # Flipped through zero -> new entry at fill price.
            avg = fill_price
        else:
            avg = pos.avg_price  # reduced but same side

        self._positions[order.symbol] = Position(symbol=order.symbol, qty=new_qty, avg_price=avg)
        fill = Fill(symbol=order.symbol, side=order.side, qty=order.qty, price=fill_price, fee=fee)
        self.fills.append(fill)
        return fill


def binance_mark_price(symbol: str = "BTCUSDT", market: str = "futures") -> float:
    """Public mark price (no API key). Lazy ``requests`` import.

    market: "futures" (USDM) or "spot".
    """
    import requests

    base = "https://fapi.binance.com/fapi/v1" if market == "futures" else "https://api.binance.com/api/v3"
    path = "/premiumIndex" if market == "futures" else "/ticker/price"
    resp = requests.get(f"{base}{path}", params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return float(data["markPrice"] if market == "futures" else data["price"])
