"""Live/testnet futures broker over ccxt (any supported venue).

ccxt is an optional dependency (not in requirements-bot.txt). Install with
``pip install ccxt`` on the execution machine to use this adapter.

Safety rails — a live order is only sent when **all** of these hold:
* ``config.live`` is True   (TRADING_LIVE=1)
* the order notional is <= ``config.max_notional_usd``
Otherwise placing an order raises, so a misconfigured run can't trade real size.
Set ``EXCHANGE_TESTNET=1`` to route everything to the exchange sandbox.
"""

from __future__ import annotations

import logging

from src.execution.broker import Broker, Fill, Order, OrderSide, Position
from src.execution.config import ExchangeConfig

LOGGER = logging.getLogger(__name__)


class CcxtBroker(Broker):
    def __init__(self, config: ExchangeConfig | None = None):
        self.config = config or ExchangeConfig.from_env()
        self.name = f"ccxt:{self.config.exchange}{'(testnet)' if self.config.testnet else ''}"
        self._client = self._build_client()

    def _build_client(self):
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "ccxt is not installed. Run `pip install ccxt` to use CcxtBroker, "
                "or use PaperBroker for simulated execution."
            ) from exc

        if not hasattr(ccxt, self.config.exchange):
            raise ValueError(f"ccxt has no exchange {self.config.exchange!r}.")
        klass = getattr(ccxt, self.config.exchange)
        client = klass({
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "password": self.config.api_password or None,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if self.config.testnet and hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)
        return client

    # -- market data --------------------------------------------------------
    def get_price(self, symbol: str) -> float:
        return float(self._client.fetch_ticker(symbol)["last"])

    def get_balance(self) -> float:
        bal = self._client.fetch_balance()
        free = bal.get("free", {})
        # USDT-margined by default; fall back to total USDT.
        return float(free.get("USDT", bal.get("total", {}).get("USDT", 0.0)) or 0.0)

    def get_position(self, symbol: str) -> Position:
        positions = self._client.fetch_positions([symbol])
        for p in positions:
            contracts = float(p.get("contracts") or 0.0)
            if contracts:
                side = p.get("side")
                qty = contracts if side == "long" else -contracts
                return Position(symbol=symbol, qty=qty, avg_price=float(p.get("entryPrice") or 0.0))
        return Position(symbol=symbol)

    # -- orders -------------------------------------------------------------
    def place_order(self, order: Order) -> Fill:
        ref_price = order.price or self.get_price(order.symbol)
        notional = ref_price * order.qty
        if notional > self.config.max_notional_usd:
            raise ValueError(
                f"Order notional ${notional:,.2f} exceeds MAX_NOTIONAL_USD "
                f"${self.config.max_notional_usd:,.2f}. Refusing."
            )
        if not self.config.live:
            raise RuntimeError(
                "Refusing to place a real order: TRADING_LIVE is not enabled. "
                "Set TRADING_LIVE=1 (and ideally EXCHANGE_TESTNET=1) to trade."
            )

        params = {"reduceOnly": True} if order.reduce_only else {}
        result = self._client.create_order(
            symbol=order.symbol, type=order.type.value, side=order.side.value,
            amount=order.qty, price=order.price, params=params,
        )
        fill_price = float(result.get("average") or result.get("price") or ref_price)
        fee = float((result.get("fee") or {}).get("cost") or 0.0)
        filled = float(result.get("filled") or order.qty)
        LOGGER.info("Placed %s %s %s @ %s", order.side.value, filled, order.symbol, fill_price)
        return Fill(symbol=order.symbol, side=order.side, qty=filled, price=fill_price, fee=fee)
