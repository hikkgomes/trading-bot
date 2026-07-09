"""Live/testnet broker over ccxt (spot or futures).

ccxt is an optional dependency (not in requirements-bot.txt). Install with
``pip install ccxt`` on the execution machine to use this adapter.

Safety rails — a live order is only sent when **all** of these hold:
* ``config.live`` is True   (TRADING_LIVE=1)
* entry/increase order notional is <= ``config.max_notional_usd``
* futures reduce-only closes may exceed ``config.max_notional_usd`` so
  emergency flatten can reduce existing risk
* the filled price is within ``config.max_fill_slippage_bps`` of the reference
  price used before submission
* the filled quantity is present, positive, and matches the requested order
  quantity within a tiny numerical tolerance
* spot sell quantity is <= current base-asset balance (no margin/shorting)
* futures reduce-only orders are <= the current broker position on the matching side
* futures margin mode is explicitly set to isolated
* futures leverage is explicitly set to ``config.max_futures_leverage``
Otherwise placing an order raises, so a misconfigured run can't trade real size.
Set ``EXCHANGE_TESTNET=1`` to route everything to the exchange sandbox where the
exchange supports one.
"""

from __future__ import annotations

import logging
import math
import re

from src.execution.broker import Broker, Fill, Order, OrderSide, OrderType, Position
from src.execution.config import ExchangeConfig

LOGGER = logging.getLogger(__name__)
QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")


class CcxtBroker(Broker):
    def __init__(self, config: ExchangeConfig | None = None):
        self.config = config or ExchangeConfig.from_env()
        self.name = (
            f"ccxt:{self.config.exchange}:{self.config.market_type}"
            f"{'(testnet)' if self.config.testnet else ''}"
        )
        self._client = self._build_client()
        self._leverage_set_symbols: set[str] = set()
        self._margin_mode_set_symbols: set[str] = set()

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
        default_type = "future" if self.config.market_type == "futures" else "spot"
        client = klass({
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "password": self.config.api_password or None,
            "enableRateLimit": True,
            "options": {"defaultType": default_type},
        })
        if self.config.testnet and hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)
        return client

    # -- market data --------------------------------------------------------
    def get_price(self, symbol: str) -> float:
        price = self._finite_number(
            self._client.fetch_ticker(self._ccxt_symbol(symbol)).get("last"),
            "Ticker price",
            positive=True,
        )
        return price

    def get_balance(self) -> float:
        bal = self._client.fetch_balance()
        free = bal.get("free", {})
        value = free.get(self.config.quote_asset, bal.get("total", {}).get(self.config.quote_asset, 0.0))
        return self._finite_number(0.0 if value is None else value, "Quote balance", non_negative=True)

    def get_position(self, symbol: str) -> Position:
        if self.config.market_type == "spot":
            base_asset = self._base_asset(symbol)
            bal = self._client.fetch_balance()
            total = bal.get("total", {})
            value = total.get(base_asset, bal.get(base_asset, {}).get("total", 0.0))
            qty = self._finite_number(
                0.0 if value is None else value,
                f"Spot {base_asset} position quantity",
                non_negative=True,
            )
            return Position(symbol=symbol, qty=qty, avg_price=0.0)

        positions = self._client.fetch_positions([self._ccxt_symbol(symbol)])
        for p in positions:
            contracts_value = p.get("contracts")
            contracts = self._finite_number(
                0.0 if contracts_value is None else contracts_value,
                "Futures position contracts",
                non_negative=True,
            )
            if contracts:
                side = p.get("side")
                if side not in {"long", "short"}:
                    raise ValueError(f"Futures position side must be long or short, got {side!r}.")
                entry_value = p.get("entryPrice")
                avg_price = self._finite_number(
                    0.0 if entry_value is None else entry_value,
                    "Futures position entry price",
                    positive=True,
                )
                qty = contracts if side == "long" else -contracts
                return Position(symbol=symbol, qty=qty, avg_price=avg_price)
        return Position(symbol=symbol)

    # -- orders -------------------------------------------------------------
    def _ensure_futures_margin_mode(self, symbol: str) -> None:
        if self.config.market_type != "futures":
            return
        client_symbol = self._ccxt_symbol(symbol)
        margin_mode = str(self.config.futures_margin_mode).lower()
        if margin_mode != "isolated":
            raise ValueError("FUTURES_MARGIN_MODE must be 'isolated' for live futures entries.")
        configured = getattr(self, "_margin_mode_set_symbols", set())
        if client_symbol in configured:
            return
        if not hasattr(self._client, "set_margin_mode"):
            raise RuntimeError(
                "Refusing futures order: ccxt client cannot set isolated margin mode, "
                "so account margin risk cannot be bounded."
            )
        try:
            self._client.set_margin_mode(margin_mode, client_symbol)
        except Exception as exc:
            message = str(exc).lower()
            already_set = "no need to change margin type" in message or "already" in message
            if not already_set:
                raise RuntimeError(f"Refusing futures order: could not set isolated margin mode: {exc}") from exc
        configured.add(client_symbol)
        self._margin_mode_set_symbols = configured

    def _ensure_futures_leverage(self, symbol: str) -> None:
        if self.config.market_type != "futures":
            return
        client_symbol = self._ccxt_symbol(symbol)
        leverage = int(self.config.max_futures_leverage)
        if not (1 <= leverage <= 3):
            raise ValueError("MAX_FUTURES_LEVERAGE must be between 1 and 3.")
        configured = getattr(self, "_leverage_set_symbols", set())
        if client_symbol in configured:
            return
        if not hasattr(self._client, "set_leverage"):
            raise RuntimeError(
                "Refusing futures order: ccxt client cannot set leverage, "
                "so account leverage cannot be bounded."
            )
        self._client.set_leverage(leverage, client_symbol)
        configured.add(client_symbol)
        self._leverage_set_symbols = configured

    def place_order(self, order: Order) -> Fill:
        if not math.isfinite(float(order.qty)) or order.qty <= 0:
            raise ValueError(f"Order quantity must be positive, got {order.qty:g}. Refusing.")
        ref_price = self._reference_price(order)
        notional = ref_price * order.qty
        enforce_notional_cap = not self._is_futures_reduce_only(order)
        if enforce_notional_cap:
            self._assert_notional_within_cap(notional)
        if not self.config.live:
            raise RuntimeError(
                "Refusing to place a real order: TRADING_LIVE is not enabled. "
                "Set TRADING_LIVE=1 (and ideally EXCHANGE_TESTNET=1) to trade."
            )
        if self.config.market_type == "spot" and order.side == OrderSide.BUY:
            self._assert_spot_buy_within_balance(order, ref_price)
        if self.config.market_type == "spot" and order.side == OrderSide.SELL:
            self._assert_spot_sell_within_balance(order)
        if self._is_futures_reduce_only(order):
            self._assert_futures_reduce_only_within_position(order)
        if self.config.market_type == "futures" and not order.reduce_only:
            self._ensure_futures_margin_mode(order.symbol)
            self._ensure_futures_leverage(order.symbol)

        client_symbol = self._ccxt_symbol(order.symbol)
        params = {}
        if order.reduce_only and self.config.market_type == "futures":
            params["reduceOnly"] = True
        result = self._client.create_order(
            symbol=client_symbol, type=order.type.value, side=order.side.value,
            amount=order.qty, price=order.price, params=params,
        )
        self._assert_order_status_accepted(result)
        self._assert_order_response_matches(result, order)
        fill_price = float(self._required_first_present(result, ("average", "price"), label="Filled order price"))
        fee = float((result.get("fee") or {}).get("cost") or 0.0)
        filled = float(self._required_first_present(result, ("filled",), label="Filled order quantity"))
        if not math.isfinite(fee) or fee < 0:
            raise ValueError(f"Filled order fee must be finite and non-negative, got {fee:g}. Refusing.")
        if not math.isfinite(fill_price) or fill_price <= 0:
            raise ValueError(f"Filled order price must be positive, got {fill_price:g}. Refusing.")
        if not math.isfinite(filled) or filled <= 0:
            raise ValueError(f"Filled order quantity must be positive, got {filled:g}. Refusing.")
        self._assert_filled_quantity_matches_request(float(order.qty), filled)
        self._assert_fill_slippage_within_cap(ref_price, fill_price)
        if enforce_notional_cap:
            self._assert_notional_within_cap(fill_price * filled, label="Filled order")
        LOGGER.info("Placed %s %s %s @ %s", order.side.value, filled, order.symbol, fill_price)
        return Fill(symbol=order.symbol, side=order.side, qty=filled, price=fill_price, fee=fee)

    def _is_futures_reduce_only(self, order: Order) -> bool:
        return self.config.market_type == "futures" and bool(order.reduce_only)

    @staticmethod
    def _required_first_present(payload: dict, keys: tuple[str, ...], *, label: str):
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value
        raise ValueError(f"{label} missing from exchange response. Refusing to assume order fill.")

    @staticmethod
    def _assert_order_status_accepted(payload: dict) -> None:
        status = payload.get("status")
        if status is None:
            return
        normalized = str(status).lower()
        if normalized not in {"closed", "filled"}:
            raise ValueError(f"Exchange order status {status!r} is not closed/filled. Refusing to accept fill.")

    def _assert_order_response_matches(self, payload: dict, order: Order) -> None:
        response_side = payload.get("side")
        if response_side is not None and str(response_side).lower() != order.side.value:
            raise ValueError(
                f"Exchange order side {response_side!r} does not match requested side "
                f"{order.side.value!r}. Refusing to accept fill."
            )
        response_symbol = payload.get("symbol")
        if response_symbol is not None and not self._symbols_match(str(response_symbol), order.symbol):
            raise ValueError(
                f"Exchange order symbol {response_symbol!r} does not match requested symbol "
                f"{order.symbol!r}. Refusing to accept fill."
            )

    def _reference_price(self, order: Order) -> float:
        if order.type == OrderType.LIMIT and order.price is None:
            raise ValueError("Limit order price is required. Refusing.")
        if order.price is not None:
            price = float(order.price)
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"Order price must be positive, got {price:g}. Refusing.")
            return price
        price = float(self.get_price(order.symbol))
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Reference price must be positive, got {price:g}. Refusing.")
        return price

    def _assert_notional_within_cap(self, notional: float, *, label: str = "Order") -> None:
        if not math.isfinite(notional) or notional <= 0:
            raise ValueError(f"{label} notional must be finite and positive, got {notional:g}. Refusing.")
        max_notional = float(self.config.max_notional_usd)
        if not math.isfinite(max_notional) or max_notional <= 0:
            raise ValueError(
                f"MAX_NOTIONAL_USD must be finite and positive, got {max_notional:g}. Refusing."
            )
        if notional > max_notional:
            raise ValueError(
                f"{label} notional ${notional:,.2f} exceeds MAX_NOTIONAL_USD "
                f"${max_notional:,.2f}. Refusing."
            )

    def _assert_fill_slippage_within_cap(self, ref_price: float, fill_price: float) -> None:
        max_bps = float(self.config.max_fill_slippage_bps)
        if not math.isfinite(max_bps) or max_bps <= 0:
            raise ValueError(f"MAX_FILL_SLIPPAGE_BPS must be finite and positive, got {max_bps:g}. Refusing.")
        slippage_bps = abs(float(fill_price) - float(ref_price)) / float(ref_price) * 10_000.0
        if slippage_bps > max_bps:
            raise ValueError(
                f"Filled order slippage {slippage_bps:.2f} bps exceeds MAX_FILL_SLIPPAGE_BPS "
                f"{max_bps:.2f}. Refusing."
            )

    @staticmethod
    def _assert_filled_quantity_matches_request(requested_qty: float, filled: float) -> None:
        tolerance = max(1e-12, requested_qty * 1e-9)
        if filled - requested_qty > tolerance:
            raise ValueError(
                f"Filled order quantity {filled:g} exceeds requested quantity {requested_qty:g}. Refusing."
            )
        if requested_qty - filled > tolerance:
            raise ValueError(
                f"Filled order quantity {filled:g} is less than requested quantity {requested_qty:g}. "
                "Refusing to accept a partial fill."
            )

    def _assert_futures_reduce_only_within_position(self, order: Order) -> None:
        position = self.get_position(order.symbol)
        if order.side == OrderSide.SELL:
            reducible_qty = float(position.qty)
            required_side = "long"
        else:
            reducible_qty = -float(position.qty)
            required_side = "short"
        if not math.isfinite(reducible_qty) or reducible_qty <= 0:
            raise ValueError(
                f"Reduce-only {order.side.value} requires an existing {required_side} futures position. "
                f"Broker reports qty {position.qty:g}. Refusing."
            )
        tolerance = max(reducible_qty * 1e-9, 1e-12)
        if float(order.qty) - reducible_qty > tolerance:
            raise ValueError(
                f"Reduce-only quantity {order.qty:g} exceeds current futures position "
                f"{abs(float(position.qty)):g}. Refusing."
            )

    @staticmethod
    def _finite_number(value, label: str, *, positive: bool = False, non_negative: bool = False) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric, got {value!r}. Refusing.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite, got {number:g}. Refusing.")
        if positive and number <= 0:
            raise ValueError(f"{label} must be positive, got {number:g}. Refusing.")
        if non_negative and number < 0:
            raise ValueError(f"{label} must be non-negative, got {number:g}. Refusing.")
        return number

    def _assert_spot_sell_within_balance(self, order: Order) -> None:
        position = self.get_position(order.symbol)
        available = max(float(position.qty or 0.0), 0.0)
        tolerance = max(1e-12, available * 1e-9)
        if order.qty - available > tolerance:
            base_asset = self._base_asset(order.symbol)
            raise ValueError(
                f"Spot sell quantity {order.qty:g} {base_asset} exceeds available "
                f"{available:g} {base_asset}. Refusing to short spot."
            )

    def _assert_spot_buy_within_balance(self, order: Order, ref_price: float) -> None:
        available = self.get_balance()
        required = float(order.qty) * float(ref_price)
        tolerance = max(1e-12, available * 1e-9)
        if required - available > tolerance:
            raise ValueError(
                f"Spot buy notional {required:g} {self.config.quote_asset} exceeds available "
                f"{available:g} {self.config.quote_asset}. Refusing to overspend spot quote balance."
            )

    @staticmethod
    def _base_asset(symbol: str) -> str:
        if "/" in symbol:
            return symbol.split("/", 1)[0].upper()
        compact = re.sub(r"[^A-Za-z]", "", symbol).upper()
        for quote in QUOTE_ASSETS:
            if compact.endswith(quote) and len(compact) > len(quote):
                return compact[: -len(quote)]
        return compact

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str, str | None]:
        raw = re.sub(r"\s+", "", str(symbol or "").upper())
        if not raw:
            raise ValueError("Symbol must be non-empty.")
        settlement = None
        pair = raw
        if ":" in pair:
            pair, settlement = pair.split(":", 1)
        if "/" in pair:
            base, quote = pair.split("/", 1)
            return base, quote, settlement
        compact = re.sub(r"[^A-Z0-9]", "", pair)
        for quote in QUOTE_ASSETS:
            if compact.endswith(quote) and len(compact) > len(quote):
                return compact[: -len(quote)], quote, settlement
        raise ValueError(f"Symbol {symbol!r} must include base and quote assets.")

    def _symbols_match(self, left: str, right: str) -> bool:
        left_base, left_quote, left_settlement = self._split_symbol(left)
        right_base, right_quote, right_settlement = self._split_symbol(right)
        if (left_base, left_quote) != (right_base, right_quote):
            return False
        if left_settlement is not None and right_settlement is not None:
            return left_settlement == right_settlement
        return True

    def _ccxt_symbol(self, symbol: str) -> str:
        base, quote, settlement = self._split_symbol(symbol)
        if self.config.market_type == "futures":
            return f"{base}/{quote}:{settlement or quote}"
        return f"{base}/{quote}"
