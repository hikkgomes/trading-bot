"""Live/testnet broker over ccxt (spot or futures).

ccxt is pinned in ``requirements-bot.txt`` because live Binance conditional
order routing is version-sensitive.

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
* futures margin mode, leverage, and position mode are read back before every entry
Otherwise placing an order raises, so a misconfigured run can't trade real size.
Set ``EXCHANGE_TESTNET=1`` to route everything to the exchange sandbox where the
exchange supports one.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from src.execution.broker import (
    Broker,
    BrokerFill,
    BrokerIncome,
    BrokerOrderAcknowledgement,
    BrokerOrderState,
    Fill,
    FuturesPositionIdentity,
    OpenOrderIdentity,
    Order,
    OrderSide,
    OrderType,
    Position,
    ProtectiveOrder,
    ProtectiveOrderStatus,
)
from src.execution.config import ExchangeConfig
from src.execution.rate_limit import RateLimitedExchangeClient, shared_exchange_rate_limiter

LOGGER = logging.getLogger(__name__)
QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")
CLIENT_ORDER_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,36}$")
NATIVE_STOP_CCXT_VERSION = "4.5.64"
BINANCE_MARGIN_MODE_ALREADY_SET_CODE = -4046
BINANCE_MARGIN_MODE_ALREADY_SET_MESSAGE = "No need to change margin type."
BINANCE_POSITION_MODE_ALREADY_SET_CODE = -4059
BINANCE_POSITION_MODE_ALREADY_SET_MESSAGE = "No need to change position side."


class CcxtBroker(Broker):
    def __init__(self, config: ExchangeConfig | None = None):
        self.config = config or ExchangeConfig.from_env()
        self.name = (
            f"ccxt:{self.config.exchange}:{self.config.market_type}"
            f"{'(testnet)' if self.config.testnet else ''}"
        )
        self._client = self._build_client()
        self._precision_markets: dict[str, dict] = {}
        self._acknowledgement_only = False
        self._submission_responses: dict[str, dict] = {}

    def submit_order(self, order: Order) -> BrokerOrderAcknowledgement:
        """Accept Binance ACK/RESULT/NEW responses without inventing a fill."""

        self._acknowledgement_only = True
        try:
            acknowledgement = self.place_order(order)
        finally:
            self._acknowledgement_only = False
        if not isinstance(acknowledgement, BrokerOrderAcknowledgement):
            raise RuntimeError("ccxt submission did not return an acknowledgement")
        return acknowledgement

    def query_order(
        self, *, symbol: str, exchange_order_id: str, client_order_id: str
    ) -> BrokerOrderState:
        if not exchange_order_id and not client_order_id:
            raise RuntimeError("exchange order lookup requires an exchange or client order ID")
        payload = self._fetch_order_state(
            symbol=symbol,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
        )
        return self._parse_order_state(
            payload,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
        )

    def _fetch_order_state(
        self, *, symbol: str, exchange_order_id: str, client_order_id: str
    ) -> dict | None:
        responses = getattr(self, "_submission_responses", {})
        cached = responses.get(exchange_order_id)
        if cached is None and client_order_id:
            cached = next(
                (
                    response
                    for response in responses.values()
                    if self._response_client_id(response) == client_order_id
                ),
                None,
            )
        fetch = getattr(self._client, "fetch_order", None)
        if callable(fetch):
            params = {"origClientOrderId": client_order_id} if not exchange_order_id else {}
            payload = fetch(exchange_order_id, self._ccxt_symbol(symbol), params)
        else:
            payload = cached
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _response_client_id(response: dict) -> str:
        raw_info = response.get("info")
        info = raw_info if isinstance(raw_info, dict) else {}
        return str(response.get("clientOrderId") or info.get("clientOrderId") or "")

    @staticmethod
    def _parse_order_state(
        payload: dict | None, *, exchange_order_id: str, client_order_id: str
    ) -> BrokerOrderState:
        if payload is None:
            raise RuntimeError("exchange order state is unavailable")
        raw_info = payload.get("info")
        info = raw_info if isinstance(raw_info, dict) else {}
        status = str(payload.get("status") or info.get("status") or "unknown").lower()
        try:
            filled = float(payload.get("filled") or info.get("executedQty") or 0.0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("exchange order state has an invalid filled quantity") from exc
        if not math.isfinite(filled) or filled < 0:
            raise RuntimeError("exchange order state has an invalid filled quantity")
        average_raw = payload.get("average") or payload.get("price")
        try:
            average = float(average_raw) if average_raw not in {None, ""} and filled > 0 else None
        except (TypeError, ValueError) as exc:
            raise RuntimeError("exchange order state has an invalid average price") from exc
        if average is not None and (not math.isfinite(average) or average <= 0):
            raise RuntimeError("exchange order state has an invalid average price")
        resolved_exchange_order_id = str(
            payload.get("id") or info.get("orderId") or exchange_order_id
        ).strip()
        resolved_client_order_id = str(
            payload.get("clientOrderId") or info.get("clientOrderId") or client_order_id
        ).strip()
        if not resolved_exchange_order_id:
            raise RuntimeError("exchange order state has no exchange order ID")
        if client_order_id and resolved_client_order_id != client_order_id:
            raise RuntimeError("exchange order state changed the client order ID")
        if exchange_order_id and resolved_exchange_order_id != exchange_order_id:
            raise RuntimeError("exchange order state changed the exchange order ID")
        return BrokerOrderState(
            exchange_order_id=resolved_exchange_order_id,
            client_order_id=resolved_client_order_id,
            status=status,
            filled_quantity=filled,
            average_price=average,
        )

    def query_order_fills(
        self, *, symbol: str, exchange_order_id: str, client_order_id: str
    ) -> tuple[BrokerFill, ...]:
        fetch = getattr(self._client, "fetch_my_trades", None)
        if not callable(fetch):
            raise RuntimeError("ccxt client cannot fetch account trades")
        params = {"orderId": exchange_order_id} if exchange_order_id else {}
        payload = fetch(self._ccxt_symbol(symbol), None, None, params)
        if not isinstance(payload, list):
            raise RuntimeError("exchange trade history response is not a list")
        fills: list[BrokerFill] = []
        for item in payload:
            if not isinstance(item, dict):
                raise RuntimeError("exchange trade history contains a non-object")
            info_value = item.get("info")
            info = info_value if isinstance(info_value, dict) else {}
            reported_symbol = item.get("symbol") or info.get("symbol")
            if reported_symbol is not None and not self._symbols_match(
                str(reported_symbol), symbol
            ):
                raise RuntimeError("exchange trade history returned a different symbol")
            raw_order_id = str(item.get("order") or info.get("orderId") or "").strip()
            raw_client_id = str(
                item.get("clientOrderId") or info.get("clientOrderId") or ""
            ).strip()
            if exchange_order_id and raw_order_id != exchange_order_id:
                continue
            if client_order_id and raw_client_id and raw_client_id != client_order_id:
                continue
            if not raw_order_id and not raw_client_id:
                continue
            trade_id = str(item.get("id") or info.get("id") or "").strip()
            if not trade_id:
                raise RuntimeError("exchange trade has no trade ID")
            quantity = self._finite_number(
                item.get("amount") or info.get("qty"), "Recovered trade quantity", positive=True
            )
            price = self._finite_number(
                item.get("price") or info.get("price"), "Recovered trade price", positive=True
            )
            side = self._trade_side(item, info)
            fee, fee_asset = self._trade_fee(item, info)
            occurred_at = self._trade_timestamp(item, info)
            fills.append(
                BrokerFill(
                    trade_id=trade_id,
                    exchange_order_id=raw_order_id or exchange_order_id,
                    client_order_id=raw_client_id or client_order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    fee=fee,
                    occurred_at=occurred_at,
                    fee_asset=fee_asset,
                )
            )
        return tuple(sorted(fills, key=lambda item: (item.occurred_at, item.trade_id)))

    def query_income(self, *, since: float | None = None) -> tuple[BrokerIncome, ...]:
        fetch = getattr(self._client, "fapiPrivateGetIncome", None)
        if not callable(fetch):
            raise RuntimeError("ccxt client cannot fetch futures income history")
        params = {"startTime": int(since * 1_000)} if since is not None else {}
        payload = fetch(params)
        if not isinstance(payload, list):
            raise RuntimeError("exchange income history response is not a list")
        result: list[BrokerIncome] = []
        for item in payload:
            if not isinstance(item, dict):
                raise RuntimeError("exchange income history contains a non-object")
            info_value = item.get("info")
            info = info_value if isinstance(info_value, dict) else {}
            income_id = str(
                item.get("id") or item.get("tranId") or info.get("tranId") or ""
            ).strip()
            if not income_id:
                raise RuntimeError("exchange income record has no identity")
            amount = self._finite_number(
                item.get("amount") or item.get("income") or info.get("income"),
                "Recovered income amount",
            )
            occurred_at = self._trade_timestamp(item, info)
            result.append(
                BrokerIncome(
                    income_id=income_id,
                    symbol=str(item.get("symbol") or info.get("symbol") or ""),
                    income_type=str(
                        item.get("type")
                        or item.get("incomeType")
                        or info.get("incomeType")
                        or "unknown"
                    ),
                    amount=amount,
                    asset=str(
                        item.get("currency") or item.get("asset") or info.get("asset") or "USDT"
                    ).upper(),
                    occurred_at=occurred_at,
                    exchange_order_id=str(
                        item.get("order") or item.get("orderId") or info.get("orderId") or ""
                    ).strip()
                    or None,
                    trade_id=str(
                        item.get("trade") or item.get("tradeId") or info.get("tradeId") or ""
                    ).strip()
                    or None,
                )
            )
        return tuple(sorted(result, key=lambda item: (item.occurred_at, item.income_id)))

    @staticmethod
    def _trade_side(item: dict, info: dict) -> OrderSide:
        raw = str(item.get("side") or "").lower()
        if raw not in {"buy", "sell"}:
            buyer = info.get("isBuyer")
            raw = "buy" if buyer is True or str(buyer).lower() == "true" else "sell"
        return OrderSide(raw)

    def _trade_fee(self, item: dict, info: dict) -> tuple[float, str | None]:
        fee_value = item.get("fee")
        fee = fee_value.get("cost") if isinstance(fee_value, dict) else None
        asset = fee_value.get("currency") if isinstance(fee_value, dict) else None
        fee = fee if fee is not None else info.get("commission", 0.0)
        asset = asset or info.get("commissionAsset")
        return self._finite_number(fee, "Recovered trade fee", non_negative=True), (
            str(asset).upper() if asset else None
        )

    @staticmethod
    def _trade_timestamp(item: dict, info: dict) -> float:
        raw = item.get("timestamp") or item.get("time") or info.get("time") or info.get("T")
        try:
            result = float(raw) / 1_000.0 if float(raw) > 10_000_000_000 else float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("exchange trade timestamp is invalid") from exc
        if not math.isfinite(result) or result <= 0:
            raise RuntimeError("exchange trade timestamp is invalid")
        return result

    def cancel_order(
        self, *, symbol: str, exchange_order_id: str, client_order_id: str
    ) -> BrokerOrderState:
        if not exchange_order_id and not client_order_id:
            raise RuntimeError("exchange order cancellation requires an order identity")
        cancel = getattr(self._client, "cancel_order", None)
        if not callable(cancel):
            raise RuntimeError("ccxt client cannot cancel exchange orders")
        lookup_id = exchange_order_id or client_order_id
        params = {"origClientOrderId": client_order_id} if not exchange_order_id else {}
        try:
            payload = cancel(lookup_id, self._ccxt_symbol(symbol), params)
        except Exception as exc:
            raise RuntimeError(f"could not cancel exchange order {lookup_id}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("exchange cancellation response is not an object")
        info_value = payload.get("info")
        info = info_value if isinstance(info_value, dict) else {}
        resolved_id = str(payload.get("id") or info.get("orderId") or exchange_order_id).strip()
        resolved_client = str(
            payload.get("clientOrderId") or info.get("clientOrderId") or client_order_id
        ).strip()
        if not resolved_id or not resolved_client:
            raise RuntimeError("exchange cancellation response has no complete order identity")
        if exchange_order_id and resolved_id != exchange_order_id:
            raise RuntimeError("exchange cancellation response changed the exchange order ID")
        if client_order_id and resolved_client != client_order_id:
            raise RuntimeError("exchange cancellation response changed the client order ID")
        status = str(payload.get("status") or info.get("status") or "canceled").lower()
        filled = float(payload.get("filled") or info.get("executedQty") or 0.0)
        average_raw = payload.get("average") or payload.get("price")
        average = float(average_raw) if average_raw not in {None, ""} and filled > 0 else None
        if not math.isfinite(filled) or filled < 0:
            raise RuntimeError("exchange cancellation response has invalid fill quantity")
        return BrokerOrderState(
            exchange_order_id=resolved_id,
            client_order_id=resolved_client,
            status=status,
            filled_quantity=filled,
            average_price=average,
        )

    @property
    def account_fingerprint(self) -> str:
        """Non-secret identity of the credential and venue this broker uses."""

        return self.config.account_fingerprint

    def account_snapshot(self, *, expected_symbols: tuple[str, ...] = ()) -> dict:
        """Read the complete authenticated account state used by live gates."""
        balance, balances, free_balances, info = self._account_balances()
        used_margin, maintenance_margin, used_fraction, liquidation_buffer = self._account_margin(
            balance, balances, info
        )
        positions, unknown_positions, regular_orders, conditional_orders = (
            self._account_positions_and_orders(balances, expected_symbols)
        )
        unknown_orders = {
            f"{item.symbol}:{item.order_id}": item.status
            for item in [*regular_orders, *conditional_orders]
            if not any(self._symbols_match(item.symbol, symbol) for symbol in expected_symbols)
        }
        position_mode = self._account_position_mode()
        return {
            "balances": balances,
            "free_balances": free_balances,
            "positions": positions,
            "regular_orders": [item.__dict__ for item in regular_orders],
            "conditional_orders": [item.__dict__ for item in conditional_orders],
            "used_margin": used_margin,
            "maintenance_margin": maintenance_margin,
            "used_margin_fraction": used_fraction,
            "liquidation_buffer_fraction": liquidation_buffer,
            "account_mode": position_mode,
            "unknown_exposure": {**unknown_positions, **unknown_orders},
            "account_state_known": True,
            "account_state_authority": "authenticated_rest",
            "account_fingerprint": self.account_fingerprint,
        }

    def _account_balances(self) -> tuple[dict, dict[str, float], dict[str, float], dict]:
        balance = self._client.fetch_balance()
        if not isinstance(balance, dict):
            raise RuntimeError("authenticated balance response is not an object")
        totals = balance.get("total")
        free = balance.get("free")
        if not isinstance(totals, dict) or not isinstance(free, dict):
            raise RuntimeError("authenticated balance response lacks total/free balances")
        balances = {
            str(asset): self._finite_number(value, "Account balance", non_negative=True)
            for asset, value in totals.items()
            if value is not None
        }
        free_balances = {
            str(asset): self._finite_number(value, "Free account balance", non_negative=True)
            for asset, value in free.items()
            if value is not None
        }
        raw_info = balance.get("info")
        info = raw_info if isinstance(raw_info, dict) else {}
        return balance, balances, free_balances, info

    def _account_number(
        self, balance: dict, info: dict, *names: str, default: float | None = None
    ) -> float:
        for name in names:
            value = info.get(name, balance.get(name))
            if value is not None:
                return self._finite_number(value, f"Account field {name}", non_negative=True)
        if default is None:
            raise RuntimeError("authenticated account response is missing a required margin field")
        return default

    def _account_margin(
        self, balance: dict, balances: Mapping[str, float], info: dict
    ) -> tuple[float, float, float, float]:
        quote_total = self._finite_number(
            balances.get(self.config.quote_asset, 0.0),
            "Account quote balance",
            non_negative=True,
        )
        margin_default = 0.0 if self.config.market_type == "spot" else None
        used = self._account_number(
            balance, info, "totalInitialMargin", "initialMargin", default=margin_default
        )
        maintenance = self._account_number(
            balance, info, "totalMaintMargin", "maintMargin", default=margin_default
        )
        margin_balance = self._account_number(
            balance,
            info,
            "totalMarginBalance",
            "marginBalance",
            "equity",
            default=quote_total if self.config.market_type == "spot" else None,
        )
        if margin_balance <= 0 and (used > 0 or maintenance > 0):
            raise RuntimeError("authenticated account margin balance is not positive")
        used_fraction = used / margin_balance if margin_balance > 0 else 0.0
        buffer = max(0.0, 1.0 - maintenance / margin_balance) if margin_balance > 0 else 0.0
        return used, maintenance, used_fraction, buffer

    def _account_positions_and_orders(
        self, balances: Mapping[str, float], expected_symbols: tuple[str, ...]
    ) -> tuple[dict[str, float], dict[str, float], list, list]:
        positions: dict[str, float] = {}
        unknown_positions: dict[str, float] = {}
        if self.config.market_type == "futures":
            position_rows = self.list_account_futures_positions()
            for item in position_rows:
                if abs(float(item.qty)) > 1e-12:
                    positions[self.platform_instrument_id(item.symbol)] = float(item.qty)
            unknown_positions = {
                f"position:{item.symbol}": float(item.qty)
                for item in position_rows
                if abs(float(item.qty)) > 1e-12
                and not any(self._symbols_match(item.symbol, symbol) for symbol in expected_symbols)
            }
        else:
            for symbol in expected_symbols:
                position = self.get_position(symbol)
                if abs(position.qty) > 1e-12:
                    positions[f"binance:spot:{symbol}"] = float(position.qty)
            expected_assets = {
                self.config.quote_asset.upper(),
                *(self._base_asset(symbol) for symbol in expected_symbols),
            }
            unknown_positions = {
                f"asset:{asset}": quantity
                for asset, quantity in balances.items()
                if abs(quantity) > 1e-12 and asset.upper() not in expected_assets
            }
        regular_orders = list(self.list_account_open_orders(conditional=False))
        conditional_orders = list(self.list_account_open_orders(conditional=True))
        return positions, unknown_positions, regular_orders, conditional_orders

    def _account_position_mode(self) -> str:
        if self.config.market_type != "futures":
            return "one_way"
        fetch_position_mode = getattr(self._client, "fetch_position_mode", None)
        if not callable(fetch_position_mode):
            raise RuntimeError(
                "authenticated futures account response lacks a position-mode reader"
            )
        mode_payload = fetch_position_mode()
        if not isinstance(mode_payload, dict):
            raise RuntimeError("authenticated futures account mode is not an object")
        if mode_payload.get("hedged") is not None:
            return "hedged" if bool(mode_payload["hedged"]) else "one_way"
        if str(mode_payload.get("mode") or "") in {"hedged", "one_way"}:
            return str(mode_payload["mode"])
        raise RuntimeError("authenticated futures account mode is missing")

    def _build_client(self):
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "ccxt is not installed. Run `pip install ccxt` to use CcxtBroker, "
                "or use PaperBroker for simulated execution."
            ) from exc
        self._assert_ccxt_version(ccxt)
        client = self._new_client(ccxt)
        self._configure_client(client)
        self._assert_client_capabilities(client)
        return client

    def _assert_ccxt_version(self, ccxt: Any) -> None:
        if not (
            self.config.live
            and self.config.market_type == "futures"
            and str(self.config.exchange).lower() == "binanceusdm"
        ):
            return
        installed_version = str(getattr(ccxt, "__version__", ""))
        if installed_version != NATIVE_STOP_CCXT_VERSION:
            raise RuntimeError(
                "Live Binance USD-M execution requires "
                f"ccxt=={NATIVE_STOP_CCXT_VERSION} for the validated conditional Algo Order API path; "
                f"found {installed_version or 'unknown'}."
            )

    def _new_client(self, ccxt: Any) -> Any:
        if not hasattr(ccxt, self.config.exchange):
            raise ValueError(f"ccxt has no exchange {self.config.exchange!r}.")
        klass = getattr(ccxt, self.config.exchange)
        default_type = "future" if self.config.market_type == "futures" else "spot"
        client = klass(
            {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "password": self.config.api_password or None,
                "enableRateLimit": True,
                "options": {
                    "defaultType": default_type,
                    "warnOnFetchOpenOrdersWithoutSymbol": False,
                },
            }
        )
        limiter = shared_exchange_rate_limiter(
            f"{self.config.exchange}:{self.config.market_type}:{self.config.testnet}",
            minimum_interval_seconds=self.config.request_min_interval_seconds,
        )
        return RateLimitedExchangeClient(client, limiter)

    def _configure_client(self, client: Any) -> None:
        if not self.config.testnet:
            return
        if str(self.config.exchange).lower() == "binanceusdm":
            enable_demo = getattr(client, "enable_demo_trading", None)
            if not callable(enable_demo):
                raise RuntimeError("Binance USD-M testnet requires CCXT demo trading support")
            enable_demo(True)
        elif hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)

    def _assert_client_capabilities(self, client: Any) -> None:
        if self.config.live:
            self._assert_methods(
                client,
                ("load_markets", "amount_to_precision", "price_to_precision"),
                "Live ccxt client lacks required precision method(s)",
            )
        if (
            self.config.live
            and self.config.market_type == "futures"
            and str(self.config.exchange).lower() == "binanceusdm"
        ):
            self._assert_methods(
                client,
                ("set_position_mode", "fetch_position_mode"),
                "Live Binance USD-M client lacks required position-mode method(s)",
            )
            self._assert_methods(
                client,
                ("fetch_open_orders", "fetch_positions"),
                "Live Binance USD-M client lacks whole-account inventory method(s)",
            )
            self._assert_methods(
                client,
                (
                    "fapiPrivatePostAlgoOrder",
                    "fapiPrivateGetAlgoOrder",
                    "fapiPrivateGetOpenAlgoOrders",
                    "fapiPrivateDeleteAlgoOrder",
                ),
                "Live Binance USD-M client lacks the Algo Order API method(s)",
            )

    @staticmethod
    def _assert_methods(client: Any, names: tuple[str, ...], message: str) -> None:
        missing = [name for name in names if not callable(getattr(client, name, None))]
        if missing:
            raise RuntimeError(f"{message}: {', '.join(missing)}.")

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
        free = bal.get("free")
        if not isinstance(free, dict):
            raise ValueError("Quote balance response must include a free-balance object.")
        value = free.get(self.config.quote_asset)
        if value is None:
            nested = bal.get(self.config.quote_asset)
            value = nested.get("free") if isinstance(nested, dict) else 0.0
        return self._finite_number(
            0.0 if value is None else value, "Quote balance", non_negative=True
        )

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
    def _has_precision_tools(self) -> bool:
        return all(
            callable(getattr(self._client, name, None))
            for name in ("load_markets", "amount_to_precision", "price_to_precision")
        )

    def _precision_market(self, symbol: str) -> tuple[str, dict | None]:
        client_symbol = self._ccxt_symbol(symbol)
        if not self._has_precision_tools():
            # Lightweight test doubles created via ``__new__`` predate these
            # hooks. Real live clients are capability-checked in _build_client.
            return client_symbol, None
        cache = getattr(self, "_precision_markets", {})
        if client_symbol in cache:
            return client_symbol, cache[client_symbol]
        markets = self._client.load_markets()
        if not isinstance(markets, dict):
            raise RuntimeError(
                "ccxt load_markets() did not return market metadata. Refusing order."
            )
        market_method = getattr(self._client, "market", None)
        market = (
            market_method(client_symbol) if callable(market_method) else markets.get(client_symbol)
        )
        if not isinstance(market, dict):
            raise ValueError(f"ccxt market metadata missing for {client_symbol}. Refusing order.")
        cache[client_symbol] = market
        self._precision_markets = cache
        return client_symbol, market

    def _market_limit(
        self,
        market: dict,
        category: str,
        bound: str,
    ) -> float | None:
        limits = market.get("limits")
        category_limits = limits.get(category) if isinstance(limits, dict) else None
        raw = category_limits.get(bound) if isinstance(category_limits, dict) else None
        if raw is None:
            return None
        value = self._finite_number(
            raw,
            f"Market {category} {bound} limit",
            non_negative=True,
        )
        return value

    @staticmethod
    def _assert_within_market_limits(
        value: float,
        *,
        minimum: float | None,
        maximum: float | None,
        label: str,
    ) -> None:
        tolerance = max(abs(value) * 1e-12, 1e-12)
        if minimum is not None and value + tolerance < minimum:
            raise ValueError(
                f"{label} {value:g} is below exchange minimum {minimum:g}. Refusing order."
            )
        if maximum is not None and value - tolerance > maximum:
            raise ValueError(
                f"{label} {value:g} exceeds exchange maximum {maximum:g}. Refusing order."
            )

    def normalize_order_qty(
        self,
        symbol: str,
        qty: float,
        *,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> float:
        raw_qty = self._finite_number(qty, "Order quantity", positive=True)
        client_symbol, market = self._precision_market(symbol)
        if market is None:
            return raw_qty
        try:
            precise = self._client.amount_to_precision(client_symbol, raw_qty)
        except Exception as exc:
            raise ValueError(
                f"Could not normalize order quantity for {client_symbol}: {exc}"
            ) from exc
        normalized = self._finite_number(
            precise,
            "Precision-normalized order quantity",
            positive=True,
        )
        increase_tolerance = max(abs(raw_qty) * 1e-12, 1e-12)
        if normalized - raw_qty > increase_tolerance:
            raise ValueError(
                f"Exchange amount precision increased intended quantity from {raw_qty:g} "
                f"to {normalized:g}. Refusing order."
            )
        self._assert_within_market_limits(
            normalized,
            minimum=self._market_limit(market, "amount", "min"),
            maximum=self._market_limit(market, "amount", "max"),
            label="Order quantity",
        )
        self._assert_within_market_limits(
            normalized,
            minimum=self._market_limit(market, "market", "min"),
            maximum=self._market_limit(market, "market", "max"),
            label="Market order quantity",
        )
        if price is not None:
            reference_price = self._finite_number(price, "Order reference price", positive=True)
            cost = normalized * reference_price
            self._assert_within_market_limits(
                cost,
                minimum=(
                    None
                    if reduce_only and self.config.market_type == "futures"
                    else self._market_limit(market, "cost", "min")
                ),
                maximum=self._market_limit(market, "cost", "max"),
                label="Order notional",
            )
        return normalized

    def normalize_order_price(self, symbol: str, price: float) -> float:
        raw_price = self._finite_number(price, "Order price", positive=True)
        client_symbol, market = self._precision_market(symbol)
        if market is None:
            return raw_price
        try:
            precise = self._client.price_to_precision(client_symbol, raw_price)
        except Exception as exc:
            raise ValueError(f"Could not normalize order price for {client_symbol}: {exc}") from exc
        normalized = self._finite_number(
            precise,
            "Precision-normalized order price",
            positive=True,
        )
        self._assert_within_market_limits(
            normalized,
            minimum=self._market_limit(market, "price", "min"),
            maximum=self._market_limit(market, "price", "max"),
            label="Order price",
        )
        return normalized

    def _ensure_futures_margin_mode(self, symbol: str) -> None:
        if self.config.market_type != "futures":
            return
        client_symbol = self._ccxt_symbol(symbol)
        margin_mode = str(self.config.futures_margin_mode).lower()
        if margin_mode != "isolated":
            raise ValueError("FUTURES_MARGIN_MODE must be 'isolated' for live futures entries.")
        if not callable(getattr(self._client, "set_margin_mode", None)):
            raise RuntimeError(
                "Refusing futures order: ccxt client cannot set isolated margin mode, "
                "so account margin risk cannot be bounded."
            )
        try:
            self._client.set_margin_mode(margin_mode, client_symbol)
        except Exception as exc:
            if not self._is_exact_binance_already_set_error(
                exc,
                code=BINANCE_MARGIN_MODE_ALREADY_SET_CODE,
                message=BINANCE_MARGIN_MODE_ALREADY_SET_MESSAGE,
            ):
                raise RuntimeError(
                    f"Refusing futures order: could not set isolated margin mode: {exc}"
                ) from exc

    def _ensure_futures_position_mode(self, symbol: str) -> None:
        if (
            self.config.market_type != "futures"
            or str(self.config.exchange).lower() != "binanceusdm"
        ):
            return
        client_symbol = self._ccxt_symbol(symbol)
        if not callable(getattr(self._client, "set_position_mode", None)):
            raise RuntimeError(
                "Refusing Binance USD-M entry: ccxt client cannot set one-way position mode."
            )
        try:
            self._client.set_position_mode(False, client_symbol)
        except Exception as exc:
            if not self._is_exact_binance_already_set_error(
                exc,
                code=BINANCE_POSITION_MODE_ALREADY_SET_CODE,
                message=BINANCE_POSITION_MODE_ALREADY_SET_MESSAGE,
            ):
                raise RuntimeError(
                    f"Refusing Binance USD-M entry: could not set one-way position mode: {exc}"
                ) from exc
        if not self.verify_one_way_position_mode(symbol):
            raise RuntimeError(
                "Refusing Binance USD-M entry: exchange did not confirm one-way position mode."
            )

    def _is_exact_binance_already_set_error(
        self,
        exc: Exception,
        *,
        code: int,
        message: str,
    ) -> bool:
        """Accept only Binance's documented idempotent-setting responses."""

        if str(self.config.exchange).strip().lower() != "binanceusdm":
            return False
        exception_code = getattr(exc, "code", None)
        try:
            if exception_code is not None and int(exception_code) == code:
                return True
        except (TypeError, ValueError):
            pass
        raw = str(exc).strip()
        code_match = re.search(r'["\']?code["\']?\s*[:=]\s*["\']?(-?\d+)', raw)
        if code_match is not None and int(code_match.group(1)) == code:
            return True
        if raw == message:
            return True
        message_match = re.search(r'["\']msg["\']\s*:\s*["\']([^"\']*)["\']', raw)
        return message_match is not None and message_match.group(1) == message

    def verify_one_way_position_mode(self, symbol: str) -> bool:
        """Read Binance USD-M position mode without changing account settings."""

        if (
            self.config.market_type != "futures"
            or str(self.config.exchange).strip().lower() != "binanceusdm"
        ):
            raise RuntimeError(
                "One-way position-mode verification is only supported for Binance USD-M futures."
            )
        fetch_mode = getattr(self._client, "fetch_position_mode", None)
        if not callable(fetch_mode):
            raise RuntimeError(
                "Refusing Binance USD-M entry: ccxt client cannot verify one-way position mode."
            )
        client_symbol = self._ccxt_symbol(symbol)
        try:
            mode = fetch_mode(client_symbol)
        except Exception as exc:
            raise RuntimeError(f"Could not read Binance USD-M position mode: {exc}") from exc
        if not isinstance(mode, dict) or not isinstance(mode.get("hedged"), bool):
            raise RuntimeError(
                "Binance USD-M position-mode response did not contain a boolean hedged flag."
            )
        return mode["hedged"] is False

    def _ensure_futures_leverage(self, symbol: str) -> None:
        if self.config.market_type != "futures":
            return
        client_symbol = self._ccxt_symbol(symbol)
        leverage = int(self.config.max_futures_leverage)
        if not (1 <= leverage <= 3):
            raise ValueError("MAX_FUTURES_LEVERAGE must be between 1 and 3.")
        if not callable(getattr(self._client, "set_leverage", None)):
            raise RuntimeError(
                "Refusing futures order: ccxt client cannot set leverage, "
                "so account leverage cannot be bounded."
            )
        self._client.set_leverage(leverage, client_symbol)

    def _verify_futures_risk_settings(self, symbol: str) -> None:
        """Read back the per-symbol isolated-margin and leverage settings."""

        if self.config.market_type != "futures":
            return
        fetch_positions = getattr(self._client, "fetch_positions", None)
        if not callable(fetch_positions):
            raise RuntimeError(
                "Refusing futures entry: ccxt client cannot read back margin and leverage settings."
            )
        client_symbol = self._ccxt_symbol(symbol)
        try:
            positions = fetch_positions([client_symbol])
        except Exception as exc:
            raise RuntimeError(
                f"Refusing futures entry: could not read back margin and leverage settings: {exc}"
            ) from exc
        if not isinstance(positions, list):
            raise RuntimeError("Refusing futures entry: position-settings response must be a list.")
        position = self._matching_position_settings(
            positions, client_symbol=client_symbol, symbol=symbol
        )
        raw_info = position.get("info")
        info = raw_info if isinstance(raw_info, dict) else {}
        self._assert_isolated_margin(position, info)
        self._assert_leverage_readback(position, info)

    @staticmethod
    def _matching_position_settings(
        positions: list[dict], *, client_symbol: str, symbol: str
    ) -> dict:
        matches: list[dict] = []
        for position in positions:
            if not isinstance(position, dict):
                continue
            raw_info = position.get("info")
            info = raw_info if isinstance(raw_info, dict) else {}
            reported_symbol = position.get("symbol") or info.get("symbol")
            if (
                reported_symbol == client_symbol
                or str(reported_symbol or "").upper() == symbol.upper()
            ):
                matches.append(position)
        if len(matches) != 1:
            raise RuntimeError(
                "Refusing futures entry: exchange did not return exactly one matching "
                f"position-settings record for {client_symbol}."
            )
        return matches[0]

    @staticmethod
    def _assert_isolated_margin(position: dict, info: dict) -> None:
        margin_mode = str(position.get("marginMode") or info.get("marginType") or "").lower()
        if margin_mode != "isolated":
            raise RuntimeError(
                "Refusing futures entry: exchange did not confirm isolated margin mode."
            )

    def _assert_leverage_readback(self, position: dict, info: dict) -> None:
        raw_leverage = position.get("leverage")
        if raw_leverage is None:
            raw_leverage = info.get("leverage")
        if isinstance(raw_leverage, bool):
            raise RuntimeError("Refusing futures entry: exchange leverage readback is not numeric.")
        try:
            leverage = self._finite_number(
                raw_leverage,
                "Futures leverage readback",
                positive=True,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Refusing futures entry: exchange leverage readback is invalid: {exc}"
            ) from exc
        expected_leverage = int(self.config.max_futures_leverage)
        if leverage != expected_leverage:
            raise RuntimeError(
                "Refusing futures entry: exchange leverage readback "
                f"{leverage:g} does not match configured leverage {expected_leverage}."
            )

    def place_order(self, order: Order) -> Fill:
        normalized_order, ref_price, enforce_notional_cap = self._prepare_order(order)
        client_symbol = self._ccxt_symbol(normalized_order.symbol)
        params = self._order_params(normalized_order)
        result = self._client.create_order(
            symbol=client_symbol,
            type=normalized_order.type.value,
            side=normalized_order.side.value,
            amount=normalized_order.qty,
            price=normalized_order.price,
            params=params,
        )
        self._assert_order_status_accepted(
            result,
            requested_quantity=normalized_order.qty,
            acknowledgement_only=getattr(self, "_acknowledgement_only", False),
        )
        self._assert_order_response_matches(result, normalized_order)
        return self._parse_order_result(
            result,
            normalized_order=normalized_order,
            ref_price=ref_price,
            enforce_notional_cap=enforce_notional_cap,
        )

    def _prepare_order(self, order: Order) -> tuple[Order, float, bool]:
        if not math.isfinite(float(order.qty)) or order.qty <= 0:
            raise ValueError(f"Order quantity must be positive, got {order.qty:g}. Refusing.")
        ref_price = self._reference_price(order)
        normalized_price = (
            self.normalize_order_price(order.symbol, float(order.price))
            if order.price is not None
            else None
        )
        if normalized_price is not None:
            ref_price = normalized_price
        normalized_qty = self.normalize_order_qty(
            order.symbol,
            float(order.qty),
            price=ref_price,
            reduce_only=bool(order.reduce_only),
        )
        normalized_order = Order(
            symbol=order.symbol,
            side=order.side,
            qty=normalized_qty,
            type=order.type,
            price=normalized_price,
            reduce_only=order.reduce_only,
            client_id=order.client_id,
        )
        enforce_notional_cap = not self._is_futures_reduce_only(order)
        if enforce_notional_cap:
            self._assert_notional_within_cap(ref_price * normalized_qty)
        self._assert_live_order_allowed()
        self._assert_order_balance_and_position(normalized_order, ref_price)
        if self.config.market_type == "futures" and not normalized_order.reduce_only:
            self._prepare_futures_entry(normalized_order.symbol)
        return normalized_order, ref_price, enforce_notional_cap

    def _assert_live_order_allowed(self) -> None:
        if not self.config.live:
            raise RuntimeError(
                "Refusing to place a real order: TRADING_LIVE is not enabled. "
                "Set TRADING_LIVE=1 (and ideally EXCHANGE_TESTNET=1) to trade."
            )

    def _assert_order_balance_and_position(self, order: Order, ref_price: float) -> None:
        if self.config.market_type == "spot" and order.side == OrderSide.BUY:
            self._assert_spot_buy_within_balance(order, ref_price)
        if self.config.market_type == "spot" and order.side == OrderSide.SELL:
            self._assert_spot_sell_within_balance(order)
        if self._is_futures_reduce_only(order):
            self._assert_futures_reduce_only_within_position(order)

    def _prepare_futures_entry(self, symbol: str) -> None:
        self._ensure_futures_margin_mode(symbol)
        self._ensure_futures_leverage(symbol)
        self._ensure_futures_position_mode(symbol)
        self._verify_futures_risk_settings(symbol)

    def _order_params(self, order: Order) -> dict[str, object]:
        params: dict[str, object] = {}
        if order.client_id is not None:
            if not isinstance(order.client_id, str) or not CLIENT_ORDER_ID_RE.fullmatch(
                order.client_id
            ):
                raise ValueError(
                    "Order client_id must be 1-36 Binance-safe characters "
                    "([A-Za-z0-9._:/-]). Refusing."
                )
            client_id_param = (
                "newClientOrderId"
                if str(self.config.exchange).lower().startswith("binance")
                else "clientOrderId"
            )
            params[client_id_param] = order.client_id
        if order.reduce_only and self.config.market_type == "futures":
            params["reduceOnly"] = True
        if self.config.market_type == "futures" and not order.reduce_only:
            self._assert_no_open_orders_before_entry(order.symbol)
            self._assert_futures_position_compatible(order)
        return params

    def _parse_order_result(
        self,
        result: dict,
        *,
        normalized_order: Order,
        ref_price: float,
        enforce_notional_cap: bool,
    ) -> Fill:
        exchange_order_id = str(
            result.get("id") or (result.get("info") or {}).get("orderId") or ""
        ).strip()
        acknowledged_client_id = str(
            result.get("clientOrderId")
            or (result.get("info") or {}).get("clientOrderId")
            or normalized_order.client_id
            or ""
        ).strip()
        if getattr(self, "_acknowledgement_only", False):
            if not exchange_order_id or not acknowledged_client_id:
                raise RuntimeError("exchange acknowledgement has no order identity")
            if not hasattr(self, "_submission_responses"):
                self._submission_responses = {}
            self._submission_responses[exchange_order_id] = dict(result)
            return BrokerOrderAcknowledgement(  # type: ignore[return-value]
                exchange_order_id=exchange_order_id,
                client_order_id=acknowledged_client_id,
                status=str(
                    result.get("status") or (result.get("info") or {}).get("status") or "ack"
                ).lower(),
                submitted_at=float(result.get("timestamp") or time.time()),
            )
        fill_price, fee, filled = self._fill_values(result)
        self._assert_fill_values(
            result,
            normalized_order=normalized_order,
            ref_price=ref_price,
            enforce_notional_cap=enforce_notional_cap,
            fill_price=fill_price,
            fee=fee,
            filled=filled,
        )
        LOGGER.info(
            "Placed %s %s %s @ %s",
            normalized_order.side.value,
            filled,
            normalized_order.symbol,
            fill_price,
        )
        return Fill(
            symbol=normalized_order.symbol,
            side=normalized_order.side,
            qty=filled,
            price=fill_price,
            fee=fee,
            timestamp=float(
                result.get("timestamp") or result.get("lastTradeTimestamp") or time.time()
            ),
            exchange_order_id=str(result.get("id") or "") or None,
            client_order_id=str(
                result.get("clientOrderId")
                or (result.get("info") or {}).get("clientOrderId")
                or normalized_order.client_id
                or ""
            )
            or None,
            fee_asset=str((result.get("fee") or {}).get("currency") or "") or None,
        )

    def _fill_values(self, result: dict) -> tuple[float, float, float]:
        fill_price = float(
            self._required_first_present(result, ("average", "price"), label="Filled order price")
        )
        fee = float((result.get("fee") or {}).get("cost") or 0.0)
        filled = float(
            self._required_first_present(result, ("filled",), label="Filled order quantity")
        )
        return fill_price, fee, filled

    def _assert_fill_values(
        self,
        result: dict,
        *,
        normalized_order: Order,
        ref_price: float,
        enforce_notional_cap: bool,
        fill_price: float,
        fee: float,
        filled: float,
    ) -> None:
        if not math.isfinite(fee) or fee < 0:
            raise ValueError(
                f"Filled order fee must be finite and non-negative, got {fee:g}. Refusing."
            )
        if not math.isfinite(fill_price) or fill_price <= 0:
            raise ValueError(f"Filled order price must be positive, got {fill_price:g}. Refusing.")
        if not math.isfinite(filled) or filled <= 0:
            raise ValueError(f"Filled order quantity must be positive, got {filled:g}. Refusing.")
        if filled < float(normalized_order.qty) and not (result.get("id") or result.get("status")):
            raise ValueError("Refusing to accept a partial fill without an exchange order identity")
        if filled > float(normalized_order.qty) + max(float(normalized_order.qty) * 1e-12, 1e-12):
            raise ValueError("Exchange fill quantity exceeds requested quantity. Refusing.")
        self._assert_fill_slippage_within_cap(ref_price, fill_price)
        if enforce_notional_cap:
            self._assert_notional_within_cap(fill_price * filled, label="Filled order")

    def supports_native_protective_stops(self) -> bool:
        return (
            self.config.market_type == "futures"
            and str(self.config.exchange).lower() == "binanceusdm"
            and str(self.config.quote_asset).upper() == "USDT"
            and str(self.config.futures_margin_mode).lower() == "isolated"
        )

    def list_open_orders(
        self,
        symbol: str,
        *,
        conditional: bool,
    ) -> tuple[OpenOrderIdentity, ...]:
        """Read and strictly sanitize Binance USD-M open-order inventory.

        Binance routes ordinary orders and conditional/algo orders through
        distinct CCXT query paths. Callers therefore request each inventory
        explicitly and must require both to be empty before adding exposure.
        This method is read-only and never cancels an order.
        """

        return self._list_open_orders(symbol=symbol, conditional=conditional)

    def list_account_open_orders(
        self,
        *,
        conditional: bool,
    ) -> tuple[OpenOrderIdentity, ...]:
        """Read every regular or conditional order in the USD-M account."""

        return self._list_open_orders(symbol=None, conditional=conditional)

    def _list_open_orders(
        self,
        *,
        symbol: str | None,
        conditional: bool,
    ) -> tuple[OpenOrderIdentity, ...]:
        exchange_name = str(self.config.exchange).lower()
        futures_supported = self.config.market_type == "futures" and exchange_name == "binanceusdm"
        spot_supported = self.config.market_type == "spot" and exchange_name == "binance"
        if not (futures_supported or spot_supported):
            raise RuntimeError(
                "Open-order inventory verification is only validated for Binance spot regular "
                "orders and Binance USD-M futures."
            )
        if futures_supported and conditional:
            algo_orders = getattr(self._client, "fapiPrivateGetOpenAlgoOrders", None)
            if callable(algo_orders):
                params = {"symbol": self._ccxt_symbol(symbol)} if symbol is not None else {}
                try:
                    payload = algo_orders(params)
                except Exception as exc:
                    raise RuntimeError(
                        "Could not query conditional open orders for "
                        f"{symbol or 'the whole account'}: {exc}"
                    ) from exc
                return self._parse_open_order_payload(
                    payload, requested_symbol=symbol, conditional=True
                )
        fetch_open_orders = getattr(self._client, "fetch_open_orders", None)
        if not callable(fetch_open_orders):
            raise RuntimeError(
                "ccxt client cannot fetch open orders; refusing to assume the inventory is empty."
            )
        client_symbol = self._ccxt_symbol(symbol) if symbol is not None else None
        params = {"trigger": True} if conditional else {}
        try:
            payload = fetch_open_orders(client_symbol, params=params)
        except Exception as exc:
            order_kind = "conditional" if conditional else "regular"
            raise RuntimeError(
                f"Could not query {order_kind} open orders for "
                f"{client_symbol or 'the whole account'}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            raise ValueError(
                "ccxt fetch_open_orders response must be a list. Refusing to assume "
                "the open-order inventory is empty."
            )

        return self._parse_open_order_payload(
            payload, requested_symbol=symbol, conditional=conditional
        )

    def _parse_open_order_payload(
        self,
        payload: Any,
        *,
        requested_symbol: str | None,
        conditional: bool,
    ) -> tuple[OpenOrderIdentity, ...]:
        if not isinstance(payload, list):
            raise ValueError(
                "ccxt open-order response must be a list. Refusing to assume "
                "the open-order inventory is empty."
            )
        orders: list[OpenOrderIdentity] = []
        seen: set[tuple[str, str, str]] = set()
        for index, item in enumerate(payload):
            is_conditional = conditional
            if self.config.market_type == "spot":
                is_conditional = self._is_conditional_spot_order(item)
                if is_conditional != conditional:
                    continue
            order = self._parse_open_order_identity(
                item,
                requested_symbol=requested_symbol,
                conditional=is_conditional,
                index=index,
            )
            identity = (order.symbol, order.order_id, order.client_id)
            if identity in seen:
                raise ValueError(
                    f"ccxt open-order response contains duplicate identity {identity!r}. Refusing."
                )
            seen.add(identity)
            orders.append(order)
        return tuple(
            sorted(orders, key=lambda order: (order.symbol, order.order_id, order.client_id))
        )

    @staticmethod
    def _is_conditional_spot_order(payload: dict) -> bool:
        """Identify Binance spot stop/OCO legs in the regular open-order feed."""
        info_value = payload.get("info")
        info = info_value if isinstance(info_value, dict) else {}
        raw_type = payload.get("type") or info.get("type") or info.get("origType")
        order_type = str(raw_type or "").upper()
        if order_type in {
            "STOP",
            "STOP_LOSS",
            "STOP_LOSS_LIMIT",
            "TAKE_PROFIT",
            "TAKE_PROFIT_LIMIT",
            "TRAILING_STOP_MARKET",
        }:
            return True
        for field in ("stopPrice", "triggerPrice", "trailingDelta"):
            value = payload.get(field, info.get(field))
            if value not in (None, "", 0, "0", 0.0):
                return True
        return False

    def list_account_futures_positions(self) -> tuple[FuturesPositionIdentity, ...]:
        """Read and sanitize every non-flat Binance USD-M account position."""
        self._assert_whole_account_positions_supported()
        fetch_positions = getattr(self._client, "fetch_positions", None)
        if not callable(fetch_positions):
            raise RuntimeError(
                "ccxt client cannot fetch account positions; refusing to assume the account is flat."
            )
        payload = self._fetch_whole_account_positions(fetch_positions)
        if not isinstance(payload, list):
            raise ValueError(
                "ccxt fetch_positions response must be a list. Refusing to assume "
                "the account is flat."
            )

        positions = [
            position
            for index, item in enumerate(payload)
            if (position := self._parse_futures_position(item, index=index)) is not None
        ]
        symbols = [position.symbol for position in positions]
        if len(symbols) != len(set(symbols)):
            duplicate = next(symbol for symbol in symbols if symbols.count(symbol) > 1)
            raise ValueError(
                f"ccxt position response contains duplicate symbol {duplicate!r}. Refusing."
            )
        return tuple(sorted(positions, key=lambda position: position.symbol))

    def _assert_whole_account_positions_supported(self) -> None:
        if (
            self.config.market_type != "futures"
            or str(self.config.exchange).lower() != "binanceusdm"
        ):
            raise RuntimeError(
                "Whole-account position verification is only validated for Binance USD-M futures."
            )

    @staticmethod
    def _fetch_whole_account_positions(fetch_positions: Callable[..., Any]) -> Any:
        try:
            return fetch_positions(None)
        except Exception as exc:
            raise RuntimeError(f"Could not query whole-account futures positions: {exc}") from exc

    def _parse_futures_position(self, item: Any, *, index: int) -> FuturesPositionIdentity | None:
        if not isinstance(item, dict):
            raise ValueError(f"ccxt position response item {index} must be an object. Refusing.")
        info_value = item.get("info")
        if info_value is not None and not isinstance(info_value, dict):
            raise ValueError(
                f"ccxt position response item {index} info must be an object. Refusing."
            )
        info = info_value or {}
        symbol = self._canonical_inventory_symbol(
            self._payload_value(item, info, "symbol"),
            label=f"Position item {index} symbol",
        )
        contracts = self._finite_number(
            self._payload_value(item, info, "contracts", "positionAmt"),
            f"Position item {index} contracts",
            non_negative=True,
        )
        if contracts == 0:
            return None
        qty = self._futures_position_quantity(item, info, contracts, index=index)
        entry_price = self._finite_number(
            self._payload_value(item, info, "entryPrice"),
            f"Position item {index} entry price",
            positive=True,
        )
        return FuturesPositionIdentity(symbol=symbol, qty=qty, avg_price=entry_price)

    def _futures_position_quantity(
        self, item: dict, info: dict, contracts: float, *, index: int
    ) -> float:
        raw_side = self._sanitized_open_order_field(
            self._payload_value(item, info, "side", "positionSide"),
            label=f"Position item {index} side",
        ).lower()
        if raw_side == "long":
            return contracts
        if raw_side == "short":
            return -contracts
        if raw_side == "both":
            signed_amount = self._finite_number(
                info.get("positionAmt"), f"Position item {index} signed amount"
            )
            quantity_tolerance = max(contracts * 1e-9, 1e-12)
            if signed_amount == 0 or abs(abs(signed_amount) - contracts) > quantity_tolerance:
                raise ValueError(
                    f"Position item {index} one-way quantity evidence is inconsistent. Refusing."
                )
            return signed_amount
        raise ValueError(f"Position item {index} side {raw_side!r} is invalid. Refusing.")

    @staticmethod
    def _sanitized_open_order_field(value, *, label: str) -> str:
        if value is None or isinstance(value, bool | dict | list | tuple | set):
            raise ValueError(f"{label} is missing or invalid in open-order response. Refusing.")
        normalized = str(value).strip()
        if not normalized or len(normalized) > 128 or not normalized.isprintable():
            raise ValueError(f"{label} is missing or invalid in open-order response. Refusing.")
        return normalized

    def _parse_open_order_identity(
        self,
        payload,
        *,
        requested_symbol: str | None,
        conditional: bool,
        index: int,
    ) -> OpenOrderIdentity:
        if not isinstance(payload, dict):
            raise ValueError(f"ccxt open-order response item {index} must be an object. Refusing.")
        info_value = payload.get("info")
        if info_value is not None and not isinstance(info_value, dict):
            raise ValueError(
                f"ccxt open-order response item {index} info must be an object. Refusing."
            )
        info = info_value or {}
        symbol_value = self._payload_value(payload, info, "symbol")
        parsed_symbol = self._canonical_inventory_symbol(
            symbol_value,
            label=f"Open order item {index} symbol",
        )
        if requested_symbol is not None and not self._symbols_match(
            parsed_symbol, requested_symbol
        ):
            raise ValueError(
                f"Open order symbol {symbol_value!r} does not match {requested_symbol!r}. Refusing."
            )
        order_id = self._sanitized_open_order_field(
            self._payload_value(payload, info, "id", "algoId", "orderId"),
            label="Open order id",
        )
        client_id = self._sanitized_open_order_field(
            self._payload_value(
                payload,
                info,
                "clientOrderId",
                "clientAlgoId",
                "origClientOrderId",
            ),
            label="Open order client id",
        )
        raw_status = self._sanitized_open_order_field(
            self._payload_value(payload, info, "status", "algoStatus"),
            label="Open order status",
        ).lower()
        if raw_status in {"open", "new", "accepted", "pending", "triggering"}:
            status = "open"
        elif raw_status in {"partially_filled", "partially-filled", "partial"}:
            status = "partially_filled"
        else:
            raise ValueError(f"Open order status {raw_status!r} is not an active status. Refusing.")
        return OpenOrderIdentity(
            symbol=requested_symbol or parsed_symbol,
            order_id=order_id,
            client_id=client_id,
            status=status,
            conditional=conditional,
        )

    def _assert_no_open_orders_before_entry(self, symbol: str) -> None:
        if (
            self.config.market_type != "futures"
            or str(self.config.exchange).lower() != "binanceusdm"
        ):
            return
        if self.config.allow_multi_symbol_positions:
            regular = self.list_open_orders(symbol, conditional=False)
            conditional = self.list_open_orders(symbol, conditional=True)
        else:
            regular = self.list_account_open_orders(conditional=False)
            conditional = self.list_account_open_orders(conditional=True)
        if regular or conditional:
            raise RuntimeError(
                "Refusing Binance USD-M entry: dedicated account has outstanding "
                f"orders (regular={len(regular)}, conditional={len(conditional)})."
            )

    def _assert_futures_position_compatible(self, order: Order) -> None:
        """Prove that account state is compatible immediately before adding risk."""

        if self.config.market_type != "futures":
            return
        if self.config.allow_multi_symbol_positions:
            position = self.get_position(order.symbol)
            if not position.is_flat and position.side is not order.side:
                raise RuntimeError(
                    "Refusing futures entry: a non-reduce-only order would oppose the "
                    f"current {order.symbol} position."
                )
            return
        positions = self.list_account_futures_positions()
        if positions:
            detail = ", ".join(f"{position.symbol}={position.qty:g}" for position in positions)
            signed_qty = f" signed qty {positions[0].qty:g};" if len(positions) == 1 else ""
            raise RuntimeError(
                f"Refusing futures entry: dedicated account is not flat;{signed_qty} "
                f"broker reports {detail}."
            )

    def place_protective_stop(
        self,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        trigger_price: float,
        client_id: str,
    ) -> ProtectiveOrder:
        """Place a Binance USD-M one-way reduce-only stop-market order.

        CCXT maps ``stopLossPrice`` on a market order to Binance's conditional
        ``STOP_MARKET`` algo endpoint.  The raw response is still validated
        here because accepting an unidentifiable or non-reducing order would
        leave the caller falsely believing the position is protected.
        """

        self._assert_protective_stop_supported()
        trigger_price = self.normalize_order_price(symbol, trigger_price)
        qty = self.normalize_order_qty(
            symbol,
            qty,
            price=trigger_price,
            reduce_only=True,
        )
        self._assert_client_order_id(client_id, label="Protective stop client_id")
        if not isinstance(side, OrderSide):
            try:
                side = OrderSide(side)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Protective stop side must be buy or sell, got {side!r}."
                ) from exc

        self._assert_futures_reduce_only_within_position(
            Order(symbol=symbol, side=side, qty=qty, reduce_only=True)
        )
        result = self._place_algo_stop(
            symbol=symbol,
            side=side,
            qty=qty,
            trigger_price=trigger_price,
            client_id=client_id,
        )
        protective = self._parse_protective_order(result, requested_symbol=symbol)
        self._assert_protective_order_matches(
            protective,
            symbol=symbol,
            side=side,
            qty=qty,
            trigger_price=trigger_price,
            client_id=client_id,
        )
        if protective.status not in {
            ProtectiveOrderStatus.OPEN,
            ProtectiveOrderStatus.TRIGGERED,
        }:
            raise ValueError(
                f"Protective stop placement returned unsafe status {protective.status.value!r}. Refusing."
            )
        if protective.status == ProtectiveOrderStatus.OPEN and protective.filled_qty != 0:
            raise ValueError(
                "New protective stop unexpectedly reports a fill. Refusing to accept protection."
            )
        LOGGER.info(
            "Placed native protective stop %s for %s %s @ trigger %s",
            protective.order_id,
            qty,
            symbol,
            trigger_price,
        )
        return protective

    def get_protective_stop(
        self,
        *,
        symbol: str,
        order_id: str | None,
        client_id: str,
    ) -> ProtectiveOrder:
        self._assert_protective_stop_supported()
        algo_get = getattr(self._client, "fapiPrivateGetAlgoOrder", None)
        if callable(algo_get):
            params = {"symbol": self._ccxt_symbol(symbol)}
            if order_id is not None:
                params["algoId"] = str(order_id)
            else:
                params["clientAlgoId"] = client_id
            result = algo_get(params)
        else:
            lookup_id, params = self._protective_lookup(order_id=order_id, client_id=client_id)
            result = self._client.fetch_order(
                lookup_id,
                self._ccxt_symbol(symbol),
                params,
            )
        protective = self._parse_protective_order(result, requested_symbol=symbol)
        self._assert_protective_identity(
            protective,
            symbol=symbol,
            order_id=order_id,
            client_id=client_id,
        )
        return protective

    def cancel_protective_stop(
        self,
        *,
        symbol: str,
        order_id: str | None,
        client_id: str,
    ) -> ProtectiveOrder:
        self._assert_protective_stop_supported()
        algo_delete = getattr(self._client, "fapiPrivateDeleteAlgoOrder", None)
        if callable(algo_delete):
            params = {"symbol": self._ccxt_symbol(symbol)}
            if order_id is not None:
                params["algoId"] = str(order_id)
            else:
                params["clientAlgoId"] = client_id
            algo_delete(params)
        else:
            lookup_id, params = self._protective_lookup(order_id=order_id, client_id=client_id)
            self._client.cancel_order(
                lookup_id,
                self._ccxt_symbol(symbol),
                params,
            )
        # Binance's cancel response may omit status and execution details.  A
        # separate trigger-order query is the acknowledgement we trust.
        protective = self.get_protective_stop(
            symbol=symbol,
            order_id=order_id,
            client_id=client_id,
        )
        if protective.status == ProtectiveOrderStatus.OPEN:
            raise RuntimeError(
                f"Protective stop {protective.order_id} is still open after cancellation. Refusing."
            )
        return protective

    def _place_algo_stop(
        self,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        trigger_price: float,
        client_id: str,
    ) -> dict:
        algo_post = getattr(self._client, "fapiPrivatePostAlgoOrder", None)
        if callable(algo_post):
            return algo_post(
                {
                    "symbol": self._ccxt_symbol(symbol),
                    "side": side.value.upper(),
                    "positionSide": "BOTH",
                    "type": "STOP_MARKET",
                    "quantity": qty,
                    "triggerPrice": trigger_price,
                    "reduceOnly": True,
                    "clientAlgoId": client_id,
                }
            )
        return self._client.create_order(
            symbol=self._ccxt_symbol(symbol),
            type=OrderType.MARKET.value,
            side=side.value,
            amount=qty,
            price=None,
            params={
                "stopLossPrice": trigger_price,
                "reduceOnly": True,
                "positionSide": "BOTH",
                "newClientOrderId": client_id,
            },
        )

    def _assert_protective_stop_supported(self) -> None:
        if not self.config.live:
            raise RuntimeError("Refusing native protective stop: TRADING_LIVE is not enabled.")
        if not self.supports_native_protective_stops():
            raise RuntimeError(
                "Native protective stops are restricted to Binance USDT-M futures "
                "with isolated margin (exchange=binanceusdm, quote=USDT)."
            )

    @staticmethod
    def _assert_client_order_id(client_id: str, *, label: str) -> None:
        if not isinstance(client_id, str) or not CLIENT_ORDER_ID_RE.fullmatch(client_id):
            raise ValueError(
                f"{label} must be 1-36 Binance-safe characters ([A-Za-z0-9._:/-]). Refusing."
            )

    def _protective_lookup(self, *, order_id: str | None, client_id: str) -> tuple[str, dict]:
        self._assert_client_order_id(client_id, label="Protective stop client_id")
        if order_id is not None:
            normalized_order_id = str(order_id).strip()
            if not normalized_order_id:
                raise ValueError("Protective stop order_id must be non-empty when supplied.")
            return normalized_order_id, {"trigger": True}
        # Current CCXT's Binance USD-M adapter recognizes clientAlgoId for the
        # conditional algo endpoint.  This path is needed after an ambiguous
        # placement response, when the deterministic client ID is all we have.
        return client_id, {"trigger": True, "clientAlgoId": client_id}

    @staticmethod
    def _payload_value(payload: dict, info: dict, *keys: str):
        for source in (payload, info):
            for key in keys:
                value = source.get(key)
                if value is not None:
                    return value
        return None

    @staticmethod
    def _normalized_protective_status(value) -> ProtectiveOrderStatus:
        normalized = str(value or "").strip().lower()
        if normalized in {"open", "new", "accepted", "partially_filled", "triggering"}:
            return ProtectiveOrderStatus.OPEN
        if normalized in {"closed", "filled", "triggered", "finished"}:
            return ProtectiveOrderStatus.TRIGGERED
        if normalized in {"canceled", "cancelled"}:
            return ProtectiveOrderStatus.CANCELED
        if normalized == "expired":
            return ProtectiveOrderStatus.EXPIRED
        if normalized == "rejected":
            return ProtectiveOrderStatus.REJECTED
        raise ValueError(f"Protective stop status {value!r} is missing or unsupported. Refusing.")

    @staticmethod
    def _response_bool(value, *, label: str) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"{label} must be boolean in exchange response, got {value!r}. Refusing.")

    def _parse_protective_order(self, payload: dict, *, requested_symbol: str) -> ProtectiveOrder:
        if not isinstance(payload, dict):
            raise ValueError("Protective stop exchange response must be an object. Refusing.")
        info_value = payload.get("info")
        info = info_value if isinstance(info_value, dict) else {}
        order_id, client_id, side = self._protective_identity(
            payload, info, requested_symbol=requested_symbol
        )
        qty = self._finite_number(
            self._payload_value(payload, info, "amount", "quantity", "origQty"),
            "Protective stop response quantity",
            positive=True,
        )
        trigger_price = self._finite_number(
            self._payload_value(payload, info, "triggerPrice", "stopPrice"),
            "Protective stop response trigger price",
            positive=True,
        )
        status = self._normalized_protective_status(
            self._payload_value(payload, info, "status", "algoStatus")
        )
        self._assert_protective_constraints(payload, info, qty=qty, status=status)
        filled_qty, average_price = self._protective_fill_values(
            payload, info, qty=qty, status=status
        )
        fee = self._protective_fee(payload, info)
        return ProtectiveOrder(
            symbol=requested_symbol,
            side=side,
            qty=qty,
            trigger_price=trigger_price,
            status=status,
            order_id=order_id,
            client_id=client_id,
            filled_qty=filled_qty,
            average_price=average_price,
            fee=fee,
        )

    def _protective_identity(
        self, payload: dict, info: dict, *, requested_symbol: str
    ) -> tuple[str, str, OrderSide]:
        order_id = str(self._payload_value(payload, info, "id", "algoId", "orderId") or "").strip()
        if not order_id:
            raise ValueError("Protective stop order id missing from exchange response. Refusing.")
        client_id = str(
            self._payload_value(payload, info, "clientOrderId", "clientAlgoId", "origClientOrderId")
            or ""
        ).strip()
        self._assert_client_order_id(client_id, label="Protective stop response client_id")
        symbol_value = self._payload_value(payload, info, "symbol")
        if symbol_value is None or not self._symbols_match(str(symbol_value), requested_symbol):
            raise ValueError(
                f"Protective stop symbol {symbol_value!r} does not match {requested_symbol!r}. Refusing."
            )
        side_value = str(self._payload_value(payload, info, "side") or "").lower()
        try:
            side = OrderSide(side_value)
        except ValueError as exc:
            raise ValueError(f"Protective stop side {side_value!r} is invalid. Refusing.") from exc
        return order_id, client_id, side

    def _assert_protective_constraints(
        self, payload: dict, info: dict, *, qty: float, status: ProtectiveOrderStatus
    ) -> None:
        reduce_only = self._payload_value(payload, info, "reduceOnly")
        if reduce_only is None or not self._response_bool(
            reduce_only, label="Protective stop reduceOnly"
        ):
            raise ValueError("Protective stop response does not prove reduceOnly=true. Refusing.")
        position_side = self._payload_value(payload, info, "positionSide")
        if position_side is not None and str(position_side).upper() != "BOTH":
            raise ValueError(
                f"Protective stop positionSide {position_side!r} is not one-way BOTH. Refusing."
            )
        filled_qty, average_price = self._protective_fill_values(
            payload, info, qty=qty, status=status
        )
        if status is ProtectiveOrderStatus.TRIGGERED:
            self._assert_triggered_protective_fill(qty, filled_qty, average_price)

    def _protective_fill_values(
        self, payload: dict, info: dict, *, qty: float, status: ProtectiveOrderStatus
    ) -> tuple[float, float | None]:
        filled_value = self._payload_value(
            payload, info, "filled", "executedQty", "actualQty", "cumQty", "aq"
        )
        if filled_value is None:
            if status is ProtectiveOrderStatus.TRIGGERED:
                raise ValueError(
                    "Triggered protective stop fill quantity is missing. Refusing adoption."
                )
            filled_qty = 0.0
        else:
            filled_qty = self._finite_number(
                filled_value, "Protective stop filled quantity", non_negative=True
            )
        average_value = self._payload_value(
            payload, info, "average", "avgPrice", "averagePrice", "actualPrice", "ap"
        )
        average_price = (
            None
            if average_value is None or str(average_value).strip() in {"", "0", "0.0", "0.00000000"}
            else self._finite_number(
                average_value, "Protective stop average fill price", positive=True
            )
        )
        return filled_qty, average_price

    @staticmethod
    def _assert_triggered_protective_fill(
        qty: float, filled_qty: float, average_price: float | None
    ) -> None:
        if filled_qty <= 0:
            raise ValueError(
                "Triggered protective stop has no positive fill quantity. Refusing adoption."
            )
        if average_price is None:
            raise ValueError(
                "Triggered protective stop average fill price is missing. Refusing adoption."
            )
        if abs(filled_qty - qty) > max(qty * 1e-9, 1e-12):
            raise ValueError("Triggered protective stop is not fully filled. Refusing adoption.")

    def _protective_fee(self, payload: dict, info: dict) -> float:
        fee_value = (
            (payload.get("fee") or {}).get("cost") if isinstance(payload.get("fee"), dict) else None
        )
        if fee_value is None:
            fee_value = self._payload_value({}, info, "commission", "fee")
        return (
            0.0
            if fee_value is None
            else self._finite_number(fee_value, "Protective stop fee", non_negative=True)
        )

    def _assert_protective_identity(
        self,
        protective: ProtectiveOrder,
        *,
        symbol: str,
        order_id: str | None,
        client_id: str,
    ) -> None:
        if not self._symbols_match(protective.symbol, symbol):
            raise ValueError("Protective stop response symbol mismatch. Refusing.")
        if order_id is not None and protective.order_id != str(order_id):
            raise ValueError("Protective stop response order id mismatch. Refusing.")
        if protective.client_id != client_id:
            raise ValueError("Protective stop response client id mismatch. Refusing.")

    def _assert_protective_order_matches(
        self,
        protective: ProtectiveOrder,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        trigger_price: float,
        client_id: str,
    ) -> None:
        self._assert_protective_identity(
            protective,
            symbol=symbol,
            order_id=None,
            client_id=client_id,
        )
        if protective.side != side:
            raise ValueError("Protective stop response side mismatch. Refusing.")
        qty_tolerance = max(abs(qty) * 1e-9, 1e-12)
        if abs(protective.qty - qty) > qty_tolerance:
            raise ValueError("Protective stop response quantity mismatch. Refusing.")
        trigger_tolerance = max(abs(trigger_price) * 1e-9, 1e-12)
        if abs(protective.trigger_price - trigger_price) > trigger_tolerance:
            raise ValueError("Protective stop response trigger price mismatch. Refusing.")

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
    def _assert_order_status_accepted(
        payload: dict, *, requested_quantity: float, acknowledgement_only: bool = False
    ) -> None:
        status = payload.get("status")
        if status is None:
            return
        normalized = str(status).lower()
        acknowledgement_statuses = {
            "new",
            "open",
            "accepted",
            "partially_filled",
            "ack",
            "result",
            "pending",
            "closed",
            "filled",
        }
        if acknowledgement_only:
            if normalized not in acknowledgement_statuses:
                raise ValueError(f"Exchange order status {status!r} is not accepted")
            return
        if normalized in {"new", "open", "accepted", "partially_filled"}:
            filled = float(payload.get("filled") or 0.0)
            amount = float(payload.get("amount") or requested_quantity)
            if 0 < filled < amount:
                return
        if normalized not in {"closed", "filled"}:
            raise ValueError(
                f"Exchange order status {status!r} is not closed/filled. Refusing to accept fill."
            )

    def _assert_order_response_matches(self, payload: dict, order: Order) -> None:
        response_side = payload.get("side")
        if response_side is not None and str(response_side).lower() != order.side.value:
            raise ValueError(
                f"Exchange order side {response_side!r} does not match requested side "
                f"{order.side.value!r}. Refusing to accept fill."
            )
        response_symbol = payload.get("symbol")
        if response_symbol is not None and not self._symbols_match(
            str(response_symbol), order.symbol
        ):
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
            raise ValueError(
                f"{label} notional must be finite and positive, got {notional:g}. Refusing."
            )
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
            raise ValueError(
                f"MAX_FILL_SLIPPAGE_BPS must be finite and positive, got {max_bps:g}. Refusing."
            )
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
    def _finite_number(
        value, label: str, *, positive: bool = False, non_negative: bool = False
    ) -> float:
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

    def platform_instrument_id(self, symbol: str) -> str:
        """Return the canonical platform identity for a Binance symbol."""

        base, quote, settlement = self._split_symbol(symbol)
        if self.config.market_type == "futures":
            return f"binance:futures:{base}{quote}:{settlement or quote}"
        return f"binance:spot:{base}{quote}"

    def _canonical_inventory_symbol(self, value, *, label: str) -> str:
        raw = self._sanitized_open_order_field(value, label=label)
        try:
            base, quote, settlement = self._split_symbol(raw)
        except ValueError as exc:
            raise ValueError(f"{label} is invalid: {exc} Refusing.") from exc
        return f"{base}/{quote}:{settlement or quote}"

    def _ccxt_symbol(self, symbol: str) -> str:
        base, quote, settlement = self._split_symbol(symbol)
        if self.config.market_type == "futures":
            return f"{base}/{quote}:{settlement or quote}"
        return f"{base}/{quote}"
