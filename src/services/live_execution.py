"""Fail-closed construction and approval gates for canonical live venues."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine

from src.autopilot.approvals import (
    artifact_digest,
    assert_loaded_artifact_live_approved,
    load_artifact,
    load_production_preflight_evidence,
)
from src.autopilot.config import load_config as load_autopilot_config
from src.autopilot.runtime import assert_recent_testnet_rehearsal
from src.data.database import instrument as instrument_table
from src.domain.instruments import Instrument, MarketType
from src.domain.orders import OrderIntent
from src.execution.ccxt_broker import CcxtBroker
from src.execution.config import ExchangeConfig
from src.execution.live_exchange import BrokerExecutionVenue
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.reconciler import ReconciliationResult, reconcile_account


class ApprovedLiveExecution:
    """Build live adapters only when split and mandatory legacy gates agree."""

    def __init__(
        self,
        *,
        engine: Engine,
        configuration: Mapping[str, Mapping[str, Any]],
        config_root: Path,
        order_manager: OrderManager,
        positions: PositionManager,
    ) -> None:
        products = _records(configuration["products"], "products", "product_id")
        accounts = _records(configuration["accounts"], "accounts", "account_id")
        live_products = {
            product_id: product
            for product_id, product in products.items()
            if product["execution_mode"] == "live"
        }
        autopilot = load_autopilot_config(config_root / "autopilot.json", strict_jobs=False)
        self.legacy_products = {product.name: product for product in autopilot.products}
        self.approval_ledger = autopilot.approval_ledger
        self.order_manager = order_manager
        self.positions = positions
        self.product_portfolios = {
            product_id: str(product["portfolio_id"]) for product_id, product in products.items()
        }
        instruments = _load_instruments(engine)
        self.venues: dict[str, BrokerExecutionVenue] = {}
        for product_id, product in live_products.items():
            legacy = self.legacy_products.get(product_id)
            if legacy is None or legacy.execution_mode != "live":
                raise ValueError(
                    f"live product {product_id} must also be live in config/autopilot.json"
                )
            account = accounts[str(product["account_id"])]
            market = "spot" if account["market"] == "spot" else "futures"
            product_instruments = {
                identity: item
                for identity, item in instruments.items()
                if item.market_type.value == market
            }
            if not product_instruments:
                raise ValueError(f"live product {product_id} has no persisted instruments")
            exchange = _exchange_config(account, market=market)
            self.venues[product_id] = BrokerExecutionVenue(
                order_manager=order_manager,
                position_manager=positions,
                broker=CcxtBroker(exchange),
                instruments=product_instruments,
            )

    def reconcile(self, product_id: str) -> ReconciliationResult:
        """Compare complete authenticated venue state with durable local state."""
        venue = self.venues[product_id]
        broker = venue.broker
        symbol_to_instrument = {
            instrument.exchange_symbol: identity
            for identity, instrument in venue.instruments.items()
        }
        self.order_manager.reload()
        self.positions.reload()
        portfolio_id = self.product_portfolios[product_id]
        local_positions = self.positions.current_quantities(portfolio_id)
        local_orders = {
            order.order_id[:36]
            for order in self.order_manager.all()
            if order.portfolio_id == portfolio_id and not order.is_terminal
        }
        if broker.config.market_type == "futures":
            exchange_positions = {
                symbol_to_instrument.get(
                    item.symbol,
                    f"binance:futures:{item.symbol}:{broker.config.quote_asset}",
                ): item.qty
                for item in broker.list_account_futures_positions()
            }
            exchange_orders = {
                item.client_id or f"exchange:{item.order_id}"
                for conditional in (False, True)
                for item in broker.list_account_open_orders(conditional=conditional)
            }
        else:
            exchange_positions = {
                identity: position.qty
                for identity, instrument in venue.instruments.items()
                if abs((position := broker.get_position(instrument.exchange_symbol)).qty) > 1e-12
            }
            exchange_orders = {
                item.client_id or f"exchange:{item.order_id}"
                for instrument in venue.instruments.values()
                for item in broker.list_open_orders(
                    instrument.exchange_symbol,
                    conditional=False,
                )
            }
        return reconcile_account(
            local_positions=local_positions,
            exchange_positions=exchange_positions,
            local_open_order_ids=local_orders,
            exchange_open_order_ids=exchange_orders,
        )

    def authorise(self, payload: Mapping[str, Any], order: OrderIntent) -> None:
        product_id = str(payload["product_id"])
        product = self.legacy_products[product_id]
        artifact = load_artifact(product.strategies_path)
        digest = artifact_digest(artifact)
        load_production_preflight_evidence(
            product,
            expected_artifact_digest=digest,
            require_fresh=True,
        )
        assert_loaded_artifact_live_approved(
            artifact,
            product.strategies_path,
            self.approval_ledger,
            product=product,
        )
        assert_recent_testnet_rehearsal(product, artifact=artifact)
        strategies = {
            str(item.get("id") or ""): item
            for item in artifact["strategies"]
            if isinstance(item, dict)
        }
        contributions = set(order.strategy_contributions)
        if not contributions or not contributions <= set(strategies):
            raise PermissionError(
                "live order strategy contributions are not bound to the approved artefact"
            )
        instrument = self.venues[product_id].instruments.get(order.instrument_id)
        if instrument is None:
            raise PermissionError("live order instrument is not persisted and tradable")
        if any(
            str(strategies[strategy_id].get("symbol") or "").upper() != instrument.exchange_symbol
            for strategy_id in contributions
        ):
            raise PermissionError(
                "live order instrument does not match each approved contributing strategy"
            )


def _records(
    payload: Mapping[str, Any], collection: str, identity: str
) -> dict[str, dict[str, Any]]:
    rows = payload.get(collection)
    if not isinstance(rows, list):
        raise ValueError(f"{collection} must be a list")
    return {str(row[identity]): dict(row) for row in rows}


def _load_instruments(engine: Engine) -> dict[str, Instrument]:
    with engine.connect() as connection:
        rows = connection.execute(select(instrument_table.c.id, instrument_table.c.payload))
        result: dict[str, Instrument] = {}
        for identity, payload in rows:
            values = dict(payload)
            values["market_type"] = MarketType(values["market_type"])
            item = Instrument(**values)
            if item.instrument_id != identity:
                raise ValueError(f"persisted instrument identity mismatch: {identity}")
            result[identity] = item
        return result


def _exchange_config(account: Mapping[str, Any], *, market: str) -> ExchangeConfig:
    config = ExchangeConfig.from_env(market_type=market)
    api_key_name = str(account["api_key_env"])
    api_secret_name = str(account["api_secret_env"])
    api_key = os.environ.get(api_key_name, "").strip()
    api_secret = os.environ.get(api_secret_name, "").strip()
    if not api_key or not api_secret:
        raise ValueError(f"live account requires {api_key_name} and {api_secret_name}")
    production = account["environment"] == "production"
    if production == config.testnet:
        expected = "0" if production else "1"
        raise ValueError(f"EXCHANGE_TESTNET must be {expected} for this account")
    if not config.live:
        raise ValueError("TRADING_LIVE must be enabled for a live product")
    return replace(
        config,
        exchange="binance" if market == "spot" else "binanceusdm",
        api_key=api_key,
        api_secret=api_secret,
        max_futures_leverage=int(account["maximum_leverage"]),
        quote_asset=str(account["quote_assets"][0]),
        allow_multi_symbol_positions=market == "futures",
    )
