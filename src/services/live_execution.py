"""Fail-closed construction and approval gates for canonical live venues."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.engine import Engine

from src.data.database import instrument as instrument_table
from src.domain._codec import canonical_hash
from src.domain.instruments import Instrument, MarketType
from src.domain.orders import OrderIntent
from src.execution.ccxt_broker import CcxtBroker
from src.execution.config import ExchangeConfig
from src.execution.live_exchange import BrokerExecutionVenue
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.reconciler import ReconciliationResult, reconcile_account
from src.research.canonical import (
    SqlActiveStrategyAssignmentRepository,
    SqlApprovalRepository,
    SqlPreflightRepository,
    SqlStrategyArtefactRepository,
    preflight_is_fresh,
)


class ApprovedLiveExecution:
    """Build live adapters only from canonical PostgreSQL-backed authority."""

    def __init__(
        self,
        *,
        engine: Engine,
        configuration: Mapping[str, Mapping[str, Any]],
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
        self.order_manager = order_manager
        self.positions = positions
        self.products = products
        self.assignments = SqlActiveStrategyAssignmentRepository(engine)
        self.artefacts = SqlStrategyArtefactRepository(engine)
        self.approvals = SqlApprovalRepository(engine)
        self.preflights = SqlPreflightRepository(engine)
        self.product_portfolios = {
            product_id: str(product["portfolio_id"]) for product_id, product in products.items()
        }
        instruments = _load_instruments(engine)
        self.venues: dict[str, BrokerExecutionVenue] = {}
        for product_id, product in live_products.items():
            if self.assignments.active(product_id) is None:
                raise ValueError(
                    f"live product {product_id} has no active canonical strategy assignment"
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
        broker = cast(CcxtBroker, venue.broker)
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
        product = self.products[product_id]
        account_id = str(product["account_id"])
        current_assignment = self.assignments.active(product_id)
        if current_assignment is None:
            raise PermissionError("live product has no active canonical assignment")
        requested_strategy_id = str(payload.get("strategy_version_id") or "")
        if (
            requested_strategy_id
            and requested_strategy_id != current_assignment["strategy_version_id"]
        ):
            raise PermissionError("live order strategy identity does not match assignment")
        requested_artefact_hash = str(payload.get("artefact_hash") or "")
        if (
            requested_artefact_hash
            and requested_artefact_hash != current_assignment["artefact_hash"]
        ):
            raise PermissionError("live order artefact identity does not match assignment")
        assignment = self.assignments.assert_binding(
            product_id=product_id,
            strategy_version_id=str(current_assignment["strategy_version_id"]),
            artefact_hash=str(current_assignment["artefact_hash"]),
            execution_mode="live",
        )
        artifact = self.artefacts.get(str(assignment["artefact_hash"]))
        declared_hash = artifact.get("artefact_hash")
        content = dict(artifact)
        content.pop("artefact_hash", None)
        if declared_hash != assignment["artefact_hash"] or canonical_hash(content) != declared_hash:
            raise PermissionError("canonical live artefact content hash does not match assignment")
        strategy_version_id = str(artifact.get("strategy_version_id") or "")
        if strategy_version_id != assignment["strategy_version_id"]:
            raise PermissionError(
                "canonical live artefact strategy identity does not match assignment"
            )
        if artifact.get("product_id") != product_id or artifact.get("account_id") != account_id:
            raise PermissionError("canonical live artefact account or product does not match")
        if artifact.get("portfolio_id") != assignment["portfolio_id"]:
            raise PermissionError("canonical live artefact portfolio does not match assignment")
        approval = self.approvals.latest(
            strategy_version_id=strategy_version_id,
            product_id=product_id,
            account_id=account_id,
        )
        if approval is None or approval["status"] != "approved":
            raise PermissionError("canonical live artefact has no current human approval")
        preflight = self.preflights.latest(
            strategy_version_id=strategy_version_id,
            product_id=product_id,
            account_id=account_id,
        )
        if preflight is None or not preflight["accepted"]:
            raise PermissionError("canonical live artefact has no accepted preflight")
        if not preflight_is_fresh(
            str(preflight["checked_at"]),
            maximum_age_seconds=int(product.get("preflight_max_age_seconds", 3_600)),
        ):
            raise PermissionError("canonical live artefact preflight is stale")
        for record, label in ((approval, "approval"), (preflight, "preflight")):
            if any(
                record[field] != artifact.get(field)
                for field in ("artefact_hash", "source_commit_hash", "engine_version")
            ):
                raise PermissionError(f"canonical {label} is not bound to the live artefact")
        if float(assignment["capital_limit"]) > min(
            float(approval["capital_cap"]), float(preflight["capital_cap"])
        ):
            raise PermissionError("active assignment exceeds approved/preflight capital cap")
        strategies = {
            str(item.get("id") or ""): item
            for item in artifact.get("strategies", [])
            if isinstance(item, dict)
        }
        if not strategies:
            strategies = {
                strategy_version_id: {
                    "id": strategy_version_id,
                    "supported_instruments": artifact.get("supported_instruments", []),
                }
            }
        contributions = set(order.strategy_contributions)
        if not contributions or not contributions <= set(strategies):
            raise PermissionError(
                "live order strategy contributions are not bound to the approved artefact"
            )
        instrument = self.venues[product_id].instruments.get(order.instrument_id)
        if instrument is None:
            raise PermissionError("live order instrument is not persisted and tradable")
        allowed_instruments = set(artifact.get("supported_instruments", []))
        if instrument.instrument_id not in allowed_instruments:
            raise PermissionError("live order instrument is not bound to the canonical artefact")


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
