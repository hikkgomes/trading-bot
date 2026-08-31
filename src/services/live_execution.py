"""Fail-closed construction and approval gates for canonical live venues."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.engine import Engine

from src.data.database import account_snapshot
from src.data.database import instrument as instrument_table
from src.domain._codec import canonical_hash, timestamp, to_primitive
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
from src.services.exposure_budget import ExposureBudgetGuard

_EXECUTION_IDENTITY_FILES = (
    Path("requirements-runtime.txt"),
    Path("src/data/binance_user_stream.py"),
    Path("src/execution/ccxt_broker.py"),
    Path("src/execution/config.py"),
    Path("src/execution/live_exchange.py"),
    Path("src/services/market_gateway.py"),
    Path("src/services/order_execution.py"),
    Path("src/services/live_execution.py"),
    Path("src/services/exposure_budget.py"),
    Path("src/services/protective_stops.py"),
    Path("src/execution/stops.py"),
)


def execution_engine_identity() -> str:
    root = Path(__file__).resolve().parents[2]
    return canonical_hash(
        {str(path): (root / path).read_text(encoding="utf-8") for path in _EXECUTION_IDENTITY_FILES}
    )


def live_authority_configuration_hash(
    *,
    product: Mapping[str, Any],
    account: Mapping[str, Any],
    instrument_payload: Mapping[str, Any],
    artefact: Mapping[str, Any],
    sleeve_id: str,
    promotion_policy: Mapping[str, Any],
    risk_configuration: Mapping[str, Any],
) -> str:
    return canonical_hash(
        {
            "product": dict(product),
            "account": dict(account),
            "instrument": dict(instrument_payload),
            "sleeve_id": sleeve_id,
            "promotion_policy": dict(promotion_policy),
            "risk_configuration": dict(risk_configuration),
            "artefact_hash": str(artefact["artefact_hash"]),
            "source_commit_hash": str(artefact["source_commit_hash"]),
            "strategy_engine_version": str(artefact["engine_version"]),
            "execution_engine_identity": execution_engine_identity(),
        }
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
        self.engine = engine
        self.products = products
        self.accounts = accounts
        self.promotion_policies = _records(configuration["promotion"], "policies", "policy_id")
        self.risk_configuration = dict(configuration["risk"])
        self.assignments = SqlActiveStrategyAssignmentRepository(engine)
        self.artefacts = SqlStrategyArtefactRepository(engine)
        self.approvals = SqlApprovalRepository(engine)
        self.preflights = SqlPreflightRepository(engine)
        self.exposure_guard = ExposureBudgetGuard()
        self.product_portfolios = {
            product_id: str(product["portfolio_id"]) for product_id, product in products.items()
        }
        instruments = _load_instruments(engine)
        self.instruments = instruments
        self.venues: dict[str, BrokerExecutionVenue] = {}
        current = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        for product_id, product in live_products.items():
            if self.assignments.active(product_id, execution_mode="live", at=current) is None:
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
                    broker.platform_instrument_id(item.symbol),
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
                for conditional in (False, True)
                for item in broker.list_open_orders(
                    instrument.exchange_symbol,
                    conditional=conditional,
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
        authority_at = timestamp(
            str(payload.get("authorisation_at") or order.created_at),
            field="live authorisation time",
        )
        order_created_at = timestamp(order.created_at, field="order.created_at")
        if authority_at < order_created_at:
            raise PermissionError("live order cannot be authorised before it was created")
        if authority_at >= order.valid_until:
            raise PermissionError("live order intent has expired")
        with self.engine.connect() as connection:
            account_row = (
                connection.execute(
                    select(account_snapshot.c.payload, account_snapshot.c.observed_at)
                    .where(
                        account_snapshot.c.account_id == account_id,
                        account_snapshot.c.observed_at <= authority_at,
                    )
                    .order_by(account_snapshot.c.observed_at.desc(), account_snapshot.c.id.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
        account_payload = account_row["payload"] if account_row is not None else None
        account_observed_at = (
            timestamp(str(account_row["observed_at"]), field="account_snapshot.observed_at")
            if account_row is not None
            else None
        )
        account_age = (
            (
                dt.datetime.fromisoformat(authority_at)
                - dt.datetime.fromisoformat(account_observed_at)
            ).total_seconds()
            if account_observed_at is not None
            else None
        )
        complete_account_fields = {
            "balances",
            "free_balances",
            "positions",
            "regular_orders",
            "conditional_orders",
            "used_margin",
            "maintenance_margin",
            "used_margin_fraction",
            "liquidation_buffer_fraction",
            "account_mode",
            "unknown_exposure",
        }
        if (
            not isinstance(account_payload, Mapping)
            or account_payload.get("product_id") != product_id
            or not complete_account_fields.issubset(account_payload)
            or account_payload.get("account_state_known") is not True
            or account_payload.get("account_state_authority")
            not in {"authenticated_rest", "authenticated_reconciled"}
            or account_payload.get("unknown_exposure")
            or account_age is None
            or account_age < 0
            or account_age > int(product.get("account_snapshot_max_age_seconds", 60))
        ):
            raise PermissionError("live order requires a recent complete account snapshot")
        expected_fingerprint = str(account_payload.get("account_fingerprint") or "")
        venue = self.venues[product_id]
        actual_fingerprint = str(getattr(venue.broker, "account_fingerprint", ""))
        if not expected_fingerprint or expected_fingerprint != actual_fingerprint:
            raise PermissionError("live order account fingerprint does not match reconciliation")
        if not order.reduce_only and self.accounts[account_id].get("market") != "spot":
            supports_stops = getattr(venue.broker, "supports_native_protective_stops", None)
            if not callable(supports_stops) or not supports_stops():
                raise PermissionError("live futures entry requires native protective stops")
        requested_strategy_id = str(payload.get("strategy_version_id") or "")
        live_assignments = self.assignments.active_assignments(product_id, at=authority_at)
        current_assignment = next(
            (
                item
                for item in live_assignments
                if item["execution_mode"] == "live"
                and item.get("instrument_id") == order.instrument_id
                and (
                    not requested_strategy_id
                    or item["strategy_version_id"] == requested_strategy_id
                )
            ),
            None,
        )
        if current_assignment is None:
            raise PermissionError("live instrument has no active canonical assignment")
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
            at=authority_at,
            instrument_id=order.instrument_id,
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
            at=authority_at,
        )
        if approval is None or approval["status"] != "approved":
            raise PermissionError("canonical live artefact has no current human approval")
        preflight = self.preflights.latest(
            strategy_version_id=strategy_version_id,
            product_id=product_id,
            account_id=account_id,
            at=authority_at,
        )
        if preflight is None or not preflight["accepted"]:
            raise PermissionError("canonical live artefact has no accepted preflight")
        if not preflight_is_fresh(
            str(preflight["checked_at"]),
            reference_at=authority_at,
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
        authority_instrument = self.instruments.get(str(assignment.get("instrument_id") or ""))
        if authority_instrument is None:
            raise PermissionError("live assignment instrument is not persisted")
        instrument_payload = dict(to_primitive(authority_instrument))
        instrument_payload["instrument_id"] = authority_instrument.instrument_id
        expected_engine_identity = execution_engine_identity()
        expected_configuration_hash = live_authority_configuration_hash(
            product=product,
            account=self.accounts[account_id],
            instrument_payload=instrument_payload,
            artefact=artifact,
            sleeve_id=str(assignment["sleeve_id"]),
            promotion_policy=self.promotion_policies[str(product["promotion_policy_id"])],
            risk_configuration=self.risk_configuration,
        )
        preflight_payload = preflight.get("payload")
        approval_payload = approval.get("payload")
        if (
            not isinstance(preflight_payload, Mapping)
            or preflight_payload.get("schema") != "platform.production-preflight/v1"
            or preflight_payload.get("environment") != self.accounts[account_id]["environment"]
            or preflight_payload.get("account_fingerprint") != actual_fingerprint
            or preflight_payload.get("execution_engine_identity") != expected_engine_identity
            or preflight_payload.get("instrument_id") != authority_instrument.instrument_id
            or preflight_payload.get("sleeve_id") != assignment["sleeve_id"]
            or preflight_payload.get("configuration_hash") != expected_configuration_hash
            or preflight.get("content_hash") != canonical_hash(dict(preflight_payload))
        ):
            raise PermissionError("canonical preflight does not match current live authority")
        if (
            not isinstance(approval_payload, Mapping)
            or approval_payload.get("schema") != "platform.strategy-approval/v1"
            or approval_payload.get("preflight_id") != preflight.get("id")
            or approval_payload.get("environment") != self.accounts[account_id]["environment"]
            or approval_payload.get("account_fingerprint") != actual_fingerprint
            or approval_payload.get("execution_engine_identity") != expected_engine_identity
            or approval_payload.get("instrument_id") != authority_instrument.instrument_id
            or approval_payload.get("sleeve_id") != assignment["sleeve_id"]
            or approval_payload.get("configuration_hash") != expected_configuration_hash
        ):
            raise PermissionError("canonical approval does not match current live authority")
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
        self.exposure_guard.enforce(
            product_id=product_id,
            product=product,
            account=self.accounts[account_id],
            assignment=assignment,
            risk_configuration=self.risk_configuration,
            account_payload=account_payload,
            order=order,
            positions=self.positions.all(),
            orders=self.order_manager.all(),
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
