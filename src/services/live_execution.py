"""Fail-closed construction and approval gates for canonical live venues."""

from __future__ import annotations

import datetime as dt
import math
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from sqlalchemy import or_, select
from sqlalchemy.engine import Engine

from src.accounting.fees import FeeConversionError, convert_fee, instrument_asset
from src.data.database import (
    account_snapshot,
    market_input_sources,
    universe_member,
    universe_snapshot,
)
from src.data.database import instrument as instrument_table
from src.domain._codec import canonical_hash, timestamp, to_primitive
from src.domain.instruments import Instrument, MarketType
from src.domain.orders import Fill, OrderIntent, OrderSide, OrderStatus, OrderType
from src.execution.broker import BrokerFill
from src.execution.ccxt_broker import CcxtBroker
from src.execution.config import ExchangeConfig
from src.execution.live_exchange import BrokerExecutionVenue
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.execution.reconciler import ReconciliationResult, reconcile_account
from src.execution.recovery import RecoveryAction, RecoveryActionType
from src.execution.stops import SqlStopStore, StopManager
from src.research.canonical import (
    SqlActiveStrategyAssignmentRepository,
    SqlApprovalRepository,
    SqlPreflightRepository,
    SqlStrategyArtefactRepository,
    preflight_is_fresh,
)
from src.risk.engine import SqlRiskSnapshotStore
from src.services.exposure_budget import ExposureBudgetGuard
from src.services.financial_application import FinancialApplication, FinancialScope
from src.services.income_recovery import checkpoint_income, income_window, reconcile_income
from src.services.runtime import utc_now


def execution_engine_identity() -> str:
    from src.services.execution_identity import execution_engine_digest

    return execution_engine_digest()


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
        ledgers: Mapping[str, Any] | None = None,
        queue_scope: str | None = None,
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
        self.ledgers = dict(ledgers or {})
        self.engine = engine
        self.queue_scope = queue_scope
        self.loaded_execution_identity = execution_engine_identity()
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
            assignments = tuple(
                row
                for row in self.assignments.active_assignments(product_id, at=current)
                if row["execution_mode"] == "live"
            )
            if not assignments:
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
            allowed_ids = _live_assignment_instrument_ids(
                engine,
                assignments,
                product=product,
                at=current,
            )
            product_instruments = {
                identity: item
                for identity, item in product_instruments.items()
                if identity in allowed_ids
            }
            configured_symbols = product.get("live_exchange_symbols")
            if isinstance(configured_symbols, list):
                allowed_symbols = {str(value).upper() for value in configured_symbols}
                product_instruments = {
                    identity: item
                    for identity, item in product_instruments.items()
                    if item.exchange_symbol.upper() in allowed_symbols
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
        local_orders: set[str] = set()
        for order in self.order_manager.all():
            if order.portfolio_id != portfolio_id or order.is_terminal:
                continue
            local_orders.add(
                str(
                    order.metadata.get("client_order_id")
                    or order.metadata.get("exchange_order_id")
                    or order.order_id
                )
            )
        if broker.config.market_type == "futures":
            local_orders.update(self._protection_order_ids(portfolio_id))
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

    def _protection_order_ids(self, portfolio_id: str) -> set[str]:
        return {
            str(client_id)
            for stop in StopManager(SqlStopStore(self.engine)).active()
            if stop.portfolio_id == portfolio_id
            for order_id, client_id in (
                (stop.native_order_id, stop.native_client_id),
                (stop.previous_native_order_id, stop.previous_native_client_id),
            )
            if order_id and client_id
        }

    def recover_action(self, product_id: str, action: RecoveryAction) -> Mapping[str, Any]:
        """Execute one bounded, idempotent recovery action on the approved venue."""

        venue = self.venues[product_id]
        broker = venue.broker
        if action.action_type is RecoveryActionType.CANCEL_ORDER:
            return self._cancel_known_order(product_id, action.target)
        if action.action_type is RecoveryActionType.CANCEL_UNKNOWN_ORDER:
            return self._cancel_unknown_order(broker, action.target)
        if action.action_type is RecoveryActionType.RECONCILE_ORDER:
            return self._reconcile_missing_order(product_id, action.target)
        if action.action_type is RecoveryActionType.RECONCILE_POSITION:
            return self._reconcile_position(product_id, action.target, action.quantity)
        if action.action_type is RecoveryActionType.EMERGENCY_FLATTEN:
            return self._emergency_flatten(
                product_id, action.target, action.quantity, operation_id=action.operation_id
            )
        raise ValueError(f"unsupported recovery action: {action.action_type}")

    def _cancel_known_order(self, product_id, identity):
        self.order_manager.reload()
        order = self.order_manager.get(identity)
        if order.portfolio_id != self.product_portfolios[product_id]:
            raise PermissionError("entry cancellation crosses the reconciled product account")
        if order.is_terminal:
            return {"order_id": identity, "status": order.status.value}
        if order.status in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}:
            self.order_manager.request_cancel(identity)
        venue = self.venues[product_id]
        venue.broker.cancel_order(
            symbol=venue.instruments[order.instrument_id].exchange_symbol,
            exchange_order_id=str(order.metadata.get("exchange_order_id") or ""),
            client_order_id=str(order.metadata.get("client_order_id") or ""),
        )
        result = self._reconcile_missing_order(product_id, identity)
        if not self.order_manager.get(identity).is_terminal:
            raise RuntimeError("triggered protection entry cancellation remains pending")
        return result

    def _cancel_unknown_order(self, broker: Any, target: str) -> Mapping[str, Any]:
        matches = []
        for conditional in (False, True):
            matches.extend(broker.list_account_open_orders(conditional=conditional))
        matching = tuple(
            item
            for item in matches
            if target in {item.order_id, item.client_id, f"exchange:{item.order_id}"}
        )
        if not matching:
            return {"action": "cancel_unknown_order", "target": target, "status": "absent"}
        if len(matching) != 1:
            raise RuntimeError(f"recovery order identity is ambiguous: {target}")
        item = matching[0]
        result = broker.cancel_order(
            symbol=item.symbol,
            exchange_order_id=item.order_id,
            client_order_id=item.client_id,
        )
        status = str(result.status).casefold()
        if status in {"open", "new", "accepted", "partially_filled"}:
            raise RuntimeError(f"unknown exchange order remains open: {target}")
        return {"action": "cancel_unknown_order", "target": target, "status": status}

    def _reconcile_missing_order(self, product_id: str, target: str) -> Mapping[str, Any]:
        self.order_manager.reload()
        order = self._recovery_order(target)
        if order is None:
            return {"action": "reconcile_order", "target": target, "status": "local_absent"}
        state, recovered = self._recovery_exchange_state(product_id, order, target)
        _validate_recovered_fills(
            order=order,
            fills=recovered,
            exchange_order_id=state.exchange_order_id
            or str(order.metadata.get("exchange_order_id") or ""),
            client_order_id=state.client_order_id
            or str(order.metadata.get("client_order_id") or ""),
        )
        _validate_recovered_quantities(
            order,
            state,
            recovered,
            self.order_manager.fills_for(order.order_id),
        )
        recovered_count = self._apply_recovered_fills(product_id, order, recovered)
        self._finish_recovered_order(order.order_id, state.status.casefold())
        numeric_ids = [int(fill.trade_id) for fill in recovered if fill.trade_id.isdigit()]
        next_id = (
            str(max(numeric_ids) + 1)
            if numeric_ids and len(numeric_ids) == len(recovered)
            else str(order.metadata.get("rest_checkpoint", {}).get("next_trade_id", "0"))
        )
        self.order_manager.checkpoint_rest(order.order_id, next_id)
        return {
            "action": "reconcile_order",
            "target": target,
            "order_id": order.order_id,
            "status": self.order_manager.get(order.order_id).status.value,
            "recovered_fills": recovered_count,
        }

    def _recovery_order(self, target: str) -> OrderIntent | None:
        matches = tuple(
            order
            for order in self.order_manager.all()
            if target
            in {
                order.order_id,
                str(order.metadata.get("client_order_id") or ""),
                str(order.metadata.get("exchange_order_id") or ""),
            }
        )
        if len(matches) > 1:
            raise RuntimeError(f"local recovery order identity is ambiguous: {target}")
        return matches[0] if matches else None

    def _recovery_exchange_state(
        self, product_id: str, order: OrderIntent, target: str
    ) -> tuple[Any, tuple[BrokerFill, ...]]:
        venue = self.venues[product_id]
        exchange_id = str(order.metadata.get("exchange_order_id") or "")
        client_id = str(order.metadata.get("client_order_id") or "")
        symbol = str(venue.instruments[order.instrument_id].exchange_symbol)
        state = venue.broker.query_order(
            symbol=symbol,
            exchange_order_id=exchange_id,
            client_order_id=client_id,
        )
        _validate_exchange_recovery_state(state, target=target)
        fills_reader = getattr(venue.broker, "query_order_fills", None)
        if not callable(fills_reader):
            raise RuntimeError("live recovery cannot prove missing fills from the exchange")
        return state, tuple(
            fills_reader(
                symbol=symbol,
                exchange_order_id=state.exchange_order_id or exchange_id,
                client_order_id=state.client_order_id or client_id,
                from_trade_id=str(
                    order.metadata.get("rest_checkpoint", {}).get("next_trade_id", "0")
                ),
            )
        )

    def _finish_recovered_order(self, order_id: str, state_status: str) -> None:
        if state_status in {"open", "new", "accepted", "partially_filled"}:
            return
        current = self.order_manager.get(order_id)
        if current.status is OrderStatus.RECONCILED:
            return
        if state_status in {"canceled", "cancelled"} and not current.is_terminal:
            if current.status not in {OrderStatus.CANCEL_PENDING, OrderStatus.RECOVERY_REQUIRED}:
                self.order_manager.recovery_required(order_id)
            self.order_manager.cancelled(order_id)
        elif state_status in {"rejected", "expired"} and current.status in {
            OrderStatus.RECOVERY_REQUIRED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            self.order_manager.transition(
                order_id,
                OrderStatus.REJECTED if state_status == "rejected" else OrderStatus.EXPIRED,
            )
        current = self.order_manager.get(order_id)
        if current.status is not OrderStatus.RECONCILED:
            self.order_manager.reconcile(order_id)

    def _apply_recovered_fills(
        self, product_id: str, order: OrderIntent, fills: tuple[BrokerFill, ...]
    ) -> int:
        existing_fills = self.order_manager.fills_for(order.order_id)
        account_id = str(self.products[product_id]["account_id"])
        ledger = self.ledgers.get(product_id)
        if ledger is None:
            raise RuntimeError("recovered fills require a product ledger")
        application = FinancialApplication(
            self.engine,
            FinancialScope(
                product_id,
                account_id,
                str(self.accounts[account_id]["environment"]),
                "binance",
                ledger.accounting_asset,
            ),
            queue_scope=self.queue_scope,
        )
        application.assert_order_reconciled(order.order_id)
        recovered = 0
        for broker_fill in fills:
            if _matches_existing_recovered_fill(existing_fills, broker_fill):
                continue
            conversion = self.fee_conversion_metadata(
                product_id,
                order.instrument_id,
                broker_fill.fee_asset,
                broker_fill.fee,
                broker_fill.price,
            )
            result = application.apply_fill(_domain_fill(order, broker_fill, metadata=conversion))
            recovered += int(result.applied)
            existing_fills = (*existing_fills, result.fill)
        self.order_manager.reload()
        self.positions.reload()
        ledger.reload()
        return recovered

    def backfill_account(self, product_id: str, at: str) -> Mapping[str, Any]:
        """Backfill commission, funding, and realised PnL from Binance REST."""

        venue = self.venues[product_id]
        query = getattr(venue.broker, "query_income", None)
        ledger = self.ledgers.get(product_id)
        if not callable(query) or ledger is None:
            raise RuntimeError("account backfill requires a broker income reader and ledger")
        ledger.reload()
        account_id = str(self.products[product_id]["account_id"])
        scope = FinancialScope(
            product_id,
            account_id,
            str(self.accounts[account_id]["environment"]),
            "binance",
            ledger.accounting_asset,
        )
        self.order_manager.reload()
        created_at = [
            order.created_at
            for order in self.order_manager.all()
            if order.portfolio_id == self.product_portfolios[product_id]
        ]
        since, until = income_window(ledger, scope, created_at, at)
        incomes = tuple(query(since=since, until=until))
        ledger.reload()
        recorded = 0
        for income in incomes:
            if not since <= income.occurred_at <= until:
                raise ValueError("exchange income is outside the requested recovery window")
            if income.income_type.casefold() not in {
                "funding_fee",
                "commission",
                "commission_fee",
                "realized_pnl",
                "realised_pnl",
            }:
                raise ValueError(
                    f"exchange income type requires explicit reconciliation: {income.income_type}"
                )
            recorded += int(reconcile_income(self.engine, ledger, scope, income))
        checkpoint_income(ledger, scope, until, at)
        return {
            "product_id": product_id,
            "income_records": len(incomes),
            "recorded": recorded,
            "through": until,
            "history_complete": until == dt.datetime.fromisoformat(at).timestamp(),
        }

    def fee_conversion_metadata(
        self,
        product_id: str,
        instrument_id: str,
        fee_asset: str | None,
        fee: float,
        price: float,
    ) -> Mapping[str, Any]:
        ledger = self.ledgers.get(product_id)
        if ledger is None:
            raise RuntimeError(f"no accounting ledger is configured for {product_id}")
        base_asset = _instrument_asset(instrument_id, "base")
        quote_asset = _instrument_asset(instrument_id, "quote")
        try:
            conversion = convert_fee(
                amount=fee,
                fee_asset=fee_asset,
                accounting_asset=ledger.accounting_asset,
                trade_price=price,
                base_asset=base_asset,
                quote_asset=quote_asset,
            )
        except FeeConversionError:
            asset = str(fee_asset or "").upper()
            conversion_price = float(
                self.venues[product_id].broker.get_price(f"{asset}{ledger.accounting_asset}")
            )
            conversion = convert_fee(
                amount=fee,
                fee_asset=asset,
                accounting_asset=ledger.accounting_asset,
                trade_price=price,
                base_asset=base_asset,
                quote_asset=quote_asset,
                metadata={"fee_conversion_rate": conversion_price},
            )
        return {
            "fee_in_accounting_asset": conversion.accounting_amount,
            "fee_conversion_rate": conversion.conversion_rate,
            "fee_conversion_source": conversion.source,
        }

    def _reconcile_position(
        self, product_id: str, instrument_id: str, quantity: float | None
    ) -> Mapping[str, Any]:
        venue = self.venues[product_id]
        broker = venue.broker
        instrument = venue.instruments.get(instrument_id)
        if instrument is None:
            raise RuntimeError(f"recovery instrument is not approved: {instrument_id}")
        observed_quantity = quantity
        average_price = 0.0
        if broker.config.market_type == "futures":
            matches = tuple(
                item
                for item in broker.list_account_futures_positions()
                if broker._symbols_match(item.symbol, instrument.exchange_symbol)
            )
            if len(matches) > 1:
                raise RuntimeError(f"recovery position identity is ambiguous: {instrument_id}")
            observed_quantity = matches[0].qty if matches else 0.0
            average_price = matches[0].avg_price if matches else 0.0
        else:
            position = broker.get_position(instrument.exchange_symbol)
            observed_quantity = position.qty
            average_price = position.avg_price
        if observed_quantity is None:
            raise RuntimeError(f"recovery position quantity is missing: {instrument_id}")
        self.positions.reload()
        local = self.positions.get(self.product_portfolios[product_id], instrument_id)
        if abs(local.quantity - float(observed_quantity)) > 1e-12:
            raise RuntimeError("position mismatch requires exact fill-history reconciliation")
        if local.quantity and abs(local.average_entry_price - average_price) > max(
            1e-8, average_price * 1e-9
        ):
            raise RuntimeError("position cost basis differs from the financial ledger")
        return {
            "action": "reconcile_position",
            "instrument_id": instrument_id,
            "quantity": float(observed_quantity),
            "status": "reconciled",
        }

    def _emergency_flatten(
        self,
        product_id: str,
        instrument_id: str,
        quantity: float | None,
        *,
        operation_id: str | None,
    ) -> Mapping[str, Any]:
        if not operation_id:
            raise ValueError("emergency recovery requires a durable operation identity")
        venue = self.venues[product_id]
        instrument = venue.instruments.get(instrument_id)
        if instrument is None:
            raise RuntimeError(f"recovery instrument is not approved: {instrument_id}")
        broker = venue.broker
        signed_quantity = float(quantity or 0.0)
        if abs(signed_quantity) <= 1e-12:
            if broker.config.market_type == "futures":
                positions = tuple(
                    item
                    for item in broker.list_account_futures_positions()
                    if broker._symbols_match(item.symbol, instrument.exchange_symbol)
                )
                if len(positions) > 1:
                    raise RuntimeError(f"recovery position identity is ambiguous: {instrument_id}")
                signed_quantity = positions[0].qty if positions else 0.0
            else:
                signed_quantity = broker.get_position(instrument.exchange_symbol).qty
        if abs(signed_quantity) <= 1e-12:
            return {"action": "emergency_flatten", "instrument_id": instrument_id, "status": "flat"}
        reference_price = float(broker.get_price(instrument.exchange_symbol))
        unsigned = {
            "operation_id": operation_id,
            "product_id": product_id,
            "instrument_id": instrument_id,
            "quantity": abs(signed_quantity),
            "side": "sell" if signed_quantity > 0 else "buy",
            "kind": "emergency_flatten",
        }
        order_id = "recovery:" + canonical_hash(unsigned).removeprefix("sha256:")[:40]
        self.order_manager.reload()
        existing = next(
            (order for order in self.order_manager.all() if order.order_id == order_id), None
        )
        if existing is not None:
            result = {
                "action": "emergency_flatten",
                "instrument_id": instrument_id,
                "order_id": order_id,
                "status": existing.status.value,
            }
            if self._recovery_instrument_flat(broker, instrument.exchange_symbol, instrument_id):
                result["status"] = "flat_reconciled"
            else:
                result["status"] = "pending_reconciliation"
            return result
        intent = OrderIntent(
            order_id=order_id,
            portfolio_id=self.product_portfolios[product_id],
            instrument_id=instrument_id,
            side=OrderSide.SELL if signed_quantity > 0 else OrderSide.BUY,
            quantity=abs(signed_quantity),
            order_type=OrderType.MARKET,
            created_at=utc_now(),
            reduce_only=True,
            strategy_contributions={"recovery": 1.0},
            metadata={
                "recovery": True,
                "reason_code": "emergency_flatten",
                "reference_price": reference_price,
            },
        )
        from src.portfolio.account_plans import prepare_book_flatten

        account = self.accounts[self.products[product_id]["account_id"]]
        intent = prepare_book_flatten(
            self.engine,
            FinancialScope(
                product_id,
                account["account_id"],
                account["environment"],
                account["venue"],
                self.products[product_id]["base_accounting_asset"],
            ),
            intent,
        )
        self.order_manager.reload()
        self.positions.reload()
        venue.submit(intent)
        status = (
            "flat_reconciled"
            if self._recovery_instrument_flat(broker, instrument.exchange_symbol, instrument_id)
            else "pending_reconciliation"
        )
        return {
            "action": "emergency_flatten",
            "instrument_id": instrument_id,
            "order_id": order_id,
            "status": status,
        }

    @staticmethod
    def _recovery_instrument_flat(broker: Any, exchange_symbol: str, instrument_id: str) -> bool:
        reader = getattr(broker, "account_snapshot", None)
        if not callable(reader):
            return False
        state = reader(expected_symbols=(exchange_symbol,))
        if not isinstance(state, Mapping):
            return False
        if (
            state.get("account_state_known") is not True
            or state.get("account_state_authority") != "authenticated_rest"
        ):
            return False
        positions = state.get("positions")
        if not isinstance(positions, Mapping):
            return False
        quantity = positions.get(instrument_id, positions.get(exchange_symbol, 0.0))
        try:
            return abs(float(quantity)) <= 1e-12
        except (TypeError, ValueError):
            return False

    def authorise(self, payload: Mapping[str, Any], order: OrderIntent) -> None:
        if execution_engine_identity() != self.loaded_execution_identity:
            raise PermissionError("execution code changed after process startup")
        if order.metadata.get("target_metadata", {}).get("account_plan_id"):
            self._authorise_account_order(payload, order)
            self._assert_market_admission(payload, order)
            return
        product_id, product, account_id, authority_at = self._order_context(payload, order)
        if not order.reduce_only:
            self._assert_financial_followups_complete(account_id)
        account_payload, actual_fingerprint = self._validated_account_authority(
            product_id=product_id,
            account_id=account_id,
            authority_at=authority_at,
            product=product,
        )
        venue = self.venues[product_id]
        if not order.reduce_only and self.accounts[account_id].get("market") != "spot":
            supports_stops = getattr(venue.broker, "supports_native_protective_stops", None)
            if not callable(supports_stops) or not supports_stops():
                raise PermissionError("live futures entry requires native protective stops")
        assignment, artifact = self._validated_strategy_authority(
            payload=payload,
            order=order,
            product_id=product_id,
            account_id=account_id,
            authority_at=authority_at,
            product=product,
            actual_fingerprint=actual_fingerprint,
        )
        self._assert_order_strategy_scope(
            order=order,
            product_id=product_id,
            artifact=artifact,
            assignment=assignment,
        )
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
        self._assert_market_admission(payload, order)

    def _assert_market_admission(self, payload, order):
        if order.reduce_only:
            return
        product_id, _product, _account, now = self._order_context(payload, order)
        maximum_age = float(
            self.risk_configuration["global"]["maximum_market_data_staleness_seconds"]
        )
        oldest = (dt.datetime.fromisoformat(now) - dt.timedelta(seconds=maximum_age)).isoformat()
        source = market_input_sources
        with self.engine.connect() as connection:
            row = connection.execute(
                select(source.c.id)
                .where(
                    source.c.payload["product_id"].as_string() == product_id,
                    source.c.payload["instrument_id"].as_string() == order.instrument_id,
                    source.c.payload["values"]["bid_depth_quote"].as_float().is_not(None),
                    source.c.payload["values"]["ask_depth_quote"].as_float().is_not(None),
                    source.c.created_at.between(oldest, now),
                    or_(
                        source.c.payload["availability_time"].as_string().is_(None),
                        source.c.payload["availability_time"].as_string() <= now,
                    ),
                )
                .order_by(source.c.created_at.desc(), source.c.id.desc())
                .limit(1)
            ).scalar_one_or_none()
        if row is None:
            raise PermissionError("live entry requires a fresh instrument book quote")
        quote = SqlRiskSnapshotStore(self.engine).get(str(row))
        observed_at = quote.get("source_event_time")
        if (
            not observed_at
            or not 0
            <= (
                dt.datetime.fromisoformat(now) - dt.datetime.fromisoformat(observed_at)
            ).total_seconds()
            <= maximum_age
        ):
            raise PermissionError("live entry requires a fresh exchange observation")
        values = quote["values"]
        side = "ask" if order.side is OrderSide.BUY else "bid"
        depth = float(values[f"{side}_depth_quote"])
        price = float(values["close"])
        spread = float(values["spread_bps"])
        if (
            not all(math.isfinite(value) and value >= 0 for value in (depth, price, spread))
            or price <= 0
        ):
            raise PermissionError("live instrument quote is invalid")
        limits = self.risk_configuration["instrument"]
        if spread > float(limits["maximum_slippage_bps"]):
            raise PermissionError("live instrument spread exceeds its risk limit")
        if order.remaining_quantity * price > depth * float(
            limits["maximum_visible_depth_fraction"]
        ):
            raise PermissionError("live order exceeds current side-specific quote depth")

    def authorise_plan(self, plan, *, now, connection):
        """Validate every current book authority before an atomic local allocation."""
        from src.portfolio.account_plans import _capital_proposals, _owner, book_reserve_fraction
        from src.portfolio.books import load_book
        from src.services.exposure_budget import enforce_book_plan

        if execution_engine_identity() != self.loaded_execution_identity:
            raise PermissionError("execution code changed after process startup")
        product_id = plan["product_id"]
        product = self.products[product_id]
        if product["execution_mode"] != "live" or product_id not in self.venues:
            raise PermissionError("account plan has no configured live authority")
        if plan["account_id"] != product["account_id"] or any(
            target["portfolio_id"] != product["portfolio_id"] or now >= target["valid_until"]
            for target in plan["targets"]
        ):
            raise PermissionError("account plan scope or validity changed")
        self._assert_financial_followups_complete(plan["account_id"])
        account_payload, fingerprint = self._validated_account_authority(
            product_id=product_id,
            account_id=plan["account_id"],
            authority_at=now,
            product=product,
        )
        assignments = tuple(
            row
            for row in self.assignments.active_assignments(product_id, at=now)
            if row["execution_mode"] == "live"
        )
        owners = {row["payload"].get("book_id"): _owner(row) for row in assignments}
        if owners != plan["books"]:
            raise PermissionError("account plan no longer contains every current live book")
        for assignment in assignments:
            book = load_book(self.engine, assignment["payload"]["book_id"], connection=connection)
            self._assert_live_book(assignment, book, product, fingerprint, now)
        clean = {
            **account_payload,
            "market": {
                instrument: {"price": price} for instrument, price in plan["prices"].items()
            },
        }
        proposed = _capital_proposals(
            self.engine,
            connection,
            assignments,
            clean,
            now,
            reserve_fraction=book_reserve_fraction(product),
        )
        planned = {
            row["book_id"]: (row["balance_changes"], row.get("btc_capital_baseline"))
            for row in plan["initial_capital"]
        }
        if any(
            planned.get(row["book_id"]) != (row["balance_changes"], row.get("btc_capital_baseline"))
            for row in proposed
        ):
            raise PermissionError("account capital allocation changed since planning")
        enforce_book_plan(
            plan,
            assignments,
            product,
            self.accounts[plan["account_id"]],
            self.risk_configuration,
            account_payload,
        )

    def _assert_live_book(self, assignment, book, product, fingerprint, now):
        if (
            book["execution_mode"] != "live"
            or book["account_id"] != product["account_id"]
            or book["artefact_hash"] != assignment["artefact_hash"]
            or book["strategy_version_id"] != assignment["strategy_version_id"]
        ):
            raise PermissionError("live assignment does not own the planned book")
        artifact = self.artefacts.get(assignment["artefact_hash"])
        product_id = product["product_id"]
        self._assert_artifact_identity(
            artifact, assignment=assignment, product_id=product_id, account_id=product["account_id"]
        )
        allowed = _live_assignment_instrument_ids(
            self.engine, (assignment,), product=product, at=now
        )
        venue = self.venues[product_id]
        for instrument in book["instrument_ids"]:
            if instrument not in allowed or instrument not in venue.instruments:
                raise PermissionError("book instrument is outside current live authority")
            self._current_approval_and_preflight(
                assignment=assignment,
                artifact=artifact,
                product_id=product_id,
                account_id=product["account_id"],
                authority_at=now,
                product=product,
                actual_fingerprint=fingerprint,
                instrument_id=instrument,
            )
        if self.accounts[product["account_id"]]["market"] != "spot":
            supports_stops = getattr(venue.broker, "supports_native_protective_stops", None)
            if not callable(supports_stops) or not supports_stops():
                raise PermissionError("live futures entry requires native protective stops")

    def _authorise_account_order(self, payload, order):
        from src.data.database import account_portfolio_plan, book_event
        from src.domain.portfolios import TargetPosition
        from src.execution.order_planner import plan_orders

        product_id, product, account_id, now = self._order_context(payload, order)
        metadata = order.metadata["target_metadata"]
        with self.engine.connect() as connection:
            plan = connection.execute(
                select(account_portfolio_plan.c.payload).where(
                    account_portfolio_plan.c.id == metadata["account_plan_id"]
                )
            ).scalar_one()
            target = next(
                row for row in plan["targets"] if row["instrument_id"] == order.instrument_id
            )
            if (
                plan["product_id"] != product_id
                or metadata != target["metadata"]
                or not metadata["book_contributions"]
            ):
                raise PermissionError("net order differs from its immutable book contributions")
            if (
                connection.execute(
                    select(book_event.c.id).where(
                        book_event.c.id == "account-plan-applied:" + plan["plan_id"]
                    )
                ).first()
                is None
            ):
                raise PermissionError("account allocation transaction has not completed")
            expected = plan_orders(
                (TargetPosition(**target),),
                current_quantities=plan["account_positions"],
                decided_at=plan["decision_at"],
                prices=plan["prices"],
            )
            if not any(
                order.side == item.side
                and order.reduce_only == item.reduce_only
                and _planned_order_size_matches(order, item)
                and order.portfolio_id == item.portfolio_id
                and order.strategy_contributions == item.strategy_contributions
                for item in expected
            ):
                raise PermissionError(
                    "net order direction or quantity differs from the approved plan"
                )
            self.authorise_plan(plan, now=now, connection=connection)
        account_payload, _ = self._validated_account_authority(
            product_id=product_id,
            account_id=account_id,
            authority_at=now,
            product=product,
        )
        active = {row["id"]: row for row in self.assignments.active_assignments(product_id, at=now)}
        self.positions.reload()
        self.order_manager.reload()
        resources = (
            self._btc_book_resources(metadata["book_authorities"], now)
            if product_id == "btc_accumulation"
            else None
        )
        # Reassess every book, including holdings with no net external contribution.
        for owner in metadata["book_authorities"]:
            self.exposure_guard.enforce(
                product_id=product_id,
                product=product,
                account=self.accounts[account_id],
                assignment=active[owner["assignment_id"]],
                risk_configuration=self.risk_configuration,
                account_payload=account_payload,
                order=order,
                positions=self.positions.all(),
                orders=self.order_manager.all(),
                book_resources=resources,
            )

    def _btc_book_resources(self, owners, at):
        from src.data.database import book_event
        from src.portfolio.books import book_balances

        with self.engine.connect() as connection:
            return {
                owner["book_id"]: {
                    "balances": book_balances(
                        self.engine, owner["book_id"], at=at, connection=connection
                    ),
                    "capital": connection.execute(
                        select(book_event.c.payload).where(
                            book_event.c.id == "live-capital:" + owner["book_id"]
                        )
                    ).scalar_one(),
                }
                for owner in owners
            }

    def _assert_financial_followups_complete(self, account_id: str) -> None:
        from src.data.database import financial_application, financial_outbox

        with self.engine.connect() as connection:
            pending = connection.execute(
                select(financial_outbox.c.id)
                .join(
                    financial_application,
                    financial_application.c.id == financial_outbox.c.id,
                )
                .where(
                    financial_application.c.account_id == account_id,
                    financial_application.c.environment == self.accounts[account_id]["environment"],
                    financial_outbox.c.processed_at.is_(None),
                )
                .limit(1)
            ).first()
        if pending is not None:
            raise PermissionError("committed financial follow-up is incomplete")

    def _order_context(
        self, payload: Mapping[str, Any], order: OrderIntent
    ) -> tuple[str, dict[str, Any], str, str]:
        product_id = str(payload["product_id"])
        product = self.products[product_id]
        account_id = str(product["account_id"])
        authority_at = timestamp(
            str(payload.get("authorisation_at") or order.created_at),
            field="live authorisation time",
            preserve_subseconds=True,
        )
        order_created_at = timestamp(order.created_at, field="order.created_at")
        if authority_at < order_created_at:
            raise PermissionError("live order cannot be authorised before it was created")
        if order.valid_until is None:
            raise PermissionError("live order intent has no expiry")
        if authority_at >= order.valid_until:
            raise PermissionError("live order intent has expired")
        return product_id, product, account_id, authority_at

    def _validated_account_authority(
        self,
        *,
        product_id: str,
        account_id: str,
        authority_at: str,
        product: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], str]:
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
        account_age = self._account_snapshot_age(authority_at, account_observed_at)
        self._assert_complete_account_snapshot(
            account_payload,
            product_id=product_id,
            account_age=account_age,
            maximum_age=int(product.get("account_snapshot_max_age_seconds", 60)),
        )
        if not isinstance(account_payload, Mapping):
            raise PermissionError("live order requires a complete account snapshot")
        expected_fingerprint = str(account_payload.get("account_fingerprint") or "")
        venue = self.venues[product_id]
        actual_fingerprint = str(getattr(venue.broker, "account_fingerprint", ""))
        if not expected_fingerprint or expected_fingerprint != actual_fingerprint:
            raise PermissionError("live order account fingerprint does not match reconciliation")
        return account_payload, actual_fingerprint

    @staticmethod
    def _account_snapshot_age(authority_at: str, observed_at: str | None) -> float | None:
        if observed_at is None:
            return None
        return (
            dt.datetime.fromisoformat(authority_at) - dt.datetime.fromisoformat(observed_at)
        ).total_seconds()

    @staticmethod
    def _assert_complete_account_snapshot(
        account_payload: Any,
        *,
        product_id: str,
        account_age: float | None,
        maximum_age: int,
    ) -> None:
        complete_fields = {
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
            or not complete_fields.issubset(account_payload)
            or account_payload.get("account_state_known") is not True
            or account_payload.get("account_state_authority")
            not in {"authenticated_rest", "authenticated_reconciled"}
            or account_payload.get("unknown_exposure")
            or account_age is None
            or account_age < 0
            or account_age > maximum_age
        ):
            raise PermissionError("live order requires a recent complete account snapshot")

    def _validated_strategy_authority(
        self,
        *,
        payload: Mapping[str, Any],
        order: OrderIntent,
        product_id: str,
        account_id: str,
        authority_at: str,
        product: Mapping[str, Any],
        actual_fingerprint: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        current = self._current_assignment(
            payload, product_id=product_id, order=order, at=authority_at
        )
        assignment = current
        artifact = self.artefacts.get(str(assignment["artefact_hash"]))
        self._assert_artifact_identity(
            artifact,
            assignment=assignment,
            product_id=product_id,
            account_id=account_id,
        )
        self._current_approval_and_preflight(
            assignment=assignment,
            artifact=artifact,
            product_id=product_id,
            account_id=account_id,
            authority_at=authority_at,
            product=product,
            actual_fingerprint=actual_fingerprint,
            instrument_id=order.instrument_id,
        )
        return assignment, artifact

    def _current_assignment(
        self,
        payload: Mapping[str, Any],
        *,
        product_id: str,
        order: OrderIntent,
        at: str,
    ) -> Mapping[str, Any]:
        requested_strategy_id = str(payload.get("strategy_version_id") or "")
        requested_assignment_id = str(payload.get("assignment_id") or "")
        live_assignments = self.assignments.active_assignments(product_id, at=at)
        candidates = tuple(
            item
            for item in live_assignments
            if item["execution_mode"] == "live"
            and order.instrument_id
            in _live_assignment_instrument_ids(
                self.engine,
                (item,),
                product=self.products[product_id],
                at=at,
            )
        )
        if requested_assignment_id:
            candidates = tuple(item for item in candidates if item["id"] == requested_assignment_id)
        if requested_strategy_id:
            candidates = tuple(
                item for item in candidates if item["strategy_version_id"] == requested_strategy_id
            )
        if len(candidates) != 1:
            reason = "no" if not candidates else "ambiguous"
            raise PermissionError(f"{reason} live assignment matches the order instrument")
        current = candidates[0]
        requested_artefact_hash = str(payload.get("artefact_hash") or "")
        if requested_artefact_hash and requested_artefact_hash != current["artefact_hash"]:
            raise PermissionError("live order artefact identity does not match assignment")
        return current

    @staticmethod
    def _assert_artifact_identity(
        artifact: Mapping[str, Any],
        *,
        assignment: Mapping[str, Any],
        product_id: str,
        account_id: str,
    ) -> None:
        declared_hash = artifact.get("artefact_hash")
        content = dict(artifact)
        content.pop("artefact_hash", None)
        if declared_hash != assignment["artefact_hash"] or canonical_hash(content) != declared_hash:
            raise PermissionError("canonical live artefact content hash does not match assignment")
        if str(artifact.get("strategy_version_id") or "") != assignment["strategy_version_id"]:
            raise PermissionError(
                "canonical live artefact strategy identity does not match assignment"
            )
        if artifact.get("product_id") != product_id or artifact.get("account_id") != account_id:
            raise PermissionError("canonical live artefact account or product does not match")
        if artifact.get("portfolio_id") != assignment["portfolio_id"]:
            raise PermissionError("canonical live artefact portfolio does not match assignment")

    def _current_approval_and_preflight(
        self,
        *,
        assignment: Mapping[str, Any],
        artifact: Mapping[str, Any],
        product_id: str,
        account_id: str,
        authority_at: str,
        product: Mapping[str, Any],
        actual_fingerprint: str,
        instrument_id: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        strategy_version_id = str(assignment["strategy_version_id"])
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
        self._assert_record_bindings(approval, preflight, artifact)
        if float(assignment["capital_limit"]) > min(
            float(approval["capital_cap"]), float(preflight["capital_cap"])
        ):
            raise PermissionError("active assignment exceeds approved/preflight capital cap")
        self._assert_current_authority_payloads(
            assignment=assignment,
            artifact=artifact,
            approval=approval,
            preflight=preflight,
            account_id=account_id,
            product=product,
            actual_fingerprint=actual_fingerprint,
            instrument_id=instrument_id,
        )
        return approval, preflight

    @staticmethod
    def _assert_record_bindings(
        approval: Mapping[str, Any],
        preflight: Mapping[str, Any],
        artifact: Mapping[str, Any],
    ) -> None:
        for record, label in ((approval, "approval"), (preflight, "preflight")):
            if any(
                record[field] != artifact.get(field)
                for field in ("artefact_hash", "source_commit_hash", "engine_version")
            ):
                raise PermissionError(f"canonical {label} is not bound to the live artefact")

    def _assert_current_authority_payloads(
        self,
        *,
        assignment: Mapping[str, Any],
        artifact: Mapping[str, Any],
        approval: Mapping[str, Any],
        preflight: Mapping[str, Any],
        account_id: str,
        product: Mapping[str, Any],
        actual_fingerprint: str,
        instrument_id: str,
    ) -> None:
        authority_instrument = self.instruments.get(instrument_id)
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
        self._assert_preflight_payload(
            preflight,
            assignment=assignment,
            account_id=account_id,
            instrument_id=instrument_id,
            actual_fingerprint=actual_fingerprint,
            expected_engine_identity=expected_engine_identity,
            expected_configuration_hash=expected_configuration_hash,
        )
        self._assert_approval_payload(
            approval,
            assignment=assignment,
            preflight=preflight,
            account_id=account_id,
            instrument_id=instrument_id,
            actual_fingerprint=actual_fingerprint,
            expected_engine_identity=expected_engine_identity,
            expected_configuration_hash=expected_configuration_hash,
        )

    def _assert_preflight_payload(
        self,
        preflight: Mapping[str, Any],
        *,
        assignment: Mapping[str, Any],
        account_id: str,
        instrument_id: str,
        actual_fingerprint: str,
        expected_engine_identity: str,
        expected_configuration_hash: str,
    ) -> None:
        payload = preflight.get("payload")
        account = self.accounts[account_id]
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "platform.production-preflight/v1"
            or payload.get("environment") != account["environment"]
            or payload.get("account_fingerprint") != actual_fingerprint
            or payload.get("execution_engine_identity") != expected_engine_identity
            or payload.get("instrument_id") != instrument_id
            or payload.get("sleeve_id") != assignment["sleeve_id"]
            or payload.get("configuration_hash") != expected_configuration_hash
            or preflight.get("content_hash") != canonical_hash(dict(payload))
        ):
            raise PermissionError("canonical preflight does not match current live authority")

    def _assert_approval_payload(
        self,
        approval: Mapping[str, Any],
        *,
        assignment: Mapping[str, Any],
        preflight: Mapping[str, Any],
        account_id: str,
        instrument_id: str,
        actual_fingerprint: str,
        expected_engine_identity: str,
        expected_configuration_hash: str,
    ) -> None:
        payload = approval.get("payload")
        account = self.accounts[account_id]
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "platform.strategy-approval/v1"
            or payload.get("preflight_id") != preflight.get("id")
            or payload.get("environment") != account["environment"]
            or payload.get("account_fingerprint") != actual_fingerprint
            or payload.get("execution_engine_identity") != expected_engine_identity
            or payload.get("instrument_id") != instrument_id
            or payload.get("sleeve_id") != assignment["sleeve_id"]
            or payload.get("configuration_hash") != expected_configuration_hash
        ):
            raise PermissionError("canonical approval does not match current live authority")

    def _assert_order_strategy_scope(
        self,
        *,
        order: OrderIntent,
        product_id: str,
        artifact: Mapping[str, Any],
        assignment: Mapping[str, Any],
    ) -> None:
        strategy_version_id = str(assignment["strategy_version_id"])
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
        if instrument.instrument_id not in set(artifact.get("supported_instruments", [])):
            raise PermissionError("live order instrument is not bound to the canonical artefact")


def _planned_order_size_matches(order, planned):
    from src.execution.order_manager import entry_notional

    reference = order.metadata.get("entry_reference")
    if reference is None:
        return order.quantity <= planned.quantity
    budget = entry_notional(planned)
    return (
        not order.reduce_only
        and reference.get("notional") == budget
        and order.quantity * float(reference["price"]) <= budget + max(1e-9, budget * 1e-12)
    )


def _domain_fill(
    order: OrderIntent,
    recovered: BrokerFill,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Fill:
    if not _instrument_matches_symbol(order.instrument_id, recovered.symbol):
        raise RuntimeError("recovered fill symbol does not match the local order")
    if recovered.side.value != order.side.value:
        raise RuntimeError("recovered fill side does not match the local order")
    return Fill(
        fill_id=canonical_hash(
            {
                "venue": "binance",
                "instrument_id": order.instrument_id,
                "trade_id": recovered.trade_id,
            }
        ),
        order_id=order.order_id,
        instrument_id=order.instrument_id,
        side=OrderSide(recovered.side.value),
        quantity=recovered.quantity,
        price=recovered.price,
        fee=recovered.fee,
        occurred_at=_epoch_time(recovered.occurred_at),
        fee_asset=recovered.fee_asset,
        metadata={
            "rest_recovery": True,
            "exchange_order_id": recovered.exchange_order_id,
            "trade_id": recovered.trade_id,
            **dict(metadata or {}),
        },
    )


def _instrument_matches_symbol(instrument_id: str, symbol: str) -> bool:
    canonical = str(instrument_id).upper()
    compact = str(symbol).replace("/", "").replace(":", "").upper()
    return compact in canonical.replace(":", "") or canonical.endswith(f":{str(symbol).upper()}")


def _matches_existing_recovered_fill(existing: tuple[Fill, ...], recovered: BrokerFill) -> bool:
    return any(str(item.metadata.get("trade_id") or "") == recovered.trade_id for item in existing)


def _validate_recovered_fills(
    *,
    order: OrderIntent,
    fills: tuple[BrokerFill, ...],
    exchange_order_id: str,
    client_order_id: str,
) -> None:
    for recovered in fills:
        _validate_recovered_fill(
            order=order,
            recovered=recovered,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
        )


def _validate_exchange_recovery_state(state: Any, *, target: str) -> None:
    status = str(state.status).casefold()
    if status not in {
        "open",
        "new",
        "accepted",
        "partially_filled",
        "closed",
        "filled",
        "canceled",
        "cancelled",
        "rejected",
        "expired",
    }:
        raise RuntimeError(f"exchange order {target} has an unsupported state: {state.status}")
    if not math.isfinite(state.filled_quantity) or state.filled_quantity < 0:
        raise RuntimeError("exchange order state has an invalid filled quantity")
    if state.average_price is not None and (
        not math.isfinite(state.average_price) or state.average_price <= 0
    ):
        raise RuntimeError("exchange order state has an invalid average price")


def _validate_recovered_fill(
    *,
    order: OrderIntent,
    recovered: BrokerFill,
    exchange_order_id: str,
    client_order_id: str,
) -> None:
    if not math.isfinite(recovered.quantity) or recovered.quantity <= 0:
        raise RuntimeError("REST recovered fill quantity is invalid")
    if not math.isfinite(recovered.price) or recovered.price <= 0:
        raise RuntimeError("REST recovered fill price is invalid")
    if not math.isfinite(recovered.fee) or recovered.fee < 0:
        raise RuntimeError("REST recovered fill fee is invalid")
    if recovered.exchange_order_id != exchange_order_id:
        raise RuntimeError("REST recovered fill exchange order identity changed")
    if client_order_id and recovered.client_order_id not in {"", client_order_id}:
        raise RuntimeError("REST recovered fill client order identity changed")
    if not _instrument_matches_symbol(order.instrument_id, recovered.symbol):
        raise RuntimeError("REST recovered fill symbol does not match the local order")
    if recovered.side.value != order.side.value:
        raise RuntimeError("REST recovered fill side does not match the local order")


def _validate_recovered_quantities(
    order: OrderIntent,
    state: Any,
    recovered: tuple[BrokerFill, ...],
    existing: tuple[Fill, ...],
) -> None:
    existing_quantity = sum(item.quantity for item in existing)
    recovered_quantity = sum(
        item.quantity for item in recovered if not _matches_existing_recovered_fill(existing, item)
    )
    total = existing_quantity + recovered_quantity
    tolerance = max(order.quantity * 1e-9, 1e-12)
    if total > order.quantity + tolerance:
        raise RuntimeError("recovered fills exceed the local order quantity")
    if abs(total - state.filled_quantity) > tolerance:
        raise RuntimeError(
            "exchange order filled quantity does not match exact REST trade quantities"
        )
    if (
        str(state.status).casefold() in {"closed", "filled"}
        and abs(total - order.quantity) > tolerance
    ):
        raise RuntimeError("closed exchange order is not fully filled")
    if state.filled_quantity > 1e-12 and not recovered and not existing:
        raise RuntimeError("exchange reports fills but REST trade history is unavailable")


def _epoch_time(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("recovered exchange timestamp is invalid")
    return dt.datetime.fromtimestamp(value, dt.UTC).replace(microsecond=0).isoformat()


def _records(
    payload: Mapping[str, Any], collection: str, identity: str
) -> dict[str, dict[str, Any]]:
    rows = payload.get(collection)
    if not isinstance(rows, list):
        raise ValueError(f"{collection} must be a list")
    return {str(row[identity]): dict(row) for row in rows}


def _live_assignment_instrument_ids(
    engine,
    assignments: tuple[Mapping[str, Any], ...],
    *,
    product: Mapping[str, Any],
    at: str,
) -> set[str]:
    direct = {str(row["instrument_id"]) for row in assignments if row.get("instrument_id")}
    universe_ids = {str(row["universe_id"]) for row in assignments if row.get("universe_id")}
    configured_universe = str(product.get("universe_id") or "")
    if any(value.startswith("product:") for value in universe_ids) and configured_universe:
        universe_ids = {
            configured_universe if value.startswith("product:") else value for value in universe_ids
        }
    if not universe_ids and product.get("live_instrument_scope") == "current_eligible_universe":
        universe_ids.add(configured_universe)
    for universe_id in tuple(value for value in universe_ids if value):
        snapshot_id = _latest_universe_snapshot_id(engine, universe_id, at=at)
        if snapshot_id is None:
            continue
        with engine.connect() as connection:
            direct.update(
                str(value)
                for value in connection.execute(
                    select(universe_member.c.instrument_id).where(
                        universe_member.c.snapshot_id == snapshot_id,
                        universe_member.c.eligible.is_(True),
                    )
                ).scalars()
            )
    return direct


def _latest_universe_snapshot_id(engine, universe_id: str, *, at: str) -> str | None:
    with engine.connect() as connection:
        value = connection.execute(
            select(universe_snapshot.c.id)
            .where(
                universe_snapshot.c.universe_id == universe_id,
                universe_snapshot.c.observed_at <= at,
            )
            .order_by(universe_snapshot.c.observed_at.desc(), universe_snapshot.c.id.desc())
            .limit(1)
        ).scalar_one_or_none()
    return None if value is None else str(value)


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


def _instrument_asset(instrument_id: str, asset: str) -> str | None:
    return instrument_asset(instrument_id, asset)


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
