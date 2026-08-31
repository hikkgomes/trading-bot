"""Authenticated exchange and explicit paper-account reconciliation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import insert, select

from src.data.database import (
    account_snapshot,
    balance_snapshot,
    instrument,
    reconciliation_event,
    universe_member,
    universe_snapshot,
)
from src.domain._codec import canonical_hash, finite, json_value, timestamp
from src.execution.ccxt_broker import CcxtBroker
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.execution.stops import SqlStopStore, StopManager
from src.services.live_execution import _exchange_config


class AccountAuthorityError(RuntimeError):
    """Authenticated account state is missing or cannot be trusted."""


def _append(connection, table, values: Mapping[str, Any]) -> None:
    existing = connection.execute(select(table.c.payload).where(table.c.id == values["id"])).first()
    if existing is not None:
        if existing[0] != values["payload"]:
            raise AccountAuthorityError(
                f"account reconciliation identity collision: {values['id']}"
            )
        return
    connection.execute(insert(table).values(**dict(values)))


class AccountReconciliationService:
    def __init__(
        self,
        *,
        engine,
        products: Mapping[str, Mapping[str, Any]],
        accounts: Mapping[str, Mapping[str, Any]],
        broker_factory: Callable[[Mapping[str, Any], str], Any] | None = None,
    ) -> None:
        self.engine = engine
        self.products = {str(key): dict(value) for key, value in products.items()}
        self.accounts = {str(key): dict(value) for key, value in accounts.items()}
        self.broker_factory = broker_factory or self._default_broker

    def reconcile_once(self, *, now: str) -> dict[str, Any]:
        now = timestamp(now, field="now")
        results = []
        for product_id, product in sorted(self.products.items()):
            account_id = str(product["account_id"])
            account = self.accounts[account_id]
            if product.get("execution_mode") == "live":
                payload = self._authenticated(account, product, observed_at=now)
                source = "authenticated_rest"
            else:
                payload = self._paper(account, product)
                source = "paper_config"
            payload = {
                **payload,
                "account_id": account_id,
                "product_id": product_id,
                "observed_at": now,
                "account_state_authority": source,
                "account_state_known": True,
            }
            snapshot_id = self._persist(
                account_id=account_id,
                payload=payload,
                source=source,
                observed_at=now,
            )
            results.append(
                {
                    "product_id": product_id,
                    "account_id": account_id,
                    "snapshot_id": snapshot_id,
                    "source": source,
                    "unknown_exposure": dict(payload["unknown_exposure"]),
                    "positions": dict(payload["positions"]),
                    "regular_orders": list(payload["regular_orders"]),
                    "conditional_orders": list(payload["conditional_orders"]),
                    "account_fingerprint": str(payload["account_fingerprint"]),
                }
            )
        return {
            "reason_code": "account_reconciliation_completed",
            "accounts": results,
        }

    def _authenticated(
        self,
        account: Mapping[str, Any],
        product: Mapping[str, Any],
        *,
        observed_at: str,
    ) -> dict[str, Any]:
        api_key_name = str(account.get("api_key_env") or "")
        api_secret_name = str(account.get("api_secret_env") or "")
        import os

        if (
            not os.environ.get(api_key_name, "").strip()
            or not os.environ.get(api_secret_name, "").strip()
        ):
            raise AccountAuthorityError(
                f"live account {account['account_id']} has no authenticated credentials"
            )
        market = "spot" if account.get("market") == "spot" else "futures"
        broker = self.broker_factory(account, market)
        expected_symbols = self._expected_symbols(
            product_id=str(product["product_id"]),
            product=product,
            account=account,
            observed_at=observed_at,
        )
        reader = getattr(broker, "account_snapshot", None)
        if not callable(reader):
            raise AccountAuthorityError("live broker has no authenticated account_snapshot reader")
        payload = reader(expected_symbols=expected_symbols)
        if not isinstance(payload, Mapping):
            raise AccountAuthorityError("authenticated account snapshot is not an object")
        required = {
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
            "account_state_known",
            "account_state_authority",
            "account_fingerprint",
        }
        if not required.issubset(payload) or payload.get("account_state_known") is not True:
            raise AccountAuthorityError("authenticated account snapshot is incomplete")
        if payload.get("account_state_authority") != "authenticated_rest":
            raise AccountAuthorityError("account snapshot is not authenticated REST authority")
        expected_fingerprint = str(getattr(broker, "account_fingerprint", ""))
        actual_fingerprint = str(payload.get("account_fingerprint") or "")
        if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
            raise AccountAuthorityError(
                "authenticated account snapshot fingerprint does not match the configured account"
            )
        reconciled = self._reconcile_platform_exposure(payload=dict(payload), product=product)
        return json_value(reconciled, field="authenticated account snapshot")

    def _expected_symbols(
        self,
        *,
        product_id: str,
        product: Mapping[str, Any],
        account: Mapping[str, Any],
        observed_at: str | None,
    ) -> tuple[str, ...]:
        """Resolve the exact exchange scope owned by the current assignment."""

        configured = product.get("exchange_symbols")
        if isinstance(configured, list | tuple):
            symbols = tuple(sorted({str(value).strip().upper() for value in configured if str(value).strip()}))
            if symbols:
                return symbols
        direct = str(product.get("exchange_symbol") or "").strip().upper()
        if direct:
            return (direct,)
        from src.research.canonical import SqlActiveStrategyAssignmentRepository

        assignments = SqlActiveStrategyAssignmentRepository(self.engine).active_assignments(
            product_id,
            at=observed_at,
        )
        instrument_ids = {
            str(row["instrument_id"])
            for row in assignments
            if row.get("instrument_id")
        }
        universe_ids = {
            str(row["universe_id"])
            for row in assignments
            if row.get("universe_id")
        }
        if universe_ids:
            with self.engine.connect() as connection:
                latest_snapshots = connection.execute(
                    select(universe_snapshot.c.id)
                    .where(
                        universe_snapshot.c.universe_id.in_(universe_ids),
                        *([universe_snapshot.c.observed_at <= observed_at] if observed_at else []),
                    )
                    .order_by(
                        universe_snapshot.c.universe_id,
                        universe_snapshot.c.observed_at.desc(),
                        universe_snapshot.c.id.desc(),
                    )
                ).scalars()
                snapshot_ids = tuple(dict.fromkeys(str(value) for value in latest_snapshots))
                if snapshot_ids:
                    instrument_ids.update(
                        str(value)
                        for value in connection.execute(
                            select(universe_member.c.instrument_id).where(
                                universe_member.c.snapshot_id.in_(snapshot_ids),
                                universe_member.c.eligible.is_(True),
                            )
                        ).scalars()
                    )
        if not instrument_ids:
            if account.get("market") != "spot":
                raise AccountAuthorityError(
                    f"live futures product {product_id} has no active instrument scope"
                )
            return ("BTCUSDT",)
        with self.engine.connect() as connection:
            symbols = connection.execute(
                select(instrument.c.exchange_symbol).where(instrument.c.id.in_(instrument_ids))
            ).scalars()
        result = tuple(sorted({str(value).upper() for value in symbols if str(value).strip()}))
        if not result:
            raise AccountAuthorityError(
                f"live product {product_id} has no persisted symbols for its active scope"
            )
        return result

    def _reconcile_platform_exposure(
        self,
        *,
        payload: dict[str, Any],
        product: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reject exchange exposure that is not present in durable platform state."""

        unknown = dict(payload["unknown_exposure"])
        portfolio_id = str(product["portfolio_id"])
        raw_positions = payload["positions"]
        if not isinstance(raw_positions, Mapping):
            raise AccountAuthorityError("authenticated account positions are not an object")
        exchange_positions = {
            str(instrument_id): finite(
                float(quantity),
                field=f"authenticated position {instrument_id}",
            )
            for instrument_id, quantity in raw_positions.items()
        }
        platform_positions = PositionManager(SqlPositionStore(self.engine)).current_quantities(
            portfolio_id
        )
        for instrument_id in sorted(set(exchange_positions) | set(platform_positions)):
            exchange_quantity = exchange_positions.get(instrument_id, 0.0)
            platform_quantity = float(platform_positions.get(instrument_id, 0.0))
            if abs(exchange_quantity - platform_quantity) > 1e-12:
                unknown[f"position:{instrument_id}"] = {
                    "exchange_quantity": exchange_quantity,
                    "platform_quantity": platform_quantity,
                }

        orders = OrderManager(SqlOrderStore(self.engine)).all()
        active_orders = tuple(
            order
            for order in orders
            if order.portfolio_id == portfolio_id and not order.is_terminal
        )
        known_exchange_ids = {
            str(order.metadata.get("exchange_order_id"))
            for order in active_orders
            if order.metadata.get("exchange_order_id")
        }
        known_client_ids = {
            str(order.metadata.get("client_order_id"))
            for order in active_orders
            if order.metadata.get("client_order_id")
        }
        known_exchange_ids.update(
            str(stop.native_order_id)
            for stop in StopManager(SqlStopStore(self.engine)).active()
            if stop.portfolio_id == portfolio_id and stop.native_order_id
        )
        for order_type in ("regular_orders", "conditional_orders"):
            raw_orders = payload[order_type]
            if not isinstance(raw_orders, list):
                raise AccountAuthorityError(f"authenticated account {order_type} are not a list")
            for row in raw_orders:
                if not isinstance(row, Mapping):
                    raise AccountAuthorityError(
                        f"authenticated account {order_type} contain a non-object"
                    )
                exchange_order_id = str(row.get("order_id") or "")
                client_order_id = str(row.get("client_id") or "")
                if exchange_order_id in known_exchange_ids or client_order_id in known_client_ids:
                    continue
                symbol = str(row.get("symbol") or "unknown")
                identity = exchange_order_id or client_order_id or "unknown"
                unknown[f"external_order:{symbol}:{identity}"] = str(row.get("status") or "open")
        return {**payload, "positions": exchange_positions, "unknown_exposure": unknown}

    @staticmethod
    def _paper(account: Mapping[str, Any], product: Mapping[str, Any]) -> dict[str, Any]:
        balances = account.get("paper_starting_balances")
        if not isinstance(balances, Mapping) or not balances:
            raise AccountAuthorityError(
                f"paper account {account['account_id']} has no explicit starting balances"
            )
        positions = account.get("paper_starting_positions", {})
        if not isinstance(positions, Mapping):
            raise AccountAuthorityError("paper_starting_positions must be an object")
        try:
            clean_balances = {
                str(key): finite(
                    float(value),
                    field=f"paper_starting_balances[{key}]",
                    minimum=0.0,
                )
                for key, value in balances.items()
            }
            clean_positions = {
                str(key): finite(float(value), field=f"paper_starting_positions[{key}]")
                for key, value in positions.items()
            }
        except (TypeError, ValueError) as exc:
            raise AccountAuthorityError(f"paper account state is invalid: {exc}") from exc
        return {
            "balances": clean_balances,
            "free_balances": dict(clean_balances),
            "positions": clean_positions,
            "regular_orders": [],
            "conditional_orders": [],
            "used_margin": 0.0,
            "maintenance_margin": 0.0,
            "used_margin_fraction": 0.0,
            "liquidation_buffer_fraction": 1.0,
            "account_mode": str(account.get("margin_mode", "cash")),
            "unknown_exposure": {},
            "account_fingerprint": canonical_hash(
                {
                    "account_id": account["account_id"],
                    "product_id": product["product_id"],
                    "source": "paper_config",
                }
            ),
        }

    @staticmethod
    def _default_broker(account: Mapping[str, Any], market: str) -> CcxtBroker:
        config = _exchange_config(account, market=market)
        return CcxtBroker(config)

    def _persist(
        self,
        *,
        account_id: str,
        payload: Mapping[str, Any],
        source: str,
        observed_at: str,
    ) -> str:
        clean = json_value(dict(payload), field="account snapshot")
        content_hash = canonical_hash(clean)
        snapshot_id = canonical_hash(
            {"account_id": account_id, "observed_at": observed_at, "content_hash": content_hash}
        )
        balance_id = canonical_hash(
            {"account_snapshot_id": snapshot_id, "observed_at": observed_at}
        )
        event_id = canonical_hash({"kind": "account_snapshot", "snapshot_id": snapshot_id})
        with self.engine.begin() as connection:
            _append(
                connection,
                account_snapshot,
                {
                    "id": snapshot_id,
                    "account_id": account_id,
                    "observed_at": observed_at,
                    "source": source,
                    "content_hash": content_hash,
                    "payload": clean,
                },
            )
            _append(
                connection,
                balance_snapshot,
                {
                    "id": balance_id,
                    "created_at": observed_at,
                    "payload": clean,
                },
            )
            _append(
                connection,
                reconciliation_event,
                {
                    "id": event_id,
                    "created_at": observed_at,
                    "payload": {
                        "kind": "account_snapshot",
                        "account_id": account_id,
                        "snapshot_id": snapshot_id,
                        "source": source,
                        "observed_at": observed_at,
                        "unknown_exposure": dict(clean["unknown_exposure"]),
                    },
                },
            )
        return snapshot_id
