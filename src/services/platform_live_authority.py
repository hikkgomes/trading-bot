"""Manual, exact-hash authority setup for a canonical live assignment."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.data.database import (
    PlatformDatabase,
    account_snapshot,
    instrument,
    production_preflight,
    strategy_approval,
)
from src.domain._codec import timestamp
from src.execution.ccxt_broker import CcxtBroker
from src.research.canonical import (
    CanonicalEvidenceError,
    SqlActiveStrategyAssignmentRepository,
    SqlApprovalRepository,
    SqlPreflightRepository,
    SqlStrategyArtefactRepository,
    latest_accepted_forward_summary,
    preflight_is_fresh,
)
from src.services.account_reconciliation import AccountReconciliationService
from src.services.config import load_platform_config, load_split_configuration
from src.services.live_execution import (
    _exchange_config,
    execution_engine_identity,
    live_authority_configuration_hash,
)
from src.services.runtime import utc_now


class PlatformLiveAuthorityError(RuntimeError):
    """Manual live authority is incomplete or does not match current state."""


class PlatformLiveAuthority:
    def __init__(
        self,
        *,
        engine,
        configuration: Mapping[str, Mapping[str, Any]],
        broker_factory: Callable[[Mapping[str, Any], str], Any] | None = None,
    ) -> None:
        self.engine = engine
        self.configuration = configuration
        self.products = _records(configuration["products"], "products", "product_id")
        self.accounts = _records(configuration["accounts"], "accounts", "account_id")
        self.policies = _records(configuration["promotion"], "policies", "policy_id")
        self.risk_configuration = dict(configuration["risk"])
        self.broker_factory = broker_factory
        self.artefacts = SqlStrategyArtefactRepository(engine)
        self.approvals = SqlApprovalRepository(engine)
        self.preflights = SqlPreflightRepository(engine)
        self.assignments = SqlActiveStrategyAssignmentRepository(engine)

    def inspect(
        self, *, product_id: str, artefact_hash: str, instrument_id: str, sleeve_id: str
    ) -> dict[str, Any]:
        product, account, artefact, instrument_payload = self._selection(
            product_id=product_id,
            artefact_hash=artefact_hash,
            instrument_id=instrument_id,
            sleeve_id=sleeve_id,
        )
        manifest = self._manifest(
            product=product,
            account=account,
            artefact=artefact,
            instrument_payload=instrument_payload,
            sleeve_id=sleeve_id,
        )
        return {
            "schema": "platform.live-authority-review/v1",
            "product_id": product_id,
            "account_id": str(account["account_id"]),
            "environment": str(account["environment"]),
            "artefact_hash": artefact_hash,
            "strategy_version_id": str(artefact["strategy_version_id"]),
            "instrument_id": instrument_id,
            "sleeve_id": sleeve_id,
            "execution_engine_identity": execution_engine_identity(),
            "maximum_canary_capital": self._maximum_canary_capital(product),
            "configuration_hash": live_authority_configuration_hash(
                product=product,
                account=account,
                instrument_payload=instrument_payload,
                artefact=artefact,
                sleeve_id=sleeve_id,
                promotion_policy=self.policies[str(product["promotion_policy_id"])],
                risk_configuration=self.risk_configuration,
            ),
            "manifest": manifest,
            "artefact": artefact,
        }

    def preflight(
        self,
        *,
        product_id: str,
        artefact_hash: str,
        instrument_id: str,
        sleeve_id: str,
        capital_cap: float,
        checked_at: str,
    ) -> dict[str, Any]:
        checked_at = timestamp(checked_at, field="checked_at")
        capital_cap = self._capital_cap(product_id, capital_cap)
        product, account, artefact, instrument_payload = self._selection(
            product_id=product_id,
            artefact_hash=artefact_hash,
            instrument_id=instrument_id,
            sleeve_id=sleeve_id,
        )
        broker = self._broker(account)
        market = _market(account)
        product_for_snapshot = {
            **product,
            "exchange_symbol": str(instrument_payload["exchange_symbol"]),
        }
        reconciliation = AccountReconciliationService(
            engine=self.engine,
            products={product_id: product_for_snapshot},
            accounts={str(account["account_id"]): account},
            broker_factory=lambda _account, _market: broker,
        ).reconcile_once(now=checked_at)
        account_result = reconciliation["accounts"][0]
        snapshot_id = str(account_result["snapshot_id"])
        with self.engine.connect() as connection:
            snapshot = connection.execute(
                select(account_snapshot.c.payload).where(account_snapshot.c.id == snapshot_id)
            ).scalar_one()
        if not isinstance(snapshot, Mapping):
            raise PlatformLiveAuthorityError("authenticated account snapshot is missing")
        expected_fingerprint = str(getattr(broker, "account_fingerprint", ""))
        price = float(broker.get_price(str(instrument_payload["exchange_symbol"])))
        checks = {
            "product_configured_live": product.get("execution_mode") == "live",
            "authenticated_account": snapshot.get("account_state_known") is True
            and snapshot.get("account_state_authority") == "authenticated_rest",
            "account_fingerprint": bool(expected_fingerprint)
            and snapshot.get("account_fingerprint") == expected_fingerprint,
            "account_mode": market != "futures" or snapshot.get("account_mode") == "one_way",
            "unknown_exposure_empty": snapshot.get("unknown_exposure") == {},
            "regular_orders_empty": snapshot.get("regular_orders") == [],
            "conditional_orders_empty": snapshot.get("conditional_orders") == [],
            "futures_positions_flat": market != "futures" or snapshot.get("positions") == {},
            "positive_quote_balance": float(
                dict(snapshot.get("balances", {})).get(str(account["quote_assets"][0]), 0.0)
            )
            > 0,
            "positive_price": math.isfinite(price) and price > 0,
            "native_futures_stops": market != "futures"
            or bool(getattr(broker, "supports_native_protective_stops", lambda: False)()),
        }
        failed = sorted(name for name, accepted in checks.items() if not accepted)
        if failed:
            raise PlatformLiveAuthorityError("preflight checks failed: " + ", ".join(failed))
        payload = {
            "schema": "platform.production-preflight/v1",
            "strategy_version_id": str(artefact["strategy_version_id"]),
            "product_id": product_id,
            "account_id": str(account["account_id"]),
            "artefact_hash": artefact_hash,
            "source_commit_hash": str(artefact["source_commit_hash"]),
            "engine_version": str(artefact["engine_version"]),
            "execution_engine_identity": execution_engine_identity(),
            "environment": str(account["environment"]),
            "account_fingerprint": expected_fingerprint,
            "instrument_id": instrument_id,
            "sleeve_id": sleeve_id,
            "account_snapshot_id": snapshot_id,
            "configuration_hash": live_authority_configuration_hash(
                product=product,
                account=account,
                instrument_payload=instrument_payload,
                artefact=artefact,
                sleeve_id=sleeve_id,
                promotion_policy=self.policies[str(product["promotion_policy_id"])],
                risk_configuration=self.risk_configuration,
            ),
            "capital_cap": capital_cap,
            "checked_at": checked_at,
            "accepted": True,
            "checks": checks,
        }
        preflight_id = self.preflights.append(payload)
        return {**payload, "preflight_id": preflight_id}

    def approve(
        self,
        *,
        product_id: str,
        artefact_hash: str,
        instrument_id: str,
        sleeve_id: str,
        expected_preflight_id: str,
        capital_cap: float,
        approved_by: str,
        approved_at: str,
        confirm: bool,
    ) -> dict[str, Any]:
        if not confirm:
            raise PlatformLiveAuthorityError("explicit operator confirmation is required")
        approved_at = timestamp(approved_at, field="approved_at")
        capital_cap = self._capital_cap(product_id, capital_cap)
        product, account, artefact, instrument_payload = self._selection(
            product_id=product_id,
            artefact_hash=artefact_hash,
            instrument_id=instrument_id,
            sleeve_id=sleeve_id,
        )
        preflight = self._current_preflight(
            product=product,
            account=account,
            artefact=artefact,
            instrument_payload=instrument_payload,
            sleeve_id=sleeve_id,
            expected_id=expected_preflight_id,
            at=approved_at,
        )
        if capital_cap > float(preflight["capital_cap"]):
            raise PlatformLiveAuthorityError("approval capital exceeds the preflight cap")
        forward = latest_accepted_forward_summary(
            self.engine,
            strategy_version_id=str(artefact["strategy_version_id"]),
            product_id=product_id,
            artefact_hash=artefact_hash,
            at=approved_at,
        )
        if forward is None:
            raise PlatformLiveAuthorityError(
                "approval requires the latest accepted forward summary"
            )
        approval_id = self.approvals.append(
            strategy_version_id=str(artefact["strategy_version_id"]),
            product_id=product_id,
            account_id=str(account["account_id"]),
            artefact_hash=artefact_hash,
            source_commit_hash=str(artefact["source_commit_hash"]),
            engine_version=str(artefact["engine_version"]),
            capital_cap=capital_cap,
            actor=approved_by,
            approved_at=approved_at,
            payload={
                "schema": "platform.strategy-approval/v1",
                "preflight_id": expected_preflight_id,
                "instrument_id": instrument_id,
                "sleeve_id": sleeve_id,
                "environment": str(account["environment"]),
                "account_fingerprint": str(preflight["payload"]["account_fingerprint"]),
                "execution_engine_identity": execution_engine_identity(),
                "configuration_hash": str(preflight["payload"]["configuration_hash"]),
                "forward_summary_id": str(forward["summary"]["id"]),
                "forward_decision_id": str(forward["decision"]["id"]),
            },
        )
        return {
            "schema": "platform.strategy-approval-result/v1",
            "approval_id": approval_id,
            "artefact_hash": artefact_hash,
            "preflight_id": expected_preflight_id,
            "product_id": product_id,
            "account_id": str(account["account_id"]),
            "instrument_id": instrument_id,
            "sleeve_id": sleeve_id,
            "capital_cap": capital_cap,
            "approved_by": approved_by.strip(),
            "approved_at": approved_at,
        }

    def assign(
        self,
        *,
        product_id: str,
        artefact_hash: str,
        instrument_id: str,
        sleeve_id: str,
        expected_preflight_id: str,
        expected_approval_id: str,
        capital_limit: float,
        risk_budget: float,
        assigned_by: str,
        assigned_at: str,
        confirm: bool,
    ) -> dict[str, Any]:
        if not confirm:
            raise PlatformLiveAuthorityError("explicit operator confirmation is required")
        assigned_at = timestamp(assigned_at, field="assigned_at")
        capital_limit = self._capital_cap(product_id, capital_limit)
        if not math.isfinite(float(risk_budget)) or not 0 <= float(risk_budget) <= capital_limit:
            raise PlatformLiveAuthorityError("risk budget must be within the capital limit")
        product, account, artefact, instrument_payload = self._selection(
            product_id=product_id,
            artefact_hash=artefact_hash,
            instrument_id=instrument_id,
            sleeve_id=sleeve_id,
        )
        preflight = self._current_preflight(
            product=product,
            account=account,
            artefact=artefact,
            instrument_payload=instrument_payload,
            sleeve_id=sleeve_id,
            expected_id=expected_preflight_id,
            at=assigned_at,
        )
        approval = self._current_approval(
            product=product,
            account=account,
            artefact=artefact,
            instrument_payload=instrument_payload,
            sleeve_id=sleeve_id,
            expected_id=expected_approval_id,
            expected_preflight_id=expected_preflight_id,
            at=assigned_at,
        )
        self._assert_forward_ready(
            product_id=product_id,
            strategy_version_id=str(artefact["strategy_version_id"]),
            artefact_hash=artefact_hash,
            at=assigned_at,
        )
        if capital_limit > min(float(preflight["capital_cap"]), float(approval["capital_cap"])):
            raise PlatformLiveAuthorityError("assignment capital exceeds approved authority")
        if not isinstance(assigned_by, str) or not assigned_by.strip():
            raise PlatformLiveAuthorityError("assignment operator is required")
        self.assignments.deactivate(
            product_id,
            at=assigned_at,
            assignment_reason="paper authority replaced by live canary",
            sleeve_id=sleeve_id,
            instrument_id=instrument_id,
        )
        assignment_id = self.assignments.assign(
            product_id=product_id,
            portfolio_id=str(product["portfolio_id"]),
            sleeve_id=sleeve_id,
            strategy_version_id=str(artefact["strategy_version_id"]),
            instrument_id=instrument_id,
            artefact_hash=artefact_hash,
            lifecycle_state="live_canary",
            execution_mode="live",
            capital_limit=capital_limit,
            risk_budget=float(risk_budget),
            assigned_at=assigned_at,
            assigned_by=assigned_by.strip(),
            assignment_reason="explicit operator-approved live canary",
            payload={
                "instrument_ids": [instrument_id],
                "sleeve_id": sleeve_id,
                "approval_id": expected_approval_id,
                "preflight_id": expected_preflight_id,
                "execution_engine_identity": execution_engine_identity(),
            },
        )
        return {
            "schema": "platform.live-assignment-result/v1",
            "assignment_id": assignment_id,
            "approval_id": expected_approval_id,
            "preflight_id": expected_preflight_id,
            "product_id": product_id,
            "artefact_hash": artefact_hash,
            "instrument_id": instrument_id,
            "sleeve_id": sleeve_id,
            "capital_limit": capital_limit,
            "risk_budget": float(risk_budget),
            "assigned_by": assigned_by.strip(),
            "assigned_at": assigned_at,
        }

    def _assert_forward_ready(
        self,
        *,
        product_id: str,
        strategy_version_id: str,
        artefact_hash: str,
        at: str,
    ) -> None:
        assignments = self.assignments.active_assignments(product_id, at=at)
        if any(
            row["execution_mode"] == "paper"
            and row["lifecycle_state"] == "live_ready"
            and row["strategy_version_id"] == strategy_version_id
            and row["artefact_hash"] == artefact_hash
            for row in assignments
        ):
            return
        raise PlatformLiveAuthorityError(
            "live assignment requires a current live_ready lifecycle assignment"
        )

    def _selection(
        self, *, product_id: str, artefact_hash: str, instrument_id: str, sleeve_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        product = self.products.get(product_id)
        if product is None:
            raise PlatformLiveAuthorityError(f"unknown product: {product_id}")
        if product.get("execution_mode") != "live":
            raise PlatformLiveAuthorityError("selected product must be configured live")
        sleeves = product.get("sleeves")
        if not isinstance(sleeves, list) or sleeve_id not in {str(value) for value in sleeves}:
            raise PlatformLiveAuthorityError("selected sleeve is not configured for the product")
        account = self.accounts[str(product["account_id"])]
        if account.get("environment") not in {"testnet", "production"}:
            raise PlatformLiveAuthorityError("selected account environment is invalid")
        try:
            artefact = self.artefacts.get(artefact_hash)
        except (CanonicalEvidenceError, KeyError, ValueError) as exc:
            raise PlatformLiveAuthorityError("canonical artefact is missing or invalid") from exc
        if (
            artefact.get("product_id") != product_id
            or artefact.get("account_id") != account["account_id"]
            or artefact.get("portfolio_id") != product["portfolio_id"]
            or artefact.get("promotion_policy_id") != product["promotion_policy_id"]
        ):
            raise PlatformLiveAuthorityError("artefact binding does not match the selected product")
        self._assert_reviewable(artefact)
        if product_id == "btc_accumulation":
            self._assert_btc_spot_only(artefact)
        supported = {str(value) for value in artefact.get("supported_instruments", [])}
        if instrument_id not in supported:
            raise PlatformLiveAuthorityError("instrument is not supported by the artefact")
        with self.engine.connect() as connection:
            row = connection.execute(
                select(instrument.c.payload).where(instrument.c.id == instrument_id)
            ).scalar_one_or_none()
        if not isinstance(row, Mapping):
            raise PlatformLiveAuthorityError("selected instrument is not persisted")
        instrument_payload = dict(row)
        instrument_payload["instrument_id"] = instrument_id
        expected_market = "spot" if account.get("market") == "spot" else "futures"
        if instrument_payload.get("market_type") != expected_market:
            raise PlatformLiveAuthorityError("instrument market does not match the account")
        configured_live_symbols = product.get("live_exchange_symbols")
        if isinstance(configured_live_symbols, list) and (
            str(instrument_payload.get("exchange_symbol") or "").upper()
            not in {str(value).upper() for value in configured_live_symbols}
        ):
            raise PlatformLiveAuthorityError(
                "selected instrument is not in the configured live product scope"
            )
        return product, account, artefact, instrument_payload

    @staticmethod
    def _assert_reviewable(artefact: Mapping[str, Any]) -> None:
        definition = artefact.get("definition")
        containers = [
            artefact,
            artefact.get("metadata"),
            artefact.get("promotion_policy"),
            definition,
            definition.get("metadata") if isinstance(definition, Mapping) else None,
            definition.get("validation_policy") if isinstance(definition, Mapping) else None,
        ]
        if any(
            isinstance(item, Mapping)
            and (item.get("diagnostic") is True or item.get("promotable") is False)
            for item in containers
        ):
            raise PlatformLiveAuthorityError(
                "diagnostic or non-promotable artefacts cannot receive live authority"
            )
        limits = artefact.get("position_limits")
        maximum = limits.get("maximum_position") if isinstance(limits, Mapping) else None
        if isinstance(maximum, bool) or not isinstance(maximum, int | float) or maximum <= 0:
            raise PlatformLiveAuthorityError("artefact has no positive maximum position")

    @staticmethod
    def _assert_btc_spot_only(artefact: Mapping[str, Any]) -> None:
        supported = tuple(str(value) for value in artefact.get("supported_instruments", ()))
        if supported != ("binance:spot:BTCUSDT",):
            raise PlatformLiveAuthorityError(
                "BTC accumulation artefacts must support BTCUSDT spot only"
            )
        definition = artefact.get("definition")
        if not isinstance(definition, Mapping):
            raise PlatformLiveAuthorityError("BTC accumulation artefact has no definition")
        universe = definition.get("universe")
        if isinstance(universe, Mapping):
            instrument_ids = universe.get("instrument_ids")
            if (
                instrument_ids is not None
                and tuple(str(value) for value in instrument_ids) != supported
            ):
                raise PlatformLiveAuthorityError(
                    "BTC accumulation universe is not restricted to BTCUSDT spot"
                )
        forbidden = {"futures", "leverage", "borrow", "borrowing", "margin"}
        containers = (
            definition.get("position_model"),
            definition.get("execution_preferences"),
            definition.get("risk_policy"),
            artefact.get("position_limits"),
            artefact.get("risk_limits"),
        )
        if any(_contains_forbidden_btc_term(value, forbidden) for value in containers):
            raise PlatformLiveAuthorityError(
                "BTC accumulation artefacts cannot use futures, borrowing, leverage, or margin"
            )

    def _broker(self, account: Mapping[str, Any]):
        market = _market(account)
        if self.broker_factory is not None:
            return self.broker_factory(account, market)
        return CcxtBroker(_exchange_config(account, market=market))

    def _maximum_canary_capital(self, product: Mapping[str, Any]) -> float:
        policy = self.policies[str(product["promotion_policy_id"])]
        return float(policy["canary_capital_limit"])

    def _capital_cap(self, product_id: str, value: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise PlatformLiveAuthorityError("capital cap must be numeric") from exc
        product = self.products.get(product_id)
        if product is None:
            raise PlatformLiveAuthorityError(f"unknown product: {product_id}")
        maximum = self._maximum_canary_capital(product)
        if not math.isfinite(result) or result <= 0 or result > maximum:
            raise PlatformLiveAuthorityError(
                f"capital must be in (0, {maximum}] for the live canary"
            )
        return result

    def _manifest(
        self,
        *,
        product: Mapping[str, Any],
        account: Mapping[str, Any],
        artefact: Mapping[str, Any],
        instrument_payload: Mapping[str, Any],
        sleeve_id: str,
    ) -> dict[str, Any]:
        return {
            "product": dict(product),
            "account": dict(account),
            "instrument": dict(instrument_payload),
            "sleeve_id": sleeve_id,
            "promotion_policy": dict(self.policies[str(product["promotion_policy_id"])]),
            "risk_configuration": dict(self.risk_configuration),
            "artefact_hash": str(artefact["artefact_hash"]),
            "source_commit_hash": str(artefact["source_commit_hash"]),
            "strategy_engine_version": str(artefact["engine_version"]),
            "execution_engine_identity": execution_engine_identity(),
        }

    def _current_preflight(
        self,
        *,
        product: Mapping[str, Any],
        account: Mapping[str, Any],
        artefact: Mapping[str, Any],
        instrument_payload: Mapping[str, Any],
        sleeve_id: str,
        expected_id: str,
        at: str,
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(production_preflight).where(production_preflight.c.id == expected_id)
                )
                .mappings()
                .first()
            )
        if row is None:
            raise PlatformLiveAuthorityError("expected preflight does not exist")
        latest = self.preflights.latest(
            strategy_version_id=str(artefact["strategy_version_id"]),
            product_id=str(product["product_id"]),
            account_id=str(account["account_id"]),
            at=at,
        )
        payload = row["payload"] if isinstance(row["payload"], Mapping) else {}
        expected_fingerprint = _exchange_config(
            account, market=_market(account)
        ).account_fingerprint
        if (
            latest is None
            or latest["id"] != expected_id
            or not row["accepted"]
            or row["artefact_hash"] != artefact["artefact_hash"]
            or row["source_commit_hash"] != artefact["source_commit_hash"]
            or row["engine_version"] != artefact["engine_version"]
            or payload.get("schema") != "platform.production-preflight/v1"
            or payload.get("instrument_id") != instrument_payload["instrument_id"]
            or payload.get("sleeve_id") != sleeve_id
            or payload.get("environment") != account["environment"]
            or payload.get("account_fingerprint") != expected_fingerprint
            or payload.get("execution_engine_identity") != execution_engine_identity()
            or payload.get("configuration_hash")
            != live_authority_configuration_hash(
                product=product,
                account=account,
                instrument_payload=instrument_payload,
                artefact=artefact,
                sleeve_id=sleeve_id,
                promotion_policy=self.policies[str(product["promotion_policy_id"])],
                risk_configuration=self.risk_configuration,
            )
            or not preflight_is_fresh(
                str(row["checked_at"]),
                reference_at=at,
                maximum_age_seconds=int(product.get("preflight_max_age_seconds", 3_600)),
            )
        ):
            raise PlatformLiveAuthorityError("preflight is not current exact authority")
        return dict(row)

    def _current_approval(
        self,
        *,
        product: Mapping[str, Any],
        account: Mapping[str, Any],
        artefact: Mapping[str, Any],
        instrument_payload: Mapping[str, Any],
        sleeve_id: str,
        expected_id: str,
        expected_preflight_id: str,
        at: str,
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(strategy_approval).where(strategy_approval.c.id == expected_id)
                )
                .mappings()
                .first()
            )
        latest = self.approvals.latest(
            strategy_version_id=str(artefact["strategy_version_id"]),
            product_id=str(product["product_id"]),
            account_id=str(account["account_id"]),
            at=at,
        )
        payload = row["payload"] if row is not None and isinstance(row["payload"], Mapping) else {}
        forward = latest_accepted_forward_summary(
            self.engine,
            strategy_version_id=str(artefact["strategy_version_id"]),
            product_id=str(product["product_id"]),
            artefact_hash=str(artefact["artefact_hash"]),
            at=at,
        )
        if (
            row is None
            or latest is None
            or latest["id"] != expected_id
            or row["status"] != "approved"
            or row["artefact_hash"] != artefact["artefact_hash"]
            or row["source_commit_hash"] != artefact["source_commit_hash"]
            or row["engine_version"] != artefact["engine_version"]
            or payload.get("schema") != "platform.strategy-approval/v1"
            or payload.get("preflight_id") != expected_preflight_id
            or payload.get("instrument_id") != instrument_payload["instrument_id"]
            or payload.get("sleeve_id") != sleeve_id
            or payload.get("environment") != account["environment"]
            or payload.get("account_fingerprint")
            != _exchange_config(account, market=_market(account)).account_fingerprint
            or payload.get("execution_engine_identity") != execution_engine_identity()
            or payload.get("configuration_hash")
            != live_authority_configuration_hash(
                product=product,
                account=account,
                instrument_payload=instrument_payload,
                artefact=artefact,
                sleeve_id=sleeve_id,
                promotion_policy=self.policies[str(product["promotion_policy_id"])],
                risk_configuration=self.risk_configuration,
            )
            or forward is None
            or payload.get("forward_summary_id") != str(forward["summary"]["id"])
            or payload.get("forward_decision_id") != str(forward["decision"]["id"])
        ):
            raise PlatformLiveAuthorityError("approval is not current exact authority")
        return dict(row)


def _records(
    payload: Mapping[str, Any], collection: str, identity: str
) -> dict[str, dict[str, Any]]:
    rows = payload.get(collection)
    if not isinstance(rows, list):
        raise PlatformLiveAuthorityError(f"{collection} must be a list")
    return {str(row[identity]): dict(row) for row in rows}


def _market(account: Mapping[str, Any]) -> str:
    return "spot" if account.get("market") == "spot" else "futures"


def _contains_forbidden_btc_term(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            bool(forbidden.intersection(_tokens(key)))
            or _contains_forbidden_btc_term(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set):
        return any(_contains_forbidden_btc_term(item, forbidden) for item in value)
    return bool(forbidden.intersection(_tokens(value)))


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in str(value).casefold().replace("-", "_").replace("/", "_").split("_")
        if token
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage exact canonical live authority.")
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    def selection(command: argparse.ArgumentParser) -> None:
        command.add_argument("--product", required=True)
        command.add_argument("--artefact-hash", required=True)
        command.add_argument("--instrument-id", required=True)
        command.add_argument("--sleeve-id", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    selection(inspect_parser)

    preflight_parser = subparsers.add_parser("preflight")
    selection(preflight_parser)
    preflight_parser.add_argument("--capital-cap", type=float, required=True)

    approve_parser = subparsers.add_parser("approve")
    selection(approve_parser)
    approve_parser.add_argument("--expected-preflight-id", required=True)
    approve_parser.add_argument("--capital-cap", type=float, required=True)
    approve_parser.add_argument("--approved-by", required=True)
    approve_parser.add_argument("--confirm", action="store_true")

    assign_parser = subparsers.add_parser("assign")
    selection(assign_parser)
    assign_parser.add_argument("--expected-preflight-id", required=True)
    assign_parser.add_argument("--expected-approval-id", required=True)
    assign_parser.add_argument("--capital-limit", type=float, required=True)
    assign_parser.add_argument("--risk-budget", type=float, required=True)
    assign_parser.add_argument("--assigned-by", required=True)
    assign_parser.add_argument("--confirm", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    platform = load_platform_config(args.config)
    database = PlatformDatabase(platform.database_url())
    try:
        database.assert_migrated()
        authority = PlatformLiveAuthority(
            engine=database.engine,
            configuration=load_split_configuration(args.config.parent),
        )
        selection = {
            "product_id": args.product,
            "artefact_hash": args.artefact_hash,
            "instrument_id": args.instrument_id,
            "sleeve_id": args.sleeve_id,
        }
        now = utc_now()
        if args.command == "inspect":
            result = authority.inspect(**selection)
        elif args.command == "preflight":
            result = authority.preflight(
                **selection,
                capital_cap=args.capital_cap,
                checked_at=now,
            )
        elif args.command == "approve":
            result = authority.approve(
                **selection,
                expected_preflight_id=args.expected_preflight_id,
                capital_cap=args.capital_cap,
                approved_by=args.approved_by,
                approved_at=now,
                confirm=args.confirm,
            )
        else:
            result = authority.assign(
                **selection,
                expected_preflight_id=args.expected_preflight_id,
                expected_approval_id=args.expected_approval_id,
                capital_limit=args.capital_limit,
                risk_budget=args.risk_budget,
                assigned_by=args.assigned_by,
                assigned_at=now,
                confirm=args.confirm,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
