"""Publish fully bound portfolio and risk state from immutable source snapshots."""

from __future__ import annotations

import datetime as dt
import statistics
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import select

from src.data.database import (
    account_snapshot,
    balance_snapshot,
    exchange_order,
    order_intent,
    position,
    reconciliation_event,
    risk_snapshot,
    service_heartbeat,
)
from src.domain._codec import canonical_hash, timestamp
from src.risk.engine import SqlRiskSnapshotStore
from src.services.portfolio_engine import _canonical_portfolio_state
from src.services.scheduler import DatabaseJobQueue


class DatabasePortfolioSourceService:
    """Publish portfolio inputs from the durable accounting and execution state."""

    def __init__(
        self,
        *,
        engine,
        store: SqlRiskSnapshotStore,
        products: Mapping[str, Mapping[str, Any]],
        accounts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.engine = engine
        self.store = store
        self.products = {str(key): dict(value) for key, value in products.items()}
        self.accounts = {str(key): dict(value) for key, value in accounts.items()}

    def publish(
        self,
        *,
        product_id: str,
        kind: str,
        observed_at: str,
        values: Mapping[str, Any],
    ) -> str:
        observed_at = timestamp(observed_at, field="observed_at")
        return self.store.save(
            {
                "kind": kind,
                "product_id": product_id,
                "observed_at": observed_at,
                "values": dict(values),
            },
            created_at=observed_at,
        )

    def refresh(self, product_id: str, now: str) -> None:
        product = self.products.get(product_id)
        if product is None:
            raise ValueError(f"portfolio source product is unknown: {product_id}")
        account_id = str(product["account_id"])
        now = timestamp(now, field="now")
        balance_payload, balance_at = self._latest_payload(balance_snapshot, account_id, now)
        if product.get("execution_mode") == "live":
            required_authority = {
                "account_state_known",
                "account_state_authority",
                "used_margin_fraction",
                "liquidation_buffer_fraction",
                "unknown_exposure",
                "positions",
                "regular_orders",
                "conditional_orders",
                "account_mode",
            }
            if (
                balance_payload is None
                or not required_authority.issubset(balance_payload)
                or balance_payload.get("account_state_known") is not True
                or balance_payload.get("account_state_authority")
                not in {"authenticated_rest", "authenticated_reconciled"}
            ):
                return
        positions, positions_at = self._positions(str(product["portfolio_id"]), now)
        open_orders, orders_at = self._open_orders(str(product["portfolio_id"]), now)
        market, market_at = self._market(product_id, now)
        if balance_payload is None or not market:
            return
        # An empty position or order book has no authoritative event timestamp.
        # Reusing the prior derived snapshot keeps an unchanged empty state
        # content-addressed instead of making it depend on the latest market
        # event's wall-clock timestamp.
        positions_at = positions_at or self._previous_observed_at(
            kind="positions", product_id=product_id, at=now
        )
        orders_at = orders_at or self._previous_observed_at(
            kind="open_orders", product_id=product_id, at=now
        )
        observed_at = max(
            value for value in (balance_at, positions_at, orders_at, market_at) if value is not None
        )
        health_values, health_at = self._health(now)
        if not health_values:
            return
        drift_values, drift_at = self._drift(
            product_id=product_id,
            portfolio_id=str(product["portfolio_id"]),
            account_id=account_id,
            at=now,
        )
        account_values = {
            "used_margin_fraction": float(balance_payload.get("used_margin_fraction", 0.0)),
            "liquidation_buffer_fraction": float(
                balance_payload.get("liquidation_buffer_fraction", 1.0)
            ),
            "unknown_exposure": dict(balance_payload.get("unknown_exposure", {})),
            "account_id": account_id,
        }
        if "account_state_known" in balance_payload:
            account_values["account_state_known"] = bool(balance_payload["account_state_known"])
        if "account_state_authority" in balance_payload:
            account_values["account_state_authority"] = str(
                balance_payload["account_state_authority"]
            )
        if "account_fingerprint" in balance_payload:
            account_values["account_fingerprint"] = str(balance_payload["account_fingerprint"])
        self.publish(
            product_id=product_id,
            kind="balances",
            observed_at=balance_at or observed_at,
            values={"balances": dict(balance_payload.get("balances", {}))},
        )
        self.publish(
            product_id=product_id,
            kind="account",
            observed_at=balance_at or observed_at,
            values=account_values,
        )
        self.publish(
            product_id=product_id,
            kind="positions",
            observed_at=positions_at or observed_at,
            values={"positions": positions},
        )
        self.publish(
            product_id=product_id,
            kind="open_orders",
            observed_at=orders_at or observed_at,
            values={"open_orders": open_orders},
        )
        self.publish(
            product_id=product_id,
            kind="market",
            observed_at=market_at or observed_at,
            values={"market": market, "correlations": {}, "beta": {}},
        )
        self.publish(
            product_id=product_id,
            kind="health",
            # Heartbeat timestamps describe liveness, not a new portfolio
            # input. Keep them out of the snapshot identity so an unchanged
            # system does not publish a new canonical state every second.
            observed_at=self._stable_observed_at(
                kind="health",
                product_id=product_id,
                values=health_values,
                candidate=health_at or observed_at,
                at=now,
            ),
            values=health_values,
        )
        self.publish(
            product_id=product_id,
            kind="drift",
            observed_at=drift_at
            or self._previous_observed_at(kind="drift", product_id=product_id, at=now)
            or observed_at,
            values=drift_values,
        )

    def _previous_observed_at(self, *, kind: str, product_id: str, at: str) -> str | None:
        try:
            _identity, snapshot = self.store.latest(kind=kind, product_id=product_id, at=at)
        except KeyError:
            return None
        value = snapshot.get("observed_at", snapshot.get("created_at"))
        return timestamp(str(value), field=f"{kind}.observed_at") if value is not None else None

    def _stable_observed_at(
        self,
        *,
        kind: str,
        product_id: str,
        values: Mapping[str, Any],
        candidate: str,
        at: str,
    ) -> str:
        try:
            _identity, snapshot = self.store.latest(kind=kind, product_id=product_id, at=at)
        except KeyError:
            return timestamp(candidate, field=f"{kind}.observed_at")
        previous_values = snapshot.get("values")
        if isinstance(previous_values, Mapping) and canonical_hash(
            dict(previous_values)
        ) == canonical_hash(dict(values)):
            previous_at = snapshot.get("observed_at", snapshot.get("created_at"))
            if previous_at is not None:
                return timestamp(str(previous_at), field=f"{kind}.observed_at")
        return timestamp(candidate, field=f"{kind}.observed_at")

    def _latest_payload(
        self, table, account_id: str, at: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        with self.engine.connect() as connection:
            if table is balance_snapshot:
                account_rows = connection.execute(
                    select(account_snapshot.c.payload, account_snapshot.c.observed_at)
                    .where(
                        account_snapshot.c.account_id == account_id,
                        account_snapshot.c.observed_at <= at,
                    )
                    .order_by(account_snapshot.c.observed_at.desc(), account_snapshot.c.id.desc())
                ).mappings()
                row = next(iter(account_rows), None)
                if row is not None and isinstance(row["payload"], dict):
                    return dict(row["payload"]), str(row["observed_at"])
            rows = connection.execute(
                select(table.c.payload, table.c.created_at)
                .where(table.c.created_at <= at)
                .order_by(table.c.created_at.desc(), table.c.id.desc())
            ).mappings()
            for row in rows:
                payload = row["payload"]
                if isinstance(payload, dict) and str(payload.get("account_id")) == account_id:
                    return dict(payload), str(row["created_at"])
        return None, None

    def _positions(self, portfolio_id: str, at: str) -> tuple[dict[str, float], str | None]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(position.c.payload, position.c.created_at)
                .where(position.c.created_at <= at)
                .order_by(position.c.created_at.desc(), position.c.id.desc())
            ).mappings()
        values: dict[str, float] = {}
        latest_at: str | None = None
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, dict) or str(payload.get("portfolio_id")) != portfolio_id:
                continue
            instrument_id = str(payload["instrument_id"])
            if instrument_id in values:
                continue
            values[instrument_id] = float(payload.get("quantity", 0.0))
            latest_at = latest_at or str(row["created_at"])
        return values, latest_at

    def _open_orders(
        self, portfolio_id: str, at: str
    ) -> tuple[tuple[dict[str, Any], ...], str | None]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(order_intent.c.payload, exchange_order.c.status, exchange_order.c.created_at)
                .select_from(
                    order_intent.join(
                        exchange_order, order_intent.c.id == exchange_order.c.order_id
                    )
                )
                .where(exchange_order.c.created_at <= at)
                .order_by(exchange_order.c.created_at.desc(), exchange_order.c.sequence.desc())
            ).mappings()
        terminal = {"cancelled", "rejected", "expired", "reconciled", "filled"}
        latest: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        latest_at: str | None = None
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, dict) or str(payload.get("portfolio_id")) != portfolio_id:
                continue
            order_id = str(payload["order_id"])
            if order_id in seen:
                continue
            seen.add(order_id)
            latest_at = latest_at or str(row["created_at"])
            if str(row["status"]) not in terminal:
                latest[order_id] = {**payload, "status": str(row["status"])}
        return tuple(latest[key] for key in sorted(latest)), latest_at

    def _market(self, product_id: str, at: str) -> tuple[dict[str, dict[str, Any]], str | None]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(risk_snapshot.c.payload, risk_snapshot.c.created_at)
                .where(risk_snapshot.c.created_at <= at)
                .order_by(risk_snapshot.c.created_at.desc(), risk_snapshot.c.id.desc())
            ).mappings()
        market: dict[str, dict[str, Any]] = {}
        product = self.products.get(product_id, {})
        account = self.accounts.get(str(product.get("account_id")), {})
        market_type = "spot" if account.get("market") == "spot" else "futures"
        latest_at: str | None = None
        fields = {
            "close": "price",
            "price": "price",
            "spread_bps": "spread_bps",
            "visible_depth": "visible_depth",
            "volatility": "volatility",
            "funding": "funding",
            "funding_rate": "funding",
        }
        latest_values: dict[str, dict[str, Any]] = {}
        latest_times: dict[str, str] = {}
        field_times: dict[str, dict[str, str]] = {}
        close_history: dict[str, list[float]] = {}
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, dict) or payload.get("kind") != "market_data_input":
                continue
            if str(payload.get("product_id")) != product_id:
                continue
            instrument_id = str(payload.get("instrument_id") or "")
            if not instrument_id:
                continue
            raw = payload.get("values")
            if not isinstance(raw, Mapping):
                continue
            values = latest_values.setdefault(instrument_id, {})
            close = raw.get("close", raw.get("price"))
            try:
                close_value = float(close)
            except (TypeError, ValueError):
                close_value = 0.0
            if close_value > 0 and close_value == close_value:
                close_history.setdefault(instrument_id, []).append(close_value)
            for source_name, target_name in fields.items():
                if target_name in values or source_name not in raw:
                    continue
                try:
                    value = float(raw[source_name])
                except (TypeError, ValueError):
                    continue
                if value != value or value in (float("inf"), float("-inf")):
                    continue
                values[target_name] = value
                field_times.setdefault(instrument_id, {})[target_name] = str(row["created_at"])
                latest_times.setdefault(instrument_id, str(row["created_at"]))
        required = {"price", "spread_bps", "visible_depth", "volatility"}
        if market_type == "futures":
            required.add("funding")
        for instrument_id, values in latest_values.items():
            if "volatility" not in values:
                closes = list(reversed(close_history.get(instrument_id, [])))
                returns = [
                    closes[index] / closes[index - 1] - 1.0
                    for index in range(1, len(closes))
                    if closes[index - 1] > 0
                ]
                if len(returns) >= 2:
                    values["volatility"] = statistics.pstdev(returns)
            if market_type == "spot":
                # Funding is not a spot market requirement. Supplying a zero
                # value keeps downstream risk payloads structurally uniform.
                values.setdefault("funding", 0.0)
            else:
                funding_at = field_times.get(instrument_id, {}).get("funding")
                if funding_at is None:
                    continue
                funding_age = (
                    dt.datetime.fromisoformat(at) - dt.datetime.fromisoformat(funding_at)
                ).total_seconds()
                maximum_funding_age = float(product.get("maximum_funding_age_seconds", 28_800))
                if funding_age < 0 or funding_age > maximum_funding_age:
                    continue
            if not required.issubset(values):
                continue
            values["market_type"] = market_type
            market[instrument_id] = values
            row_time = latest_times[instrument_id]
            latest_at = row_time if latest_at is None else max(latest_at, row_time)
        return market, latest_at

    def _health(self, at: str) -> tuple[dict[str, Any], str | None]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    service_heartbeat.c.service_name,
                    service_heartbeat.c.node_id,
                    service_heartbeat.c.observed_at,
                    service_heartbeat.c.healthy,
                    service_heartbeat.c.payload,
                )
                .where(service_heartbeat.c.observed_at <= at)
                .order_by(service_heartbeat.c.observed_at.desc(), service_heartbeat.c.id.desc())
            ).mappings()
        latest: dict[tuple[str, str], bool] = {}
        latest_observed_at: str | None = None
        for row in rows:
            service_name = str(row["service_name"])
            # The state service must not use its own heartbeat as a source.
            # Its heartbeat is written immediately before refresh_sources and
            # would otherwise make every state identity wall-clock dependent.
            if service_name == "portfolio-state-service":
                continue
            observed_at = timestamp(str(row["observed_at"]), field="health.observed_at")
            latest_observed_at = latest_observed_at or observed_at
            key = (service_name, str(row["node_id"]))
            if key in latest:
                continue
            latest[key] = bool(row["healthy"])
        if not latest:
            return {}, None
        statuses = {
            f"{service}@{node}": healthy for (service, node), healthy in sorted(latest.items())
        }
        return (
            {
                "data_age_seconds": 0.0,
                "clock_skew_seconds": 0.0,
                "exchange_connected": all(statuses.values()) if statuses else True,
                "database_healthy": self._database_healthy(),
                "services": statuses,
            },
            latest_observed_at,
        )

    def _database_healthy(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1").scalar_one()
            return True
        except Exception:
            return False

    def _drift(
        self,
        *,
        product_id: str,
        portfolio_id: str,
        account_id: str,
        at: str,
    ) -> tuple[dict[str, bool], str | None]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(reconciliation_event.c.payload, reconciliation_event.c.created_at)
                .where(reconciliation_event.c.created_at <= at)
                .order_by(
                    reconciliation_event.c.created_at.desc(), reconciliation_event.c.id.desc()
                )
            ).mappings()
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, dict) or not (
                str(payload.get("product_id")) == product_id
                or str(payload.get("portfolio_id")) == portfolio_id
                or str(payload.get("account_id")) == account_id
            ):
                continue
            return {
                "execution_drift": bool(payload.get("execution_drift", False)),
                "model_drift": bool(payload.get("model_drift", False)),
            }, str(row["created_at"])
        return {"execution_drift": False, "model_drift": False}, None


class CanonicalPortfolioStatePublisher:
    def __init__(self, store: SqlRiskSnapshotStore):
        self.store = store

    def publish(self, payload: Mapping[str, Any]) -> str:
        product_id = str(payload.get("product_id") or "")
        clean = _canonical_portfolio_state(payload, product_id=product_id)
        observed_at = timestamp(str(clean["observed_at"]), field="observed_at")
        sources = clean["source_snapshot_ids"]
        receipt = {
            "product_id": product_id,
            "observed_at": observed_at,
            "source_snapshot_ids": sources,
        }
        record = {
            **clean,
            "assembly_receipt": {**receipt, "receipt_hash": canonical_hash(receipt)},
        }
        return self.store.save(record, created_at=observed_at)


class DatabasePortfolioStateWorker:
    """Assemble canonical state only from immutable source snapshot identities."""

    REQUIRED_SOURCES = frozenset(
        {"balances", "positions", "open_orders", "account", "market", "health", "drift"}
    )

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        store: SqlRiskSnapshotStore,
        lease_seconds: int = 60,
        refresh_sources: Callable[[str, str], None] | None = None,
        job_name: str = "portfolio_state_publish",
        job_id_prefix: str = "portfolio-state",
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.publisher = CanonicalPortfolioStatePublisher(store)
        self.lease_seconds = lease_seconds
        self.refresh_sources = refresh_sources
        self.job_name = job_name
        self.job_id_prefix = job_id_prefix
        self.last_schedule_reason = "portfolio_state_queue_empty"

    def schedule_from_latest(
        self,
        *,
        products: Mapping[str, Mapping[str, Any]],
        state_policies: Mapping[str, Mapping[str, Any]],
        now: str,
    ) -> int:
        """Enqueue content-addressed assemblies when every source is available."""

        now = timestamp(now, field="now")
        scheduled = 0
        missing_sources = False
        for product_id in sorted(products):
            if self.refresh_sources is not None:
                self.refresh_sources(product_id, now)
            source_ids: dict[str, str] = {}
            source_observed_at: list[str] = []
            try:
                for source in sorted(self.REQUIRED_SOURCES):
                    identity, snapshot = self.store.latest(
                        kind=source, product_id=product_id, at=now
                    )
                    source_ids[source] = identity
                    observed_at = snapshot.get("observed_at", snapshot.get("created_at"))
                    if observed_at is not None:
                        source_observed_at.append(
                            timestamp(str(observed_at), field=f"{source}.observed_at")
                        )
            except KeyError:
                missing_sources = True
                continue
            policy = state_policies.get(product_id)
            if not isinstance(policy, Mapping):
                raise ValueError(f"portfolio state policy is missing for {product_id}")
            policy_hash = canonical_hash(dict(policy))
            observed_at = max(source_observed_at, default=now)
            payload = {
                "product_id": product_id,
                "observed_at": observed_at,
                "source_snapshot_ids": source_ids,
                "risk_policy": dict(policy),
                "risk_policy_hash": policy_hash,
            }
            identity_payload = {
                "product_id": product_id,
                "source_snapshot_ids": source_ids,
                "risk_policy_hash": policy_hash,
            }
            identity = canonical_hash(identity_payload).removeprefix("sha256:")
            if self.queue.enqueue_if_absent(
                job_id=f"{self.job_id_prefix}:{identity}",
                name=self.job_name,
                payload=payload,
                available_at=observed_at,
                priority=25,
                producer_identity="portfolio-state-service",
            ):
                scheduled += 1
        self.last_schedule_reason = (
            "portfolio_state_jobs_scheduled"
            if scheduled
            else "portfolio_state_waiting_for_source_snapshots"
            if missing_sources
            else "portfolio_state_idle"
        )
        return scheduled

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=(self.job_name,),
        )
        if claimed is None:
            return {"reason_code": "portfolio_state_queue_empty"}
        try:
            source_ids = claimed.payload.get("source_snapshot_ids")
            if not isinstance(source_ids, Mapping) or set(source_ids) != self.REQUIRED_SOURCES:
                raise ValueError("portfolio state source snapshot identities are incomplete")
            policy_hash = str(claimed.payload.get("risk_policy_hash") or "")
            expected_job_id = (
                self.job_id_prefix
                + ":"
                + canonical_hash(
                    {
                        "product_id": str(claimed.payload["product_id"]),
                        "source_snapshot_ids": dict(source_ids),
                        "risk_policy_hash": policy_hash,
                    }
                ).removeprefix("sha256:")
            )
            if claimed.job_id != expected_job_id:
                raise ValueError("portfolio state job identity is not content-addressed")
            assembled: dict[str, Any] = {
                "kind": "canonical_portfolio_risk_state",
                "product_id": str(claimed.payload["product_id"]),
                "source_snapshot_ids": dict(source_ids),
                "risk_policy_hash": policy_hash,
            }
            source_observed_at: list[str] = []
            for source, identity in source_ids.items():
                snapshot = self.store.get(str(identity))
                if snapshot.get("kind") not in {source, f"{source}_snapshot"}:
                    raise ValueError(f"portfolio state {source} snapshot has the wrong kind")
                if str(snapshot.get("product_id") or "") != assembled["product_id"]:
                    raise ValueError(
                        f"portfolio state {source} snapshot belongs to another product"
                    )
                observed_at = snapshot.get("observed_at", snapshot.get("created_at"))
                if observed_at is None:
                    raise ValueError(f"portfolio state {source} snapshot has no timestamp")
                source_observed_at.append(
                    timestamp(str(observed_at), field=f"{source}.observed_at")
                )
                values = snapshot.get("values", snapshot)
                if not isinstance(values, Mapping):
                    raise ValueError(f"portfolio state {source} snapshot has no values")
                for key, value in values.items():
                    if key not in {"kind", "product_id", "observed_at", "created_at"}:
                        assembled[str(key)] = value
            latest_source_at = max(source_observed_at)
            claimed_observed_at = timestamp(
                str(claimed.payload.get("observed_at")), field="observed_at"
            )
            if claimed_observed_at != latest_source_at:
                raise ValueError("portfolio state observed_at is not the latest source timestamp")
            assembled["observed_at"] = latest_source_at
            policy = claimed.payload.get("risk_policy")
            if not isinstance(policy, Mapping):
                raise ValueError("portfolio state job requires immutable risk policy values")
            if str(claimed.payload.get("risk_policy_hash") or "") != canonical_hash(dict(policy)):
                raise ValueError("portfolio state risk policy hash is invalid")
            assembled.update(policy)
            state_id = self.publisher.publish(assembled)
        except Exception as exc:
            retry_at = (
                dt.datetime.fromisoformat(now) + dt.timedelta(seconds=self.lease_seconds)
            ).replace(microsecond=0).isoformat()
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=retry_at,
            )
            return {
                "reason_code": "portfolio_state_publish_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "canonical_portfolio_state_published",
            "job_id": claimed.job_id,
            "state_id": state_id,
        }


def portfolio_state_policies(
    configuration: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the canonical state policy bound to each configured product."""

    risk = configuration["risk"]
    global_limits = risk["global"]
    instrument = risk["instrument"]
    sleeve = risk["sleeve"]
    product_limits = risk["products"]
    result: dict[str, dict[str, Any]] = {}
    for product_id, product in products.items():
        risk_policy_id = str(product["risk_policy_id"])
        limits = product_limits[risk_policy_id]
        sleeves = tuple(str(item) for item in product.get("sleeves", ()))
        sleeve_budget = 1.0 / len(sleeves) if sleeves else 1.0
        result[product_id] = {
            "maximum_state_age_seconds": min(
                float(global_limits["maximum_market_data_staleness_seconds"]),
                float(global_limits["maximum_database_staleness_seconds"]),
            ),
            "risk_policy_ids": [risk_policy_id],
            "portfolio_risk_budget": float(
                limits.get("maximum_exposure", limits.get("maximum_gross", 1.0))
            ),
            "maximum_symbol_fraction": float(instrument["maximum_fraction"]),
            "maximum_abs_beta": float(sleeve["maximum_abs_beta"]),
            "maximum_correlation": float(sleeve["maximum_correlation"]),
            "maximum_turnover_fraction": float(sleeve["maximum_turnover_fraction"]),
            "maximum_cluster_fraction": float(sleeve["maximum_fraction"]),
            "maximum_product_drawdown_fraction": float(limits["maximum_drawdown"]),
            "maximum_depth_participation": float(instrument["maximum_visible_depth_fraction"]),
            "product_drawdown_fraction": 0.0,
            "daily_pnl_fraction": 0.0,
            "global_drawdown_fraction": 0.0,
            "trades_today": 0,
            "sleeve_budgets": {name: sleeve_budget for name in sleeves},
            "clusters": {},
            "cluster_fraction_caps": {},
        }
    return result
