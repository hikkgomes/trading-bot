"""Publish fully bound portfolio and risk state from immutable source snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import select

from src.data.database import (
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
        positions, positions_at = self._positions(str(product["portfolio_id"]), now)
        open_orders, orders_at = self._open_orders(str(product["portfolio_id"]), now)
        market, market_at = self._market(product_id, now)
        if balance_payload is None or not market:
            return
        observed_at = max(
            value for value in (balance_at, positions_at, orders_at, market_at) if value is not None
        )
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
            values={
                "used_margin_fraction": float(balance_payload.get("used_margin_fraction", 0.0)),
                "liquidation_buffer_fraction": float(
                    balance_payload.get("liquidation_buffer_fraction", 1.0)
                ),
                "unknown_exposure": dict(balance_payload.get("unknown_exposure", {})),
                "account_id": account_id,
            },
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
        health_values = self._health(now)
        self.publish(
            product_id=product_id,
            kind="health",
            # Heartbeat timestamps describe liveness, not a new portfolio
            # input. Keep them out of the snapshot identity so an unchanged
            # system does not publish a new canonical state every second.
            observed_at=observed_at,
            values=health_values,
        )
        drift_values, drift_at = self._drift(str(product["portfolio_id"]), now)
        self.publish(
            product_id=product_id,
            kind="drift",
            observed_at=drift_at or observed_at,
            values=drift_values,
        )

    def _latest_payload(
        self, table, account_id: str, at: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        with self.engine.connect() as connection:
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

    def _market(self, product_id: str, at: str) -> tuple[dict[str, dict[str, float]], str | None]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(risk_snapshot.c.payload, risk_snapshot.c.created_at)
                .where(risk_snapshot.c.created_at <= at)
                .order_by(risk_snapshot.c.created_at.desc(), risk_snapshot.c.id.desc())
            ).mappings()
        market: dict[str, dict[str, float]] = {}
        latest_at: str | None = None
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, dict) or payload.get("kind") != "market_data_input":
                continue
            if str(payload.get("product_id")) != product_id:
                continue
            instrument_id = str(payload["instrument_id"])
            raw = payload.get("values")
            if not isinstance(raw, Mapping) or not {
                "close",
                "spread_bps",
                "visible_depth",
                "volatility",
                "funding",
            }.issubset(raw):
                continue
            market[instrument_id] = {
                "price": float(raw["close"]),
                "spread_bps": float(raw["spread_bps"]),
                "visible_depth": float(raw["visible_depth"]),
                "volatility": float(raw["volatility"]),
                "funding": float(raw["funding"]),
            }
            latest_at = latest_at or str(row["created_at"])
        return market, latest_at

    def _health(self, at: str) -> dict[str, Any]:
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
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            service_name = str(row["service_name"])
            # The state service must not use its own heartbeat as a source.
            # Its heartbeat is written immediately before refresh_sources and
            # would otherwise make every state identity wall-clock dependent.
            if service_name == "portfolio-state-service":
                continue
            key = (service_name, str(row["node_id"]))
            if key in latest:
                continue
            latest[key] = {
                "healthy": bool(row["healthy"]),
                "observed_at": str(row["observed_at"]),
                "payload": dict(row["payload"]) if isinstance(row["payload"], dict) else {},
            }
        statuses = {
            f"{service}@{node}": value["healthy"]
            for (service, node), value in sorted(latest.items())
        }
        return {
            "data_age_seconds": 0.0,
            "clock_skew_seconds": 0.0,
            "exchange_connected": all(statuses.values()) if statuses else True,
            "database_healthy": self._database_healthy(),
            "services": statuses,
        }

    def _database_healthy(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1").scalar_one()
            return True
        except Exception:
            return False

    def _drift(self, portfolio_id: str, at: str) -> tuple[dict[str, bool], str | None]:
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
            if not isinstance(payload, dict) or str(payload.get("portfolio_id")) != portfolio_id:
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
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.publisher = CanonicalPortfolioStatePublisher(store)
        self.lease_seconds = lease_seconds
        self.refresh_sources = refresh_sources

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
                job_id=f"portfolio-state:{identity}",
                name="portfolio_state_publish",
                payload=payload,
                available_at=observed_at,
                priority=25,
                producer_identity="portfolio-state-service",
            ):
                scheduled += 1
        return scheduled

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("portfolio_state_publish",),
        )
        if claimed is None:
            return {"reason_code": "portfolio_state_queue_empty"}
        try:
            source_ids = claimed.payload.get("source_snapshot_ids")
            if not isinstance(source_ids, Mapping) or set(source_ids) != self.REQUIRED_SOURCES:
                raise ValueError("portfolio state source snapshot identities are incomplete")
            policy_hash = str(claimed.payload.get("risk_policy_hash") or "")
            expected_job_id = "portfolio-state:" + canonical_hash(
                {
                    "product_id": str(claimed.payload["product_id"]),
                    "source_snapshot_ids": dict(source_ids),
                    "risk_policy_hash": policy_hash,
                }
            ).removeprefix("sha256:")
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
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=now,
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
