"""Publish fully bound portfolio and risk state from immutable source snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.domain._codec import canonical_hash, timestamp
from src.risk.engine import SqlRiskSnapshotStore
from src.services.portfolio_engine import _canonical_portfolio_state
from src.services.scheduler import DatabaseJobQueue


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
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.publisher = CanonicalPortfolioStatePublisher(store)
        self.lease_seconds = lease_seconds

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
            source_ids: dict[str, str] = {}
            try:
                for source in sorted(self.REQUIRED_SOURCES):
                    identity, _ = self.store.latest(kind=source, product_id=product_id, at=now)
                    source_ids[source] = identity
            except KeyError:
                continue
            policy = state_policies.get(product_id)
            if not isinstance(policy, Mapping):
                raise ValueError(f"portfolio state policy is missing for {product_id}")
            payload = {
                "product_id": product_id,
                "observed_at": now,
                "source_snapshot_ids": source_ids,
                "risk_policy": dict(policy),
            }
            identity = canonical_hash(payload).removeprefix("sha256:")
            if self.queue.enqueue_if_absent(
                job_id=f"portfolio-state:{identity}",
                name="portfolio_state_publish",
                payload=payload,
                available_at=now,
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
            assembled: dict[str, Any] = {
                "kind": "canonical_portfolio_risk_state",
                "product_id": str(claimed.payload["product_id"]),
                "observed_at": str(claimed.payload["observed_at"]),
                "source_snapshot_ids": dict(source_ids),
            }
            for source, identity in source_ids.items():
                snapshot = self.store.get(str(identity))
                if snapshot.get("kind") not in {source, f"{source}_snapshot"}:
                    raise ValueError(f"portfolio state {source} snapshot has the wrong kind")
                observed_at = snapshot.get("observed_at", snapshot.get("created_at"))
                if (
                    observed_at is not None
                    and timestamp(str(observed_at), field=f"{source}.observed_at")
                    > assembled["observed_at"]
                ):
                    raise ValueError(f"portfolio state {source} snapshot is from the future")
                values = snapshot.get("values", snapshot)
                if not isinstance(values, Mapping):
                    raise ValueError(f"portfolio state {source} snapshot has no values")
                for key, value in values.items():
                    if key not in {"kind", "product_id", "observed_at", "created_at"}:
                        assembled[str(key)] = value
            policy = claimed.payload.get("risk_policy")
            if not isinstance(policy, Mapping):
                raise ValueError("portfolio state job requires immutable risk policy values")
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
