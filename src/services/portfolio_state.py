"""Publish fully bound portfolio and risk state from immutable source snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.domain._codec import canonical_hash, timestamp
from src.risk.engine import SqlRiskSnapshotStore
from src.services.portfolio_engine import _canonical_portfolio_state


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
