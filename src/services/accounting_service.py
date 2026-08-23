"""Persist balances, NAV, funding, fees, and attribution evidence."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.accounting.ledger import Ledger
from src.accounting.nav import NavSnapshot
from src.data.database import (
    balance_snapshot,
    fee_entry,
    funding_entry,
    nav_snapshot,
    trade_attribution,
)
from src.domain._codec import canonical_hash, json_value, timestamp
from src.services.scheduler import DatabaseJobQueue


class AccountingService:
    def __init__(self, *, engine: Engine, ledgers: dict[str, Ledger]):
        self.engine = engine
        self.ledgers = dict(ledgers)

    def record_nav(self, snapshot: NavSnapshot) -> str:
        payload = json_value(snapshot.__dict__, field="NAV snapshot")
        identity = canonical_hash(payload)
        self._append(nav_snapshot, identity, snapshot.observed_at, payload)
        return identity

    def record_balances(
        self,
        *,
        account_id: str,
        observed_at: str,
        balances: dict[str, float],
    ) -> str:
        payload = json_value(
            {"account_id": account_id, "balances": balances}, field="balance snapshot"
        )
        identity = canonical_hash({**payload, "observed_at": observed_at})
        self._append(balance_snapshot, identity, observed_at, payload)
        return identity

    def record_funding(
        self,
        *,
        product_id: str,
        entry_id: str,
        amount: Decimal,
        occurred_at: str,
        attribution: dict[str, Any],
    ) -> None:
        ledger = self.ledgers[product_id]
        entry = ledger.record_funding(
            entry_id=entry_id,
            amount=amount,
            occurred_at=occurred_at,
            attribution=attribution,
        )
        self._append(funding_entry, entry_id, occurred_at, entry.to_dict())
        self._append(
            trade_attribution,
            f"{entry_id}:attribution",
            occurred_at,
            {"entry_id": entry_id, **attribution, "pnl_effect": str(amount)},
        )

    def record_fee_evidence(
        self,
        *,
        product_id: str,
        entry_id: str,
        amount: Decimal,
        occurred_at: str,
        attribution: dict[str, Any],
    ) -> None:
        payload = {
            "product_id": product_id,
            "entry_id": entry_id,
            "amount": str(amount),
            "attribution": attribution,
        }
        self._append(fee_entry, entry_id, occurred_at, payload)
        self._append(
            trade_attribution,
            f"{entry_id}:attribution",
            occurred_at,
            {"entry_id": entry_id, **attribution, "pnl_effect": str(-abs(amount))},
        )

    def _append(self, table, identity: str, created_at: str, payload: dict[str, Any]) -> None:
        created_at = timestamp(created_at, field="created_at")
        clean = json_value(payload, field=table.name)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(table.c.payload).where(table.c.id == identity)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != clean:
                    raise ValueError(f"{table.name} identity collision")
                return
            connection.execute(
                insert(table).values(id=identity, created_at=created_at, payload=clean)
            )


class DatabaseAccountingWorker:
    """Apply typed balance, NAV, funding, and fee-evidence jobs."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        service: AccountingService,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.service = service
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("accounting_event",),
        )
        if claimed is None:
            return {"reason_code": "accounting_queue_empty"}
        try:
            kind = str(claimed.payload["kind"])
            payload = claimed.payload
            if kind == "balance":
                identity = self.service.record_balances(
                    account_id=str(payload["account_id"]),
                    observed_at=str(payload["observed_at"]),
                    balances={str(key): float(value) for key, value in payload["balances"].items()},
                )
            elif kind == "nav":
                identity = self.service.record_nav(NavSnapshot(**dict(payload["snapshot"])))
            elif kind == "funding":
                self.service.record_funding(
                    product_id=str(payload["product_id"]),
                    entry_id=str(payload["entry_id"]),
                    amount=Decimal(str(payload["amount"])),
                    occurred_at=str(payload["occurred_at"]),
                    attribution=dict(payload["attribution"]),
                )
                identity = str(payload["entry_id"])
            elif kind == "fee_evidence":
                self.service.record_fee_evidence(
                    product_id=str(payload["product_id"]),
                    entry_id=str(payload["entry_id"]),
                    amount=Decimal(str(payload["amount"])),
                    occurred_at=str(payload["occurred_at"]),
                    attribution=dict(payload["attribution"]),
                )
                identity = str(payload["entry_id"])
            else:
                raise ValueError(f"unsupported accounting event kind: {kind}")
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "accounting_event_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "accounting_event_recorded",
            "job_id": claimed.job_id,
            "record_id": identity,
            "kind": kind,
        }


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
