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
from src.risk.engine import SqlRiskSnapshotStore
from src.services.scheduler import DatabaseJobQueue


class AccountingService:
    def __init__(
        self,
        *,
        engine: Engine,
        ledgers: dict[str, Ledger] | None = None,
        snapshot_store: SqlRiskSnapshotStore | None = None,
    ):
        self.engine = engine
        self.ledgers = dict(ledgers or {})
        self.snapshot_store = snapshot_store

    def record_nav(self, snapshot: NavSnapshot) -> str:
        snapshot = NavSnapshot(**dict(snapshot.__dict__))
        payload = json_value(snapshot.__dict__, field="NAV snapshot")
        identity = canonical_hash(payload)
        self._append(nav_snapshot, identity, snapshot.observed_at, payload)
        return identity

    def latest_nav(self, *, product_id: str, at: str) -> NavSnapshot | None:
        at = timestamp(at, field="NAV lookup time")
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(nav_snapshot.c.payload)
                .where(nav_snapshot.c.created_at <= at)
                .order_by(nav_snapshot.c.created_at.desc(), nav_snapshot.c.id.desc())
            ).scalars()
            for payload in rows:
                if isinstance(payload, dict) and str(payload.get("product_id")) == product_id:
                    return NavSnapshot(**payload)
        return None

    def record_balances(
        self,
        *,
        account_id: str,
        observed_at: str,
        balances: dict[str, float],
        product_id: str | None = None,
        used_margin_fraction: float = 0.0,
        liquidation_buffer_fraction: float = 1.0,
        unknown_exposure: dict[str, float] | None = None,
        account_state_known: bool = True,
        account_state_authority: str = "accounting_event",
    ) -> str:
        payload = json_value(
            {
                "account_id": account_id,
                "balances": balances,
                "used_margin_fraction": float(used_margin_fraction),
                "liquidation_buffer_fraction": float(liquidation_buffer_fraction),
                "unknown_exposure": dict(unknown_exposure or {}),
                "account_state_known": bool(account_state_known),
                "account_state_authority": str(account_state_authority),
            },
            field="balance snapshot",
        )
        identity = canonical_hash({**payload, "observed_at": observed_at})
        self._append(balance_snapshot, identity, observed_at, payload)
        if self.snapshot_store is not None and product_id is not None:
            self.snapshot_store.save(
                {
                    "kind": "balances",
                    "product_id": product_id,
                    "observed_at": observed_at,
                    "values": {"balances": dict(balances)},
                },
                created_at=observed_at,
            )
            self.snapshot_store.save(
                {
                    "kind": "account",
                    "product_id": product_id,
                    "observed_at": observed_at,
                    "values": {
                        "used_margin_fraction": float(payload["used_margin_fraction"]),
                        "liquidation_buffer_fraction": float(
                            payload["liquidation_buffer_fraction"]
                        ),
                        "unknown_exposure": dict(payload["unknown_exposure"]),
                    },
                },
                created_at=observed_at,
            )
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
        job_name: str = "accounting_event",
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.service = service
        self.lease_seconds = lease_seconds
        self.job_name = job_name

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=(self.job_name,),
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
                    product_id=(str(payload["product_id"]) if payload.get("product_id") else None),
                    used_margin_fraction=float(payload.get("used_margin_fraction", 0.0)),
                    liquidation_buffer_fraction=float(
                        payload.get("liquidation_buffer_fraction", 1.0)
                    ),
                    unknown_exposure={
                        str(key): float(value)
                        for key, value in dict(payload.get("unknown_exposure", {})).items()
                    },
                    account_state_known=bool(payload.get("account_state_known", False)),
                    account_state_authority=str(
                        payload.get("account_state_authority", "user_stream_delta")
                    ),
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
