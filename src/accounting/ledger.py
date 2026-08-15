"""Asset-specific, double-entry ledger with immutable hash-chained entries."""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import func, insert, select, text
from sqlalchemy.engine import Engine

from src.data.database import accounting_entry
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp


@dataclass(frozen=True)
class LedgerEntry:
    entry_id: str
    product_id: str
    accounting_asset: str
    occurred_at: str
    postings: Mapping[str, Decimal]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    previous_hash: str = "0" * 64

    def __post_init__(self) -> None:
        for attribute in ("entry_id", "product_id", "accounting_asset"):
            object.__setattr__(
                self, attribute, non_empty(getattr(self, attribute), field=attribute)
            )
        object.__setattr__(self, "accounting_asset", self.accounting_asset.upper())
        object.__setattr__(self, "occurred_at", timestamp(self.occurred_at, field="occurred_at"))
        if len(self.postings) < 2:
            raise ValueError("ledger entries need at least two postings")
        normalised = {
            non_empty(account, field="ledger account"): Decimal(amount)
            for account, amount in self.postings.items()
        }
        if sum(normalised.values()) != 0:
            raise ValueError("ledger postings must balance to zero")
        object.__setattr__(self, "postings", normalised)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), field="metadata"))
        if len(self.previous_hash) != 64:
            raise ValueError("previous_hash must be a SHA-256 hex digest")

    @property
    def entry_hash(self) -> str:
        return canonical_hash(
            {
                "entry_id": self.entry_id,
                "product_id": self.product_id,
                "accounting_asset": self.accounting_asset,
                "occurred_at": self.occurred_at,
                "postings": {key: str(value) for key, value in self.postings.items()},
                "metadata": self.metadata,
                "previous_hash": self.previous_hash,
            }
        ).removeprefix("sha256:")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "product_id": self.product_id,
            "accounting_asset": self.accounting_asset,
            "occurred_at": self.occurred_at,
            "postings": {key: str(value) for key, value in self.postings.items()},
            "metadata": dict(self.metadata),
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }


class JsonlLedgerStore:
    """Append-only ledger persistence with hash-chain verification."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, entry: LedgerEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> tuple[LedgerEntry, ...]:
        if not self.path.exists():
            return ()
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("ledger journal must be a regular file")
        entries: list[LedgerEntry] = []
        previous_hash = "0" * 64
        seen: set[str] = set()
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                payload = json.loads(line)
                saved_hash = payload.pop("entry_hash")
                payload["postings"] = {
                    key: Decimal(value) for key, value in payload["postings"].items()
                }
                entry = LedgerEntry(**payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid ledger entry at line {line_number}") from exc
            if entry.entry_id in seen:
                raise ValueError(f"duplicate ledger entry at line {line_number}")
            if entry.previous_hash != previous_hash or entry.entry_hash != saved_hash:
                raise ValueError(f"ledger hash chain is invalid at line {line_number}")
            entries.append(entry)
            seen.add(entry.entry_id)
            previous_hash = entry.entry_hash
        return tuple(entries)


class LedgerStore(Protocol):
    def append(self, entry: LedgerEntry) -> None: ...

    def read(self) -> tuple[LedgerEntry, ...]: ...


class SqlLedgerStore:
    """Product-isolated immutable accounting entries in PostgreSQL."""

    def __init__(self, engine: Engine, *, product_id: str):
        self.engine = engine
        self.product_id = non_empty(product_id, field="product_id")

    def append(self, entry: LedgerEntry) -> None:
        if entry.product_id != self.product_id:
            raise ValueError("ledger entry product does not match SQL store")
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:product_id))"),
                    {"product_id": self.product_id},
                )
            sequence = (
                int(
                    connection.execute(
                        select(func.coalesce(func.max(accounting_entry.c.sequence), -1)).where(
                            accounting_entry.c.product_id == self.product_id
                        )
                    ).scalar_one()
                )
                + 1
            )
            connection.execute(
                insert(accounting_entry).values(
                    id=entry.entry_id,
                    product_id=self.product_id,
                    sequence=sequence,
                    created_at=entry.occurred_at,
                    entry_hash=entry.entry_hash,
                    payload=entry.to_dict(),
                )
            )

    def read(self) -> tuple[LedgerEntry, ...]:
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(accounting_entry.c.payload)
                .where(accounting_entry.c.product_id == self.product_id)
                .order_by(accounting_entry.c.sequence)
            ).scalars()
            entries: list[LedgerEntry] = []
            previous_hash = "0" * 64
            for payload in payloads:
                values = dict(payload)
                saved_hash = values.pop("entry_hash")
                values["postings"] = {
                    key: Decimal(value) for key, value in values["postings"].items()
                }
                entry = LedgerEntry(**values)
                if entry.previous_hash != previous_hash or entry.entry_hash != saved_hash:
                    raise ValueError("SQL ledger hash chain is invalid")
                entries.append(entry)
                previous_hash = entry.entry_hash
            return tuple(entries)


class Ledger:
    """A product may only contain entries denominated in its accounting asset."""

    def __init__(
        self,
        *,
        product_id: str,
        accounting_asset: str,
        store: LedgerStore | None = None,
    ) -> None:
        self.product_id = non_empty(product_id, field="product_id")
        self.accounting_asset = non_empty(accounting_asset, field="accounting_asset").upper()
        self.store = store
        self._entries = list(store.read()) if store is not None else []
        if any(
            entry.product_id != self.product_id or entry.accounting_asset != self.accounting_asset
            for entry in self._entries
        ):
            raise ValueError("persisted ledger identity does not match configured product")

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def append(
        self,
        *,
        entry_id: str,
        postings: Mapping[str, Decimal],
        metadata: Mapping[str, Any] | None = None,
        occurred_at: str | None = None,
    ) -> LedgerEntry:
        if self.store is not None:
            self._entries = list(self.store.read())
        if any(item.entry_id == entry_id for item in self._entries):
            raise ValueError(f"duplicate ledger entry: {entry_id}")
        entry = LedgerEntry(
            entry_id=entry_id,
            product_id=self.product_id,
            accounting_asset=self.accounting_asset,
            occurred_at=occurred_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            postings=postings,
            metadata=metadata or {},
            previous_hash=self._entries[-1].entry_hash if self._entries else "0" * 64,
        )
        if self.store is not None:
            self.store.append(entry)
        self._entries.append(entry)
        return entry

    def balances(self) -> dict[str, Decimal]:
        balances: dict[str, Decimal] = {}
        for entry in self._entries:
            for account, amount in entry.postings.items():
                balances[account] = balances.get(account, Decimal("0")) + amount
        return balances

    def nav(self) -> Decimal:
        """Reconstruct NAV from immutable asset-account postings."""
        return sum(
            (
                amount
                for account, amount in self.balances().items()
                if account.startswith("assets:")
            ),
            Decimal("0"),
        )

    def record_capital(
        self, *, entry_id: str, amount: Decimal, occurred_at: str | None = None
    ) -> LedgerEntry:
        amount = Decimal(amount)
        if amount <= 0:
            raise ValueError("capital amount must be positive")
        return self.append(
            entry_id=entry_id,
            postings={"assets:cash": amount, "equity:capital": -amount},
            metadata={"kind": "capital", "pnl_effect": "0"},
            occurred_at=occurred_at,
        )

    def record_fee(
        self,
        *,
        entry_id: str,
        amount: Decimal,
        occurred_at: str | None = None,
        attribution: Mapping[str, Any] | None = None,
    ) -> LedgerEntry:
        amount = Decimal(amount)
        if amount < 0:
            raise ValueError("fee amount cannot be negative")
        return self.append(
            entry_id=entry_id,
            postings={"expense:fees": amount, "assets:cash": -amount},
            metadata={
                "kind": "fee",
                "pnl_effect": str(-amount),
                **dict(attribution or {}),
            },
            occurred_at=occurred_at,
        )

    def record_slippage(
        self,
        *,
        entry_id: str,
        amount: Decimal,
        occurred_at: str | None = None,
        attribution: Mapping[str, Any] | None = None,
    ) -> LedgerEntry:
        amount = Decimal(amount)
        if amount < 0:
            raise ValueError("slippage amount cannot be negative")
        return self.append(
            entry_id=entry_id,
            postings={"expense:slippage": amount, "assets:cash": -amount},
            metadata={
                "kind": "slippage",
                "pnl_effect": str(-amount),
                **dict(attribution or {}),
            },
            occurred_at=occurred_at,
        )

    def record_funding(
        self,
        *,
        entry_id: str,
        amount: Decimal,
        occurred_at: str | None = None,
        attribution: Mapping[str, Any] | None = None,
    ) -> LedgerEntry:
        amount = Decimal(amount)
        return self.append(
            entry_id=entry_id,
            postings={"assets:cash": amount, "income:funding": -amount},
            metadata={"kind": "funding", "pnl_effect": str(amount), **dict(attribution or {})},
            occurred_at=occurred_at,
        )

    def record_realised_pnl(
        self,
        *,
        entry_id: str,
        amount: Decimal,
        occurred_at: str | None = None,
        attribution: Mapping[str, Any] | None = None,
    ) -> LedgerEntry:
        amount = Decimal(amount)
        return self.append(
            entry_id=entry_id,
            postings={"assets:cash": amount, "income:realised_pnl": -amount},
            metadata={
                "kind": "realised_pnl",
                "pnl_effect": str(amount),
                **dict(attribution or {}),
            },
            occurred_at=occurred_at,
        )

    def attribution(self, dimension: str) -> dict[str, Decimal]:
        result: dict[str, Decimal] = {}
        for entry in self._entries:
            key = str(entry.metadata.get(dimension) or "unattributed")
            effect = Decimal(str(entry.metadata.get("pnl_effect") or "0"))
            result[key] = result.get(key, Decimal("0")) + effect
        return result
