"""Durable paper-account state derived from the product accounting ledger."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import IntegrityError

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.domain._codec import canonical_hash, timestamp
from src.domain.orders import Fill, OrderIntent, OrderSide
from src.domain.positions import Position


def configured_paper_accounts(engine, configuration) -> dict[str, PaperAccountLedger]:
    accounts = {row["account_id"]: row for row in configuration["accounts"]["accounts"]}
    return {
        product["product_id"]: PaperAccountLedger(
            engine=engine,
            account_id=product["account_id"],
            product_id=product["product_id"],
            accounting_asset=product["base_accounting_asset"],
            starting_balances=accounts[product["account_id"]]["paper_starting_balances"],
            starting_positions=accounts[product["account_id"]].get("paper_starting_positions", {}),
        )
        for product in configuration["products"]["products"]
    }


@dataclass(frozen=True)
class PaperAccountState:
    account_id: str
    product_id: str
    observed_at: str
    balances: Mapping[str, float]
    positions: Mapping[str, float]
    position_entry_prices: Mapping[str, float] = field(default_factory=dict)
    last_fill_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "product_id": self.product_id,
            "observed_at": self.observed_at,
            "balances": dict(self.balances),
            "positions": dict(self.positions),
            "position_entry_prices": dict(self.position_entry_prices or {}),
            "last_fill_id": self.last_fill_id,
        }


class PaperAccountLedger:
    """Make one product ledger the durable authority for simulated balances."""

    def __init__(
        self,
        *,
        engine,
        account_id: str,
        product_id: str,
        accounting_asset: str,
        starting_balances: Mapping[str, Any],
        starting_positions: Mapping[str, Any] | None = None,
        connection=None,
        book_id: str | None = None,
    ) -> None:
        if not account_id or not product_id:
            raise ValueError("paper account identity is required")
        if not isinstance(starting_balances, Mapping) or not starting_balances:
            raise ValueError("paper account needs explicit starting balances")
        self.account_id = account_id
        if book_id is not None and book_id != account_id:
            raise ValueError("paper book balances require an isolated matching account")
        self.engine, self.connection, self.book_id = engine, connection, book_id
        self.product_id = product_id
        self.starting_balances = _balances(starting_balances)
        self.starting_positions = _positions(starting_positions or {})
        if accounting_asset.upper() == "BTC" and not self.starting_positions:
            btc_balance = self.starting_balances.get("BTC", 0.0)
            if btc_balance > 0.0:
                self.starting_positions = {"binance:spot:BTCUSDT": btc_balance}
        self.ledger = Ledger(
            product_id=product_id,
            accounting_asset=accounting_asset,
            store=SqlLedgerStore(engine, product_id=product_id, connection=connection),
        )

    def _reload(self) -> None:
        self.ledger = Ledger(
            product_id=self.product_id,
            accounting_asset=self.ledger.accounting_asset,
            store=self.ledger.store,
        )

    def ensure_seed(self, *, observed_at: str) -> PaperAccountState:
        self._reload()
        current = self.state()
        if current is not None:
            return current
        observed_at = timestamp(observed_at, field="paper seed timestamp")
        payload = {
            "kind": "paper_account_state",
            "book_id": self.book_id,
            "account_id": self.account_id,
            "product_id": self.product_id,
            "observed_at": observed_at,
            "balances": dict(self.starting_balances),
            "positions": dict(self.starting_positions),
            "last_fill_id": None,
            "seed": True,
        }
        try:
            self.ledger.append(
                entry_id=f"paper-seed:{self.account_id}:{self.product_id}",
                postings={"assets:paper_state": Decimal("0"), "equity:paper_state": Decimal("0")},
                metadata=payload,
                occurred_at=observed_at,
            )
        except (IntegrityError, ValueError):
            # Another process may have seeded the same product between the
            # read and append. Reloading gives the durable winner.
            self._reload()
        result = self.state()
        if result is None:
            raise RuntimeError("paper account seed was not persisted")
        return result

    def state(self, *, at: str | None = None) -> PaperAccountState | None:
        if at is not None:
            at = timestamp(at, field="paper state as-of time", preserve_subseconds=True)
        self.ledger.reload()
        entries = self.ledger.entries
        for entry in reversed(entries):
            metadata = entry.metadata
            if (
                metadata.get("kind") == "paper_account_state"
                and metadata.get("account_id") == self.account_id
                and metadata.get("product_id") == self.product_id
                and (at is None or metadata["observed_at"] <= at)
            ):
                state = _state_from_payload(metadata)
                if self.book_id is not None:
                    from src.portfolio.books import book_balance_snapshot

                    balances, observed_at = book_balance_snapshot(
                        self.engine, self.book_id, at=at, connection=self.connection
                    )
                    return replace(
                        state,
                        balances=balances,
                        observed_at=max(state.observed_at, observed_at or state.observed_at),
                    )
                return state
        return None

    def recorded_fill_ids(self) -> frozenset[str]:
        """Return every fill already represented by the paper account ledger."""

        return frozenset(
            str(entry.metadata["last_fill_id"])
            for entry in self.ledger.entries
            if entry.metadata.get("kind") == "paper_account_state"
            and entry.metadata.get("account_id") == self.account_id
            and entry.metadata.get("last_fill_id")
        )

    def record_fill(
        self,
        *,
        order: OrderIntent,
        fill: Fill,
        position: Position,
        previous_position: Position,
    ) -> PaperAccountState:
        self.ensure_seed(observed_at=fill.occurred_at)
        self._reload()
        state = self.state()
        if state is None:
            raise RuntimeError("paper account seed was not persisted")
        if state.last_fill_id == fill.fill_id or any(
            entry.metadata.get("last_fill_id") == fill.fill_id
            for entry in self.ledger.entries
            if entry.metadata.get("kind") == "paper_account_state"
            and entry.metadata.get("account_id") == self.account_id
        ):
            return state
        balances = dict(state.balances)
        if ":spot:" in order.instrument_id:
            self._apply_spot_fill(balances, order, fill)
        else:
            self._apply_futures_fill(balances, order, fill, previous_position)
        position_map = dict(state.positions)
        position_map[order.instrument_id] = float(position.quantity)
        payload = {
            "kind": "paper_account_state",
            "book_id": self.book_id,
            "account_id": self.account_id,
            "product_id": self.product_id,
            "observed_at": timestamp(
                fill.occurred_at, field="paper fill timestamp", preserve_subseconds=True
            ),
            "balances": balances,
            "positions": position_map,
            "position_entry_prices": {
                **dict(state.position_entry_prices or {}),
                order.instrument_id: float(position.average_entry_price),
            },
            "last_fill_id": fill.fill_id,
            "order_id": order.order_id,
            "instrument_id": order.instrument_id,
            "fill_hash": canonical_hash(fill),
            "strategy_version_ids": sorted(order.strategy_contributions),
            "assignment_id": _order_assignment_id(order),
        }
        try:
            self.ledger.append(
                entry_id=f"paper-state:{fill.fill_id}",
                postings={"assets:paper_state": Decimal("0"), "equity:paper_state": Decimal("0")},
                metadata=payload,
                occurred_at=fill.occurred_at,
            )
        except IntegrityError:
            self._reload()
            existing = self.state()
            if existing is None or existing.last_fill_id != fill.fill_id:
                raise
            return existing
        return _state_from_payload(payload)

    def account_payload(self, *, observed_at: str) -> dict[str, Any]:
        state = self.ensure_seed(observed_at=observed_at)
        return {
            "balances": dict(state.balances),
            "free_balances": dict(state.balances),
            "positions": dict(state.positions),
            "position_entry_prices": dict(state.position_entry_prices or {}),
            "regular_orders": [],
            "conditional_orders": [],
            "used_margin": 0.0,
            "maintenance_margin": 0.0,
            "used_margin_fraction": 0.0,
            "liquidation_buffer_fraction": 1.0,
            "account_mode": "paper",
            "unknown_exposure": {},
            "account_fingerprint": canonical_hash(
                {
                    "account_id": self.account_id,
                    "product_id": self.product_id,
                    "source": "paper_ledger",
                }
            ),
        }

    def _apply_spot_fill(self, balances: dict[str, float], order: OrderIntent, fill: Fill) -> None:
        base, quote = _spot_assets(order.instrument_id)
        quantity = float(fill.quantity)
        quote_value = quantity * float(fill.price)
        fee_asset = str(fill.fee_asset or quote).upper()
        base_delta = quantity if fill.side is OrderSide.BUY else -quantity
        quote_delta = -quote_value if fill.side is OrderSide.BUY else quote_value
        if fee_asset == base:
            base_delta -= float(fill.fee)
        elif fee_asset == quote:
            quote_delta -= float(fill.fee)
        else:
            balances[fee_asset] = balances.get(fee_asset, 0.0) - float(fill.fee)
        balances[base] = balances.get(base, 0.0) + base_delta
        balances[quote] = balances.get(quote, 0.0) + quote_delta
        _assert_nonnegative(balances)

    def _apply_futures_fill(
        self,
        balances: dict[str, float],
        order: OrderIntent,
        fill: Fill,
        previous_position: Position,
    ) -> None:
        quote = "USDT"
        signed = float(fill.quantity) if fill.side is OrderSide.BUY else -float(fill.quantity)
        closing = previous_position.quantity != 0 and (previous_position.quantity > 0) != (
            signed > 0
        )
        realised = 0.0
        if closing:
            quantity = min(abs(previous_position.quantity), abs(signed))
            entry_price = previous_position.average_entry_price
            reference_price = fill.price
            direction = 1.0 if previous_position.quantity > 0 else -1.0
            realised = quantity * (reference_price - entry_price) * direction
        balances[quote] = balances.get(quote, 0.0) + realised
        fee_asset = str(fill.fee_asset or quote).upper()
        balances[fee_asset] = balances.get(fee_asset, 0.0) - float(fill.fee)
        _assert_nonnegative(balances)


def _state_from_payload(payload: Mapping[str, Any]) -> PaperAccountState:
    return PaperAccountState(
        account_id=str(payload["account_id"]),
        product_id=str(payload["product_id"]),
        observed_at=timestamp(
            str(payload["observed_at"]), field="paper state observed_at", preserve_subseconds=True
        ),
        balances=_balances(payload["balances"]),
        positions=_positions(payload.get("positions", {})),
        position_entry_prices=_entry_prices(payload.get("position_entry_prices", {})),
        last_fill_id=(str(payload["last_fill_id"]) if payload.get("last_fill_id") else None),
    )


def _balances(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("paper balances must be an object")
    result = {str(key).upper(): float(amount) for key, amount in value.items()}
    _assert_nonnegative(result)
    return result


def _positions(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("paper positions must be an object")
    result = {str(key): float(amount) for key, amount in value.items()}
    if any(not math.isfinite(amount) for amount in result.values()):
        raise ValueError("paper positions must be finite")
    return result


def _entry_prices(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("paper position entry prices must be an object")
    result = {str(key): float(amount) for key, amount in value.items()}
    if any(not math.isfinite(amount) or amount < 0.0 for amount in result.values()):
        raise ValueError("paper position entry prices must be finite and non-negative")
    return result


def _assert_nonnegative(values: Mapping[str, float]) -> None:
    if any(not math.isfinite(float(value)) or float(value) < -1e-12 for value in values.values()):
        raise ValueError("paper account balance became negative")


def _spot_assets(instrument_id: str) -> tuple[str, str]:
    parts = instrument_id.split(":")
    symbol = parts[2] if len(parts) >= 3 else instrument_id
    for quote in ("USDT", "USDC", "BUSD", "BTC", "ETH"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[: -len(quote)], quote
    return symbol, "USDT"


def _order_assignment_id(order: OrderIntent) -> str | None:
    target_metadata = order.metadata.get("target_metadata")
    if isinstance(target_metadata, Mapping) and target_metadata.get("assignment_id"):
        return str(target_metadata["assignment_id"])
    if order.metadata.get("assignment_id"):
        return str(order.metadata["assignment_id"])
    return None
