"""Immutable initial capital and protected BTC reserve for one account/product."""

from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.data.database import capital_baseline
from src.domain._codec import canonical_hash, timestamp


def btc_capital_baseline(
    engine,
    *,
    product_id: str,
    account_id: str,
    mode: str,
    state_id: str,
    observed_at: str,
    btc: float,
    quote: float,
    price: float,
    reserve_fraction: float,
) -> Mapping[str, Any]:
    identity = canonical_hash({"product_id": product_id, "account_id": account_id, "mode": mode})
    with engine.begin() as connection:
        existing = connection.execute(
            select(capital_baseline.c.payload).where(capital_baseline.c.id == identity)
        ).scalar_one_or_none()
        if existing is not None:
            return existing
        if mode != "paper":
            raise ValueError("live BTC capital baseline requires an approved capital event")
        initial = btc + quote / price
        payload = {
            "product_id": product_id,
            "account_id": account_id,
            "execution_mode": mode,
            "initial_btc": btc,
            "initial_quote": quote,
            "initial_price": price,
            "initial_btc_equivalent": initial,
            "core_btc_quantity": initial * reserve_fraction,
            "reserve_fraction": reserve_fraction,
            "source_state_id": state_id,
        }
        insert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
        connection.execute(
            insert(capital_baseline)
            .values(
                id=identity,
                created_at=timestamp(observed_at, field="capital observed_at"),
                payload=payload,
            )
            .on_conflict_do_nothing(index_elements=[capital_baseline.c.id])
        )
        return connection.execute(
            select(capital_baseline.c.payload).where(capital_baseline.c.id == identity)
        ).scalar_one()
