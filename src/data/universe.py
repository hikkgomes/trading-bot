"""Point-in-time dynamic instrument-universe snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import (
    instrument as instrument_table,
)
from src.data.database import (
    instrument_status,
    universe,
    universe_member,
    universe_snapshot,
)
from src.domain._codec import canonical_hash, timestamp, to_primitive
from src.domain.instruments import Instrument, MarketType


@dataclass(frozen=True)
class UniverseEligibilityPolicy:
    minimum_listing_age_days: float = 30.0
    minimum_quote_volume: float = 10_000_000.0
    minimum_trade_count: int = 1_000
    maximum_spread_bps: float = 20.0
    minimum_open_interest: float = 1_000_000.0
    maximum_absolute_funding_rate: float = 0.01
    minimum_realised_volatility: float = 0.0
    maximum_realised_volatility: float = 3.0
    minimum_depth_notional: float = 100_000.0
    minimum_data_completeness: float = 0.99


@dataclass(frozen=True)
class InstrumentObservation:
    instrument: Instrument
    listing_age_days: float
    quote_volume: float
    trade_count: int
    spread_bps: float
    open_interest: float
    funding_rate: float
    realised_volatility: float
    depth_notional: float
    data_completeness: float
    strategy_eligibility: tuple[str, ...] = ()

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "listing_age_days": self.listing_age_days,
            "quote_volume": self.quote_volume,
            "trade_count": self.trade_count,
            "spread_bps": self.spread_bps,
            "open_interest": self.open_interest,
            "funding_rate": self.funding_rate,
            "realised_volatility": self.realised_volatility,
            "depth_notional": self.depth_notional,
            "data_completeness": self.data_completeness,
        }


@dataclass(frozen=True)
class UniverseMembership:
    snapshot_id: str
    observed_at: str
    instrument: Instrument
    eligible: bool
    reason_code: str
    metrics: Mapping[str, Any]
    strategy_eligibility: tuple[str, ...]


def eligibility_reason(
    observation: InstrumentObservation,
    policy: UniverseEligibilityPolicy,
) -> str:
    instrument = observation.instrument
    checks = (
        (instrument.status != "trading", "listing_not_trading"),
        (instrument.market_type is not MarketType.FUTURES, "not_futures"),
        (instrument.quote_asset != "USDT", "not_usdt_quoted"),
        (observation.listing_age_days < policy.minimum_listing_age_days, "listing_too_new"),
        (observation.quote_volume < policy.minimum_quote_volume, "quote_volume_too_low"),
        (observation.trade_count < policy.minimum_trade_count, "trade_count_too_low"),
        (observation.spread_bps > policy.maximum_spread_bps, "spread_too_wide"),
        (observation.open_interest < policy.minimum_open_interest, "open_interest_too_low"),
        (
            abs(observation.funding_rate) > policy.maximum_absolute_funding_rate,
            "funding_out_of_bounds",
        ),
        (
            not policy.minimum_realised_volatility
            <= observation.realised_volatility
            <= policy.maximum_realised_volatility,
            "volatility_out_of_bounds",
        ),
        (observation.depth_notional < policy.minimum_depth_notional, "depth_too_low"),
        (
            observation.data_completeness < policy.minimum_data_completeness,
            "data_incomplete",
        ),
    )
    return next((reason for failed, reason in checks if failed), "eligible")


class SqlUniverseStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def record_snapshot(
        self,
        *,
        universe_id: str,
        observed_at: str,
        observations: Iterable[InstrumentObservation],
        policy: UniverseEligibilityPolicy,
    ) -> str:
        observed_at = timestamp(observed_at, field="observed_at")
        materialised = tuple(sorted(observations, key=lambda item: item.instrument.instrument_id))
        if not materialised:
            raise ValueError("universe snapshot requires at least one instrument")
        if len({item.instrument.instrument_id for item in materialised}) != len(materialised):
            raise ValueError("universe snapshot contains duplicate instruments")
        snapshot_payload = {
            "universe_id": universe_id,
            "observed_at": observed_at,
            "policy": to_primitive(policy),
            "observations": [
                {
                    "instrument": to_primitive(item.instrument),
                    "metrics": item.metrics,
                    "strategy_eligibility": list(item.strategy_eligibility),
                    "reason_code": eligibility_reason(item, policy),
                }
                for item in materialised
            ],
        }
        content_hash = canonical_hash(snapshot_payload)
        snapshot_id = canonical_hash({"universe_id": universe_id, "content_hash": content_hash})
        with self.engine.begin() as connection:
            if (
                connection.execute(
                    select(universe.c.id).where(universe.c.id == universe_id)
                ).first()
                is None
            ):
                connection.execute(
                    insert(universe).values(
                        id=universe_id,
                        created_at=observed_at,
                        payload={"dynamic": True, "fixed_maximum": None},
                    )
                )
            existing = connection.execute(
                select(universe_snapshot.c.id).where(universe_snapshot.c.id == snapshot_id)
            ).first()
            if existing is not None:
                return snapshot_id
            connection.execute(
                insert(universe_snapshot).values(
                    id=snapshot_id,
                    universe_id=universe_id,
                    observed_at=observed_at,
                    content_hash=content_hash,
                    payload=snapshot_payload,
                )
            )
            for item in materialised:
                self._record_instrument(connection, item.instrument, observed_at=observed_at)
                reason = eligibility_reason(item, policy)
                connection.execute(
                    insert(universe_member).values(
                        id=f"{snapshot_id}:{item.instrument.instrument_id}",
                        snapshot_id=snapshot_id,
                        instrument_id=item.instrument.instrument_id,
                        eligible=reason == "eligible",
                        reason_code=reason,
                        payload={
                            "instrument": to_primitive(item.instrument),
                            "metrics": item.metrics,
                            "strategy_eligibility": list(item.strategy_eligibility),
                        },
                    )
                )
        return snapshot_id

    @staticmethod
    def _record_instrument(connection, item: Instrument, *, observed_at: str) -> None:
        existing = connection.execute(
            select(instrument_table.c.payload).where(instrument_table.c.id == item.instrument_id)
        ).scalar_one_or_none()
        payload = to_primitive(item)
        if existing is None:
            connection.execute(
                insert(instrument_table).values(
                    id=item.instrument_id,
                    venue=item.venue,
                    market_type=item.market_type.value,
                    exchange_symbol=item.exchange_symbol,
                    base_asset=item.base_asset,
                    quote_asset=item.quote_asset,
                    settlement_asset=item.settlement_asset,
                    payload=payload,
                )
            )
        status_id = canonical_hash(
            {"instrument_id": item.instrument_id, "observed_at": observed_at, "status": item.status}
        )
        if (
            connection.execute(
                select(instrument_status.c.id).where(instrument_status.c.id == status_id)
            ).first()
            is None
        ):
            connection.execute(
                insert(instrument_status).values(
                    id=status_id,
                    instrument_id=item.instrument_id,
                    observed_at=observed_at,
                    status=item.status,
                    payload={},
                )
            )

    def members_at(
        self,
        *,
        universe_id: str,
        observed_at: str,
        eligible_only: bool = True,
    ) -> tuple[UniverseMembership, ...]:
        observed_at = timestamp(observed_at, field="observed_at")
        with self.engine.connect() as connection:
            snapshot_row = (
                connection.execute(
                    select(universe_snapshot)
                    .where(
                        universe_snapshot.c.universe_id == universe_id,
                        universe_snapshot.c.observed_at <= observed_at,
                    )
                    .order_by(universe_snapshot.c.observed_at.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if snapshot_row is None:
                return ()
            statement = select(universe_member).where(
                universe_member.c.snapshot_id == snapshot_row["id"]
            )
            if eligible_only:
                statement = statement.where(universe_member.c.eligible.is_(True))
            rows = connection.execute(
                statement.order_by(universe_member.c.instrument_id)
            ).mappings()
            memberships: list[UniverseMembership] = []
            for row in rows:
                payload = dict(row["payload"])
                instrument_payload = dict(payload["instrument"])
                instrument_payload["market_type"] = MarketType(instrument_payload["market_type"])
                memberships.append(
                    UniverseMembership(
                        snapshot_id=snapshot_row["id"],
                        observed_at=snapshot_row["observed_at"],
                        instrument=Instrument(**instrument_payload),
                        eligible=bool(row["eligible"]),
                        reason_code=str(row["reason_code"]),
                        metrics=dict(payload["metrics"]),
                        strategy_eligibility=tuple(payload["strategy_eligibility"]),
                    )
                )
            return tuple(memberships)
