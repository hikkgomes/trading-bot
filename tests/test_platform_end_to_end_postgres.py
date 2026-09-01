from __future__ import annotations

import datetime as dt
import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import insert, select

from src.data.database import (
    PlatformDatabase,
    cost_model_manifest,
    dataset_snapshot,
    feature_manifest,
)
from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.domain._codec import canonical_hash
from src.domain.instruments import Instrument, MarketType
from src.research.dataset_service import DatabaseDatasetBundleService
from src.research.datasets import CORE_RESEARCH_BUNDLE_ROLES, SqlDatasetBundleRepository


@pytest.mark.skipif(
    not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith("postgresql"),
    reason="requires a PostgreSQL platform fixture",
)
def test_platform_postgres_schema_is_migrated() -> None:
    database = PlatformDatabase(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    database.assert_migrated()
    database.dispose()


@pytest.mark.skipif(
    not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith("postgresql"),
    reason="requires a PostgreSQL platform fixture",
)
def test_platform_postgres_builds_idempotent_bundle_from_real_parquet(tmp_path: Path) -> None:
    run_id = uuid.uuid4().hex
    database = PlatformDatabase(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    database.migrate()
    instrument = Instrument(
        venue="binance",
        market_type=MarketType.SPOT,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset=None,
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5.0,
    )
    now = "2026-08-30T10:00:00+00:00"
    universe_id = f"postgres-e2e-{run_id}"
    product_id = "btc_accumulation"
    feature_id = canonical_hash(
        {"schema": "platform.feature_manifest/v1", "market_type": "spot", "run_id": run_id}
    )
    cost_id = canonical_hash(
        {"schema": "platform.cost_model/v1", "product_id": product_id, "run_id": run_id}
    )
    try:
        SqlUniverseStore(database.engine).record_snapshot(
            universe_id=universe_id,
            observed_at=now,
            observations=(
                InstrumentObservation(
                    instrument=instrument,
                    listing_age_days=365.0,
                    quote_volume=1_000_000_000.0,
                    trade_count=1_000_000,
                    spread_bps=1.0,
                    open_interest=0.0,
                    funding_rate=0.0,
                    realised_volatility=0.2,
                    depth_notional=10_000_000.0,
                    data_completeness=1.0,
                ),
            ),
            policy=UniverseEligibilityPolicy(),
        )
        with database.engine.begin() as connection:
            connection.execute(
                insert(feature_manifest).values(
                    id=feature_id,
                    created_at=now,
                    payload={
                        "schema": "platform.feature_manifest/v1",
                        "market_type": "spot",
                        "run_id": run_id,
                    },
                )
            )
            connection.execute(
                insert(cost_model_manifest).values(
                    id=cost_id,
                    created_at=now,
                    payload={
                        "schema": "platform.cost_model/v1",
                        "product_id": product_id,
                        "run_id": run_id,
                    },
                )
            )
        partition = tmp_path / "bars" / "binance" / "spot" / "BTCUSDT" / "1m" / "date=2026-08-20"
        partition.mkdir(parents=True)
        close_times = [
            int(dt.datetime(2026, 8, 20 + index, tzinfo=dt.UTC).timestamp() * 1_000)
            for index in range(8)
        ]
        pq.write_table(
            pa.table(
                {
                    "event_id": [f"{run_id}-{index}" for index in range(8)],
                    "instrument_id": [instrument.instrument_id] * 8,
                    "close_time_ms": close_times,
                    "availability_time": [now] * 8,
                    "open": [100.0 + index for index in range(8)],
                    "high": [101.0 + index for index in range(8)],
                    "low": [99.0 + index for index in range(8)],
                    "close": [100.0 + index for index in range(8)],
                    "volume": [10.0] * 8,
                }
            ),
            partition / "bars.parquet",
        )
        service = DatabaseDatasetBundleService(database.engine, tmp_path)
        first = service.run(
            product_id=product_id,
            universe_id=universe_id,
            market_type="spot",
            created_at=now,
        )
        second = service.run(
            product_id=product_id,
            universe_id=universe_id,
            market_type="spot",
            created_at=now,
        )

        assert first.state == second.state == "ready"
        assert first.bundle_id == second.bundle_id
        assert first.bundle_id is not None
        bundle = SqlDatasetBundleRepository(database.engine).get(first.bundle_id)
        assert set(bundle.stage_snapshot_ids) == set(CORE_RESEARCH_BUNDLE_ROLES)
        with database.engine.connect() as connection:
            payload = connection.execute(
                select(dataset_snapshot.c.payload).where(
                    dataset_snapshot.c.id == bundle.stage_snapshot_ids["development"]
                )
            ).scalar_one()
        assert payload["payload"]["market_frame"]
        assert payload["payload"]["data_quality"]["rows"] > 0
    finally:
        database.dispose()
