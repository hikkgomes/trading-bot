from __future__ import annotations

import os

import pytest

from src.data.database import PlatformDatabase


@pytest.mark.skipif(
    not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith("postgresql"),
    reason="requires a PostgreSQL platform fixture",
)
def test_platform_postgres_schema_is_migrated() -> None:
    database = PlatformDatabase(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    database.assert_migrated()
    database.dispose()
