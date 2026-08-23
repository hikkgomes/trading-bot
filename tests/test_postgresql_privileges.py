from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL roles",
)
def test_privilege_fixture_is_explicitly_enabled() -> None:
    # Role grants are applied by the owner Alembic revision. This test remains
    # opt-in because local developer databases do not normally contain roles.
    assert os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") == "1"
