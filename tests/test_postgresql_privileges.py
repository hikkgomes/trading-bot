from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError


@pytest.mark.skipif(
    not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL roles",
)
def test_privilege_fixture_is_explicitly_enabled() -> None:
    # Role grants are applied by the owner Alembic revision. This test remains
    # opt-in because local developer databases do not normally contain roles.
    assert os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") == "1"


def _as_role(role: str, statement: str, parameters: dict | None = None):
    engine = create_engine(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    try:
        with engine.begin() as connection:
            connection.execute(text(f"SET LOCAL ROLE {role}"))
            result = connection.execute(text(statement), parameters or {})
            return result.all() if result.returns_rows else []
    finally:
        engine.dispose()


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="requires PostgreSQL role fixture",
)
def test_every_platform_role_has_only_its_declared_authority() -> None:
    run_id = uuid.uuid4().hex
    research_job_id = f"role-test-research-job-{run_id}"
    agent_proposal_id = f"role-test-agent-proposal-{run_id}"
    _as_role("trading_runtime", "SELECT id FROM balance_snapshot LIMIT 1")
    _as_role("trading_research", "SELECT id FROM instrument LIMIT 1")
    with pytest.raises(DBAPIError):
        _as_role("trading_research", "SELECT id FROM balance_snapshot LIMIT 1")
    with pytest.raises(DBAPIError):
        _as_role(
            "trading_research",
            "INSERT INTO job (id, name, state, priority, available_at, attempts, "
            "producer_identity, content_hash, payload) VALUES "
            "('forbidden', 'evaluate_candidate', 'queued', 0, CURRENT_TIMESTAMP, 0, "
            "'research', 'sha256:' || repeat('1', 64), '{}'::jsonb)",
        )
    rows = _as_role(
        "trading_research",
        "SELECT submit_typed_research_job(:id, 'evaluate_candidate', '{}'::jsonb, "
        "CURRENT_TIMESTAMP, 0, 'research:test', :hash)",
        {"id": research_job_id, "hash": "sha256:" + "1" * 64},
    )
    assert rows == [(research_job_id,)]

    _as_role(
        "trading_agent",
        "INSERT INTO agent_proposal (id, created_at, payload) "
        "VALUES (:id, CURRENT_TIMESTAMP, '{}'::jsonb)",
        {"id": agent_proposal_id},
    )
    with pytest.raises(DBAPIError):
        _as_role("trading_agent", "SELECT id FROM strategy_approval LIMIT 1")
    with pytest.raises(DBAPIError):
        _as_role(
            "trading_agent",
            "SELECT submit_typed_research_job(:id, 'live_order_submit', '{}'::jsonb, "
            "CURRENT_TIMESTAMP, 0, 'agent:test', :hash)",
            {"id": "role-test-forbidden-job", "hash": "sha256:" + "2" * 64},
        )

    with pytest.raises(DBAPIError):
        _as_role("trading_platform_owner", "SELECT id FROM balance_snapshot LIMIT 1")
