from __future__ import annotations

import pytest
from sqlalchemy import select

from src.data.database import PlatformDatabase, job
from src.services.control_api import ControlMode, DatabaseControlPlane
from src.services.health import DatabaseHeartbeatStore

NOW = "2026-08-31T10:00:00+00:00"
LATER = "2026-08-31T10:01:00+00:00"


def _control(tmp_path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'control.sqlite3'}")
    database.create_schema()
    return database, DatabaseControlPlane(database.engine, DatabaseHeartbeatStore(database.engine))


def test_global_risk_block_does_not_pause_critical_reconciliation_services(tmp_path) -> None:
    database, control = _control(tmp_path)
    try:
        control.set_mode(
            target="global",
            mode=ControlMode.BLOCK_NEW_RISK,
            reason_code="risk_review",
            requested_by="operator",
            changed_at=NOW,
        )
        assert control.effective_mode(product_id="active_income") is ControlMode.BLOCK_NEW_RISK
        assert control.blocks_new_risk(product_id="active_income") is True
        assert control.service_is_paused("market-gateway") is False
        assert control.service_is_paused("account-reconciliation") is False
        assert control.service_is_paused("live-execution") is False
    finally:
        database.dispose()


def test_strategy_suspend_has_precedence_and_resume_requires_confirmation(tmp_path) -> None:
    database, control = _control(tmp_path)
    try:
        control.set_mode(
            target="product:active_income",
            mode=ControlMode.BLOCK_NEW_RISK,
            reason_code="product_review",
            requested_by="operator",
            changed_at=NOW,
        )
        control.set_mode(
            target="strategy:strategy-1",
            mode=ControlMode.SUSPENDED,
            reason_code="strategy_fault",
            requested_by="operator",
            changed_at=LATER,
        )
        assert control.effective_mode(product_id="active_income", strategy_id="strategy-1") is ControlMode.SUSPENDED
        with pytest.raises(PermissionError, match="confirmation"):
            control.set_mode(
                target="global",
                mode=ControlMode.RUN,
                reason_code="resume",
                requested_by="operator",
                changed_at=LATER,
            )
        state = control.set_mode(
            target="global",
            mode=ControlMode.RUN,
            reason_code="resume",
            requested_by="operator",
            changed_at=LATER,
            confirm_resume=True,
        )
        assert state.mode == ControlMode.RUN.value
    finally:
        database.dispose()


def test_emergency_flatten_control_enqueues_durable_command(tmp_path) -> None:
    database, control = _control(tmp_path)
    try:
        control.set_mode(
            target="product:active_income",
            mode=ControlMode.EMERGENCY_FLATTEN,
            reason_code="stop_failure",
            requested_by="operator",
            changed_at=NOW,
        )
        with database.engine.connect() as connection:
            rows = connection.execute(select(job.c.name, job.c.state)).all()
        assert rows == [("emergency_flatten", "pending")]
    finally:
        database.dispose()
