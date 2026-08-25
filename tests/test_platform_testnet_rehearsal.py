from __future__ import annotations

import pytest

from src.services.platform_testnet_rehearsal import (
    PlatformTestnetRehearsal,
    validate_testnet_rehearsal_configuration,
)


class _Worker:
    def __init__(self, reason_code: str):
        self.reason_code = reason_code

    def run_once(self, *, now: str):
        return {"reason_code": self.reason_code, "observed_at": now}


def test_platform_testnet_rehearsal_runs_live_user_stream_accounting_and_recovery() -> None:
    pending = {"user": True, "accounting": True, "recovery": True}

    def consume(name: str):
        if not pending[name]:
            return False
        pending[name] = False
        return True

    rehearsal = PlatformTestnetRehearsal(
        active_assignment=_Worker("active_assignment_loaded"),
        strategy_evaluator=_Worker("strategy_evaluation_recorded"),
        portfolio=_Worker("portfolio_target_created"),
        risk=_Worker("risk_decision_recorded"),
        live_execution=_Worker("live_order_acknowledged"),
        user_stream=_Worker("user_stream_event_recorded"),
        accounting=_Worker("accounting_event_recorded"),
        recovery=_Worker("live_recovery_plan_created"),
        has_pending_user_stream=lambda: consume("user"),
        has_pending_accounting=lambda: consume("accounting"),
        has_pending_recovery=lambda: consume("recovery"),
    )

    report = rehearsal.run(now="2026-08-24T00:00:00+00:00")

    assert report.ok
    assert [stage["stage"] for stage in report.stages] == [
        "active_assignment",
        "strategy_evaluation",
        "portfolio_target",
        "risk_decision",
        "live_order_submission",
        "user_stream",
        "accounting",
        "recovery",
    ]


def test_testnet_rehearsal_requires_explicit_testnet_environment() -> None:
    assert (
        validate_testnet_rehearsal_configuration(
            {"environment": "testnet", "product_id": "active_income"}
        )["strategy_evaluator"]
        == "DatabaseStrategyEvaluator"
    )


def test_platform_testnet_rehearsal_requires_postgresql_platform_queue() -> None:
    with pytest.raises(ValueError, match="PostgreSQL queue"):
        validate_testnet_rehearsal_configuration(
            {
                "environment": "testnet",
                "product_id": "active_income",
                "queue_backend": "sqlite",
            }
        )
    with pytest.raises(ValueError, match="legacy autopilot"):
        validate_testnet_rehearsal_configuration(
            {
                "environment": "testnet",
                "product_id": "active_income",
                "legacy_autopilot": True,
            }
        )
