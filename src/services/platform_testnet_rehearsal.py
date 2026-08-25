"""Platform-service testnet rehearsal orchestration.

The rehearsal is deliberately built from the same durable live-submission,
user-stream, accounting, and recovery workers used by the platform.  A broker
adapter and captured testnet events are injected by the runner, so this module
does not create an alternate execution path or place an order implicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from src.domain._codec import non_empty, timestamp


class RehearsalWorker(Protocol):
    def run_once(self, *, now: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PlatformTestnetRehearsalReport:
    schema: str
    ok: bool
    stages: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ok": self.ok,
            "stages": [dict(stage) for stage in self.stages],
        }


class PlatformTestnetRehearsal:
    """Run a bounded live-order, user-stream, accounting, and recovery cycle."""

    def __init__(
        self,
        *,
        active_assignment: RehearsalWorker,
        strategy_evaluator: RehearsalWorker,
        portfolio: RehearsalWorker,
        risk: RehearsalWorker,
        live_execution: RehearsalWorker,
        user_stream: RehearsalWorker,
        accounting: RehearsalWorker,
        recovery: RehearsalWorker,
        has_pending_user_stream: Callable[[], bool],
        has_pending_accounting: Callable[[], bool],
        has_pending_recovery: Callable[[], bool],
    ) -> None:
        self.active_assignment = active_assignment
        self.strategy_evaluator = strategy_evaluator
        self.portfolio = portfolio
        self.risk = risk
        self.live_execution = live_execution
        self.user_stream = user_stream
        self.accounting = accounting
        self.recovery = recovery
        self.has_pending_user_stream = has_pending_user_stream
        self.has_pending_accounting = has_pending_accounting
        self.has_pending_recovery = has_pending_recovery

    def run(self, *, now: str) -> PlatformTestnetRehearsalReport:
        now = timestamp(now, field="now")
        stages: list[Mapping[str, Any]] = []
        for name, worker in (
            ("active_assignment", self.active_assignment),
            ("strategy_evaluation", self.strategy_evaluator),
            ("portfolio_target", self.portfolio),
            ("risk_decision", self.risk),
        ):
            stages.append({"stage": name, **dict(worker.run_once(now=now))})
        stages.append(
            {"stage": "live_order_submission", **dict(self.live_execution.run_once(now=now))}
        )
        while self.has_pending_user_stream():
            stages.append({"stage": "user_stream", **dict(self.user_stream.run_once(now=now))})
        while self.has_pending_accounting():
            stages.append({"stage": "accounting", **dict(self.accounting.run_once(now=now))})
        while self.has_pending_recovery():
            stages.append({"stage": "recovery", **dict(self.recovery.run_once(now=now))})
        required = {
            "active_assignment",
            "strategy_evaluation",
            "portfolio_target",
            "risk_decision",
            "live_order_submission",
            "user_stream",
            "accounting",
            "recovery",
        }
        completed = {str(stage.get("stage")) for stage in stages}
        ok = required.issubset(completed) and all(
            str(stage.get("reason_code", "")).endswith(
                ("acknowledged", "recorded", "created", "loaded", "filled")
            )
            or str(stage.get("reason_code", "")).endswith("filled")
            or str(stage.get("reason_code", "")).endswith("recovery_plan_created")
            for stage in stages
        )
        return PlatformTestnetRehearsalReport(
            schema="platform.testnet-rehearsal/v1",
            ok=ok,
            stages=tuple(stages),
        )


def validate_testnet_rehearsal_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a PostgreSQL-backed platform testnet rehearsal declaration."""

    environment = non_empty(str(configuration.get("environment") or ""), field="environment")
    if environment != "testnet":
        raise ValueError("platform testnet rehearsal requires environment=testnet")
    queue_backend = str(configuration.get("queue_backend") or "postgresql")
    if queue_backend != "postgresql":
        raise ValueError("platform testnet rehearsal requires the PostgreSQL queue")
    if configuration.get("legacy_autopilot") is True:
        raise ValueError("platform testnet rehearsal cannot use the legacy autopilot path")
    product_id = non_empty(str(configuration.get("product_id") or ""), field="product_id")
    return {
        "schema": "platform.testnet-rehearsal-config/v1",
        "environment": environment,
        "queue_backend": queue_backend,
        "legacy_autopilot": False,
        "product_id": product_id,
        "active_assignment": "SqlActiveStrategyAssignmentRepository",
        "strategy_evaluator": "DatabaseStrategyEvaluator",
        "portfolio": "DatabasePortfolioTargetWorker",
        "risk": "DatabaseRiskWorker",
        "order_submission": "DatabaseLiveExecutionWorker",
        "user_stream": "DatabaseUserStreamWorker",
        "accounting": "DatabaseAccountingWorker",
        "recovery": "DatabaseLiveRecoveryWorker",
    }
