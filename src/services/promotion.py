"""Deterministic strategy lifecycle and promotion policy."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from enum import StrEnum
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import (
    experiment,
    forward_paper_decision,
    forward_paper_summary,
    holdout_claim,
    holdout_outcome,
    production_preflight,
    promotion_event,
    promotion_policy,
    strategy_approval,
    strategy_artefact,
    validation_stage,
)
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.research.canonical import SqlActiveStrategyAssignmentRepository, preflight_is_fresh
from src.research.objectives import objective_unit
from src.services.scheduler import DatabaseJobQueue


class LifecycleState(StrEnum):
    REGISTERED = "registered"
    DEVELOPMENT = "development"
    FORWARD_PAPER = "forward_paper"
    LIVE_READY = "live_ready"
    LIVE_CANARY = "live_canary"
    LIVE = "live"
    SUSPENDED = "suspended"
    RETIRED = "retired"


@dataclass(frozen=True)
class PromotionRequest:
    """Identity-only request accepted by the promotion queue."""

    strategy_version_id: str
    requested_transition: str
    requested_capital: float
    evaluated_at: str
    instrument_id: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PromotionRequest:
        if not isinstance(payload, Mapping):
            raise ValueError("promotion request must be an object")
        allowed = {
            "strategy_version_id",
            "requested_transition",
            "requested_capital",
            "evaluated_at",
            "instrument_id",
        }
        forbidden = sorted(
            set(payload)
            & {"evidence", "policy", "accepted", "validation", "outcome", "current_state"}
        )
        if forbidden:
            raise ValueError(
                "submitted promotion policy/evidence is not accepted: " + ", ".join(forbidden)
            )
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("promotion request contains unsupported fields: " + ", ".join(unknown))
        strategy_version_id = non_empty(
            str(payload.get("strategy_version_id") or ""), field="strategy_version_id"
        )
        transition = non_empty(
            str(payload.get("requested_transition") or ""), field="requested_transition"
        )
        if transition not in {
            "forward_paper",
            "live_ready",
            "live_canary",
            "live",
            "suspend",
            "retire",
            "resume",
        }:
            raise ValueError("requested_transition is not a supported lifecycle transition")
        raw_capital = payload.get("requested_capital")
        if isinstance(raw_capital, bool) or not isinstance(raw_capital, int | float | str):
            raise ValueError("requested_capital must be numeric")
        try:
            capital = float(raw_capital)
        except (TypeError, ValueError) as exc:
            raise ValueError("requested_capital must be numeric") from exc
        if not math.isfinite(capital) or capital < 0:
            raise ValueError("requested_capital must be finite and non-negative")
        return cls(
            strategy_version_id=strategy_version_id,
            requested_transition=transition,
            requested_capital=capital,
            evaluated_at=timestamp(str(payload.get("evaluated_at") or ""), field="evaluated_at"),
            instrument_id=(
                non_empty(str(payload["instrument_id"]), field="instrument_id")
                if payload.get("instrument_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PromotionPolicy:
    automatic_paper_promotion: bool
    automatic_live_canary_promotion: bool
    canary_capital_limit: float
    required_forward_evidence_days: int
    maximum_drawdown: float
    maximum_execution_drift: float
    maximum_model_drift: float
    minimum_forward_independent_decisions: int = 1
    minimum_forward_net_pnl: float = 0.0
    minimum_forward_objective_excess_fraction: float = 0.0
    maximum_forward_data_gaps: int = 0
    automatic_live_ready_promotion: bool = True
    paper_capital_limit: float = 1.0
    minimum_forward_effective_trades: int = 0
    minimum_forward_fill_rate: float = 0.0
    maximum_forward_slippage: float = 1.0
    minimum_forward_data_uptime: float = 0.0
    maximum_forward_rejected_orders: int = 0


@dataclass(frozen=True)
class PromotionEvidence:
    strategy_artefact_hash: str
    source_commit_hash: str
    validation_accepted: bool
    protected_holdout_accepted: bool
    forward_evidence_days: int
    forward_evidence_accepted: bool
    drawdown: float
    execution_drift: float
    model_drift: float
    portfolio_capacity: float
    requested_capital: float
    risk_budget_available: float
    live_approval: bool
    fresh_preflight: bool
    product_id: str | None = None
    account_id: str | None = None
    artefact_record_id: str | None = None
    preflight_record_id: str | None = None
    portfolio_id: str | None = None
    market_making: bool = False
    market_making_live_capability: bool = False
    event_replay_passed: bool = False
    event_replay_fills: int = 0
    forward_summary_id: str | None = None
    forward_decision_id: str | None = None
    forward_independent_decisions: int = 0
    forward_net_pnl: float = 0.0
    forward_benchmark_pnl: float = 0.0
    forward_excess_benchmark_pnl: float = 0.0
    forward_objective_unit: str | None = None
    forward_objective_value: float | None = None
    forward_benchmark_value: float | None = None
    forward_objective_excess: float | None = None
    forward_objective_excess_fraction: float | None = None
    forward_data_gaps: int = 0
    canary_evidence_accepted: bool = False
    forward_effective_trades: int = 0
    forward_fill_rate: float = 1.0
    forward_slippage: float = 0.0
    forward_data_uptime: float = 0.0
    forward_rejected_orders: int = 0
    supported_instruments: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromotionDecision:
    strategy_version_id: str
    prior_state: LifecycleState
    next_state: LifecycleState
    accepted: bool
    reason_code: str
    evaluated_at: str
    capital_limit: float
    evidence_hash: str


class SqlPromotionPolicyStore:
    """Authoritative promotion-policy records stored in PostgreSQL."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def put(self, policy_id: str, policy: PromotionPolicy, *, created_at: str) -> str:
        payload = json_value(
            {
                "policy_id": policy_id,
                "automatic_paper_promotion": policy.automatic_paper_promotion,
                "automatic_live_canary_promotion": policy.automatic_live_canary_promotion,
                "canary_capital_limit": policy.canary_capital_limit,
                "required_forward_evidence_days": policy.required_forward_evidence_days,
                "maximum_drawdown": policy.maximum_drawdown,
                "maximum_execution_drift": policy.maximum_execution_drift,
                "maximum_model_drift": policy.maximum_model_drift,
                "minimum_forward_independent_decisions": policy.minimum_forward_independent_decisions,
                "minimum_forward_net_pnl": policy.minimum_forward_net_pnl,
                "minimum_forward_objective_excess_fraction": policy.minimum_forward_objective_excess_fraction,
                "maximum_forward_data_gaps": policy.maximum_forward_data_gaps,
                "automatic_live_ready_promotion": policy.automatic_live_ready_promotion,
                "paper_capital_limit": policy.paper_capital_limit,
                "minimum_forward_effective_trades": policy.minimum_forward_effective_trades,
                "minimum_forward_fill_rate": policy.minimum_forward_fill_rate,
                "maximum_forward_slippage": policy.maximum_forward_slippage,
                "minimum_forward_data_uptime": policy.minimum_forward_data_uptime,
                "maximum_forward_rejected_orders": policy.maximum_forward_rejected_orders,
            },
            field="promotion policy",
        )
        identity = canonical_hash(payload)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(promotion_policy.c.payload).where(promotion_policy.c.id == policy_id)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != payload:
                    raise ValueError(f"promotion policy identity collision: {policy_id}")
                return identity
            connection.execute(
                insert(promotion_policy).values(
                    id=policy_id,
                    created_at=timestamp(created_at, field="created_at"),
                    payload=payload,
                )
            )
        return identity

    def get(self, policy_id: str) -> PromotionPolicy:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(promotion_policy.c.payload).where(promotion_policy.c.id == policy_id)
            ).scalar_one_or_none()
        if payload is None:
            raise KeyError(f"promotion policy does not exist: {policy_id}")
        fields = PromotionPolicy.__dataclass_fields__
        values = {
            key: payload[key]
            if field.default is MISSING and field.default_factory is MISSING
            else payload.get(key, field.default)
            for key, field in fields.items()
        }
        return PromotionPolicy(**values)

    def first(self) -> PromotionPolicy:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(promotion_policy.c.payload).order_by(promotion_policy.c.created_at).limit(1)
            ).scalar_one_or_none()
        if payload is None:
            raise KeyError("no authoritative promotion policy exists")
        fields = PromotionPolicy.__dataclass_fields__
        return PromotionPolicy(
            **{
                key: payload[key]
                if field.default is MISSING and field.default_factory is MISSING
                else payload.get(key, field.default)
                for key, field in fields.items()
            }
        )


class SqlCanonicalPromotionEvidence:
    """Build promotion evidence from canonical rows, never from a job payload."""

    def __init__(self, engine: Engine, policy_store: SqlPromotionPolicyStore | None = None):
        self.engine = engine
        self.policy_store = policy_store or SqlPromotionPolicyStore(engine)

    def build(
        self, request: PromotionRequest
    ) -> tuple[LifecycleState, PromotionEvidence, PromotionPolicy]:
        with self.engine.connect() as connection:
            artefact_rows = [
                row
                for row in connection.execute(select(strategy_artefact)).mappings()
                if isinstance(row["payload"], dict)
                and row["payload"].get("strategy_version_id") == request.strategy_version_id
            ]
            if len(artefact_rows) != 1:
                raise ValueError(
                    "canonical strategy artefact must have exactly one immutable version"
                )
            artefact_row = artefact_rows[0]
            if artefact_row is None:
                raise ValueError("canonical strategy artefact is missing")
            artefact = dict(artefact_row["payload"])
            artefact_hash = str(artefact.get("artefact_hash") or "")
            if artefact_hash != str(artefact_row["id"]):
                raise ValueError("canonical strategy artefact identity is invalid")
            content = dict(artefact)
            content.pop("artefact_hash", None)
            if canonical_hash(content) != artefact_hash:
                raise ValueError("canonical strategy artefact content hash is invalid")
            product_id = str(
                artefact.get("product_id") or (artefact.get("supported_products") or [""])[0] or ""
            )
            portfolio_id = str(artefact.get("portfolio_id") or "")
            if not portfolio_id:
                raise ValueError("canonical strategy artefact has no portfolio identity")
            account_id = str(artefact.get("account_id") or "")
            if not product_id:
                raise ValueError("canonical strategy artefact has no product identity")
            if not account_id:
                raise ValueError("canonical strategy artefact has no account identity")
            if not artefact.get("engine_version"):
                raise ValueError("canonical strategy artefact has no engine version")
            if not artefact.get("promotion_policy_id"):
                raise ValueError("canonical strategy artefact has no promotion policy identity")
            authoritative = artefact.get("authoritative_evidence")
            if not isinstance(authoritative, Mapping):
                raise ValueError("canonical strategy artefact has no authoritative evidence map")
            validation_stage_ids = authoritative.get("validation_stage_ids")
            holdout_claim_id = str(authoritative.get("holdout_claim_id") or "")
            if (
                not isinstance(validation_stage_ids, list)
                or not validation_stage_ids
                or not holdout_claim_id
            ):
                raise ValueError("canonical strategy artefact has incomplete evidence identities")
            binding_fields = {
                "strategy_version_id": request.strategy_version_id,
                "product_id": product_id,
                "portfolio_id": portfolio_id,
                "account_id": account_id,
                "promotion_policy_id": str(artefact.get("promotion_policy_id")),
                "engine_version": str(artefact.get("engine_version")),
            }
            if any(
                authoritative.get(field) not in {None, expected}
                for field, expected in binding_fields.items()
            ):
                raise ValueError("canonical artefact authoritative binding is inconsistent")
            validation_experiment_ids = {
                str(row["id"])
                for row in connection.execute(
                    select(experiment).where(
                        experiment.c.strategy_version_id == request.strategy_version_id
                    )
                ).mappings()
            }
            validation_by_id = {
                str(row["id"]): row
                for row in connection.execute(select(validation_stage)).mappings()
            }
            missing_validation = [
                str(identity)
                for identity in validation_stage_ids
                if str(identity) not in validation_by_id
            ]
            if missing_validation:
                raise ValueError(
                    "canonical validation stage is missing: " + ", ".join(missing_validation)
                )
            validation_rows = [validation_by_id[str(identity)] for identity in validation_stage_ids]
            if any(
                str(row["experiment_id"]) not in validation_experiment_ids
                and str(row["experiment_id"]) != request.strategy_version_id
                for row in validation_rows
            ):
                raise ValueError("canonical validation stage is bound to another experiment")
            claim_ids = {
                str(row["id"])
                for row in connection.execute(select(holdout_claim)).mappings()
                if isinstance(row["payload"], dict)
                and row["payload"].get("strategy_version_id") == request.strategy_version_id
            }
            holdout_rows = [
                row
                for row in connection.execute(select(holdout_outcome)).mappings()
                if str(row["holdout_claim_id"]) in claim_ids
            ]
            claim_row = next(
                (
                    row
                    for row in connection.execute(select(holdout_claim)).mappings()
                    if str(row["id"]) == holdout_claim_id
                ),
                None,
            )
            if claim_row is None or str(claim_row["id"]) not in claim_ids:
                raise ValueError("canonical holdout claim is missing or misbound")
            holdout_rows = [
                row for row in holdout_rows if str(row["holdout_claim_id"]) == holdout_claim_id
            ]
            if not holdout_rows:
                raise ValueError("canonical holdout outcome is missing")
            forward_summaries = [
                row
                for row in connection.execute(select(forward_paper_summary)).mappings()
                if row["strategy_version_id"] == request.strategy_version_id
                and row["product_id"] == product_id
                and row["artefact_hash"] == artefact_hash
            ]
            forward_summary = max(
                forward_summaries,
                key=lambda row: str(row["created_at"]),
                default=None,
            )
            forward_decisions = [
                row
                for row in connection.execute(select(forward_paper_decision)).mappings()
                if row["strategy_version_id"] == request.strategy_version_id
                and row["product_id"] == product_id
                and row["artefact_hash"] == artefact_hash
            ]
            forward_decision = (
                max(
                    (
                        row
                        for row in forward_decisions
                        if forward_summary is not None
                        and row["summary_id"] == forward_summary["id"]
                    ),
                    key=lambda row: str(row["decided_at"]),
                    default=None,
                )
                if forward_summary is not None
                else None
            )
            approvals = [
                row
                for row in connection.execute(select(strategy_approval)).mappings()
                if row["strategy_version_id"] == request.strategy_version_id
                and row["product_id"] == product_id
                and row["account_id"] == account_id
                and row["artefact_hash"] == artefact_hash
            ]
            preflights = [
                row
                for row in connection.execute(select(production_preflight)).mappings()
                if row["strategy_version_id"] == request.strategy_version_id
                and row["product_id"] == product_id
                and row["account_id"] == account_id
                and row["artefact_hash"] == artefact_hash
            ]
            events = list(connection.execute(select(promotion_event)).mappings())

        latest_approval = (
            max(approvals, key=lambda row: str(row["approved_at"])) if approvals else None
        )
        latest_preflight = (
            max(preflights, key=lambda row: str(row["checked_at"])) if preflights else None
        )
        latest_event = max(
            (
                row
                for row in events
                if isinstance(row["payload"], dict)
                and row["payload"].get("strategy_version_id") == request.strategy_version_id
            ),
            key=lambda row: str(row["created_at"]),
            default=None,
        )
        current_state = (
            LifecycleState(str(latest_event["payload"]["next_state"]))
            if latest_event is not None
            else LifecycleState.REGISTERED
        )
        summary_payload = (
            forward_summary["payload"]
            if forward_summary is not None and isinstance(forward_summary["payload"], Mapping)
            else {}
        )
        forward_days = float(summary_payload.get("elapsed_days", 0.0))
        observation_values = {
            field: [float(summary_payload[field])]
            if isinstance(summary_payload.get(field), int | float)
            else []
            for field in (
                "drawdown",
                "execution_drift",
                "model_drift",
                "portfolio_capacity",
                "risk_budget_available",
            )
        }
        source_commit_hash = str(
            artefact.get("source_commit_hash") or artefact.get("source_commit") or ""
        )
        if not source_commit_hash:
            raise ValueError("canonical strategy artefact has no source commit hash")
        artifact_engine = str(artefact.get("engine_version") or "")
        validation_accepted = bool(validation_rows) and all(
            row["accepted"] for row in validation_rows
        )
        protected_accepted = bool(holdout_rows) and all(row["accepted"] for row in holdout_rows)
        forward_accepted = bool(
            forward_decision is not None and forward_decision["accepted"] is True
        )
        fresh_preflight = bool(
            latest_preflight
            and latest_preflight["accepted"]
            and latest_preflight["artefact_hash"] == artefact_hash
            and latest_preflight["source_commit_hash"] == source_commit_hash
            and (not artifact_engine or latest_preflight["engine_version"] == artifact_engine)
            and preflight_is_fresh(
                str(latest_preflight["checked_at"]),
                reference_at=request.evaluated_at,
                maximum_age_seconds=int(artefact.get("preflight_max_age_seconds", 3_600)),
            )
        )
        definition = artefact.get("definition")
        market_making = (
            isinstance(definition, Mapping) and definition.get("family") == "market_making"
        )
        capability = artefact.get("promotion_policy")
        replay = authoritative.get("market_making_event_replay")
        evidence = PromotionEvidence(
            strategy_artefact_hash=artefact_hash,
            source_commit_hash=source_commit_hash,
            validation_accepted=validation_accepted,
            protected_holdout_accepted=protected_accepted,
            forward_evidence_days=int(forward_days),
            forward_evidence_accepted=forward_accepted,
            drawdown=max(observation_values.get("drawdown") or [0.0]),
            execution_drift=max(observation_values.get("execution_drift") or [0.0]),
            model_drift=max(observation_values.get("model_drift") or [0.0]),
            portfolio_capacity=max(observation_values.get("portfolio_capacity") or [0.0]),
            requested_capital=request.requested_capital,
            risk_budget_available=max(observation_values.get("risk_budget_available") or [0.0]),
            live_approval=bool(latest_approval and latest_approval["status"] == "approved"),
            fresh_preflight=fresh_preflight,
            product_id=product_id,
            account_id=account_id or None,
            artefact_record_id=str(artefact_row["id"]),
            preflight_record_id=(str(latest_preflight["id"]) if latest_preflight else None),
            portfolio_id=portfolio_id,
            market_making=market_making,
            market_making_live_capability=(
                isinstance(capability, Mapping)
                and capability.get("market_making_live_enabled") is True
            ),
            event_replay_passed=isinstance(replay, Mapping) and replay.get("passed") is True,
            event_replay_fills=(int(replay.get("fills", 0)) if isinstance(replay, Mapping) else 0),
            forward_summary_id=(str(forward_summary["id"]) if forward_summary else None),
            forward_decision_id=(str(forward_decision["id"]) if forward_decision else None),
            forward_independent_decisions=int(summary_payload.get("independent_decisions", 0)),
            forward_net_pnl=float(summary_payload.get("net_pnl", 0.0)),
            forward_benchmark_pnl=float(summary_payload.get("benchmark_pnl", 0.0)),
            forward_excess_benchmark_pnl=float(summary_payload.get("excess_benchmark_pnl", 0.0)),
            forward_objective_unit=(
                str(summary_payload["objective_unit"])
                if summary_payload.get("objective_unit") is not None
                else None
            ),
            forward_objective_value=(
                float(summary_payload["objective_value"])
                if summary_payload.get("objective_value") is not None
                else None
            ),
            forward_benchmark_value=(
                float(summary_payload["benchmark_value"])
                if summary_payload.get("benchmark_value") is not None
                else None
            ),
            forward_objective_excess=(
                float(summary_payload["objective_excess"])
                if summary_payload.get("objective_excess") is not None
                else None
            ),
            forward_objective_excess_fraction=(
                float(summary_payload["objective_excess_fraction"])
                if summary_payload.get("objective_excess_fraction") is not None
                else None
            ),
            forward_data_gaps=int(summary_payload.get("data_gaps", 0)),
            canary_evidence_accepted=bool(summary_payload.get("canary_evidence_accepted") is True),
            forward_effective_trades=int(summary_payload.get("effective_trades", 0)),
            forward_fill_rate=float(summary_payload.get("fill_rate", 1.0)),
            forward_slippage=float(summary_payload.get("slippage", 0.0)),
            forward_data_uptime=float(summary_payload.get("data_uptime", 0.0)),
            forward_rejected_orders=int(summary_payload.get("rejected_orders", 0)),
            supported_instruments=tuple(
                sorted(
                    str(value)
                    for value in artefact.get("supported_instruments", ())
                    if str(value).strip()
                )
            ),
        )
        policy_id = str(artefact.get("promotion_policy_id") or "")
        policy = self.policy_store.get(policy_id) if policy_id else self.policy_store.first()
        return current_state, evidence, policy


def decide_promotion(
    *,
    strategy_version_id: str,
    current_state: LifecycleState,
    evidence: PromotionEvidence,
    policy: PromotionPolicy,
    evaluated_at: str,
    requested_transition: str | None = None,
) -> PromotionDecision:
    strategy_version_id = non_empty(strategy_version_id, field="strategy_version_id")
    evaluated_at = timestamp(evaluated_at, field="evaluated_at")
    requested_transition = (
        non_empty(requested_transition, field="requested_transition")
        if requested_transition is not None
        else None
    )
    if requested_transition == "suspend":
        next_state = (
            current_state if current_state is LifecycleState.RETIRED else LifecycleState.SUSPENDED
        )
        return _decision(
            strategy_version_id,
            current_state,
            next_state,
            current_state is not LifecycleState.RETIRED,
            "suspended_by_request",
            evaluated_at,
            0.0,
            evidence,
        )
    if requested_transition == "retire":
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.RETIRED,
            current_state is not LifecycleState.RETIRED,
            "retired_by_request",
            evaluated_at,
            0.0,
            evidence,
        )
    if requested_transition == "resume":
        if current_state is not LifecycleState.SUSPENDED:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                "resume_requires_suspended_state",
                evaluated_at,
                0.0,
                evidence,
            )
        if not evidence.forward_evidence_accepted:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                "resume_forward_evidence_missing",
                evaluated_at,
                0.0,
                evidence,
            )
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.LIVE_READY,
            True,
            "resumed_to_live_ready",
            evaluated_at,
            policy.paper_capital_limit,
            evidence,
        )
    if requested_transition in {"live", "live_canary"} and current_state in {
        LifecycleState.REGISTERED,
        LifecycleState.DEVELOPMENT,
    }:
        return _decision(
            strategy_version_id,
            current_state,
            current_state,
            False,
            "forward_paper_required",
            evaluated_at,
            0.0,
            evidence,
        )
    if requested_transition == "live" and current_state in {
        LifecycleState.FORWARD_PAPER,
        LifecycleState.LIVE_READY,
    }:
        return _decision(
            strategy_version_id,
            current_state,
            current_state,
            False,
            "canary_required_before_live",
            evaluated_at,
            0.0,
            evidence,
        )
    if evidence.drawdown > policy.maximum_drawdown:
        next_state = (
            LifecycleState.SUSPENDED
            if current_state
            in {
                LifecycleState.FORWARD_PAPER,
                LifecycleState.LIVE_READY,
                LifecycleState.LIVE_CANARY,
                LifecycleState.LIVE,
            }
            else current_state
        )
        return _decision(
            strategy_version_id,
            current_state,
            next_state,
            False,
            "drawdown_limit",
            evaluated_at,
            0.0,
            evidence,
        )
    if evidence.execution_drift > policy.maximum_execution_drift:
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.SUSPENDED,
            False,
            "execution_drift_limit",
            evaluated_at,
            0.0,
            evidence,
        )
    if evidence.model_drift > policy.maximum_model_drift:
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.SUSPENDED,
            False,
            "model_drift_limit",
            evaluated_at,
            0.0,
            evidence,
        )
    if current_state in {LifecycleState.REGISTERED, LifecycleState.DEVELOPMENT}:
        checks = (
            (not evidence.validation_accepted, "validation_not_accepted"),
            (not evidence.protected_holdout_accepted, "protected_holdout_not_accepted"),
            (not policy.automatic_paper_promotion, "automatic_paper_promotion_disabled"),
        )
        reason = next((reason for failed, reason in checks if failed), None)
        if reason:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                reason,
                evaluated_at,
                0.0,
                evidence,
            )
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.FORWARD_PAPER,
            True,
            "forward_paper_promoted",
            evaluated_at,
            policy.paper_capital_limit,
            evidence,
        )
    if current_state in {LifecycleState.FORWARD_PAPER, LifecycleState.LIVE_READY}:
        failures = _forward_evidence_failures(evidence, policy)
        if failures:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                failures[0],
                evaluated_at,
                0.0,
                evidence,
            )
        if requested_transition == "live_canary":
            reason = _live_authority_failure(evidence)
            if reason:
                return _decision(
                    strategy_version_id,
                    current_state,
                    current_state,
                    False,
                    reason,
                    evaluated_at,
                    0.0,
                    evidence,
                )
            capital = min(
                policy.canary_capital_limit,
                evidence.requested_capital or policy.canary_capital_limit,
                evidence.portfolio_capacity,
                evidence.risk_budget_available,
            )
            return _decision(
                strategy_version_id,
                current_state,
                LifecycleState.LIVE_CANARY,
                True,
                "live_canary_promoted",
                evaluated_at,
                capital,
                evidence,
            )
        if current_state is LifecycleState.LIVE_READY:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                True,
                "live_ready_maintained",
                evaluated_at,
                policy.paper_capital_limit,
                evidence,
            )
        if policy.automatic_live_ready_promotion and requested_transition in {
            None,
            "forward_paper",
            "live_ready",
        }:
            return _decision(
                strategy_version_id,
                current_state,
                LifecycleState.LIVE_READY,
                True,
                "live_ready_promoted",
                evaluated_at,
                policy.paper_capital_limit,
                evidence,
            )
        return _decision(
            strategy_version_id,
            current_state,
            current_state,
            True,
            "forward_paper_maintained",
            evaluated_at,
            policy.paper_capital_limit,
            evidence,
        )
    if current_state is LifecycleState.LIVE_CANARY and requested_transition == "live":
        reason = _live_authority_failure(evidence, require_canary=True)
        if reason:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                reason,
                evaluated_at,
                0.0,
                evidence,
            )
        capital = min(
            evidence.requested_capital,
            evidence.portfolio_capacity,
            evidence.risk_budget_available,
        )
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.LIVE,
            True,
            "live_promoted",
            evaluated_at,
            capital,
            evidence,
        )
    return _decision(
        strategy_version_id,
        current_state,
        current_state,
        True,
        "lifecycle_state_maintained",
        evaluated_at,
        min(evidence.requested_capital, evidence.risk_budget_available),
        evidence,
    )


def _forward_evidence_failures(
    evidence: PromotionEvidence, policy: PromotionPolicy
) -> tuple[str, ...]:
    requires_objective = evidence.product_id in {"btc_accumulation", "active_income"}
    expected_unit = objective_unit(str(evidence.product_id)) if requires_objective else None
    objective_missing = requires_objective and (
        evidence.forward_objective_unit != expected_unit
        or evidence.forward_objective_value is None
        or evidence.forward_benchmark_value is None
        or evidence.forward_objective_excess is None
        or evidence.forward_objective_excess_fraction is None
    )
    objective_failure = (
        objective_missing
        or (
            requires_objective
            and evidence.forward_objective_excess_fraction is not None
            and evidence.forward_objective_excess_fraction
            <= policy.minimum_forward_objective_excess_fraction
        )
    )
    checks = (
        (evidence.forward_summary_id is None, "forward_summary_missing"),
        (
            evidence.forward_evidence_days < policy.required_forward_evidence_days,
            "forward_evidence_duration_insufficient",
        ),
        (
            evidence.forward_independent_decisions < policy.minimum_forward_independent_decisions,
            "forward_decisions_insufficient",
        ),
        (
            objective_failure
            if requires_objective
            else evidence.forward_net_pnl <= policy.minimum_forward_net_pnl,
            "forward_objective_evidence_missing"
            if objective_missing
            else (
                "forward_objective_excess_threshold"
                if requires_objective
                else "forward_net_pnl_threshold"
            ),
        ),
        (evidence.forward_data_gaps > policy.maximum_forward_data_gaps, "forward_data_gaps"),
        (not evidence.forward_evidence_accepted, "forward_evidence_not_accepted"),
        (
            evidence.forward_effective_trades < policy.minimum_forward_effective_trades,
            "forward_effective_trades_insufficient",
        ),
        (
            evidence.forward_fill_rate < policy.minimum_forward_fill_rate,
            "forward_fill_rate_insufficient",
        ),
        (
            evidence.forward_slippage > policy.maximum_forward_slippage,
            "forward_slippage_limit",
        ),
        (
            evidence.forward_data_uptime < policy.minimum_forward_data_uptime,
            "forward_data_uptime_insufficient",
        ),
        (
            evidence.forward_rejected_orders > policy.maximum_forward_rejected_orders,
            "forward_rejected_orders_limit",
        ),
        (evidence.forward_data_uptime <= 0, "forward_data_uptime_unavailable"),
        (evidence.portfolio_capacity <= 0, "portfolio_capacity_unavailable"),
        (evidence.risk_budget_available <= 0, "risk_budget_unavailable"),
        (
            evidence.market_making and not evidence.market_making_live_capability,
            "market_making_live_capability_missing",
        ),
        (
            evidence.market_making
            and (not evidence.event_replay_passed or evidence.event_replay_fills < 500),
            "market_making_event_replay_insufficient",
        ),
    )
    return tuple(reason for failed, reason in checks if failed)


def _live_authority_failure(
    evidence: PromotionEvidence, *, require_canary: bool = False
) -> str | None:
    checks = (
        (require_canary and not evidence.canary_evidence_accepted, "canary_evidence_not_accepted"),
        (not evidence.live_approval, "live_approval_missing"),
        (not evidence.fresh_preflight, "fresh_preflight_missing"),
        (evidence.portfolio_capacity <= 0, "portfolio_capacity_unavailable"),
        (evidence.risk_budget_available <= 0, "risk_budget_unavailable"),
    )
    return next((reason for failed, reason in checks if failed), None)


def _decision(
    strategy_version_id: str,
    prior_state: LifecycleState,
    next_state: LifecycleState,
    accepted: bool,
    reason_code: str,
    evaluated_at: str,
    capital_limit: float,
    evidence: PromotionEvidence,
) -> PromotionDecision:
    return PromotionDecision(
        strategy_version_id=strategy_version_id,
        prior_state=prior_state,
        next_state=next_state,
        accepted=accepted,
        reason_code=reason_code,
        evaluated_at=evaluated_at,
        capital_limit=max(0.0, capital_limit),
        evidence_hash=canonical_hash(evidence),
    )


class SqlPromotionStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, decision: PromotionDecision) -> str:
        payload = json_value(
            {
                "strategy_version_id": decision.strategy_version_id,
                "prior_state": decision.prior_state.value,
                "next_state": decision.next_state.value,
                "accepted": decision.accepted,
                "reason_code": decision.reason_code,
                "evaluated_at": decision.evaluated_at,
                "capital_limit": decision.capital_limit,
                "evidence_hash": decision.evidence_hash,
            },
            field="promotion decision",
        )
        identity = canonical_hash(payload)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(promotion_event.c.payload).where(promotion_event.c.id == identity)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != payload:
                    raise ValueError("promotion decision identity collision")
                return identity
            connection.execute(
                insert(promotion_event).values(
                    id=identity,
                    created_at=decision.evaluated_at,
                    payload=payload,
                )
            )
        return identity

    def latest(self, strategy_version_id: str) -> PromotionDecision | None:
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(promotion_event.c.payload).order_by(promotion_event.c.created_at.desc())
            ).scalars()
            for payload in payloads:
                if payload.get("strategy_version_id") != strategy_version_id:
                    continue
                return PromotionDecision(
                    strategy_version_id=payload["strategy_version_id"],
                    prior_state=LifecycleState(payload["prior_state"]),
                    next_state=LifecycleState(payload["next_state"]),
                    accepted=bool(payload["accepted"]),
                    reason_code=payload["reason_code"],
                    evaluated_at=payload["evaluated_at"],
                    capital_limit=float(payload["capital_limit"]),
                    evidence_hash=payload["evidence_hash"],
                )
        return None


class DatabasePromotionWorker:
    """Evaluate deterministic lifecycle jobs without granting execution authority."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        store: SqlPromotionStore,
        policy_store: SqlPromotionPolicyStore | None = None,
        strict_identity_contract: bool | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.policy_store = policy_store or SqlPromotionPolicyStore(store.engine)
        self.strict_identity_contract = (
            store.engine.dialect.name == "postgresql"
            if strict_identity_contract is None
            else strict_identity_contract
        )
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("promotion_evaluation",),
        )
        if claimed is None:
            return {"reason_code": "promotion_queue_empty"}
        try:
            if self.strict_identity_contract:
                request = PromotionRequest.from_payload(claimed.payload)
                current_state, evidence, policy = SqlCanonicalPromotionEvidence(
                    self.store.engine, self.policy_store
                ).build(request)
                strategy_version_id = request.strategy_version_id
                evaluated_at = request.evaluated_at
            else:
                # SQLite is retained for isolated unit tests. Those fixtures
                # may still call the pure lifecycle function with an explicit
                # evidence object, but production PostgreSQL never enters this
                # branch.
                raw_evidence = claimed.payload.get("evidence")
                raw_policy = claimed.payload.get("policy")
                if isinstance(raw_evidence, dict) and isinstance(raw_policy, dict):
                    evidence = PromotionEvidence(**raw_evidence)
                    policy = PromotionPolicy(**raw_policy)
                    current_state = LifecycleState(str(claimed.payload["current_state"]))
                    strategy_version_id = str(claimed.payload["strategy_version_id"])
                    evaluated_at = str(claimed.payload.get("evaluated_at") or now)
                else:
                    request = PromotionRequest.from_payload(claimed.payload)
                    current_state, evidence, policy = SqlCanonicalPromotionEvidence(
                        self.store.engine, self.policy_store
                    ).build(request)
                    strategy_version_id = request.strategy_version_id
                    evaluated_at = request.evaluated_at
            decision = decide_promotion(
                strategy_version_id=strategy_version_id,
                current_state=current_state,
                evidence=evidence
                if isinstance(evidence, PromotionEvidence)
                else PromotionEvidence(**evidence),
                policy=policy if isinstance(policy, PromotionPolicy) else PromotionPolicy(**policy),
                evaluated_at=evaluated_at,
                requested_transition=(
                    str(claimed.payload["requested_transition"])
                    if claimed.payload.get("requested_transition") is not None
                    else None
                ),
            )
            identity = self.store.append(decision)
            if self.strict_identity_contract:
                assignments = SqlActiveStrategyAssignmentRepository(self.store.engine)
                if decision.accepted and decision.next_state in {
                    LifecycleState.FORWARD_PAPER,
                    LifecycleState.LIVE_READY,
                }:
                    if not evidence.product_id or not evidence.portfolio_id:
                        raise ValueError(
                            "accepted promotion evidence lacks product/portfolio identity"
                        )
                    instrument_ids = evidence.supported_instruments or (None,)
                    for instrument_id in instrument_ids:
                        assignments.assign(
                            product_id=evidence.product_id,
                            portfolio_id=evidence.portfolio_id,
                            strategy_version_id=decision.strategy_version_id,
                            artefact_hash=evidence.strategy_artefact_hash,
                            lifecycle_state=decision.next_state.value,
                            execution_mode="paper",
                            capital_limit=decision.capital_limit,
                            assigned_at=decision.evaluated_at,
                            assigned_by=self.worker_id,
                            instrument_id=instrument_id,
                            risk_budget=min(
                                decision.capital_limit,
                                evidence.risk_budget_available,
                            ),
                            payload={
                                "promotion_event_id": identity,
                                "source": "canonical_promotion_worker",
                            },
                        )
                elif decision.next_state in {LifecycleState.SUSPENDED, LifecycleState.RETIRED}:
                    if evidence.product_id:
                        assignments.deactivate(evidence.product_id)
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "promotion_evaluation_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": decision.reason_code,
            "job_id": claimed.job_id,
            "promotion_event_id": identity,
            "accepted": decision.accepted,
            "next_state": decision.next_state.value,
            "capital_limit": decision.capital_limit,
        }


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
