"""Deterministic strategy lifecycle and promotion policy."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import (
    experiment,
    forward_evidence,
    forward_paper_observation,
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
from src.services.scheduler import DatabaseJobQueue


class LifecycleState(StrEnum):
    REGISTERED = "registered"
    DEVELOPMENT = "development"
    FORWARD_PAPER = "forward_paper"
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

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PromotionRequest:
        if not isinstance(payload, Mapping):
            raise ValueError("promotion request must be an object")
        allowed = {
            "strategy_version_id",
            "requested_transition",
            "requested_capital",
            "evaluated_at",
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
        if transition not in {"forward_paper", "live_canary", "live", "suspend", "retire"}:
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
        values = {key: payload[key] for key in fields}
        return PromotionPolicy(**values)

    def first(self) -> PromotionPolicy:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(promotion_policy.c.payload).order_by(promotion_policy.c.created_at).limit(1)
            ).scalar_one_or_none()
        if payload is None:
            raise KeyError("no authoritative promotion policy exists")
        fields = PromotionPolicy.__dataclass_fields__
        return PromotionPolicy(**{key: payload[key] for key in fields})


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
            forward_evidence_id = str(authoritative.get("forward_evidence_id") or "")
            if (
                not isinstance(validation_stage_ids, list)
                or not validation_stage_ids
                or not holdout_claim_id
                or not forward_evidence_id
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
            forward_rows = [
                row
                for row in connection.execute(select(forward_paper_observation)).mappings()
                if row["strategy_version_id"] == request.strategy_version_id
                and row["artefact_hash"] == artefact_hash
            ]
            if not any(str(row["id"]) == forward_evidence_id for row in forward_rows):
                forward_record = (
                    connection.execute(
                        select(forward_evidence).where(forward_evidence.c.id == forward_evidence_id)
                    )
                    .mappings()
                    .first()
                )
                if forward_record is None or not isinstance(forward_record["payload"], dict):
                    raise ValueError("canonical forward evidence is missing or misbound")
                forward_payload = forward_record["payload"]
                if (
                    forward_payload.get("strategy_version_id") != request.strategy_version_id
                    or forward_payload.get("product_id") != product_id
                ):
                    raise ValueError("canonical forward evidence is bound to another product")
                forward_rows = [forward_record]
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
        observations = []
        for row in forward_rows:
            if not isinstance(row["payload"], dict):
                continue
            payload = dict(row["payload"])
            summary = payload.get("evidence") or payload.get("observation")
            observations.append(
                {**dict(summary), **payload} if isinstance(summary, Mapping) else payload
            )
        observation_values = {
            field: [
                float(item[field])
                for item in observations
                if isinstance(item.get(field), int | float)
            ]
            for field in (
                "drawdown",
                "execution_drift",
                "model_drift",
                "portfolio_capacity",
                "risk_budget_available",
            )
        }
        forward_days = max([float(item.get("evidence_days", 0)) for item in observations] + [0.0])
        if len(forward_rows) >= 2:
            first = min(str(row["observed_at"]) for row in forward_rows)
            last = max(str(row["observed_at"]) for row in forward_rows)
            try:
                delta = dt.datetime.fromisoformat(last) - dt.datetime.fromisoformat(first)
                forward_days = max(forward_days, delta.total_seconds() / 86_400)
            except ValueError:
                pass
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
        forward_accepted = bool(forward_rows) and all(
            item.get("accepted", True) is True for item in observations
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
) -> PromotionDecision:
    strategy_version_id = non_empty(strategy_version_id, field="strategy_version_id")
    evaluated_at = timestamp(evaluated_at, field="evaluated_at")
    if evidence.drawdown > policy.maximum_drawdown:
        next_state = (
            LifecycleState.SUSPENDED
            if current_state
            in {
                LifecycleState.FORWARD_PAPER,
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
        if not evidence.validation_accepted:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                "validation_not_accepted",
                evaluated_at,
                0.0,
                evidence,
            )
        if not evidence.protected_holdout_accepted:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                "protected_holdout_not_accepted",
                evaluated_at,
                0.0,
                evidence,
            )
        if not policy.automatic_paper_promotion:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                "automatic_paper_promotion_disabled",
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
            0.0,
            evidence,
        )
    if current_state is LifecycleState.FORWARD_PAPER:
        checks = (
            (
                evidence.forward_evidence_days < policy.required_forward_evidence_days,
                "forward_evidence_duration_insufficient",
            ),
            (not evidence.forward_evidence_accepted, "forward_evidence_not_accepted"),
            (not policy.automatic_live_canary_promotion, "automatic_live_canary_disabled"),
            (not evidence.live_approval, "live_approval_missing"),
            (not evidence.fresh_preflight, "fresh_preflight_missing"),
            (evidence.portfolio_capacity <= 0, "portfolio_capacity_unavailable"),
            (evidence.risk_budget_available <= 0, "risk_budget_unavailable"),
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
        capital = min(
            policy.canary_capital_limit,
            evidence.requested_capital,
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
                evidence = claimed.payload.get("evidence")
                policy = claimed.payload.get("policy")
                if isinstance(evidence, dict) and isinstance(policy, dict):
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
            )
            identity = self.store.append(decision)
            if self.strict_identity_contract:
                assignments = SqlActiveStrategyAssignmentRepository(self.store.engine)
                if decision.accepted and decision.next_state in {
                    LifecycleState.FORWARD_PAPER,
                    LifecycleState.LIVE_CANARY,
                    LifecycleState.LIVE,
                }:
                    if not evidence.product_id or not evidence.portfolio_id:
                        raise ValueError(
                            "accepted promotion evidence lacks product/portfolio identity"
                        )
                    assignments.assign(
                        product_id=evidence.product_id,
                        portfolio_id=evidence.portfolio_id,
                        strategy_version_id=decision.strategy_version_id,
                        artefact_hash=evidence.strategy_artefact_hash,
                        lifecycle_state=decision.next_state.value,
                        execution_mode=(
                            "live"
                            if decision.next_state
                            in {LifecycleState.LIVE_CANARY, LifecycleState.LIVE}
                            else "paper"
                        ),
                        capital_limit=decision.capital_limit,
                        assigned_at=decision.evaluated_at,
                        assigned_by=self.worker_id,
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
