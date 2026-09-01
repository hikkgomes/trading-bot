"""Repositories for immutable research evidence and lifecycle authority."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import insert, select, text
from sqlalchemy.engine import Engine

from src.data.database import (
    accounting_entry,
    active_strategy_assignment,
    alpha_forecast,
    dataset_snapshot,
    experiment,
    experiment_run,
    fill,
    forward_paper_decision,
    forward_paper_observation,
    forward_paper_summary,
    holdout_claim,
    holdout_outcome,
    order_intent,
    production_preflight,
    risk_snapshot,
    strategy_approval,
    strategy_artefact,
    strategy_definition,
    strategy_version,
    target_position,
    validation_stage,
)
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.research.objectives import objective_unit


class CanonicalEvidenceError(RuntimeError):
    """Canonical evidence is invalid, missing, or conflicts with history."""


_VALIDATION_STAGES = frozenset({"screening", "development", "robustness", "protected", "forward"})
_AUTOMATION_APPROVAL_ACTORS = frozenset(
    {
        "agent",
        "automation",
        "autopilot",
        "bot",
        "ci",
        "cron",
        "github-actions",
        "github-actions[bot]",
        "robot",
        "scheduler",
        "service",
        "system",
        "trading-bot",
    }
)


def _human_actor(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalEvidenceError(f"{field} must identify a human operator")
    actor = value.strip()
    key = actor.casefold().replace("_", "-").replace(" ", "-")
    if key in _AUTOMATION_APPROVAL_ACTORS or key.endswith("-bot") or key.endswith("[bot]"):
        raise CanonicalEvidenceError(f"{field} must identify a human operator")
    return actor


def preflight_is_fresh(
    checked_at: str,
    *,
    reference_at: str | None = None,
    maximum_age_seconds: int = 3_600,
) -> bool:
    if isinstance(maximum_age_seconds, bool) or maximum_age_seconds <= 0:
        raise CanonicalEvidenceError("maximum preflight age must be positive")
    try:
        checked = dt.datetime.fromisoformat(timestamp(checked_at, field="checked_at"))
        reference = dt.datetime.fromisoformat(
            timestamp(reference_at or dt.datetime.now(dt.UTC), field="reference_at")
        )
    except ValueError as exc:
        raise CanonicalEvidenceError("preflight timestamps are invalid") from exc
    age = (reference - checked).total_seconds()
    return 0 <= age <= maximum_age_seconds


def latest_accepted_forward_summary(
    engine: Engine,
    *,
    strategy_version_id: str,
    product_id: str,
    artefact_hash: str,
    at: str | None = None,
) -> dict[str, Any] | None:
    """Return the latest summary only when its latest decision is accepted."""

    with engine.connect() as connection:
        statement = select(forward_paper_summary).where(
            forward_paper_summary.c.strategy_version_id == strategy_version_id,
            forward_paper_summary.c.product_id == product_id,
            forward_paper_summary.c.artefact_hash == artefact_hash,
        )
        if at is not None:
            statement = statement.where(
                forward_paper_summary.c.created_at <= timestamp(at, field="forward summary time")
            )
        summary = (
            connection.execute(
                statement.order_by(
                    forward_paper_summary.c.created_at.desc(),
                    forward_paper_summary.c.id.desc(),
                ).limit(1)
            )
            .mappings()
            .first()
        )
        if summary is None:
            return None
        decision_statement = select(forward_paper_decision).where(
            forward_paper_decision.c.summary_id == summary["id"],
        )
        if at is not None:
            decision_statement = decision_statement.where(
                forward_paper_decision.c.decided_at <= timestamp(at, field="forward decision time")
            )
        decision = (
            connection.execute(
                decision_statement.order_by(
                    forward_paper_decision.c.decided_at.desc(),
                    forward_paper_decision.c.id.desc(),
                ).limit(1)
            )
            .mappings()
            .first()
        )
    if decision is None or decision["accepted"] is not True:
        return None
    return {"summary": dict(summary), "decision": dict(decision)}


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise CanonicalEvidenceError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalEvidenceError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise CanonicalEvidenceError(f"{field} must be finite and non-negative")
    return result


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise CanonicalEvidenceError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalEvidenceError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise CanonicalEvidenceError(f"{field} must be finite")
    return result


def _object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CanonicalEvidenceError(f"{field} must be an object")
    return json_value(dict(value), field=field)


def _hash(value: object, *, field: str) -> str:
    digest = canonical_hash(value)
    if not digest.startswith("sha256:"):
        raise CanonicalEvidenceError(f"{field} did not produce a SHA-256 identity")
    return digest


def _identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise CanonicalEvidenceError(f"{field} must be a sha256: identity")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise CanonicalEvidenceError(f"{field} must be a sha256: identity") from exc
    return value


def _immutable_insert(connection, table, values: dict[str, Any]) -> str:
    identity = str(values["id"])
    existing = connection.execute(select(table).where(table.c.id == identity)).mappings().first()
    if existing is None:
        connection.execute(insert(table).values(**values))
        return identity
    for key, value in values.items():
        if existing[key] != value:
            raise CanonicalEvidenceError(
                f"immutable identity collision in {table.name}: {identity}"
            )
    return identity


def _assert_canonical_artifact(connection, artefact_hash: str) -> dict[str, Any]:
    row = (
        connection.execute(
            select(strategy_artefact.c.payload, strategy_artefact.c.created_at).where(
                strategy_artefact.c.id == artefact_hash
            )
        )
        .mappings()
        .first()
    )
    if row is None or not isinstance(row["payload"], Mapping):
        raise CanonicalEvidenceError(f"strategy artefact does not exist: {artefact_hash}")
    payload = dict(row["payload"])
    if payload.get("schema") != "platform.strategy_artefact/v2":
        raise CanonicalEvidenceError("canonical strategy artefact schema is unsupported")
    payload_created_at = payload.get("created_at")
    if not isinstance(payload_created_at, str) or timestamp(
        payload_created_at, field="artefact.created_at"
    ) != timestamp(str(row["created_at"]), field="artefact.created_at"):
        raise CanonicalEvidenceError(
            "canonical strategy artefact creation timestamp does not match immutable storage"
        )
    content = dict(payload)
    content.pop("artefact_hash", None)
    if payload.get("artefact_hash") != artefact_hash or canonical_hash(content) != artefact_hash:
        raise CanonicalEvidenceError("canonical strategy artefact content hash is invalid")
    return payload


def _assert_artefact_binding(
    payload: Mapping[str, Any],
    *,
    strategy_version_id: str,
    product_id: str,
    portfolio_id: str | None = None,
    account_id: str | None = None,
) -> None:
    expected = {
        "strategy_version_id": strategy_version_id,
        "product_id": product_id,
        "portfolio_id": portfolio_id,
        "account_id": account_id,
    }
    for field, value in expected.items():
        if value is not None and payload.get(field) != value:
            raise CanonicalEvidenceError(f"canonical artefact {field} binding is invalid")
    if product_id not in set(payload.get("supported_products") or []):
        raise CanonicalEvidenceError("canonical artefact does not support the product")


class SqlValidationRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append_stage(
        self,
        *,
        experiment_id: str,
        stage: str,
        source_run_id: str,
        evaluated_at: str,
        accepted: bool,
        reason_code: str | None,
        evidence: Mapping[str, Any],
    ) -> str:
        if not isinstance(accepted, bool):
            raise CanonicalEvidenceError("validation accepted must be a boolean")
        if not accepted and not reason_code:
            raise CanonicalEvidenceError("rejected validation stages need a reason code")
        if stage not in _VALIDATION_STAGES:
            raise CanonicalEvidenceError(f"unsupported validation stage: {stage}")
        experiment_id = str(experiment_id).strip()
        source_run_id = str(source_run_id).strip()
        if not experiment_id or not source_run_id:
            raise CanonicalEvidenceError("validation stages require experiment and source run IDs")
        evaluated_at = timestamp(evaluated_at, field="evaluated_at")
        evidence_payload = _object(evidence, field="validation evidence")
        evidence_hash = _hash(evidence_payload, field="evidence")
        payload = {
            "experiment_id": experiment_id,
            "stage": stage,
            "source_run_id": source_run_id,
            "evaluated_at": evaluated_at,
            "accepted": accepted,
            "reason_code": reason_code,
            "evidence": evidence_payload,
        }
        identity = _hash(payload, field="validation stage")
        with self.engine.begin() as connection:
            if (
                connection.execute(
                    select(experiment.c.id).where(experiment.c.id == experiment_id)
                ).first()
                is None
            ):
                raise CanonicalEvidenceError(f"experiment does not exist: {experiment_id}")
            source_run = (
                connection.execute(
                    select(experiment_run).where(experiment_run.c.id == source_run_id)
                )
                .mappings()
                .first()
            )
            if source_run is None:
                raise CanonicalEvidenceError(f"source run does not exist: {source_run_id}")
            if (
                not isinstance(source_run["payload"], Mapping)
                or source_run["payload"].get("candidate_id") != experiment_id
            ):
                raise CanonicalEvidenceError("validation source run belongs to another experiment")
            existing = (
                connection.execute(
                    select(validation_stage)
                    .where(
                        validation_stage.c.experiment_id == experiment_id,
                        validation_stage.c.stage == stage,
                    )
                    .limit(1)
                )
                .mappings()
                .first()
            )
            values = {
                "id": identity,
                "experiment_id": experiment_id,
                "stage": stage,
                "source_run_id": source_run_id,
                "evaluated_at": evaluated_at,
                "state": "accepted" if accepted else "rejected",
                "accepted": accepted,
                "reason_code": reason_code,
                "evidence_hash": evidence_hash,
                "payload": payload,
            }
            if existing is not None:
                if dict(existing) != values:
                    raise CanonicalEvidenceError(
                        f"validation stage already exists with different evidence: {experiment_id}:{stage}"
                    )
                return identity
            return _immutable_insert(connection, validation_stage, values)

    def stages(self, experiment_id: str) -> tuple[dict[str, Any], ...]:
        with self.engine.connect() as connection:
            return tuple(
                dict(row)
                for row in connection.execute(
                    select(validation_stage)
                    .where(validation_stage.c.experiment_id == experiment_id)
                    .order_by(validation_stage.c.evaluated_at, validation_stage.c.stage)
                ).mappings()
            )


class SqlHoldoutRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def claim(
        self,
        *,
        strategy_version_id: str,
        data_snapshot_id: str,
        cohort_id: str,
        source_hashes: Mapping[str, str],
        claimed_at: str,
    ) -> str:
        claimed_at = timestamp(claimed_at, field="claimed_at")
        strategy_version_id = str(strategy_version_id).strip()
        data_snapshot_id = _identity(data_snapshot_id, field="data_snapshot_id")
        cohort_id = str(cohort_id).strip()
        if not strategy_version_id or not data_snapshot_id or not cohort_id:
            raise CanonicalEvidenceError(
                "holdout claims require strategy, data snapshot, and cohort identities"
            )
        hashes = _object(source_hashes, field="holdout source hashes")
        payload = {
            "strategy_version_id": strategy_version_id,
            "data_snapshot_id": data_snapshot_id,
            "cohort_id": cohort_id,
            "source_hashes": hashes,
            "claimed_at": claimed_at,
        }
        identity = _hash(payload, field="holdout claim")
        with self.engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:claim_key))"),
                    {"claim_key": f"holdout-claim:{strategy_version_id}"},
                )
            if (
                connection.execute(
                    select(strategy_version.c.id).where(
                        strategy_version.c.id == strategy_version_id
                    )
                ).first()
                is None
            ):
                raise CanonicalEvidenceError(
                    f"strategy version does not exist: {strategy_version_id}"
                )
            if (
                connection.execute(
                    select(dataset_snapshot.c.id).where(dataset_snapshot.c.id == data_snapshot_id)
                ).first()
                is None
            ):
                raise CanonicalEvidenceError(f"dataset snapshot does not exist: {data_snapshot_id}")
            existing_claims = connection.execute(select(holdout_claim.c.payload)).scalars()
            for existing_payload in existing_claims:
                if (
                    isinstance(existing_payload, Mapping)
                    and existing_payload.get("strategy_version_id") == strategy_version_id
                    and canonical_hash(existing_payload) != identity
                ):
                    raise CanonicalEvidenceError(
                        "a strategy version may have only one protected holdout claim"
                    )
            return _immutable_insert(
                connection,
                holdout_claim,
                {"id": identity, "created_at": claimed_at, "payload": payload},
            )

    def record_outcome(
        self,
        *,
        claim_id: str,
        evaluated_at: str,
        accepted: bool,
        outcome: Mapping[str, Any],
    ) -> str:
        if not isinstance(accepted, bool):
            raise CanonicalEvidenceError("holdout accepted must be a boolean")
        evaluated_at = timestamp(evaluated_at, field="evaluated_at")
        outcome_payload = _object(outcome, field="holdout outcome")
        outcome_hash = _hash(outcome_payload, field="holdout outcome")
        identity = _hash(
            {"claim_id": claim_id, "evaluated_at": evaluated_at, "outcome_hash": outcome_hash},
            field="holdout outcome identity",
        )
        with self.engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:outcome_key))"),
                    {"outcome_key": f"holdout-outcome:{claim_id}"},
                )
            if (
                connection.execute(
                    select(holdout_claim.c.id).where(holdout_claim.c.id == claim_id)
                ).first()
                is None
            ):
                raise KeyError(f"holdout claim does not exist: {claim_id}")
            existing_outcomes = connection.execute(
                select(holdout_outcome).where(holdout_outcome.c.holdout_claim_id == claim_id)
            ).mappings()
            for existing in existing_outcomes:
                if str(existing["id"]) != identity:
                    raise CanonicalEvidenceError(
                        "a protected holdout claim may have only one outcome"
                    )
            return _immutable_insert(
                connection,
                holdout_outcome,
                {
                    "id": identity,
                    "holdout_claim_id": claim_id,
                    "evaluated_at": evaluated_at,
                    "accepted": accepted,
                    "outcome_hash": outcome_hash,
                    "payload": outcome_payload,
                },
            )

    def outcome_for(self, claim_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(holdout_outcome).where(holdout_outcome.c.holdout_claim_id == claim_id)
                )
                .mappings()
                .first()
            )
        return None if row is None else dict(row)


@dataclass(frozen=True)
class ForwardPaperSummary:
    strategy_version_id: str
    product_id: str
    artefact_hash: str
    observed_from: str
    observed_until: str
    elapsed_days: float
    independent_decisions: int
    net_pnl: float
    benchmark_pnl: float
    excess_benchmark_pnl: float
    drawdown: float
    execution_drift: float
    model_drift: float
    portfolio_capacity: float
    risk_budget_available: float
    data_gaps: int
    strategy_decay: float
    observation_ids: tuple[str, ...]
    effective_trades: int = 0
    fill_rate: float = 1.0
    slippage: float = 0.0
    data_uptime: float = 0.0
    rejected_orders: int = 0
    objective_unit: str | None = None
    objective_value: float | None = None
    benchmark_value: float | None = None
    objective_excess: float | None = None
    objective_excess_fraction: float | None = None
    trading_days: int = 0
    cycles: int = 0
    effective_independent_episodes: int = 0
    tail_loss: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "platform.forward_paper_summary/v1",
            "strategy_version_id": self.strategy_version_id,
            "product_id": self.product_id,
            "artefact_hash": self.artefact_hash,
            "observed_from": self.observed_from,
            "observed_until": self.observed_until,
            "elapsed_days": self.elapsed_days,
            "independent_decisions": self.independent_decisions,
            "net_pnl": self.net_pnl,
            "benchmark_pnl": self.benchmark_pnl,
            "excess_benchmark_pnl": self.excess_benchmark_pnl,
            "drawdown": self.drawdown,
            "execution_drift": self.execution_drift,
            "model_drift": self.model_drift,
            "portfolio_capacity": self.portfolio_capacity,
            "risk_budget_available": self.risk_budget_available,
            "data_gaps": self.data_gaps,
            "strategy_decay": self.strategy_decay,
            "observation_ids": list(self.observation_ids),
            "effective_trades": self.effective_trades,
            "fill_rate": self.fill_rate,
            "slippage": self.slippage,
            "data_uptime": self.data_uptime,
            "rejected_orders": self.rejected_orders,
            "objective_unit": self.objective_unit,
            "objective_value": self.objective_value,
            "benchmark_value": self.benchmark_value,
            "objective_excess": self.objective_excess,
            "objective_excess_fraction": self.objective_excess_fraction,
            "trading_days": self.trading_days,
            "cycles": self.cycles,
            "effective_independent_episodes": self.effective_independent_episodes,
            "tail_loss": self.tail_loss,
        }


def _validated_observation_facts(row: object) -> tuple[str, ...]:
    if not isinstance(row, Mapping):
        raise CanonicalEvidenceError("forward observation facts are missing")
    observation = row.get("observation")
    facts = observation.get("facts") if isinstance(observation, Mapping) else None
    if (
        not isinstance(facts, Mapping)
        or facts.get("schema") != "platform.forward_evidence_facts/v1"
    ):
        raise CanonicalEvidenceError("forward observations need immutable evidence facts")
    saved_hash = facts.get("facts_hash")
    content = dict(facts)
    content.pop("facts_hash", None)
    if saved_hash != canonical_hash(content):
        raise CanonicalEvidenceError("forward evidence facts hash is invalid")
    source_ids = facts.get("source_event_ids")
    metrics = facts.get("metrics")
    if not isinstance(source_ids, list | tuple) or not source_ids:
        raise CanonicalEvidenceError("forward evidence facts need source identities")
    if not isinstance(metrics, Mapping):
        raise CanonicalEvidenceError("forward evidence facts need metrics")
    required_metrics = {
        "net_pnl",
        "benchmark_pnl",
        "drawdown",
        "execution_drift",
        "model_drift",
        "portfolio_capacity",
        "risk_budget_available",
        "data_gaps",
        "effective_trades",
        "fill_rate",
        "slippage",
        "data_uptime",
        "rejected_orders",
    }
    if not required_metrics.issubset(metrics):
        raise CanonicalEvidenceError("forward evidence facts metrics are incomplete")
    return tuple(str(source_id) for source_id in source_ids)


def _has_known_observation_source(connection, source_ids: tuple[str, ...], tables) -> bool:
    for source_id in source_ids:
        if any(
            connection.execute(select(table.c.id).where(table.c.id == source_id)).first()
            for table in tables
        ):
            return True
    return False


_SUMMARY_OBJECTIVE_FIELDS = (
    "objective_value",
    "benchmark_value",
    "objective_excess",
    "objective_excess_fraction",
)
_SUMMARY_NONNEGATIVE_FIELDS = (
    "drawdown",
    "execution_drift",
    "model_drift",
    "portfolio_capacity",
    "risk_budget_available",
    "strategy_decay",
    "slippage",
    "tail_loss",
)
_SUMMARY_FRACTION_FIELDS = ("fill_rate", "data_uptime")
_SUMMARY_INTEGER_FIELDS = (
    "effective_trades",
    "rejected_orders",
    "trading_days",
    "cycles",
    "effective_independent_episodes",
)


def _normalise_summary_objective(payload: dict[str, Any], product_id: str) -> None:
    has_objective = payload.get("objective_unit") is not None or any(
        payload.get(field) is not None for field in _SUMMARY_OBJECTIVE_FIELDS
    )
    if not has_objective:
        return
    expected_unit = objective_unit(product_id)
    if expected_unit is None or payload.get("objective_unit") != expected_unit:
        raise CanonicalEvidenceError("forward summary objective unit is invalid")
    for field_name in _SUMMARY_OBJECTIVE_FIELDS:
        payload[field_name] = _finite_number(payload.get(field_name), field=f"summary {field_name}")
    payload["objective_unit"] = expected_unit


def _normalise_summary_metrics(payload: dict[str, Any]) -> None:
    for field_name in _SUMMARY_NONNEGATIVE_FIELDS:
        payload[field_name] = _finite_nonnegative(
            payload.get(field_name, 0.0), field=f"summary {field_name}"
        )
    for field_name in _SUMMARY_FRACTION_FIELDS:
        value = _finite_nonnegative(payload.get(field_name, 0.0), field=f"summary {field_name}")
        if value > 1:
            raise CanonicalEvidenceError(f"summary {field_name} must be at most one")
        payload[field_name] = value
    for field_name in _SUMMARY_INTEGER_FIELDS:
        payload[field_name] = int(
            _finite_nonnegative(payload.get(field_name, 0), field=f"summary {field_name}")
        )
    payload["data_gaps"] = int(
        _finite_nonnegative(payload.get("data_gaps", 0), field="summary data gaps")
    )


def _summary_observation_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    observation_ids = payload.get("observation_ids", ())
    if (
        not isinstance(observation_ids, list | tuple)
        or not observation_ids
        or any(not str(value).strip() for value in observation_ids)
    ):
        raise CanonicalEvidenceError("forward summary needs observation identities")
    normalised = tuple(
        _identity(str(value), field="summary observation identity") for value in observation_ids
    )
    if len(set(normalised)) != len(normalised):
        raise CanonicalEvidenceError("forward summary observation identities must be unique")
    payload["observation_ids"] = list(normalised)
    return normalised


def _prepare_summary_payload(
    *,
    strategy_version_id: str,
    product_id: str,
    observed_at: str,
    artefact_hash: str | None,
    evidence: Mapping[str, Any],
) -> tuple[str, str, str, dict[str, Any], tuple[str, ...], str]:
    strategy_version_id = str(strategy_version_id).strip()
    product_id = str(product_id).strip()
    if not strategy_version_id or not product_id:
        raise CanonicalEvidenceError("forward summaries require strategy and product identities")
    observed_at = timestamp(observed_at, field="summary.observed_at")
    payload = _object(evidence, field="forward summary")
    if "accepted" in payload:
        raise CanonicalEvidenceError(
            "artefact-bound forward summary cannot contain a raw acceptance flag"
        )
    artefact = _identity(
        artefact_hash or str(payload.get("artefact_hash") or ""), field="artefact_hash"
    )
    payload.update(
        {
            "strategy_version_id": strategy_version_id,
            "product_id": product_id,
            "artefact_hash": artefact,
            "observed_from": timestamp(
                str(payload.get("observed_from") or observed_at), field="summary.observed_from"
            ),
            "observed_until": timestamp(
                str(payload.get("observed_until") or observed_at), field="summary.observed_until"
            ),
            "elapsed_days": _finite_nonnegative(
                payload.get("elapsed_days", 0.0), field="summary elapsed days"
            ),
            "independent_decisions": int(
                _finite_nonnegative(
                    payload.get("independent_decisions", 0),
                    field="summary independent decisions",
                )
            ),
            "net_pnl": _finite_number(payload.get("net_pnl", 0.0), field="summary net_pnl"),
        }
    )
    if payload["observed_until"] < payload["observed_from"]:
        raise CanonicalEvidenceError("forward summary interval is not chronological")
    _normalise_summary_objective(payload, product_id)
    _normalise_summary_metrics(payload)
    observation_ids = _summary_observation_ids(payload)
    identity = _hash(payload, field="forward summary")
    return strategy_version_id, product_id, observed_at, payload, observation_ids, identity


def _assert_summary_bindings(
    connection,
    *,
    strategy_version_id: str,
    product_id: str,
    artefact: str,
    payload: Mapping[str, Any],
    observation_ids: tuple[str, ...],
) -> None:
    if (
        connection.execute(
            select(strategy_version.c.id).where(strategy_version.c.id == strategy_version_id)
        ).first()
        is None
    ):
        raise CanonicalEvidenceError(f"strategy version does not exist: {strategy_version_id}")
    artefact_payload = _assert_canonical_artifact(connection, artefact)
    _assert_artefact_binding(
        artefact_payload,
        strategy_version_id=strategy_version_id,
        product_id=product_id,
    )
    if payload["observed_from"] <= timestamp(
        str(artefact_payload["created_at"]), field="artefact.created_at"
    ):
        raise CanonicalEvidenceError("forward summary must follow artefact creation")
    observations = (
        connection.execute(
            select(
                forward_paper_observation.c.id,
                forward_paper_observation.c.strategy_version_id,
                forward_paper_observation.c.product_id,
                forward_paper_observation.c.artefact_hash,
                forward_paper_observation.c.observed_at,
            ).where(forward_paper_observation.c.id.in_(observation_ids))
        )
        .mappings()
        .all()
    )
    if len(observations) != len(observation_ids):
        raise CanonicalEvidenceError("forward summary references an unknown observation identity")
    for observation_row in observations:
        if (
            str(observation_row["strategy_version_id"]) != strategy_version_id
            or str(observation_row["product_id"]) != product_id
            or str(observation_row["artefact_hash"]) != artefact
            or not payload["observed_from"]
            <= str(observation_row["observed_at"])
            <= payload["observed_until"]
        ):
            raise CanonicalEvidenceError(
                "forward summary observation binding or interval is invalid"
            )


class SqlForwardEvidenceRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(
        self,
        *,
        strategy_version_id: str,
        product_id: str,
        instrument_id: str,
        observed_at: str,
        artefact_hash: str,
        observation: Mapping[str, Any],
        evaluation_time: str | None = None,
    ) -> str:
        observed_at = timestamp(observed_at, field="observed_at")
        evaluation_at = (
            timestamp(evaluation_time, field="evaluation_time")
            if evaluation_time is not None
            else None
        )
        strategy_version_id = str(strategy_version_id).strip()
        product_id = str(product_id).strip()
        instrument_id = str(instrument_id).strip()
        if not strategy_version_id or not product_id or not instrument_id:
            raise CanonicalEvidenceError(
                "forward observations require strategy, product, and instrument identities"
            )
        artefact_hash = _identity(artefact_hash, field="artefact_hash")
        payload = _object(observation, field="forward observation")
        if "accepted" in payload:
            raise CanonicalEvidenceError(
                "forward observations are immutable facts and cannot contain acceptance"
            )
        identity_payload = {
            "strategy_version_id": strategy_version_id,
            "product_id": product_id,
            "instrument_id": instrument_id,
            "observed_at": observed_at,
            "artefact_hash": artefact_hash,
            "observation": payload,
        }
        if evaluation_at is not None:
            identity_payload["evaluation_time"] = evaluation_at
        observation_hash = _hash(identity_payload, field="forward observation")
        with self.engine.begin() as connection:
            if (
                connection.execute(
                    select(strategy_version.c.id).where(
                        strategy_version.c.id == strategy_version_id
                    )
                ).first()
                is None
            ):
                raise CanonicalEvidenceError(
                    f"strategy version does not exist: {strategy_version_id}"
                )
            artefact = _assert_canonical_artifact(connection, artefact_hash)
            created_at = artefact.get("created_at")
            if not isinstance(created_at, str):
                raise CanonicalEvidenceError("canonical artefact creation timestamp is missing")
            if observed_at <= timestamp(created_at, field="artefact.created_at"):
                raise CanonicalEvidenceError(
                    "forward observation must occur after artefact creation"
                )
            if evaluation_at is not None and observed_at > evaluation_at:
                raise CanonicalEvidenceError(
                    "forward observation must not occur after evaluation time"
                )
            _assert_artefact_binding(
                artefact,
                strategy_version_id=strategy_version_id,
                product_id=product_id,
            )
            supported_products = set(artefact.get("supported_products") or [])
            supported_instruments = set(artefact.get("supported_instruments") or [])
            if product_id not in supported_products or instrument_id not in supported_instruments:
                raise CanonicalEvidenceError("forward observation is outside artefact support")
            return _immutable_insert(
                connection,
                forward_paper_observation,
                {
                    "id": observation_hash,
                    "strategy_version_id": strategy_version_id,
                    "product_id": product_id,
                    "instrument_id": instrument_id,
                    "observed_at": observed_at,
                    "artefact_hash": artefact_hash,
                    "observation_hash": observation_hash,
                    "payload": identity_payload,
                },
            )

    def append_summary(
        self,
        *,
        strategy_version_id: str,
        product_id: str,
        observed_at: str,
        artefact_hash: str | None = None,
        evidence: Mapping[str, Any],
    ) -> str:
        (
            strategy_version_id,
            product_id,
            observed_at,
            payload,
            normalised_observation_ids,
            identity,
        ) = _prepare_summary_payload(
            strategy_version_id=strategy_version_id,
            product_id=product_id,
            observed_at=observed_at,
            artefact_hash=artefact_hash,
            evidence=evidence,
        )
        artefact = str(payload["artefact_hash"])
        with self.engine.begin() as connection:
            _assert_summary_bindings(
                connection,
                strategy_version_id=strategy_version_id,
                product_id=product_id,
                artefact=artefact,
                payload=payload,
                observation_ids=normalised_observation_ids,
            )
            self._assert_observation_facts(connection, normalised_observation_ids)
            return _immutable_insert(
                connection,
                forward_paper_summary,
                {
                    "id": identity,
                    "strategy_version_id": strategy_version_id,
                    "product_id": product_id,
                    "artefact_hash": artefact,
                    "observed_from": payload["observed_from"],
                    "observed_until": payload["observed_until"],
                    "created_at": observed_at,
                    "content_hash": identity,
                    "payload": payload,
                },
            )

    def build_summary(
        self,
        *,
        strategy_version_id: str,
        product_id: str,
        artefact_hash: str,
        observed_at: str,
    ) -> tuple[str, ForwardPaperSummary]:
        observed_at = timestamp(observed_at, field="summary.observed_at")
        artefact_hash = _identity(artefact_hash, field="artefact_hash")
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    forward_paper_observation.c.id,
                    forward_paper_observation.c.observed_at,
                    forward_paper_observation.c.payload,
                )
                .where(
                    forward_paper_observation.c.strategy_version_id == strategy_version_id,
                    forward_paper_observation.c.product_id == product_id,
                    forward_paper_observation.c.artefact_hash == artefact_hash,
                    forward_paper_observation.c.observed_at <= observed_at,
                )
                .order_by(forward_paper_observation.c.observed_at, forward_paper_observation.c.id)
            ).mappings()
            materialised = tuple(row for row in rows if isinstance(row["payload"], Mapping))
        if not materialised:
            raise CanonicalEvidenceError("no forward observations exist for summary")
        with self.engine.connect() as connection:
            self._assert_observation_facts(
                connection, tuple(str(row["id"]) for row in materialised)
            )

        def value(payload: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
            source = payload["observation"]
            facts = source["facts"]
            metrics = facts["metrics"]
            for name in names:
                if metrics.get(name) is not None:
                    if name in {"net_pnl", "cost_adjusted_return", "pnl"}:
                        return _finite_number(metrics[name], field=f"forward observation {name}")
                    return _finite_nonnegative(metrics[name], field=f"forward observation {name}")
            return default

        def optional_value(payload: Mapping[str, Any], name: str) -> float | None:
            source = payload["observation"]
            metrics = source["facts"]["metrics"]
            raw = metrics.get(name)
            return None if raw is None else _finite_number(raw, field=f"forward observation {name}")

        observed_from = min(str(row["observed_at"]) for row in materialised)
        observed_until = max(str(row["observed_at"]) for row in materialised)
        elapsed = max(
            0.0,
            (
                dt.datetime.fromisoformat(observed_until) - dt.datetime.fromisoformat(observed_from)
            ).total_seconds()
            / 86_400,
        )
        payloads = [dict(row["payload"]) for row in materialised]
        decisions = {
            str(
                payload["observation"].get(
                    "decision_id",
                    payload["observation"].get("forecast_id", row_id),
                )
            )
            for payload, row_id in zip(
                payloads, (str(row["id"]) for row in materialised), strict=False
            )
        }
        returns = [value(payload, "net_pnl", "cost_adjusted_return", "pnl") for payload in payloads]
        benchmark_returns = [
            value(payload, "benchmark_pnl", "benchmark_return") for payload in payloads
        ]
        objective_units = tuple(
            str(payload["observation"]["facts"]["metrics"].get("objective_unit"))
            for payload in payloads
            if payload["observation"]["facts"]["metrics"].get("objective_unit") is not None
        )
        objective_values = [optional_value(payload, "objective_value") for payload in payloads]
        benchmark_values = [optional_value(payload, "benchmark_value") for payload in payloads]
        objective_excesses = [optional_value(payload, "objective_excess") for payload in payloads]
        objective_excess_fractions = [
            optional_value(payload, "objective_excess_fraction") for payload in payloads
        ]
        complete_objective = all(
            value is not None
            for value in (
                objective_values[-1] if objective_values else None,
                benchmark_values[-1] if benchmark_values else None,
                objective_excesses[-1] if objective_excesses else None,
                objective_excess_fractions[-1] if objective_excess_fractions else None,
            )
        )
        first_return = returns[0] if returns else 0.0
        last_return = returns[-1] if returns else 0.0
        summary = ForwardPaperSummary(
            strategy_version_id=strategy_version_id,
            product_id=product_id,
            artefact_hash=artefact_hash,
            observed_from=observed_from,
            observed_until=observed_until,
            elapsed_days=max(
                elapsed,
                max(value(payload, "evidence_days") for payload in payloads),
            ),
            independent_decisions=len(decisions),
            net_pnl=sum(returns),
            benchmark_pnl=sum(benchmark_returns),
            excess_benchmark_pnl=sum(returns) - sum(benchmark_returns),
            drawdown=max(value(payload, "drawdown") for payload in payloads),
            execution_drift=max(value(payload, "execution_drift") for payload in payloads),
            model_drift=max(value(payload, "model_drift") for payload in payloads),
            portfolio_capacity=max(value(payload, "portfolio_capacity") for payload in payloads),
            risk_budget_available=min(
                value(payload, "risk_budget_available", default=0.0) for payload in payloads
            ),
            data_gaps=int(
                sum(value(payload, "data_gaps", "missing_data_count") for payload in payloads)
            ),
            strategy_decay=max(0.0, first_return - last_return),
            observation_ids=tuple(str(row["id"]) for row in materialised),
            effective_trades=int(sum(value(payload, "effective_trades") for payload in payloads)),
            fill_rate=(
                sum(value(payload, "fill_rate") for payload in payloads) / len(payloads)
                if payloads
                else 0.0
            ),
            slippage=max(value(payload, "slippage") for payload in payloads),
            data_uptime=(
                sum(value(payload, "data_uptime") for payload in payloads) / len(payloads)
                if payloads
                else 0.0
            ),
            rejected_orders=int(sum(value(payload, "rejected_orders") for payload in payloads)),
            trading_days=int(sum(value(payload, "trading_days") for payload in payloads)),
            cycles=int(sum(value(payload, "cycles") for payload in payloads)),
            effective_independent_episodes=int(
                sum(value(payload, "effective_independent_episodes") for payload in payloads)
            ),
            tail_loss=max(value(payload, "tail_loss") for payload in payloads),
            objective_unit=objective_units[-1] if objective_units and complete_objective else None,
            objective_value=objective_values[-1] if complete_objective else None,
            benchmark_value=benchmark_values[-1] if complete_objective else None,
            objective_excess=objective_excesses[-1] if complete_objective else None,
            objective_excess_fraction=(
                objective_excess_fractions[-1] if complete_objective else None
            ),
        )
        summary_id = self.append_summary(
            strategy_version_id=strategy_version_id,
            product_id=product_id,
            observed_at=observed_at,
            artefact_hash=artefact_hash,
            evidence=summary.to_payload(),
        )
        return summary_id, summary

    @staticmethod
    def _assert_observation_facts(connection, observation_ids: tuple[str, ...]) -> None:
        known_tables = (
            strategy_version,
            alpha_forecast,
            target_position,
            order_intent,
            fill,
            risk_snapshot,
            accounting_entry,
            strategy_artefact,
        )
        for observation_id in observation_ids:
            row = connection.execute(
                select(forward_paper_observation.c.payload).where(
                    forward_paper_observation.c.id == observation_id
                )
            ).scalar_one_or_none()
            source_ids = _validated_observation_facts(row)
            if not _has_known_observation_source(connection, source_ids, known_tables):
                raise CanonicalEvidenceError("forward evidence facts reference no canonical source")

    def append_decision(
        self,
        *,
        summary_id: str,
        decided_at: str,
        accepted: bool,
        reason_code: str | None = None,
    ) -> str:
        if not isinstance(accepted, bool):
            raise CanonicalEvidenceError("forward decision accepted must be a boolean")
        if not accepted and not str(reason_code or "").strip():
            raise CanonicalEvidenceError("rejected forward decisions need a reason code")
        decided_at = timestamp(decided_at, field="forward decision timestamp")
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(forward_paper_summary).where(forward_paper_summary.c.id == summary_id)
                )
                .mappings()
                .first()
            )
        if row is None or not isinstance(row["payload"], Mapping):
            raise CanonicalEvidenceError("forward summary does not exist")
        payload = {
            "schema": "platform.forward_paper_decision/v1",
            "summary_id": summary_id,
            "strategy_version_id": str(row["strategy_version_id"]),
            "product_id": str(row["product_id"]),
            "artefact_hash": str(row["artefact_hash"]),
            "decided_at": decided_at,
            "accepted": accepted,
            "reason_code": reason_code,
        }
        identity = _hash(payload, field="forward paper decision")
        with self.engine.begin() as connection:
            return _immutable_insert(
                connection,
                forward_paper_decision,
                {
                    "id": identity,
                    "summary_id": summary_id,
                    "strategy_version_id": str(row["strategy_version_id"]),
                    "product_id": str(row["product_id"]),
                    "artefact_hash": str(row["artefact_hash"]),
                    "decided_at": decided_at,
                    "accepted": accepted,
                    "reason_code": reason_code,
                    "content_hash": identity,
                    "payload": payload,
                },
            )

    def decide_summary(
        self,
        summary_id: str,
        *,
        decided_at: str,
        minimum_days: int,
        minimum_decisions: int = 1,
        minimum_net_pnl: float = 0.0,
        minimum_objective_excess_fraction: float | None = None,
        maximum_drawdown: float = 1.0,
        maximum_data_gaps: int = 0,
        minimum_effective_trades: int = 0,
        minimum_fill_rate: float = 0.0,
        maximum_slippage: float = 1.0,
        minimum_data_uptime: float = 0.0,
        maximum_rejected_orders: int = 0,
        minimum_trading_days: int = 0,
        minimum_cycles: int = 0,
        minimum_effective_episodes: int = 0,
        maximum_tail_loss: float = 1.0,
    ) -> tuple[str, bool, str | None]:
        limits = _normalise_decision_limits(
            minimum_days=minimum_days,
            minimum_decisions=minimum_decisions,
            minimum_net_pnl=minimum_net_pnl,
            minimum_objective_excess_fraction=minimum_objective_excess_fraction,
            maximum_drawdown=maximum_drawdown,
            maximum_data_gaps=maximum_data_gaps,
            minimum_effective_trades=minimum_effective_trades,
            minimum_fill_rate=minimum_fill_rate,
            maximum_slippage=maximum_slippage,
            minimum_data_uptime=minimum_data_uptime,
            maximum_rejected_orders=maximum_rejected_orders,
            minimum_trading_days=minimum_trading_days,
            minimum_cycles=minimum_cycles,
            minimum_effective_episodes=minimum_effective_episodes,
            maximum_tail_loss=maximum_tail_loss,
        )
        decided_at = timestamp(decided_at, field="forward decision timestamp")
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(forward_paper_summary.c.payload).where(
                    forward_paper_summary.c.id == summary_id
                )
            ).scalar_one_or_none()
        if not isinstance(payload, Mapping):
            raise CanonicalEvidenceError("forward summary does not exist")
        checks = _forward_decision_checks(payload, limits)
        reason = next((reason for failed, reason in checks if failed), None)
        decision_id = self.append_decision(
            summary_id=summary_id,
            decided_at=decided_at,
            accepted=reason is None,
            reason_code=reason,
        )
        return decision_id, reason is None, reason


@dataclass(frozen=True)
class _ForwardDecisionLimits:
    minimum_days: int
    minimum_decisions: int
    minimum_net_pnl: float
    minimum_objective_excess_fraction: float | None
    maximum_drawdown: float
    maximum_data_gaps: int
    minimum_effective_trades: int
    minimum_fill_rate: float
    maximum_slippage: float
    minimum_data_uptime: float
    maximum_rejected_orders: int
    minimum_trading_days: int
    minimum_cycles: int
    minimum_effective_episodes: int
    maximum_tail_loss: float


def _decision_integer(value: int, *, message: str) -> int:
    if isinstance(value, bool) or value < 0:
        raise CanonicalEvidenceError(message)
    return value


def _decision_fraction(value: float, *, field: str, at_most_one: bool = False) -> float:
    result = _finite_nonnegative(value, field=field)
    if at_most_one and result > 1:
        raise CanonicalEvidenceError(f"{field} must be at most one")
    return result


def _normalise_decision_limits(
    *,
    minimum_days: int,
    minimum_decisions: int,
    minimum_net_pnl: float,
    minimum_objective_excess_fraction: float | None,
    maximum_drawdown: float,
    maximum_data_gaps: int,
    minimum_effective_trades: int,
    minimum_fill_rate: float,
    maximum_slippage: float,
    minimum_data_uptime: float,
    maximum_rejected_orders: int,
    minimum_trading_days: int,
    minimum_cycles: int,
    minimum_effective_episodes: int,
    maximum_tail_loss: float,
) -> _ForwardDecisionLimits:
    objective = (
        None
        if minimum_objective_excess_fraction is None
        else _finite_number(
            minimum_objective_excess_fraction,
            field="minimum forward objective excess fraction",
        )
    )
    return _ForwardDecisionLimits(
        minimum_days=_decision_integer(
            minimum_days, message="minimum forward days must be non-negative"
        ),
        minimum_decisions=_decision_integer(
            minimum_decisions, message="minimum forward decisions must be non-negative"
        ),
        minimum_net_pnl=_finite_number(minimum_net_pnl, field="minimum forward net_pnl"),
        minimum_objective_excess_fraction=objective,
        maximum_drawdown=_finite_nonnegative(maximum_drawdown, field="maximum forward drawdown"),
        maximum_data_gaps=_decision_integer(
            maximum_data_gaps, message="maximum forward data gaps must be non-negative"
        ),
        minimum_effective_trades=_decision_integer(
            minimum_effective_trades,
            message="minimum effective forward trades must be non-negative",
        ),
        minimum_fill_rate=_decision_fraction(
            minimum_fill_rate, field="minimum forward fill rate", at_most_one=True
        ),
        maximum_slippage=_finite_nonnegative(maximum_slippage, field="maximum forward slippage"),
        minimum_data_uptime=_decision_fraction(
            minimum_data_uptime, field="minimum forward data uptime", at_most_one=True
        ),
        maximum_rejected_orders=_decision_integer(
            maximum_rejected_orders,
            message="maximum rejected forward orders must be non-negative",
        ),
        minimum_trading_days=_decision_integer(
            minimum_trading_days,
            message="minimum forward trading days must be a non-negative integer",
        ),
        minimum_cycles=_decision_integer(
            minimum_cycles,
            message="minimum forward cycles must be a non-negative integer",
        ),
        minimum_effective_episodes=_decision_integer(
            minimum_effective_episodes,
            message="minimum forward independent episodes must be a non-negative integer",
        ),
        maximum_tail_loss=_finite_nonnegative(maximum_tail_loss, field="maximum forward tail loss"),
    )


def _objective_decision_failed(payload: Mapping[str, Any], threshold: float | None) -> bool:
    if threshold is None:
        return False
    expected_unit = objective_unit(str(payload.get("product_id") or ""))
    return (
        expected_unit is None
        or payload.get("objective_unit") != expected_unit
        or not all(
            payload.get(field) is not None
            for field in (
                "objective_value",
                "benchmark_value",
                "objective_excess",
                "objective_excess_fraction",
            )
        )
        or float(payload.get("objective_excess_fraction", 0.0)) <= threshold
    )


def _forward_decision_checks(
    payload: Mapping[str, Any], limits: _ForwardDecisionLimits
) -> tuple[tuple[bool, str], ...]:
    objective_required = limits.minimum_objective_excess_fraction is not None
    objective_failed = _objective_decision_failed(payload, limits.minimum_objective_excess_fraction)
    return (
        (
            float(payload.get("elapsed_days", 0.0)) < limits.minimum_days,
            "forward_evidence_duration_insufficient",
        ),
        (
            int(payload.get("independent_decisions", 0)) < limits.minimum_decisions,
            "forward_decisions_insufficient",
        ),
        (
            int(payload.get("trading_days", 0)) < limits.minimum_trading_days,
            "forward_trading_days_insufficient",
        ),
        (
            int(payload.get("cycles", 0)) < limits.minimum_cycles,
            "forward_cycles_insufficient",
        ),
        (
            int(payload.get("effective_independent_episodes", 0))
            < limits.minimum_effective_episodes,
            "forward_independent_episodes_insufficient",
        ),
        (
            objective_failed
            if objective_required
            else float(payload.get("net_pnl", 0.0)) <= limits.minimum_net_pnl,
            (
                "forward_objective_excess_threshold"
                if objective_required
                else "forward_net_pnl_threshold"
            ),
        ),
        (
            float(payload.get("drawdown", 0.0)) > limits.maximum_drawdown,
            "forward_drawdown_limit",
        ),
        (
            float(payload.get("tail_loss", 0.0)) > limits.maximum_tail_loss,
            "forward_tail_loss_limit",
        ),
        (int(payload.get("data_gaps", 0)) > limits.maximum_data_gaps, "forward_data_gaps"),
        (
            int(payload.get("effective_trades", 0)) < limits.minimum_effective_trades,
            "forward_effective_trades_insufficient",
        ),
        (
            float(payload.get("fill_rate", 0.0)) < limits.minimum_fill_rate,
            "forward_fill_rate_insufficient",
        ),
        (
            float(payload.get("slippage", 0.0)) > limits.maximum_slippage,
            "forward_slippage_limit",
        ),
        (
            float(payload.get("data_uptime", 0.0)) < limits.minimum_data_uptime,
            "forward_data_uptime_insufficient",
        ),
        (
            int(payload.get("rejected_orders", 0)) > limits.maximum_rejected_orders,
            "forward_rejected_orders_limit",
        ),
        (
            float(payload.get("portfolio_capacity", 0.0)) <= 0.0,
            "portfolio_capacity_unavailable",
        ),
        (
            float(payload.get("risk_budget_available", 0.0)) <= 0.0,
            "risk_budget_unavailable",
        ),
    )


class SqlStrategyArtefactRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def put(self, artefact_hash: str, payload: Mapping[str, Any], *, created_at: str) -> str:
        payload = _object(payload, field="strategy artefact")
        if payload.get("schema") != "platform.strategy_artefact/v2":
            raise CanonicalEvidenceError("only StrategyArtefact v2 may enter canonical storage")
        authoritative = payload.get("authoritative_evidence")
        if not isinstance(authoritative, Mapping):
            raise CanonicalEvidenceError("canonical strategy artefact needs authoritative evidence")
        if "forward_evidence_id" in authoritative or "forward_evidence" in payload:
            raise CanonicalEvidenceError(
                "canonical strategy artefacts cannot contain pre-artefact forward evidence fields"
            )
        required_fields = (
            "strategy_version_id",
            "product_id",
            "portfolio_id",
            "account_id",
            "promotion_policy_id",
            "engine_version",
        )
        missing_fields = [
            field for field in required_fields if not str(payload.get(field) or "").strip()
        ]
        if missing_fields:
            raise CanonicalEvidenceError(
                "canonical strategy artefact is missing fields: " + ", ".join(missing_fields)
            )
        artefact_hash = _identity(artefact_hash, field="artefact_hash")
        if payload.get("artefact_hash") != artefact_hash:
            raise CanonicalEvidenceError("strategy artefact hash does not match its payload")
        content = dict(payload)
        content.pop("artefact_hash", None)
        if canonical_hash(content) != artefact_hash:
            raise CanonicalEvidenceError("strategy artefact content hash is invalid")
        created_at = timestamp(created_at, field="created_at")
        if (
            timestamp(str(payload.get("created_at") or ""), field="artefact.created_at")
            != created_at
        ):
            raise CanonicalEvidenceError(
                "strategy artefact payload creation timestamp does not match storage"
            )
        with self.engine.begin() as connection:
            version_row = connection.execute(
                select(strategy_version.c.id, strategy_definition.c.product_id)
                .select_from(
                    strategy_version.join(
                        strategy_definition,
                        strategy_version.c.definition_id == strategy_definition.c.id,
                    )
                )
                .where(strategy_version.c.id == payload["strategy_version_id"])
            ).first()
            if version_row is None:
                raise CanonicalEvidenceError(
                    f"strategy version does not exist: {payload['strategy_version_id']}"
                )
            if str(payload["product_id"]) != str(version_row.product_id):
                raise CanonicalEvidenceError(
                    "canonical artefact product does not match its version"
                )
            return _immutable_insert(
                connection,
                strategy_artefact,
                {"id": artefact_hash, "created_at": created_at, "payload": payload},
            )

    def get(self, artefact_hash: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            try:
                return _assert_canonical_artifact(connection, artefact_hash)
            except CanonicalEvidenceError as exc:
                if "does not exist" in str(exc):
                    raise KeyError(f"strategy artefact does not exist: {artefact_hash}") from exc
                raise


class SqlApprovalRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(
        self,
        *,
        strategy_version_id: str,
        product_id: str,
        account_id: str,
        artefact_hash: str,
        source_commit_hash: str,
        engine_version: str,
        capital_cap: float,
        actor: str,
        approved_at: str,
        status: str = "approved",
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        actor = _human_actor(actor, field="human approval actor")
        capital_cap = _finite_nonnegative(capital_cap, field="approval capital cap")
        strategy_version_id = str(strategy_version_id).strip()
        product_id = str(product_id).strip()
        account_id = str(account_id).strip()
        engine_version = str(engine_version).strip()
        if not strategy_version_id or not product_id or not account_id or not engine_version:
            raise CanonicalEvidenceError("approval binding fields cannot be empty")
        artefact_hash = _identity(artefact_hash, field="artefact_hash")
        source_commit_hash = _identity(source_commit_hash, field="source_commit_hash")
        if status not in {"approved", "revoked", "expired"}:
            raise CanonicalEvidenceError("approval status is not supported")
        approved_at = timestamp(approved_at, field="approved_at")
        record = {
            "strategy_version_id": strategy_version_id,
            "product_id": product_id,
            "account_id": account_id,
            "artefact_hash": artefact_hash,
            "source_commit_hash": source_commit_hash,
            "engine_version": engine_version,
            "capital_cap": capital_cap,
            "approved_by": actor,
            "approved_at": approved_at,
            "status": status,
            "payload": _object(payload or {}, field="approval payload"),
        }
        identity = _hash(record, field="approval")
        with self.engine.begin() as connection:
            if (
                connection.execute(
                    select(strategy_version.c.id).where(
                        strategy_version.c.id == strategy_version_id
                    )
                ).first()
                is None
            ):
                raise CanonicalEvidenceError(
                    f"strategy version does not exist: {strategy_version_id}"
                )
            artefact = _assert_canonical_artifact(connection, artefact_hash)
            _assert_artefact_binding(
                artefact,
                strategy_version_id=strategy_version_id,
                product_id=product_id,
                account_id=account_id,
            )
            return _immutable_insert(connection, strategy_approval, {"id": identity, **record})

    def latest(
        self,
        *,
        strategy_version_id: str,
        product_id: str,
        account_id: str,
        at: str | None = None,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            statement = select(strategy_approval).where(
                strategy_approval.c.strategy_version_id == strategy_version_id,
                strategy_approval.c.product_id == product_id,
                strategy_approval.c.account_id == account_id,
            )
            if at is not None:
                statement = statement.where(
                    strategy_approval.c.approved_at <= timestamp(at, field="approval timestamp")
                )
            row = (
                connection.execute(
                    statement.order_by(
                        strategy_approval.c.approved_at.desc(), strategy_approval.c.id.desc()
                    ).limit(1)
                )
                .mappings()
                .first()
            )
        return None if row is None else dict(row)


class SqlPreflightRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, record: Mapping[str, Any]) -> str:
        payload = _object(record, field="production preflight")
        required = (
            "strategy_version_id",
            "product_id",
            "account_id",
            "artefact_hash",
            "source_commit_hash",
            "engine_version",
            "capital_cap",
            "checked_at",
            "accepted",
        )
        missing = [field for field in required if field not in payload]
        if missing:
            raise CanonicalEvidenceError(f"production preflight is missing fields: {missing}")
        if not isinstance(payload["accepted"], bool):
            raise CanonicalEvidenceError("production preflight accepted must be a boolean")
        if payload["accepted"] is False and not str(payload.get("reason_code") or "").strip():
            raise CanonicalEvidenceError("rejected production preflights need a reason code")
        for field_name in (
            "strategy_version_id",
            "product_id",
            "account_id",
            "engine_version",
        ):
            payload[field_name] = str(payload[field_name]).strip()
            if not payload[field_name]:
                raise CanonicalEvidenceError(f"production preflight {field_name} cannot be empty")
        payload["artefact_hash"] = _identity(payload["artefact_hash"], field="artefact_hash")
        payload["source_commit_hash"] = _identity(
            payload["source_commit_hash"], field="source_commit_hash"
        )
        capital_cap = _finite_nonnegative(payload["capital_cap"], field="preflight capital cap")
        checked_at = timestamp(str(payload["checked_at"]), field="checked_at")
        payload["capital_cap"] = capital_cap
        payload["checked_at"] = checked_at
        content_hash = _hash(payload, field="production preflight")
        identity = content_hash
        with self.engine.begin() as connection:
            if (
                connection.execute(
                    select(strategy_version.c.id).where(
                        strategy_version.c.id == payload["strategy_version_id"]
                    )
                ).first()
                is None
            ):
                raise CanonicalEvidenceError(
                    f"strategy version does not exist: {payload['strategy_version_id']}"
                )
            artefact = _assert_canonical_artifact(connection, payload["artefact_hash"])
            _assert_artefact_binding(
                artefact,
                strategy_version_id=str(payload["strategy_version_id"]),
                product_id=str(payload["product_id"]),
                account_id=str(payload["account_id"]),
            )
            return _immutable_insert(
                connection,
                production_preflight,
                {
                    "id": identity,
                    "strategy_version_id": str(payload["strategy_version_id"]),
                    "product_id": str(payload["product_id"]),
                    "account_id": str(payload["account_id"]),
                    "artefact_hash": str(payload["artefact_hash"]),
                    "source_commit_hash": str(payload["source_commit_hash"]),
                    "engine_version": str(payload["engine_version"]),
                    "capital_cap": capital_cap,
                    "checked_at": checked_at,
                    "content_hash": content_hash,
                    "accepted": payload["accepted"],
                    "payload": payload,
                },
            )

    def latest(
        self,
        *,
        strategy_version_id: str,
        product_id: str,
        account_id: str,
        at: str | None = None,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            statement = select(production_preflight).where(
                production_preflight.c.strategy_version_id == strategy_version_id,
                production_preflight.c.product_id == product_id,
                production_preflight.c.account_id == account_id,
            )
            if at is not None:
                statement = statement.where(
                    production_preflight.c.checked_at <= timestamp(at, field="preflight timestamp")
                )
            row = (
                connection.execute(
                    statement.order_by(
                        production_preflight.c.checked_at.desc(),
                        production_preflight.c.id.desc(),
                    ).limit(1)
                )
                .mappings()
                .first()
            )
        return None if row is None else dict(row)


_ASSIGNMENT_LIFECYCLE_STATES = frozenset(
    {
        "registered",
        "development",
        "forward_paper",
        "live_ready",
        "live_canary",
        "live",
        "suspended",
        "retired",
    }
)


def _assignment_record(
    *,
    product_id: str,
    portfolio_id: str,
    strategy_version_id: str,
    artefact_hash: str,
    lifecycle_state: str,
    execution_mode: str,
    capital_limit: float,
    assigned_at: str,
    assigned_by: str,
    sleeve_id: str,
    instrument_id: str | None,
    universe_id: str | None,
    risk_budget: float,
    active_until: str | None,
    assignment_reason: str,
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if execution_mode not in {"paper", "live"}:
        raise CanonicalEvidenceError("active assignment execution mode must be paper or live")
    if lifecycle_state not in _ASSIGNMENT_LIFECYCLE_STATES:
        raise CanonicalEvidenceError("active assignment lifecycle state is not supported")
    capital_limit = _finite_nonnegative(capital_limit, field="assignment capital limit")
    risk_budget = _finite_nonnegative(risk_budget, field="assignment risk budget")
    product_id = str(product_id).strip()
    portfolio_id = str(portfolio_id).strip()
    strategy_version_id = str(strategy_version_id).strip()
    sleeve_id = str(sleeve_id).strip()
    if not product_id or not portfolio_id or not strategy_version_id or not sleeve_id:
        raise CanonicalEvidenceError("active assignment binding fields cannot be empty")
    source_payload = payload or {}
    if instrument_id is None and universe_id is None:
        instrument_id = str(source_payload.get("instrument_id") or "").strip() or None
        universe_id = str(source_payload.get("universe_id") or "").strip() or None
    if instrument_id is None and universe_id is None:
        universe_id = f"product:{product_id}"
    if (instrument_id is None) == (universe_id is None):
        raise CanonicalEvidenceError("assignment needs exactly one instrument_id or universe_id")
    if not isinstance(assigned_by, str) or not assigned_by.strip():
        raise CanonicalEvidenceError("active assignment actor must be non-empty")
    assigned_at = timestamp(assigned_at, field="assigned_at")
    return {
        "product_id": product_id,
        "portfolio_id": portfolio_id,
        "sleeve_id": sleeve_id,
        "strategy_version_id": strategy_version_id,
        "instrument_id": instrument_id,
        "universe_id": universe_id,
        "assignment_scope_id": (
            f"instrument:{instrument_id}" if instrument_id else f"universe:{universe_id}"
        ),
        "artefact_hash": _identity(artefact_hash, field="artefact_hash"),
        "lifecycle_state": lifecycle_state,
        "execution_mode": execution_mode,
        "capital_limit": capital_limit,
        "risk_budget": risk_budget,
        "assigned_at": assigned_at,
        "active_until": (
            timestamp(active_until, field="active_until") if active_until is not None else None
        ),
        "assigned_by": assigned_by,
        "assignment_reason": non_empty(assignment_reason, field="assignment_reason"),
        "active": True,
        "payload": _object(source_payload, field="assignment payload"),
    }


class SqlActiveStrategyAssignmentRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    @staticmethod
    def _lock_key(record: Mapping[str, Any]) -> str:
        product_id = str(record["product_id"])
        if record["execution_mode"] == "live":
            return (
                f"active-assignment-live:{product_id}:{record['portfolio_id']}:"
                f"{record['sleeve_id']}:{record['instrument_id'] or record['universe_id']}"
            )
        return (
            f"active-assignment:{product_id}:{record['portfolio_id']}:{record['sleeve_id']}:"
            f"{record['strategy_version_id']}:{record['instrument_id'] or record['universe_id']}:"
            f"{record['execution_mode']}"
        )

    @staticmethod
    def _assert_references(connection, record: Mapping[str, Any]) -> dict[str, Any]:
        strategy_version_id = str(record["strategy_version_id"])
        product_id = str(record["product_id"])
        portfolio_id = str(record["portfolio_id"])
        if (
            connection.execute(
                select(strategy_version.c.id).where(strategy_version.c.id == strategy_version_id)
            ).first()
            is None
        ):
            raise CanonicalEvidenceError(f"strategy version does not exist: {strategy_version_id}")
        artefact = _assert_canonical_artifact(connection, str(record["artefact_hash"]))
        _assert_artefact_binding(
            artefact,
            strategy_version_id=strategy_version_id,
            product_id=product_id,
            portfolio_id=portfolio_id,
        )
        assigned_at = str(record["assigned_at"])
        if assigned_at < timestamp(str(artefact["created_at"]), field="artefact.created_at"):
            raise CanonicalEvidenceError(
                "active assignment cannot activate before artefact creation"
            )
        active_until = record["active_until"]
        if isinstance(active_until, str) and active_until <= assigned_at:
            raise CanonicalEvidenceError("active assignment expiry must be after activation")
        return artefact

    @staticmethod
    def _authority_rows(connection, record: Mapping[str, Any], artefact: Mapping[str, Any]):
        account_id = str(artefact.get("account_id") or "")
        if (
            not account_id
            or not artefact.get("source_commit_hash")
            or not artefact.get("engine_version")
        ):
            raise CanonicalEvidenceError(
                "live assignment artefact has incomplete authority bindings"
            )
        strategy_version_id = str(record["strategy_version_id"])
        product_id = str(record["product_id"])
        assigned_at = str(record["assigned_at"])
        approval = (
            connection.execute(
                select(strategy_approval)
                .where(
                    strategy_approval.c.strategy_version_id == strategy_version_id,
                    strategy_approval.c.product_id == product_id,
                    strategy_approval.c.account_id == account_id,
                    strategy_approval.c.approved_at <= assigned_at,
                )
                .order_by(
                    strategy_approval.c.approved_at.desc(),
                    strategy_approval.c.id.desc(),
                )
                .limit(1)
            )
            .mappings()
            .first()
        )
        preflight = (
            connection.execute(
                select(production_preflight)
                .where(
                    production_preflight.c.strategy_version_id == strategy_version_id,
                    production_preflight.c.product_id == product_id,
                    production_preflight.c.account_id == account_id,
                    production_preflight.c.checked_at <= assigned_at,
                )
                .order_by(
                    production_preflight.c.checked_at.desc(),
                    production_preflight.c.id.desc(),
                )
                .limit(1)
            )
            .mappings()
            .first()
        )
        forward = latest_accepted_forward_summary(
            connection.engine,
            strategy_version_id=strategy_version_id,
            product_id=product_id,
            artefact_hash=str(record["artefact_hash"]),
            at=assigned_at,
        )
        return approval, preflight, forward

    @staticmethod
    def _assert_live_authority(
        connection, record: Mapping[str, Any], artefact: Mapping[str, Any]
    ) -> None:
        approval, preflight, forward = SqlActiveStrategyAssignmentRepository._authority_rows(
            connection, record, artefact
        )
        approval_payload = (
            approval["payload"]
            if approval is not None and isinstance(approval["payload"], Mapping)
            else {}
        )
        preflight_payload = (
            preflight["payload"]
            if preflight is not None and isinstance(preflight["payload"], Mapping)
            else {}
        )
        forward_summary = forward["summary"] if forward is not None else None
        forward_decision = forward["decision"] if forward is not None else None
        instrument_id = record["instrument_id"]
        sleeve_id = str(record["sleeve_id"])
        exact_authority = (
            instrument_id is not None
            and approval_payload.get("schema") == "platform.strategy-approval/v1"
            and preflight_payload.get("schema") == "platform.production-preflight/v1"
            and approval_payload.get("preflight_id")
            == (preflight["id"] if preflight is not None else None)
            and approval_payload.get("instrument_id") == instrument_id
            and preflight_payload.get("instrument_id") == instrument_id
            and approval_payload.get("sleeve_id") == sleeve_id
            and preflight_payload.get("sleeve_id") == sleeve_id
            and approval_payload.get("environment") == preflight_payload.get("environment")
            and approval_payload.get("environment") in {"testnet", "production"}
            and isinstance(approval_payload.get("account_fingerprint"), str)
            and bool(approval_payload.get("account_fingerprint"))
            and approval_payload.get("account_fingerprint")
            == preflight_payload.get("account_fingerprint")
            and approval_payload.get("execution_engine_identity")
            == preflight_payload.get("execution_engine_identity")
            and approval_payload.get("configuration_hash")
            == preflight_payload.get("configuration_hash")
            and approval_payload.get("forward_summary_id")
            == (forward_summary["id"] if forward_summary is not None else None)
            and approval_payload.get("forward_decision_id")
            == (forward_decision["id"] if forward_decision is not None else None)
        )
        if exact_authority:
            _identity(
                approval_payload["execution_engine_identity"],
                field="live execution engine identity",
            )
            _identity(approval_payload["configuration_hash"], field="live configuration hash")
        source_commit_hash = str(artefact.get("source_commit_hash") or "")
        engine_version = str(artefact.get("engine_version") or "")
        assigned_at = str(record["assigned_at"])
        valid = (
            approval is not None
            and approval["status"] == "approved"
            and approval["artefact_hash"] == record["artefact_hash"]
            and approval["source_commit_hash"] == source_commit_hash
            and approval["engine_version"] == engine_version
            and _human_actor(approval["approved_by"], field="approval actor")
            == approval["approved_by"]
            and preflight is not None
            and preflight["accepted"]
            and preflight["artefact_hash"] == record["artefact_hash"]
            and preflight["source_commit_hash"] == source_commit_hash
            and preflight["engine_version"] == engine_version
            and exact_authority
            and preflight_is_fresh(str(preflight["checked_at"]), reference_at=assigned_at)
        )
        if not valid:
            raise CanonicalEvidenceError(
                "live assignment requires matching approval and fresh accepted preflight"
            )
        if float(record["capital_limit"]) > min(
            float(approval["capital_cap"]), float(preflight["capital_cap"])
        ):
            raise CanonicalEvidenceError(
                "live assignment exceeds approved or preflight capital cap"
            )

    @staticmethod
    def _assert_no_live_conflict(connection, record: Mapping[str, Any]) -> None:
        product_id = str(record["product_id"])
        portfolio_id = str(record["portfolio_id"])
        sleeve_id = str(record["sleeve_id"])
        scope_id = str(record["assignment_scope_id"])
        assigned_at = str(record["assigned_at"])
        rows = (
            connection.execute(
                select(active_strategy_assignment)
                .where(
                    active_strategy_assignment.c.product_id == product_id,
                    active_strategy_assignment.c.portfolio_id == portfolio_id,
                    active_strategy_assignment.c.sleeve_id == sleeve_id,
                    active_strategy_assignment.c.assignment_scope_id == scope_id,
                    active_strategy_assignment.c.execution_mode == "live",
                )
                .order_by(
                    active_strategy_assignment.c.assigned_at,
                    active_strategy_assignment.c.id,
                )
            )
            .mappings()
            .all()
        )
        if rows and assigned_at < max(str(row["assigned_at"]) for row in rows):
            raise CanonicalEvidenceError("live assignment cannot predate existing live authority")
        if any(
            row["active"] and (row["active_until"] is None or row["active_until"] > assigned_at)
            for row in SqlActiveStrategyAssignmentRepository._current_assignment_rows(rows)
        ):
            raise CanonicalEvidenceError(
                f"product {product_id} already has an active live assignment"
            )

    def _persist(self, connection, record: Mapping[str, Any], identity: str) -> str:
        if connection.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:assignment_key))"),
                {"assignment_key": self._lock_key(record)},
            )
        existing_identity = (
            connection.execute(
                select(active_strategy_assignment).where(
                    active_strategy_assignment.c.id == identity
                )
            )
            .mappings()
            .first()
        )
        if existing_identity is not None:
            return _immutable_insert(
                connection, active_strategy_assignment, {"id": identity, **record}
            )
        artefact = self._assert_references(connection, record)
        if record["execution_mode"] == "live":
            self._assert_live_authority(connection, record, artefact)
            self._assert_no_live_conflict(connection, record)
        return _immutable_insert(connection, active_strategy_assignment, {"id": identity, **record})

    def assign(
        self,
        *,
        product_id: str,
        portfolio_id: str,
        strategy_version_id: str,
        artefact_hash: str,
        lifecycle_state: str,
        execution_mode: str,
        capital_limit: float,
        assigned_at: str,
        assigned_by: str,
        sleeve_id: str = "default",
        instrument_id: str | None = None,
        universe_id: str | None = None,
        risk_budget: float = 0.0,
        active_until: str | None = None,
        assignment_reason: str = "unspecified",
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        record = _assignment_record(
            product_id=product_id,
            portfolio_id=portfolio_id,
            strategy_version_id=strategy_version_id,
            artefact_hash=artefact_hash,
            lifecycle_state=lifecycle_state,
            execution_mode=execution_mode,
            capital_limit=capital_limit,
            assigned_at=assigned_at,
            assigned_by=assigned_by,
            sleeve_id=sleeve_id,
            instrument_id=instrument_id,
            universe_id=universe_id,
            risk_budget=risk_budget,
            active_until=active_until,
            assignment_reason=assignment_reason,
            payload=payload,
        )
        identity = _hash(record, field="active assignment")
        with self.engine.begin() as connection:
            return self._persist(connection, record, identity)

    def active(
        self,
        product_id: str,
        *,
        execution_mode: str | None = None,
        at: str | None = None,
    ) -> dict[str, Any] | None:
        rows = self.active_assignments(product_id, at=at)
        if execution_mode is not None:
            rows = tuple(row for row in rows if row["execution_mode"] == execution_mode)
        return rows[0] if rows else None

    def by_id(self, assignment_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(active_strategy_assignment).where(
                        active_strategy_assignment.c.id == assignment_id
                    )
                )
                .mappings()
                .first()
            )
        return None if row is None else dict(row)

    def active_assignments(
        self, product_id: str, *, at: str | None = None
    ) -> tuple[dict[str, Any], ...]:
        with self.engine.connect() as connection:
            statement = select(active_strategy_assignment).where(
                active_strategy_assignment.c.product_id == product_id,
            )
            if at is not None:
                at = timestamp(at, field="assignment timestamp")
                statement = statement.where(
                    active_strategy_assignment.c.assigned_at <= at,
                )
            rows = connection.execute(statement).mappings().all()
        return tuple(
            row
            for row in self._current_assignment_rows(rows)
            if row["active"]
            and (at is None or row["active_until"] is None or row["active_until"] > at)
        )

    @staticmethod
    def _current_assignment_rows(rows: list[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        """Resolve immutable assignment events into the current active events."""

        materialised = [dict(row) for row in rows]
        superseded = {
            str(payload["supersedes_assignment_id"])
            for row in materialised
            if not row["active"]
            and isinstance(row.get("payload"), Mapping)
            and (payload := row["payload"]).get("supersedes_assignment_id")
        }
        current: dict[tuple[str, ...], dict[str, Any]] = {}
        lifecycle_rank = {
            "registered": 0,
            "development": 1,
            "forward_paper": 2,
            "live_ready": 3,
            "live_canary": 4,
            "live": 5,
            "suspended": 6,
            "retired": 7,
        }
        for row in materialised:
            if not row["active"] or str(row["id"]) in superseded:
                continue
            key = (
                str(row["product_id"]),
                str(row["portfolio_id"]),
                str(row["sleeve_id"]),
                str(row["strategy_version_id"]),
                str(row["assignment_scope_id"]),
                str(row["execution_mode"]),
            )
            previous = current.get(key)
            if previous is None or (
                str(row["assigned_at"]),
                lifecycle_rank.get(str(row["lifecycle_state"]), -1),
                str(row["id"]),
            ) > (
                str(previous["assigned_at"]),
                lifecycle_rank.get(str(previous["lifecycle_state"]), -1),
                str(previous["id"]),
            ):
                current[key] = row
        return tuple(current.values())

    def deactivate(
        self,
        product_id: str,
        *,
        at: str | None = None,
        assignment_reason: str = "deactivated",
        strategy_version_id: str | None = None,
        artefact_hash: str | None = None,
        sleeve_id: str | None = None,
        instrument_id: str | None = None,
        universe_id: str | None = None,
    ) -> None:
        """Remove execution authority while retaining the assignment history."""
        deactivated_at = (
            timestamp(at, field="deactivation timestamp")
            if at is not None
            else dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        )
        active = self.active_assignments(product_id, at=deactivated_at)
        if strategy_version_id is not None:
            active = tuple(
                row
                for row in active
                if row["strategy_version_id"] == strategy_version_id
            )
        if artefact_hash is not None:
            active = tuple(row for row in active if row["artefact_hash"] == artefact_hash)
        if sleeve_id is not None:
            active = tuple(row for row in active if row["sleeve_id"] == sleeve_id)
        if instrument_id is not None:
            active = tuple(row for row in active if row["instrument_id"] == instrument_id)
        if universe_id is not None:
            active = tuple(row for row in active if row["universe_id"] == universe_id)
        with self.engine.begin() as connection:
            for row in active:
                existing_payload = (
                    dict(row["payload"]) if isinstance(row.get("payload"), Mapping) else {}
                )
                existing_payload.update(
                    {
                        "supersedes_assignment_id": str(row["id"]),
                        "deactivation_timestamp": deactivated_at,
                    }
                )
                event = {
                    **{key: value for key, value in row.items() if key != "id"},
                    "active": False,
                    "assigned_at": deactivated_at,
                    "active_until": deactivated_at,
                    "assignment_reason": non_empty(assignment_reason, field="assignment_reason"),
                    "payload": existing_payload,
                }
                identity = _hash(event, field="assignment deactivation")
                _immutable_insert(
                    connection,
                    active_strategy_assignment,
                    {"id": identity, **event},
                )

    def assert_binding(
        self,
        *,
        product_id: str,
        strategy_version_id: str,
        artefact_hash: str,
        execution_mode: str,
        at: str | None = None,
        instrument_id: str | None = None,
        sleeve_id: str | None = None,
    ) -> dict[str, Any]:
        candidates = tuple(
            item
            for item in self.active_assignments(product_id, at=at)
            if item["execution_mode"] == execution_mode
            and (instrument_id is None or item.get("instrument_id") == instrument_id)
            and (sleeve_id is None or item.get("sleeve_id") == sleeve_id)
        )
        row = next(
            (
                item
                for item in candidates
                if item["strategy_version_id"] == strategy_version_id
                and item["artefact_hash"] == artefact_hash
            ),
            candidates[0] if candidates else None,
        )
        if row is None:
            raise CanonicalEvidenceError(
                f"no active canonical strategy assignment for {product_id}"
            )
        expected = {
            "strategy_version_id": strategy_version_id,
            "artefact_hash": artefact_hash,
            "execution_mode": execution_mode,
        }
        actual = {key: row[key] for key in expected}
        if actual != expected:
            raise CanonicalEvidenceError(
                f"active canonical assignment mismatch for {product_id}: expected={expected} actual={actual}"
            )
        return row
