"""Repositories for immutable research evidence and lifecycle authority."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from typing import Any

from sqlalchemy import insert, select, text, update
from sqlalchemy.engine import Engine

from src.data.database import (
    active_strategy_assignment,
    dataset_snapshot,
    experiment,
    experiment_run,
    forward_evidence,
    forward_paper_observation,
    holdout_claim,
    holdout_outcome,
    production_preflight,
    strategy_approval,
    strategy_artefact,
    strategy_definition,
    strategy_version,
    validation_stage,
)
from src.domain._codec import canonical_hash, json_value, timestamp


class CanonicalEvidenceError(RuntimeError):
    """Canonical evidence is invalid, missing, or conflicts with history."""


_VALIDATION_STAGES = frozenset({"screening", "development", "robustness", "protected", "forward"})


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
    payload = connection.execute(
        select(strategy_artefact.c.payload).where(strategy_artefact.c.id == artefact_hash)
    ).scalar_one_or_none()
    if not isinstance(payload, Mapping):
        raise CanonicalEvidenceError(f"strategy artefact does not exist: {artefact_hash}")
    payload = dict(payload)
    if payload.get("schema") != "platform.strategy_artefact/v2":
        raise CanonicalEvidenceError("canonical strategy artefact schema is unsupported")
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
    ) -> str:
        observed_at = timestamp(observed_at, field="observed_at")
        strategy_version_id = str(strategy_version_id).strip()
        product_id = str(product_id).strip()
        instrument_id = str(instrument_id).strip()
        if not strategy_version_id or not product_id or not instrument_id:
            raise CanonicalEvidenceError(
                "forward observations require strategy, product, and instrument identities"
            )
        artefact_hash = _identity(artefact_hash, field="artefact_hash")
        payload = _object(observation, field="forward observation")
        identity_payload = {
            "strategy_version_id": strategy_version_id,
            "product_id": product_id,
            "instrument_id": instrument_id,
            "observed_at": observed_at,
            "artefact_hash": artefact_hash,
            "observation": payload,
        }
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
        evidence: Mapping[str, Any],
    ) -> str:
        """Append an artefact-independent forward-paper evidence summary.

        The summary is created before a deployable artefact. Per-observation
        rows can then bind the exact artefact without creating a hash cycle.
        """

        strategy_version_id = str(strategy_version_id).strip()
        product_id = str(product_id).strip()
        if not strategy_version_id or not product_id:
            raise CanonicalEvidenceError(
                "forward evidence requires strategy and product identities"
            )
        observed_at = timestamp(observed_at, field="observed_at")
        payload = {
            "strategy_version_id": strategy_version_id,
            "product_id": product_id,
            "observed_at": observed_at,
            "evidence": _object(evidence, field="forward evidence"),
        }
        identity = _hash(payload, field="forward evidence")
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
            return _immutable_insert(
                connection,
                forward_evidence,
                {"id": identity, "created_at": observed_at, "payload": payload},
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
            payload = connection.execute(
                select(strategy_artefact.c.payload).where(strategy_artefact.c.id == artefact_hash)
            ).scalar_one_or_none()
        if payload is None:
            raise KeyError(f"strategy artefact does not exist: {artefact_hash}")
        return dict(payload)


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
        if not isinstance(actor, str) or not actor.strip():
            raise CanonicalEvidenceError("human approval actor must be non-empty")
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
            "approved_by": actor.strip(),
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
        self, *, strategy_version_id: str, product_id: str, account_id: str
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(strategy_approval)
                    .where(
                        strategy_approval.c.strategy_version_id == strategy_version_id,
                        strategy_approval.c.product_id == product_id,
                        strategy_approval.c.account_id == account_id,
                    )
                    .order_by(strategy_approval.c.approved_at.desc())
                    .limit(1)
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
        self, *, strategy_version_id: str, product_id: str, account_id: str
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(production_preflight)
                    .where(
                        production_preflight.c.strategy_version_id == strategy_version_id,
                        production_preflight.c.product_id == product_id,
                        production_preflight.c.account_id == account_id,
                    )
                    .order_by(production_preflight.c.checked_at.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
        return None if row is None else dict(row)


class SqlActiveStrategyAssignmentRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

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
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        if execution_mode not in {"paper", "live"}:
            raise CanonicalEvidenceError("active assignment execution mode must be paper or live")
        if lifecycle_state not in {
            "registered",
            "development",
            "forward_paper",
            "live_canary",
            "live",
            "suspended",
            "retired",
        }:
            raise CanonicalEvidenceError("active assignment lifecycle state is not supported")
        capital_limit = _finite_nonnegative(capital_limit, field="assignment capital limit")
        product_id = str(product_id).strip()
        portfolio_id = str(portfolio_id).strip()
        strategy_version_id = str(strategy_version_id).strip()
        if not product_id or not portfolio_id or not strategy_version_id:
            raise CanonicalEvidenceError("active assignment binding fields cannot be empty")
        artefact_hash = _identity(artefact_hash, field="artefact_hash")
        if not isinstance(assigned_by, str) or not assigned_by.strip():
            raise CanonicalEvidenceError("active assignment actor must be non-empty")
        assigned_at = timestamp(assigned_at, field="assigned_at")
        record = {
            "product_id": product_id,
            "portfolio_id": portfolio_id,
            "strategy_version_id": strategy_version_id,
            "artefact_hash": artefact_hash,
            "lifecycle_state": lifecycle_state,
            "execution_mode": execution_mode,
            "capital_limit": capital_limit,
            "assigned_at": assigned_at,
            "assigned_by": assigned_by,
            "active": True,
            "payload": _object(payload or {}, field="assignment payload"),
        }
        identity = _hash(record, field="active assignment")
        with self.engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:assignment_key))"),
                    {"assignment_key": f"active-assignment:{product_id}"},
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
            artefact = _assert_canonical_artifact(connection, artefact_hash)
            _assert_artefact_binding(
                artefact,
                strategy_version_id=strategy_version_id,
                product_id=product_id,
                portfolio_id=portfolio_id,
            )
            if execution_mode == "live":
                approved = connection.execute(
                    select(strategy_approval.c.id).where(
                        strategy_approval.c.strategy_version_id == strategy_version_id,
                        strategy_approval.c.product_id == product_id,
                        strategy_approval.c.artefact_hash == artefact_hash,
                        strategy_approval.c.status == "approved",
                    )
                ).first()
                preflight = (
                    connection.execute(
                        select(production_preflight)
                        .where(
                            production_preflight.c.strategy_version_id == strategy_version_id,
                            production_preflight.c.product_id == product_id,
                            production_preflight.c.artefact_hash == artefact_hash,
                            production_preflight.c.accepted.is_(True),
                        )
                        .order_by(production_preflight.c.checked_at.desc())
                        .limit(1)
                    )
                    .mappings()
                    .first()
                )
                if (
                    approved is None
                    or preflight is None
                    or not preflight_is_fresh(
                        str(preflight["checked_at"]),
                        reference_at=assigned_at,
                    )
                ):
                    raise CanonicalEvidenceError(
                        "live assignment requires matching approval and fresh accepted preflight"
                    )
            connection.execute(
                update(active_strategy_assignment)
                .where(
                    active_strategy_assignment.c.product_id == product_id,
                    active_strategy_assignment.c.active.is_(True),
                )
                .values(active=False)
            )
            return _immutable_insert(
                connection, active_strategy_assignment, {"id": identity, **record}
            )

    def active(self, product_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(active_strategy_assignment).where(
                        active_strategy_assignment.c.product_id == product_id,
                        active_strategy_assignment.c.active.is_(True),
                    )
                )
                .mappings()
                .first()
            )
        return None if row is None else dict(row)

    def deactivate(self, product_id: str) -> None:
        """Remove execution authority while retaining the assignment history."""

        with self.engine.begin() as connection:
            connection.execute(
                update(active_strategy_assignment)
                .where(
                    active_strategy_assignment.c.product_id == product_id,
                    active_strategy_assignment.c.active.is_(True),
                )
                .values(active=False)
            )

    def assert_binding(
        self,
        *,
        product_id: str,
        strategy_version_id: str,
        artefact_hash: str,
        execution_mode: str,
    ) -> dict[str, Any]:
        row = self.active(product_id)
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
