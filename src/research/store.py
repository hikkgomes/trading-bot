"""Database repository for the provider-neutral research queue."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from src.data.database import (
    experiment,
    experiment_metric,
    experiment_run,
    strategy_definition,
    strategy_identity,
    strategy_version,
    thesis_trial,
    validation_result,
)
from src.domain._codec import canonical_hash, json_value, timestamp, to_primitive
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.coordinator import Candidate, CandidateState, ResearchResult


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"persisted strategy {field} must be an object")
    return value


def _definition_from_dict(payload: dict[str, object]) -> StrategyDefinition:
    return StrategyDefinition(
        identity=str(payload["identity"]),
        version=str(payload["version"]),
        family=str(payload["family"]),
        product=str(payload["product"]),
        universe=_mapping(payload["universe"], field="universe"),
        data_requirements=_mapping(payload["data_requirements"], field="data_requirements"),
        feature_graph=_mapping(payload["feature_graph"], field="feature_graph"),
        signal_model=_mapping(payload["signal_model"], field="signal_model"),
        position_model=_mapping(payload["position_model"], field="position_model"),
        execution_preferences=_mapping(
            payload["execution_preferences"], field="execution_preferences"
        ),
        risk_policy=_mapping(payload["risk_policy"], field="risk_policy"),
        validation_policy=_mapping(payload["validation_policy"], field="validation_policy"),
        source_type=StrategySourceType(str(payload["source_type"])),
        source_hash=str(payload["source_hash"]),
        metadata=_mapping(payload.get("metadata", {}), field="metadata"),
    )


class SqlResearchStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def save_candidate(self, candidate: Candidate) -> None:
        candidate_id = candidate.candidate_id
        definition_id = candidate.definition.definition_hash
        version_id = candidate.definition.strategy_version_id
        definition_payload = to_primitive(candidate.definition)
        metadata_payload = json_value(dict(candidate.metadata), field="candidate metadata")
        parent_hashes = metadata_payload.get("parent_hashes", [])
        with self.engine.begin() as connection:
            trial = connection.execute(
                select(thesis_trial).where(thesis_trial.c.candidate_id == candidate_id)
            ).mappings().first()
            if trial is None:
                raise ValueError("candidate trial must be claimed before candidate registration")
            if trial["thesis_id"] != candidate.thesis_id or trial["lineage_id"] != candidate.lineage_id:
                raise ValueError("candidate thesis lineage does not match its claimed trial")
            existing = (
                connection.execute(select(experiment).where(experiment.c.id == candidate_id))
                .mappings()
                .first()
            )
            if existing is not None:
                if self._candidate_from_row(connection, existing) != candidate:
                    raise ValueError("persisted research candidate does not match candidate hash")
                return
            existing_definition = (
                connection.execute(
                    select(strategy_definition)
                    .where(strategy_definition.c.id == definition_id)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if existing_definition is None:
                connection.execute(
                    insert(strategy_definition).values(
                        id=definition_id,
                        identity=candidate.definition.identity,
                        product_id=candidate.definition.product,
                        source_type=candidate.definition.source_type.value,
                        source_hash=candidate.definition.source_hash,
                        definition=definition_payload,
                    )
                )
                connection.execute(
                    insert(strategy_version).values(
                        id=version_id,
                        definition_id=definition_id,
                        version=candidate.definition.version,
                        created_at=candidate.submitted_at,
                        payload={"definition_hash": definition_id},
                    )
                )
            else:
                expected_definition = {
                    "id": definition_id,
                    "identity": candidate.definition.identity,
                    "product_id": candidate.definition.product,
                    "source_type": candidate.definition.source_type.value,
                    "source_hash": candidate.definition.source_hash,
                    "definition": definition_payload,
                }
                if any(
                    existing_definition[field] != value
                    for field, value in expected_definition.items()
                ):
                    raise ValueError("persisted strategy definition does not match definition hash")
                existing_version = (
                    connection.execute(
                        select(strategy_version).where(strategy_version.c.id == version_id).limit(1)
                    )
                    .mappings()
                    .first()
                )
                if existing_version is None:
                    connection.execute(
                        insert(strategy_version).values(
                            id=version_id,
                            definition_id=definition_id,
                            version=candidate.definition.version,
                            created_at=candidate.submitted_at,
                            payload={"definition_hash": definition_id},
                        )
                    )
                elif (
                    existing_version["definition_id"] != definition_id
                    or existing_version["version"] != candidate.definition.version
                    or existing_version["payload"] != {"definition_hash": definition_id}
                ):
                    raise ValueError("persisted strategy version does not match version hash")
            identity_rows = connection.execute(
                select(strategy_identity.c.id).where(
                    strategy_identity.c.behavior_hash == definition_id,
                    strategy_identity.c.id != candidate_id,
                )
            ).all()
            existing_identity = (
                connection.execute(
                    select(strategy_identity).where(strategy_identity.c.id == candidate_id).limit(1)
                )
                .mappings()
                .first()
            )
            identity_values = {
                "id": candidate_id,
                "behavior_hash": definition_id,
                "submitted_spec": definition_payload,
                "generation_method": candidate.definition.source_type.value,
                "metadata": metadata_payload,
                "parent_hashes": parent_hashes,
                "is_duplicate": bool(identity_rows),
                "created_at": candidate.submitted_at,
            }
            if existing_identity is None:
                connection.execute(insert(strategy_identity).values(**identity_values))
            elif any(existing_identity[field] != value for field, value in identity_values.items()):
                raise ValueError("persisted strategy identity does not match candidate hash")
            connection.execute(
                insert(experiment).values(
                    id=candidate_id,
                    strategy_version_id=version_id,
                    provider=candidate.provider,
                    state=CandidateState.QUEUED.value,
                    submitted_at=candidate.submitted_at,
                    dataset_snapshot_hashes=list(candidate.dataset_snapshot_hashes),
                    metadata=metadata_payload,
                )
            )

    def load_candidates(self) -> tuple[Candidate, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(experiment).order_by(experiment.c.id)).mappings()
            return tuple(self._candidate_from_row(connection, row) for row in rows)

    def get_candidate(self, candidate_id: str) -> Candidate:
        with self.engine.connect() as connection:
            row = (
                connection.execute(select(experiment).where(experiment.c.id == candidate_id))
                .mappings()
                .first()
            )
            if row is None:
                raise KeyError(f"research candidate does not exist: {candidate_id}")
            candidate = self._candidate_from_row(connection, row)
            if candidate.candidate_id != candidate_id:
                raise ValueError("persisted research candidate identity does not match its payload")
            return candidate

    @staticmethod
    def _candidate_from_row(connection, row) -> Candidate:
        trial = connection.execute(
            select(thesis_trial).where(thesis_trial.c.candidate_id == row["id"])
        ).mappings().one()
        definition_payload = connection.execute(
            select(strategy_definition.c.definition)
            .select_from(
                strategy_definition.join(
                    strategy_version,
                    strategy_definition.c.id == strategy_version.c.definition_id,
                )
            )
            .where(strategy_version.c.id == row["strategy_version_id"])
        ).scalar_one()
        return Candidate(
            definition=_definition_from_dict(definition_payload),
            thesis_id=trial["thesis_id"],
            lineage_id=trial["lineage_id"],
            provider=row["provider"],
            dataset_snapshot_hashes=tuple(row["dataset_snapshot_hashes"]),
            submitted_at=row["submitted_at"],
            metadata=row["metadata"],
        )

    def claim_trial(self, candidate: Candidate) -> None:
        from src.research.theses import SqlThesisRegistry

        SqlThesisRegistry(self.engine).claim_trial(
            thesis_id=candidate.thesis_id,
            candidate_id=candidate.candidate_id,
            lineage_id=candidate.lineage_id,
            claimed_at=candidate.submitted_at,
        )

    def save_result(self, result: ResearchResult) -> None:
        result_id = f"{result.candidate_id}:{result.state.value}"
        with self.engine.begin() as connection:
            existing = (
                connection.execute(
                    select(validation_result).where(validation_result.c.id == result_id)
                )
                .mappings()
                .first()
            )
            if existing is not None:
                restored = self._result_from_row(existing)
                if restored != result:
                    raise ValueError("persisted research result does not match result identity")
                return
            connection.execute(
                insert(validation_result).values(
                    id=result_id,
                    experiment_id=result.candidate_id,
                    state=result.state.value,
                    accepted=result.accepted,
                    reason_code=result.reason_code,
                    evidence=dict(result.evidence),
                )
            )
            connection.execute(
                update(experiment)
                .where(experiment.c.id == result.candidate_id)
                .values(state=result.state.value)
            )

    def load_results(self) -> tuple[ResearchResult, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(validation_result).order_by(validation_result.c.id)
            ).mappings()
            return tuple(self._result_from_row(row) for row in rows)

    def runs(self, candidate_id: str) -> tuple[dict[str, object], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(experiment_run).order_by(experiment_run.c.created_at, experiment_run.c.id)
            ).mappings()
            return tuple(
                dict(row)
                for row in rows
                if isinstance(row["payload"], dict)
                and row["payload"].get("candidate_id") == candidate_id
            )

    def save_run(
        self,
        *,
        candidate_id: str,
        run_name: str,
        created_at: str,
        evidence: dict[str, object],
        metrics: dict[str, float],
        receipt: Mapping[str, Any] | None = None,
    ) -> str:
        created_at = timestamp(created_at, field="created_at")
        if not str(run_name).strip():
            raise ValueError("experiment run name cannot be empty")
        clean_evidence = json_value(evidence, field="experiment run evidence")
        if not clean_evidence:
            raise ValueError("experiment run evidence cannot be empty")
        clean_metrics = {str(key): float(value) for key, value in metrics.items()}
        if not clean_metrics:
            raise ValueError("experiment run metrics cannot be empty")
        clean_receipt = json_value(dict(receipt or {}), field="experiment execution receipt")
        if receipt is not None and not {
            "candidate_id",
            "dataset_snapshot_ids",
            "executor_version",
            "input_hash",
        }.issubset(clean_receipt):
            raise ValueError("experiment execution receipt is incomplete")
        run_id = canonical_hash(
            {
                "candidate_id": candidate_id,
                "run_name": run_name,
                "evidence": clean_evidence,
                "metrics": clean_metrics,
                "receipt": clean_receipt,
            }
        )
        run_payload = {
            "candidate_id": candidate_id,
            "run_name": run_name,
            "evidence": clean_evidence,
            "metrics": clean_metrics,
            "receipt": clean_receipt,
        }
        with self.engine.begin() as connection:
            if (
                connection.execute(
                    select(experiment.c.id).where(experiment.c.id == candidate_id)
                ).first()
                is None
            ):
                raise KeyError(f"research candidate does not exist: {candidate_id}")
            existing = connection.execute(
                select(experiment_run.c.payload).where(experiment_run.c.id == run_id)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != run_payload:
                    raise ValueError("experiment run identity collision")
                return run_id
            connection.execute(
                insert(experiment_run).values(
                    id=run_id,
                    created_at=created_at,
                    payload=run_payload,
                )
            )
            for name, value in sorted(clean_metrics.items()):
                metric_payload = {
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "name": name,
                    "value": value,
                }
                connection.execute(
                    insert(experiment_metric).values(
                        id=canonical_hash(metric_payload),
                        created_at=created_at,
                        payload=metric_payload,
                    )
                )
        return run_id

    @staticmethod
    def _result_from_row(row) -> ResearchResult:
        return ResearchResult(
            candidate_id=row["experiment_id"],
            state=CandidateState(row["state"]),
            accepted=bool(row["accepted"]),
            reason_code=row["reason_code"],
            evidence=row["evidence"],
        )
