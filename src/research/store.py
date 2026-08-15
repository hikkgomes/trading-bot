"""Database repository for the provider-neutral research queue."""

from __future__ import annotations

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from src.data.database import (
    experiment,
    experiment_metric,
    experiment_run,
    strategy_definition,
    strategy_version,
    validation_result,
)
from src.domain._codec import canonical_hash, json_value, timestamp, to_primitive
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.coordinator import Candidate, CandidateState, ResearchResult


def _definition_from_dict(payload: dict[str, object]) -> StrategyDefinition:
    values = dict(payload)
    values["source_type"] = StrategySourceType(values["source_type"])
    return StrategyDefinition(**values)


class SqlResearchStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def save_candidate(self, candidate: Candidate) -> None:
        candidate_id = candidate.candidate_id
        definition_id = candidate.definition.definition_hash
        version_id = definition_id
        with self.engine.begin() as connection:
            existing = (
                connection.execute(select(experiment).where(experiment.c.id == candidate_id))
                .mappings()
                .first()
            )
            if existing is not None:
                if self._candidate_from_row(connection, existing) != candidate:
                    raise ValueError("persisted research candidate does not match candidate hash")
                return
            if (
                connection.execute(
                    select(strategy_definition.c.id).where(
                        strategy_definition.c.id == definition_id
                    )
                ).first()
                is None
            ):
                connection.execute(
                    insert(strategy_definition).values(
                        id=definition_id,
                        identity=candidate.definition.identity,
                        product_id=candidate.definition.product,
                        source_type=candidate.definition.source_type.value,
                        source_hash=candidate.definition.source_hash,
                        definition=to_primitive(candidate.definition),
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
            connection.execute(
                insert(experiment).values(
                    id=candidate_id,
                    strategy_version_id=version_id,
                    provider=candidate.provider,
                    state=CandidateState.QUEUED.value,
                    submitted_at=candidate.submitted_at,
                    dataset_snapshot_hashes=list(candidate.dataset_snapshot_hashes),
                    metadata=dict(candidate.metadata),
                )
            )

    def load_candidates(self) -> tuple[Candidate, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(experiment).order_by(experiment.c.id)).mappings()
            return tuple(self._candidate_from_row(connection, row) for row in rows)

    @staticmethod
    def _candidate_from_row(connection, row) -> Candidate:
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
            provider=row["provider"],
            dataset_snapshot_hashes=tuple(row["dataset_snapshot_hashes"]),
            submitted_at=row["submitted_at"],
            metadata=row["metadata"],
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

    def save_run(
        self,
        *,
        candidate_id: str,
        run_name: str,
        created_at: str,
        evidence: dict[str, object],
        metrics: dict[str, float],
    ) -> str:
        created_at = timestamp(created_at, field="created_at")
        clean_evidence = json_value(evidence, field="experiment run evidence")
        clean_metrics = {str(key): float(value) for key, value in metrics.items()}
        run_id = canonical_hash(
            {
                "candidate_id": candidate_id,
                "run_name": run_name,
                "evidence": clean_evidence,
                "metrics": clean_metrics,
            }
        )
        run_payload = {
            "candidate_id": candidate_id,
            "run_name": run_name,
            "evidence": clean_evidence,
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
