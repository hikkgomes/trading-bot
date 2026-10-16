"""Database repository for the provider-neutral research queue."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from sqlalchemy import column, func, insert, or_, select, table, text, update
from sqlalchemy.engine import Connection, Engine

from src.data.database import (
    experiment,
    experiment_metric,
    experiment_run,
    research_epoch,
    strategy_definition,
    strategy_identity,
    strategy_version,
    thesis_trial,
    validation_result,
    validation_stage,
)
from src.data.transactions import connection_scope
from src.domain._codec import canonical_hash, json_value, timestamp, to_primitive
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.coordinator import (
    Candidate,
    CandidateState,
    ResearchResult,
    TrialCapacityError,
)
from src.research.datasets import CandidateDatasetPlan
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.rounds import assert_epoch_open, lock_epoch
from src.strategies.behaviour import behaviour_hash_for_definition

_ADAPTIVE_CANDIDATE = table(
    "adaptive_candidate", *(column(item.name, item.type) for item in experiment.columns)
)


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
    DAILY_TRIAL_BUDGET = 100
    EPOCH_TRIAL_BUDGET = 100
    CATALOGUE_EPOCH_SHARE = 0.8

    def __init__(
        self,
        engine: Engine,
        *,
        connection: Connection | None = None,
        daily_trial_budget: int = DAILY_TRIAL_BUDGET,
        epoch_trial_budget: int = EPOCH_TRIAL_BUDGET,
    ):
        self.engine = engine
        self.connection = connection
        self.daily_trial_budget = daily_trial_budget
        self.epoch_trial_budget = epoch_trial_budget
        self.candidates = _ADAPTIVE_CANDIDATE if engine.dialect.name == "postgresql" else experiment

    def register_candidate(self, candidate: Candidate, *, producer: str) -> None:
        """Reserve one unique trial and persist its candidate in one transaction."""
        from src.research.theses import SqlThesisRegistry

        day = candidate.submitted_at[:10]
        with connection_scope(self.engine, self.connection, write=True) as connection:
            lock_epoch(connection, candidate.definition.product)
            self._lock_daily_capacity(connection, day)
            if self._existing_candidate_matches(connection, candidate):
                return
            epoch = self._ensure_epoch(connection, candidate)
            is_catalogue = producer == "catalogue" or producer.startswith(
                "registered_strategy_catalogue"
            )
            self._assert_epoch_capacity(connection, epoch, candidate, is_catalogue)
            self._assert_daily_capacity(connection, day)
            SqlThesisRegistry(self.engine, connection=connection).claim_trial(
                thesis_id=candidate.thesis_id,
                candidate_id=candidate.candidate_id,
                lineage_id=candidate.lineage_id,
                claimed_at=candidate.submitted_at,
            )
            self._record_epoch_reservation(connection, epoch, candidate, is_catalogue)
            stored_candidate = candidate
            if is_catalogue:
                stored_candidate = replace(
                    candidate,
                    metadata={**dict(candidate.metadata), "_research_producer": "catalogue"},
                )
            SqlResearchStore(self.engine, connection=connection).save_candidate(stored_candidate)

    def _lock_daily_capacity(self, connection: Connection, day: str) -> None:
        if self.engine.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:scope))"),
                {"scope": f"research-trial-budget:{day}"},
            )

    def _assert_epoch_capacity(self, connection, epoch, candidate, is_catalogue) -> None:
        if epoch is not None:
            from src.research.rounds import EpochState

            if epoch["state"] in {
                EpochState.CLOSED_FOR_SELECTION.value,
                EpochState.EXHAUSTED.value,
            }:
                raise TrialCapacityError("research epoch is closed to new candidates")
            if epoch["consumed"] >= epoch["budget"]:
                raise TrialCapacityError("research epoch trial budget is exhausted")
            used = self._catalogue_consumed(connection, epoch, candidate)
            budget = epoch["budget"]
        else:
            total, used = self._unbundled_usage(connection, candidate)
            budget = self.epoch_trial_budget
            if total >= budget:
                raise TrialCapacityError("research epoch trial budget is exhausted")
        if is_catalogue and used >= int(budget * self.CATALOGUE_EPOCH_SHARE):
            raise TrialCapacityError("catalogue allocation reached its 80% epoch limit")

    def _unbundled_usage(self, connection: Connection, candidate: Candidate) -> tuple[int, int]:
        source = self.candidates.join(
            strategy_version,
            self.candidates.c.strategy_version_id == strategy_version.c.id,
        ).join(
            strategy_definition,
            strategy_version.c.definition_id == strategy_definition.c.id,
        )
        scope = (
            self.candidates.c.dataset_bundle_id.is_(None),
            self.candidates.c.dataset_snapshot_hashes == list(candidate.dataset_snapshot_hashes),
            strategy_definition.c.product_id == candidate.definition.product,
        )
        counts = connection.execute(
            select(
                func.count().label("total"),
                func.count()
                .filter(
                    or_(
                        self.candidates.c.provider == "registered_strategy_catalogue",
                        self.candidates.c.metadata["_research_producer"].as_string() == "catalogue",
                    )
                )
                .label("catalogue"),
            )
            .select_from(source)
            .where(*scope)
        ).one()
        return int(counts.total), int(counts.catalogue)

    def _catalogue_consumed(self, connection, epoch, candidate) -> int:
        payload = dict(epoch["payload"] or {})
        used = payload.get("catalogue_consumed")
        if used is not None:
            return int(used)
        return int(
            connection.execute(
                select(func.count())
                .select_from(self.candidates)
                .where(
                    self.candidates.c.dataset_bundle_id == candidate.dataset_bundle_id,
                    or_(
                        self.candidates.c.provider == "registered_strategy_catalogue",
                        self.candidates.c.metadata["_research_producer"].as_string() == "catalogue",
                    ),
                )
            ).scalar_one()
        )

    def _assert_daily_capacity(self, connection: Connection, day: str) -> None:
        tomorrow = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
        used = connection.execute(
            select(func.count())
            .select_from(thesis_trial)
            .where(
                thesis_trial.c.claimed_at >= f"{day}T00:00:00+00:00",
                thesis_trial.c.claimed_at < f"{tomorrow}T00:00:00+00:00",
            )
        ).scalar_one()
        if used >= self.daily_trial_budget:
            raise TrialCapacityError("daily research trial budget is exhausted")

    def _record_epoch_reservation(self, connection, epoch, candidate, is_catalogue) -> None:
        if epoch is None:
            return
        payload = dict(epoch["payload"] or {})
        if is_catalogue:
            payload["catalogue_consumed"] = (
                self._catalogue_consumed(connection, epoch, candidate) + 1
            )
        connection.execute(
            update(research_epoch)
            .where(research_epoch.c.id == epoch["id"])
            .values(consumed=research_epoch.c.consumed + 1, payload=payload)
        )

    def _ensure_epoch(self, connection: Connection, candidate: Candidate) -> dict[str, Any] | None:
        if candidate.dataset_bundle_id is None:
            return None
        statement = select(research_epoch).where(
            research_epoch.c.dataset_bundle_id == candidate.dataset_bundle_id
        )
        if self.engine.dialect.name == "postgresql":
            statement = statement.with_for_update()
        row = connection.execute(statement).mappings().first()
        if row is not None:
            return dict(row)
        existing = list(
            connection.execute(
                select(
                    self.candidates.c.id,
                    self.candidates.c.provider,
                    self.candidates.c.metadata,
                ).where(self.candidates.c.dataset_bundle_id == candidate.dataset_bundle_id)
            ).mappings()
        )
        existing_ids = [str(item["id"]) for item in existing]
        catalogue_count = sum(
            item["provider"] == "registered_strategy_catalogue"
            or (item["metadata"] or {}).get("_research_producer") == "catalogue"
            for item in existing
        )
        previous = connection.execute(
            select(research_epoch.c.payload)
            .where(research_epoch.c.product_id == candidate.definition.product)
            .order_by(research_epoch.c.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        previous = dict(previous or {})
        catalogue_closed = bool(previous.get("catalogue_closed", False))
        catalogue_cursor = 0 if catalogue_closed else int(previous.get("catalogue_cursor", 0))
        identity = canonical_hash(
            {
                "product_id": candidate.definition.product,
                "bundle_id": candidate.dataset_bundle_id,
                "engine_version": ECONOMIC_ENGINE_VERSION,
            }
        )
        epoch = {
            "id": identity,
            "product_id": candidate.definition.product,
            "dataset_bundle_id": candidate.dataset_bundle_id,
            "state": "active",
            "round_cursor": 0,
            "budget": max(self.epoch_trial_budget, len(existing_ids)),
            "consumed": len(existing_ids),
            "created_at": candidate.submitted_at,
            "payload": {
                "engine_version": ECONOMIC_ENGINE_VERSION,
                "adopted_candidate_ids": existing_ids,
                "catalogue_consumed": catalogue_count,
                "catalogue_cursor": catalogue_cursor,
                "catalogue_closed": False,
                "last_completion": None,
            },
        }
        connection.execute(insert(research_epoch).values(**epoch))
        return epoch

    def save_candidate(self, candidate: Candidate) -> None:
        definition_id = candidate.definition.definition_hash
        version_id = candidate.definition.strategy_version_id
        definition_payload = to_primitive(candidate.definition)
        metadata_payload = self._candidate_metadata(candidate)
        with connection_scope(self.engine, self.connection, write=True) as connection:
            lock_epoch(connection, candidate.definition.product)
            self._assert_claimed_trial(connection, candidate)
            if self._existing_candidate_matches(connection, candidate):
                return
            assert_epoch_open(connection, candidate.dataset_bundle_id)
            self._ensure_strategy_version(
                connection,
                candidate,
                definition_id=definition_id,
                version_id=version_id,
                definition_payload=definition_payload,
            )
            self._ensure_strategy_identity(
                connection,
                candidate,
                definition_id=definition_id,
                definition_payload=definition_payload,
                metadata_payload=metadata_payload,
            )
            connection.execute(
                insert(experiment).values(
                    **self._experiment_values(
                        candidate,
                        version_id=version_id,
                        metadata_payload=metadata_payload,
                    )
                )
            )

    @staticmethod
    def _candidate_metadata(candidate: Candidate) -> dict[str, Any]:
        metadata_values = dict(candidate.metadata)
        if candidate.dataset_plan is not None:
            metadata_values["_dataset_plan"] = candidate.dataset_plan.to_payload()
        return json_value(metadata_values, field="candidate metadata")

    @staticmethod
    def _assert_claimed_trial(connection, candidate: Candidate) -> None:
        trial = (
            connection.execute(
                select(thesis_trial).where(thesis_trial.c.candidate_id == candidate.candidate_id)
            )
            .mappings()
            .first()
        )
        if trial is None:
            raise ValueError("candidate trial must be claimed before candidate registration")
        if trial["thesis_id"] != candidate.thesis_id or trial["lineage_id"] != candidate.lineage_id:
            raise ValueError("candidate thesis lineage does not match its claimed trial")

    def _existing_candidate_matches(self, connection, candidate: Candidate) -> bool:
        existing = (
            connection.execute(
                select(self.candidates).where(self.candidates.c.id == candidate.candidate_id)
            )
            .mappings()
            .first()
        )
        if existing is None:
            return False
        if self._candidate_from_row(connection, existing).candidate_id != candidate.candidate_id:
            raise ValueError("persisted research candidate does not match candidate hash")
        return True

    @staticmethod
    def _ensure_strategy_version(
        connection,
        candidate: Candidate,
        *,
        definition_id: str,
        version_id: str,
        definition_payload: dict[str, Any],
    ) -> None:
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
            return
        expected_definition = {
            "id": definition_id,
            "identity": candidate.definition.identity,
            "product_id": candidate.definition.product,
            "source_type": candidate.definition.source_type.value,
            "source_hash": candidate.definition.source_hash,
        }
        if (
            any(existing_definition[field] != value for field, value in expected_definition.items())
            or _definition_from_dict(existing_definition["definition"]).definition_hash
            != definition_id
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
        elif existing_version["definition_id"] != definition_id or existing_version["payload"] != {
            "definition_hash": definition_id
        }:
            raise ValueError("persisted strategy version does not match version hash")

    @staticmethod
    def _ensure_strategy_identity(
        connection,
        candidate: Candidate,
        *,
        definition_id: str,
        definition_payload: dict[str, Any],
        metadata_payload: dict[str, Any],
    ) -> None:
        behaviour_hash = behaviour_hash_for_definition(candidate.definition)
        identity_rows = connection.execute(
            select(strategy_identity.c.id).where(
                strategy_identity.c.behavior_hash == behaviour_hash,
                strategy_identity.c.id != candidate.candidate_id,
            )
        ).all()
        identity_values = {
            "id": candidate.candidate_id,
            "behavior_hash": behaviour_hash,
            "submitted_spec": definition_payload,
            "generation_method": candidate.definition.source_type.value,
            "metadata": metadata_payload,
            "parent_hashes": metadata_payload.get("parent_hashes", []),
            "is_duplicate": bool(identity_rows),
            "created_at": candidate.submitted_at,
        }
        existing_identity = (
            connection.execute(
                select(strategy_identity)
                .where(strategy_identity.c.id == candidate.candidate_id)
                .limit(1)
            )
            .mappings()
            .first()
        )
        if existing_identity is None:
            connection.execute(insert(strategy_identity).values(**identity_values))
        elif any(existing_identity[field] != value for field, value in identity_values.items()):
            raise ValueError("persisted strategy identity does not match candidate hash")

    @staticmethod
    def _experiment_values(
        candidate: Candidate, *, version_id: str, metadata_payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "id": candidate.candidate_id,
            "strategy_version_id": version_id,
            "provider": candidate.provider,
            "state": CandidateState.QUEUED.value,
            "submitted_at": candidate.submitted_at,
            "dataset_snapshot_hashes": list(candidate.dataset_snapshot_hashes),
            "dataset_bundle_id": candidate.dataset_bundle_id,
            "metadata": metadata_payload,
        }

    def development_parents(
        self, *, limit: int = 20, as_of: str | None = None, stage_ids: list[str] | None = None
    ) -> tuple[Candidate, ...]:
        if limit < 1:
            raise ValueError("parent selection limit must be positive")
        net_return = validation_stage.c.payload["evidence"]["cost_adjusted_return"].as_float()
        statement = (
            select(validation_stage.c.experiment_id)
            .where(
                validation_stage.c.stage == "development",
                validation_stage.c.accepted.is_(True),
                validation_stage.c.engine_version == ECONOMIC_ENGINE_VERSION,
            )
            .order_by(net_return.desc().nulls_last(), validation_stage.c.experiment_id)
            .limit(limit)
        )
        if as_of is not None:
            statement = statement.where(validation_stage.c.evaluated_at <= as_of)
        if stage_ids is not None:
            statement = statement.where(validation_stage.c.id.in_(stage_ids))
        with connection_scope(self.engine, self.connection, write=False) as connection:
            ids = tuple(connection.execute(statement).scalars())
        return tuple(self.get_candidate(candidate_id) for candidate_id in dict.fromkeys(ids))

    def load_candidates(self) -> tuple[Candidate, ...]:
        with connection_scope(self.engine, self.connection, write=False) as connection:
            rows = connection.execute(
                select(self.candidates)
                .where(self.candidates.c.state != "legacy_import")
                .order_by(self.candidates.c.id)
            ).mappings()
            return tuple(self._candidate_from_row(connection, row) for row in rows)

    def get_candidate(self, candidate_id: str) -> Candidate:
        with connection_scope(self.engine, self.connection, write=False) as connection:
            row = (
                connection.execute(
                    select(self.candidates).where(self.candidates.c.id == candidate_id)
                )
                .mappings()
                .first()
            )
            if row is None:
                raise KeyError(f"research candidate does not exist: {candidate_id}")
            if row["state"] == "legacy_import":
                raise KeyError(f"research candidate is immutable legacy history: {candidate_id}")
            candidate = self._candidate_from_row(connection, row)
            if candidate.candidate_id != candidate_id:
                raise ValueError("persisted research candidate identity does not match its payload")
            return candidate

    @staticmethod
    def _candidate_from_row(connection, row) -> Candidate:
        trial = (
            connection.execute(select(thesis_trial).where(thesis_trial.c.candidate_id == row["id"]))
            .mappings()
            .one()
        )
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
        metadata = dict(row["metadata"] or {})
        raw_plan = metadata.pop("_dataset_plan", None)
        metadata.pop("_research_producer", None)
        dataset_plan = CandidateDatasetPlan.from_payload(raw_plan) if raw_plan is not None else None
        return Candidate(
            definition=_definition_from_dict(definition_payload),
            thesis_id=trial["thesis_id"],
            lineage_id=trial["lineage_id"],
            provider=row["provider"],
            dataset_snapshot_hashes=tuple(row["dataset_snapshot_hashes"]),
            submitted_at=row["submitted_at"],
            metadata=metadata,
            dataset_bundle_id=row.get("dataset_bundle_id"),
            dataset_plan=dataset_plan,
        )

    def claim_trial(self, candidate: Candidate) -> None:
        from src.research.theses import SqlThesisRegistry

        SqlThesisRegistry(self.engine, connection=self.connection).claim_trial(
            thesis_id=candidate.thesis_id,
            candidate_id=candidate.candidate_id,
            lineage_id=candidate.lineage_id,
            claimed_at=candidate.submitted_at,
        )

    def save_result(self, result: ResearchResult) -> None:
        result_id = f"{result.candidate_id}:{result.state.value}"
        with connection_scope(self.engine, self.connection, write=True) as connection:
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
        with connection_scope(self.engine, self.connection, write=False) as connection:
            rows = connection.execute(
                select(validation_result).order_by(validation_result.c.id)
            ).mappings()
            return tuple(self._result_from_row(row) for row in rows)

    def runs(self, candidate_id: str) -> tuple[dict[str, object], ...]:
        with connection_scope(self.engine, self.connection, write=False) as connection:
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
        with connection_scope(self.engine, self.connection, write=True) as connection:
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
