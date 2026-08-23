"""Immutable thesis registration and lineage-wide trial accounting."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine

from src.data.database import research_thesis, thesis_trial
from src.domain._codec import canonical_hash, to_primitive
from src.domain.strategies import MechanismCategory, ResearchThesis


class ThesisError(RuntimeError):
    pass


@dataclass(frozen=True)
class ThesisTrial:
    thesis_id: str
    candidate_id: str
    lineage_id: str
    ordinal: int


class ThesisRegistry:
    """Register theses before results and share one budget across all variants."""

    def __init__(self) -> None:
        self._theses: dict[str, ResearchThesis] = {}
        self._trials: dict[str, dict[str, ThesisTrial]] = {}

    def register(self, thesis: ResearchThesis) -> str:
        existing = self._theses.get(thesis.thesis_id)
        if existing is not None and existing != thesis:
            raise ThesisError("thesis identity collision")
        self._theses[thesis.thesis_id] = thesis
        return thesis.thesis_id

    def get(self, thesis_id: str) -> ResearchThesis:
        try:
            return self._theses[thesis_id]
        except KeyError as exc:
            raise ThesisError(f"thesis is not registered: {thesis_id}") from exc

    def claim_trial(self, *, thesis_id: str, candidate_id: str, lineage_id: str) -> ThesisTrial:
        thesis = self.get(thesis_id)
        trials = self._trials.setdefault(thesis_id, {})
        existing = trials.get(candidate_id)
        if existing is not None:
            return existing
        if len(trials) >= thesis.cumulative_trial_budget:
            raise ThesisError("thesis cumulative trial budget is exhausted")
        trial = ThesisTrial(thesis_id, candidate_id, lineage_id, len(trials) + 1)
        trials[candidate_id] = trial
        return trial

    def trials(self, thesis_id: str) -> tuple[ThesisTrial, ...]:
        self.get(thesis_id)
        return tuple(self._trials.get(thesis_id, {}).values())


class SqlThesisRegistry:
    """Durable append-only thesis authority with atomic trial budgets."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def register(self, thesis: ResearchThesis) -> str:
        payload = to_primitive(thesis)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(research_thesis).where(research_thesis.c.id == thesis.thesis_id)
            ).mappings().first()
            values = {
                "id": thesis.thesis_id,
                "created_at": thesis.created_at,
                "creator_identity": thesis.creator_identity,
                "cumulative_trial_budget": thesis.cumulative_trial_budget,
                "payload": payload,
            }
            if existing is None:
                connection.execute(insert(research_thesis).values(**values))
            elif any(existing[key] != value for key, value in values.items()):
                raise ThesisError("persisted thesis identity collision")
        return thesis.thesis_id

    def get(self, thesis_id: str) -> ResearchThesis:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(research_thesis.c.payload).where(research_thesis.c.id == thesis_id)
            ).scalar_one_or_none()
        if payload is None:
            raise ThesisError(f"thesis is not registered: {thesis_id}")
        return _thesis_from_payload(payload)

    def claim_trial(
        self, *, thesis_id: str, candidate_id: str, lineage_id: str, claimed_at: str
    ) -> ThesisTrial:
        with self.engine.begin() as connection:
            statement = select(research_thesis).where(research_thesis.c.id == thesis_id)
            if self.engine.dialect.name == "postgresql":
                statement = statement.with_for_update()
            thesis_row = connection.execute(statement).mappings().first()
            if thesis_row is None:
                raise ThesisError(f"thesis is not registered: {thesis_id}")
            existing = connection.execute(
                select(thesis_trial).where(thesis_trial.c.candidate_id == candidate_id)
            ).mappings().first()
            if existing is not None:
                if existing["thesis_id"] != thesis_id or existing["lineage_id"] != lineage_id:
                    raise ThesisError("candidate trial identity collision")
                return ThesisTrial(thesis_id, candidate_id, lineage_id, existing["ordinal"])
            count = connection.execute(
                select(func.count()).select_from(thesis_trial).where(
                    thesis_trial.c.thesis_id == thesis_id
                )
            ).scalar_one()
            if count >= thesis_row["cumulative_trial_budget"]:
                raise ThesisError("thesis cumulative trial budget is exhausted")
            ordinal = count + 1
            trial_id = canonical_hash(
                {"thesis_id": thesis_id, "candidate_id": candidate_id, "lineage_id": lineage_id}
            )
            connection.execute(
                insert(thesis_trial).values(
                    id=trial_id,
                    thesis_id=thesis_id,
                    candidate_id=candidate_id,
                    lineage_id=lineage_id,
                    ordinal=ordinal,
                    claimed_at=claimed_at,
                )
            )
            return ThesisTrial(thesis_id, candidate_id, lineage_id, ordinal)


def _thesis_from_payload(payload: dict[str, object]) -> ResearchThesis:
    return ResearchThesis(
        mechanism_category=MechanismCategory(str(payload["mechanism_category"])),
        market_rationale=str(payload["market_rationale"]),
        expected_causal_chain=tuple(payload["expected_causal_chain"]),
        expected_direction=str(payload["expected_direction"]),
        expected_horizon=str(payload["expected_horizon"]),
        required_data=tuple(payload["required_data"]),
        permitted_features=tuple(payload["permitted_features"]),
        instrument_universe=tuple(payload["instrument_universe"]),
        generalisation_scope=dict(payload["generalisation_scope"]),
        failure_regimes=tuple(payload["failure_regimes"]),
        falsification_tests=tuple(payload["falsification_tests"]),
        negative_controls=tuple(payload["negative_controls"]),
        execution_capacity_assumptions=dict(payload["execution_capacity_assumptions"]),
        parent_thesis_ids=tuple(payload["parent_thesis_ids"]),
        cumulative_trial_budget=int(payload["cumulative_trial_budget"]),
        created_at=str(payload["created_at"]),
        creator_identity=str(payload["creator_identity"]),
    )


REQUIRED_NEGATIVE_CONTROLS = (
    "block_permutation",
    "synthetic_autocorrelated_null",
    "placebo_event_times",
    "feature_ablation",
    "parameter_neighbourhood",
    "predeclared_universe_holdout",
    "cross_instrument",
)
