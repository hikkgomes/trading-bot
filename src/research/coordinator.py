"""Provider-neutral research queue with reproducible rejection reasons."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.domain.strategies import StrategyDefinition
from src.research.datasets import CandidateDatasetPlan


class CandidateState(StrEnum):
    QUEUED = "queued"
    SCREEN_REJECTED = "screen_rejected"
    DEVELOPMENT_REJECTED = "development_rejected"
    ROBUSTNESS_REJECTED = "robustness_rejected"
    PROTECTED_REJECTED = "protected_rejected"
    FORWARD_PAPER = "forward_paper"


def _sha256_identity(value: str, *, field: str) -> str:
    value = non_empty(value, field=field)
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{field} must be a SHA-256 identity")
    return value


def _validate_snapshot_hashes(values: tuple[str, ...]) -> None:
    if not values or any(
        not isinstance(item, str) or len(item) != 71 or not item.startswith("sha256:")
        for item in values
    ):
        raise ValueError("dataset_snapshot_hashes must contain SHA-256 hashes")
    if len(set(values)) != len(values):
        raise ValueError("dataset_snapshot_hashes must not contain duplicates")


def _validate_dataset_plan(candidate: Candidate) -> None:
    if candidate.dataset_plan is None:
        return
    if candidate.dataset_plan.product_id != candidate.definition.product:
        raise ValueError("candidate dataset plan product does not match definition")
    if set(candidate.dataset_snapshot_hashes) != set(candidate.dataset_plan.all_snapshot_ids):
        raise ValueError("candidate dataset identities do not match its typed plan")


@dataclass(frozen=True)
class Candidate:
    definition: StrategyDefinition
    thesis_id: str
    lineage_id: str
    provider: str
    dataset_snapshot_hashes: tuple[str, ...]
    submitted_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dataset_bundle_id: str | None = None
    dataset_plan: CandidateDatasetPlan | None = None

    def __post_init__(self) -> None:
        for field_name in ("thesis_id", "lineage_id"):
            object.__setattr__(
                self, field_name, _sha256_identity(getattr(self, field_name), field=field_name)
            )
        object.__setattr__(self, "provider", non_empty(self.provider, field="provider"))
        _validate_snapshot_hashes(self.dataset_snapshot_hashes)
        object.__setattr__(self, "submitted_at", timestamp(self.submitted_at, field="submitted_at"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), field="metadata"))
        if self.dataset_bundle_id is not None:
            object.__setattr__(
                self,
                "dataset_bundle_id",
                _sha256_identity(self.dataset_bundle_id, field="dataset_bundle_id"),
            )
        _validate_dataset_plan(self)

    @property
    def candidate_id(self) -> str:
        payload: dict[str, Any] = {
            "definition_hash": self.definition.definition_hash,
            "thesis_id": self.thesis_id,
            "lineage_id": self.lineage_id,
            "provider": self.provider,
            "dataset_snapshot_hashes": self.dataset_snapshot_hashes,
        }
        if self.dataset_bundle_id is not None:
            payload["dataset_bundle_id"] = self.dataset_bundle_id
        if self.dataset_plan is not None:
            payload["dataset_plan"] = self.dataset_plan.to_payload()
        return canonical_hash(payload)


@dataclass(frozen=True)
class CandidateEvaluationView:
    """Candidate metadata view limited to the datasets for one adaptive run."""

    candidate_id: str
    definition: StrategyDefinition
    thesis_id: str
    lineage_id: str
    provider: str
    dataset_snapshot_hashes: tuple[str, ...]
    submitted_at: str
    metadata: Mapping[str, Any]
    dataset_bundle_id: str | None = None
    dataset_plan: CandidateDatasetPlan | None = None

    @classmethod
    def from_candidate(
        cls, candidate: Candidate, dataset_snapshot_hashes: tuple[str, ...]
    ) -> CandidateEvaluationView:
        snapshots = tuple(dataset_snapshot_hashes)
        if not snapshots or not set(snapshots).issubset(set(candidate.dataset_snapshot_hashes)):
            raise ValueError("evaluation view datasets must be a non-empty candidate subset")
        return cls(
            candidate_id=candidate.candidate_id,
            definition=candidate.definition,
            thesis_id=candidate.thesis_id,
            lineage_id=candidate.lineage_id,
            provider=candidate.provider,
            dataset_snapshot_hashes=snapshots,
            submitted_at=candidate.submitted_at,
            metadata=candidate.metadata,
            dataset_bundle_id=candidate.dataset_bundle_id,
            dataset_plan=candidate.dataset_plan,
        )


@dataclass(frozen=True)
class ResearchResult:
    candidate_id: str
    state: CandidateState
    accepted: bool
    reason_code: str | None
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", non_empty(self.candidate_id, field="candidate_id"))
        if self.accepted != (self.reason_code is None):
            raise ValueError(
                "accepted research results must have no reason_code and rejected results need one"
            )
        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be an object")
        object.__setattr__(self, "evidence", json_value(dict(self.evidence), field="evidence"))


Validator = Callable[[Candidate], tuple[bool, str | None, Mapping[str, Any]]]


class ResearchStore(Protocol):
    def save_candidate(self, candidate: Candidate) -> None: ...

    def load_candidates(self) -> tuple[Candidate, ...]: ...

    def save_result(self, result: ResearchResult) -> None: ...

    def load_results(self) -> tuple[ResearchResult, ...]: ...


class ResearchCoordinator:
    """One queue for registered, generated, ML, and agent candidates.

    The coordinator does not evaluate protected data itself. A protected-stage
    validator receives only pre-approved data and returns immutable evidence.
    """

    def __init__(self, store: ResearchStore | None = None) -> None:
        self.store = store
        candidates = store.load_candidates() if store is not None else ()
        results = store.load_results() if store is not None else ()
        self._queue = {item.candidate_id: item for item in candidates}
        self._results = {item.candidate_id: item for item in results}

    def submit(self, candidate: Candidate) -> str:
        candidate_id = candidate.candidate_id
        existing = self._queue.get(candidate_id)
        if existing is not None and existing != candidate:
            raise ValueError("candidate hash collision")
        if self.store is not None:
            claim_trial = getattr(self.store, "claim_trial", None)
            if claim_trial is not None:
                claim_trial(candidate)
            self.store.save_candidate(candidate)
        self._queue[candidate_id] = candidate
        return candidate_id

    def register(self, candidates: Iterable[Candidate]) -> tuple[str, ...]:
        return tuple(self.submit(candidate) for candidate in candidates)

    def pending(self) -> tuple[Candidate, ...]:
        return tuple(self._queue[key] for key in sorted(self._queue) if key not in self._results)

    def evaluate(
        self,
        candidate_id: str,
        *,
        screening: Validator,
        development: Validator,
        robustness: Validator,
        protected: Validator,
        evaluated_at: str | None = None,
    ) -> ResearchResult:
        candidate = self._queue[candidate_id]
        stages = (
            (CandidateState.SCREEN_REJECTED, screening),
            (CandidateState.DEVELOPMENT_REJECTED, development),
            (CandidateState.ROBUSTNESS_REJECTED, robustness),
            (CandidateState.PROTECTED_REJECTED, protected),
        )
        evidence: dict[str, Any] = {
            "candidate_id": candidate_id,
            "definition_hash": candidate.definition.definition_hash,
            "evaluated_at": timestamp(
                evaluated_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
                field="evaluated_at",
            ),
            "stages": {},
        }
        for rejected_state, validator in stages:
            accepted, reason_code, stage_evidence = validator(candidate)
            if not isinstance(accepted, bool) or (not accepted and not reason_code):
                raise ValueError("research validator returned an invalid result")
            evidence["stages"][rejected_state.value] = {
                "accepted": accepted,
                "reason_code": reason_code,
                "evidence": dict(stage_evidence),
            }
            if not accepted:
                result = ResearchResult(
                    candidate_id=candidate_id,
                    state=rejected_state,
                    accepted=False,
                    reason_code=str(reason_code),
                    evidence=evidence,
                )
                if self.store is not None:
                    self.store.save_result(result)
                self._results[candidate_id] = result
                return result
        result = ResearchResult(
            candidate_id=candidate_id,
            state=CandidateState.FORWARD_PAPER,
            accepted=True,
            reason_code=None,
            evidence=evidence,
        )
        if self.store is not None:
            self.store.save_result(result)
        self._results[candidate_id] = result
        return result

    def result(self, candidate_id: str) -> ResearchResult | None:
        return self._results.get(candidate_id)

    def development_feedback(self) -> tuple[dict[str, Any], ...]:
        """Return adaptive feedback without protected or forward evidence."""
        feedback: list[dict[str, Any]] = []
        for candidate_id in sorted(self._results):
            result = self._results[candidate_id]
            stages = result.evidence.get("stages")
            stages = stages if isinstance(stages, Mapping) else {}
            feedback.append(
                {
                    "candidate_id": candidate_id,
                    "state": result.state.value,
                    "reason_code": result.reason_code,
                    "stages": {
                        key: value
                        for key, value in stages.items()
                        if key
                        not in {
                            "protected",
                            CandidateState.PROTECTED_REJECTED.value,
                            "forward",
                            "forward_paper",
                        }
                    },
                }
            )
        return tuple(feedback)
