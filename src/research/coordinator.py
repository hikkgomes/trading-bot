"""Provider-neutral research queue with reproducible rejection reasons."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.domain.strategies import StrategyDefinition


class CandidateState(StrEnum):
    QUEUED = "queued"
    SCREEN_REJECTED = "screen_rejected"
    DEVELOPMENT_REJECTED = "development_rejected"
    ROBUSTNESS_REJECTED = "robustness_rejected"
    PROTECTED_REJECTED = "protected_rejected"
    FORWARD_PAPER = "forward_paper"


@dataclass(frozen=True)
class Candidate:
    definition: StrategyDefinition
    provider: str
    dataset_snapshot_hashes: tuple[str, ...]
    submitted_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", non_empty(self.provider, field="provider"))
        if not self.dataset_snapshot_hashes or any(
            not item.startswith("sha256:") for item in self.dataset_snapshot_hashes
        ):
            raise ValueError("dataset_snapshot_hashes must contain SHA-256 hashes")
        object.__setattr__(self, "submitted_at", timestamp(self.submitted_at, field="submitted_at"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), field="metadata"))

    @property
    def candidate_id(self) -> str:
        return canonical_hash(
            {
                "definition_hash": self.definition.definition_hash,
                "provider": self.provider,
                "dataset_snapshot_hashes": self.dataset_snapshot_hashes,
            }
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
                        if key != CandidateState.PROTECTED_REJECTED.value
                    },
                }
            )
        return tuple(feedback)
