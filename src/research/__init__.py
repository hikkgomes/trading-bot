"""Unified strategy-registration and research-coordination contracts."""

from src.research.artefacts import StrategyArtefact, StrategyArtefactStore
from src.research.canonical import (
    CanonicalEvidenceError,
    SqlActiveStrategyAssignmentRepository,
    SqlApprovalRepository,
    SqlForwardEvidenceRepository,
    SqlHoldoutRepository,
    SqlPreflightRepository,
    SqlStrategyArtefactRepository,
    SqlValidationRepository,
)
from src.research.coordinator import Candidate, ResearchCoordinator, ResearchResult
from src.research.evaluation import (
    CanonicalResearchEvaluator,
    EvaluationRequest,
    StageEvaluation,
)
from src.research.providers import provider_candidate
from src.research.store import SqlResearchStore

__all__ = [
    "Candidate",
    "CanonicalEvidenceError",
    "CanonicalResearchEvaluator",
    "EvaluationRequest",
    "ResearchCoordinator",
    "ResearchResult",
    "SqlResearchStore",
    "SqlActiveStrategyAssignmentRepository",
    "SqlApprovalRepository",
    "SqlForwardEvidenceRepository",
    "SqlHoldoutRepository",
    "SqlPreflightRepository",
    "SqlStrategyArtefactRepository",
    "SqlValidationRepository",
    "StageEvaluation",
    "StrategyArtefact",
    "StrategyArtefactStore",
    "provider_candidate",
]
