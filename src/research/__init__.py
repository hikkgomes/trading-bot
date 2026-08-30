"""Unified strategy-registration and research-coordination contracts."""

from src.research.artefacts import StrategyArtefact, StrategyArtefactStore
from src.research.canonical import (
    CanonicalEvidenceError,
    ForwardPaperSummary,
    SqlActiveStrategyAssignmentRepository,
    SqlApprovalRepository,
    SqlForwardEvidenceRepository,
    SqlHoldoutRepository,
    SqlPreflightRepository,
    SqlStrategyArtefactRepository,
    SqlValidationRepository,
)
from src.research.coordinator import Candidate, ResearchCoordinator, ResearchResult
from src.research.datasets import CandidateDatasetPlan
from src.research.evaluation import (
    CanonicalResearchEvaluator,
    EvaluationRequest,
    ProtectedHoldoutWorker,
    StageEvaluation,
)
from src.research.generation import (
    CAMPAIGNS,
    CampaignAllocation,
    CampaignSpec,
    DuplicateMatch,
    GeneratedHypothesis,
    GenerationAllocator,
    GenerationError,
    GenerationFeedback,
    SqlGenerationFeedbackStore,
    SqlHypothesisMemory,
    build_hypothesis,
    campaign_thesis,
    hypothesis_signature,
    semantic_distance,
)
from src.research.providers import provider_candidate
from src.research.returns import PositionReturnLedger, PositionReturnReport, ReturnLedgerError
from src.research.store import SqlResearchStore
from src.research.theses import ThesisRegistry

__all__ = [
    "Candidate",
    "CandidateDatasetPlan",
    "CanonicalEvidenceError",
    "CanonicalResearchEvaluator",
    "EvaluationRequest",
    "ProtectedHoldoutWorker",
    "ResearchCoordinator",
    "ResearchResult",
    "SqlResearchStore",
    "SqlActiveStrategyAssignmentRepository",
    "SqlApprovalRepository",
    "SqlForwardEvidenceRepository",
    "ForwardPaperSummary",
    "SqlHoldoutRepository",
    "SqlPreflightRepository",
    "SqlStrategyArtefactRepository",
    "SqlValidationRepository",
    "StageEvaluation",
    "StrategyArtefact",
    "StrategyArtefactStore",
    "ThesisRegistry",
    "provider_candidate",
    "PositionReturnLedger",
    "PositionReturnReport",
    "ReturnLedgerError",
    "CAMPAIGNS",
    "CampaignAllocation",
    "CampaignSpec",
    "DuplicateMatch",
    "GeneratedHypothesis",
    "GenerationAllocator",
    "GenerationFeedback",
    "GenerationError",
    "SqlGenerationFeedbackStore",
    "SqlHypothesisMemory",
    "build_hypothesis",
    "campaign_thesis",
    "hypothesis_signature",
    "semantic_distance",
]
