"""Unified strategy-registration and research-coordination contracts."""

from src.research.artefacts import StrategyArtefact, StrategyArtefactStore
from src.research.coordinator import Candidate, ResearchCoordinator, ResearchResult
from src.research.providers import provider_candidate
from src.research.store import SqlResearchStore

__all__ = [
    "Candidate",
    "ResearchCoordinator",
    "ResearchResult",
    "SqlResearchStore",
    "StrategyArtefact",
    "StrategyArtefactStore",
    "provider_candidate",
]
