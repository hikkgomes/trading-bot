"""Trusted compilation of inert OpenClaw theses into research candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping

from src.agents.proposals import AgentAction, AgentProposal
from src.domain._codec import canonical_hash
from src.domain.strategies import ResearchThesis, StrategyDefinition, StrategySourceType
from src.research.datasets import CandidateDatasetPlan, DatasetBundle
from src.research.providers import provider_candidate


class AgentCompilationError(ValueError):
    """An untrusted proposal cannot be compiled into a safe candidate."""


_ALLOWED_FAMILIES = frozenset(
    {
        "time_series",
        "mean_reversion",
        "cross_sectional",
        "relative_value",
        "microstructure",
        "machine_learning",
        "advanced_alpha",
        "meta_strategy",
        "execution",
        "market_making",
    }
)
_ALLOWED_DIRECTIONS = frozenset({"long", "short", "signed", "market_neutral", "hedged"})


def _validated_proposal(
    proposal: AgentProposal,
    *,
    bundle: DatasetBundle,
) -> tuple[ResearchThesis, str, Mapping[str, object]]:
    thesis = proposal.economic_thesis
    if thesis is None:
        raise AgentCompilationError("OpenClaw proposal has no typed economic thesis")
    if proposal.action is not AgentAction.CREATE_DSL:
        raise AgentCompilationError("only typed DSL proposals can enter the candidate funnel")
    if bundle.product_id != proposal.product_id:
        raise AgentCompilationError("proposal and dataset bundle products differ")
    family = str(thesis.generalisation_scope.get("family") or "")
    if family not in _ALLOWED_FAMILIES:
        raise AgentCompilationError("OpenClaw thesis family is not allowlisted")
    rule = proposal.provenance.get("rule")
    if not isinstance(rule, Mapping):
        raise AgentCompilationError("OpenClaw DSL proposals require a typed rule")
    return thesis, family, rule


def _validated_rule(
    proposal: AgentProposal,
    *,
    bundle: DatasetBundle,
) -> tuple[ResearchThesis, str, str, str, float, str]:
    thesis, family, rule = _validated_proposal(proposal, bundle=bundle)
    feature = str(rule.get("feature") or "")
    operator = str(rule.get("operator") or "")
    raw_threshold = rule.get("threshold")
    direction = str(rule.get("direction") or thesis.expected_direction)
    if feature not in thesis.permitted_features:
        raise AgentCompilationError("OpenClaw rule feature is outside its thesis")
    if operator not in {"gt", "ge", "lt", "le"}:
        raise AgentCompilationError("OpenClaw rule operator is unsupported")
    if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, int | float):
        raise AgentCompilationError("OpenClaw rule threshold must be numeric")
    threshold = float(raw_threshold)
    if not math.isfinite(threshold):
        raise AgentCompilationError("OpenClaw rule threshold must be finite")
    if direction not in _ALLOWED_DIRECTIONS:
        raise AgentCompilationError("OpenClaw rule direction is unsupported")
    return thesis, family, feature, operator, threshold, direction


def compile_openclaw_candidate(
    proposal: AgentProposal,
    *,
    bundle: DatasetBundle,
) -> tuple[StrategyDefinition, CandidateDatasetPlan]:
    thesis, family, feature, operator, threshold, direction = _validated_rule(
        proposal, bundle=bundle
    )
    plan = CandidateDatasetPlan.from_bundle(bundle)
    source_payload = {
        "kind": "typed_rule",
        "rule": {
            "feature": feature,
            "operator": operator,
            "threshold": threshold,
            "direction": direction,
        },
        "compiler": "openclaw-trusted-compiler/v1",
        "proposal_id": proposal.proposal_id,
    }
    definition = StrategyDefinition(
        identity=f"openclaw:{proposal.proposal_id}",
        version="openclaw-v1",
        family=family,
        product=proposal.product_id,
        universe={
            "type": "point_in_time",
            "symbols": list(thesis.instrument_universe),
            "universe_snapshot_id": bundle.universe_snapshot_id,
        },
        data_requirements={"required": list(thesis.required_data)},
        feature_graph={
            "version": "canonical-features/v2",
            "required_nodes": list(thesis.permitted_features),
        },
        signal_model=source_payload,
        position_model={"kind": "volatility_scaled", "signal_timing": "next_bar"},
        execution_preferences={"policy": "market", "paper_only_until_approved": True},
        risk_policy={"product_policy": proposal.product_id},
        validation_policy={
            "evidence_type": str(
                proposal.provenance.get(
                    "evidence_type",
                    "btc_allocation" if proposal.product_id == "btc_accumulation" else "swing",
                )
            ),
            "promotable": True,
        },
        source_type=StrategySourceType.GENERATED_DSL,
        source_hash=canonical_hash(source_payload),
        metadata={
            "openclaw_proposal_id": proposal.proposal_id,
            "trusted_compiler": "openclaw-trusted-compiler/v1",
            "promotable": True,
        },
    )
    return definition, plan


def compile_openclaw_candidate_payload(
    proposal: AgentProposal,
    *,
    bundle: DatasetBundle,
    submitted_at: str,
):
    definition, plan = compile_openclaw_candidate(proposal, bundle=bundle)
    return provider_candidate(
        identity=definition.identity,
        version=definition.version,
        family=definition.family,
        product=definition.product,
        thesis_id=proposal.economic_thesis.thesis_id if proposal.economic_thesis else "",
        lineage_id=canonical_hash(
            {
                "proposal_id": proposal.proposal_id,
                "thesis_id": proposal.economic_thesis.thesis_id if proposal.economic_thesis else "",
                "parent_thesis_ids": proposal.economic_thesis.parent_thesis_ids
                if proposal.economic_thesis
                else (),
            }
        ),
        provider="openclaw_trusted_compiler",
        source_type=definition.source_type,
        source_payload=definition.signal_model,
        dataset_snapshot_hashes=plan.all_snapshot_ids,
        submitted_at=submitted_at,
        universe=definition.universe,
        data_requirements=definition.data_requirements,
        feature_graph=definition.feature_graph,
        position_model=definition.position_model,
        execution_preferences=definition.execution_preferences,
        risk_policy=definition.risk_policy,
        validation_policy=definition.validation_policy,
        metadata=definition.metadata,
        dataset_bundle_id=bundle.bundle_id,
        dataset_plan=plan,
    )
