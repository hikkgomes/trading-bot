"""Bounded hypothesis generation and durable research-memory contracts."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import generation_feedback, strategy_identity
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp, to_primitive
from src.domain.instruments import MarketType, canonical_instrument_id
from src.domain.strategies import (
    MechanismCategory,
    ResearchThesis,
    StrategyDefinition,
    StrategySourceType,
)
from src.research.coordinator import Candidate
from src.research.theses import FAMILY_NEGATIVE_CONTROLS


class GenerationError(ValueError):
    """A generation request violates a bounded research contract."""


@dataclass(frozen=True)
class CampaignSpec:
    """One predeclared economic mechanism and its executable rule family."""

    name: str
    product: str
    family: str
    mechanism: MechanismCategory
    required_data: tuple[str, ...]
    permitted_features: tuple[str, ...]
    feature: str
    operator: str
    thresholds: tuple[float, ...]
    expected_direction: str
    expected_horizon: str
    evidence_type: str
    failure_regimes: tuple[str, ...] = (
        "structural break",
        "insufficient liquidity",
        "capacity breach",
    )
    cumulative_trial_budget: int = 12

    def __post_init__(self) -> None:
        for name in (
            "name",
            "product",
            "family",
            "feature",
            "operator",
            "expected_direction",
            "expected_horizon",
            "evidence_type",
        ):
            non_empty(getattr(self, name), field=name)
        if self.operator not in {"gt", "ge", "lt", "le"}:
            raise GenerationError(f"unsupported campaign operator: {self.operator}")
        if not self.required_data or not self.permitted_features or not self.thresholds:
            raise GenerationError("campaign data, feature, and threshold contracts are required")
        if self.feature not in self.permitted_features:
            raise GenerationError("campaign rule feature is not permitted by its thesis")
        if self.cumulative_trial_budget < len(self.thresholds):
            raise GenerationError("campaign trial budget cannot be smaller than its variants")
        if any(not math.isfinite(float(value)) for value in self.thresholds):
            raise GenerationError("campaign thresholds must be finite")


def _campaign(
    name: str,
    product: str,
    family: str,
    mechanism: MechanismCategory,
    required_data: tuple[str, ...],
    features: tuple[str, ...],
    feature: str,
    operator: str,
    thresholds: tuple[float, ...],
    direction: str,
    horizon: str,
    evidence_type: str,
) -> CampaignSpec:
    return CampaignSpec(
        name=name,
        product=product,
        family=family,
        mechanism=mechanism,
        required_data=required_data,
        permitted_features=features,
        feature=feature,
        operator=operator,
        thresholds=thresholds,
        expected_direction=direction,
        expected_horizon=horizon,
        evidence_type=evidence_type,
    )


CAMPAIGNS: tuple[CampaignSpec, ...] = (
    _campaign(
        "btc_trend_breakout",
        "btc_accumulation",
        "time_series",
        MechanismCategory.BEHAVIOURAL,
        ("closed_ohlcv_bars",),
        ("returns", "trend", "breakout", "realised_volatility"),
        "trend",
        "gt",
        (-0.25, 0.0, 0.25),
        "long",
        "1h to 7d",
        "btc_allocation",
    ),
    _campaign(
        "btc_pullback_reversion",
        "btc_accumulation",
        "mean_reversion",
        MechanismCategory.LIQUIDITY,
        ("closed_ohlcv_bars",),
        ("normalised_price_deviation", "oscillator", "range_state"),
        "normalised_price_deviation",
        "lt",
        (-2.0, -1.0, -0.5),
        "long",
        "1h to 3d",
        "btc_allocation",
    ),
    _campaign(
        "futures_trend_following",
        "active_income",
        "time_series",
        MechanismCategory.BEHAVIOURAL,
        ("closed_ohlcv_bars",),
        ("returns", "trend", "breakout", "realised_volatility"),
        "trend",
        "gt",
        (-0.25, 0.0, 0.25),
        "signed",
        "15m to 3d",
        "swing",
    ),
    _campaign(
        "futures_mean_reversion",
        "active_income",
        "mean_reversion",
        MechanismCategory.MARKET_STRUCTURE,
        ("closed_ohlcv_bars",),
        ("normalised_price_deviation", "oscillator", "range_state"),
        "normalised_price_deviation",
        "lt",
        (-2.0, -1.0, -0.5),
        "signed",
        "5m to 12h",
        "intraday",
    ),
    _campaign(
        "cross_sectional_carry",
        "active_income",
        "cross_sectional",
        MechanismCategory.CARRY,
        ("point_in_time_instrument_panel",),
        ("point_in_time_rank", "relative_return", "funding_rank"),
        "funding_rank",
        "gt",
        (-0.5, 0.0, 0.5),
        "market_neutral",
        "4h to 7d",
        "cross_sectional",
    ),
    _campaign(
        "basis_convergence",
        "active_income",
        "relative_value",
        MechanismCategory.RELATIVE_VALUE,
        ("synchronised_linked_instruments",),
        ("causal_spread", "hedge_ratio", "basis", "funding_differential"),
        "basis",
        "lt",
        (-2.0, -1.0, -0.5),
        "hedged",
        "1h to 30d",
        "pairs",
    ),
    _campaign(
        "liquidation_flow_reversion",
        "active_income",
        "microstructure",
        MechanismCategory.FORCED_FLOW,
        ("sequenced_order_book", "public_trades"),
        ("spread", "depth_imbalance", "microprice", "aggressor_flow", "liquidation_flow"),
        "liquidation_flow",
        "lt",
        (-2.0, -1.0, -0.5),
        "signed",
        "event-time to 5m",
        "scalping",
    ),
    _campaign(
        "btc_risk_off_reentry",
        "btc_accumulation",
        "time_series",
        MechanismCategory.BEHAVIOURAL,
        ("closed_ohlcv_bars",),
        ("returns", "trend", "breakout", "realised_volatility"),
        "realised_volatility",
        "gt",
        (0.5, 1.0, 1.5),
        "long",
        "1h to 7d",
        "btc_allocation",
    ),
    _campaign(
        "futures_breakout",
        "active_income",
        "time_series",
        MechanismCategory.BEHAVIOURAL,
        ("closed_ohlcv_bars",),
        ("returns", "trend", "breakout", "realised_volatility"),
        "breakout",
        "gt",
        (0.25, 0.5, 1.0),
        "signed",
        "15m to 3d",
        "swing",
    ),
    _campaign(
        "cross_sectional_momentum",
        "active_income",
        "cross_sectional",
        MechanismCategory.INFORMATION_DIFFUSION,
        ("point_in_time_instrument_panel",),
        ("point_in_time_rank", "relative_return", "funding_rank"),
        "relative_return",
        "gt",
        (0.25, 0.5, 1.0),
        "market_neutral",
        "4h to 7d",
        "cross_sectional",
    ),
    _campaign(
        "funding_carry",
        "active_income",
        "cross_sectional",
        MechanismCategory.CARRY,
        ("point_in_time_instrument_panel",),
        ("point_in_time_rank", "relative_return", "funding_rank"),
        "funding_rank",
        "gt",
        (0.25, 0.5, 1.0),
        "market_neutral",
        "4h to 7d",
        "funding_carry",
    ),
    _campaign(
        "pairs_mean_reversion",
        "active_income",
        "relative_value",
        MechanismCategory.RELATIVE_VALUE,
        ("synchronised_linked_instruments",),
        ("causal_spread", "hedge_ratio", "basis", "funding_differential"),
        "causal_spread",
        "lt",
        (-2.0, -1.0, -0.5),
        "hedged",
        "1h to 30d",
        "pairs",
    ),
    _campaign(
        "event_microstructure",
        "active_income",
        "microstructure",
        MechanismCategory.FORCED_FLOW,
        ("sequenced_order_book", "public_trades"),
        ("spread", "depth_imbalance", "microprice", "aggressor_flow", "liquidation_flow"),
        "aggressor_flow",
        "gt",
        (0.5, 1.0, 2.0),
        "signed",
        "event-time to 5m",
        "scalping",
    ),
    _campaign(
        "ensemble_regime",
        "active_income",
        "meta_strategy",
        MechanismCategory.RISK_PREMIUM,
        ("as_of_alpha_forecasts",),
        ("forecast_conflict", "forecast_decay", "correlation", "regime"),
        "regime",
        "gt",
        (0.25, 0.5, 1.0),
        "signed",
        "15m to 3d",
        "intraday",
    ),
)


@dataclass(frozen=True)
class GenerationFeedback:
    campaign: str
    outcome: str
    observed_at: str
    candidate_id: str | None = None
    reason_code: str | None = None
    semantic_signature: str | None = None
    distance: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        non_empty(self.campaign, field="campaign")
        if self.outcome not in {
            "generated",
            "accepted",
            "rejected",
            "duplicate_exact",
            "duplicate_near",
            "data_unavailable",
            "resource_budget_exhausted",
            "retired",
        }:
            raise GenerationError(f"unsupported generation feedback outcome: {self.outcome}")
        object.__setattr__(self, "observed_at", timestamp(self.observed_at, field="observed_at"))
        if self.candidate_id is not None:
            non_empty(self.candidate_id, field="candidate_id")
        if self.semantic_signature is not None:
            if (
                not self.semantic_signature.startswith("sha256:")
                or len(self.semantic_signature) != 71
            ):
                raise GenerationError("semantic_signature must be a sha256 identity")
        if self.distance is not None and not 0.0 <= float(self.distance) <= 1.0:
            raise GenerationError("generation feedback distance must be between zero and one")
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), field="metadata"))

    @property
    def feedback_id(self) -> str:
        return canonical_hash(self)


class GenerationFeedbackStore(Protocol):
    def append(self, feedback: GenerationFeedback) -> str: ...

    def load(self, *, campaign: str | None = None) -> tuple[GenerationFeedback, ...]: ...


class SqlGenerationFeedbackStore:
    """Append-only PostgreSQL/SQLite memory for allocator feedback."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def append(self, feedback: GenerationFeedback) -> str:
        payload = to_primitive(feedback)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(generation_feedback.c.payload).where(
                    generation_feedback.c.id == feedback.feedback_id
                )
            ).scalar_one_or_none()
            if existing is None:
                connection.execute(
                    insert(generation_feedback).values(
                        id=feedback.feedback_id,
                        created_at=feedback.observed_at,
                        payload=payload,
                    )
                )
            elif existing != payload:
                raise GenerationError("generation feedback identity collision")
        return feedback.feedback_id

    def load(self, *, campaign: str | None = None) -> tuple[GenerationFeedback, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(generation_feedback.c.payload).order_by(
                    generation_feedback.c.created_at,
                    generation_feedback.c.id,
                )
            ).scalars()
        result = tuple(_feedback_from_payload(payload) for payload in rows)
        if campaign is None:
            return result
        return tuple(item for item in result if item.campaign == campaign)


@dataclass(frozen=True)
class CampaignAllocation:
    campaign: str
    trials: int
    score: float


class GenerationAllocator:
    """Allocate a finite trial budget with a guaranteed exploration floor."""

    def __init__(self, *, minimum_exploration_fraction: float = 0.2) -> None:
        if not 0.0 <= minimum_exploration_fraction <= 1.0:
            raise GenerationError("minimum exploration fraction must be between zero and one")
        self.minimum_exploration_fraction = minimum_exploration_fraction

    def allocate(
        self,
        campaigns: Sequence[CampaignSpec],
        *,
        total_budget: int,
        feedback: Iterable[GenerationFeedback] = (),
    ) -> tuple[CampaignAllocation, ...]:
        if total_budget < 1:
            raise GenerationError("generation budget must be positive")
        if not campaigns:
            return ()
        feedback_by_campaign: dict[str, list[GenerationFeedback]] = {}
        for item in feedback:
            feedback_by_campaign.setdefault(item.campaign, []).append(item)
        scores = {
            campaign.name: self._score(feedback_by_campaign.get(campaign.name, ()))
            for campaign in campaigns
        }
        count = len(campaigns)
        floor_total = min(
            total_budget, max(0, int(total_budget * self.minimum_exploration_fraction))
        )
        floor_each = floor_total // count
        allocations = {campaign.name: floor_each for campaign in campaigns}
        remaining = total_budget - floor_each * count
        if remaining > 0:
            weights = {name: max(score, 0.05) for name, score in scores.items()}
            weight_total = sum(weights.values())
            raw = {name: remaining * weight / weight_total for name, weight in weights.items()}
            for name, value in raw.items():
                allocations[name] += int(value)
            left = total_budget - sum(allocations.values())
            order = sorted(
                campaigns, key=lambda item: (-(raw[item.name] - int(raw[item.name])), item.name)
            )
            for campaign in order[:left]:
                allocations[campaign.name] += 1
        return tuple(
            CampaignAllocation(campaign.name, allocations[campaign.name], scores[campaign.name])
            for campaign in sorted(campaigns, key=lambda item: item.name)
        )

    @staticmethod
    def _score(feedback: Sequence[GenerationFeedback]) -> float:
        if not feedback:
            return 1.0
        counts = Counter(item.outcome for item in feedback)
        successes = counts["accepted"]
        failures = counts["rejected"] + counts["data_unavailable"] + counts["retired"]
        duplicates = counts["duplicate_exact"] + counts["duplicate_near"]
        return max(0.05, (1.0 + successes) / (1.0 + failures + duplicates))


@dataclass(frozen=True)
class DuplicateMatch:
    candidate_id: str
    kind: str
    distance: float
    semantic_signature: str


def _semantic_payload(value: StrategyDefinition | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, StrategyDefinition):
        payload = to_primitive(value)
    else:
        payload = dict(value)
    return {
        key: payload.get(key)
        for key in (
            "family",
            "product",
            "universe",
            "data_requirements",
            "feature_graph",
            "signal_model",
            "position_model",
            "execution_preferences",
            "risk_policy",
            "validation_policy",
        )
    }


def hypothesis_signature(value: StrategyDefinition | Mapping[str, Any]) -> str:
    """Return a semantic identity independent of display name and source hash."""

    return canonical_hash(_semantic_payload(value))


def semantic_distance(
    left: StrategyDefinition | Mapping[str, Any], right: StrategyDefinition | Mapping[str, Any]
) -> float:
    left_payload = _semantic_payload(left)
    right_payload = _semantic_payload(right)
    if left_payload.get("product") != right_payload.get("product"):
        return 1.0
    if left_payload.get("family") != right_payload.get("family"):
        return 1.0
    keys = tuple(sorted(set(left_payload) | set(right_payload)))
    if not keys:
        return 0.0
    equal = sum(left_payload.get(key) == right_payload.get(key) for key in keys)
    return 1.0 - (equal / len(keys))


class HypothesisMemory(Protocol):
    def find(
        self, candidate: Candidate, *, maximum_distance: float = 0.2
    ) -> DuplicateMatch | None: ...


class SqlHypothesisMemory:
    """Search immutable strategy identities for exact and semantic duplicates."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def find(self, candidate: Candidate, *, maximum_distance: float = 0.2) -> DuplicateMatch | None:
        if not 0.0 <= maximum_distance <= 1.0:
            raise GenerationError("maximum duplicate distance must be between zero and one")
        signature = hypothesis_signature(candidate.definition)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    strategy_identity.c.id,
                    strategy_identity.c.behavior_hash,
                    strategy_identity.c.submitted_spec,
                    strategy_identity.c.metadata,
                ).order_by(strategy_identity.c.created_at, strategy_identity.c.id)
            ).mappings()
            for row in rows:
                if str(row["behavior_hash"]) == candidate.definition.definition_hash:
                    return DuplicateMatch(str(row["id"]), "exact", 0.0, signature)
                distance = semantic_distance(candidate.definition, row["submitted_spec"])
                if distance <= maximum_distance:
                    return DuplicateMatch(str(row["id"]), "near", distance, signature)
        return None


@dataclass(frozen=True)
class GeneratedHypothesis:
    campaign: CampaignSpec
    thesis: ResearchThesis
    candidate: Candidate
    semantic_signature: str


class HypothesisGenerator:
    """Compile bounded campaign allocations into normal queue candidates."""

    def __init__(
        self,
        *,
        product: str,
        instrument_universe: tuple[str, ...],
        allocator: GenerationAllocator | None = None,
        memory: HypothesisMemory | None = None,
        feedback_store: GenerationFeedbackStore | None = None,
    ) -> None:
        self.product = non_empty(product, field="product")
        if not instrument_universe:
            raise GenerationError("hypothesis generation requires an instrument universe")
        self.instrument_universe = tuple(sorted(set(instrument_universe)))
        if any(not item for item in self.instrument_universe):
            raise GenerationError("instrument universe contains an empty symbol")
        if self.product == "btc_accumulation" and self.instrument_universe != ("BTCUSDT",):
            raise GenerationError("BTC accumulation research is restricted to BTCUSDT spot")
        self.allocator = allocator or GenerationAllocator()
        self.memory = memory
        self.feedback_store = feedback_store

    def generate(
        self,
        *,
        dataset_snapshot_hashes: tuple[str, ...],
        submitted_at: str,
        total_budget: int,
        campaigns: Sequence[CampaignSpec] | None = None,
        dataset_bundle_id: str | None = None,
        universe_snapshot_id: str | None = None,
        parent_thesis_ids: tuple[str, ...] = (),
        parent_candidates: Sequence[Candidate] = (),
    ) -> tuple[GeneratedHypothesis, ...]:
        selected = tuple(
            campaign for campaign in (campaigns or CAMPAIGNS) if campaign.product == self.product
        )
        allocations = self.allocator.allocate(
            selected,
            total_budget=total_budget,
            feedback=self.feedback_store.load() if self.feedback_store is not None else (),
        )
        by_name = {campaign.name: campaign for campaign in selected}
        generated: list[GeneratedHypothesis] = []
        for allocation in allocations:
            campaign = by_name[allocation.campaign]
            variants = min(allocation.trials, len(campaign.thresholds))
            for variant in range(variants):
                hypothesis = build_hypothesis(
                    campaign,
                    variant=variant,
                    instrument_universe=self.instrument_universe,
                    dataset_snapshot_hashes=dataset_snapshot_hashes,
                    submitted_at=submitted_at,
                    dataset_bundle_id=dataset_bundle_id,
                    universe_snapshot_id=universe_snapshot_id,
                    parent_thesis_ids=parent_thesis_ids,
                )
                match = self.memory.find(hypothesis.candidate) if self.memory is not None else None
                if match is not None:
                    derived = _derive_duplicate_hypothesis(
                        hypothesis,
                        match=match,
                        parent_candidates=parent_candidates,
                        submitted_at=submitted_at,
                    )
                    if derived is not None:
                        derived_match = (
                            self.memory.find(derived.candidate, maximum_distance=0.0)
                            if self.memory is not None
                            else None
                        )
                        if derived_match is None:
                            generated.append(derived)
                            self._record_outcome(derived, "generated", submitted_at)
                            continue
                    self._record_duplicate(hypothesis, match, submitted_at)
                    continue
                generated.append(hypothesis)
                self._record_outcome(hypothesis, "generated", submitted_at)
        return tuple(generated)

    def _record_duplicate(
        self, hypothesis: GeneratedHypothesis, match: DuplicateMatch, observed_at: str
    ) -> None:
        self._record_outcome(
            hypothesis,
            "duplicate_exact" if match.kind == "exact" else "duplicate_near",
            observed_at,
            candidate_id=match.candidate_id,
            distance=match.distance,
        )

    def _record_outcome(
        self,
        hypothesis: GeneratedHypothesis,
        outcome: str,
        observed_at: str,
        *,
        candidate_id: str | None = None,
        distance: float | None = None,
    ) -> None:
        if self.feedback_store is None:
            return
        self.feedback_store.append(
            GenerationFeedback(
                campaign=hypothesis.campaign.name,
                outcome=outcome,
                observed_at=observed_at,
                candidate_id=candidate_id,
                semantic_signature=hypothesis.semantic_signature,
                distance=distance,
            )
        )


def campaign_thesis(
    campaign: CampaignSpec,
    *,
    instrument_universe: tuple[str, ...],
    created_at: str,
    parent_thesis_ids: tuple[str, ...] = (),
) -> ResearchThesis:
    if not instrument_universe:
        raise GenerationError("generated hypotheses require a non-empty instrument universe")
    return ResearchThesis(
        mechanism_category=campaign.mechanism,
        market_rationale=(f"Predeclared {campaign.mechanism.value} mechanism: {campaign.name}."),
        expected_causal_chain=(
            "observable market state",
            "mechanism-specific imbalance",
            "subsequent risk-adjusted return",
        ),
        expected_direction=campaign.expected_direction,
        expected_horizon=campaign.expected_horizon,
        required_data=campaign.required_data,
        permitted_features=campaign.permitted_features,
        instrument_universe=instrument_universe,
        generalisation_scope={
            "product": campaign.product,
            "family": campaign.family,
            "campaign": campaign.name,
            "predeclared": True,
        },
        failure_regimes=campaign.failure_regimes,
        falsification_tests=(
            "chronological holdout",
            "cost stress",
            "mechanism-specific negative controls",
        ),
        negative_controls=FAMILY_NEGATIVE_CONTROLS.get(
            campaign.family,
            ("block_permutation", "feature_ablation"),
        ),
        execution_capacity_assumptions={
            "maximum_participation": 0.01,
            "market_impact_model_required": True,
        },
        parent_thesis_ids=parent_thesis_ids,
        cumulative_trial_budget=campaign.cumulative_trial_budget,
        created_at=created_at,
        creator_identity="bounded-hypothesis-generator/v1",
    )


def _derive_duplicate_hypothesis(
    base: GeneratedHypothesis,
    *,
    match: DuplicateMatch,
    parent_candidates: Sequence[Candidate],
    submitted_at: str,
) -> GeneratedHypothesis | None:
    parents = tuple(
        candidate
        for candidate in parent_candidates
        if candidate.definition.product == base.campaign.product
        and candidate.definition.family == base.campaign.family
        and isinstance(candidate.definition.signal_model.get("rule"), Mapping)
    )
    if not parents:
        return None
    if len(parents) >= 2 and match.kind == "near":
        return _derive_hypothesis(
            base,
            parents=parents[:2],
            method="crossover",
            submitted_at=submitted_at,
        )
    return _derive_hypothesis(
        base,
        parents=(parents[0],),
        method="mutation",
        submitted_at=submitted_at,
    )


def _derive_hypothesis(
    base: GeneratedHypothesis,
    *,
    parents: tuple[Candidate, ...],
    method: str,
    submitted_at: str,
) -> GeneratedHypothesis:
    if method not in {"mutation", "crossover"} or not parents:
        raise GenerationError("unsupported derived hypothesis method")
    rules = [
        dict(candidate.definition.signal_model["rule"])
        for candidate in parents
        if isinstance(candidate.definition.signal_model.get("rule"), Mapping)
    ]
    if len(rules) != len(parents):
        raise GenerationError("derived hypotheses need executable parent rules")
    if method == "mutation":
        rule = dict(rules[0])
        threshold = float(rule.get("threshold", base.campaign.thresholds[0]))
        step = max(abs(threshold) * 0.1, 0.1)
        rule["threshold"] = threshold + step
    else:
        rule = {
            "feature": base.campaign.feature,
            "operator": base.campaign.operator,
            "threshold": sum(float(item.get("threshold", 0.0)) for item in rules) / len(rules),
            "direction": (
                rules[0].get("direction")
                if all(item.get("direction") == rules[0].get("direction") for item in rules)
                else "signed"
            ),
        }
    source_payload = {
        "kind": "typed_rule",
        "method": method,
        "rule": rule,
        "parameters": {"threshold": float(rule["threshold"])},
        "parameter_free": False,
        "parents": [candidate.candidate_id for candidate in parents],
        "generator_schema": "bounded_hypothesis/v1",
    }
    identity = f"{method}:{base.campaign.name}:{canonical_hash(source_payload).removeprefix('sha256:')[:16]}"
    definition = replace(
        base.candidate.definition,
        identity=identity,
        version=f"{method}-v1",
        signal_model=source_payload,
        source_type=StrategySourceType.GENERATED_DSL,
        source_hash=canonical_hash(source_payload),
        metadata={
            **dict(base.candidate.definition.metadata),
            "generated_method": method,
            "parent_candidate_ids": [candidate.candidate_id for candidate in parents],
            "promotable": True,
        },
    )
    thesis = campaign_thesis(
        base.campaign,
        instrument_universe=tuple(base.candidate.definition.universe.get("symbols", ())),
        created_at=submitted_at,
        parent_thesis_ids=tuple(dict.fromkeys(candidate.thesis_id for candidate in parents)),
    )
    candidate = replace(
        base.candidate,
        definition=definition,
        thesis_id=thesis.thesis_id,
        lineage_id=canonical_hash(
            {
                "method": method,
                "parents": [candidate.lineage_id for candidate in parents],
                "thesis_id": thesis.thesis_id,
            }
        ),
        provider=f"bounded_hypothesis_{method}",
    )
    return GeneratedHypothesis(
        campaign=base.campaign,
        thesis=thesis,
        candidate=candidate,
        semantic_signature=hypothesis_signature(definition),
    )


def build_hypothesis(
    campaign: CampaignSpec,
    *,
    variant: int,
    instrument_universe: tuple[str, ...],
    dataset_snapshot_hashes: tuple[str, ...],
    submitted_at: str,
    dataset_bundle_id: str | None = None,
    universe_snapshot_id: str | None = None,
    parent_thesis_ids: tuple[str, ...] = (),
) -> GeneratedHypothesis:
    if variant < 0 or variant >= len(campaign.thresholds):
        raise GenerationError("campaign variant is outside the declared threshold set")
    if not dataset_snapshot_hashes:
        raise GenerationError("generated hypotheses require canonical dataset identities")
    thesis = campaign_thesis(
        campaign,
        instrument_universe=instrument_universe,
        created_at=submitted_at,
        parent_thesis_ids=parent_thesis_ids,
    )
    threshold = campaign.thresholds[variant]
    rule = {
        "feature": campaign.feature,
        "operator": campaign.operator,
        "threshold": float(threshold),
        "direction": campaign.expected_direction,
    }
    source_payload = {
        "kind": "typed_rule",
        "campaign": campaign.name,
        "rule": rule,
        "parameters": {"threshold": float(threshold)},
        "parameter_free": False,
        "generator_schema": "bounded_hypothesis/v1",
    }
    market_type = MarketType.SPOT if campaign.product == "btc_accumulation" else MarketType.FUTURES
    universe: dict[str, Any] = {
        "type": "fixed",
        "symbols": list(instrument_universe),
        "instrument_ids": [
            canonical_instrument_id(
                symbol,
                market_type=market_type,
                settlement_asset="USDT" if market_type is MarketType.FUTURES else None,
            )
            for symbol in instrument_universe
        ],
    }
    if universe_snapshot_id is not None:
        universe = {
            "type": "point_in_time",
            "symbols": list(instrument_universe),
            "instrument_ids": list(universe["instrument_ids"]),
            "universe_snapshot_id": universe_snapshot_id,
        }
    definition = StrategyDefinition(
        identity=f"generated:{campaign.name}:{variant}",
        version="generated-v1",
        family=campaign.family,
        product=campaign.product,
        universe=universe,
        data_requirements={"required": list(campaign.required_data)},
        feature_graph={
            "version": "canonical-features/v2",
            "required_nodes": list(campaign.permitted_features),
        },
        signal_model=source_payload,
        position_model={"kind": "volatility_scaled", "signal_timing": "next_bar"},
        execution_preferences={"policy": "market", "paper_only_until_approved": True},
        risk_policy={"product_policy": campaign.product},
        validation_policy={"evidence_type": campaign.evidence_type, "promotable": True},
        source_type=StrategySourceType.GENERATED_DSL,
        source_hash=canonical_hash(source_payload),
        metadata={
            "generated": True,
            "campaign": campaign.name,
            "variant": variant,
            "thesis_id": thesis.thesis_id,
            "promotable": True,
        },
    )
    candidate = Candidate(
        definition=definition,
        thesis_id=thesis.thesis_id,
        lineage_id=canonical_hash(
            {
                "campaign": campaign.name,
                "thesis_id": thesis.thesis_id,
                "parent_thesis_ids": parent_thesis_ids,
            }
        ),
        provider="bounded_hypothesis_generator",
        dataset_snapshot_hashes=dataset_snapshot_hashes,
        submitted_at=submitted_at,
        metadata={
            "generation_campaign": campaign.name,
            "generation_variant": variant,
            "parent_hashes": list(parent_thesis_ids),
        },
        dataset_bundle_id=dataset_bundle_id,
    )
    return GeneratedHypothesis(campaign, thesis, candidate, hypothesis_signature(definition))


def _feedback_from_payload(payload: object) -> GenerationFeedback:
    if not isinstance(payload, Mapping):
        raise GenerationError("persisted generation feedback must be an object")
    return GenerationFeedback(
        campaign=str(payload["campaign"]),
        outcome=str(payload["outcome"]),
        observed_at=str(payload["observed_at"]),
        candidate_id=str(payload["candidate_id"])
        if payload.get("candidate_id") is not None
        else None,
        reason_code=str(payload["reason_code"]) if payload.get("reason_code") is not None else None,
        semantic_signature=(
            str(payload["semantic_signature"])
            if payload.get("semantic_signature") is not None
            else None
        ),
        distance=float(payload["distance"]) if payload.get("distance") is not None else None,
        metadata=dict(payload.get("metadata") or {}),
    )
