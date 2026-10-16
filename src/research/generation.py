"""Bounded hypothesis generation and durable research-memory contracts."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

from sqlalchemy import insert, select
from sqlalchemy.engine import Connection, Engine

from src.data.database import generation_feedback, strategy_identity
from src.data.transactions import connection_scope
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp, to_primitive
from src.domain.instruments import MarketType, canonical_instrument_id
from src.domain.strategies import (
    MechanismCategory,
    ResearchThesis,
    StrategyDefinition,
    StrategySourceType,
)
from src.research.coordinator import Candidate
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.theses import FAMILY_NEGATIVE_CONTROLS
from src.strategies.behaviour import behaviour_hash_for_definition


class GenerationError(ValueError):
    """A generation request violates a bounded research contract."""


@dataclass(frozen=True)
class BtcAllocationLimits:
    product_max_tactical_fraction: float
    instrument_maximum_fraction: float
    strategy_maximum_fraction: float
    sleeve_maximum_fraction: float
    minimum_btc_fraction: float

    def __post_init__(self) -> None:
        for name in (
            "product_max_tactical_fraction",
            "instrument_maximum_fraction",
            "strategy_maximum_fraction",
            "sleeve_maximum_fraction",
            "minimum_btc_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise GenerationError(f"{name} must be a finite fraction from zero to one")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> BtcAllocationLimits:
        required = (
            "product_max_tactical_fraction",
            "instrument_maximum_fraction",
            "strategy_maximum_fraction",
            "sleeve_maximum_fraction",
            "minimum_btc_fraction",
        )
        if any(name not in values for name in required):
            raise GenerationError("BTC allocation limits are incomplete")
        try:
            return cls(**{name: float(values[name]) for name in required})
        except (TypeError, ValueError) as exc:
            raise GenerationError("BTC allocation limits must be numeric") from exc

    @classmethod
    def conservative_default(cls) -> BtcAllocationLimits:
        return cls(0.0, 0.0, 0.0, 0.0, 1.0)

    @property
    def maximum_tactical_fraction(self) -> float:
        return min(
            self.product_max_tactical_fraction,
            self.instrument_maximum_fraction,
            self.strategy_maximum_fraction,
            self.sleeve_maximum_fraction,
            1.0 - self.minimum_btc_fraction,
        )

    def to_payload(self) -> dict[str, float]:
        return {
            "product_max_tactical_fraction": self.product_max_tactical_fraction,
            "instrument_maximum_fraction": self.instrument_maximum_fraction,
            "strategy_maximum_fraction": self.strategy_maximum_fraction,
            "sleeve_maximum_fraction": self.sleeve_maximum_fraction,
            "minimum_btc_fraction": self.minimum_btc_fraction,
            "maximum_tactical_fraction": self.maximum_tactical_fraction,
        }


def _btc_limits(value: BtcAllocationLimits | Mapping[str, Any] | None) -> BtcAllocationLimits:
    if isinstance(value, BtcAllocationLimits):
        return value
    if isinstance(value, Mapping):
        return BtcAllocationLimits.from_mapping(value)
    return BtcAllocationLimits.conservative_default()


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
    enabled_by_default: bool = True
    unavailable_reason: str | None = None
    failure_regimes: tuple[str, ...] = (
        "structural break",
        "insufficient liquidity",
        "capacity breach",
    )
    cumulative_trial_budget: int = 12

    @property
    def source_type(self) -> StrategySourceType:
        return {
            "cross_sectional": StrategySourceType.CROSS_SECTIONAL,
            "relative_value": StrategySourceType.RELATIVE_VALUE,
            "microstructure": StrategySourceType.MICROSTRUCTURE,
            "meta_strategy": StrategySourceType.ENSEMBLE,
        }.get(self.family, StrategySourceType.GENERATED_DSL)

    @property
    def semantic_strategy(self) -> str | None:
        return {
            "cross_sectional_carry": "funding_adjusted_ranking",
            "cross_sectional_momentum": "relative_momentum",
            "funding_carry": "funding_adjusted_ranking",
            "basis_convergence": "beta_neutral_spreads",
            "pairs_mean_reversion": "beta_neutral_spreads",
            "liquidation_flow_reversion": "bid_ask_depth_imbalance",
            "event_microstructure": "microprice_displacement",
            "ensemble_regime": "correlation_aware_ensemble",
        }.get(self.name)

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
    *,
    enabled_by_default: bool = True,
    unavailable_reason: str | None = None,
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
        enabled_by_default=enabled_by_default and unavailable_reason is None,
        unavailable_reason=unavailable_reason,
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
        (0.0005, 0.001, 0.002),
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
        (0.0005, 0.001, 0.002),
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
        (0.1, 0.25, 0.5),
        "market_neutral",
        "4h to 7d",
        "cross_sectional",
        unavailable_reason="funding-aware multi-leg settlement accounting is not implemented",
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
        (-0.005, -0.002, -0.001),
        "hedged",
        "1h to 30d",
        "pairs",
        unavailable_reason="spot inventory and a training-only hedge model are not implemented",
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
        enabled_by_default=False,
        unavailable_reason="historical order-book replay and latency execution are unavailable",
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
        (0.005, 0.01, 0.02),
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
        (0.002, 0.005, 0.01),
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
        (0.002, 0.005, 0.01),
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
        (0.1, 0.25, 0.5),
        "market_neutral",
        "4h to 7d",
        "funding_carry",
        unavailable_reason="funding-aware multi-leg settlement accounting is not implemented",
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
        (-0.005, -0.002, -0.001),
        "hedged",
        "1h to 30d",
        "pairs",
        unavailable_reason="training-only pair selection and frozen hedge estimates are not implemented",
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
        (0.1, 0.2, 0.4),
        "signed",
        "event-time to 5m",
        "scalping",
        enabled_by_default=False,
        unavailable_reason="historical order-book replay and latency execution are unavailable",
    ),
    _campaign(
        "ensemble_regime",
        "active_income",
        "meta_strategy",
        MechanismCategory.RISK_PREMIUM,
        ("as_of_alpha_forecasts",),
        ("correlation",),
        "correlation",
        "gt",
        (0.25, 0.5, 0.75),
        "signed",
        "15m to 3d",
        "intraday",
        enabled_by_default=False,
        unavailable_reason="portfolio allocation is not an independent alpha hypothesis",
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

    def __init__(
        self,
        engine: Engine,
        *,
        connection: Connection | None = None,
        as_of: str | None = None,
        feedback_ids: list[str] | None = None,
    ) -> None:
        self.engine = engine
        self.connection = connection
        self.as_of = as_of
        self.feedback_ids = feedback_ids

    def append(self, feedback: GenerationFeedback) -> str:
        payload = to_primitive(feedback)
        with connection_scope(self.engine, self.connection, write=True) as connection:
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
        query = select(generation_feedback.c.payload).order_by(
            generation_feedback.c.created_at, generation_feedback.c.id
        )
        if self.as_of is not None:
            query = query.where(generation_feedback.c.created_at <= self.as_of)
        if self.feedback_ids is not None:
            query = query.where(generation_feedback.c.id.in_(self.feedback_ids))
        with connection_scope(self.engine, self.connection, write=False) as connection:
            rows = connection.execute(query).scalars().all()
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
        allocations = _deficit_allocations(
            scores, feedback_by_campaign, total_budget, self.minimum_exploration_fraction
        )
        return tuple(
            CampaignAllocation(campaign.name, allocations[campaign.name], scores[campaign.name])
            for campaign in sorted(campaigns, key=lambda item: item.name)
        )

    @staticmethod
    def _score(feedback: Sequence[GenerationFeedback]) -> float:
        measured = {item.candidate_id: item for item in feedback if _economic_feedback(item)}
        if not measured:
            return 1.0
        counts = Counter(item.outcome for item in measured.values())
        successes = counts["accepted"]
        failures = counts["rejected"]
        return max(0.05, (1.0 + successes) / (1.0 + failures))


def _deficit_allocations(scores, feedback, budget: int, exploration: float) -> dict[str, int]:
    counts = {
        name: len(
            {
                item.candidate_id
                for item in feedback.get(name, ())
                if item.outcome == "generated" and item.candidate_id is not None
            }
        )
        for name in scores
    }
    total = sum(counts.values())
    weight_sum = sum(scores.values())
    weights = {
        name: exploration / len(scores) + (1 - exploration) * score / weight_sum
        for name, score in scores.items()
    }
    allocations = dict.fromkeys(scores, 0)
    for slot in range(budget):
        chosen = min(
            scores,
            key=lambda name: (
                -(weights[name] * (total + slot + 1) - counts[name] - allocations[name]),
                name,
            ),
        )
        allocations[chosen] += 1
    return allocations


def _allocated_variants(allocations, campaigns) -> tuple[tuple[CampaignSpec, int], ...]:
    primary = [
        (allocation.campaign, variant)
        for allocation in allocations
        for variant in range(min(allocation.trials, len(campaigns[allocation.campaign].thresholds)))
    ]
    remaining = [
        (name, variant)
        for name in sorted(campaigns)
        for variant in range(len(campaigns[name].thresholds))
        if (name, variant) not in primary
    ]
    return tuple((campaigns[name], variant) for name, variant in (*primary, *remaining))


def _economic_feedback(item: GenerationFeedback) -> bool:
    value = item.metadata.get("net_return")
    return (
        item.candidate_id is not None
        and item.outcome in {"accepted", "rejected"}
        and item.metadata.get("stage") == "development"
        and item.metadata.get("economic_engine") == ECONOMIC_ENGINE_VERSION
        and isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


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
    return sum(
        _value_distance(left_payload.get(key), right_payload.get(key)) for key in keys
    ) / len(keys)


def _value_distance(left: Any, right: Any) -> float:
    if left == right:
        return 0.0
    if (
        isinstance(left, int | float)
        and not isinstance(left, bool)
        and isinstance(right, int | float)
        and not isinstance(right, bool)
    ):
        difference = abs(float(left) - float(right))
        return min(1.0, difference / (1.0 + max(abs(float(left)), abs(float(right)))))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = tuple(sorted(set(left) | set(right)))
        return sum(_value_distance(left.get(key), right.get(key)) for key in keys) / max(
            1, len(keys)
        )
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        if not left and not right:
            return 0.0
        length = max(len(left), len(right))
        return (
            sum(
                _value_distance(
                    left[index] if index < len(left) else None,
                    right[index] if index < len(right) else None,
                )
                for index in range(length)
            )
            / length
        )
    return 1.0


class HypothesisMemory(Protocol):
    def find(
        self, candidate: Candidate, *, maximum_distance: float = 0.2
    ) -> DuplicateMatch | None: ...


class SqlHypothesisMemory:
    """Search immutable strategy identities for exact and semantic duplicates."""

    def __init__(self, engine: Engine, *, connection: Connection | None = None) -> None:
        self.engine = engine
        self.connection = connection

    def find(self, candidate: Candidate, *, maximum_distance: float = 0.2) -> DuplicateMatch | None:
        if not 0.0 <= maximum_distance <= 1.0:
            raise GenerationError("maximum duplicate distance must be between zero and one")
        signature = hypothesis_signature(candidate.definition)
        nearest = None
        with connection_scope(self.engine, self.connection, write=False) as connection:
            rows = connection.execute(
                select(
                    strategy_identity.c.id,
                    strategy_identity.c.behavior_hash,
                    strategy_identity.c.submitted_spec,
                    strategy_identity.c.metadata,
                ).order_by(strategy_identity.c.created_at, strategy_identity.c.id)
            ).mappings()
            for row in rows:
                if str(row["behavior_hash"]) == behaviour_hash_for_definition(candidate.definition):
                    return DuplicateMatch(str(row["id"]), "exact", 0.0, signature)
                distance = semantic_distance(candidate.definition, row["submitted_spec"])
                if distance <= maximum_distance and (
                    nearest is None or distance < nearest.distance
                ):
                    nearest = DuplicateMatch(str(row["id"]), "near", distance, signature)
        return nearest


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
        btc_allocation_limits: BtcAllocationLimits | Mapping[str, Any] | None = None,
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
        self.btc_allocation_limits = _btc_limits(btc_allocation_limits)

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
        dataset_plan: Any | None = None,
        instrument_cursor: int = 0,
    ) -> tuple[GeneratedHypothesis, ...]:
        if instrument_cursor < 0:
            raise GenerationError("instrument cursor cannot be negative")
        available_campaigns = CAMPAIGNS if campaigns is None else campaigns
        selected = tuple(
            campaign
            for campaign in available_campaigns
            if campaign.product == self.product
            and campaign.unavailable_reason is None
            and (campaigns is not None or campaign.enabled_by_default)
        )
        allocations = self.allocator.allocate(
            selected,
            total_budget=total_budget,
            feedback=self.feedback_store.load() if self.feedback_store is not None else (),
        )
        by_name = {campaign.name: campaign for campaign in selected}
        generated: list[GeneratedHypothesis] = []
        scope_cursor = instrument_cursor
        for campaign, variant in _allocated_variants(allocations, by_name):
            if len(generated) >= total_budget:
                break
            candidate_universe = self.instrument_universe
            if (
                self.product == "active_income"
                and campaign.source_type is StrategySourceType.GENERATED_DSL
            ):
                candidate_universe = (
                    self.instrument_universe[scope_cursor % len(self.instrument_universe)],
                )
                scope_cursor += 1
            try:
                hypothesis = build_hypothesis(
                    campaign,
                    variant=variant,
                    instrument_universe=candidate_universe,
                    dataset_snapshot_hashes=dataset_snapshot_hashes,
                    submitted_at=submitted_at,
                    dataset_bundle_id=dataset_bundle_id,
                    universe_snapshot_id=universe_snapshot_id,
                    parent_thesis_ids=parent_thesis_ids,
                    btc_allocation_limits=self.btc_allocation_limits,
                )
            except GenerationError as exc:
                self._record_generation_failure(
                    campaign,
                    variant=variant,
                    observed_at=submitted_at,
                    reason_code=str(exc),
                )
                continue
            if dataset_plan is not None:
                candidate = replace(hypothesis.candidate, dataset_plan=dataset_plan)
                hypothesis = replace(
                    hypothesis,
                    candidate=candidate,
                    semantic_signature=hypothesis_signature(candidate.definition),
                )
            match = self.memory.find(hypothesis.candidate) if self.memory is not None else None
            if match is not None and match.kind == "exact":
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

    def _record_generation_failure(
        self,
        campaign: CampaignSpec,
        *,
        variant: int,
        observed_at: str,
        reason_code: str,
    ) -> None:
        if self.feedback_store is None:
            return
        self.feedback_store.append(
            GenerationFeedback(
                campaign=campaign.name,
                outcome="retired",
                observed_at=observed_at,
                reason_code=reason_code,
                semantic_signature=canonical_hash({"campaign": campaign.name, "variant": variant}),
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
        and candidate.definition.universe == base.candidate.definition.universe
        and isinstance(candidate.definition.signal_model, Mapping)
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
    parent_models = [candidate.definition.signal_model for candidate in parents]
    if not all(isinstance(model, Mapping) for model in parent_models):
        raise GenerationError("derived hypotheses need executable parent models")
    source_payload = _derived_source_payload(
        base.campaign,
        parent_models,
        parents=parents,
        method=method,
        btc_limits=_btc_limits(
            base.candidate.definition.position_model.get("btc_allocation_limits")
            if isinstance(base.candidate.definition.position_model, Mapping)
            else None
        ),
    )
    identity = f"{method}:{base.campaign.name}:{canonical_hash(source_payload).removeprefix('sha256:')[:16]}"
    definition = replace(
        base.candidate.definition,
        identity=identity,
        version=f"{method}-v1",
        signal_model=source_payload,
        position_model=_derived_position_model(
            base.candidate.definition.position_model,
            base.campaign,
            method=method,
            btc_limits=_btc_limits(
                base.candidate.definition.position_model.get("btc_allocation_limits")
                if isinstance(base.candidate.definition.position_model, Mapping)
                else None
            ),
        ),
        source_type=base.campaign.source_type,
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


def _derived_source_payload(
    campaign: CampaignSpec,
    parent_models: Sequence[Mapping[str, Any]],
    *,
    parents: Sequence[Candidate],
    method: str,
    btc_limits: BtcAllocationLimits,
) -> dict[str, Any]:
    """Mutate the economic contract, not only a display threshold."""

    first = deepcopy(dict(parent_models[0]))
    if campaign.source_type is StrategySourceType.GENERATED_DSL:
        return _derived_dsl_source_payload(
            campaign,
            first,
            parent_models,
            parents=parents,
            method=method,
            btc_limits=btc_limits,
        )
    contract = dict(first)
    contract["method"] = method
    contract["parents"] = [candidate.candidate_id for candidate in parents]
    contract["mutation_contract"] = {
        "mechanism": campaign.mechanism.value,
        "feature_space": list(campaign.permitted_features),
        "horizon": campaign.expected_horizon,
        "position_model": _campaign_position_model(campaign, btc_limits=btc_limits),
    }
    contract["parameters"] = _mutate_contract_parameters(
        campaign,
        parent_models,
        method=method,
    )
    if campaign.source_type is StrategySourceType.CROSS_SECTIONAL:
        panel = dict(contract.get("panel_contract") or {})
        panel["long_count"] = contract["parameters"]["top_k"]
        panel["short_count"] = contract["parameters"]["top_k"]
        panel["minimum_score"] = abs(float(contract["parameters"].get("threshold", 0.0)))
        contract["panel_contract"] = panel
    elif campaign.source_type is StrategySourceType.RELATIVE_VALUE:
        hedge_ratio = float(contract["parameters"]["hedge_ratio"])
        contract["hedge_ratio"] = hedge_ratio
        legs = list(contract.get("paired_legs") or ())
        if len(legs) >= 2 and isinstance(legs[1], Mapping):
            legs[1] = {**dict(legs[1]), "weight": -hedge_ratio}
        contract["paired_legs"] = legs
    elif campaign.source_type is StrategySourceType.MICROSTRUCTURE:
        event_contract = dict(contract.get("event_contract") or {})
        event_contract["minimum_score"] = contract["parameters"]["minimum_score"]
        contract["event_contract"] = event_contract
    contract["generator_schema"] = "economic_contract/v1"
    return contract


def _derived_dsl_source_payload(
    campaign: CampaignSpec,
    first: Mapping[str, Any],
    parent_models: Sequence[Mapping[str, Any]],
    *,
    parents: Sequence[Candidate],
    method: str,
    btc_limits: BtcAllocationLimits,
) -> dict[str, Any]:
    base_rule = first.get("rule")
    if not isinstance(base_rule, Mapping):
        raise GenerationError("directional derived strategy has no typed rule")
    rule = dict(base_rule)
    if method == "mutation":
        _mutate_rule(rule, campaign)
    else:
        sibling = parent_models[1].get("rule") if len(parent_models) > 1 else None
        if not isinstance(sibling, Mapping):
            raise GenerationError("directional crossover has no second typed rule")
        rule = _crossover_rule(rule, sibling, campaign)
    parent_ids = [candidate.candidate_id for candidate in parents]
    if campaign.product != "btc_accumulation":
        return {
            "kind": "typed_rule",
            "method": method,
            "mechanism": campaign.mechanism.value,
            "horizon": campaign.expected_horizon,
            "rule": rule,
            "entry_contract": {
                "feature_space": list(campaign.permitted_features),
                "threshold_unit": _threshold_unit(campaign),
            },
            "exit_contract": {
                "kind": "signal_reversal_or_terminal_close",
                "horizon": campaign.expected_horizon,
            },
            "parameters": _rule_parameters(rule),
            "parameter_free": False,
            "parents": parent_ids,
            "generator_schema": "bounded_hypothesis/v2",
        }
    if first.get("kind") != "btc_target_allocation":
        raise GenerationError("BTC derived strategy lost its allocation contract")
    allocation_limits = btc_limits.to_payload()
    maximum_tactical = btc_limits.maximum_tactical_fraction
    parameters = dict(first.get("parameters") or {})
    for group in ("negative_conditions", "positive_conditions"):
        conditions = rule.get(group)
        if isinstance(conditions, list) and conditions:
            parameters["threshold"] = conditions[0].get("threshold", parameters.get("threshold"))
            break
    return {
        "kind": "btc_target_allocation",
        "method": method,
        "mechanism": campaign.mechanism.value,
        "horizon": campaign.expected_horizon,
        "rule": rule,
        "entry_contract": {
            "feature_space": list(campaign.permitted_features),
            "threshold_unit": _threshold_unit(campaign),
        },
        "exit_contract": {
            "kind": "state_reversal_or_terminal_close",
            "horizon": campaign.expected_horizon,
        },
        "target_states": {
            "risk_on": {"btc_fraction": 1.0, "tactical_fraction": 0.0},
            "risk_off": {
                "btc_fraction": 1.0 - maximum_tactical,
                "tactical_fraction": maximum_tactical,
            },
            "reentry": {"btc_fraction": 1.0, "tactical_fraction": 0.0},
        },
        "minimum_btc_fraction": btc_limits.minimum_btc_fraction,
        "maximum_tactical_fraction": maximum_tactical,
        "btc_allocation_limits": allocation_limits,
        "parameters": parameters,
        "parameter_free": False,
        "parents": parent_ids,
        "generator_schema": "btc_allocation_contract/v1",
    }


def _mutate_rule(rule: dict[str, Any], campaign: CampaignSpec) -> None:
    original_feature = str(rule.get("feature") or campaign.feature)
    alternate = next(
        (feature for feature in campaign.permitted_features if feature != original_feature),
        original_feature,
    )
    changed_feature = False
    for group in (
        "conditions",
        "positive_conditions",
        "negative_conditions",
        "long_conditions",
        "short_conditions",
    ):
        conditions = rule.get(group)
        if not isinstance(conditions, list):
            continue
        for condition in conditions:
            if isinstance(condition, dict) and "threshold" in condition:
                threshold = float(condition["threshold"])
                condition["threshold"] = threshold + max(abs(threshold) * 0.1, 0.001)
                if str(condition.get("feature") or "") == original_feature:
                    condition["feature"] = alternate
                    condition["threshold"] = _mutated_feature_threshold(
                        alternate, condition["threshold"]
                    )
                    changed_feature = changed_feature or alternate != original_feature
                break
        else:
            continue
        break
    else:
        threshold = float(rule.get("threshold", campaign.thresholds[0]))
        rule["threshold"] = threshold + max(abs(threshold) * 0.1, 0.001)
        if str(rule.get("feature") or "") == original_feature:
            rule["feature"] = alternate
            rule["threshold"] = _mutated_feature_threshold(alternate, rule["threshold"])
            changed_feature = changed_feature or alternate != original_feature
    if len(campaign.permitted_features) < 2:
        return
    if "feature" in rule:
        entry = {
            "feature": str(rule["feature"]),
            "operator": str(rule["operator"]),
            "threshold": float(rule["threshold"]),
        }
        direction = str(rule.get("direction") or campaign.expected_direction)
        if direction in {"signed", "market_neutral", "hedged"}:
            rule.pop("feature", None)
            rule.pop("operator", None)
            rule.pop("threshold", None)
            rule["positive_conditions"] = [entry]
            rule["negative_conditions"] = [
                {
                    "feature": entry["feature"],
                    "operator": "lt",
                    "threshold": -abs(float(cast(Any, entry["threshold"]))),
                }
            ]
        else:
            rule.pop("feature", None)
            rule.pop("operator", None)
            rule.pop("threshold", None)
            rule["conditions"] = [entry]
    rule["mutation_feature"] = alternate
    rule["mutation_changed_feature"] = changed_feature
    exit_threshold = 1.0 if "z_score" in _threshold_unit(campaign) else 0.0
    rule["exit_conditions"] = [
        {
            "feature": alternate,
            "operator": "gt",
            "threshold": exit_threshold,
        }
    ]


def _crossover_rule(
    left: Mapping[str, Any], right: Mapping[str, Any], campaign: CampaignSpec
) -> dict[str, Any]:
    result = dict(left)
    for group in (
        "conditions",
        "positive_conditions",
        "negative_conditions",
        "long_conditions",
        "short_conditions",
    ):
        merged = _crossover_condition_group(left.get(group), right.get(group), campaign)
        if merged is not None:
            result[group] = merged
    for key in ("threshold", "positive_threshold", "negative_threshold"):
        if key in left and key in right:
            result[key] = (float(left[key]) + float(right[key])) / 2.0
    if left.get("feature") != right.get("feature"):
        result["feature"] = campaign.feature
    if left.get("direction") != right.get("direction"):
        result["direction"] = campaign.expected_direction
    return result


def _crossover_condition_group(
    left: Any, right: Any, campaign: CampaignSpec
) -> list[dict[str, Any]] | None:
    if not isinstance(left, list | tuple) or not isinstance(right, list | tuple):
        return None
    if len(left) != len(right):
        return None
    merged: list[dict[str, Any]] = []
    for first, second in zip(left, right, strict=True):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            return None
        item = dict(first)
        item["feature"] = (
            first.get("feature")
            if first.get("feature") == second.get("feature")
            else campaign.feature
        )
        item["operator"] = (
            first.get("operator")
            if first.get("operator") == second.get("operator")
            else campaign.operator
        )
        if first.get("threshold") is not None and second.get("threshold") is not None:
            item["threshold"] = (float(first["threshold"]) + float(second["threshold"])) / 2.0
        merged.append(item)
    return merged


def _rule_parameters(rule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: rule[key]
        for key in ("feature", "threshold", "positive_threshold", "negative_threshold")
        if key in rule
    }


def _mutated_feature_threshold(feature: str, threshold: float) -> float:
    """Keep a mutation in the units of the feature it actually uses."""

    scales = {
        "returns": 0.001,
        "trend": 0.001,
        "relative_return": 0.001,
        "breakout": 0.001,
        "realised_volatility": 0.005,
        "normalised_price_deviation": 0.5,
        "oscillator": 0.5,
        "range_state": 0.001,
    }
    minimum = scales.get(feature, 0.001)
    return math.copysign(max(abs(threshold), minimum), threshold or 1.0)


def _derived_position_model(
    base: Mapping[str, Any],
    campaign: CampaignSpec,
    *,
    method: str,
    btc_limits: BtcAllocationLimits,
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    result.update(_campaign_position_model(campaign, btc_limits=btc_limits))
    result["derived_method"] = method
    return result


def _mutate_contract_parameters(
    campaign: CampaignSpec,
    parent_models: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    parameters: list[Mapping[str, Any]] = []
    for model in parent_models:
        raw_parameters = model.get("parameters")
        parameters.append(raw_parameters if isinstance(raw_parameters, Mapping) else {})
    if campaign.source_type is StrategySourceType.CROSS_SECTIONAL:
        values = [float(item.get("top_k", 1)) for item in parameters]
        thresholds = [float(item.get("threshold", 0.0)) for item in parameters]
        threshold_step = 0.05 if campaign.feature == "funding_rank" else 0.001
        return {
            "top_k": max(
                1, int(round(sum(values) / len(values))) + (1 if method == "mutation" else 0)
            ),
            "threshold": abs(sum(thresholds) / len(thresholds))
            + (threshold_step if method == "mutation" else 0.0),
        }
    if campaign.source_type is StrategySourceType.RELATIVE_VALUE:
        values = [float(item.get("hedge_ratio", 1.0)) for item in parameters]
        ratio = sum(values) / len(values)
        return {"hedge_ratio": ratio * (1.05 if method == "mutation" else 1.0)}
    if campaign.source_type is StrategySourceType.MICROSTRUCTURE:
        values = [
            abs(float(item.get("minimum_score", item.get("threshold", 0.1)))) for item in parameters
        ]
        return {
            "minimum_score": min(
                1.0, sum(values) / len(values) + (0.02 if method == "mutation" else 0.0)
            )
        }
    aggregation_values = [str(item.get("aggregation", "correlation_aware")) for item in parameters]
    return {
        "aggregation": (
            aggregation_values[0] if len(set(aggregation_values)) == 1 else "correlation_aware"
        )
    }


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
    btc_allocation_limits: BtcAllocationLimits | Mapping[str, Any] | None = None,
) -> GeneratedHypothesis:
    if campaign.unavailable_reason is not None:
        raise GenerationError(
            f"unsupported campaign {campaign.name}: {campaign.unavailable_reason}"
        )
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
    source_type = campaign.source_type
    if (
        source_type
        in {
            StrategySourceType.CROSS_SECTIONAL,
            StrategySourceType.RELATIVE_VALUE,
            StrategySourceType.MICROSTRUCTURE,
            StrategySourceType.ENSEMBLE,
        }
        and len(instrument_universe) < 2
    ):
        raise GenerationError(
            f"{campaign.name} requires at least two instruments for its economic contract"
        )
    market_type = MarketType.SPOT if campaign.product == "btc_accumulation" else MarketType.FUTURES
    resolved_btc_limits = (
        _btc_limits(btc_allocation_limits) if campaign.product == "btc_accumulation" else None
    )
    instrument_ids = tuple(
        canonical_instrument_id(
            symbol,
            market_type=market_type,
            settlement_asset="USDT" if market_type is MarketType.FUTURES else None,
        )
        for symbol in instrument_universe
    )
    source_payload = _campaign_source_payload(
        campaign,
        threshold=float(threshold),
        instrument_universe=instrument_ids,
        btc_limits=resolved_btc_limits,
    )
    universe: dict[str, Any] = {
        "type": "fixed",
        "symbols": list(instrument_universe),
        "instrument_ids": list(instrument_ids),
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
        position_model=_campaign_position_model(campaign, btc_limits=resolved_btc_limits),
        execution_preferences={"policy": "market", "paper_only_until_approved": True},
        risk_policy={
            "product_policy": campaign.product,
            **(
                {"btc_allocation_limits": resolved_btc_limits.to_payload()}
                if resolved_btc_limits is not None
                else {}
            ),
        },
        validation_policy={"evidence_type": campaign.evidence_type, "promotable": True},
        source_type=source_type,
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


def _campaign_source_payload(
    campaign: CampaignSpec,
    *,
    threshold: float,
    instrument_universe: tuple[str, ...],
    btc_limits: BtcAllocationLimits | None = None,
) -> dict[str, Any]:
    if campaign.source_type is StrategySourceType.GENERATED_DSL:
        if campaign.product == "btc_accumulation":
            limits = btc_limits or BtcAllocationLimits.conservative_default()
            allocation_limits = limits.to_payload()
            maximum_tactical = limits.maximum_tactical_fraction
            risk_off: dict[str, Any] = {
                "feature": campaign.feature,
                "operator": campaign.operator,
                "threshold": threshold,
            }
            risk_on: dict[str, Any] = {
                "feature": campaign.feature,
                "operator": "le" if campaign.operator in {"gt", "ge"} else "ge",
                "threshold": threshold,
            }
            if campaign.name != "btc_risk_off_reentry":
                risk_on, risk_off = risk_off, risk_on
            rule = {
                "direction": "signed",
                "positive_conditions": [risk_on],
                "negative_conditions": [risk_off],
                "condition_mode": "all",
            }
            return {
                "kind": "btc_target_allocation",
                "campaign": campaign.name,
                "mechanism": campaign.mechanism.value,
                "horizon": campaign.expected_horizon,
                "rule": rule,
                "entry_contract": {
                    "feature": campaign.feature,
                    "operator": campaign.operator,
                    "threshold_unit": _threshold_unit(campaign),
                },
                "exit_contract": {
                    "kind": "state_reversal_or_terminal_close",
                    "horizon": campaign.expected_horizon,
                },
                "target_states": {
                    "risk_on": {"btc_fraction": 1.0, "tactical_fraction": 0.0},
                    "risk_off": {
                        "btc_fraction": 1.0 - maximum_tactical,
                        "tactical_fraction": maximum_tactical,
                    },
                    "reentry": {"btc_fraction": 1.0, "tactical_fraction": 0.0},
                },
                "minimum_btc_fraction": limits.minimum_btc_fraction,
                "maximum_tactical_fraction": maximum_tactical,
                "btc_allocation_limits": allocation_limits,
                "parameters": {
                    "feature": campaign.feature,
                    "threshold": threshold,
                    "threshold_unit": _threshold_unit(campaign),
                },
                "parameter_free": False,
                "generator_schema": "btc_allocation_contract/v1",
            }
        directional_rule = {
            "feature": campaign.feature,
            "operator": campaign.operator,
            "threshold": threshold,
            "direction": campaign.expected_direction,
        }
        if campaign.expected_direction == "signed":
            directional_rule = _signed_campaign_rule(campaign, threshold)
        return {
            "kind": "signed_directional_forecast",
            "campaign": campaign.name,
            "mechanism": campaign.mechanism.value,
            "horizon": campaign.expected_horizon,
            "rule": directional_rule,
            "entry_contract": {
                "feature": campaign.feature,
                "operator": campaign.operator,
                "threshold_unit": _threshold_unit(campaign),
            },
            "exit_contract": {
                "kind": "signal_reversal_or_terminal_close",
                "horizon": campaign.expected_horizon,
            },
            "parameters": {
                "feature": campaign.feature,
                "threshold": threshold,
                "threshold_unit": _threshold_unit(campaign),
            },
            "parameter_free": False,
            "generator_schema": "directional_contract/v1",
        }
    contract: dict[str, Any] = {
        "kind": {
            StrategySourceType.CROSS_SECTIONAL: "ranked_long_short_panel",
            StrategySourceType.RELATIVE_VALUE: "paired_legs_hedged",
            StrategySourceType.MICROSTRUCTURE: "event_time_microstructure",
            StrategySourceType.ENSEMBLE: "forecast_ensemble",
        }[campaign.source_type],
        "campaign": campaign.name,
        "mechanism": campaign.mechanism.value,
        "horizon": campaign.expected_horizon,
        "semantic_strategy": campaign.semantic_strategy,
        "instrument_scope": list(instrument_universe),
        "parameters": {"threshold": threshold, "threshold_unit": _threshold_unit(campaign)},
        "parameter_free": False,
        "generator_schema": "economic_contract/v1",
        "execution_contract": {
            "signal_timing": "next_observation",
            "terminal_close_required": True,
            "leg_failure_recovery": "block_new_exposure",
        },
    }
    if campaign.source_type is StrategySourceType.CROSS_SECTIONAL:
        contract["panel_contract"] = {
            "rank_feature": campaign.feature,
            "long_count": 1,
            "short_count": 1,
            "minimum_score": abs(threshold),
            "gross_target_fraction": 1.0,
            "net_target_fraction": 0.0,
        }
    elif campaign.source_type is StrategySourceType.RELATIVE_VALUE:
        contract["paired_legs"] = [
            {"instrument_id": instrument_universe[0], "weight": 1.0},
            {"instrument_id": instrument_universe[1], "weight": -1.0},
        ]
        contract["hedge_ratio"] = 1.0
    elif campaign.source_type is StrategySourceType.MICROSTRUCTURE:
        contract["parameters"]["minimum_score"] = abs(threshold)
        contract["event_contract"] = {
            "requires_sequence": True,
            "requires_bid_ask": True,
            "requires_trade_flow": True,
            "minimum_score": abs(threshold),
            "maximum_spread_bps": 10.0,
        }
    else:
        contract["forecast_inputs"] = ["directional", "cross_sectional", "relative_value"]
        contract["aggregation"] = "correlation_aware"
    return contract


def _signed_campaign_rule(campaign: CampaignSpec, threshold: float) -> dict[str, Any]:
    opposite = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge"}
    return {
        "direction": "signed",
        "long_conditions": [
            {
                "feature": campaign.feature,
                "operator": campaign.operator,
                "threshold": threshold,
            }
        ],
        "short_conditions": [
            {
                "feature": campaign.feature,
                "operator": opposite[campaign.operator],
                "threshold": -threshold,
            }
        ],
    }


def _campaign_position_model(
    campaign: CampaignSpec,
    *,
    btc_limits: BtcAllocationLimits | None = None,
) -> dict[str, Any]:
    if campaign.product == "btc_accumulation":
        limits = btc_limits or BtcAllocationLimits.conservative_default()
        return {
            "kind": "target_allocation",
            "base_asset": "BTC",
            "minimum_btc_fraction": limits.minimum_btc_fraction,
            "maximum_tactical_fraction": limits.maximum_tactical_fraction,
            "btc_allocation_limits": limits.to_payload(),
            "states": ("risk_on", "risk_off", "reentry"),
            "signal_timing": "next_bar",
        }
    if campaign.source_type is StrategySourceType.CROSS_SECTIONAL:
        return {"kind": "ranked_panel_targets", "net_exposure": 0.0, "gross_exposure": 1.0}
    if campaign.source_type is StrategySourceType.RELATIVE_VALUE:
        return {"kind": "paired_hedged_targets", "net_beta": 0.0, "legs": 2}
    if campaign.source_type is StrategySourceType.MICROSTRUCTURE:
        return {"kind": "event_time_signed_position", "maximum_holding_seconds": 300}
    if campaign.source_type is StrategySourceType.ENSEMBLE:
        return {"kind": "portfolio_forecast_aggregation", "net_exposure": 0.5}
    from src.strategies.position_state import directional_position_model

    return directional_position_model(32 if campaign.evidence_type == "intraday" else 96)


def _threshold_unit(campaign: CampaignSpec) -> str:
    return {
        "breakout": "fraction_of_prior_high",
        "trend": "fractional_return",
        "realised_volatility": "fractional_volatility",
        "normalised_price_deviation": "z_score",
        "oscillator": "oscillator_unit",
        "range_state": "fractional_range",
        "funding_rank": "cross_sectional_rank",
        "relative_return": "fractional_return",
        "causal_spread": "standardised_spread",
        "basis": "fractional_basis",
        "liquidation_flow": "signed_flow_z_score",
        "aggressor_flow": "signed_flow_z_score",
        "regime": "regime_score",
        "correlation": "correlation_unit",
    }.get(campaign.feature, "feature_units")


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
