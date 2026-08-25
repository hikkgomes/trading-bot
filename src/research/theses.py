"""Immutable thesis registration and lineage-wide trial accounting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

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
            existing = (
                connection.execute(
                    select(research_thesis).where(research_thesis.c.id == thesis.thesis_id)
                )
                .mappings()
                .first()
            )
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
            existing = (
                connection.execute(
                    select(thesis_trial).where(thesis_trial.c.candidate_id == candidate_id)
                )
                .mappings()
                .first()
            )
            if existing is not None:
                if existing["thesis_id"] != thesis_id or existing["lineage_id"] != lineage_id:
                    raise ThesisError("candidate trial identity collision")
                return ThesisTrial(thesis_id, candidate_id, lineage_id, existing["ordinal"])
            count = connection.execute(
                select(func.count())
                .select_from(thesis_trial)
                .where(thesis_trial.c.thesis_id == thesis_id)
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
    def strings(name: str) -> tuple[str, ...]:
        value = payload[name]
        if not isinstance(value, list | tuple):
            raise ThesisError(f"persisted thesis {name} must be a list")
        return tuple(str(item) for item in value)

    def mapping(name: str) -> dict[str, Any]:
        value = payload[name]
        if not isinstance(value, Mapping):
            raise ThesisError(f"persisted thesis {name} must be an object")
        return dict(value)

    return ResearchThesis(
        mechanism_category=MechanismCategory(str(payload["mechanism_category"])),
        market_rationale=str(payload["market_rationale"]),
        expected_causal_chain=strings("expected_causal_chain"),
        expected_direction=str(payload["expected_direction"]),
        expected_horizon=str(payload["expected_horizon"]),
        required_data=strings("required_data"),
        permitted_features=strings("permitted_features"),
        instrument_universe=strings("instrument_universe"),
        generalisation_scope=mapping("generalisation_scope"),
        failure_regimes=strings("failure_regimes"),
        falsification_tests=strings("falsification_tests"),
        negative_controls=strings("negative_controls"),
        execution_capacity_assumptions=mapping("execution_capacity_assumptions"),
        parent_thesis_ids=strings("parent_thesis_ids"),
        cumulative_trial_budget=int(str(payload["cumulative_trial_budget"])),
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


ThesisBuilder = Callable[[str, tuple[str, ...], str], ResearchThesis]


class StrategyThesisFactory:
    """Mechanism-specific thesis authority for every semantic family."""

    def __init__(self) -> None:
        self._builders: dict[str, ThesisBuilder] = {}

    def register(self, family: str, builder: ThesisBuilder) -> None:
        if family in self._builders:
            raise ThesisError(f"thesis builder already registered: {family}")
        self._builders[family] = builder

    def build(
        self, *, name: str, family: str, product: str, instrument_universe: tuple[str, ...]
    ) -> ResearchThesis:
        try:
            thesis = self._builders[family](name, instrument_universe, product)
        except KeyError as exc:
            raise ThesisError(f"no thesis builder is registered for {family}") from exc
        self.validate(name=name, family=family, thesis=thesis)
        return thesis

    @staticmethod
    def validate(*, name: str, family: str, thesis: ResearchThesis) -> None:
        allowed = _FAMILY_MECHANISMS[family]
        if thesis.mechanism_category not in allowed:
            raise ThesisError(f"{name} thesis mechanism does not match {family}")
        required = _FAMILY_REQUIRED_DATA[family]
        if not required.issubset(thesis.required_data):
            raise ThesisError(f"{name} thesis required-data contract does not match {family}")
        if name in thesis.permitted_features:
            raise ThesisError("strategy names cannot be used as permitted features")

    @classmethod
    def default(cls) -> StrategyThesisFactory:
        factory = cls()
        for family in _FAMILY_MECHANISMS:
            factory.register(family, _build_family_thesis)
        return factory


_FAMILY_MECHANISMS = {
    "time_series": frozenset({MechanismCategory.BEHAVIOURAL, MechanismCategory.RISK_PREMIUM}),
    "mean_reversion": frozenset(
        {
            MechanismCategory.BEHAVIOURAL,
            MechanismCategory.LIQUIDITY,
            MechanismCategory.MARKET_STRUCTURE,
        }
    ),
    "cross_sectional": frozenset(
        {
            MechanismCategory.INFORMATION_DIFFUSION,
            MechanismCategory.RISK_PREMIUM,
            MechanismCategory.CARRY,
        }
    ),
    "relative_value": frozenset({MechanismCategory.RELATIVE_VALUE, MechanismCategory.CARRY}),
    "microstructure": frozenset(
        {
            MechanismCategory.LIQUIDITY,
            MechanismCategory.MARKET_STRUCTURE,
            MechanismCategory.FORCED_FLOW,
        }
    ),
    "machine_learning": frozenset({MechanismCategory.INFORMATION_DIFFUSION}),
    "advanced_alpha": frozenset(
        {
            MechanismCategory.BEHAVIOURAL,
            MechanismCategory.INFORMATION_DIFFUSION,
            MechanismCategory.RISK_PREMIUM,
        }
    ),
    "meta_strategy": frozenset({MechanismCategory.RISK_PREMIUM}),
    "execution": frozenset({MechanismCategory.EXECUTION, MechanismCategory.LIQUIDITY}),
    "market_making": frozenset({MechanismCategory.LIQUIDITY, MechanismCategory.EXECUTION}),
}

_FAMILY_REQUIRED_DATA = {
    "time_series": frozenset({"closed_ohlcv_bars"}),
    "mean_reversion": frozenset({"closed_ohlcv_bars"}),
    "cross_sectional": frozenset({"point_in_time_instrument_panel"}),
    "relative_value": frozenset({"synchronised_linked_instruments"}),
    "microstructure": frozenset({"sequenced_order_book", "public_trades"}),
    "machine_learning": frozenset({"immutable_feature_manifest", "frozen_model_artefact"}),
    "advanced_alpha": frozenset({"point_in_time_market_data"}),
    "meta_strategy": frozenset({"as_of_alpha_forecasts"}),
    "execution": frozenset({"target_delta", "market_state"}),
    "market_making": frozenset({"sequenced_order_book", "private_order_events"}),
}

_FAMILY_FEATURES = {
    "time_series": ("returns", "trend", "breakout", "realised_volatility"),
    "mean_reversion": ("normalised_price_deviation", "oscillator", "range_state"),
    "cross_sectional": ("point_in_time_rank", "relative_return", "funding_rank"),
    "relative_value": ("causal_spread", "hedge_ratio", "basis", "funding_differential"),
    "microstructure": (
        "spread",
        "depth_imbalance",
        "microprice",
        "aggressor_flow",
        "liquidation_flow",
    ),
    "machine_learning": ("ordered_frozen_feature_vector",),
    "advanced_alpha": (
        "calendar_effect",
        "realised_volatility_state",
        "change_point_score",
        "sentiment_score",
    ),
    "meta_strategy": ("forecast_conflict", "forecast_decay", "correlation", "regime"),
    "execution": ("spread", "visible_depth", "volume_curve", "urgency"),
    "market_making": ("spread", "inventory", "queue_position", "adverse_selection"),
}


def _build_family_thesis(name: str, universe: tuple[str, ...], product: str) -> ResearchThesis:
    from src.strategies.manifest import manifest_by_name

    entry = manifest_by_name()[name]
    family = entry.family
    mechanism = {
        "time_series": MechanismCategory.BEHAVIOURAL,
        "mean_reversion": MechanismCategory.LIQUIDITY,
        "cross_sectional": MechanismCategory.INFORMATION_DIFFUSION,
        "relative_value": MechanismCategory.RELATIVE_VALUE,
        "microstructure": MechanismCategory.MARKET_STRUCTURE,
        "machine_learning": MechanismCategory.INFORMATION_DIFFUSION,
        "advanced_alpha": MechanismCategory.INFORMATION_DIFFUSION,
        "meta_strategy": MechanismCategory.RISK_PREMIUM,
        "execution": MechanismCategory.EXECUTION,
        "market_making": MechanismCategory.LIQUIDITY,
    }[family]
    if name == "funding_adjusted_ranking":
        mechanism = MechanismCategory.CARRY
    elif name == "spot_perpetual_basis":
        mechanism = MechanismCategory.RELATIVE_VALUE
    elif "liquidation" in name:
        mechanism = MechanismCategory.FORCED_FLOW
    causal = {
        "time_series": ("persistent demand imbalance", "price trend", "subsequent return"),
        "mean_reversion": ("temporary displacement", "liquidity provision", "price reversion"),
        "cross_sectional": (
            "uneven information diffusion",
            "relative rank",
            "cross-sectional return",
        ),
        "relative_value": ("linked-price divergence", "hedged convergence", "spread return"),
        "microstructure": (
            "order-flow imbalance",
            "short-lived price pressure",
            "event-time return",
        ),
        "machine_learning": ("point-in-time state", "frozen model inference", "calibrated return"),
        "advanced_alpha": (
            "predeclared observable state",
            "causal statistical transformation",
            "subsequent return",
        ),
        "meta_strategy": ("forecast regime and decay", "capital weighting", "portfolio return"),
        "execution": ("target delta and liquidity", "order scheduling", "implementation shortfall"),
        "market_making": (
            "two-sided liquidity demand",
            "inventory-aware quoting",
            "maker spread capture",
        ),
    }[family]
    return ResearchThesis(
        mechanism_category=mechanism,
        market_rationale=f"Predeclared {family} mechanism for {name}.",
        expected_causal_chain=causal,
        expected_direction="family contract output",
        expected_horizon={
            "microstructure": "event-time to 5m",
            "execution": "order lifetime",
            "relative_value": "1h to 30d",
        }.get(family, "1h to 30d"),
        required_data=tuple(sorted(_FAMILY_REQUIRED_DATA[family])),
        permitted_features=_FAMILY_FEATURES[family],
        instrument_universe=universe,
        generalisation_scope={"product": product, "family": family, "predeclared": True},
        failure_regimes=("structural break", "insufficient liquidity", "capacity breach"),
        falsification_tests=(
            "chronological holdout",
            "cost stress",
            "mechanism-specific negative controls",
        ),
        negative_controls=REQUIRED_NEGATIVE_CONTROLS,
        execution_capacity_assumptions={
            "maximum_participation": 0.01,
            "market_impact_model_required": True,
        },
        parent_thesis_ids=(),
        cumulative_trial_budget=12,
        created_at="2026-01-01T00:00:00+00:00",
        creator_identity="strategy-thesis-factory/v2",
    )
