from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest

from src.data.database import PlatformDatabase
from src.data.feature_graph import (
    AvailableValue,
    FeatureGraph,
    FeatureGraphEngine,
    FeatureGraphError,
    FeatureNode,
    FeatureNodeType,
)
from src.domain._codec import canonical_hash
from src.domain.strategies import (
    MechanismCategory,
    ResearchThesis,
    StrategyDefinition,
    StrategySourceType,
)
from src.research.executors import ProviderExecutorRegistry
from src.research.theses import SqlThesisRegistry, ThesisError, ThesisRegistry
from src.strategies.frozen_model import FrozenLinearModel
from src.strategies.manifest import (
    execution_policy_manifest,
    portfolio_meta_manifest,
    predictive_alpha_manifest,
)
from src.strategies.semantic import (
    SEMANTIC_STRATEGIES,
    HedgedTargets,
    LinkedInstrumentState,
    MicrostructureForecast,
    MicrostructureState,
    PointInTimePanel,
    RankedTargets,
    SemanticFamily,
)

NOW = dt.datetime(2026, 8, 23, tzinfo=dt.UTC).isoformat()


def _thesis(*, budget: int = 2) -> ResearchThesis:
    return ResearchThesis(
        mechanism_category=MechanismCategory.FORCED_FLOW,
        market_rationale="Forced liquidation flow can temporarily displace price.",
        expected_causal_chain=("liquidation", "temporary imbalance", "reversion"),
        expected_direction="reversion",
        expected_horizon="5m-1h",
        required_data=("liquidations", "trades"),
        permitted_features=("liquidation_imbalance",),
        instrument_universe=("BTCUSDT", "ETHUSDT"),
        generalisation_scope={"venues": ["binance"], "symbols": ["BTCUSDT", "ETHUSDT"]},
        failure_regimes=("persistent deleveraging",),
        falsification_tests=("no post-event reversal",),
        negative_controls=("placebo_event_times", "block_permutation"),
        execution_capacity_assumptions={"maximum_participation": 0.01},
        parent_thesis_ids=(),
        cumulative_trial_budget=budget,
        created_at=NOW,
        creator_identity="researcher:test",
    )


def test_strategy_version_uses_definition_content_not_display_version() -> None:
    values = {
        "identity": "trend",
        "family": "time_series",
        "product": "active_income",
        "universe": {"symbols": ["BTCUSDT"]},
        "data_requirements": {"bars": "1h"},
        "feature_graph": {"version": "v1"},
        "signal_model": {"kind": "trend"},
        "position_model": {"kind": "vol_scaled"},
        "execution_preferences": {"policy": "market"},
        "risk_policy": {"id": "risk-v1"},
        "validation_policy": {"id": "validation-v1"},
        "source_type": StrategySourceType.REGISTERED_PYTHON,
        "source_hash": "sha256:" + "1" * 64,
    }
    first = StrategyDefinition(version="display-1", **values)
    renamed = StrategyDefinition(version="display-2", **values)
    changed = StrategyDefinition(version="display-1", **{**values, "source_hash": "sha256:" + "2" * 64})

    assert first.strategy_version_id == renamed.strategy_version_id
    assert changed.strategy_version_id != first.strategy_version_id


def test_thesis_registry_shares_budget_across_lineage_variants() -> None:
    registry = ThesisRegistry()
    thesis_id = registry.register(_thesis(budget=2))

    registry.claim_trial(thesis_id=thesis_id, candidate_id="a", lineage_id="root")
    registry.claim_trial(thesis_id=thesis_id, candidate_id="b", lineage_id="mutation:a")
    with pytest.raises(ThesisError, match="budget"):
        registry.claim_trial(thesis_id=thesis_id, candidate_id="c", lineage_id="crossover:a:b")


def test_sql_thesis_registry_persists_and_enforces_one_lineage_budget(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'theses.sqlite3'}")
    database.create_schema()
    registry = SqlThesisRegistry(database.engine)
    thesis = _thesis(budget=2)
    assert registry.register(thesis) == thesis.thesis_id
    for ordinal, suffix in enumerate(("a", "b"), 1):
        trial = registry.claim_trial(
            thesis_id=thesis.thesis_id,
            candidate_id="sha256:" + suffix * 64,
            lineage_id="sha256:" + str(ordinal) * 64,
            claimed_at=NOW,
        )
        assert trial.ordinal == ordinal
    with pytest.raises(ThesisError, match="budget"):
        registry.claim_trial(
            thesis_id=thesis.thesis_id,
            candidate_id="sha256:" + "c" * 64,
            lineage_id="sha256:" + "3" * 64,
            claimed_at=NOW,
        )


def test_catalogues_keep_alpha_meta_and_execution_semantics_separate() -> None:
    assert predictive_alpha_manifest()
    assert {item.catalogue for item in predictive_alpha_manifest()} == {"predictive_alpha"}
    assert {item.catalogue for item in portfolio_meta_manifest()} == {"portfolio_meta"}
    assert {item.catalogue for item in execution_policy_manifest()} == {"execution_policy"}
    assert {item.output_contract for item in execution_policy_manifest()} == {"order_intents"}


def test_default_executor_registry_covers_every_promotable_source_type() -> None:
    registry = ProviderExecutorRegistry.default()
    for source_type in (
        StrategySourceType.REGISTERED_PYTHON,
        StrategySourceType.GENERATED_DSL,
        StrategySourceType.MACHINE_LEARNING,
        StrategySourceType.CROSS_SECTIONAL,
        StrategySourceType.RELATIVE_VALUE,
        StrategySourceType.MICROSTRUCTURE,
        StrategySourceType.ENSEMBLE,
        StrategySourceType.AGENT_GENERATED_PYTHON,
    ):
        assert callable(registry.executor_for(source_type))


def test_feature_graph_rejects_unavailable_inputs_and_non_determinism() -> None:
    graph = FeatureGraph(
        version="features-v1",
        nodes=(FeatureNode("return", FeatureNodeType.BAR_INDICATOR, (), {}),),
    )
    calls = 0

    def unstable(_node, _dependencies, _inputs):
        nonlocal calls
        calls += 1
        return calls

    evaluators = {node_type: unstable for node_type in FeatureNodeType}
    engine = FeatureGraphEngine(evaluators)
    with pytest.raises(FeatureGraphError, match="not available"):
        engine.evaluate(
            graph,
            information_timestamp=NOW,
            inputs={
                "close": AvailableValue(
                    1.0,
                    information_time=NOW,
                    availability_time="2026-08-23T00:00:01+00:00",
                )
            },
        )
    with pytest.raises(FeatureGraphError, match="non-deterministic"):
        engine.evaluate(graph, information_timestamp=NOW, inputs={})


@pytest.mark.parametrize(
    ("family", "value", "output_type"),
    (
        (
            SemanticFamily.CROSS_SECTIONAL,
            PointInTimePanel(
                NOW,
                {
                    "BTCUSDT": {"momentum": 0.1, "funding": 0.01},
                    "ETHUSDT": {"momentum": -0.1, "funding": 0.02},
                },
            ),
            RankedTargets,
        ),
        (
            SemanticFamily.RELATIVE_VALUE,
            LinkedInstrumentState(
                NOW,
                {
                    "btc_spot": {"price": 100.0, "beta": 1.0},
                    "btc_perp": {"price": 101.0, "beta": 1.0},
                },
            ),
            HedgedTargets,
        ),
        (
            SemanticFamily.MICROSTRUCTURE,
            MicrostructureState(NOW, 1, 99.0, 101.0, 12.0, 8.0, 2.0),
            MicrostructureForecast,
        ),
    ),
)
def test_every_semantic_alpha_registration_is_typed_deterministic_and_family_specific(
    family, value, output_type
) -> None:
    registrations = SEMANTIC_STRATEGIES.by_family(family)
    assert registrations
    for registration in registrations:
        output = registration.evaluate(value)
        assert isinstance(output, output_type)
        if isinstance(output, RankedTargets):
            assert sum(output.target_fractions.values()) == pytest.approx(0.0)
        elif isinstance(output, HedgedTargets):
            assert output.hedge_error == pytest.approx(0.0)
        else:
            assert output.expected_direction in {-1, 0, 1}


def test_frozen_model_requires_exact_artefact_and_ordered_feature_manifest(tmp_path) -> None:
    payload = {
        "model_type": "linear_return_v1",
        "feature_names": ["momentum", "funding"],
        "weights": [0.5, -0.25],
        "intercept": 0.01,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / "model.json"
    path.write_bytes(encoded)
    artefact_hash = "sha256:" + hashlib.sha256(encoded).hexdigest()
    manifest_hash = canonical_hash({"feature_names": ("momentum", "funding")})
    model = FrozenLinearModel.load(
        path,
        expected_artefact_hash=artefact_hash,
        expected_feature_manifest_hash=manifest_hash,
    )

    first = model.evaluate({"momentum": 0.2, "funding": 0.01})
    second = model.evaluate({"momentum": 0.2, "funding": 0.01})
    assert first == second
    with pytest.raises(ValueError, match="ordered manifest"):
        model.evaluate({"funding": 0.01, "momentum": 0.2})
