from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json

import pandas as pd
import pytest
from sqlalchemy import insert

from src.data.database import PlatformDatabase, dataset_snapshot, universe, universe_snapshot
from src.data.feature_graph import (
    AvailableValue,
    FeatureGraph,
    FeatureGraphEngine,
    FeatureGraphError,
    FeatureGraphRegistry,
    FeatureNode,
    FeatureNodeType,
    default_feature_engine,
)
from src.domain._codec import canonical_hash, to_primitive
from src.domain.strategies import (
    MechanismCategory,
    ResearchThesis,
    StrategyDefinition,
    StrategySourceType,
)
from src.research.catalogue import registered_strategy_candidates
from src.research.coordinator import ResearchCoordinator
from src.research.datasets import CanonicalDatasetResolver, SqlCanonicalDatasetRepository
from src.research.evaluation import (
    CanonicalResearchEvaluator,
    EvaluationRequest,
    EvidencePolicy,
)
from src.research.executors import ProviderContextBuilderRegistry, ProviderExecutorRegistry
from src.research.providers import provider_candidate
from src.research.store import SqlResearchStore
from src.research.theses import (
    SqlThesisRegistry,
    StrategyThesisFactory,
    ThesisError,
    ThesisRegistry,
)
from src.risk.engine import SqlRiskSnapshotStore
from src.services.artefact_dispatcher import ArtefactDispatcher, ArtefactDispatchError
from src.services.market_gateway import DatabaseMarketGateway, UserStreamAccount
from src.services.portfolio_state import DatabasePortfolioStateWorker
from src.services.promotion import (
    LifecycleState,
    PromotionEvidence,
    PromotionPolicy,
    decide_promotion,
)
from src.services.scheduler import DatabaseJobQueue
from src.strategies.advanced import (
    MarketMakingCapabilityPolicy,
    PortfolioMetaState,
    QuoteState,
    StrategyGenome,
    adaptive_portfolio_targets,
    discover_pairs,
    execution_schedule,
    genetic_fitness,
    market_making_quotes,
    microstructure_scalping_signal,
    recognise_statistical_pattern,
    replay_market_making,
)
from src.strategies.advanced import (
    MicrostructureState as AdvancedMicrostructureState,
)
from src.strategies.frozen_model import SAFE_MODEL_TYPES, FrozenLinearModel, FrozenSafeModel
from src.strategies.manifest import (
    execution_policy_manifest,
    portfolio_meta_manifest,
    predictive_alpha_manifest,
)
from src.strategies.registry import get as get_registered_strategy
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
    changed = StrategyDefinition(
        version="display-1", **{**values, "source_hash": "sha256:" + "2" * 64}
    )

    assert first.strategy_version_id == renamed.strategy_version_id
    assert changed.strategy_version_id != first.strategy_version_id


def test_default_feature_graph_is_dependency_closed_and_deterministic() -> None:
    registry = FeatureGraphRegistry.default()
    graph = registry.graph(("bid_ask_spread", "microprice"))
    inputs = {
        name: AvailableValue(value, NOW, NOW)
        for name, value in {
            "bid_price": 99.0,
            "ask_price": 101.0,
            "bid_depth": 3.0,
            "ask_depth": 1.0,
        }.items()
    }
    engine = default_feature_engine()
    first = engine.evaluate(graph, information_timestamp=NOW, inputs=inputs)
    second = engine.evaluate(graph, information_timestamp=NOW, inputs=inputs)

    assert first == second
    assert {"bid_ask_spread", "depth_imbalance", "microprice"}.issubset(first)


def test_thesis_factory_builds_family_specific_contracts() -> None:
    factory = StrategyThesisFactory.default()
    thesis = factory.build(
        name="pairs_trading",
        family="relative_value",
        product="active_income",
        instrument_universe=("BTCUSDT", "ETHUSDT"),
    )

    assert thesis.mechanism_category is MechanismCategory.RELATIVE_VALUE
    assert "synchronised_linked_instruments" in thesis.required_data
    assert "pairs_trading" not in thesis.permitted_features


def test_dispatcher_verifies_hashes_and_emits_execution_receipt() -> None:
    definition = {
        "source_type": "generated_dsl",
        "signal_model": {"rule": {"feature": "bar_return", "operator": "gt", "threshold": 0}},
    }
    definition_hash = canonical_hash(definition)
    artefact = {
        "definition": definition,
        "definition_hash": definition_hash,
        "position_limits": {"maximum_position": 0.2},
    }
    artefact["artefact_hash"] = canonical_hash(artefact)

    result = ArtefactDispatcher.default().evaluate({"bar_return": 0.01}, artefact)
    assert result["direction"] == "long"
    assert result["execution_receipt"]["definition_hash"] == definition_hash
    with pytest.raises(ArtefactDispatchError, match="content hash"):
        ArtefactDispatcher.default().evaluate({"bar_return": 0.01}, {**artefact, "x": 1})


def test_dispatcher_executes_every_autonomous_candidate_source_type() -> None:
    dispatcher = ArtefactDispatcher.default()
    for source_type in (
        "parameter_search",
        "mutation",
        "crossover",
        "agent_generated_python",
    ):
        definition = {
            "source_type": source_type,
            "signal_model": {
                "production_rule": {
                    "kind": "linear_feature_score/v1",
                    "terms": [{"feature": "bar_return", "scale": 1.0, "weight": 1.0}],
                }
            },
            "metadata": (
                {"sandbox_receipt": "sha256:" + "a" * 64}
                if source_type == "agent_generated_python"
                else {"derived_from": "sha256:" + "b" * 64}
            ),
        }
        definition_hash = canonical_hash(
            {key: value for key, value in definition.items() if key != "metadata"}
        )
        artefact = {
            "definition": definition,
            "definition_hash": definition_hash,
            "position_limits": {"maximum_position": 0.2},
        }
        artefact["artefact_hash"] = canonical_hash(artefact)
        result = dispatcher.evaluate({"bar_return": 0.01}, artefact)
        assert result["direction"] == "long"
        assert result["execution_receipt"]["source_type"] == source_type


def test_every_registered_catalogue_candidate_has_an_executable_live_graph() -> None:
    candidates = registered_strategy_candidates(
        product="active_income",
        dataset_snapshot_hashes=("sha256:" + "f" * 64,),
        instrument_universe=("BTCUSDT",),
    )
    dispatcher = ArtefactDispatcher.default()
    for candidate in candidates:
        definition = to_primitive(candidate.definition)
        required = tuple(definition["feature_graph"]["required_nodes"])
        assert required
        artefact = {
            "definition": definition,
            "definition_hash": candidate.definition.definition_hash,
            "position_limits": {"maximum_position": 0.2},
        }
        artefact["artefact_hash"] = canonical_hash(artefact)
        result = dispatcher.evaluate({name: 0.0 for name in required}, artefact)
        assert result["direction"] in {"long", "short", "flat"}
        assert result["execution_receipt"]["artefact_hash"] == artefact["artefact_hash"]


def test_advanced_execution_market_making_and_genetic_gates() -> None:
    schedule = execution_schedule("twap", quantity=4.0, slices=4, forecast_volume=100.0)
    assert len(schedule) == 4
    assert sum(item.quantity for item in schedule) == pytest.approx(4.0)
    quotes = market_making_quotes(QuoteState(NOW, 99.0, 101.0, 0.01, 10.0, 10.0, 0.1, 2.0, 2.0))
    assert quotes.emergency_inventory_exit is True
    assert MarketMakingCapabilityPolicy().authorise(mode="live", event_replay={}) is False
    genome = StrategyGenome(
        "thesis", (), "lineage", 0, ("BTCUSDT",), ("trend",), {"risk": 1.0}, {"lookback": 20.0}
    )
    assert genome.mutate({"lookback": 21.0}).thesis_id == genome.thesis_id
    with pytest.raises(ValueError, match="development evidence"):
        genetic_fitness({"stage": "development", "cost_adjusted_return": 1.0, "holdout": {}})


def test_advanced_alpha_and_meta_contracts_are_semantically_executable() -> None:
    rising = tuple(100.0 + index for index in range(20))
    paired = tuple(value * 2 for value in rising)
    discoveries = discover_pairs({"BTC": rising, "ETH": paired}, minimum_correlation=0.5)
    assert discoveries and discoveries[0].excursion_count >= 0
    pattern = recognise_statistical_pattern(
        tuple(0.001 * ((index % 3) - 1) for index in range(24)),
        tuple(0.2 for _ in range(24)),
    )
    assert pattern.flow_state == "buying"
    scalp = microstructure_scalping_signal(
        AdvancedMicrostructureState(NOW, 0.5, 1.0, -0.2, 0.3, 10.0, 100.0, 0.01)
    )
    assert 0 <= scalp.fill_probability <= 1
    targets = adaptive_portfolio_targets(
        PortfolioMetaState(
            forecasts={"trend": 0.4, "reversion": -0.2},
            regimes={"trend": "active", "reversion": "inactive"},
            performance_decay={"trend": 1.0, "reversion": 1.0},
            drift={"trend": 0.0, "reversion": 0.0},
            correlations={"trend": {"reversion": 0.2}, "reversion": {"trend": 0.2}},
            sleeve_budgets={"directional": 2.0, "relative_value": 1.0},
        )
    )
    assert targets.strategy_weights["trend"] == pytest.approx(1.0)
    assert "reversion" in targets.suppressed


def test_market_making_has_deterministic_paper_event_replay() -> None:
    states = tuple(
        QuoteState(
            f"2026-08-23T00:00:{index:02d}+00:00",
            99.0,
            101.0,
            0.01,
            0.0,
            10.0,
            0.1,
            1.0,
            1.0,
        )
        for index in range(10)
    )
    first = replay_market_making(states, maker_fee_bps=0.0)
    second = replay_market_making(states, maker_fee_bps=0.0)
    assert first == second
    assert first.passed is True
    assert MarketMakingCapabilityPolicy(live_enabled=True, minimum_event_replay_fills=10).authorise(
        mode="live", event_replay={"passed": first.passed, "fills": first.fills}
    )


def test_market_making_live_promotion_requires_capability_and_replay() -> None:
    policy = PromotionPolicy(True, True, 0.01, 30, 0.1, 0.1, 0.1)
    base = PromotionEvidence(
        strategy_artefact_hash="sha256:" + "1" * 64,
        source_commit_hash="sha256:" + "2" * 64,
        validation_accepted=True,
        protected_holdout_accepted=True,
        forward_evidence_days=60,
        forward_evidence_accepted=True,
        drawdown=0.01,
        execution_drift=0.0,
        model_drift=0.0,
        portfolio_capacity=0.1,
        requested_capital=0.01,
        risk_budget_available=0.1,
        live_approval=True,
        fresh_preflight=True,
        market_making=True,
    )
    missing_capability = decide_promotion(
        strategy_version_id="market-maker:v1",
        current_state=LifecycleState.FORWARD_PAPER,
        evidence=base,
        policy=policy,
        evaluated_at=NOW,
    )
    missing_replay = decide_promotion(
        strategy_version_id="market-maker:v1",
        current_state=LifecycleState.FORWARD_PAPER,
        evidence=PromotionEvidence(**{**base.__dict__, "market_making_live_capability": True}),
        policy=policy,
        evaluated_at=NOW,
    )
    accepted = decide_promotion(
        strategy_version_id="market-maker:v1",
        current_state=LifecycleState.FORWARD_PAPER,
        evidence=PromotionEvidence(
            **{
                **base.__dict__,
                "market_making_live_capability": True,
                "event_replay_passed": True,
                "event_replay_fills": 500,
            }
        ),
        policy=policy,
        evaluated_at=NOW,
    )
    assert missing_capability.reason_code == "market_making_live_capability_missing"
    assert missing_replay.reason_code == "market_making_event_replay_insufficient"
    assert accepted.next_state is LifecycleState.LIVE_CANARY


def test_thesis_registry_shares_budget_across_lineage_variants() -> None:
    registry = ThesisRegistry()
    thesis_id = registry.register(_thesis(budget=2))

    registry.claim_trial(thesis_id=thesis_id, candidate_id="a", lineage_id="root")
    registry.claim_trial(thesis_id=thesis_id, candidate_id="b", lineage_id="mutation:a")
    with pytest.raises(ThesisError, match="budget"):
        registry.claim_trial(thesis_id=thesis_id, candidate_id="c", lineage_id="crossover:a:b")


def test_portfolio_state_service_schedules_automatically_from_latest_sources(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'state.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="state-worker",
        node_id="node",
        role="portfolio-state-service",
        capabilities=("portfolio_state_publish",),
        observed_at=NOW,
    )
    store = SqlRiskSnapshotStore(database.engine)
    product_id = "active_income"
    values = {
        "balances": {"USDT": 1000.0},
        "positions": {},
        "open_orders": [],
        "used_margin_fraction": 0.0,
        "liquidation_buffer_fraction": 1.0,
        "unknown_exposure": {},
        "market": {
            "binance:futures:BTCUSDT:USDT": {
                "price": 100.0,
                "spread_bps": 1.0,
                "visible_depth": 10000.0,
                "volatility": 0.2,
                "funding": 0.0,
            }
        },
        "correlations": {},
        "beta": {},
        "product_drawdown_fraction": 0.0,
        "daily_pnl_fraction": 0.0,
        "global_drawdown_fraction": 0.0,
        "data_age_seconds": 0.0,
        "clock_skew_seconds": 0.0,
        "exchange_connected": True,
        "database_healthy": True,
        "execution_drift": False,
        "model_drift": False,
        "trades_today": 0,
    }
    for source in DatabasePortfolioStateWorker.REQUIRED_SOURCES:
        store.save(
            {
                "kind": source,
                "product_id": product_id,
                "observed_at": NOW,
                "values": values if source == "balances" else {},
            },
            created_at=NOW,
        )
    worker = DatabasePortfolioStateWorker(queue=queue, worker_id="state-worker", store=store)
    policy = {
        "maximum_state_age_seconds": 5.0,
        "risk_policy_ids": ["active-income"],
        "portfolio_risk_budget": 0.5,
        "maximum_symbol_fraction": 0.2,
        "maximum_abs_beta": 1.0,
        "maximum_correlation": 0.8,
        "maximum_turnover_fraction": 1.0,
        "maximum_cluster_fraction": 0.5,
        "maximum_product_drawdown_fraction": 0.1,
        "maximum_depth_participation": 0.1,
        "sleeve_budgets": {"directional": 1.0},
        "clusters": {},
        "cluster_fraction_caps": {},
    }
    assert (
        worker.schedule_from_latest(
            products={product_id: {}}, state_policies={product_id: policy}, now=NOW
        )
        == 1
    )
    assert worker.run_once(now=NOW)["reason_code"] == "canonical_portfolio_state_published"


def test_authenticated_gateway_stream_flushes_and_reconnects_independently(monkeypatch) -> None:
    class Streams:
        def __init__(self):
            self.calls = 0
            self._listen_keys = {"account": "expired"}

        async def capture(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection lost")
            raise asyncio.CancelledError

    class Sink:
        def __init__(self):
            self.flushes = 0

        def flush(self):
            self.flushes += 1

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr("src.services.market_gateway.asyncio.sleep", no_delay)
    gateway = DatabaseMarketGateway.__new__(DatabaseMarketGateway)
    gateway.user_streams = Streams()
    sink = Sink()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            gateway._capture_user_forever(
                object(),
                UserStreamAccount("account", "futures", "key", "secret"),
                sink,
            )
        )
    assert gateway.user_streams.calls == 2
    assert gateway.user_streams._listen_keys == {}
    assert sink.flushes == 1


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
    builders = ProviderContextBuilderRegistry.default()
    for source_type in (
        StrategySourceType.REGISTERED_PYTHON,
        StrategySourceType.PARAMETER_SEARCH,
        StrategySourceType.MUTATION,
        StrategySourceType.CROSSOVER,
        StrategySourceType.GENERATED_DSL,
        StrategySourceType.MACHINE_LEARNING,
        StrategySourceType.CROSS_SECTIONAL,
        StrategySourceType.RELATIVE_VALUE,
        StrategySourceType.MICROSTRUCTURE,
        StrategySourceType.ENSEMBLE,
        StrategySourceType.AGENT_GENERATED_PYTHON,
    ):
        assert callable(registry.executor_for(source_type))
        assert callable(builders.builder_for(source_type))


def test_real_registered_candidate_completes_adaptive_canonical_stages(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'research.sqlite3'}")
    database.create_schema()
    thesis = _thesis(budget=1)
    SqlThesisRegistry(database.engine).register(thesis)
    rows = [
        {
            "open": (160.0 - index) if index < 60 else (40.0 + index),
            "high": (161.0 - index) if index < 60 else (41.0 + index),
            "low": (159.0 - index) if index < 60 else (39.0 + index),
            "close": (160.0 - index) if index < 60 else (40.0 + index),
            "volume": 1000.0,
        }
        for index in range(120)
    ]
    frame = pd.DataFrame(rows)
    signals = get_registered_strategy("sma_cross")(
        fast=2, slow=4, allow_short=True
    ).generate_signals(frame)
    returns = [
        float(frame["close"].iloc[index] / frame["close"].iloc[index - 1] - 1.0)
        for index in range(1, len(frame))
    ]
    strategy_returns = [
        float(signals.iloc[index]) * returns[index] for index in range(len(returns))
    ]
    data = {
        "market_frame": rows,
        "returns": returns,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "funding_rate": 0.0,
        "features_valid": True,
        "causality_valid": True,
        "symbol_returns": {"BTCUSDT": strategy_returns},
        "active_strategy_returns": {
            "existing_strategy": [0.0001 if index % 2 else -0.0001 for index in range(len(returns))]
        },
        "negative_control_returns": {
            name: [0.0 for _ in returns]
            for name in (
                "placebo_event_times",
                "block_permutation",
                "feature_ablation",
                "parameter_neighbourhood",
                "cross_instrument",
                "predeclared_universe_holdout",
                "synthetic_autocorrelated_null",
            )
        },
    }
    snapshot_id = "sha256:" + "a" * 64
    feature_id = "sha256:" + "b" * 64
    cost_id = "sha256:" + "c" * 64
    parameter_id = "sha256:" + "d" * 64
    universe_payload = {
        "universe_id": "universe:btc",
        "observed_at": NOW,
        "policy": {"predeclared": True},
        "observations": [],
    }
    universe_content_hash = canonical_hash(universe_payload)
    universe_snapshot_id = canonical_hash(
        {"universe_id": "universe:btc", "content_hash": universe_content_hash}
    )
    with database.engine.begin() as connection:
        connection.execute(
            insert(universe).values(
                id="universe:btc",
                created_at=NOW,
                payload={"dynamic": False, "fixed_maximum": 1},
            )
        )
        connection.execute(
            insert(universe_snapshot).values(
                id=universe_snapshot_id,
                universe_id="universe:btc",
                observed_at=NOW,
                content_hash=universe_content_hash,
                payload=universe_payload,
            )
        )
        connection.execute(
            insert(dataset_snapshot).values(
                id=snapshot_id,
                created_at=NOW,
                payload={
                    "snapshot_id": snapshot_id,
                    "content_hash": canonical_hash(data),
                    "interval": {
                        "start": "2026-01-01T00:00:00+00:00",
                        "end": "2026-06-01T00:00:00+00:00",
                    },
                    "universe_snapshot_id": universe_snapshot_id,
                    "availability_timestamp": NOW,
                    "feature_manifest_id": feature_id,
                    "cost_model_id": cost_id,
                    "parameter_set_id": parameter_id,
                    "product_id": "active_income",
                    "instrument_scope": ["BTCUSDT"],
                    "engine_version": "research-engine/v1",
                    "payload": data,
                },
            )
        )
    candidate = provider_candidate(
        identity="sma_cross",
        version="v1",
        family="time_series",
        product="active_income",
        thesis_id=thesis.thesis_id,
        lineage_id=canonical_hash({"lineage": "real-stage-integration"}),
        provider="registered_strategy_catalogue",
        source_type=StrategySourceType.REGISTERED_PYTHON,
        source_payload={
            "registered_strategy": "sma_cross",
            "parameters": {"fast": 2, "slow": 4, "allow_short": True},
        },
        dataset_snapshot_hashes=(snapshot_id,),
        submitted_at=NOW,
        universe={"symbols": ["BTCUSDT"]},
        data_requirements={"bars": "1h"},
        feature_graph={"required_nodes": ["sma"]},
        position_model={"kind": "volatility_scaled"},
        execution_preferences={"policy": "market"},
        risk_policy={"id": "active-income"},
        validation_policy={"id": "canonical"},
    )
    store = SqlResearchStore(database.engine)
    ResearchCoordinator(store).submit(candidate)
    resolved = CanonicalDatasetResolver(
        SqlCanonicalDatasetRepository(database.engine)
    ).resolve_context(
        snapshot_ids=(snapshot_id,),
        feature_manifest_id=feature_id,
        cost_model_id=cost_id,
        parameter_set_id=parameter_id,
    )
    context = ProviderContextBuilderRegistry.default().build(candidate, resolved)
    identity_hash = canonical_hash({"test_artefact": candidate.definition.definition_hash})
    context.update(
        {
            "artefact_hash": identity_hash,
            "runtime_artefact_hash": identity_hash,
            "engine_hash": "provider-executors/v3",
            "runtime_engine_hash": "provider-executors/v3",
            "runtime_cost_model_id": cost_id,
            "production_execution_mode": "production",
            "runtime_execution_mode": "production",
            "drift_measurements": {
                "execution": 0.0,
                "model": 0.0,
                "maximum_execution": 0.1,
                "maximum_model": 0.1,
            },
        }
    )
    executors = ProviderExecutorRegistry.default()
    policy = EvidencePolicy(minimum_deflated_sharpe=0.0)

    evaluator = CanonicalResearchEvaluator(
        store,
        executors=executors,
        provider_context=context,
        evidence_policy=policy,
    )
    results = []
    for stage in ("screening", "development", "robustness"):
        result = evaluator.evaluate(
            EvaluationRequest(
                candidate_id=candidate.candidate_id,
                evaluation_policy_id="canonical",
                dataset_snapshot_ids=(snapshot_id,),
                requested_stage=stage,
                evaluated_at=NOW,
                code_hash=candidate.definition.source_hash,
                feature_manifest_id=feature_id,
                cost_model_id=cost_id,
                parameter_set_id=parameter_id,
                evaluator_version="research-engine/v1",
                producer_identity="test:canonical-research",
                content_hash=canonical_hash({"stage": stage}),
            )
        )
        assert result.accepted, f"{result.reason_code}: {result.evidence}"
        results.append(result)
    assert [result.stage for result in results] == ["screening", "development", "robustness"]
    assert all(result.accepted for result in results)


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


@pytest.mark.parametrize("model_type", sorted(SAFE_MODEL_TYPES))
def test_every_safe_frozen_model_type_has_data_only_inference(tmp_path, model_type) -> None:
    payload = {
        "model_type": model_type,
        "feature_names": ["momentum", "funding"],
        "weights": [0.5, -0.25],
        "intercept": 0.01,
    }
    if model_type.startswith("lightgbm"):
        payload.pop("weights")
        payload.pop("intercept")
        payload.update(
            {
                "base_score": 0.0,
                "learning_rate": 0.1,
                "trees": [
                    {
                        "feature": "momentum",
                        "threshold": 0.0,
                        "left": {"value": -1.0},
                        "right": {"value": 1.0},
                    }
                ],
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / f"{model_type}.json"
    path.write_bytes(encoded)
    model = FrozenSafeModel.load(
        path,
        expected_artefact_hash="sha256:" + hashlib.sha256(encoded).hexdigest(),
        expected_feature_manifest_hash=canonical_hash({"feature_names": ("momentum", "funding")}),
    )
    forecast = model.evaluate({"momentum": 0.2, "funding": 0.01})
    assert -1 <= forecast.score <= 1
