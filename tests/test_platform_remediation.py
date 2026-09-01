from __future__ import annotations

import datetime as dt
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import select, update

from src.accounting.fees import FeeConversionError, convert_fee, instrument_asset
from src.data.database import (
    PlatformDatabase,
    cost_model_manifest,
    dataset_snapshot,
    experiment,
    feature_manifest,
    job,
)
from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.domain._codec import canonical_hash
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.domain.instruments import Instrument, MarketType
from src.domain.orders import OrderSide
from src.domain.portfolios import TargetPosition
from src.execution.order_planner import plan_orders
from src.portfolio.optimiser import PortfolioConstraints, optimise_targets
from src.products.btc_accumulation import (
    BtcAllocationPolicy,
    btc_step_aside_metadata,
    target_btc_allocation,
)
from src.research.accounting import (
    BtcAccumulationAccounting,
    BtcResearchAccounting,
    FuturesIncomeAccounting,
    FuturesResearchAccounting,
    ProductAccountingError,
)
from src.research.artefacts import StrategyArtefact
from src.research.canonical import CanonicalEvidenceError, _prepare_summary_payload
from src.research.catalogue import registered_strategy_candidates
from src.research.coordinator import ResearchCoordinator
from src.research.dataset_service import DatabaseDatasetBundleService, _bounded_instrument_rows
from src.research.datasets import (
    CORE_RESEARCH_BUNDLE_ROLES,
    CanonicalResearchDatasetBuilder,
    DatasetLifecycleState,
    DatasetResolutionError,
    SqlDatasetBundleRepository,
)
from src.research.evaluation import (
    EvaluationDecision,
    EvidencePolicy,
    EvidenceProfile,
    EvidenceStatus,
)
from src.research.evidence import (
    cross_symbol_stability_passes,
    drawdown_passes,
    holdout_degradation_passes,
    monte_carlo_passes,
    parameter_stability_passes,
    sample_evidence_passes,
)
from src.research.executors import (
    _cross_symbol_stability,
    _negative_control_evidence,
    _pbo_measurements,
    _portfolio_overlap,
    _product_accounting,
)
from src.research.generation import CAMPAIGNS, build_hypothesis
from src.research.objectives import objective_passes
from src.research.returns import PositionReturnLedger
from src.research.store import SqlResearchStore
from src.research.theses import SqlThesisRegistry
from src.services.artefact_dispatcher import ArtefactDispatcher
from src.services.forward_metrics import ForwardEvidenceCollector
from src.services.order_execution import _validate_btc_spot_orders
from src.services.readiness import _dataset_readiness, _ready_dataset_roles
from src.services.research_jobs import DatabaseResearchJobHandlers
from src.services.scheduler import ClaimedJob, DatabaseJobQueue, PlatformScheduler
from src.strategies.behaviour import RegisteredStrategyBehaviour, TypedRuleBehaviour

NOW = "2026-08-30T10:00:00+00:00"


def test_forward_summary_rejects_fractional_independent_decisions() -> None:
    identity = "sha256:" + "a" * 64

    with pytest.raises(CanonicalEvidenceError, match="independent decisions must be an integer"):
        _prepare_summary_payload(
            strategy_version_id="strategy:v1",
            product_id="active_income",
            observed_at=NOW,
            artefact_hash=identity,
            evidence={
                "independent_decisions": 1.5,
                "observation_ids": [identity],
            },
        )


def test_job_retries_end_in_a_durable_dead_letter_state(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'queue.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="worker",
        node_id="node",
        role="research-worker",
        capabilities=("research",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="bounded",
        name="research",
        payload={"candidate": "bounded"},
        available_at=NOW,
        max_attempts=2,
    )

    first = queue.claim(worker_id="worker", now=NOW, lease_seconds=10)
    assert first is not None
    queue.fail(
        first,
        completed_at="2026-08-30T10:00:01+00:00",
        error="first failure",
        retry_at="2026-08-30T10:00:02+00:00",
    )
    second = queue.claim(worker_id="worker", now="2026-08-30T10:00:02+00:00", lease_seconds=10)
    assert second is not None and second.attempt == 2
    queue.fail(
        second,
        completed_at="2026-08-30T10:00:03+00:00",
        error="terminal failure",
        retry_at="2026-08-30T10:00:04+00:00",
    )

    with database.engine.connect() as connection:
        row = connection.execute(
            select(job.c.state, job.c.attempts, job.c.max_attempts, job.c.terminal_reason).where(
                job.c.id == "bounded"
            )
        ).one()
    assert row.state == "dead_letter"
    assert row.attempts == row.max_attempts == 2
    assert row.terminal_reason == "terminal failure"
    assert (
        queue.claim(worker_id="worker", now="2026-08-30T10:00:10+00:00", lease_seconds=10) is None
    )


def test_dead_lettered_stage_is_durably_blocked_from_recurring_resubmission(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'dead-letter-stage.sqlite3'}")
    database.create_schema()
    identity = "sha256:" + "f" * 64
    intervals = {
        role: {
            "start": f"2026-08-{20 + index:02d}T00:00:00+00:00",
            "end": f"2026-08-{21 + index:02d}T00:00:00+00:00",
        }
        for index, role in enumerate(CORE_RESEARCH_BUNDLE_ROLES)
    }
    bundle = CanonicalResearchDatasetBuilder(database.engine).build(
        "active_income",
        intervals=intervals,
        payload_by_role={
            role: {"market_frame": [{"close": 100.0}], "rows": [1]}
            for role in CORE_RESEARCH_BUNDLE_ROLES
        },
        universe_snapshot_id=identity,
        feature_manifest_id=identity,
        cost_model_id=identity,
        parameter_set_id=identity,
        instrument_scope=("binance:futures:BTCUSDT:USDT",),
        availability_timestamp=NOW,
        created_at=NOW,
    )
    hypothesis = build_hypothesis(
        CAMPAIGNS[2],
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=tuple(bundle.stage_snapshot_ids.values()),
        submitted_at=NOW,
    )
    SqlThesisRegistry(database.engine).register(hypothesis.thesis)
    ResearchCoordinator(SqlResearchStore(database.engine)).submit(hypothesis.candidate)
    with database.engine.begin() as connection:
        connection.execute(
            update(experiment)
            .where(experiment.c.id == hypothesis.candidate.candidate_id)
            .values(state="screening")
        )
    scheduler = PlatformScheduler(
        engine=database.engine,
        products={"active_income": {"universe_id": "unused"}},
        node_id="linux-optiplex",
    )
    snapshot_id = bundle.stage_snapshot_ids["development"]
    with database.engine.connect() as connection:
        snapshot_payload = connection.execute(
            select(dataset_snapshot.c.payload).where(dataset_snapshot.c.id == snapshot_id)
        ).scalar_one()
    request = scheduler._research_request(
        candidate_id=hypothesis.candidate.candidate_id,
        snapshot_id=snapshot_id,
        snapshot_payload=snapshot_payload,
        requested_stage="development",
        evaluated_at=NOW,
    )
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="dead-letter-worker",
        node_id="node",
        role="research-worker",
        capabilities=("evaluate_candidate",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="dead-letter-evaluation",
        name="evaluate_candidate",
        payload=request,
        available_at=NOW,
        max_attempts=1,
    )
    claimed = queue.claim(
        worker_id="dead-letter-worker", now=NOW, lease_seconds=10, names=("evaluate_candidate",)
    )
    assert claimed is not None
    queue.fail(
        claimed,
        completed_at="2026-08-30T10:00:01+00:00",
        error="poison evaluation",
        retry_at="2026-08-30T10:00:02+00:00",
    )

    assert scheduler._candidate_evaluation_jobs("active_income", NOW, NOW) == ()
    with database.engine.connect() as connection:
        state, metadata = connection.execute(
            select(experiment.c.state, experiment.c.metadata).where(
                experiment.c.id == hypothesis.candidate.candidate_id
            )
        ).one()
    assert state == "blocked_dead_letter:development"
    assert metadata["blocked_reason"]["reason_code"] == "evaluation_job_dead_letter"


def test_evidence_policy_preserves_applicability_and_thesis_scoped_controls() -> None:
    policy = EvidencePolicy()
    input_hash = "sha256:" + "a" * 64
    evidence = {
        "parameter_stability": {"status": "not_applicable", "passed": True},
        "cross_symbol_stability": {"status": "not_applicable", "passed": True},
        "portfolio_overlap": {"status": "not_applicable", "passed": True},
        "negative_control_results": {
            "placebo_event_times": {
                "passed": True,
                "observations": 10,
                "input_hash": input_hash,
            }
        },
    }
    statuses = policy.statuses(
        "development",
        evidence,
        ("placebo_event_times",),
    )
    assert statuses["parameter_stability"] is EvidenceStatus.NOT_APPLICABLE
    assert statuses["cross_symbol_stability"] is EvidenceStatus.NOT_APPLICABLE
    assert statuses["portfolio_overlap"] is EvidenceStatus.NOT_APPLICABLE
    robustness_statuses = policy.statuses(
        "robustness",
        evidence,
        ("placebo_event_times",),
    )
    assert robustness_statuses["negative_control_results"] is EvidenceStatus.PASS


def test_evidence_policy_returns_a_structured_next_stage_decision() -> None:
    policy = EvidencePolicy()
    identity = "sha256:" + "a" * 64
    parity = {
        "schema": "typed_rule_parity/v1",
        "behaviour_hash": identity,
        "input_hash": identity,
        "signals": [1],
    }
    evidence = {
        "evidence_policy_hash": policy.policy_hash,
        "compiled": True,
        "features_valid": True,
        "causality_valid": True,
        "data_integrity": {
            "passed": True,
            "dataset_snapshot_ids": [identity],
            "input_hash": identity,
        },
        "semantic_parity": {
            "passed": True,
            "behaviour_hash": identity,
            "parity_receipt": {**parity, "receipt_hash": canonical_hash(parity)},
        },
        "realistic_costs": {
            "passed": True,
            "fee_bps": 1.0,
            "slippage_bps": 1.0,
            "funding_rate": 0.0,
        },
        "family_evidence": {"passed": True, "family": "time_series"},
        "signal_frequency": 0.2,
        "turnover": 0.1,
    }

    decision = policy.decide("screening", evidence, ())

    assert isinstance(decision, EvaluationDecision)
    assert decision.accepted is True
    assert decision.fatal_failures == ()
    assert decision.next_stage == "development"
    assert 0.0 < decision.evidence_score <= 1.0
    assert decision.to_payload()["next_stage"] == "development"


def test_forward_evidence_is_deferred_until_sample_is_sufficient() -> None:
    identity = "sha256:" + "c" * 64
    policy = EvidencePolicy(
        profiles=(
            EvidenceProfile(
                stage="forward",
                family="swing",
                minimum_calendar_days=60.0,
                minimum_closed_trades=20,
                minimum_effective_episodes=20,
            ),
        )
    )
    parity = {
        "schema": "typed_rule_parity/v1",
        "behaviour_hash": identity,
        "input_hash": identity,
        "signals": [1],
    }
    parity["receipt_hash"] = canonical_hash(parity)
    evidence = {
        "evidence_policy_hash": policy.policy_hash,
        "data_integrity": {
            "passed": True,
            "dataset_snapshot_ids": [identity],
            "input_hash": identity,
        },
        "semantic_parity": {
            "passed": True,
            "behaviour_hash": identity,
            "parity_receipt": parity,
        },
        "realistic_costs": {
            "passed": True,
            "fee_bps": 1.0,
            "slippage_bps": 1.0,
            "funding_rate": 0.0,
        },
        "family_evidence": {"passed": True, "family": "swing"},
        "production_equivalent": {"passed": True, "mode": "production"},
        "exact_strategy_identity": {"passed": True, "expected": identity},
        "exact_artefact_hash": {"passed": True, "expected": identity},
        "exact_engine_hash": {"passed": True, "expected": identity},
        "exact_cost_model": {"passed": True, "expected": identity},
        "drift_checks": {"passed": True, "execution": 0.0},
        "duration": 1.0,
        "evidence_units": 1.0,
        "sample_evidence": {
            "passed": False,
            "observations": 1,
            "closed_trades": 0,
            "effective_independent_episodes": 0,
            "trading_days": 0,
        },
        "forward_duration": {
            "passed": False,
            "calendar_days": 1.0,
            "trading_days": 0,
            "cycles": 0,
        },
    }

    statuses = policy.statuses("forward", evidence, (), family="swing")
    decision = policy.decide("forward", evidence, (), family="swing")

    assert statuses["sample_evidence"] is EvidenceStatus.UNAVAILABLE
    assert statuses["forward_duration"] is EvidenceStatus.UNAVAILABLE
    assert decision.accepted is False
    assert decision.deferred is True
    assert decision.fatal_failures == ("sample_evidence", "forward_duration")
    assert decision.to_payload()["deferred"] is True


def test_robustness_uses_one_primary_confidence_gate() -> None:
    policy = EvidencePolicy(
        minimum_deflated_sharpe=0.95,
        minimum_bootstrap_observations=30,
        confidence_method="bootstrap",
    )
    identity = "sha256:" + "b" * 64
    parity = {
        "schema": "typed_rule_parity/v1",
        "behaviour_hash": identity,
        "input_hash": identity,
        "signals": [1],
    }
    evidence = {
        "evidence_policy_hash": policy.policy_hash,
        "data_integrity": {
            "passed": True,
            "dataset_snapshot_ids": [identity],
            "input_hash": identity,
        },
        "semantic_parity": {
            "passed": True,
            "behaviour_hash": identity,
            "parity_receipt": {**parity, "receipt_hash": canonical_hash(parity)},
        },
        "realistic_costs": {
            "passed": True,
            "fee_bps": 1.0,
            "slippage_bps": 1.0,
            "funding_rate": 0.0,
        },
        "family_evidence": {"passed": True, "family": "time_series"},
        "walk_forward": {"passed": True, "window_count": 5, "pass_fraction": 0.8},
        "purged": True,
        "embargo": 1,
        "cost_stress": {"passed": True, "return": 0.1},
        "delay_stress": {"passed": True, "return": 0.1},
        "adverse_fill_stress": {"passed": True, "return": 0.1},
        "missing_data_stress": {"passed": True, "return": 0.1},
        "funding_stress": {"passed": True, "return": 0.1},
        "monte_carlo_trade_order": {
            "passed": True,
            "iterations": 250,
            "maximum_drawdown": 0.1,
            "tail_loss": 0.01,
        },
        "bootstrap_confidence": {
            "passed": True,
            "observations": 30,
            "lower_bound": 0.1,
        },
        "probability_backtest_overfitting": 0.1,
        "deflated_sharpe": 0.0,
        "statistical_procedures": {
            "bootstrap": policy.bootstrap_method,
            "multiple_testing": policy.multiple_testing_method,
            "pbo": policy.pbo_method,
        },
        "drawdown_stability": {
            "passed": True,
            "maximum_drawdown": 0.1,
            "tail_loss": 0.01,
        },
        "null_results": {"passed": True, "tests": 1},
        "negative_control_results": {},
    }

    decision = policy.decide("robustness", evidence, ())

    assert decision.accepted is True
    assert "deflated_sharpe" not in decision.fatal_failures
    assert "deflated_sharpe:fail" in decision.diagnostics


def test_single_symbol_and_empty_portfolio_overlap_are_not_applicable() -> None:
    cross_symbol = _cross_symbol_stability(
        {"instrument_scope": ("BTCUSDT",)},
        [0.001, -0.0005],
    )
    overlap = _portfolio_overlap({}, [0.001, -0.0005])

    assert cross_symbol["status"] == "not_applicable"
    assert cross_symbol["passed"] is True
    assert overlap["status"] == "not_applicable"
    assert overlap["passed"] is True


def test_missing_declared_negative_control_is_unavailable_not_a_fake_null() -> None:
    evidence = _negative_control_evidence(
        signals=[1.0, 1.0],
        returns=[0.01, 0.01],
        candidate_return=0.02,
        controls=("block_permutation",),
    )

    assert evidence["block_permutation"]["status"] == "unavailable"
    assert evidence["block_permutation"]["passed"] is False


def test_negative_controls_are_deterministic_when_immutable_market_inputs_are_available() -> None:
    kwargs = {
        "signals": [1.0, -1.0, 1.0, 0.0, -1.0, 1.0],
        "returns": [0.01, -0.02, 0.03, 0.0, -0.01, 0.02],
        "candidate_return": 0.01,
        "controls": ("block_permutation", "feature_ablation", "parameter_neighbourhood"),
        "seed_material": {"candidate": "candidate-1", "dataset": "dataset-1"},
    }
    first = _negative_control_evidence(**kwargs)
    second = _negative_control_evidence(**kwargs)

    assert first == second
    assert first["feature_ablation"]["control_return"] == 0.0
    assert first["block_permutation"]["source"] == "derived_immutable_inputs"
    assert first["parameter_neighbourhood"]["method"] == "parameter_neighbourhood_shift_v1"


def test_active_income_registered_time_series_candidates_are_symbol_isolated() -> None:
    candidates = registered_strategy_candidates(
        product="active_income",
        dataset_snapshot_hashes=("sha256:" + "1" * 64,),
        instrument_universe=("BTCUSDT", "ETHUSDT"),
    )

    sma_candidates = [
        candidate for candidate in candidates if candidate.definition.identity == "sma_cross"
    ]
    assert len(sma_candidates) == 2
    assert {tuple(candidate.definition.universe["symbols"]) for candidate in sma_candidates} == {
        ("BTCUSDT",),
        ("ETHUSDT",),
    }
    assert all(
        candidate.definition.metadata["research_scope"] == "symbol" for candidate in sma_candidates
    )


def test_strategy_universe_rejects_legacy_and_wrong_btc_scope() -> None:
    definition = registered_strategy_candidates(
        product="btc_accumulation",
        dataset_snapshot_hashes=("sha256:" + "1" * 64,),
    )[0].definition

    with pytest.raises(ValueError, match="unsupported fields"):
        replace(definition, universe={"symbols": ["BTCUSDT"], "predeclared": True})
    with pytest.raises(ValueError, match="BTCUSDT spot only"):
        replace(definition, universe={"symbols": ["ETHUSDT"]})


def test_typed_rule_behaviour_has_one_research_and_production_signal_contract() -> None:
    behaviour = TypedRuleBehaviour(
        {"feature": "bar_return", "operator": "gt", "threshold": 0.0, "direction": "short"}
    )
    rows = [{"bar_return": -0.01}, {"bar_return": 0.01}, {"bar_return": 0.0}]

    assert behaviour.generate_signals(rows) == tuple(behaviour.signal(row) for row in rows)
    assert behaviour.parity_receipt(rows)["behaviour_hash"] == behaviour.behaviour_hash


def test_typed_rule_behaviour_supports_signed_long_and_short_outputs() -> None:
    behaviour = TypedRuleBehaviour(
        {
            "feature": "trend",
            "operator": "gt",
            "threshold": 0.25,
            "direction": "signed",
        }
    )

    assert behaviour.generate_signals([{"trend": 0.5}, {"trend": -0.5}, {"trend": 0.1}]) == (
        1,
        -1,
        0,
    )


def test_typed_rule_behaviour_supports_multi_condition_entries_and_exits() -> None:
    behaviour = TypedRuleBehaviour(
        {
            "conditions": [
                {"feature": "trend", "operator": "gt", "threshold": 0.1},
                {"feature": "volatility", "operator": "lt", "threshold": 0.5},
            ],
            "condition_mode": "all",
            "exit_conditions": [{"feature": "volatility", "operator": "ge", "threshold": 0.9}],
            "direction": "long",
        }
    )

    assert behaviour.generate_signals(
        [
            {"trend": 0.2, "volatility": 0.2},
            {"trend": 0.2, "volatility": 0.9},
            {"trend": 0.0, "volatility": 0.2},
        ]
    ) == (1, 0, 0)


def test_typed_rule_behaviour_validates_directional_groups() -> None:
    behaviour = TypedRuleBehaviour(
        {
            "positive_conditions": [{"feature": "momentum", "operator": "gt", "threshold": 0.1}],
            "negative_conditions": [{"feature": "momentum", "operator": "lt", "threshold": -0.1}],
            "condition_mode": "any",
            "direction": "signed",
        }
    )

    assert behaviour.generate_signals(
        [{"momentum": 0.2}, {"momentum": -0.2}, {"momentum": 0.0}]
    ) == (1, -1, 0)
    with pytest.raises(ValueError, match="no entry conditions"):
        TypedRuleBehaviour(
            {"exit_conditions": [{"feature": "volatility", "operator": "gt", "threshold": 0.9}]}
        )


def _evidence_hashes(index: int) -> dict[str, str]:
    return {
        "run_id": "sha256:" + str(index) * 64,
        "input_hash": "sha256:" + str(index + 1) * 64,
    }


def test_cross_symbol_evidence_uses_panel_statistics_not_all_positive_symbols() -> None:
    per_symbol = {
        symbol: {
            "return": value,
            "observations": 20,
            "passed": value >= 0,
            **_evidence_hashes(index),
        }
        for index, (symbol, value) in enumerate(
            (("BTCUSDT", 0.10), ("ETHUSDT", 0.08), ("SOLUSDT", 0.06), ("XRPUSDT", -0.04))
        )
    }
    evidence = {
        "passed": False,
        "symbols": 4,
        "per_symbol": per_symbol,
        "median_return": 0.07,
        "pooled_return": 0.05,
        "positive_symbol_fraction": 0.75,
    }

    assert cross_symbol_stability_passes(evidence, EvidenceProfile()) is True


def test_parameter_surface_rejects_a_cliff_but_accepts_smooth_degradation() -> None:
    def surface(values: tuple[float, ...]) -> dict[str, object]:
        return {
            "passed": True,
            "base_return": 0.10,
            "neighbours_tested": len(values),
            "results": [
                {
                    "return": value,
                    "observations": 20,
                    "passed": value >= 0.05,
                    **_evidence_hashes(index),
                }
                for index, value in enumerate(values)
            ],
        }

    assert parameter_stability_passes(surface((0.09, 0.08, 0.07)), EvidenceProfile()) is True
    assert parameter_stability_passes(surface((0.09, 0.08, -0.50)), EvidenceProfile()) is False


def test_profile_enforces_effective_sample_and_quantitative_tail_limits() -> None:
    profile = EvidenceProfile(
        minimum_closed_trades=20,
        minimum_effective_episodes=8,
        maximum_drawdown=0.20,
        maximum_tail_loss=0.05,
    )
    sample = {
        "passed": True,
        "observations": 40,
        "closed_trades": 20,
        "effective_independent_episodes": 8,
    }
    assert sample_evidence_passes(sample, profile) is True
    assert sample_evidence_passes({**sample, "closed_trades": 19}, profile) is False
    assert (
        drawdown_passes({"passed": True, "maximum_drawdown": 0.12, "tail_loss": 0.04}, profile)
        is True
    )
    assert (
        monte_carlo_passes(
            {"passed": True, "iterations": 250, "maximum_drawdown": 0.21, "tail_loss": 0.01},
            profile,
        )
        is False
    )


def test_evidence_profiles_select_the_most_specific_product_horizon_policy() -> None:
    policy = EvidencePolicy(
        profiles=(
            EvidenceProfile(stage="forward", product_id="*", family="swing"),
            EvidenceProfile(
                stage="forward",
                product_id="active_income",
                family="swing",
                minimum_closed_trades=20,
            ),
        )
    )

    selected = policy.profile_for(
        "forward", product_id="active_income", family="swing", horizon="1d"
    )

    assert selected.minimum_closed_trades == 20


def test_protected_holdout_allows_configured_moderate_degradation() -> None:
    profile = EvidenceProfile(allowed_holdout_degradation=0.50)
    assert (
        holdout_degradation_passes(
            {"objective_excess_fraction": 0.10}, {"objective_excess_fraction": 0.07}, profile
        )
        is True
    )
    assert (
        holdout_degradation_passes(
            {"objective_excess_fraction": 0.10}, {"objective_excess_fraction": 0.01}, profile
        )
        is False
    )


def test_registered_strategy_behaviour_is_shared_by_research_and_dispatch() -> None:
    candidates = registered_strategy_candidates(
        product="active_income",
        dataset_snapshot_hashes=("sha256:" + "b" * 64,),
        instrument_universe=("BTCUSDT",),
    )
    candidate = next(item for item in candidates if item.definition.identity == "sma_cross")
    frame = [
        {
            "open": float(index),
            "high": float(index) + 1.0,
            "low": float(index) - 1.0,
            "close": float(index) + 0.5,
            "volume": 100.0,
        }
        for index in range(1, 40)
    ]
    artefact = StrategyArtefact(
        definition=candidate.definition,
        dependency_hash="sha256:" + "c" * 64,
        dataset_snapshot_hashes=candidate.dataset_snapshot_hashes,
        feature_set_version="features-v1",
        cost_model_version="costs-v1",
        validation_evidence={"accepted": True},
        holdout_claim={"accepted": True},
        promotion_policy={"paper": True},
        position_limits={"maximum_position": 0.2, "target_volatility": 0.2},
        risk_limits={"policy": "active-income"},
        model_hashes=(),
        supported_products=("active_income",),
        supported_instruments=("BTCUSDT",),
        created_at=NOW,
        product_id="active_income",
        portfolio_id="portfolio-active-income",
        account_id="account-usdt",
        promotion_policy_id="promotion-v1",
        engine_version="strategy-v1",
    )
    expected_signal = RegisteredStrategyBehaviour.from_definition(
        candidate.definition
    ).latest_signal(frame)
    dispatched = ArtefactDispatcher.default().evaluate({"market_frame": frame}, artefact.to_dict())

    assert (
        dispatched["direction"]
        == {
            -1: "short",
            0: "flat",
            1: "long",
        }[expected_signal]
    )
    assert dispatched["behaviour_hash"] == artefact.behaviour_hash
    assert dispatched["execution_receipt"]["deployment_hash"] == artefact.artefact_hash
    assert dispatched["execution_receipt"]["behaviour_hash"] == artefact.behaviour_hash


def test_btc_accounting_measures_sell_rebuy_in_btc_and_counts_costs() -> None:
    report = BtcAccumulationAccounting().evaluate(
        initial_btc=1.0,
        initial_price=100.0,
        trade_events=(
            {"timestamp": NOW, "side": "sell", "quantity": 0.5, "price": 100.0},
            {
                "timestamp": "2026-08-30T10:01:00+00:00",
                "side": "buy",
                "quantity": 0.5,
                "price": 80.0,
                "fee": 0.001,
                "fee_asset": "BTC",
            },
        ),
        marks=(
            {"timestamp": NOW, "price": 100.0, "regime": "trend"},
            {"timestamp": "2026-08-30T10:01:00+00:00", "price": 80.0, "regime": "risk_off"},
        ),
    )

    assert report.objective_unit == "BTC"
    assert report.final_btc_nav == pytest.approx(1.124)
    assert report.excess_btc == pytest.approx(0.124)
    assert report.fees_btc == pytest.approx(0.001)
    assert report.cycles == 1
    assert report.round_trip_btc_gain == pytest.approx(0.124)
    assert report.worst_reentry_slippage == pytest.approx(-0.2)


def test_btc_accounting_converts_bnb_fee_and_enforces_core_reserve() -> None:
    report = BtcAccumulationAccounting().evaluate(
        initial_stablecoin=1_000.0,
        initial_price=100.0,
        trade_events=(
            {
                "timestamp": NOW,
                "side": "buy",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.01,
                "fee_asset": "BNB",
                "fee_conversion_price": 200.0,
            },
        ),
        marks=({"timestamp": NOW, "price": 100.0},),
    )

    assert report.fees_btc == pytest.approx(0.02)
    assert report.event_receipts[0]["fee_quote"] == pytest.approx(2.0)
    assert report.event_receipts[0]["fee_conversion"] == "explicit_fee_conversion_price"
    with pytest.raises(ProductAccountingError, match="core BTC reserve"):
        BtcAccumulationAccounting().evaluate(
            initial_btc=1.0,
            initial_price=100.0,
            reserve_fraction=0.8,
            trade_events=({"timestamp": NOW, "side": "sell", "quantity": 0.5, "price": 100.0},),
        )
    with pytest.raises(ProductAccountingError, match="deterministic BTC or quote conversion"):
        BtcAccumulationAccounting().evaluate(
            initial_btc=1.0,
            initial_price=100.0,
            trade_events=(
                {
                    "timestamp": NOW,
                    "side": "sell",
                    "quantity": 0.1,
                    "price": 100.0,
                    "fee": 0.01,
                    "fee_asset": "BNB",
                },
            ),
        )


def test_runtime_fee_conversion_is_explicit_and_accounting_asset_bound() -> None:
    quote_to_btc = convert_fee(
        amount=1.0,
        fee_asset="USDT",
        accounting_asset="BTC",
        trade_price=100.0,
        base_asset="BTC",
        quote_asset="USDT",
    )
    assert quote_to_btc.accounting_amount == pytest.approx(0.01)
    assert quote_to_btc.source == "quote_to_base"

    bnb_to_usdt = convert_fee(
        amount=0.01,
        fee_asset="BNB",
        accounting_asset="USDT",
        trade_price=100.0,
        metadata={"fee_conversion_rate": 200.0},
    )
    assert bnb_to_usdt.accounting_amount == pytest.approx(2.0)
    assert bnb_to_usdt.source == "explicit_rate"
    with pytest.raises(FeeConversionError, match="deterministic conversion"):
        convert_fee(
            amount=0.01,
            fee_asset="BNB",
            accounting_asset="USDT",
            trade_price=100.0,
        )


def test_instrument_asset_parsing_handles_canonical_futures_symbols() -> None:
    instrument = "binance:futures:BTCUSDT:USDT"

    assert instrument_asset(instrument, "base") == "BTC"
    assert instrument_asset(instrument, "quote") == "USDT"
    assert instrument_asset("BTC/USDT:USDT", "base") == "BTC"
    assert instrument_asset("BTC/USDT:USDT", "quote") == "USDT"


def test_btc_accounting_allows_initial_stablecoin_deployment_with_reserve() -> None:
    report = BtcAccumulationAccounting().evaluate(
        initial_stablecoin=100.0,
        initial_price=100.0,
        reserve_fraction=0.8,
        trade_events=({"timestamp": NOW, "side": "buy", "quantity": 1.0, "price": 100.0},),
    )

    assert report.final_btc_nav == pytest.approx(1.0)


def test_btc_round_trip_includes_sell_and_buy_fees() -> None:
    report = BtcAccumulationAccounting().evaluate(
        initial_btc=1.0,
        initial_price=100.0,
        trade_events=(
            {
                "timestamp": NOW,
                "side": "sell",
                "quantity": 0.5,
                "price": 100.0,
                "fee": 1.0,
                "fee_asset": "USDT",
            },
            {
                "timestamp": "2026-08-30T10:01:00+00:00",
                "side": "buy",
                "quantity": 0.5,
                "price": 80.0,
                "fee": 0.001,
                "fee_asset": "BTC",
            },
        ),
        marks=(
            {"timestamp": NOW, "price": 100.0},
            {"timestamp": "2026-08-30T10:01:00+00:00", "price": 80.0},
        ),
    )

    assert report.round_trip_btc_gain == pytest.approx(0.1115)


def test_research_accounting_types_are_distinct_product_ledgers() -> None:
    assert issubclass(BtcResearchAccounting, BtcAccumulationAccounting)
    assert issubclass(FuturesResearchAccounting, FuturesIncomeAccounting)
    assert BtcResearchAccounting is not BtcAccumulationAccounting
    assert FuturesResearchAccounting is not FuturesIncomeAccounting


def test_btc_accounting_tracks_external_flows_and_tactical_limit() -> None:
    report = BtcAccumulationAccounting().evaluate(
        initial_btc=1.0,
        initial_price=100.0,
        external_events=(
            {
                "type": "deposit",
                "timestamp": NOW,
                "asset": "USDT",
                "amount": 100.0,
            },
            {
                "type": "withdrawal",
                "timestamp": "2026-08-30T10:01:00+00:00",
                "asset": "USDT",
                "amount": 100.0,
            },
        ),
        marks=(
            {"timestamp": NOW, "price": 100.0},
            {"timestamp": "2026-08-30T10:01:00+00:00", "price": 100.0},
        ),
        max_tactical_fraction=0.5,
    )

    assert report.excess_btc == pytest.approx(0.0)
    assert report.external_deposits_btc == pytest.approx(1.0)
    assert report.external_withdrawals_btc == pytest.approx(1.0)
    with pytest.raises(ProductAccountingError, match="tactical allocation"):
        BtcAccumulationAccounting().evaluate(
            initial_btc=1.0,
            initial_price=100.0,
            trade_events=({"timestamp": NOW, "side": "sell", "quantity": 0.3, "price": 100.0},),
            marks=({"timestamp": NOW, "price": 100.0},),
            max_tactical_fraction=0.2,
        )


def test_executor_derives_btc_objective_from_canonical_bar_frame() -> None:
    frame = [
        {"timestamp": NOW, "close": 100.0},
        {"timestamp": "2026-08-30T10:01:00+00:00", "close": 80.0},
        {"timestamp": "2026-08-30T10:02:00+00:00", "close": 80.0},
    ]

    accounting = _product_accounting(
        {
            "product_id": "btc_accumulation",
            "market_frame": frame,
            "signals": [-1.0, -1.0, 0.0],
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
        }
    )

    assert accounting is not None
    assert accounting["objective_unit"] == "BTC"
    assert accounting["objective_excess"] == pytest.approx(0.075)
    assert accounting["objective_value"] > accounting["benchmark_value"]


def test_executor_derives_signed_futures_events_with_mark_to_market() -> None:
    frame = [
        {"timestamp": NOW, "close": 100.0, "funding_rate": 0.0},
        {
            "timestamp": "2026-08-30T10:01:00+00:00",
            "close": 90.0,
            "funding_rate": 0.01,
            "funding_event": True,
        },
    ]

    accounting = _product_accounting(
        {
            "product_id": "active_income",
            "market_frame": frame,
            "signals": [-1.0, -1.0],
            "initial_cash": 1_000.0,
            "maximum_position": 0.1,
            "leverage": 2.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
        }
    )

    assert accounting is not None
    assert accounting["objective_unit"] == "USDT"
    assert accounting["unrealised_pnl"] == pytest.approx(20.0)
    assert accounting["funding_pnl"] == pytest.approx(2.0)
    assert accounting["max_leverage"] == pytest.approx(0.2)


def test_executor_does_not_charge_non_event_funding_quotes() -> None:
    accounting = _product_accounting(
        {
            "product_id": "active_income",
            "market_frame": [
                {"timestamp": NOW, "close": 100.0, "funding_rate": 0.01},
                {"timestamp": "2026-08-30T10:01:00+00:00", "close": 100.0, "funding_rate": 0.01},
            ],
            "signals": [1.0, 1.0],
            "initial_cash": 1_000.0,
            "maximum_position": 0.1,
            "leverage": 2.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
        }
    )

    assert accounting is not None
    assert accounting["funding_pnl"] == pytest.approx(0.0)


def test_forward_metrics_use_canonical_nav_for_drawdown() -> None:
    rows = (
        {
            "id": "nav-1",
            "created_at": NOW,
            "payload": {"product_id": "active_income", "observed_at": NOW, "nav": 100.0},
        },
        {
            "id": "nav-2",
            "created_at": "2026-08-30T10:01:00+00:00",
            "payload": {
                "product_id": "active_income",
                "observed_at": "2026-08-30T10:01:00+00:00",
                "nav": 90.0,
            },
        },
    )

    assert ForwardEvidenceCollector._nav_drawdown(
        rows,
        start=NOW,
        at="2026-08-30T10:01:00+00:00",
    ) == pytest.approx(0.1)


def test_btc_spot_execution_is_limited_to_owned_inventory_and_quote_proceeds() -> None:
    instrument = "binance:spot:BTCUSDT"
    target = TargetPosition(
        portfolio_id="btc",
        instrument_id=instrument,
        target_quantity=0.7,
        target_notional=70.0,
        target_fraction=0.7,
        strategy_contributions={"btc-risk-off": -0.3},
        risk_budget=0.3,
        valid_until="2026-08-30T11:00:00+00:00",
    )
    orders = plan_orders(
        (target,),
        current_quantities={instrument: 1.0},
        decided_at=NOW,
        prices={instrument: 100.0},
    )

    _validate_btc_spot_orders(
        orders,
        current={instrument: 1.0},
        balances={"USDT": 0.0},
        prices={instrument: 100.0},
        execution_costs={"fee_bps": 10.0, "slippage_bps": 2.0},
    )
    assert orders[0].side is OrderSide.SELL

    buy_target = TargetPosition(
        **{
            **target.__dict__,
            "target_quantity": 1.0,
            "target_notional": 100.0,
        }
    )
    buy_orders = plan_orders(
        (buy_target,),
        current_quantities={instrument: 0.7},
        decided_at=NOW,
        prices={instrument: 100.0},
    )
    with pytest.raises(ValueError, match="quote balance"):
        _validate_btc_spot_orders(
            buy_orders,
            current={instrument: 0.7},
            balances={"USDT": 0.0},
            prices={instrument: 100.0},
            execution_costs={"fee_bps": 10.0, "slippage_bps": 2.0},
        )


def test_btc_step_aside_metadata_binds_lot_and_quote_budget() -> None:
    metadata = btc_step_aside_metadata(
        instrument_id="binance:spot:BTCUSDT",
        current_btc=1.0,
        target_btc=0.7,
        price=100.0,
        stablecoin_balance=10.0,
        state_id="state-1",
        fee_bps=10.0,
        slippage_bps=2.0,
    )

    assert metadata["btc_cycle_state"] == "step_aside"
    assert metadata["btc_step_aside_lot_id"]
    assert metadata["btc_step_aside_sold_quantity"] == pytest.approx(0.3)
    assert metadata["btc_step_aside_quote_proceeds"] == pytest.approx(29.964006)
    assert metadata["btc_quote_reinvest_budget"] == pytest.approx(39.964006)


def test_btc_spot_quote_capacity_includes_fees() -> None:
    target = TargetPosition(
        portfolio_id="btc-accumulation-portfolio",
        instrument_id="binance:spot:BTCUSDT",
        target_quantity=1.0,
        target_notional=100.0,
        target_fraction=1.0,
        strategy_contributions={"strategy": 1.0},
        risk_budget=0.3,
        valid_until="2026-08-30T11:00:00+00:00",
        metadata={},
    )
    orders = plan_orders(
        (target,),
        current_quantities={"binance:spot:BTCUSDT": 0.7},
        decided_at=NOW,
        prices={"binance:spot:BTCUSDT": 100.0},
    )

    with pytest.raises(ValueError, match="quote balance"):
        _validate_btc_spot_orders(
            orders,
            current={"binance:spot:BTCUSDT": 0.7},
            balances={"USDT": 30.0},
            prices={"binance:spot:BTCUSDT": 100.0},
            execution_costs={"fee_bps": 10.0, "slippage_bps": 2.0},
        )


def test_product_objective_rejects_positive_usdt_result_that_loses_btc() -> None:
    assert not objective_passes(
        {
            "objective_status": "measured",
            "objective_unit": "BTC",
            "objective_value": 0.9,
            "benchmark_value": 1.0,
            "objective_excess": -0.1,
            "objective_excess_fraction": -0.1,
        },
        product_id="btc_accumulation",
        minimum_excess_fraction=0.0,
    )


def test_btc_allocation_is_neutral_without_an_explicit_reserve() -> None:
    neutral = target_btc_allocation(())
    reserved = target_btc_allocation(
        (), policy=BtcAllocationPolicy(core_btc_fraction=0.8, max_tactical_fraction=0.0)
    )

    assert neutral.target_btc_fraction == pytest.approx(1.0)
    assert neutral.stablecoin_fraction == pytest.approx(0.0)
    assert reserved.target_btc_fraction == pytest.approx(0.8)
    assert reserved.stablecoin_fraction == pytest.approx(0.2)


def test_futures_accounting_keeps_short_pnl_and_funding_signed() -> None:
    report = FuturesIncomeAccounting().evaluate(
        initial_cash=1_000.0,
        leverage=2.0,
        events=(
            {
                "type": "fill",
                "timestamp": NOW,
                "symbol": "BTCUSDT",
                "side": "sell",
                "quantity": 1.0,
                "price": 100.0,
            },
            {
                "type": "mark",
                "timestamp": "2026-08-30T10:01:00+00:00",
                "symbol": "BTCUSDT",
                "mark_price": 90.0,
            },
            {
                "type": "funding",
                "timestamp": "2026-08-30T10:02:00+00:00",
                "symbol": "BTCUSDT",
                "mark_price": 90.0,
                "funding_rate": 0.01,
            },
        ),
    )

    assert report.unrealised_pnl == pytest.approx(10.0)
    assert report.funding_pnl == pytest.approx(0.9)
    assert report.net_pnl == pytest.approx(10.9)
    assert report.liquidation is False


def test_futures_accounting_applies_funding_only_at_scheduled_timestamps() -> None:
    scheduled = "2026-08-30T10:02:00+00:00"
    report = FuturesIncomeAccounting().evaluate(
        initial_cash=1_000.0,
        events=(
            {
                "type": "fill",
                "timestamp": NOW,
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 1.0,
                "price": 100.0,
            },
            {
                "type": "funding",
                "timestamp": "2026-08-30T10:01:00+00:00",
                "symbol": "BTCUSDT",
                "mark_price": 100.0,
                "funding_rate": 0.01,
            },
            {
                "type": "funding",
                "timestamp": scheduled,
                "symbol": "BTCUSDT",
                "mark_price": 100.0,
                "funding_rate": 0.01,
            },
        ),
        funding_timestamps=(scheduled,),
    )

    assert report.funding_pnl == pytest.approx(-1.0)
    assert report.event_receipts[1]["funding_applied"] is False
    assert report.event_receipts[2]["funding_applied"] is True


def test_futures_accounting_reports_leverage_and_margin_policy_violations() -> None:
    report = FuturesIncomeAccounting().evaluate(
        initial_cash=100.0,
        leverage=2.0,
        max_margin_fraction=0.5,
        events=(
            {
                "type": "fill",
                "timestamp": NOW,
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 2.0,
                "price": 100.0,
            },
        ),
    )

    assert report.capacity_violations >= 1
    assert report.max_leverage == pytest.approx(2.0)


def test_futures_accounting_reports_target_notional_and_shortfall_metrics() -> None:
    report = FuturesIncomeAccounting().evaluate(
        initial_cash=1_000.0,
        leverage=2.0,
        target_notional={"BTCUSDT": 100.0},
        liquidation_buffer_fraction=0.05,
        events=(
            {
                "type": "fill",
                "timestamp": NOW,
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 2.0,
                "price": 100.0,
                "requested_quantity": 3.0,
                "fee": 1.0,
                "reference_price": 99.0,
            },
        ),
    )

    assert report.capacity_violations >= 1
    assert report.partial_fills == 1
    assert report.margin_mode == "isolated"
    assert report.liquidation_buffer_fraction == pytest.approx(0.05)
    assert report.implementation_shortfall == pytest.approx(3.0)
    assert report.turnover_notional == pytest.approx(200.0)


def test_short_forecast_is_ranked_and_allocated_after_funding() -> None:
    forecast = AlphaForecast(
        strategy_version_id="short-strategy",
        product_id="active_income",
        instrument_id="BTCUSDT",
        direction=ForecastDirection.SHORT,
        score=1.0,
        expected_return=-0.1,
        confidence=1.0,
        horizon_seconds=3_600,
        valid_from=NOW,
        valid_until="2026-08-30T11:00:00+00:00",
        target_volatility=0.1,
        maximum_position=0.2,
    )
    targets = optimise_targets(
        (forecast,),
        prices={"BTCUSDT": 100.0},
        valid_until="2026-08-30T11:00:00+00:00",
        constraints=PortfolioConstraints(
            portfolio_id="active_income",
            equity=1_000.0,
            max_net_fraction=1.0,
            max_symbol_fraction=0.5,
        ),
        funding_rates={"BTCUSDT": 0.01},
    )

    assert len(targets) == 1
    assert targets[0].target_fraction < 0
    assert targets[0].metadata["net_expected_return"] == pytest.approx(0.11)


def test_position_return_ledger_accumulates_signed_costs_once() -> None:
    report = PositionReturnLedger(
        fee_rate=0.01,
        slippage_rate=0.005,
        funding_rate=0.01,
    ).measure(
        positions=(1.0, -1.0, -1.0),
        market_returns=(0.10, -0.05),
    )

    assert report.gross_pnl == pytest.approx(0.15)
    assert report.turnover == pytest.approx(2.0)
    assert report.fees == pytest.approx(0.02)
    assert report.slippage == pytest.approx(0.01)
    assert report.funding_pnl == pytest.approx(0.0)
    assert report.net_pnl == pytest.approx(0.12)
    assert report.period_turnover == pytest.approx((2.0, 0.0))
    assert report.period_fees == pytest.approx((0.02, 0.0))
    assert report.period_slippage == pytest.approx((0.01, 0.0))
    assert report.period_funding_pnl == pytest.approx((-0.01, 0.01))


def test_position_return_ledger_keeps_funding_signed_per_period() -> None:
    report = PositionReturnLedger(fee_rate=0.01).measure(
        positions=(0.0, 1.0, 1.0),
        market_returns=(0.10, -0.10),
        funding_rates=(0.02, -0.03),
    )

    assert report.period_turnover == pytest.approx((1.0, 0.0))
    assert report.period_fees == pytest.approx((0.01, 0.0))
    assert report.period_funding_pnl == pytest.approx((0.0, 0.03))
    assert report.net_returns == pytest.approx((-0.01, -0.07))
    assert report.net_pnl == pytest.approx(-0.08)


def test_futures_return_ledger_scales_fractional_costs_into_usdt() -> None:
    report = PositionReturnLedger(fee_rate=0.01, slippage_rate=0.005).measure(
        positions=(0.0, 1.0, 1.0),
        market_returns=(0.10, -0.10),
        funding_rates=(0.02, -0.03),
    )

    accounting = _product_accounting(
        {"product_id": "active_income", "initial_cash": 1_000.0},
        fallback_return=report,
    )

    assert accounting is not None
    assert accounting["fees"] == pytest.approx(10.0)
    assert accounting["slippage_cost"] == pytest.approx(5.0)
    assert accounting["funding_pnl"] == pytest.approx(30.0)
    assert accounting["turnover_notional"] == pytest.approx(1_000.0)
    assert accounting["implementation_shortfall"] == pytest.approx(15.0)


def test_pbo_uses_the_configured_window_count_and_parameter_cohort() -> None:
    pbo, matrix = _pbo_measurements(
        {"walk_forward_windows": 5},
        {"results": [{"window_returns": [0.1, 0.2, 0.1, 0.2, 0.1]}]},
        [0.05, 0.05, 0.05, 0.05, 0.05],
    )

    assert isinstance(pbo, float)
    assert matrix == [
        [0.05, 0.05, 0.05, 0.05, 0.05],
        [0.1, 0.2, 0.1, 0.2, 0.1],
    ]


def test_dataset_bundle_supports_explicit_pending_lifecycle_and_verifies_stages(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'datasets.sqlite3'}")
    database.create_schema()
    identity = "sha256:" + "a" * 64
    builder = CanonicalResearchDatasetBuilder(database.engine)
    bundle = builder.build(
        "active_income",
        intervals={
            "screening": {
                "start": "2026-08-29T00:00:00+00:00",
                "end": "2026-08-29T06:00:00+00:00",
            },
            "development": {
                "start": "2026-08-29T06:00:00+00:00",
                "end": "2026-08-29T12:00:00+00:00",
            },
        },
        payload_by_role={"screening": {"rows": 1}, "development": {"rows": 2}},
        universe_snapshot_id=identity,
        feature_manifest_id=identity,
        cost_model_id=identity,
        parameter_set_id=identity,
        instrument_scope=("BTCUSDT",),
        availability_timestamp=NOW,
        created_at=NOW,
        lifecycle_state=DatasetLifecycleState.DATA_PENDING,
    )

    assert bundle.lifecycle_state is DatasetLifecycleState.DATA_PENDING
    assert set(bundle.stage_snapshot_ids) == {"screening", "development"}
    assert SqlDatasetBundleRepository(database.engine).get(bundle.bundle_id) == bundle


def test_ready_dataset_bundle_rejects_overlapping_intervals(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'datasets.sqlite3'}")
    database.create_schema()
    identity = "sha256:" + "b" * 64
    kwargs = {
        "universe_snapshot_id": identity,
        "feature_manifest_id": identity,
        "cost_model_id": identity,
        "parameter_set_id": identity,
        "instrument_scope": ("BTCUSDT",),
        "availability_timestamp": NOW,
        "created_at": NOW,
    }
    with pytest.raises(DatasetResolutionError, match="overlap"):
        CanonicalResearchDatasetBuilder(database.engine).build(
            "active_income",
            intervals={
                role: {
                    "start": "2026-08-29T00:00:00+00:00",
                    "end": "2026-08-29T12:00:00+00:00",
                }
                for role in (
                    "screening",
                    "development",
                    "robustness",
                    "protected_holdout",
                    "forward_observation",
                )
            },
            payload_by_role={
                role: {"role": role}
                for role in (
                    "screening",
                    "development",
                    "robustness",
                    "protected_holdout",
                    "forward_observation",
                )
            },
            **kwargs,
        )


def test_dataset_builder_selects_only_available_bars_for_every_stage(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'bar-datasets.sqlite3'}")
    database.create_schema()
    roles = ("screening", "development", "robustness", "protected_holdout")
    intervals = {
        role: {
            "start": f"2026-08-{26 + index:02d}T00:00:00+00:00",
            "end": f"2026-08-{27 + index:02d}T00:00:00+00:00",
        }
        for index, role in enumerate(roles)
    }
    identity = "sha256:" + "c" * 64
    bars = [
        {
            "instrument_id": "binance:futures:BTCUSDT:USDT",
            "close_timestamp": f"2026-08-{26 + index:02d}T12:00:00+00:00",
            "availability_time": NOW,
            "close": 100.0 + index,
        }
        for index in range(len(roles))
    ]

    bundle = CanonicalResearchDatasetBuilder(database.engine).build_from_bars(
        "active_income",
        bars=bars,
        intervals=intervals,
        universe_snapshot_id=identity,
        feature_manifest_id=identity,
        cost_model_id=identity,
        parameter_set_id=identity,
        instrument_scope=("binance:futures:BTCUSDT:USDT",),
        created_at=NOW,
    )

    assert bundle.lifecycle_state is DatasetLifecycleState.READY
    assert set(bundle.stage_snapshot_ids) == set(roles)
    for role, snapshot_id in bundle.stage_snapshot_ids.items():
        payload = SqlDatasetBundleRepository(database.engine).get(bundle.bundle_id)
        assert payload.stage_snapshot_ids[role] == snapshot_id

    with pytest.raises(DatasetResolutionError, match="data_pending"):
        CanonicalResearchDatasetBuilder(database.engine).build_from_bars(
            "active_income",
            bars=bars[:2],
            intervals=intervals,
            universe_snapshot_id=identity,
            feature_manifest_id=identity,
            cost_model_id=identity,
            parameter_set_id=identity,
            instrument_scope=("binance:futures:BTCUSDT:USDT",),
            created_at=NOW,
        )


def test_readiness_excludes_synthetic_bundles_and_does_not_require_forward_data() -> None:
    identity = "sha256:" + "d" * 64
    stages = {
        role: f"sha256:{chr(100 + index) * 64}"
        for index, role in enumerate(
            ("screening", "development", "robustness", "protected_holdout")
        )
    }
    bundle = {
        "product_id": "active_income",
        "lifecycle_state": "ready",
        "stage_snapshot_ids": stages,
    }
    snapshots = [
        {"id": snapshot_id, "payload": {"role": role, "payload": {"bars": [1]}}}
        for role, snapshot_id in stages.items()
    ]

    assert _ready_dataset_roles([bundle], snapshots) == {"active_income": set(stages)}
    assert (
        _dataset_readiness(
            {"active_income": set(stages)}, {"dataset_bundles": 1}, {"active_income"}
        )["ok"]
        is True
    )

    synthetic = [
        {
            "id": snapshot_id,
            "payload": {
                "role": role,
                "payload": {"bars": [1], "synthetic": role == "screening"},
            },
        }
        for role, snapshot_id in stages.items()
    ]
    assert _ready_dataset_roles([bundle], synthetic) == {}
    assert identity.startswith("sha256:")


def test_latest_ready_bundle_excludes_synthetic_diagnostic_data(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'diagnostic.sqlite3'}")
    database.create_schema()
    identity = "sha256:" + "e" * 64
    intervals = {
        role: {
            "start": f"2026-08-{26 + index:02d}T00:00:00+00:00",
            "end": f"2026-08-{27 + index:02d}T00:00:00+00:00",
        }
        for index, role in enumerate(CORE_RESEARCH_BUNDLE_ROLES)
    }
    builder = CanonicalResearchDatasetBuilder(database.engine)
    bundle = builder.build(
        "active_income",
        intervals=intervals,
        payload_by_role={
            role: {"bars": [1], "diagnostic": True, "synthetic": True}
            for role in CORE_RESEARCH_BUNDLE_ROLES
        },
        universe_snapshot_id=identity,
        feature_manifest_id=identity,
        cost_model_id=identity,
        parameter_set_id=identity,
        instrument_scope=("BTCUSDT",),
        availability_timestamp=NOW,
        created_at=NOW,
    )

    assert SqlDatasetBundleRepository(database.engine).get(bundle.bundle_id) == bundle
    assert SqlDatasetBundleRepository(database.engine).latest_ready("active_income", at=NOW) is None


def test_dataset_service_builds_a_real_bundle_from_point_in_time_bars(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'dataset-service.sqlite3'}")
    database.create_schema()
    instrument = Instrument(
        venue="binance",
        market_type=MarketType.SPOT,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset=None,
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5.0,
    )
    universe_id = "service-btc-universe"
    SqlUniverseStore(database.engine).record_snapshot(
        universe_id=universe_id,
        observed_at=NOW,
        observations=(
            InstrumentObservation(
                instrument=instrument,
                listing_age_days=365.0,
                quote_volume=1_000_000_000.0,
                trade_count=1_000_000,
                spread_bps=1.0,
                open_interest=0.0,
                funding_rate=0.0,
                realised_volatility=0.2,
                depth_notional=10_000_000.0,
                data_completeness=1.0,
            ),
        ),
        policy=UniverseEligibilityPolicy(),
    )
    feature_id = canonical_hash({"schema": "platform.feature_manifest/v1", "market_type": "spot"})
    cost_id = canonical_hash({"schema": "platform.cost_model/v1", "product_id": "btc_accumulation"})
    with database.engine.begin() as connection:
        connection.execute(
            feature_manifest.insert().values(
                id=feature_id,
                created_at=NOW,
                payload={"schema": "platform.feature_manifest/v1", "market_type": "spot"},
            )
        )
        connection.execute(
            cost_model_manifest.insert().values(
                id=cost_id,
                created_at=NOW,
                payload={"schema": "platform.cost_model/v1", "product_id": "btc_accumulation"},
            )
        )
    root = tmp_path / "data"
    partition = root / "bars" / "binance" / "spot" / "BTCUSDT" / "1m" / "date=2026-08-20"
    partition.mkdir(parents=True)
    timestamps = [
        int(dt.datetime(2026, 8, 20 + index, tzinfo=dt.UTC).timestamp() * 1_000)
        for index in range(8)
    ]
    pq.write_table(
        pa.table(
            {
                "event_id": [f"event-{index}" for index in range(8)],
                "instrument_id": [instrument.instrument_id] * 8,
                "close_time_ms": timestamps,
                "availability_time": [NOW] * 8,
                "open": [100.0 + index for index in range(8)],
                "high": [101.0 + index for index in range(8)],
                "low": [99.0 + index for index in range(8)],
                "close": [100.0 + index for index in range(8)],
                "volume": [10.0] * 8,
            }
        ),
        partition / "bars.parquet",
    )

    result = DatabaseDatasetBundleService(database.engine, root).run(
        product_id="btc_accumulation",
        universe_id=universe_id,
        market_type="spot",
        created_at=NOW,
    )

    assert result.state == "ready"
    assert result.bundle_id is not None
    assert len(result.source_partition_hashes) == 1
    bundle = SqlDatasetBundleRepository(database.engine).get(str(result.bundle_id))
    assert set(bundle.stage_snapshot_ids) == set(CORE_RESEARCH_BUNDLE_ROLES)
    with database.engine.connect() as connection:
        snapshot_payload = connection.execute(
            select(dataset_snapshot.c.payload).where(
                dataset_snapshot.c.id == bundle.stage_snapshot_ids["screening"]
            )
        ).scalar_one()
    assert snapshot_payload["payload"]["market_frame"]

    forward = DatabaseDatasetBundleService(database.engine, root).run_forward(
        product_id="btc_accumulation",
        universe_id=universe_id,
        market_type="spot",
        artefact_created_at="2026-08-19T00:00:00+00:00",
        created_at=NOW,
    )
    assert forward.state == "ready"
    assert forward.reason_code == "forward_dataset_ready"
    assert len(forward.snapshot_ids) == 1
    forward_bundle = SqlDatasetBundleRepository(database.engine).get(str(forward.bundle_id))
    assert set(forward_bundle.stage_snapshot_ids) == {"forward_observation"}

    pending = DatabaseDatasetBundleService(database.engine, root, minimum_history_days=365).run(
        product_id="btc_accumulation",
        universe_id=universe_id,
        market_type="spot",
        created_at=NOW,
    )
    assert pending.state == "waiting_for_dataset"
    assert pending.reason_code == "historical_history_insufficient"

    claimed = ClaimedJob(
        job_id="catalogue-service",
        name="register_strategy_catalogue",
        payload={
            "product_id": "btc_accumulation",
            "instrument_universe": ["BTCUSDT"],
            "dataset_snapshot_hashes": [],
            "dataset_bundle_id": None,
            "universe_id": universe_id,
            "market_type": "spot",
            "dataset_timeframe": "1m",
            "available_at": NOW,
            "catalogue_submitted_at": NOW,
        },
        worker_id="catalogue-worker",
        attempt=1,
        lease_expires_at="2026-08-30T10:01:00+00:00",
    )
    catalogue = DatabaseResearchJobHandlers(
        SqlResearchStore(database.engine),
        dataset_bundle_service=DatabaseDatasetBundleService(database.engine, root),
    ).register_strategy_catalogue(
        claimed,
        lambda: claimed,
    )
    assert catalogue["registered_candidates"] > 0


def test_dataset_service_keeps_delisted_symbols_in_their_historical_roles(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'historical-universe.sqlite3'}")
    database.create_schema()
    btc = Instrument(
        venue="binance",
        market_type=MarketType.SPOT,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset=None,
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5.0,
    )
    eth = Instrument(
        venue="binance",
        market_type=MarketType.SPOT,
        base_asset="ETH",
        quote_asset="USDT",
        settlement_asset=None,
        exchange_symbol="ETHUSDT",
        price_precision=2,
        quantity_precision=5,
        minimum_quantity=0.00001,
        minimum_notional=5.0,
    )
    universe_id = "historical-spot-universe"
    policy = UniverseEligibilityPolicy()

    def observation(instrument: Instrument) -> InstrumentObservation:
        return InstrumentObservation(
            instrument=instrument,
            listing_age_days=365.0,
            quote_volume=1_000_000_000.0,
            trade_count=1_000_000,
            spread_bps=1.0,
            open_interest=0.0,
            funding_rate=0.0,
            realised_volatility=0.2,
            depth_notional=10_000_000.0,
            data_completeness=1.0,
        )

    store = SqlUniverseStore(database.engine)
    store.record_snapshot(
        universe_id=universe_id,
        observed_at="2026-08-01T00:00:00+00:00",
        observations=(observation(btc), observation(eth)),
        policy=policy,
    )
    store.record_snapshot(
        universe_id=universe_id,
        observed_at="2026-08-03T00:00:00+00:00",
        observations=(observation(btc),),
        policy=policy,
    )
    feature_id = canonical_hash({"schema": "platform.feature_manifest/v1", "market_type": "spot"})
    cost_id = canonical_hash({"schema": "platform.cost_model/v1", "product_id": "active_income"})
    with database.engine.begin() as connection:
        connection.execute(
            feature_manifest.insert().values(
                id=feature_id,
                created_at=NOW,
                payload={"schema": "platform.feature_manifest/v1", "market_type": "spot"},
            )
        )
        connection.execute(
            cost_model_manifest.insert().values(
                id=cost_id,
                created_at=NOW,
                payload={"schema": "platform.cost_model/v1", "product_id": "active_income"},
            )
        )
    root = tmp_path / "data"
    rows_by_instrument = {
        btc.instrument_id: (100.0, 101.0, 102.0, 103.0),
        eth.instrument_id: (10.0, 10.1, 10.2, 10.3),
    }
    for item, prices in rows_by_instrument.items():
        symbol = "BTCUSDT" if item == btc.instrument_id else "ETHUSDT"
        partition = root / "bars" / "binance" / "spot" / symbol / "1m" / "history"
        partition.mkdir(parents=True)
        times = [
            int(dt.datetime(2026, 8, 1 + index, tzinfo=dt.UTC).timestamp() * 1_000)
            for index in range(4)
        ]
        pq.write_table(
            pa.table(
                {
                    "instrument_id": [item] * 4,
                    "close_time_ms": times,
                    "availability_time": [NOW] * 4,
                    "open": list(prices),
                    "high": [price + 1.0 for price in prices],
                    "low": [price - 1.0 for price in prices],
                    "close": list(prices),
                    "volume": [10.0] * 4,
                }
            ),
            partition / "bars.parquet",
        )

    result = DatabaseDatasetBundleService(database.engine, root).run(
        product_id="active_income",
        universe_id=universe_id,
        market_type="spot",
        created_at=NOW,
    )

    assert result.state == "ready"
    bundle = SqlDatasetBundleRepository(database.engine).get(str(result.bundle_id))
    with database.engine.connect() as connection:
        payloads = {
            role: connection.execute(
                select(dataset_snapshot.c.payload).where(dataset_snapshot.c.id == snapshot_id)
            ).scalar_one()
            for role, snapshot_id in bundle.stage_snapshot_ids.items()
        }
    screening_rows = payloads["screening"]["payload"]["market_frame"]
    robustness_rows = payloads["robustness"]["payload"]["market_frame"]
    assert any(row["instrument_id"] == eth.instrument_id for row in screening_rows)
    assert all(row["instrument_id"] != eth.instrument_id for row in robustness_rows)


def test_dataset_row_budget_preserves_each_instrument_and_history_endpoints() -> None:
    def rows(instrument_id: str) -> list[dict[str, object]]:
        return [
            {
                "instrument_id": instrument_id,
                "close_timestamp": f"2026-08-{20 + index:02d}T00:00:00+00:00",
            }
            for index in range(10)
        ]

    sampled = _bounded_instrument_rows(
        {"instrument-a": rows("instrument-a"), "instrument-b": rows("instrument-b")},
        maximum_rows=8,
    )

    assert len(sampled) == 8
    assert {str(row["instrument_id"]) for row in sampled} == {
        "instrument-a",
        "instrument-b",
    }
    for instrument_id in ("instrument-a", "instrument-b"):
        timestamps = [
            str(row["close_timestamp"]) for row in sampled if row["instrument_id"] == instrument_id
        ]
        assert timestamps[0] == "2026-08-20T00:00:00+00:00"
        assert timestamps[-1] == "2026-08-29T00:00:00+00:00"
