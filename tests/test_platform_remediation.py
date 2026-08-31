from __future__ import annotations

import pytest
from sqlalchemy import select

from src.data.database import PlatformDatabase, job
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.portfolio.optimiser import PortfolioConstraints, optimise_targets
from src.products.btc_accumulation import BtcAllocationPolicy, target_btc_allocation
from src.research.accounting import (
    BtcAccumulationAccounting,
    BtcResearchAccounting,
    FuturesIncomeAccounting,
    FuturesResearchAccounting,
    ProductAccountingError,
)
from src.research.artefacts import StrategyArtefact
from src.research.catalogue import registered_strategy_candidates
from src.research.datasets import (
    CanonicalResearchDatasetBuilder,
    DatasetLifecycleState,
    DatasetResolutionError,
    SqlDatasetBundleRepository,
)
from src.research.evaluation import EvidencePolicy, EvidenceProfile, EvidenceStatus
from src.research.evidence import (
    cross_symbol_stability_passes,
    drawdown_passes,
    holdout_degradation_passes,
    monte_carlo_passes,
    parameter_stability_passes,
    sample_evidence_passes,
)
from src.research.executors import _cross_symbol_stability, _portfolio_overlap, _product_accounting
from src.research.objectives import objective_passes
from src.research.returns import PositionReturnLedger
from src.services.artefact_dispatcher import ArtefactDispatcher
from src.services.scheduler import DatabaseJobQueue
from src.strategies.behaviour import RegisteredStrategyBehaviour

NOW = "2026-08-30T10:00:00+00:00"


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
        {"timestamp": "2026-08-30T10:01:00+00:00", "close": 90.0, "funding_rate": 0.01},
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
