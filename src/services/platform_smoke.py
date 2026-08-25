"""Run the real PostgreSQL event-to-accounting service chain for both products."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.data.binance_market import normalise_public_event
from src.data.database import (
    PlatformDatabase,
    fee_entry,
    strategy_definition,
    strategy_version,
    trade_attribution,
)
from src.data.feature_store import SqlFeatureStore
from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.domain._codec import canonical_hash, to_primitive
from src.domain.instruments import Instrument, MarketType
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.execution.order_groups import OrderGroupManager, SqlOrderGroupStore
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.observability.decision_trace import SqlDecisionTraceStore
from src.research.artefacts import StrategyArtefact
from src.research.canonical import (
    SqlActiveStrategyAssignmentRepository,
    SqlStrategyArtefactRepository,
)
from src.risk.engine import SqlRiskDecisionStore, SqlRiskPolicyStore, SqlRiskSnapshotStore
from src.services.accounting_service import AccountingService, DatabaseAccountingWorker
from src.services.config import load_split_configuration
from src.services.data_writer import DatabaseMarketDataWriter
from src.services.feature_worker import DatabaseFeatureWorker
from src.services.order_execution import DatabaseExecutionWorker, DatabasePaperExecutionWorker
from src.services.portfolio_engine import (
    DatabasePortfolioTargetBuilder,
    DatabasePortfolioTargetWorker,
)
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.portfolio_state import (
    DatabasePortfolioSourceService,
    DatabasePortfolioStateWorker,
)
from src.services.risk_service import DatabaseRiskWorker
from src.services.scheduler import DatabaseJobQueue
from src.services.strategy_evaluator import DatabaseStrategyEvaluator
from src.strategies.manifest import registered_live_contract

_STAGES = (
    "closed_candle",
    "features",
    "strategy_assignment",
    "state",
    "alpha_forecast",
    "target_position",
    "risk_decision",
    "order_intent",
    "partial_fill",
    "complete_fill",
    "position",
    "accounting_entry",
    "attribution",
    "decision_trace",
)


def run_smoke(
    database_url: str, *, config_path: Path = Path("config/platform.json")
) -> dict[str, Any]:
    database = PlatformDatabase(database_url)
    if not database.is_postgresql:
        raise ValueError("platform smoke requires PostgreSQL")
    database.assert_migrated()
    split = load_split_configuration(config_path.parent)
    accounts = {str(item["account_id"]): dict(item) for item in split["accounts"]["accounts"]}
    results: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="platform-smoke-") as directory:
            for index, product in enumerate(split["products"]["products"]):
                results.append(
                    _product_fixture(database, dict(product), accounts, Path(directory), index)
                )
    finally:
        database.dispose()
    return {
        "schema": "platform.smoke/v2",
        "ok": all(item["ok"] for item in results),
        "postgresql_service_chain": True,
        "products": results,
    }


def _product_fixture(
    database: PlatformDatabase,
    product: dict[str, Any],
    accounts: dict[str, dict[str, Any]],
    root: Path,
    index: int,
) -> dict[str, Any]:
    product_id = str(product["product_id"])
    observed = dt.datetime(2026, 8, 23, tzinfo=dt.UTC)
    now = observed.isoformat()
    market = "spot" if product_id == "btc_accumulation" else "futures"
    instrument = Instrument(
        venue="binance",
        market_type=MarketType.SPOT if market == "spot" else MarketType.FUTURES,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset=None if market == "spot" else "USDT",
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5.0,
    )
    prefix = f"smoke:{product_id}:{index}"
    queue = DatabaseJobQueue(database.engine)
    workers_ids = {
        name: f"{prefix}:{name}"
        for name in (
            "data",
            "feature",
            "strategy",
            "state",
            "portfolio",
            "risk",
            "execution",
            "paper",
            "accounting",
        )
    }
    capabilities = {
        "data": ("market_event_write",),
        "feature": ("live_feature_calculation",),
        "strategy": ("strategy_evaluation",),
        "state": ("portfolio_state_publish",),
        "portfolio": ("portfolio_target_build",),
        "risk": ("risk_assessment",),
        "execution": ("execute_targets",),
        "paper": ("paper_order_submit", "paper_order_continue"),
        "accounting": ("accounting_event",),
    }
    for role, worker_id in workers_ids.items():
        queue.register_worker(
            worker_id=worker_id,
            node_id="linux-optiplex",
            role=role,
            capabilities=capabilities[role],
            observed_at=now,
        )
    universe_snapshot_id = SqlUniverseStore(database.engine).record_snapshot(
        universe_id=prefix + ":universe",
        observed_at=now,
        observations=(
            InstrumentObservation(instrument, 1_000, 1e9, 1_000_000, 1.0, 1e9, 0.0, 0.2, 1e7, 1.0),
        ),
        policy=UniverseEligibilityPolicy(),
    )
    assignment_id = _seed_strategy(database, product, instrument, universe_snapshot_id, now, prefix)
    assignments = SqlActiveStrategyAssignmentRepository(database.engine)
    snapshots = SqlRiskSnapshotStore(database.engine)
    _install_risk_policy(database, str(product["risk_policy_id"]))

    close_ms = int(observed.timestamp() * 1_000) - 1
    event = normalise_public_event(
        market=market,
        stream="btcusdt@kline_1m",
        receive_timestamp=now,
        payload={
            "e": "kline",
            "E": close_ms + 1,
            "s": "BTCUSDT",
            "k": {
                "t": close_ms - 59_999,
                "T": close_ms,
                "i": "1m",
                "o": "100000",
                "h": "102500",
                "l": "99750",
                "c": "102000",
                "v": "25",
                "spread_bps": "1",
                "visible_depth": "10000000",
                "volatility": "0.2",
                "funding": "0.0",
                "x": True,
            },
        },
    )
    queue.enqueue(
        job_id=prefix + ":event",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": market,
            "symbol": "BTCUSDT",
            "event": to_primitive(event),
        },
        available_at=now,
    )
    repository = SqlPortfolioRepository(database.engine, require_pipeline_identity=True)
    feature_store = SqlFeatureStore(database.engine)
    positions = PositionManager(SqlPositionStore(database.engine))
    order_manager = OrderManager(SqlOrderStore(database.engine))
    traces = SqlDecisionTraceStore(database.engine)
    risk_store = SqlRiskDecisionStore(database.engine)
    ledger = Ledger(
        product_id=product_id,
        accounting_asset=str(product["base_accounting_asset"]),
        store=SqlLedgerStore(database.engine, product_id=product_id),
    )
    queue.enqueue(
        job_id=prefix + ":initial-balance",
        name="accounting_event",
        payload={
            "kind": "balance",
            "product_id": product_id,
            "account_id": str(product["account_id"]),
            "observed_at": now,
            "balances": (
                {"BTC": 0.01, "USDT": 1000.0}
                if product_id == "btc_accumulation"
                else {"USDT": 10000.0}
            ),
        },
        available_at=now,
    )
    accounting_service = AccountingService(
        engine=database.engine,
        ledgers={product_id: ledger},
        snapshot_store=snapshots,
    )
    initial_accounting = DatabaseAccountingWorker(
        queue=queue,
        worker_id=workers_ids["accounting"],
        service=accounting_service,
    ).run_once(now=now)
    order_groups = OrderGroupManager(SqlOrderGroupStore(database.engine))
    data_worker = DatabaseMarketDataWriter(
        queue=queue,
        worker_id=workers_ids["data"],
        root=root / product_id,
        snapshot_store=snapshots,
        product_ids_by_market={market: (product_id,)},
    )
    feature_worker = DatabaseFeatureWorker(
        queue=queue,
        worker_id=workers_ids["feature"],
        store=feature_store,
        job_names=("live_feature_calculation",),
        parquet_root=root / product_id,
        active_assignments=lambda instrument_id: tuple(
            item
            for item in assignments.active_assignments(product_id)
            if item.get("instrument_id") == instrument_id
        ),
        snapshot_store=snapshots,
        feature_graph_for_assignment=lambda _assignment: {"required_nodes": ["bar_return"]},
    )
    strategy_worker = DatabaseStrategyEvaluator(
        queue=queue,
        worker_id=workers_ids["strategy"],
        feature_store=feature_store,
        portfolio=repository,
        assignments=assignments,
    )
    portfolio_worker = DatabasePortfolioTargetWorker(
        queue=queue,
        worker_id=workers_ids["portfolio"],
        build_target=DatabasePortfolioTargetBuilder(
            repository=repository,
            snapshot_store=snapshots,
            positions=positions,
            product_configuration={product_id: product},
            account_configuration=accounts,
        ),
    )
    risk_worker = DatabaseRiskWorker(
        queue=queue,
        worker_id=workers_ids["risk"],
        store=risk_store,
        snapshot_store=snapshots,
        execution_modes={product_id: "paper"},
    )
    execution_worker = DatabaseExecutionWorker(
        queue=queue,
        worker_id=workers_ids["execution"],
        order_manager=order_manager,
        positions=positions,
        risk_store=risk_store,
        trace_store=traces,
        order_groups=order_groups,
        snapshot_store=snapshots,
        product_execution={
            product_id: {
                "execution_mode": "paper",
                "execution_costs": product["execution_costs"],
                "base_accounting_asset": product["base_accounting_asset"],
                "fill_fraction": 0.5,
            }
        },
    )
    paper_worker = DatabasePaperExecutionWorker(
        queue=queue,
        worker_id=workers_ids["paper"],
        order_manager=order_manager,
        positions=positions,
        ledgers={product_id: ledger},
        trace_store=traces,
        order_groups=order_groups,
    )
    data_result = data_worker.run_once(now=now)
    feature_result = feature_worker.run_once(now=now)
    source_service = DatabasePortfolioSourceService(
        engine=database.engine,
        store=snapshots,
        products={product_id: product},
        accounts=accounts,
    )
    state_worker = DatabasePortfolioStateWorker(
        queue=queue,
        worker_id=workers_ids["state"],
        store=snapshots,
        refresh_sources=source_service.refresh,
    )
    if (
        state_worker.schedule_from_latest(
            products={product_id: product},
            state_policies={product_id: _smoke_state_policy(product, instrument)},
            now=now,
        )
        != 1
    ):
        raise RuntimeError(f"{product_id} smoke did not schedule canonical portfolio state")
    state_result = state_worker.run_once(now=now)
    results = {
        "data": data_result,
        "feature": feature_result,
        "initial_accounting": initial_accounting,
        "state": state_result,
        "strategy": strategy_worker.run_once(now=now),
        "portfolio": portfolio_worker.run_once(now=now),
        "risk": risk_worker.run_once(now=now),
        "execution": execution_worker.run_once(now=now),
        "partial": paper_worker.run_once(now=now),
        "complete": paper_worker.run_once(now=now),
    }
    fills = order_manager.all_fills()
    if not fills:
        raise RuntimeError(f"{product_id} smoke produced no fills: {results}")
    final_fill = fills[-1]
    queue.enqueue(
        job_id=prefix + ":accounting",
        name="accounting_event",
        payload={
            "kind": "fee_evidence",
            "product_id": product_id,
            "entry_id": prefix + ":fee:" + final_fill.fill_id,
            "amount": str(final_fill.fee),
            "occurred_at": final_fill.occurred_at,
            "attribution": {
                "fill_id": final_fill.fill_id,
                "order_id": final_fill.order_id,
                "strategy_assignment_id": assignment_id,
            },
        },
        available_at=now,
    )
    accounting = DatabaseAccountingWorker(
        queue=queue,
        worker_id=workers_ids["accounting"],
        service=accounting_service,
    ).run_once(now=now)
    assessment = risk_store.assessment(str(results["risk"].get("assessment_id", "")))
    fee_entry_id = str(accounting.get("record_id", ""))
    with database.engine.connect() as connection:
        accounting_entries = int(
            connection.execute(
                select(func.count()).select_from(fee_entry).where(fee_entry.c.id == fee_entry_id)
            ).scalar_one()
        )
        attributions = int(
            connection.execute(
                select(func.count())
                .select_from(trade_attribution)
                .where(trade_attribution.c.id == f"{fee_entry_id}:attribution")
            ).scalar_one()
        )
    counts = {
        "closed_candle": int(results["data"].get("reason_code") == "market_event_written"),
        "features": int(results["feature"].get("features", 0)),
        "strategy_assignment": int(assignments.by_id(assignment_id) is not None),
        "state": int(results["state"].get("reason_code") == "canonical_portfolio_state_published"),
        "alpha_forecast": len(repository.active_forecasts(product_id=product_id, at=now)),
        "target_position": len(
            repository.latest_targets(portfolio_id=str(product["portfolio_id"]))
        ),
        "risk_decision": len(assessment.decisions),
        "order_intent": len(order_manager.all()),
        "partial_fill": int(
            results["partial"].get("reason_code") == "paper_order_partially_filled"
        ),
        "complete_fill": int(results["complete"].get("reason_code") == "paper_order_filled"),
        "position": len(
            tuple(item for item in positions.all() if item.portfolio_id == product["portfolio_id"])
        ),
        "accounting_entry": accounting_entries,
        "attribution": attributions,
        "decision_trace": len(
            tuple(item for item in traces.read() if item[1].event_id.startswith(event.event_id))
        ),
    }
    ok = all(counts[stage] > 0 for stage in _STAGES) and counts["risk_decision"] == 6
    return {
        "product_id": product_id,
        "ok": ok,
        "first_blocked_stage": next((stage for stage in _STAGES if not counts[stage]), None),
        "counts": counts,
        "worker_results": results,
        "accounting_result": accounting,
    }


def _seed_strategy(
    database: PlatformDatabase,
    product: dict[str, Any],
    instrument: Instrument,
    universe_snapshot_id: str,
    now: str,
    prefix: str,
    strategy_name: str = "momentum_roc",
) -> str:
    product_id = str(product["product_id"])
    feature_nodes, production_rule = registered_live_contract(strategy_name)
    definition = StrategyDefinition(
        identity=prefix + ":" + strategy_name,
        version="smoke-v1",
        family="time_series",
        product=product_id,
        universe={"snapshot_id": universe_snapshot_id, "symbols": [instrument.exchange_symbol]},
        data_requirements={"bars": "1m", "closed_only": True},
        feature_graph={"version": "core-bars-v1", "required_nodes": list(feature_nodes)},
        signal_model={
            "registered_strategy": strategy_name,
            "parameters": {},
            "production_rule": production_rule,
        },
        position_model={"kind": "volatility_scaled"},
        execution_preferences={"policy": "market"},
        risk_policy={"id": product["risk_policy_id"]},
        validation_policy={"id": "smoke-validated"},
        source_type=StrategySourceType.REGISTERED_PYTHON,
        source_hash=canonical_hash({"source": "platform-smoke", "product": product_id}),
    )
    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_definition).values(
                id=definition.definition_hash,
                identity=definition.identity,
                product_id=product_id,
                source_type=definition.source_type.value,
                source_hash=definition.source_hash,
                definition=to_primitive(definition),
            )
        )
        connection.execute(
            insert(strategy_version).values(
                id=definition.strategy_version_id,
                definition_id=definition.definition_hash,
                version=definition.version,
                created_at=now,
                payload={"definition_hash": definition.definition_hash},
            )
        )
    artefact = StrategyArtefact(
        definition=definition,
        dependency_hash=canonical_hash({"dependencies": "smoke"}),
        dataset_snapshot_hashes=(canonical_hash({"dataset": prefix}),),
        feature_set_version="core-bars-v1",
        cost_model_version="smoke-costs-v1",
        validation_evidence={"accepted": True},
        holdout_claim={"accepted": True},
        forward_evidence={"accepted": True},
        promotion_policy={"paper": True},
        position_limits={"maximum_position": 0.1, "target_volatility": 0.1},
        risk_limits={"risk_policy_id": product["risk_policy_id"]},
        model_hashes=(),
        supported_products=(product_id,),
        supported_instruments=(instrument.instrument_id,),
        created_at=now,
        authoritative_evidence={"smoke": True},
        product_id=product_id,
        portfolio_id=str(product["portfolio_id"]),
        account_id=str(product["account_id"]),
        promotion_policy_id="smoke-paper",
        engine_version="strategy-evaluator/v1",
    )
    SqlStrategyArtefactRepository(database.engine).put(
        artefact.artefact_hash, artefact.to_dict(), created_at=now
    )
    return SqlActiveStrategyAssignmentRepository(database.engine).assign(
        product_id=product_id,
        portfolio_id=str(product["portfolio_id"]),
        strategy_version_id=definition.strategy_version_id,
        artefact_hash=artefact.artefact_hash,
        lifecycle_state="forward_paper",
        execution_mode="paper",
        capital_limit=0.1,
        risk_budget=0.1,
        assigned_at=now,
        assigned_by="platform-smoke",
        assignment_reason="real service-chain smoke",
        instrument_id=instrument.instrument_id,
        payload={"instrument_ids": [instrument.instrument_id]},
    )


def _install_risk_policy(database: PlatformDatabase, policy_id: str) -> None:
    SqlRiskPolicyStore(database.engine).save(
        policy_id,
        {
            "strategy": {
                "max_position_fraction": 1.0,
                "max_turnover_fraction": 2.0,
                "max_trades_per_day": 100,
                "max_slippage_bps": 50.0,
                "max_funding_cost_fraction": 0.1,
            },
            "instrument": {
                "max_position_notional": 100000.0,
                "max_order_notional": 100000.0,
                "max_visible_depth_fraction": 0.5,
                "max_spread_bps": 50.0,
                "max_volatility": 3.0,
                "max_concentration_fraction": 1.0,
            },
            "sleeve": {
                "max_capital_fraction": 1.0,
                "max_drawdown_fraction": 0.2,
                "max_correlation": 1.0,
                "max_abs_beta": 2.0,
                "max_turnover_fraction": 2.0,
            },
            "product": {
                "max_gross_fraction": 2.0,
                "max_net_fraction": 1.0,
                "max_drawdown_fraction": 0.2,
                "max_margin_fraction": 1.0,
                "max_daily_loss_fraction": 0.1,
            },
            "account": {
                "max_used_margin_fraction": 1.0,
                "min_liquidation_buffer_fraction": 0.2,
                "reject_unknown_exposure": True,
            },
            "global": {
                "max_drawdown_fraction": 0.2,
                "max_data_age_seconds": 5.0,
                "max_clock_skew_seconds": 1.0,
            },
        },
    )


def _smoke_state_policy(product: dict[str, Any], instrument: Instrument) -> dict[str, Any]:
    return {
        "maximum_state_age_seconds": 5,
        "product_drawdown_fraction": 0.0,
        "daily_pnl_fraction": 0.0,
        "global_drawdown_fraction": 0.0,
        "risk_policy_ids": [str(product["risk_policy_id"])],
        "portfolio_risk_budget": 0.5,
        "maximum_symbol_fraction": 0.2,
        "maximum_abs_beta": 1.0,
        "maximum_correlation": 0.8,
        "maximum_turnover_fraction": 1.0,
        "maximum_cluster_fraction": 1.0,
        "maximum_product_drawdown_fraction": 0.2,
        "maximum_depth_participation": 0.1,
        "sleeve_budgets": {"directional": 1.0},
        "clusters": {instrument.instrument_id: "btc"},
        "cluster_fraction_caps": {"btc": 1.0},
        "trades_today": 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    args = parser.parse_args(argv)
    report = run_smoke(args.database_url, config_path=args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
