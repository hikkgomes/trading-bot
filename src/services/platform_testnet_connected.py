"""Confirmed, connected Binance testnet rehearsal for the new platform."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import os
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.autopilot.event_capture import load_event_capture_config
from src.data.database import (
    PlatformDatabase,
    accounting_entry,
    job,
    platform_rehearsal_report,
)
from src.data.feature_store import SqlFeatureStore
from src.domain._codec import canonical_hash
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.domain.orders import OrderIntent, OrderSide, OrderStatus, OrderType
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.observability.decision_trace import SqlDecisionTraceStore
from src.research.canonical import (
    SqlActiveStrategyAssignmentRepository,
    SqlStrategyArtefactRepository,
)
from src.risk.engine import SqlRiskDecisionStore, SqlRiskSnapshotStore
from src.services.account_reconciliation import AccountReconciliationService
from src.services.accounting_service import AccountingService, DatabaseAccountingWorker
from src.services.artefact_dispatcher import ArtefactDispatcher
from src.services.config import load_platform_config, load_split_configuration
from src.services.live_execution import ApprovedLiveExecution, execution_engine_identity
from src.services.market_gateway import DatabaseMarketGateway, UserStreamAccount
from src.services.order_execution import DatabaseLiveExecutionWorker, DatabaseUserStreamWorker
from src.services.portfolio_engine import (
    DatabasePortfolioTargetBuilder,
    DatabasePortfolioTargetWorker,
)
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.portfolio_state import (
    DatabasePortfolioSourceService,
    DatabasePortfolioStateWorker,
    portfolio_state_policies,
)
from src.services.risk_service import DatabaseRiskWorker
from src.services.runtime import utc_now
from src.services.scheduler import DatabaseJobQueue


class ConnectedTestnetError(RuntimeError):
    """The connected rehearsal cannot prove a safe real testnet round trip."""


_STATE_JOB_NAME = "connected_testnet_portfolio_state_publish"
_TARGET_JOB_NAME = "connected_testnet_portfolio_target_build"
_RISK_JOB_NAME = "connected_testnet_risk_assessment"
_LIVE_JOB_NAME = "connected_testnet_live_order_submit"
_USER_STREAM_JOB_NAME = "connected_testnet_user_stream_event"
_ACCOUNTING_JOB_NAME = "connected_testnet_accounting_event"


def validate_connected_testnet_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    if str(configuration.get("environment") or "") != "testnet":
        raise ConnectedTestnetError("connected rehearsal requires environment=testnet")
    if str(configuration.get("queue_backend") or "postgresql") != "postgresql":
        raise ConnectedTestnetError("connected rehearsal requires the PostgreSQL platform queue")
    if configuration.get("legacy_autopilot") is True:
        raise ConnectedTestnetError("connected rehearsal cannot use the legacy autopilot path")
    if (
        configuration.get("injected_broker") is not None
        or configuration.get("captured_events") is True
    ):
        raise ConnectedTestnetError(
            "connected rehearsal cannot use an injected broker or captured events"
        )
    product_id = str(configuration.get("product_id") or "")
    if product_id != "active_income":
        raise ConnectedTestnetError("connected rehearsal is restricted to active_income")
    notional = float(configuration.get("notional_usd", 10.0))
    if notional <= 0 or notional > 100.0:
        raise ConnectedTestnetError("connected rehearsal notional must be in (0, 100]")
    return {
        "schema": "platform.connected-testnet/v1",
        "environment": "testnet",
        "queue_backend": "postgresql",
        "product_id": product_id,
        "notional_usd": notional,
        "real_exchange": True,
        "user_stream_required": True,
    }


def run_connected_testnet_rehearsal(
    *,
    database_url: str,
    config_path: Path,
    product_id: str,
    notional_usd: float,
    confirm: bool,
) -> dict[str, Any]:
    validate_connected_testnet_configuration(
        {
            "environment": "testnet",
            "queue_backend": "postgresql",
            "product_id": product_id,
            "notional_usd": notional_usd,
        }
    )
    if not confirm:
        raise ConnectedTestnetError("set CONFIRM=1 to place real testnet orders")
    if os.environ.get("TRADING_LIVE") != "1" or os.environ.get("EXCHANGE_TESTNET") != "1":
        raise ConnectedTestnetError(
            "connected rehearsal requires TRADING_LIVE=1 and EXCHANGE_TESTNET=1"
        )
    signing_key = os.environ.get("TRADING_PLATFORM_REHEARSAL_SIGNING_KEY", "")
    if not signing_key:
        raise ConnectedTestnetError("TRADING_PLATFORM_REHEARSAL_SIGNING_KEY is required")
    split = load_split_configuration(config_path.parent)
    products = {str(item["product_id"]): dict(item) for item in split["products"]["products"]}
    accounts = {str(item["account_id"]): dict(item) for item in split["accounts"]["accounts"]}
    product = products.get(product_id)
    if product is None or product.get("execution_mode") != "live":
        raise ConnectedTestnetError("the selected product must be configured live for rehearsal")
    account = accounts[str(product["account_id"])]
    if account.get("environment") != "testnet":
        raise ConnectedTestnetError("the selected account must be a Binance testnet account")
    database = PlatformDatabase(database_url)
    if not database.is_postgresql:
        raise ConnectedTestnetError("connected rehearsal requires PostgreSQL")
    database.assert_migrated()
    now = utc_now()
    queue = DatabaseJobQueue(database.engine)
    order_manager = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    assignments = SqlActiveStrategyAssignmentRepository(database.engine)
    assignment = assignments.active(product_id, execution_mode="live", at=now)
    if assignment is None or assignment["execution_mode"] != "live":
        raise ConnectedTestnetError("no immutable live assignment is available")
    try:
        initial_account = AccountReconciliationService(
            engine=database.engine,
            products={product_id: product},
            accounts=accounts,
        ).reconcile_once(now=now)
        if any(
            item.get("unknown_exposure")
            or item.get("positions")
            or item.get("regular_orders")
            or item.get("conditional_orders")
            for item in initial_account["accounts"]
        ):
            raise ConnectedTestnetError(
                "testnet account must be flat and order-free before rehearsal"
            )
        account_detail = initial_account["accounts"][0]
        state = _refresh_reconciled_state(
            database=database,
            queue=queue,
            split=split,
            product_id=product_id,
            product=product,
            accounts=accounts,
            reconciled_at=now,
            account_fingerprint=str(account_detail["account_fingerprint"]),
        )
        if not state:
            raise ConnectedTestnetError("no reconciled canonical portfolio/risk state is available")
        approved_live = ApprovedLiveExecution(
            engine=database.engine,
            configuration=split,
            order_manager=order_manager,
            positions=positions,
        )
        venue = approved_live.venues[product_id]
        instrument_id = str(assignment.get("instrument_id") or "")
        instrument = venue.instruments.get(instrument_id)
        if instrument is None:
            raise ConnectedTestnetError("live assignment instrument is not persisted")
        broker = venue.broker
        price = float(broker.get_price(instrument.exchange_symbol))
        if price <= 0:
            raise ConnectedTestnetError("testnet price is not positive")
        pipeline = _prepare_rehearsal_pipeline(
            database=database,
            queue=queue,
            product=product,
            account=account,
            assignment=assignment,
            instrument_id=instrument_id,
            now=now,
        )
        target_quantity = float(pipeline["target_quantity"])
        quantity = min(abs(target_quantity), notional_usd / price)
        if quantity <= 0:
            raise ConnectedTestnetError("canonical portfolio target has no actionable quantity")
        opening_side = OrderSide.BUY if target_quantity > 0 else OrderSide.SELL
        with database.engine.connect() as connection:
            accounting_before = int(
                connection.execute(
                    select(func.count())
                    .select_from(accounting_entry)
                    .where(accounting_entry.c.product_id == product_id)
                ).scalar_one()
            )
        gateway = _connected_gateway(
            database=database,
            config_path=config_path,
            account_payload=account,
        )
        user_stream_worker, accounting_worker = _event_workers(
            database=database,
            queue=queue,
            product=product,
            account_id=str(account["account_id"]),
            order_manager=order_manager,
            positions=positions,
        )
        live_worker = _live_worker(
            database=database,
            queue=queue,
            worker_id=f"connected-testnet:live:{account['account_id']}",
            product=product,
            venue=venue,
            order_manager=order_manager,
            positions=positions,
            approved_live=approved_live,
        )
        gateway.start()
        try:
            if not gateway.wait_for_user_stream(
                str(account["account_id"]),
                timeout_seconds=min(
                    30.0,
                    float(os.environ.get("PLATFORM_TESTNET_TIMEOUT_SECONDS", "120")),
                ),
            ):
                raise ConnectedTestnetError(
                    "Binance testnet user stream did not connect before order submission"
                )
            opened = _submit_and_wait(
                order_manager=order_manager,
                product=product,
                instrument_id=instrument_id,
                quantity=quantity,
                side=opening_side,
                now=now,
                queue=queue,
                account_id=str(account["account_id"]),
                live_worker=live_worker,
                user_stream_worker=user_stream_worker,
                accounting_worker=accounting_worker,
                strategy_version_id=str(assignment["strategy_version_id"]),
                artefact_hash=str(assignment["artefact_hash"]),
            )
            open_quantity = float(order_manager.get(opened["order_id"]).filled_quantity)
            if open_quantity <= 0:
                raise ConnectedTestnetError("testnet open order was not filled")
            open_recovery = _verify_recovery_lookup(
                venue=venue,
                symbol=instrument.exchange_symbol,
                submission=opened,
            )
            open_reconciliation = approved_live.reconcile(product_id)
            if not open_reconciliation.matched:
                raise ConnectedTestnetError("testnet account did not reconcile after open fill")
            mid_account = AccountReconciliationService(
                engine=database.engine,
                products={product_id: product},
                accounts=accounts,
            ).reconcile_once(now=utc_now())
            if any(item.get("unknown_exposure") for item in mid_account["accounts"]):
                raise ConnectedTestnetError(
                    "testnet account reconciliation found unknown exposure after open"
                )
            closed = _submit_and_wait(
                order_manager=order_manager,
                product=product,
                instrument_id=instrument_id,
                quantity=open_quantity,
                side=OrderSide.SELL if opening_side is OrderSide.BUY else OrderSide.BUY,
                now=utc_now(),
                queue=queue,
                account_id=str(account["account_id"]),
                reduce_only=True,
                live_worker=live_worker,
                user_stream_worker=user_stream_worker,
                accounting_worker=accounting_worker,
                strategy_version_id=str(assignment["strategy_version_id"]),
                artefact_hash=str(assignment["artefact_hash"]),
            )
            if order_manager.get(closed["order_id"]).status is not OrderStatus.FILLED:
                raise ConnectedTestnetError("testnet close order was not filled")
            close_recovery = _verify_recovery_lookup(
                venue=venue,
                symbol=instrument.exchange_symbol,
                submission=closed,
            )
        finally:
            gateway.stop()
        reconciliation = approved_live.reconcile(product_id)
        if not reconciliation.matched:
            raise ConnectedTestnetError("testnet account did not reconcile flat after close")
        final_account = AccountReconciliationService(
            engine=database.engine,
            products={product_id: product},
            accounts=accounts,
        ).reconcile_once(now=utc_now())
        final_details = final_account["accounts"]
        if any(
            item["unknown_exposure"]
            or item["positions"]
            or item["regular_orders"]
            or item["conditional_orders"]
            for item in final_details
        ):
            raise ConnectedTestnetError("testnet account is not flat and order-free after close")
        with database.engine.connect() as connection:
            accounting_after = int(
                connection.execute(
                    select(func.count())
                    .select_from(accounting_entry)
                    .where(accounting_entry.c.product_id == product_id)
                ).scalar_one()
            )
        if accounting_after <= accounting_before:
            raise ConnectedTestnetError("testnet fills did not produce accounting entries")
        result = {
            "schema": "platform.connected-testnet-report/v1",
            "environment": "testnet",
            "real_exchange": True,
            "product_id": product_id,
            "account_id": str(account["account_id"]),
            "assignment_id": str(assignment["id"]),
            "artefact_hash": str(assignment["artefact_hash"]),
            "forecast_id": pipeline["forecast_id"],
            "target_position_snapshot_id": pipeline["target_position_snapshot_id"],
            "risk_assessment_id": pipeline["risk_assessment_id"],
            "risk_scopes": pipeline["risk_scopes"],
            "risk_accepted": pipeline["risk_accepted"],
            "target_quantity": target_quantity,
            "opening_side": opening_side.value,
            "requested_notional_usd": notional_usd,
            "execution_engine_identity": _execution_engine_identity(),
            "account_fingerprint": str(getattr(broker, "account_fingerprint", "")),
            "open_order_id": opened["order_id"],
            "open_exchange_order_id": opened["exchange_order_id"],
            "close_order_id": closed["order_id"],
            "close_exchange_order_id": closed["exchange_order_id"],
            "open_acknowledged": True,
            "close_acknowledged": True,
            "user_stream_fill": True,
            "accounting_reconciled": accounting_after > accounting_before,
            "recovery_identifiers": {
                "open": open_recovery,
                "close": close_recovery,
            },
            "recovery_lookup": True,
            "open_reconciliation": {
                "matched": open_reconciliation.matched,
                "recovery_required": open_reconciliation.recovery_required,
            },
            "flat_reconciliation": True,
            "final_account": final_account,
        }
        report_hash = canonical_hash(result)
        signature = hmac.new(signing_key.encode(), report_hash.encode(), hashlib.sha256).hexdigest()
        report = {**result, "report_hash": report_hash, "signature": signature}
        with database.engine.begin() as connection:
            existing = connection.execute(
                select(platform_rehearsal_report.c.payload).where(
                    platform_rehearsal_report.c.id == report_hash
                )
            ).scalar_one_or_none()
            if existing is None:
                connection.execute(
                    insert(platform_rehearsal_report).values(
                        id=report_hash,
                        product_id=product_id,
                        account_id=str(account["account_id"]),
                        created_at=utc_now(),
                        content_hash=report_hash,
                        accepted=True,
                        payload=report,
                    )
                )
            elif existing != report:
                raise ConnectedTestnetError("testnet report identity collision")
        return report
    finally:
        database.dispose()


def _refresh_reconciled_state(
    *,
    database: PlatformDatabase,
    queue: DatabaseJobQueue,
    split: Mapping[str, Mapping[str, Any]],
    product_id: str,
    product: Mapping[str, Any],
    accounts: Mapping[str, Mapping[str, Any]],
    reconciled_at: str,
    account_fingerprint: str,
) -> dict[str, Any]:
    store = SqlRiskSnapshotStore(database.engine)
    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=store,
        products={product_id: product},
        accounts=accounts,
    )
    worker_id = "connected-testnet:portfolio-state"
    queue.register_worker(
        worker_id=worker_id,
        node_id="linux-optiplex",
        role="connected-testnet-portfolio-state",
        capabilities=(_STATE_JOB_NAME,),
        observed_at=reconciled_at,
    )
    worker = DatabasePortfolioStateWorker(
        queue=queue,
        worker_id=worker_id,
        store=store,
        refresh_sources=source.refresh,
        job_name=_STATE_JOB_NAME,
        job_id_prefix="connected-testnet-state",
    )
    policies = portfolio_state_policies(
        {"risk": split["risk"]},
        {product_id: product},
    )
    if (
        worker.schedule_from_latest(
            products={product_id: product},
            state_policies=policies,
            now=reconciled_at,
        )
        != 1
    ):
        raise ConnectedTestnetError(
            "connected rehearsal could not schedule a reconciled canonical state"
        )
    result = worker.run_once(now=reconciled_at)
    if result.get("reason_code") != "canonical_portfolio_state_published":
        raise ConnectedTestnetError(
            f"connected rehearsal state refresh failed: {result.get('reason_code')}"
        )
    state = store.get(str(result["state_id"]))
    source_ids = state.get("source_snapshot_ids")
    if not isinstance(source_ids, Mapping):
        raise ConnectedTestnetError("reconciled canonical state has no source identities")
    account_source = store.get(str(source_ids.get("account") or ""))
    values = account_source.get("values")
    if (
        not isinstance(values, Mapping)
        or values.get("account_state_known") is not True
        or values.get("account_state_authority") != "authenticated_rest"
        or values.get("account_fingerprint") != account_fingerprint
        or account_source.get("observed_at") != reconciled_at
    ):
        raise ConnectedTestnetError(
            "canonical state is not bound to the current authenticated reconciliation"
        )
    return dict(state)


def _prepare_rehearsal_pipeline(
    *,
    database: PlatformDatabase,
    queue: DatabaseJobQueue,
    product: Mapping[str, Any],
    account: Mapping[str, Any],
    assignment: Mapping[str, Any],
    instrument_id: str,
    now: str,
) -> dict[str, Any]:
    """Create and validate a real forecast, target, and six-scope risk decision."""
    product_id = str(product["product_id"])
    try:
        artefact = SqlStrategyArtefactRepository(database.engine).get(
            str(assignment["artefact_hash"])
        )
    except (KeyError, ValueError) as exc:
        raise ConnectedTestnetError("live rehearsal artefact is missing or invalid") from exc
    if not isinstance(artefact, Mapping):
        raise ConnectedTestnetError("live rehearsal artefact is missing")
    if (
        artefact.get("product_id") != product_id
        or artefact.get("account_id") != account.get("account_id")
        or artefact.get("portfolio_id") != product.get("portfolio_id")
    ):
        raise ConnectedTestnetError("live rehearsal artefact is bound to the wrong account")
    definition = artefact.get("definition")
    graph = definition.get("feature_graph") if isinstance(definition, Mapping) else None
    required_nodes = graph.get("required_nodes", ()) if isinstance(graph, Mapping) else ()
    required_names = tuple(
        str(item["name"] if isinstance(item, Mapping) else item) for item in required_nodes
    )
    feature_version = str(artefact.get("feature_set_version") or "")
    available = SqlFeatureStore(database.engine).available(
        instrument_id=instrument_id,
        at=now,
        feature_set_version=feature_version,
    )
    latest: dict[str, Any] = {}
    for value in available:
        if (
            value.feature_name not in latest
            or value.availability_time > latest[value.feature_name].availability_time
        ):
            latest[value.feature_name] = value
    missing = sorted(set(required_names) - set(latest))
    if missing:
        raise ConnectedTestnetError(
            "connected rehearsal requires fresh canonical features: " + ", ".join(missing)
        )
    features = {name: float(latest[name].value) for name in sorted(latest)}
    forecast_values = dict(ArtefactDispatcher.default().evaluate(features, artefact))
    direction = ForecastDirection(str(forecast_values["direction"]))
    if direction is ForecastDirection.FLAT or float(forecast_values["maximum_position"]) <= 0:
        raise ConnectedTestnetError("live artefact produced no actionable rehearsal forecast")
    valid_until = (
        (dt.datetime.fromisoformat(now) + dt.timedelta(seconds=300))
        .replace(microsecond=0)
        .isoformat()
    )
    event_id = canonical_hash(
        {
            "kind": "connected-testnet-rehearsal",
            "product_id": product_id,
            "assignment_id": str(assignment["id"]),
            "evaluated_at": now,
        }
    )
    forecast = AlphaForecast(
        strategy_version_id=str(assignment["strategy_version_id"]),
        product_id=product_id,
        instrument_id=instrument_id,
        direction=direction,
        score=float(forecast_values["score"]),
        expected_return=float(forecast_values["expected_return"]),
        confidence=float(forecast_values["confidence"]),
        horizon_seconds=300,
        valid_from=now,
        valid_until=valid_until,
        target_volatility=float(forecast_values["target_volatility"]),
        maximum_position=float(forecast_values["maximum_position"]),
        metadata={
            "market_event_id": event_id,
            "feature_ids": [latest[name].feature_id for name in sorted(latest)],
            "artefact_hash": str(assignment["artefact_hash"]),
            "engine_version": "connected-testnet-rehearsal/v1",
            "execution_receipt": forecast_values.get("execution_receipt", {}),
        },
    )
    forecast_id = SqlPortfolioRepository(
        database.engine, require_pipeline_identity=True
    ).save_forecast(forecast)
    target_payload = {
        "event_id": event_id,
        "product_id": product_id,
        "forecast_id": forecast_id,
        "evaluated_at": now,
        "producer_identity": "connected-testnet-rehearsal",
    }
    target_payload["content_hash"] = canonical_hash(target_payload)
    target_job_id = f"connected-testnet:target:{forecast_id.removeprefix('sha256:')}"
    queue.enqueue_if_absent(
        job_id=target_job_id,
        name=_TARGET_JOB_NAME,
        payload=target_payload,
        available_at=now,
        priority=100,
        producer_identity="connected-testnet-rehearsal",
    )
    repository = SqlPortfolioRepository(database.engine, require_pipeline_identity=True)
    target_worker_id = "connected-testnet:portfolio-target"
    queue.register_worker(
        worker_id=target_worker_id,
        node_id="linux-optiplex",
        role="connected-testnet-portfolio-target",
        capabilities=("portfolio_target_build",),
        observed_at=now,
    )
    target_worker = DatabasePortfolioTargetWorker(
        queue=queue,
        worker_id=target_worker_id,
        build_target=DatabasePortfolioTargetBuilder(
            repository=repository,
            snapshot_store=SqlRiskSnapshotStore(database.engine),
            positions=PositionManager(SqlPositionStore(database.engine)),
            product_configuration={product_id: dict(product)},
            account_configuration={str(account["account_id"]): dict(account)},
        ),
        job_name=_TARGET_JOB_NAME,
        risk_job_name=_RISK_JOB_NAME,
    )
    target_result = target_worker.run_once(now=now)
    if target_result.get("reason_code") != "risk_assessment_enqueued":
        raise ConnectedTestnetError(
            f"connected rehearsal target pipeline failed: {target_result.get('reason_code')}"
        )
    risk_worker_id = "connected-testnet:risk"
    queue.register_worker(
        worker_id=risk_worker_id,
        node_id="linux-optiplex",
        role="connected-testnet-risk",
        capabilities=("risk_assessment",),
        observed_at=now,
    )
    risk_worker = DatabaseRiskWorker(
        queue=queue,
        worker_id=risk_worker_id,
        store=SqlRiskDecisionStore(database.engine),
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        job_name=_RISK_JOB_NAME,
    )
    risk_result = risk_worker.run_once(now=now)
    if risk_result.get("reason_code") != "risk_assessment_accepted":
        raise ConnectedTestnetError(
            f"connected rehearsal risk pipeline failed: {risk_result.get('reason_code')}"
        )
    risk_job_id = str(target_result["risk_job_id"])
    with database.engine.connect() as connection:
        risk_payload = connection.execute(
            select(job.c.payload).where(job.c.id == risk_job_id)
        ).scalar_one()
    risk_scopes = (
        SqlRiskDecisionStore(database.engine)
        .assessment(str(risk_result["assessment_id"]))
        .decisions
    )
    if tuple(item.scope for item in risk_scopes) != (
        "strategy",
        "instrument",
        "sleeve",
        "product",
        "account",
        "global",
    ):
        raise ConnectedTestnetError("connected rehearsal risk scopes are incomplete")
    target_snapshot_id = str(risk_payload["target_position_snapshot_id"])
    target_snapshot = SqlRiskSnapshotStore(database.engine).get(target_snapshot_id)
    targets = target_snapshot.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ConnectedTestnetError("connected rehearsal target snapshot has no targets")
    target_quantity = float(targets[0]["target_quantity"])
    return {
        "event_id": event_id,
        "forecast_id": forecast_id,
        "target_position_snapshot_id": target_snapshot_id,
        "risk_assessment_id": str(risk_result["assessment_id"]),
        "risk_scopes": [item.scope for item in risk_scopes],
        "risk_accepted": True,
        "target_quantity": target_quantity,
    }


def _connected_gateway(
    *, database: PlatformDatabase, config_path: Path, account_payload: Mapping[str, Any]
):
    account = UserStreamAccount.from_config(account_payload)
    if account is None:
        raise ConnectedTestnetError("testnet user stream credentials are missing")
    capture_config = load_event_capture_config(config_path.parent / "event_capture.json")
    testnet_urls = {
        "spot": "wss://stream.testnet.binance.vision/stream",
        "futures": "wss://demo-fstream.binance.com/stream",
    }
    capture_config = replace(
        capture_config,
        sources=tuple(
            replace(source, url=testnet_urls[source.market]) for source in capture_config.sources
        ),
    )
    return DatabaseMarketGateway(
        queue=DatabaseJobQueue(database.engine),
        capture_config=capture_config,
        accounts=(account,),
        testnet=True,
        user_stream_job_name=_USER_STREAM_JOB_NAME,
        user_stream_job_prefix="connected-testnet-user-stream",
    )


def _submit_and_wait(
    *,
    order_manager: OrderManager,
    product: Mapping[str, Any],
    instrument_id: str,
    quantity: float,
    side: OrderSide,
    now: str,
    queue: DatabaseJobQueue,
    account_id: str,
    live_worker: DatabaseLiveExecutionWorker,
    reduce_only: bool = False,
    user_stream_worker,
    accounting_worker,
    strategy_version_id: str,
    artefact_hash: str,
) -> dict[str, Any]:
    order = OrderIntent(
        order_id=canonical_hash(
            {
                "rehearsal": "connected-testnet",
                "side": side.value,
                "quantity": quantity,
                "at": now,
            }
        ),
        portfolio_id=str(product["portfolio_id"]),
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        created_at=now,
        reduce_only=reduce_only,
        strategy_contributions={strategy_version_id: 1.0},
        metadata={"connected_testnet_rehearsal": True, "account_id": account_id},
    )
    order_manager.create(order)
    order_manager.persist_for_submission(order.order_id)
    job_id = f"connected-testnet:submit:{order.order_id}"
    queue.enqueue_if_absent(
        job_id=job_id,
        name=_LIVE_JOB_NAME,
        payload={
            "order_id": order.order_id,
            "product_id": str(product["product_id"]),
            "strategy_version_id": strategy_version_id,
            "artefact_hash": artefact_hash,
            "account_id": account_id,
        },
        available_at=now,
        priority=100,
    )
    submission = live_worker.run_once(now=now)
    if submission.get("reason_code") != "live_order_acknowledged":
        raise ConnectedTestnetError(
            f"durable live submission failed: {submission.get('reason_code')}"
        )
    deadline = time.monotonic() + float(os.environ.get("PLATFORM_TESTNET_TIMEOUT_SECONDS", "120"))
    while time.monotonic() < deadline:
        user_stream_worker.run_once(now=utc_now())
        accounting_worker.run_once(now=utc_now())
        order_manager.reload()
        current = order_manager.get(order.order_id)
        if current.status is OrderStatus.FILLED:
            return {
                "order_id": order.order_id,
                "exchange_order_id": submission["exchange_order_id"],
                "client_order_id": submission["client_order_id"],
                "status": current.status.value,
            }
        time.sleep(1.0)
    raise ConnectedTestnetError("no user-stream fill was received before timeout")


def _verify_recovery_lookup(*, venue, symbol: str, submission: Mapping[str, Any]) -> dict[str, str]:
    """Prove the exchange/client identity lookup used by live recovery."""
    query = getattr(venue.broker, "query_order", None)
    if not callable(query):
        raise ConnectedTestnetError("connected broker has no exchange order recovery lookup")
    exchange_order_id = str(submission.get("exchange_order_id") or "")
    client_order_id = str(submission.get("client_order_id") or "")
    if not exchange_order_id or not client_order_id:
        raise ConnectedTestnetError("testnet acknowledgement has incomplete recovery identity")
    exchange_state = query(
        symbol=symbol,
        exchange_order_id=exchange_order_id,
        client_order_id=client_order_id,
    )
    if str(getattr(exchange_state, "exchange_order_id", "")) != exchange_order_id:
        raise ConnectedTestnetError("exchange recovery lookup returned the wrong order ID")
    client_state = query(
        symbol=symbol,
        exchange_order_id="",
        client_order_id=client_order_id,
    )
    if str(getattr(client_state, "exchange_order_id", "")) != exchange_order_id:
        raise ConnectedTestnetError("client recovery lookup returned the wrong exchange order ID")
    if str(getattr(client_state, "client_order_id", "")) != client_order_id:
        raise ConnectedTestnetError("client recovery lookup returned the wrong order ID")
    return {
        "exchange_order_id": exchange_order_id,
        "client_order_id": client_order_id,
    }


def _live_worker(
    *, database, queue, worker_id, product, venue, order_manager, positions, approved_live
) -> DatabaseLiveExecutionWorker:
    queue.register_worker(
        worker_id=worker_id,
        node_id="linux-optiplex",
        role="connected-testnet-live",
        capabilities=("live_order_submit",),
        observed_at=utc_now(),
    )
    ledger = Ledger(
        product_id=str(product["product_id"]),
        accounting_asset=str(product["base_accounting_asset"]),
        store=SqlLedgerStore(database.engine, product_id=str(product["product_id"])),
    )
    return DatabaseLiveExecutionWorker(
        queue=queue,
        worker_id=worker_id,
        order_manager=order_manager,
        positions=positions,
        ledgers={str(product["product_id"]): ledger},
        trace_store=SqlDecisionTraceStore(database.engine),
        venues={str(product["product_id"]): venue},
        authorise=approved_live.authorise,
        job_name=_LIVE_JOB_NAME,
    )


def _event_workers(*, database, queue, product, account_id, order_manager, positions):
    worker_id = f"connected-testnet:{account_id}"
    queue.register_worker(
        worker_id=worker_id,
        node_id="linux-optiplex",
        role="connected-testnet",
        capabilities=("user_stream_event", "accounting_event"),
        observed_at=utc_now(),
    )
    trace_store = SqlDecisionTraceStore(database.engine)
    ledgers = {
        str(product["product_id"]): Ledger(
            product_id=str(product["product_id"]),
            accounting_asset=str(product["base_accounting_asset"]),
            store=SqlLedgerStore(database.engine, product_id=str(product["product_id"])),
        )
    }
    user_worker = DatabaseUserStreamWorker(
        engine=database.engine,
        queue=queue,
        worker_id=worker_id,
        order_manager=order_manager,
        positions=positions,
        ledgers=ledgers,
        trace_store=trace_store,
        account_products={account_id: str(product["product_id"])},
        job_name=_USER_STREAM_JOB_NAME,
        accounting_job_name=_ACCOUNTING_JOB_NAME,
        accounting_job_prefix="connected-testnet-accounting",
    )
    accounting_worker = DatabaseAccountingWorker(
        queue=queue,
        worker_id=worker_id,
        service=AccountingService(
            engine=database.engine,
            ledgers=ledgers,
        ),
        job_name=_ACCOUNTING_JOB_NAME,
    )
    return user_worker, accounting_worker


def _execution_engine_identity() -> str:
    return execution_engine_identity()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the connected Binance testnet rehearsal.")
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    parser.add_argument("--product", default="active_income")
    parser.add_argument("--notional-usd", type=float, default=10.0)
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_platform_config(args.config)
    report = run_connected_testnet_rehearsal(
        database_url=config.database_url(),
        config_path=args.config,
        product_id=args.product,
        notional_usd=args.notional_usd,
        confirm=args.confirm,
    )
    print(report)


if __name__ == "__main__":
    main()
