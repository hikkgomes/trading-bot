"""Leased product coordination and target-position optimisation services."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from src.domain._codec import canonical_hash, timestamp, to_primitive
from src.domain.portfolios import TargetPosition
from src.execution.position_manager import PositionManager
from src.observability.decision_trace import (
    DecisionTrace,
    DecisionTraceStage,
    SqlDecisionTraceStore,
)
from src.portfolio.aggregation import aggregate_forecasts
from src.portfolio.optimiser import PortfolioConstraints, optimise_targets
from src.products.active_income import ActiveIncomePortfolio
from src.products.btc_accumulation import BtcAllocationPolicy, target_btc_allocation
from src.risk.engine import SqlRiskDecisionStore, SqlRiskSnapshotStore
from src.services.job_schemas import build_content_hash, validate_job_payload
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.scheduler import DatabaseJobQueue


@dataclass(frozen=True)
class TargetRiskReferences:
    """Canonical IDs produced after a preliminary target has been persisted."""

    target_position_snapshot_id: str
    account_snapshot_id: str
    positions_snapshot_id: str
    balances_snapshot_id: str
    market_data_snapshot_id: str
    risk_policy_ids: tuple[str, ...]


class DatabasePortfolioTargetBuilder:
    """Build one preliminary target from immutable forecast and account inputs."""

    def __init__(
        self,
        *,
        repository: SqlPortfolioRepository,
        snapshot_store: SqlRiskSnapshotStore,
        positions: PositionManager,
        product_configuration: Mapping[str, Mapping[str, Any]],
        account_configuration: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.repository = repository
        self.snapshot_store = snapshot_store
        self.positions = positions
        self.product_configuration = {
            str(key): dict(value) for key, value in product_configuration.items()
        }
        self.account_configuration = {
            str(key): dict(value) for key, value in account_configuration.items()
        }

    def __call__(self, payload: Mapping[str, Any]) -> TargetRiskReferences:
        product_id, product, evaluated_at, seed, forecasts = self._forecast_context(payload)
        state_id, state = self.snapshot_store.latest(
            kind="canonical_portfolio_risk_state", product_id=product_id, at=evaluated_at
        )
        clean = _canonical_portfolio_state(state, product_id=product_id)
        _assert_state_fresh(clean, evaluated_at)
        balances = {str(key): float(value) for key, value in clean["balances"].items()}
        current_positions = {str(key): float(value) for key, value in clean["positions"].items()}
        market = {str(key): dict(value) for key, value in clean["market"].items()}
        prices = _validated_prices(forecasts, market)
        targets, equity, current_positions = self._product_targets(
            product_id=product_id,
            product=product,
            product_family=str(product.get("product_family") or product_id),
            seed=seed,
            forecasts=forecasts,
            clean=clean,
            balances=balances,
            current_positions=current_positions,
            market=market,
            prices=prices,
            state_id=state_id,
        )
        if not targets:
            raise ValueError("portfolio optimiser produced no targets")

        event_id = str(payload["event_id"])
        target_ids = self.repository.save_targets(
            event_id=event_id, targets=targets, created_at=evaluated_at
        )
        target_scopes = _target_risk_scopes(
            targets=targets,
            target_ids=target_ids,
            current_positions=current_positions,
            equity=equity,
            market=market,
            clean=clean,
        )
        target_snapshot_id = self.snapshot_store.save(
            {
                "kind": "target_position",
                "product_id": product_id,
                "event_id": event_id,
                "canonical_state_id": state_id,
                "target_position_ids": list(target_ids),
                "targets": [to_primitive(item) for item in targets],
                "prices": prices,
                "balances": balances,
                "reconciled_positions": current_positions,
                "scopes": target_scopes,
            },
            created_at=evaluated_at,
        )
        (
            account_snapshot_id,
            positions_snapshot_id,
            balances_snapshot_id,
            market_data_snapshot_id,
        ) = self._save_risk_inputs(
            product_id=product_id,
            portfolio_id=str(product["portfolio_id"]),
            account_id=str(product["account_id"]),
            state_id=state_id,
            evaluated_at=evaluated_at,
            clean=clean,
            balances=balances,
            current_positions=current_positions,
            market=market,
        )
        return TargetRiskReferences(
            target_position_snapshot_id=target_snapshot_id,
            account_snapshot_id=account_snapshot_id,
            positions_snapshot_id=positions_snapshot_id,
            balances_snapshot_id=balances_snapshot_id,
            market_data_snapshot_id=market_data_snapshot_id,
            risk_policy_ids=tuple(str(item) for item in clean["risk_policy_ids"]),
        )

    def _forecast_context(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any], str, Any, tuple[Any, ...]]:
        product_id = str(payload["product_id"])
        product = self.product_configuration[product_id]
        evaluated_at = timestamp(str(payload["evaluated_at"]), field="evaluated_at")
        seed = self.repository.forecast(str(payload["forecast_id"]))
        if seed.product_id != product_id:
            raise ValueError("forecast product differs from target product")
        forecasts = self.repository.active_forecasts(product_id=product_id, at=evaluated_at)
        if seed not in forecasts:
            raise ValueError("trigger forecast is not active at the rebalance timestamp")
        return product_id, product, evaluated_at, seed, forecasts

    def _product_targets(
        self,
        *,
        product_id: str,
        product: Mapping[str, Any],
        product_family: str,
        seed: Any,
        forecasts: tuple[Any, ...],
        clean: Mapping[str, Any],
        balances: Mapping[str, float],
        current_positions: dict[str, float],
        market: Mapping[str, Mapping[str, Any]],
        prices: Mapping[str, float],
        state_id: str,
    ) -> tuple[tuple[TargetPosition, ...], float, dict[str, float]]:
        if product_family == "btc_accumulation":
            return self._btc_targets(
                product_id=product_id,
                product=product,
                seed=seed,
                forecasts=forecasts,
                clean=clean,
                balances=balances,
                current_positions=current_positions,
                prices=prices,
                state_id=state_id,
            )
        return self._active_income_targets(
            product=product,
            forecasts=forecasts,
            clean=clean,
            balances=balances,
            current_positions=current_positions,
            market=market,
            prices=prices,
        )

    def _btc_targets(
        self,
        *,
        product_id: str,
        product: Mapping[str, Any],
        seed: Any,
        forecasts: tuple[Any, ...],
        clean: Mapping[str, Any],
        balances: Mapping[str, float],
        current_positions: dict[str, float],
        prices: Mapping[str, float],
        state_id: str,
    ) -> tuple[tuple[TargetPosition, ...], float, dict[str, float]]:
        instrument_id = seed.instrument_id
        price = prices[instrument_id]
        current_positions[instrument_id] = float(balances.get("BTC", 0.0))
        equity_btc = balances.get("BTC", 0.0) + balances.get("USDT", 0.0) / price
        if equity_btc <= 0:
            raise ValueError("BTC accumulation equity must be positive")
        policy = BtcAllocationPolicy(
            core_btc_fraction=float(product.get("btc_core_fraction", 1.0)),
            max_tactical_fraction=float(product.get("btc_max_tactical_fraction", 0.0)),
        )
        allocation = target_btc_allocation(forecasts, policy=policy)
        quantity = max(0.0, allocation.target_btc_fraction * equity_btc)
        target = TargetPosition(
            portfolio_id=str(product["portfolio_id"]),
            instrument_id=instrument_id,
            target_quantity=quantity,
            target_notional=quantity * price,
            target_fraction=allocation.target_btc_fraction,
            strategy_contributions=allocation.contributions,
            risk_budget=float(clean["portfolio_risk_budget"]),
            valid_until=min(item.valid_until for item in forecasts),
            metadata={
                "product_id": product_id,
                "canonical_state_id": state_id,
                "assignment_id": seed.metadata.get("assignment_id"),
                "core_btc_fraction": allocation.core_btc_fraction,
                "tactical_btc_fraction": allocation.tactical_btc_fraction,
                "stablecoin_fraction": allocation.stablecoin_fraction,
                "quote_currency": "USDT",
                "actual_stablecoin_balance": balances.get("USDT", 0.0),
            },
        )
        return (target,), equity_btc * price, current_positions

    def _active_income_targets(
        self,
        *,
        product: Mapping[str, Any],
        forecasts: tuple[Any, ...],
        clean: Mapping[str, Any],
        balances: Mapping[str, float],
        current_positions: Mapping[str, float],
        market: Mapping[str, Mapping[str, Any]],
        prices: Mapping[str, float],
    ) -> tuple[tuple[TargetPosition, ...], float, dict[str, float]]:
        equity = balances.get("USDT", 0.0)
        if equity <= 0:
            raise ValueError("active-income USDT equity must be positive")
        account = self.account_configuration.get(str(product["account_id"]), {})
        constraints = PortfolioConstraints(
            portfolio_id=str(product["portfolio_id"]),
            equity=equity,
            max_positions=int(product.get("maximum_positions", 12)),
            max_gross_fraction=float(product.get("maximum_gross", 1.5)),
            max_net_fraction=float(product.get("maximum_net", 0.5)),
            max_symbol_fraction=float(clean["maximum_symbol_fraction"]),
            max_abs_beta=float(clean["maximum_abs_beta"]),
            max_correlation=float(clean["maximum_correlation"]),
            max_margin_fraction=float(product.get("maximum_margin", 0.5)),
            max_turnover_fraction=float(clean["maximum_turnover_fraction"]),
            max_cluster_fraction=float(clean["maximum_cluster_fraction"]),
            max_drawdown_fraction=float(clean["maximum_product_drawdown_fraction"]),
        )
        targets = optimise_targets(
            aggregate_forecasts(forecasts),
            prices=prices,
            valid_until=min(item.valid_until for item in forecasts),
            constraints=constraints,
            correlations=clean["correlations"],
            beta_by_instrument=clean["beta"],
            observed_volatility={key: float(value["volatility"]) for key, value in market.items()},
            liquidity_fraction_caps={
                key: min(
                    1.0,
                    float(value["visible_depth"])
                    * float(clean["maximum_depth_participation"])
                    / equity,
                )
                for key, value in market.items()
            },
            funding_rates={key: float(value["funding"]) for key, value in market.items()},
            current_quantities=current_positions,
            sleeve_budgets=clean["sleeve_budgets"],
            cluster_by_instrument=clean["clusters"],
            cluster_fraction_caps=clean["cluster_fraction_caps"],
            product_drawdown_fraction=float(clean["product_drawdown_fraction"]),
            available_margin_fraction=max(0.0, 1.0 - float(clean["used_margin_fraction"])),
            risk_budget=float(clean["portfolio_risk_budget"]),
            protective_stop_fraction=float(product.get("protective_stop_fraction", 0.02)),
            default_leverage=float(account.get("maximum_leverage", 1.0)),
            leverage_by_instrument={
                key: float(value.get("leverage", account.get("maximum_leverage", 1.0)))
                for key, value in market.items()
                if isinstance(value, Mapping) and value.get("leverage") is not None
            },
        )
        return tuple(targets), equity, dict(current_positions)

    def _save_risk_inputs(
        self,
        *,
        product_id: str,
        portfolio_id: str,
        account_id: str,
        state_id: str,
        evaluated_at: str,
        clean: Mapping[str, Any],
        balances: Mapping[str, float],
        current_positions: Mapping[str, float],
        market: Mapping[str, Mapping[str, Any]],
    ) -> tuple[str, str, str, str]:
        account_snapshot_id = self.snapshot_store.save(
            {
                "kind": "account_risk_input",
                "scope": "account",
                "product_id": product_id,
                "inputs": {
                    "used_margin_fraction": float(clean["used_margin_fraction"]),
                    "liquidation_buffer_fraction": float(clean["liquidation_buffer_fraction"]),
                    "unknown_positions": clean["unknown_exposure"],
                },
                "decision_id": canonical_hash({"scope": "account", "state_id": state_id}),
                "canonical_state_id": state_id,
            },
            created_at=evaluated_at,
        )
        positions_snapshot_id = self.snapshot_store.save(
            {
                "kind": "positions",
                "product_id": product_id,
                "portfolio_id": portfolio_id,
                "positions": dict(current_positions),
                "open_orders": clean["open_orders"],
                "canonical_state_id": state_id,
            },
            created_at=evaluated_at,
        )
        balances_snapshot_id = self.snapshot_store.save(
            {
                "kind": "balances",
                "product_id": product_id,
                "account_id": account_id,
                "balances": dict(balances),
                "canonical_state_id": state_id,
            },
            created_at=evaluated_at,
        )
        market_data_snapshot_id = self.snapshot_store.save(
            {
                "kind": "global_risk_input",
                "scope": "global",
                "product_id": product_id,
                "inputs": {
                    "drawdown_fraction": float(clean["global_drawdown_fraction"]),
                    "exchange_connected": bool(clean["exchange_connected"]),
                    "data_age_seconds": float(clean["data_age_seconds"]),
                    "clock_skew_seconds": float(clean["clock_skew_seconds"]),
                    "database_healthy": bool(clean["database_healthy"]),
                    "execution_drift": bool(clean["execution_drift"]),
                    "model_drift": bool(clean["model_drift"]),
                },
                "decision_id": canonical_hash({"scope": "global", "state_id": state_id}),
                "canonical_state_id": state_id,
                "market": dict(market),
            },
            created_at=evaluated_at,
        )
        return (
            account_snapshot_id,
            positions_snapshot_id,
            balances_snapshot_id,
            market_data_snapshot_id,
        )


def _assert_state_fresh(clean: Mapping[str, Any], evaluated_at: str) -> None:
    maximum_age = float(clean["maximum_state_age_seconds"])
    observed_at = dt.datetime.fromisoformat(str(clean["observed_at"]))
    evaluated = dt.datetime.fromisoformat(evaluated_at)
    age = (evaluated - observed_at).total_seconds()
    if age < 0 or age > maximum_age:
        raise ValueError("canonical portfolio/risk state is stale")


def _validated_prices(
    forecasts: tuple[Any, ...], market: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    missing = sorted({item.instrument_id for item in forecasts} - set(market))
    if missing:
        raise ValueError("canonical market state is missing: " + ", ".join(missing))
    prices = {key: float(value["price"]) for key, value in market.items()}
    if any(value <= 0 for value in prices.values()):
        raise ValueError("canonical market prices must be positive")
    return prices


def _target_risk_scopes(
    *,
    targets: tuple[TargetPosition, ...],
    target_ids: tuple[str, ...],
    current_positions: Mapping[str, float],
    equity: float,
    market: Mapping[str, Mapping[str, Any]],
    clean: Mapping[str, Any],
) -> dict[str, Any]:
    gross = sum(abs(item.target_fraction) for item in targets)
    net = sum(item.target_fraction for item in targets)
    turnover = sum(
        abs(item.target_quantity - current_positions.get(item.instrument_id, 0.0))
        * float(market[item.instrument_id]["price"])
        / equity
        for item in targets
    )
    worst_spread = max(float(market[item.instrument_id]["spread_bps"]) for item in targets)
    worst_volatility = max(float(market[item.instrument_id]["volatility"]) for item in targets)
    depth_fraction = max(
        abs(item.target_notional) / max(float(market[item.instrument_id]["visible_depth"]), 1e-12)
        for item in targets
    )
    maximum_position = max(abs(item.target_fraction) for item in targets)
    maximum_funding = max(abs(float(market[item.instrument_id]["funding"])) for item in targets)

    def identity(scope: str) -> str:
        return canonical_hash({"scope": scope, "target_ids": target_ids})

    return {
        "strategy": {
            "inputs": {
                "position_fraction": maximum_position,
                "turnover_fraction": turnover,
                "trades_today": int(clean["trades_today"]),
                "expected_slippage_bps": worst_spread,
                "expected_funding_cost_fraction": maximum_funding,
            },
            "decision_id": identity("strategy"),
        },
        "instrument": {
            "inputs": {
                "position_notional": max(abs(item.target_notional) for item in targets),
                "order_notional": max(abs(item.target_notional) for item in targets),
                "visible_depth_fraction": depth_fraction,
                "spread_bps": worst_spread,
                "volatility": worst_volatility,
                "concentration_fraction": maximum_position,
            },
            "decision_id": identity("instrument"),
        },
        "sleeve": {
            "inputs": {
                "capital_fraction": gross,
                "drawdown_fraction": float(clean["product_drawdown_fraction"]),
                "maximum_correlation": _maximum_correlation(clean["correlations"]),
                "beta": sum(
                    item.target_fraction * float(clean["beta"].get(item.instrument_id, 0.0))
                    for item in targets
                ),
                "turnover_fraction": turnover,
            },
            "decision_id": identity("sleeve"),
        },
        "product": {
            "inputs": {
                "gross_fraction": gross,
                "net_fraction": net,
                "drawdown_fraction": float(clean["product_drawdown_fraction"]),
                "margin_fraction": float(clean["used_margin_fraction"]),
                "daily_pnl_fraction": float(clean["daily_pnl_fraction"]),
            },
            "decision_id": identity("product"),
        },
    }


_PORTFOLIO_STATE_FIELDS = frozenset(
    {
        "kind",
        "product_id",
        "observed_at",
        "source_snapshot_ids",
        "maximum_state_age_seconds",
        "balances",
        "positions",
        "open_orders",
        "used_margin_fraction",
        "liquidation_buffer_fraction",
        "unknown_exposure",
        "market",
        "correlations",
        "beta",
        "risk_data_available",
        "risk_data_missing",
        "product_drawdown_fraction",
        "daily_pnl_fraction",
        "global_drawdown_fraction",
        "data_age_seconds",
        "clock_skew_seconds",
        "exchange_connected",
        "database_healthy",
        "execution_drift",
        "model_drift",
        "risk_policy_ids",
        "portfolio_risk_budget",
        "maximum_symbol_fraction",
        "maximum_abs_beta",
        "maximum_correlation",
        "maximum_turnover_fraction",
        "maximum_cluster_fraction",
        "maximum_product_drawdown_fraction",
        "maximum_depth_participation",
        "sleeve_budgets",
        "clusters",
        "cluster_fraction_caps",
        "trades_today",
    }
)


def _canonical_portfolio_state(state: Mapping[str, Any], *, product_id: str) -> dict[str, Any]:
    _validate_state_identity(state, product_id)
    clean = dict(state)
    clean["observed_at"] = timestamp(str(clean["observed_at"]), field="observed_at")
    _validate_state_shapes(clean)
    _validate_state_sources(clean)
    _validate_state_market(clean)
    _validate_state_health(clean)
    return clean


def _validate_state_identity(state: Mapping[str, Any], product_id: str) -> None:
    missing = sorted(_PORTFOLIO_STATE_FIELDS - set(state))
    if missing:
        raise ValueError("canonical portfolio/risk state is missing: " + ", ".join(missing))
    if state["kind"] != "canonical_portfolio_risk_state" or state["product_id"] != product_id:
        raise ValueError("canonical portfolio/risk state has the wrong identity")


def _validate_state_shapes(clean: Mapping[str, Any]) -> None:
    object_fields = (
        "balances",
        "positions",
        "unknown_exposure",
        "market",
        "correlations",
        "beta",
        "sleeve_budgets",
        "clusters",
        "cluster_fraction_caps",
    )
    if any(not isinstance(clean[field], Mapping) for field in object_fields):
        field = next(field for field in object_fields if not isinstance(clean[field], Mapping))
        raise ValueError(f"canonical portfolio/risk state {field} must be an object")
    if not isinstance(clean["open_orders"], list | tuple):
        raise ValueError("canonical portfolio/risk state open_orders must be a list")
    if not isinstance(clean["risk_data_available"], bool):
        raise ValueError("canonical portfolio/risk state risk_data_available must be boolean")
    missing = clean["risk_data_missing"]
    if not isinstance(missing, list | tuple) or any(
        not isinstance(value, str) or not value for value in missing
    ):
        raise ValueError("canonical portfolio/risk state risk_data_missing has invalid values")
    if not isinstance(clean["risk_policy_ids"], list | tuple) or not clean["risk_policy_ids"]:
        raise ValueError("canonical portfolio/risk state needs risk_policy_ids")


def _validate_state_sources(clean: Mapping[str, Any]) -> None:
    required = {"balances", "positions", "open_orders", "account", "market", "health", "drift"}
    source_ids = clean["source_snapshot_ids"]
    if (
        not isinstance(source_ids, Mapping)
        or set(source_ids) != required
        or any(
            not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
            for value in source_ids.values()
        )
    ):
        raise ValueError("canonical portfolio/risk state source identities are incomplete")


def _validate_state_market(clean: Mapping[str, Any]) -> None:
    required = {"price", "spread_bps", "visible_depth", "volatility"}
    for instrument_id, values in clean["market"].items():
        if not isinstance(values, Mapping) or not required.issubset(values):
            raise ValueError(f"canonical market state is incomplete for {instrument_id}")
        market_type = str(values.get("market_type") or "")
        if not market_type:
            market_type = "spot" if ":spot:" in str(instrument_id) else "futures"
        if market_type == "spot":
            continue
        elif market_type == "futures" and "funding" not in values:
            raise ValueError(f"canonical futures market state has no funding for {instrument_id}")
        elif market_type not in {"spot", "futures"}:
            raise ValueError(f"canonical market type is unsupported for {instrument_id}")


def _validate_state_health(clean: Mapping[str, Any]) -> None:
    if clean["risk_data_available"] is not True:
        missing = ", ".join(str(value) for value in clean["risk_data_missing"])
        raise ValueError(
            "canonical portfolio/risk state has unavailable factor measurements"
            + (f": {missing}" if missing else "")
        )
    if clean["unknown_exposure"]:
        raise ValueError("unknown exposure rejects new portfolio targets")
    if clean.get("account_state_known") is False:
        raise ValueError("account state authority is unknown")
    if not clean["exchange_connected"] or not clean["database_healthy"]:
        raise ValueError("exchange and database health are required for new exposure")
    if clean["execution_drift"] or clean["model_drift"]:
        raise ValueError("execution or model drift rejects new portfolio targets")


def _maximum_correlation(values: Mapping[str, Any]) -> float:
    correlations = [
        abs(float(value))
        for row in values.values()
        if isinstance(row, Mapping)
        for other, value in row.items()
        if other not in {"self", ""} and abs(float(value)) < 1.0 - 1e-12
    ]
    return max(correlations, default=0.0)


class DatabasePortfolioTargetWorker:
    """Turn an immutable forecast into a target, then submit snapshot IDs to risk."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        build_target: Callable[[Mapping[str, Any]], TargetRiskReferences],
        lease_seconds: int = 60,
        job_name: str = "portfolio_target_build",
        risk_job_name: str = "risk_assessment",
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.build_target = build_target
        self.lease_seconds = lease_seconds
        self.job_name = job_name
        self.risk_job_name = risk_job_name

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=(self.job_name,),
        )
        if claimed is None:
            return {"reason_code": "portfolio_target_queue_empty"}
        try:
            payload = validate_job_payload("portfolio_target_build", claimed.payload)
            references = self.build_target(payload)
            risk_payload = {
                "assessment_id": canonical_hash(
                    {
                        "event_id": payload["event_id"],
                        "product_id": payload["product_id"],
                        "forecast_id": payload["forecast_id"],
                        "target_position_snapshot_id": references.target_position_snapshot_id,
                    }
                ),
                "product_id": payload["product_id"],
                "event_id": payload["event_id"],
                "target_position_snapshot_id": references.target_position_snapshot_id,
                "account_snapshot_id": references.account_snapshot_id,
                "positions_snapshot_id": references.positions_snapshot_id,
                "balances_snapshot_id": references.balances_snapshot_id,
                "market_data_snapshot_id": references.market_data_snapshot_id,
                "risk_policy_ids": list(references.risk_policy_ids),
                "evaluated_at": payload["evaluated_at"],
                "producer_identity": self.worker_id,
            }
            risk_payload["content_hash"] = build_content_hash(risk_payload)
            risk_job_id = f"risk:{risk_payload['assessment_id'].removeprefix('sha256:')}"
            self.queue.enqueue_if_absent(
                job_id=risk_job_id,
                name=self.risk_job_name,
                payload=risk_payload,
                available_at=str(payload["evaluated_at"]),
                priority=18,
                producer_identity=self.worker_id,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "portfolio_target_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "risk_assessment_enqueued",
            "job_id": claimed.job_id,
            "risk_job_id": risk_job_id,
        }


class DatabaseProductCoordinator:
    """Move product triggers into the portfolio queue without doing execution work."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("active_income_cycle", "btc_accumulation_cycle"),
        )
        if claimed is None:
            return {"reason_code": "product_cycle_queue_empty"}
        try:
            name = (
                "active_income_portfolio"
                if claimed.name == "active_income_cycle"
                else "btc_accumulation_portfolio"
            )
            identity = canonical_hash(
                {"source_job_id": claimed.job_id, "name": name, "payload": claimed.payload}
            )
            job_id = f"portfolio:{identity.removeprefix('sha256:')}"
            self.queue.enqueue_if_absent(
                job_id=job_id,
                name=name,
                payload={**claimed.payload, "source_job_id": claimed.job_id},
                available_at=now,
                priority=10,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "product_coordination_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "portfolio_cycle_enqueued",
            "job_id": claimed.job_id,
            "portfolio_job_id": job_id,
        }


class DatabasePortfolioWorker:
    """Build durable targets, then hand them to the independent execution service."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        repository: SqlPortfolioRepository,
        positions: PositionManager,
        active_income: ActiveIncomePortfolio,
        risk_store: SqlRiskDecisionStore,
        trace_store: SqlDecisionTraceStore,
        execution_modes: Mapping[str, str],
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.repository = repository
        self.positions = positions
        self.active_income = active_income
        self.risk_store = risk_store
        self.trace_store = trace_store
        self.execution_modes = dict(execution_modes)
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("active_income_portfolio", "btc_accumulation_portfolio"),
        )
        if claimed is None:
            return {"reason_code": "portfolio_queue_empty"}
        try:
            self.positions.reload()
            product_id = (
                "active_income" if claimed.name == "active_income_portfolio" else "btc_accumulation"
            )
            payload = claimed.payload
            evaluated_at = timestamp(str(payload["evaluated_at"]), field="evaluated_at")
            assessment = self.risk_store.assessment(str(payload["risk_assessment_id"]))
            if assessment.aggregate.input_snapshot.get("product_id") != product_id:
                raise ValueError("risk assessment belongs to another product")
            targets, prices, reconciled_positions = self._targets(
                product_id=product_id,
                payload=payload,
                evaluated_at=evaluated_at,
            )
            event_id = str(payload["event_id"])
            target_ids = self.repository.save_targets(
                event_id=event_id,
                targets=targets,
                created_at=evaluated_at,
            )
            if not targets:
                reason = (
                    "no_actionable_forecast"
                    if not self.repository.active_forecasts(product_id=product_id, at=evaluated_at)
                    else "portfolio_no_target"
                )
                instrument_id = str(payload.get("instrument_id") or sorted(prices)[0])
                trace = (
                    DecisionTrace.start(event_id=event_id, instrument_id=instrument_id)
                    .pass_stage(DecisionTraceStage.DATA_AVAILABLE)
                    .pass_stage(DecisionTraceStage.FEATURE_AVAILABLE)
                    .pass_stage(DecisionTraceStage.STRATEGY_EVALUATED)
                    .pass_stage(DecisionTraceStage.REGIME_PASSED)
                    .pass_stage(DecisionTraceStage.SETUP_PASSED)
                    .pass_stage(DecisionTraceStage.TRIGGER_PASSED)
                )
                trace = (
                    trace.block(DecisionTraceStage.SIGNAL_PRODUCED, reason_code=reason)
                    if reason == "no_actionable_forecast"
                    else trace.pass_stage(DecisionTraceStage.SIGNAL_PRODUCED).block(
                        DecisionTraceStage.PORTFOLIO_ACCEPTED,
                        reason_code=reason,
                    )
                )
                self.trace_store.append(trace)
                self.queue.complete(claimed, completed_at=now)
                return {
                    "reason_code": reason,
                    "job_id": claimed.job_id,
                    "targets": 0,
                    "first_blocked_stage": trace.first_blocked_stage,
                }
            execution_payload = {
                "product_id": product_id,
                "event_id": event_id,
                "evaluated_at": evaluated_at,
                "risk_assessment_id": assessment.aggregate.decision_id,
                "execution_mode": self.execution_modes[product_id],
                "prices": prices,
                "reconciled_positions": reconciled_positions,
                "targets": [to_primitive(item) for item in targets],
            }
            identity = canonical_hash(execution_payload)
            execution_job_id = f"execution:{identity.removeprefix('sha256:')}"
            self.queue.enqueue_if_absent(
                job_id=execution_job_id,
                name="execute_targets",
                payload=execution_payload,
                available_at=evaluated_at,
                priority=20,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "portfolio_cycle_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "execution_cycle_enqueued",
            "job_id": claimed.job_id,
            "execution_job_id": execution_job_id,
            "targets": len(target_ids),
        }

    def _targets(
        self,
        *,
        product_id: str,
        payload: Mapping[str, Any],
        evaluated_at: str,
    ) -> tuple[tuple[TargetPosition, ...], dict[str, float], dict[str, float]]:
        forecasts = self.repository.active_forecasts(product_id=product_id, at=evaluated_at)
        if product_id == "active_income":
            prices = {str(key): float(value) for key, value in payload["prices"].items()}
            equity = float(payload["equity"])
            portfolio = ActiveIncomePortfolio(
                replace(self.active_income.constraints, equity=equity)
            )
            current = self.positions.current_quantities(portfolio.constraints.portfolio_id)
            targets = portfolio.target_positions(
                forecasts,
                prices=prices,
                valid_until=str(payload["valid_until"]),
                correlations=payload.get("correlations"),
                beta_by_instrument=payload.get("beta_by_instrument"),
                observed_volatility=payload.get("observed_volatility"),
                liquidity_fraction_caps=payload.get("liquidity_fraction_caps"),
                funding_rates=payload.get("funding_rates"),
                current_quantities=current,
                sleeve_budgets=payload.get("sleeve_budgets"),
                cluster_by_instrument=payload.get("cluster_by_instrument"),
                cluster_fraction_caps=payload.get("cluster_fraction_caps"),
                product_drawdown_fraction=float(payload.get("product_drawdown_fraction", 0.0)),
                available_margin_fraction=float(payload.get("available_margin_fraction", 1.0)),
            )
            return targets, prices, {}
        instrument_id = str(payload["instrument_id"])
        price = float(payload["stablecoin_per_btc"])
        btc_balance = float(payload["btc_balance"])
        stablecoin_balance = float(payload["stablecoin_balance"])
        if btc_balance < 0 or stablecoin_balance < 0 or price <= 0:
            raise ValueError("BTC balances must be non-negative and price positive")
        btc_policy = BtcAllocationPolicy()
        allocation = target_btc_allocation(forecasts, policy=btc_policy)
        btc_nav = btc_balance + stablecoin_balance / price
        target_quantity = btc_nav * allocation.target_btc_fraction
        target = TargetPosition(
            portfolio_id="btc-accumulation-portfolio",
            instrument_id=instrument_id,
            target_quantity=target_quantity,
            target_notional=target_quantity * price,
            target_fraction=allocation.target_btc_fraction,
            strategy_contributions=allocation.contributions
            or {"btc_allocation:core": allocation.core_btc_fraction},
            risk_budget=btc_policy.max_tactical_fraction,
            valid_until=str(payload["valid_until"]),
            metadata={
                "sleeve": "btc_tactical",
                "btc_nav_before_costs": btc_nav,
                "stablecoin_balance": stablecoin_balance,
                "stablecoin_per_btc": price,
            },
        )
        return (target,), {instrument_id: price}, {instrument_id: btc_balance}


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
