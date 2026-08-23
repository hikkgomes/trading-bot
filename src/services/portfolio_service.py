"""PostgreSQL forecast and target-position service boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import alpha_forecast as alpha_forecast_table
from src.data.database import target_position as target_position_table
from src.domain._codec import canonical_hash, timestamp, to_primitive
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.domain.portfolios import TargetPosition
from src.risk.engine import HierarchicalRiskAssessment, SqlRiskDecisionStore
from src.services.product_supervisor import (
    ActiveIncomeProductSupervisor,
    BtcAccumulationProductSupervisor,
    BtcProductCycleResult,
    ProductCycleResult,
)
from src.services.scheduler import DatabaseJobQueue


def _forecast_from_dict(payload: Mapping[str, Any]) -> AlphaForecast:
    values = dict(payload)
    values["direction"] = ForecastDirection(values["direction"])
    return AlphaForecast(**values)


class SqlPortfolioRepository:
    def __init__(self, engine: Engine, *, require_pipeline_identity: bool = False):
        self.engine = engine
        self.require_pipeline_identity = require_pipeline_identity

    def save_forecast(self, forecast: AlphaForecast) -> str:
        required_metadata = {"market_event_id", "feature_ids", "artefact_hash", "engine_version"}
        if self.require_pipeline_identity and not required_metadata.issubset(forecast.metadata):
            raise ValueError(
                "alpha forecast is missing event, feature, artefact, or engine identity"
            )
        if self.require_pipeline_identity and (
            not isinstance(forecast.metadata["feature_ids"], list | tuple)
            or not forecast.metadata["feature_ids"]
        ):
            raise ValueError("alpha forecast must bind at least one feature identity")
        payload = to_primitive(forecast)
        forecast_id = canonical_hash(payload)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(alpha_forecast_table.c.payload).where(
                    alpha_forecast_table.c.id == forecast_id
                )
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != payload:
                    raise ValueError("alpha forecast content-hash collision")
                return forecast_id
            connection.execute(
                insert(alpha_forecast_table).values(
                    id=forecast_id,
                    created_at=forecast.valid_from,
                    payload=payload,
                )
            )
        return forecast_id

    def active_forecasts(self, *, product_id: str, at: str) -> tuple[AlphaForecast, ...]:
        at = timestamp(at, field="at")
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(alpha_forecast_table.c.payload).order_by(alpha_forecast_table.c.id)
            ).scalars()
            forecasts = tuple(_forecast_from_dict(payload) for payload in payloads)
        return tuple(
            forecast
            for forecast in forecasts
            if forecast.product_id == product_id
            and forecast.valid_from <= at
            and at < forecast.valid_until
        )

    def forecast(self, forecast_id: str) -> AlphaForecast:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(alpha_forecast_table.c.payload).where(
                    alpha_forecast_table.c.id == forecast_id
                )
            ).scalar_one_or_none()
        if not isinstance(payload, Mapping):
            raise KeyError(f"alpha forecast does not exist: {forecast_id}")
        forecast = _forecast_from_dict(payload)
        if canonical_hash(to_primitive(forecast)) != forecast_id:
            raise ValueError("alpha forecast identity does not match its content")
        return forecast

    def save_targets(
        self,
        *,
        event_id: str,
        targets: Iterable[TargetPosition],
        created_at: str,
    ) -> tuple[str, ...]:
        created_at = timestamp(created_at, field="created_at")
        identities: list[str] = []
        with self.engine.begin() as connection:
            for target in targets:
                payload = to_primitive(target)
                identity = canonical_hash({"event_id": event_id, "target": payload})
                existing = connection.execute(
                    select(target_position_table.c.payload).where(
                        target_position_table.c.id == identity
                    )
                ).scalar_one_or_none()
                if existing is None:
                    connection.execute(
                        insert(target_position_table).values(
                            id=identity,
                            created_at=created_at,
                            payload={"event_id": event_id, "target": payload},
                        )
                    )
                elif dict(existing) != {"event_id": event_id, "target": payload}:
                    raise ValueError("target-position content-hash collision")
                identities.append(identity)
        return tuple(identities)

    def latest_targets(self, *, portfolio_id: str) -> tuple[TargetPosition, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(target_position_table).order_by(
                    target_position_table.c.created_at.desc(),
                    target_position_table.c.id.desc(),
                )
            ).mappings()
            latest: dict[str, TargetPosition] = {}
            for row in rows:
                payload = dict(row["payload"])
                target_payload = payload.get("target")
                if not isinstance(target_payload, dict):
                    raise ValueError("stored target position is invalid")
                target = TargetPosition(**target_payload)
                if target.portfolio_id == portfolio_id and target.instrument_id not in latest:
                    latest[target.instrument_id] = target
        return tuple(latest[key] for key in sorted(latest))


class DatabaseProductSupervisor:
    """Run both products from active database forecasts and persist their targets."""

    def __init__(
        self,
        *,
        repository: SqlPortfolioRepository,
        active_income: ActiveIncomeProductSupervisor,
        btc_accumulation: BtcAccumulationProductSupervisor,
    ) -> None:
        self.repository = repository
        self.active_income = active_income
        self.btc_accumulation = btc_accumulation

    def run_active_income(
        self,
        *,
        event_id: str,
        evaluated_at: str,
        prices: Mapping[str, float],
        valid_until: str,
        risk_assessment: HierarchicalRiskAssessment,
        correlations: Mapping[str, Mapping[str, float]] | None = None,
        beta_by_instrument: Mapping[str, float] | None = None,
        observed_volatility: Mapping[str, float] | None = None,
        liquidity_fraction_caps: Mapping[str, float] | None = None,
        funding_rates: Mapping[str, float] | None = None,
        sleeve_budgets: Mapping[str, float] | None = None,
        cluster_by_instrument: Mapping[str, str] | None = None,
        cluster_fraction_caps: Mapping[str, float] | None = None,
        product_drawdown_fraction: float = 0.0,
        available_margin_fraction: float = 1.0,
        equity: float | None = None,
    ) -> ProductCycleResult:
        if not prices:
            raise ValueError("active-income cycle requires at least one instrument price")
        forecasts = self.repository.active_forecasts(product_id="active_income", at=evaluated_at)
        result = self.active_income.process_forecasts(
            event_id=event_id,
            event_instrument_id=sorted(prices)[0],
            forecasts=forecasts,
            prices=prices,
            valid_until=valid_until,
            risk_assessment=risk_assessment,
            correlations=correlations,
            beta_by_instrument=beta_by_instrument,
            observed_volatility=observed_volatility,
            liquidity_fraction_caps=liquidity_fraction_caps,
            funding_rates=funding_rates,
            sleeve_budgets=sleeve_budgets,
            cluster_by_instrument=cluster_by_instrument,
            cluster_fraction_caps=cluster_fraction_caps,
            product_drawdown_fraction=product_drawdown_fraction,
            available_margin_fraction=available_margin_fraction,
            equity=equity,
        )
        self.repository.save_targets(
            event_id=event_id,
            targets=result.targets,
            created_at=evaluated_at,
        )
        return result

    def run_btc_accumulation(
        self,
        *,
        event_id: str,
        evaluated_at: str,
        instrument_id: str,
        btc_balance: float,
        stablecoin_balance: float,
        stablecoin_per_btc: float,
        valid_until: str,
        risk_assessment: HierarchicalRiskAssessment,
    ) -> BtcProductCycleResult:
        forecasts = self.repository.active_forecasts(product_id="btc_accumulation", at=evaluated_at)
        result = self.btc_accumulation.process_forecasts(
            event_id=event_id,
            instrument_id=instrument_id,
            forecasts=forecasts,
            btc_balance=btc_balance,
            stablecoin_balance=stablecoin_balance,
            stablecoin_per_btc=stablecoin_per_btc,
            valid_until=valid_until,
            risk_assessment=risk_assessment,
        )
        self.repository.save_targets(
            event_id=event_id,
            targets=(result.target,),
            created_at=evaluated_at,
        )
        return result


class DatabaseProductCycleWorker:
    """Claim product cycles without any dependency on a research worker."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        supervisor: DatabaseProductSupervisor,
        risk_store: SqlRiskDecisionStore,
        update_prices: Callable[[Mapping[str, float]], None] | None = None,
        is_paused: Callable[[str], bool] | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.supervisor = supervisor
        self.risk_store = risk_store
        self.update_prices = update_prices
        self.is_paused = is_paused or (lambda _target: False)
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        if self.is_paused("global") or self.is_paused("product-supervisor"):
            return {"reason_code": "product_supervisor_paused"}
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("active_income_cycle", "btc_accumulation_cycle"),
        )
        if claimed is None:
            return {"reason_code": "product_cycle_queue_empty"}
        payload = claimed.payload
        try:
            product_id = (
                "active_income" if claimed.name == "active_income_cycle" else "btc_accumulation"
            )
            if self.is_paused(product_id):
                raise RuntimeError(f"product_paused:{product_id}")
            risk = self.risk_store.assessment(str(payload["risk_assessment_id"]))
            if claimed.name == "active_income_cycle":
                prices = {str(key): float(value) for key, value in payload["prices"].items()}
                if self.update_prices is not None:
                    self.update_prices(prices)
                result = self.supervisor.run_active_income(
                    event_id=str(payload["event_id"]),
                    evaluated_at=str(payload["evaluated_at"]),
                    prices=prices,
                    valid_until=str(payload["valid_until"]),
                    risk_assessment=risk,
                    correlations=payload.get("correlations"),
                    beta_by_instrument=payload.get("beta_by_instrument"),
                    observed_volatility=payload.get("observed_volatility"),
                    liquidity_fraction_caps=payload.get("liquidity_fraction_caps"),
                    funding_rates=payload.get("funding_rates"),
                    sleeve_budgets=payload.get("sleeve_budgets"),
                    cluster_by_instrument=payload.get("cluster_by_instrument"),
                    cluster_fraction_caps=payload.get("cluster_fraction_caps"),
                    product_drawdown_fraction=float(payload.get("product_drawdown_fraction", 0.0)),
                    available_margin_fraction=float(payload.get("available_margin_fraction", 1.0)),
                    equity=float(payload["equity"]),
                )
            else:
                stablecoin_per_btc = float(payload["stablecoin_per_btc"])
                if self.update_prices is not None:
                    self.update_prices({str(payload["instrument_id"]): stablecoin_per_btc})
                result = self.supervisor.run_btc_accumulation(
                    event_id=str(payload["event_id"]),
                    evaluated_at=str(payload["evaluated_at"]),
                    instrument_id=str(payload["instrument_id"]),
                    btc_balance=float(payload["btc_balance"]),
                    stablecoin_balance=float(payload["stablecoin_balance"]),
                    stablecoin_per_btc=stablecoin_per_btc,
                    valid_until=str(payload["valid_until"]),
                    risk_assessment=risk,
                )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "product_cycle_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "product_cycle_completed",
            "job_id": claimed.job_id,
            "orders": len(result.orders),
            "fills": len(result.fills),
        }


def _retry_at(value: str, seconds: int) -> str:
    import datetime as dt

    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
