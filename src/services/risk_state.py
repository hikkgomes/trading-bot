"""Derive portfolio risk measurements from durable platform state."""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import select

from src.data.database import (
    account_snapshot,
    accounting_entry,
    fill,
    instrument,
    order_intent,
    risk_snapshot,
)
from src.domain._codec import timestamp


@dataclass(frozen=True)
class PortfolioRiskMeasurements:
    """Measured risk values used by the canonical portfolio state."""

    product_drawdown_fraction: float
    daily_pnl_fraction: float
    global_drawdown_fraction: float
    trades_today: int
    correlations: dict[str, dict[str, float]]
    beta: dict[str, float]
    risk_data_available: bool
    risk_data_missing: tuple[str, ...]
    clusters: dict[str, str]
    cluster_fraction_caps: dict[str, float]
    open_exposure_fraction: float
    pending_exposure_fraction: float


class PortfolioRiskCalculator:
    """Calculate risk from ledger, authenticated balances, fills and market marks."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def calculate(
        self,
        *,
        product_id: str,
        account_id: str,
        product: Mapping[str, object],
        account: Mapping[str, object],
        balances: Mapping[str, object],
        positions: Mapping[str, object],
        open_orders: tuple[Mapping[str, object], ...],
        market: Mapping[str, Mapping[str, object]],
        at: str,
    ) -> PortfolioRiskMeasurements:
        at = timestamp(at, field="risk measurement time")
        product_ids = tuple(sorted(str(value) for value in self._products_for_account(account_id)))
        if product_id not in product_ids:
            product_ids = tuple(sorted((*product_ids, product_id)))
        effects = self._ledger_effects(product_ids, at)
        initial = self._initial_equity(
            account_id=account_id,
            product_id=product_id,
            product=product,
            account=account,
            market=market,
        )
        product_effects = effects.get(product_id, ())
        global_effects = tuple(item for values in effects.values() for item in values)
        clusters = self._clusters(tuple(str(key) for key in market))
        cluster_limit = _positive_number(product.get("maximum_cluster_fraction")) or 1.0
        correlations, beta, risk_data_missing = self._factor_measurements(
            product_id=product_id,
            at=at,
            instrument_ids=tuple(str(key) for key in market),
        )
        return PortfolioRiskMeasurements(
            product_drawdown_fraction=_drawdown(product_effects, initial),
            daily_pnl_fraction=_daily_pnl_fraction(product_effects, initial, at),
            global_drawdown_fraction=_drawdown(
                global_effects,
                sum(
                    self._initial_equity(
                        account_id=account_id,
                        product_id=value,
                        product=product,
                        account=account,
                        market=market,
                    )
                    for value in product_ids
                ),
            ),
            trades_today=self._trades_today(
                product_id=product_id,
                portfolio_id=str(product.get("portfolio_id") or ""),
                at=at,
            ),
            correlations=correlations,
            beta=beta,
            risk_data_available=not risk_data_missing,
            risk_data_missing=risk_data_missing,
            clusters=clusters,
            cluster_fraction_caps={
                cluster: cluster_limit for cluster in sorted(set(clusters.values()))
            },
            open_exposure_fraction=_exposure_fraction(
                positions,
                market,
                self._current_equity(product_id, balances, market),
            ),
            pending_exposure_fraction=_order_exposure_fraction(
                open_orders,
                market,
                self._current_equity(product_id, balances, market),
            ),
        )

    def _products_for_account(self, account_id: str) -> tuple[str, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(account_snapshot.c.payload)).mappings()
        products = {
            str(payload.get("product_id"))
            for row in rows
            if isinstance((payload := row["payload"]), Mapping)
            and str(payload.get("account_id")) == account_id
            and str(payload.get("product_id") or "")
        }
        return tuple(sorted(products))

    def _ledger_effects(
        self, product_ids: tuple[str, ...], at: str
    ) -> dict[str, tuple[tuple[str, float], ...]]:
        if not product_ids:
            return {}
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    accounting_entry.c.product_id,
                    accounting_entry.c.created_at,
                    accounting_entry.c.payload,
                )
                .where(
                    accounting_entry.c.product_id.in_(product_ids),
                    accounting_entry.c.created_at <= at,
                )
                .order_by(accounting_entry.c.created_at, accounting_entry.c.sequence)
            ).mappings()
        result: dict[str, list[tuple[str, float]]] = {key: [] for key in product_ids}
        for row in rows:
            payload = row["payload"]
            metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
            if not isinstance(metadata, Mapping) or metadata.get("pnl_effect") is None:
                continue
            value = _finite_number(metadata["pnl_effect"], "ledger pnl_effect")
            result.setdefault(str(row["product_id"]), []).append((str(row["created_at"]), value))
        return {key: tuple(value) for key, value in result.items()}

    def _initial_equity(
        self,
        *,
        account_id: str,
        product_id: str,
        product: Mapping[str, object],
        account: Mapping[str, object],
        market: Mapping[str, Mapping[str, object]],
    ) -> float:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(account_snapshot.c.payload)
                .where(account_snapshot.c.account_id == account_id)
                .order_by(account_snapshot.c.observed_at, account_snapshot.c.id)
            ).scalars()
            payload = next(
                (
                    value
                    for value in rows
                    if isinstance(value, Mapping) and str(value.get("product_id")) == product_id
                ),
                None,
            )
        if not isinstance(payload, Mapping):
            payload = {"balances": account.get("paper_starting_balances", {})}
        balances = payload.get("balances")
        if not isinstance(balances, Mapping):
            balances = {}
        equity = _equity(product_id, balances, market)
        if equity > 0:
            return equity
        starting = account.get("paper_starting_balances")
        return _equity(product_id, starting if isinstance(starting, Mapping) else {}, market)

    @staticmethod
    def _current_equity(
        product_id: str,
        balances: Mapping[str, object],
        market: Mapping[str, Mapping[str, object]],
    ) -> float:
        equity = _equity(product_id, balances, market)
        if equity <= 0.0:
            raise ValueError("risk measurement requires positive current equity")
        return equity

    def _trades_today(self, *, product_id: str, portfolio_id: str, at: str) -> int:
        current = dt.datetime.fromisoformat(at)
        start = current.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(fill.c.created_at, order_intent.c.payload)
                .select_from(fill.join(order_intent, fill.c.order_id == order_intent.c.id))
                .where(fill.c.created_at > start, fill.c.created_at <= at)
                .order_by(fill.c.created_at, fill.c.id)
            ).mappings()
        return sum(
            1
            for row in rows
            if isinstance(row["payload"], Mapping)
            and str(row["payload"].get("product_id")) == product_id
            and (not portfolio_id or str(row["payload"].get("portfolio_id")) == portfolio_id)
        )

    def _market_history(self, *, product_id: str, at: str) -> dict[str, list[float]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(risk_snapshot.c.payload, risk_snapshot.c.created_at)
                .where(risk_snapshot.c.created_at <= at)
                .order_by(risk_snapshot.c.created_at, risk_snapshot.c.id)
            ).mappings()
        history: dict[str, list[float]] = {}
        for row in rows:
            payload = row["payload"]
            if (
                not isinstance(payload, Mapping)
                or payload.get("kind") != "market_data_input"
                or str(payload.get("product_id")) != product_id
            ):
                continue
            values = payload.get("values")
            if not isinstance(values, Mapping):
                continue
            price = _positive_number(values.get("close", values.get("price")))
            if price:
                history.setdefault(str(payload.get("instrument_id")), []).append(price)
        return history

    def _correlations(self, *, product_id: str, at: str) -> dict[str, dict[str, float]]:
        history = self._market_history(product_id=product_id, at=at)
        returns = {key: _returns(values) for key, values in history.items()}
        return _correlations_from_returns(returns)

    def _factor_measurements(
        self,
        *,
        product_id: str,
        at: str,
        instrument_ids: tuple[str, ...],
    ) -> tuple[dict[str, dict[str, float]], dict[str, float], tuple[str, ...]]:
        history = self._market_history(product_id=product_id, at=at)
        returns = {key: _returns(history.get(key, [])) for key in instrument_ids}
        correlations = _correlations_from_returns(returns)
        benchmark_key = next((key for key in returns if _is_btc_benchmark(key)), None)
        benchmark = returns.get(benchmark_key, []) if benchmark_key is not None else []
        beta = {
            key: 1.0 if _is_btc_benchmark(key) else _beta(values, benchmark)
            for key, values in sorted(returns.items())
            if _is_btc_benchmark(key) or (len(values) >= 2 and len(benchmark) >= 2)
        }
        missing: set[str] = set()
        for instrument_id, values in sorted(returns.items()):
            if _is_btc_benchmark(instrument_id):
                continue
            if len(values) < 2 or len(benchmark) < 2:
                missing.add(f"beta:{instrument_id}")
        keys = sorted(returns)
        for index, left in enumerate(keys):
            for right in keys[index + 1 :]:
                if len(returns[left]) < 2 or len(returns[right]) < 2:
                    missing.add(f"correlation:{left}:{right}")
        return correlations, beta, tuple(sorted(missing))

    def _beta(self, *, product_id: str, at: str) -> dict[str, float]:
        history = self._market_history(product_id=product_id, at=at)
        returns = {key: _returns(values) for key, values in history.items()}
        benchmark_key = next((key for key in returns if _is_btc_benchmark(key)), None)
        benchmark = returns.get(benchmark_key, []) if benchmark_key is not None else []
        return {
            key: 1.0 if _is_btc_benchmark(key) else _beta(values, benchmark)
            for key, values in sorted(returns.items())
            if _is_btc_benchmark(key) or (len(values) >= 2 and len(benchmark) >= 2)
        }

    def _clusters(self, instrument_ids: tuple[str, ...]) -> dict[str, str]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(instrument.c.id, instrument.c.payload).where(
                    instrument.c.id.in_(instrument_ids)
                )
            ).mappings()
        payloads = {str(row["id"]): row["payload"] for row in rows}
        return {
            instrument_id: "base:" + _base_asset(payloads.get(instrument_id), instrument_id)
            for instrument_id in instrument_ids
            if _base_asset(payloads.get(instrument_id), instrument_id)
        }


def _drawdown(effects: tuple[tuple[str, float], ...], initial: float) -> float:
    if initial <= 0:
        return 1.0 if effects else 0.0
    equity = initial
    peak = initial
    worst = 0.0
    for _occurred_at, effect in effects:
        equity += effect
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return max(0.0, worst)


def _daily_pnl_fraction(effects: tuple[tuple[str, float], ...], initial: float, at: str) -> float:
    if initial <= 0:
        return 0.0
    day = dt.datetime.fromisoformat(at).date()
    return (
        sum(
            effect
            for occurred_at, effect in effects
            if dt.datetime.fromisoformat(occurred_at).date() == day
        )
        / initial
    )


def _equity(
    product_id: str,
    balances: Mapping[str, object],
    market: Mapping[str, Mapping[str, object]],
) -> float:
    if product_id == "btc_accumulation":
        btc = _positive_number(balances.get("BTC"))
        usdt = _positive_number(balances.get("USDT"))
        price = next(
            (
                _positive_number(values.get("price"))
                for values in market.values()
                if _positive_number(values.get("price"))
            ),
            0.0,
        )
        return btc + usdt / price if price else btc
    return _positive_number(balances.get("USDT"))


def _exposure_fraction(
    positions: Mapping[str, object],
    market: Mapping[str, Mapping[str, object]],
    equity: float,
) -> float:
    missing = sorted(
        str(instrument_id)
        for instrument_id, quantity in positions.items()
        if abs(_finite_number(quantity, "position quantity")) > 1e-12
        and _positive_number(market.get(str(instrument_id), {}).get("price")) <= 0.0
    )
    if missing:
        raise ValueError("risk measurement has no price for: " + ", ".join(missing))
    return (
        sum(
            abs(_finite_number(quantity, "position quantity"))
            * _positive_number(market.get(str(instrument_id), {}).get("price"))
            for instrument_id, quantity in positions.items()
        )
        / equity
    )


def _order_exposure_fraction(
    orders: tuple[Mapping[str, object], ...],
    market: Mapping[str, Mapping[str, object]],
    equity: float,
) -> float:
    missing = sorted(
        str(order.get("instrument_id"))
        for order in orders
        if not bool(order.get("reduce_only"))
        and _positive_number(market.get(str(order.get("instrument_id")), {}).get("price")) <= 0.0
    )
    if missing:
        raise ValueError("risk measurement has no pending-order price for: " + ", ".join(missing))
    return (
        sum(
            abs(_positive_number(order.get("remaining_quantity", order.get("quantity"))))
            * _positive_number(market.get(str(order.get("instrument_id")), {}).get("price"))
            for order in orders
            if not bool(order.get("reduce_only"))
        )
        / equity
    )


def _returns(values: list[float]) -> list[float]:
    return [values[index] / values[index - 1] - 1.0 for index in range(1, len(values))]


def _correlations_from_returns(
    returns: Mapping[str, list[float]],
) -> dict[str, dict[str, float]]:
    result = {key: {key: 1.0} for key, values in returns.items() if len(values) >= 2}
    keys = sorted(returns)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            if len(returns[left]) < 2 or len(returns[right]) < 2:
                continue
            value = _correlation(returns[left], returns[right])
            result.setdefault(left, {})[right] = value
            result.setdefault(right, {})[left] = value
    return result


def _is_btc_benchmark(instrument_id: str) -> bool:
    return bool(re.search(r"(?:^|[:/ -])BTC/?USDT(?:$|[:/ -])", instrument_id.upper()))


def _correlation(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size < 2:
        return 0.0
    left = left[-size:]
    right = right[-size:]
    left_mean = sum(left) / size
    right_mean = sum(right) / size
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=False))
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    return covariance / denominator if denominator else 0.0


def _beta(values: list[float], benchmark: list[float] | None) -> float:
    if benchmark is None:
        return 0.0
    size = min(len(values), len(benchmark))
    if size < 2:
        return 0.0
    values = values[-size:]
    benchmark = benchmark[-size:]
    benchmark_mean = sum(benchmark) / size
    variance = sum((value - benchmark_mean) ** 2 for value in benchmark)
    if variance == 0:
        return 0.0
    value_mean = sum(values) / size
    covariance = sum(
        (value - value_mean) * (reference - benchmark_mean)
        for value, reference in zip(values, benchmark, strict=False)
    )
    return covariance / variance


def _base_asset(payload: object, instrument_id: str) -> str:
    if isinstance(payload, Mapping) and str(payload.get("base_asset") or "").strip():
        return str(payload["base_asset"]).upper()
    value = instrument_id.upper().removesuffix("USDT").removesuffix("USD")
    return value or "UNKNOWN"


def _positive_number(value: object) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result > 0 else 0.0


def _finite_number(value: object, field_name: str) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result
