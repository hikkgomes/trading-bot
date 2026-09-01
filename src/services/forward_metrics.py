"""Collect immutable, interval-scoped forward-paper evidence facts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from src.data.database import (
    account_snapshot,
    accounting_entry,
    exchange_order,
    fill,
    nav_snapshot,
    order_intent,
    reconciliation_event,
    risk_snapshot,
)
from src.domain._codec import canonical_hash, timestamp


@dataclass(frozen=True)
class ForwardEvidenceMetrics:
    """Metrics for one forward observation interval.

    The values are facts, not acceptance decisions.  Promotion policy is
    evaluated later from an immutable summary of these intervals.
    """

    net_pnl: float
    benchmark_pnl: float
    drawdown: float
    execution_drift: float
    model_drift: float
    portfolio_capacity: float
    risk_budget_available: float
    data_gaps: int
    effective_trades: int
    fill_rate: float
    slippage: float
    data_uptime: float
    rejected_orders: int
    source_event_ids: tuple[str, ...]
    window_start: str
    objective_unit: str | None = None
    objective_value: float | None = None
    benchmark_value: float | None = None
    objective_excess: float | None = None
    objective_excess_fraction: float | None = None
    trading_days: int = 0
    cycles: int = 0
    effective_independent_episodes: int = 0
    tail_loss: float = 0.0

    def to_payload(
        self, *, forecast: Mapping[str, Any], target: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        facts = {
            "schema": "platform.forward_evidence_facts/v1",
            "window_start": self.window_start,
            "source_event_ids": list(self.source_event_ids),
            "metrics": {
                "net_pnl": self.net_pnl,
                "benchmark_pnl": self.benchmark_pnl,
                "drawdown": self.drawdown,
                "execution_drift": self.execution_drift,
                "model_drift": self.model_drift,
                "portfolio_capacity": self.portfolio_capacity,
                "risk_budget_available": self.risk_budget_available,
                "data_gaps": self.data_gaps,
                "effective_trades": self.effective_trades,
                "fill_rate": self.fill_rate,
                "slippage": self.slippage,
                "data_uptime": self.data_uptime,
                "rejected_orders": self.rejected_orders,
                "objective_unit": self.objective_unit,
                "objective_value": self.objective_value,
                "benchmark_value": self.benchmark_value,
                "objective_excess": self.objective_excess,
                "objective_excess_fraction": self.objective_excess_fraction,
                "trading_days": self.trading_days,
                "cycles": self.cycles,
                "effective_independent_episodes": self.effective_independent_episodes,
                "tail_loss": self.tail_loss,
            },
            "forecast_hash": canonical_hash(dict(forecast)),
            "target_hash": canonical_hash(dict(target)) if target is not None else None,
        }
        facts["facts_hash"] = canonical_hash(facts)
        return facts


class ForwardEvidenceCollector:
    """Read the durable platform state needed for one forward interval."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def collect(
        self,
        *,
        assignment: Mapping[str, Any],
        product_id: str,
        instrument_id: str,
        artefact_created_at: str,
        evaluation_time: str,
        forecast: Mapping[str, Any],
        target: Mapping[str, Any] | None,
        previous_observed_at: str | None,
        strategy_version_id: str | None = None,
        assignment_id: str | None = None,
    ) -> ForwardEvidenceMetrics:
        created_at = timestamp(artefact_created_at, field="artefact_created_at")
        evaluated_at = timestamp(evaluation_time, field="evaluation_time")
        window_start = timestamp(previous_observed_at or created_at, field="window_start")
        if window_start < created_at:
            window_start = created_at
        orders = self._orders(
            portfolio_id=str(assignment["portfolio_id"]),
            instrument_id=instrument_id,
            start=window_start,
            at=evaluated_at,
            strategy_version_id=strategy_version_id,
            assignment_id=assignment_id,
        )
        order_ids = tuple(str(row["id"]) for row in orders)
        fills = self._fills(order_ids, start=window_start, at=evaluated_at)
        ledger_rows = self._ledger_rows(
            product_id=product_id,
            start=window_start,
            at=evaluated_at,
            strategy_version_id=strategy_version_id,
            assignment_id=assignment_id,
            instrument_id=instrument_id,
        )
        all_ledger_rows = self._ledger_rows(
            product_id=product_id,
            start=created_at,
            at=evaluated_at,
            strategy_version_id=strategy_version_id,
            assignment_id=assignment_id,
            instrument_id=instrument_id,
        )
        all_market_rows = self._market_rows(
            product_id=product_id,
            instrument_id=instrument_id,
            start=created_at,
            at=evaluated_at,
        )
        market_rows = self._market_rows(
            product_id=product_id,
            instrument_id=instrument_id,
            start=window_start,
            at=evaluated_at,
        )
        nav_rows = self._nav_rows(product_id=product_id, at=evaluated_at)
        reconciliations = self._reconciliation_rows(
            product_id=product_id,
            instrument_id=instrument_id,
            start=window_start,
            at=evaluated_at,
        )
        net_pnl = self._ledger_pnl(ledger_rows)
        benchmark_pnl = self._benchmark_pnl(product_id, market_rows, assignment)
        drawdown = self._drawdown(
            product_id,
            all_ledger_rows,
            assignment,
            at=created_at,
        )
        drawdown = max(
            drawdown,
            self._nav_drawdown(nav_rows, start=created_at, at=evaluated_at),
        )
        slippage = self._slippage(fills)
        execution_drift = self._execution_drift(fills)
        model_drift = self._model_drift(net_pnl, forecast, bool(fills))
        objective = self._objective_metrics(
            product_id=product_id,
            assignment=assignment,
            artefact_created_at=created_at,
            evaluation_time=evaluated_at,
            ledger_rows=all_ledger_rows,
            market_rows=all_market_rows,
            nav_rows=nav_rows,
        )
        attempted = len(orders)
        completed = sum(1 for row in orders if self._order_filled(row["id"]))
        rejected = sum(1 for row in orders if self._order_rejected(row["id"]))
        data_gaps = sum(1 for row in reconciliations if self._is_data_gap(row))
        source_ids = tuple(
            dict.fromkeys(
                [
                    str(forecast.get("forecast_id") or ""),
                    str(target.get("target_position_id") or "") if target else "",
                    *order_ids,
                    *(str(row["id"]) for row in fills),
                    *(str(row["id"]) for row in market_rows),
                    *(str(row["id"]) for row in ledger_rows),
                    *(str(row["id"]) for row in all_market_rows),
                    *(str(row["id"]) for row in nav_rows),
                    *objective[5],
                ]
            )
        )
        source_ids = tuple(value for value in source_ids if value)
        if not source_ids:
            raise ValueError("forward evidence has no durable source identities")
        capacity = _positive_or_zero(assignment.get("capital_limit"))
        configured_budget = _positive_or_zero(assignment.get("risk_budget"))
        risk_budget = min(capacity, configured_budget)
        trading_days = len(
            {str(row.get("created_at") or row.get("observed_at"))[:10] for row in market_rows}
        )
        cycles = self._cycles(fills)
        ledger_returns = self._ledger_returns(ledger_rows)
        independent_episodes = cycles if cycles > 0 else completed
        data_uptime = self._data_uptime(
            market_rows,
            data_gaps=data_gaps,
        )
        return ForwardEvidenceMetrics(
            net_pnl=net_pnl,
            benchmark_pnl=benchmark_pnl,
            drawdown=drawdown,
            execution_drift=execution_drift,
            model_drift=model_drift,
            portfolio_capacity=capacity,
            risk_budget_available=risk_budget,
            data_gaps=data_gaps,
            effective_trades=len(fills),
            fill_rate=(completed / attempted if attempted else 1.0),
            slippage=slippage,
            data_uptime=data_uptime,
            rejected_orders=rejected,
            source_event_ids=source_ids,
            window_start=window_start,
            objective_unit=objective[0],
            objective_value=objective[1],
            benchmark_value=objective[2],
            objective_excess=objective[3],
            objective_excess_fraction=objective[4],
            trading_days=trading_days,
            cycles=cycles,
            effective_independent_episodes=independent_episodes,
            tail_loss=self._tail_loss(ledger_returns),
        )

    def latest_observed_at(
        self,
        *,
        strategy_version_id: str,
        product_id: str,
        instrument_id: str,
        artefact_hash: str,
    ) -> str | None:
        from src.data.database import forward_paper_observation

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(forward_paper_observation.c.observed_at, forward_paper_observation.c.payload)
                .where(
                    forward_paper_observation.c.strategy_version_id == strategy_version_id,
                    forward_paper_observation.c.product_id == product_id,
                    forward_paper_observation.c.instrument_id == instrument_id,
                    forward_paper_observation.c.artefact_hash == artefact_hash,
                )
                .order_by(
                    forward_paper_observation.c.observed_at.desc(),
                    forward_paper_observation.c.id.desc(),
                )
            ).mappings()
            row = next(iter(rows), None)
        return (
            None
            if row is None
            else timestamp(str(row["observed_at"]), field="previous_observed_at")
        )

    def _orders(
        self,
        *,
        portfolio_id: str,
        instrument_id: str,
        start: str,
        at: str,
        strategy_version_id: str | None = None,
        assignment_id: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(order_intent.c.id, order_intent.c.created_at, order_intent.c.payload)
                .where(
                    order_intent.c.created_at > start,
                    order_intent.c.created_at <= at,
                )
                .order_by(order_intent.c.created_at, order_intent.c.id)
            ).mappings()
            result = []
            for row in rows:
                payload = row["payload"]
                if (
                    isinstance(payload, Mapping)
                    and str(payload.get("portfolio_id")) == portfolio_id
                    and str(payload.get("instrument_id")) == instrument_id
                    and _order_belongs_to_strategy(payload, strategy_version_id=strategy_version_id)
                    and _order_belongs_to_assignment(payload, assignment_id=assignment_id)
                ):
                    result.append(row)
        return tuple(result)

    def _fills(
        self, order_ids: tuple[str, ...], *, start: str, at: str
    ) -> tuple[Mapping[str, Any], ...]:
        if not order_ids:
            return ()
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(fill.c.id, fill.c.created_at, fill.c.order_id, fill.c.payload)
                .where(
                    fill.c.order_id.in_(order_ids),
                    fill.c.created_at > start,
                    fill.c.created_at <= at,
                )
                .order_by(fill.c.created_at, fill.c.id)
            ).mappings()
            return tuple(row for row in rows if isinstance(row["payload"], Mapping))

    def _ledger_rows(
        self,
        *,
        product_id: str,
        start: str,
        at: str,
        strategy_version_id: str | None = None,
        assignment_id: str | None = None,
        instrument_id: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    accounting_entry.c.id, accounting_entry.c.created_at, accounting_entry.c.payload
                )
                .where(
                    accounting_entry.c.product_id == product_id,
                    accounting_entry.c.created_at > start,
                    accounting_entry.c.created_at <= at,
                )
                .order_by(accounting_entry.c.created_at, accounting_entry.c.sequence)
            ).mappings()
            return tuple(
                row
                for row in rows
                if isinstance(row["payload"], Mapping)
                and _ledger_belongs_to_scope(
                    row["payload"],
                    strategy_version_id=strategy_version_id,
                    assignment_id=assignment_id,
                    instrument_id=instrument_id,
                )
            )

    def _market_rows(
        self, *, product_id: str, instrument_id: str, start: str, at: str
    ) -> tuple[Mapping[str, Any], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(risk_snapshot.c.id, risk_snapshot.c.created_at, risk_snapshot.c.payload)
                .where(
                    risk_snapshot.c.created_at > start,
                    risk_snapshot.c.created_at <= at,
                )
                .order_by(risk_snapshot.c.created_at, risk_snapshot.c.id)
            ).mappings()
        return tuple(
            row
            for row in rows
            if isinstance(row["payload"], Mapping)
            and row["payload"].get("kind") == "market_data_input"
            and str(row["payload"].get("product_id")) == product_id
            and str(row["payload"].get("instrument_id")) == instrument_id
        )

    def _nav_rows(self, *, product_id: str, at: str) -> tuple[Mapping[str, Any], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(nav_snapshot.c.id, nav_snapshot.c.created_at, nav_snapshot.c.payload)
                .where(nav_snapshot.c.created_at <= at)
                .order_by(nav_snapshot.c.created_at, nav_snapshot.c.id)
            ).mappings()
        return tuple(
            row
            for row in rows
            if isinstance(row["payload"], Mapping)
            and str(row["payload"].get("product_id")) == product_id
        )

    def _reconciliation_rows(
        self, *, product_id: str, instrument_id: str, start: str, at: str
    ) -> tuple[Mapping[str, Any], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(reconciliation_event.c.id, reconciliation_event.c.payload)
                .where(
                    reconciliation_event.c.created_at > start,
                    reconciliation_event.c.created_at <= at,
                )
                .order_by(reconciliation_event.c.created_at, reconciliation_event.c.id)
            ).mappings()
        return tuple(
            row
            for row in rows
            if isinstance(row["payload"], Mapping)
            and str(row["payload"].get("product_id")) == product_id
            and (
                not row["payload"].get("instrument_id")
                or str(row["payload"].get("instrument_id")) == instrument_id
            )
        )

    def _order_statuses(self, order_id: str) -> tuple[str, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(exchange_order.c.status)
                .where(exchange_order.c.order_id == order_id)
                .order_by(exchange_order.c.sequence)
            ).scalars()
        return tuple(str(value).casefold() for value in rows)

    def _order_filled(self, order_id: str) -> bool:
        return (
            bool(self._order_statuses(order_id)) and self._order_statuses(order_id)[-1] == "filled"
        )

    def _order_rejected(self, order_id: str) -> bool:
        return bool(self._order_statuses(order_id)) and self._order_statuses(order_id)[-1] in {
            "rejected",
            "expired",
        }

    @staticmethod
    def _ledger_pnl(rows: tuple[Mapping[str, Any], ...]) -> float:
        total = 0.0
        for row in rows:
            payload = row["payload"]
            metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
            if isinstance(metadata, Mapping) and metadata.get("pnl_effect") is not None:
                total += _number(metadata["pnl_effect"], "ledger pnl_effect")
        return total

    @staticmethod
    def _ledger_returns(rows: tuple[Mapping[str, Any], ...]) -> tuple[float, ...]:
        values: list[float] = []
        for row in rows:
            payload = row["payload"]
            metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
            if isinstance(metadata, Mapping) and metadata.get("pnl_effect") is not None:
                values.append(_number(metadata["pnl_effect"], "ledger pnl_effect"))
        return tuple(values)

    @staticmethod
    def _cycles(rows: tuple[Mapping[str, Any], ...]) -> int:
        previous: dict[str, str] = {}
        cycles = 0
        for row in rows:
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                continue
            symbol = str(payload.get("symbol") or payload.get("instrument_id") or "*")
            side = str(payload.get("side") or "").casefold()
            if side == "buy" and previous.get(symbol) == "sell":
                cycles += 1
            elif side == "sell" and previous.get(symbol) == "buy":
                cycles += 1
            if side in {"buy", "sell"}:
                previous[symbol] = side
        return cycles

    @staticmethod
    def _tail_loss(values: tuple[float, ...]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        count = max(1, math.ceil(len(ordered) * 0.05))
        return max(0.0, -sum(ordered[:count]) / count)

    @staticmethod
    def _benchmark_pnl(
        product_id: str,
        rows: tuple[Mapping[str, Any], ...],
        assignment: Mapping[str, Any],
    ) -> float:
        if product_id != "btc_accumulation" or len(rows) < 2:
            return 0.0
        prices = []
        for row in rows:
            payload = row["payload"]
            values = payload.get("values") if isinstance(payload, Mapping) else None
            if isinstance(values, Mapping):
                raw = values.get("close", values.get("price"))
                if raw is not None and _positive_or_zero(raw) > 0:
                    prices.append(_positive_or_zero(raw))
        if len(prices) < 2:
            return 0.0
        return (prices[-1] / prices[0] - 1.0) * _positive_or_zero(assignment.get("capital_limit"))

    def _drawdown(
        self,
        product_id: str,
        rows: tuple[Mapping[str, Any], ...],
        assignment: Mapping[str, Any],
        *,
        at: str,
    ) -> float:
        if not rows:
            return 0.0
        base = self._starting_equity(product_id, assignment, at=at)
        running = 0.0
        peak = 0.0
        worst = 0.0
        for row in rows:
            payload = row["payload"]
            metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
            if not isinstance(metadata, Mapping) or metadata.get("pnl_effect") is None:
                continue
            running += _number(metadata["pnl_effect"], "ledger pnl_effect")
            peak = max(peak, running)
            worst = max(worst, (peak - running) / base)
        return max(0.0, worst)

    @staticmethod
    def _nav_drawdown(
        rows: tuple[Mapping[str, Any], ...], *, start: str, at: str
    ) -> float:
        points: list[tuple[str, float]] = []
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, Mapping) or payload.get("nav") is None:
                continue
            observed_at = str(payload.get("observed_at") or row["created_at"])
            if observed_at > at:
                continue
            try:
                value = _positive_or_zero(payload["nav"])
            except (KeyError, TypeError, ValueError):
                continue
            if value > 0.0:
                points.append((observed_at, value))
        if not points:
            return 0.0
        baseline = [value for observed_at, value in points if observed_at <= start]
        after = [value for observed_at, value in points if observed_at > start]
        values = ([baseline[-1]] if baseline else [after[0]]) + after
        peak = values[0]
        worst = 0.0
        for value in values:
            peak = max(peak, value)
            worst = max(worst, (peak - value) / peak)
        return max(0.0, worst)

    @staticmethod
    def _data_uptime(market_rows: tuple[Mapping[str, Any], ...], *, data_gaps: int) -> float:
        observed = len(market_rows)
        if observed == 0:
            return 0.0
        return max(0.0, min(1.0, observed / max(1, observed + data_gaps)))

    def _starting_equity(
        self,
        product_id: str,
        assignment: Mapping[str, Any],
        *,
        at: str | None = None,
    ) -> float:
        account_id = _assignment_account_id(assignment)
        assignment_payload = assignment.get("payload")
        if not account_id and isinstance(assignment_payload, Mapping):
            account_id = str(assignment_payload.get("account_id") or "")
        with self.engine.connect() as connection:
            statement = select(account_snapshot.c.payload).where(
                account_snapshot.c.account_id == account_id
            )
            if at is not None:
                statement = statement.where(account_snapshot.c.observed_at <= at)
            row = connection.execute(
                statement.order_by(
                    account_snapshot.c.observed_at.desc(), account_snapshot.c.id.desc()
                ).limit(1)
            ).scalar_one_or_none()
        if isinstance(row, Mapping):
            balances = row.get("balances")
            if isinstance(balances, Mapping):
                asset = "BTC" if product_id == "btc_accumulation" else "USDT"
                value = _positive_or_zero(balances.get(asset))
                if value > 0:
                    return value
        return max(_positive_or_zero(assignment.get("capital_limit")), 1.0)

    def _account_snapshot_rows(self, *, account_id: str, at: str) -> tuple[Mapping[str, Any], ...]:
        if not account_id:
            return ()
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    account_snapshot.c.id,
                    account_snapshot.c.observed_at,
                    account_snapshot.c.payload,
                )
                .where(
                    account_snapshot.c.account_id == account_id,
                    account_snapshot.c.observed_at <= at,
                )
                .order_by(account_snapshot.c.observed_at, account_snapshot.c.id)
            ).mappings()
            return tuple(row for row in rows if isinstance(row["payload"], Mapping))

    @staticmethod
    def _market_prices(rows: tuple[Mapping[str, Any], ...]) -> tuple[float, ...]:
        prices: list[float] = []
        for row in rows:
            payload = row["payload"]
            values = payload.get("values") if isinstance(payload, Mapping) else None
            if not isinstance(values, Mapping):
                continue
            price = _positive_or_zero(values.get("close", values.get("price")))
            if price > 0.0:
                prices.append(price)
        return tuple(prices)

    def _objective_metrics(
        self,
        *,
        product_id: str,
        assignment: Mapping[str, Any],
        artefact_created_at: str,
        evaluation_time: str,
        ledger_rows: tuple[Mapping[str, Any], ...],
        market_rows: tuple[Mapping[str, Any], ...],
        nav_rows: tuple[Mapping[str, Any], ...],
    ) -> tuple[str, float | None, float | None, float | None, float | None, tuple[str, ...]]:
        nav_points = self._objective_nav_points(nav_rows, start=artefact_created_at)
        if product_id == "active_income":
            if nav_points is not None:
                initial, final, source_ids = nav_points
                excess = final - initial
                return (
                    "USDT",
                    final,
                    initial,
                    excess,
                    excess / initial if initial > 0.0 else None,
                    source_ids,
                )
            initial = self._starting_equity(product_id, assignment, at=artefact_created_at)
            excess = self._ledger_pnl(ledger_rows)
            return (
                "USDT",
                initial + excess,
                initial,
                excess,
                excess / initial if initial > 0.0 else None,
                (),
            )
        if product_id != "btc_accumulation":
            return (product_id, None, None, None, None, ())
        if nav_points is not None:
            initial, final, source_ids = nav_points
            final_payload = next(
                (
                    row["payload"]
                    for row in reversed(nav_rows)
                    if isinstance(row["payload"], Mapping)
                    and str(row["payload"].get("observed_at") or row["created_at"])
                    > artefact_created_at
                ),
                None,
            )
            benchmark = (
                _positive_or_zero(final_payload.get("passive_benchmark_nav"))
                if isinstance(final_payload, Mapping)
                else 0.0
            )
            benchmark = benchmark or initial
            excess = final - benchmark
            return (
                "BTC",
                final,
                benchmark,
                excess,
                excess / benchmark if benchmark > 0.0 else None,
                source_ids,
            )
        snapshots = self._account_snapshot_rows(
            account_id=_assignment_account_id(assignment), at=evaluation_time
        )
        prices = self._market_prices(market_rows)
        source_ids = tuple(str(row["id"]) for row in snapshots)
        if not snapshots or not prices:
            return ("BTC", None, None, None, None, source_ids)
        before_creation = tuple(
            row for row in snapshots if str(row["observed_at"]) <= artefact_created_at
        )
        first_snapshot = before_creation[-1] if before_creation else snapshots[0]
        last_snapshot = snapshots[-1]
        initial_btc_nav = self._btc_nav(first_snapshot["payload"], prices[0])
        final_btc_nav = self._btc_nav(last_snapshot["payload"], prices[-1])
        if initial_btc_nav is None or final_btc_nav is None or initial_btc_nav <= 0.0:
            return ("BTC", None, None, None, None, source_ids)
        excess = final_btc_nav - initial_btc_nav
        return (
            "BTC",
            final_btc_nav,
            initial_btc_nav,
            excess,
            excess / initial_btc_nav,
            source_ids,
        )

    @staticmethod
    def _objective_nav_points(
        rows: tuple[Mapping[str, Any], ...], *, start: str
    ) -> tuple[float, float, tuple[str, ...]] | None:
        points = [
            row
            for row in rows
            if isinstance(row["payload"], Mapping)
            and _positive_or_zero(row["payload"].get("nav")) > 0.0
        ]
        if not points:
            return None
        baseline = [
            row
            for row in points
            if str(row["payload"].get("observed_at") or row["created_at"]) <= start
        ]
        after = [
            row
            for row in points
            if str(row["payload"].get("observed_at") or row["created_at"]) > start
        ]
        if not after:
            return None
        initial_row = baseline[-1] if baseline else after[0]
        final_row = after[-1]
        source_ids = tuple(str(row["id"]) for row in (*baseline[-1:], *after))
        return (
            _positive_or_zero(initial_row["payload"]["nav"]),
            _positive_or_zero(final_row["payload"]["nav"]),
            source_ids,
        )

    @staticmethod
    def _btc_nav(payload: Mapping[str, Any], price: float) -> float | None:
        balances = payload.get("balances")
        if not isinstance(balances, Mapping) or price <= 0.0:
            return None
        btc = _positive_or_zero(balances.get("BTC"))
        stable = _positive_or_zero(balances.get("USDT", balances.get("USDC", balances.get("BUSD"))))
        return btc + stable / price

    @staticmethod
    def _slippage(rows: tuple[Mapping[str, Any], ...]) -> float:
        total = 0.0
        notional = 0.0
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, Mapping):
                continue
            quantity = _positive_or_zero(payload.get("quantity"))
            price = _positive_or_zero(payload.get("price"))
            metadata = payload.get("metadata")
            cost = (
                _positive_or_zero(metadata.get("slippage_cost"))
                if isinstance(metadata, Mapping)
                else 0.0
            )
            total += cost
            notional += quantity * price
        return total / notional if notional > 0 else 0.0

    @staticmethod
    def _execution_drift(rows: tuple[Mapping[str, Any], ...]) -> float:
        drifts: list[float] = []
        for row in rows:
            payload = row["payload"]
            if not isinstance(payload, Mapping):
                continue
            price = _positive_or_zero(payload.get("price"))
            metadata = payload.get("metadata")
            reference = _positive_or_zero(
                metadata.get("reference_price") if isinstance(metadata, Mapping) else None
            )
            if price > 0 and reference > 0:
                drifts.append(abs(price / reference - 1.0))
        return max(drifts, default=0.0)

    @staticmethod
    def _model_drift(net_pnl: float, forecast: Mapping[str, Any], has_fills: bool) -> float:
        if not has_fills:
            return 0.0
        expected = _number(forecast.get("expected_return", 0.0), "forecast expected_return")
        scale = max(abs(expected), 1.0)
        return min(1.0, abs(net_pnl - expected) / scale)

    @staticmethod
    def _is_data_gap(row: Mapping[str, Any]) -> bool:
        payload = row["payload"]
        return isinstance(payload, Mapping) and bool(payload.get("data_gap"))


def _number(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _positive_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result > 0 else 0.0


def _order_belongs_to_strategy(
    payload: Mapping[str, Any], *, strategy_version_id: str | None
) -> bool:
    if strategy_version_id is None:
        return True
    declared = payload.get("strategy_version_id")
    if declared is not None:
        return str(declared) == strategy_version_id
    contributions = payload.get("strategy_contributions")
    return isinstance(contributions, Mapping) and strategy_version_id in {
        str(key) for key in contributions
    }


def _order_belongs_to_assignment(payload: Mapping[str, Any], *, assignment_id: str | None) -> bool:
    if assignment_id is None:
        return True
    declared = payload.get("assignment_id")
    metadata = payload.get("metadata")
    if declared is None and isinstance(metadata, Mapping):
        declared = metadata.get("assignment_id")
        target_metadata = metadata.get("target_metadata")
        if declared is None and isinstance(target_metadata, Mapping):
            declared = target_metadata.get("assignment_id")
    return declared is not None and str(declared) == assignment_id


def _assignment_account_id(assignment: Mapping[str, Any]) -> str:
    account_id = str(assignment.get("account_id") or "").strip()
    payload = assignment.get("payload")
    if not account_id and isinstance(payload, Mapping):
        account_id = str(payload.get("account_id") or "").strip()
    return account_id


def _ledger_belongs_to_scope(
    payload: Mapping[str, Any],
    *,
    strategy_version_id: str | None,
    assignment_id: str | None,
    instrument_id: str | None,
) -> bool:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return not any((strategy_version_id, assignment_id, instrument_id))
    if assignment_id is not None:
        declared_assignment = metadata.get("assignment_id")
        if declared_assignment is None or str(declared_assignment) != assignment_id:
            return False
    if strategy_version_id is not None:
        declared_strategy = metadata.get("strategy_version_id", metadata.get("strategy"))
        if declared_strategy is None or str(declared_strategy) not in {
            strategy_version_id,
            "ensemble",
        }:
            return False
    if instrument_id is not None:
        declared_instrument = metadata.get("instrument_id", metadata.get("symbol"))
        if declared_instrument is None or str(declared_instrument) != instrument_id:
            return False
    return True
