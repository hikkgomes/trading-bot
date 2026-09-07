"""Collect immutable, interval-scoped forward-paper evidence facts."""

from __future__ import annotations

import datetime as dt
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
    position,
    position_event,
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
    data_gap_ids: tuple[str, ...] = ()
    objective_unit: str | None = None
    objective_value: float | None = None
    benchmark_value: float | None = None
    objective_excess: float | None = None
    objective_excess_fraction: float | None = None
    trading_days: int = 0
    cycles: int = 0
    effective_independent_episodes: int = 0
    tail_loss: float = 0.0
    evidence_days: float = 0.0
    valid_attempts: int = 0
    completed_attempts: int = 0
    expected_observations: int = 0
    observed_observations: int = 0
    capacity_remaining: float = 0.0
    forecast_return: float = 0.0
    realised_return: float = 0.0
    fill_rate_applicable: bool = False
    trading_dates: tuple[str, ...] = ()
    order_ids: tuple[str, ...] = ()
    fill_ids: tuple[str, ...] = ()
    episode_ids: tuple[str, ...] = ()

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
                "data_gap_ids": list(self.data_gap_ids),
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
                "evidence_days": self.evidence_days,
                "valid_attempts": self.valid_attempts,
                "completed_attempts": self.completed_attempts,
                "expected_observations": self.expected_observations,
                "observed_observations": self.observed_observations,
                "capacity_remaining": self.capacity_remaining,
                "forecast_return": self.forecast_return,
                "realised_return": self.realised_return,
                "fill_rate_applicable": self.fill_rate_applicable,
                "trading_dates": list(self.trading_dates),
                "order_ids": list(self.order_ids),
                "fill_ids": list(self.fill_ids),
                "episode_ids": list(self.episode_ids),
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
        all_orders = self._orders(
            portfolio_id=str(assignment["portfolio_id"]),
            instrument_id=instrument_id,
            start=created_at,
            at=evaluated_at,
            strategy_version_id=strategy_version_id,
            assignment_id=assignment_id,
        )
        all_fills = self._fills(
            tuple(str(row["id"]) for row in all_orders),
            start=created_at,
            at=evaluated_at,
        )
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
        nav_rows = self._nav_rows(
            product_id=product_id,
            portfolio_id=str(assignment.get("portfolio_id") or ""),
            at=evaluated_at,
            assignment_id=assignment_id,
        )
        reconciliations = self._reconciliation_rows(
            product_id=product_id,
            instrument_id=instrument_id,
            start=window_start,
            at=evaluated_at,
        )
        net_pnl = self._ledger_pnl(ledger_rows)
        market_prices = self._market_prices(market_rows)
        initial_equity = self._starting_equity(
            product_id,
            assignment,
            at=created_at,
            price=market_prices[0] if market_prices else None,
        )
        if product_id == "btc_accumulation" and market_prices:
            starting_balances = self._paper_starting_balances(
                assignment=assignment,
                product_id=product_id,
                at=created_at,
            ) or self._paper_starting_balances(
                assignment=assignment,
                product_id=product_id,
                at=evaluated_at,
            )
            if isinstance(starting_balances, Mapping):
                initial_equity = _positive_or_zero(starting_balances.get("BTC")) + (
                    _positive_or_zero(
                        starting_balances.get(
                            "USDT",
                            starting_balances.get("USDC", starting_balances.get("BUSD")),
                        )
                    )
                    / market_prices[0]
                )
        benchmark_pnl = self._benchmark_pnl(
            product_id, market_rows, assignment, initial_equity=initial_equity
        )
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
        model_drift = self._model_drift(
            net_pnl,
            forecast,
            bool(fills),
            initial_equity=initial_equity,
        )
        objective = self._objective_metrics(
            product_id=product_id,
            assignment=assignment,
            artefact_created_at=created_at,
            evaluation_time=evaluated_at,
            ledger_rows=all_ledger_rows,
            market_rows=all_market_rows,
            nav_rows=nav_rows,
            assignment_id=assignment_id,
            fill_rows=all_fills,
        )
        attempted = len(orders)
        completed = sum(1 for row in orders if self._order_filled(row["id"]))
        rejected = sum(1 for row in orders if self._order_rejected(row["id"]))
        data_gap_ids = tuple(
            sorted({str(row["id"]) for row in reconciliations if self._is_data_gap(row)})
        )
        data_gaps = len(data_gap_ids)
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
        capacity, risk_budget = self._capacity_and_budget(
            assignment=assignment,
            market_rows=market_rows,
            orders=orders,
        )
        instrument_count = len(
            {
                str(
                    row["payload"].get("instrument_id")
                    if isinstance(row.get("payload"), Mapping)
                    else ""
                )
                for row in market_rows
                if isinstance(row.get("payload"), Mapping)
            }
        )
        trading_dates = tuple(
            sorted(
                {
                    _utc_day(str(row.get("created_at") or row.get("observed_at")))
                    for row in market_rows
                }
            )
        )
        initial_positions = self._positions_at(
            portfolio_id=str(assignment["portfolio_id"]),
            instrument_id=instrument_id,
            at=window_start,
            assignment_id=assignment_id,
        )
        episode_ids = self._closed_episode_ids(fills, initial_positions=initial_positions)
        cycles = len(episode_ids)
        ledger_returns = tuple(
            value / initial_equity for value in self._ledger_returns(ledger_rows)
        )
        independent_episodes = cycles
        expected_observations = self._expected_observations(
            window_start=window_start,
            evaluated_at=evaluated_at,
            forecast=forecast,
            instrument_count=instrument_count,
        )
        observed_observations = len(
            {
                (
                    str(
                        row["payload"].get("instrument_id")
                        if isinstance(row.get("payload"), Mapping)
                        else ""
                    ),
                    str(row.get("created_at") or row.get("observed_at")),
                )
                for row in market_rows
            }
        )
        if expected_observations > observed_observations and observed_observations == 0:
            data_gap_ids = tuple(
                sorted(
                    {
                        *data_gap_ids,
                        f"schedule-gap:{window_start}:{evaluated_at}",
                    }
                )
            )
            data_gaps = len(data_gap_ids)
        data_uptime = self._data_uptime(
            observed_observations,
            expected_observations=expected_observations,
        )
        realised_return = net_pnl / initial_equity if initial_equity > 0 else 0.0
        return ForwardEvidenceMetrics(
            net_pnl=net_pnl,
            benchmark_pnl=benchmark_pnl,
            drawdown=drawdown,
            execution_drift=execution_drift,
            model_drift=model_drift,
            portfolio_capacity=capacity,
            risk_budget_available=risk_budget,
            data_gaps=data_gaps,
            data_gap_ids=data_gap_ids,
            effective_trades=cycles,
            fill_rate=(completed / attempted if attempted else 0.0),
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
            trading_days=len(trading_dates),
            cycles=cycles,
            effective_independent_episodes=independent_episodes,
            tail_loss=self._tail_loss(ledger_returns),
            evidence_days=max(
                0.0,
                (
                    dt.datetime.fromisoformat(evaluated_at) - dt.datetime.fromisoformat(created_at)
                ).total_seconds()
                / 86_400,
            ),
            valid_attempts=attempted,
            completed_attempts=completed,
            expected_observations=expected_observations,
            observed_observations=observed_observations,
            capacity_remaining=capacity,
            forecast_return=_forecast_return(forecast),
            realised_return=realised_return,
            fill_rate_applicable=attempted > 0,
            trading_dates=trading_dates,
            order_ids=order_ids,
            fill_ids=tuple(str(row["id"]) for row in fills),
            episode_ids=episode_ids,
        )

    def latest_observed_at(
        self,
        *,
        strategy_version_id: str,
        product_id: str,
        instrument_id: str,
        artefact_hash: str,
        assignment_id: str | None = None,
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
            row = next(
                (
                    candidate
                    for candidate in rows
                    if assignment_id is None
                    or _payload_assignment_id(candidate.get("payload")) == str(assignment_id)
                ),
                None,
            )
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

    def _nav_rows(
        self,
        *,
        product_id: str,
        portfolio_id: str,
        at: str,
        assignment_id: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
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
            and (not portfolio_id or str(row["payload"].get("portfolio_id") or "") == portfolio_id)
            and _nav_belongs_to_assignment(row["payload"], assignment_id=assignment_id)
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
    def _closed_episodes(rows: tuple[Mapping[str, Any], ...]) -> int:
        return len(ForwardEvidenceCollector._closed_episode_ids(rows))

    @staticmethod
    def _closed_episode_ids(
        rows: tuple[Mapping[str, Any], ...],
        *,
        initial_positions: Mapping[str, float] | None = None,
    ) -> tuple[str, ...]:
        """Count position round trips, keeping all fills of one order together."""

        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                continue
            order_id = str(payload.get("order_id") or row.get("order_id") or row.get("id"))
            item = grouped.setdefault(
                order_id,
                {
                    "order_id": order_id,
                    "instrument_id": str(
                        payload.get("instrument_id") or payload.get("symbol") or "*"
                    ),
                    "side": str(payload.get("side") or "").casefold(),
                    "quantity": 0.0,
                    "created_at": str(row.get("created_at") or ""),
                },
            )
            item["quantity"] += _positive_or_zero(
                payload.get("quantity", payload.get("filled_quantity"))
            )
        positions: dict[str, float] = {
            str(symbol): float(quantity)
            for symbol, quantity in (initial_positions or {}).items()
            if abs(float(quantity)) > 1e-12
        }
        episodes: list[str] = []
        for item in sorted(grouped.values(), key=lambda value: value["created_at"]):
            symbol = str(item["instrument_id"])
            signed = item["quantity"] if item["side"] == "buy" else -item["quantity"]
            previous = positions.get(symbol, 0.0)
            current = previous + signed
            if previous != 0.0 and (previous > 0) != (current > 0):
                episodes.append(str(item.get("order_id") or symbol))
            if previous != 0.0 and abs(current) <= 1e-12:
                episodes.append(str(item.get("order_id") or symbol))
            positions[symbol] = current
        return tuple(dict.fromkeys(episodes))

    def _positions_at(
        self,
        *,
        portfolio_id: str,
        instrument_id: str,
        at: str,
        assignment_id: str | None = None,
    ) -> dict[str, float]:
        """Return the last durable position state before a forward window."""

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(position_event.c.created_at, position_event.c.payload)
                .where(position_event.c.created_at <= at)
                .order_by(position_event.c.created_at, position_event.c.id)
            ).mappings()
        latest: dict[str, tuple[str, float]] = {}
        for row in rows:
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if str(payload.get("portfolio_id")) != portfolio_id:
                continue
            symbol = str(payload.get("instrument_id") or "")
            if symbol != instrument_id:
                continue
            if assignment_id is not None:
                metadata = payload.get("metadata")
                if not isinstance(metadata, Mapping) or str(
                    metadata.get("assignment_id") or ""
                ) != str(assignment_id):
                    continue
            try:
                quantity = float(payload.get("quantity", 0.0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(quantity):
                latest[symbol] = (str(row["created_at"]), quantity)
        return {symbol: quantity for symbol, (_observed_at, quantity) in latest.items()}

    def _capacity_and_budget(
        self,
        *,
        assignment: Mapping[str, Any],
        market_rows: tuple[Mapping[str, Any], ...],
        orders: tuple[Mapping[str, Any], ...],
    ) -> tuple[float, float]:
        capital = _positive_or_zero(assignment.get("capital_limit"))
        configured_budget = _positive_or_zero(assignment.get("risk_budget"))
        assignment_payload = assignment.get("payload")
        participation = (
            _positive_or_zero(assignment_payload.get("maximum_participation_fraction"))
            if isinstance(assignment_payload, Mapping)
            else 0.01
        ) or 0.01
        liquidity_caps: list[float] = []
        latest_prices: dict[str, float] = {}
        for row in market_rows:
            payload = row.get("payload")
            values = payload.get("values") if isinstance(payload, Mapping) else None
            if not isinstance(values, Mapping):
                continue
            depth = _positive_or_zero(values.get("visible_depth"))
            price = _positive_or_zero(values.get("close", values.get("price")))
            instrument_id = (
                str(payload.get("instrument_id") or "") if isinstance(payload, Mapping) else ""
            )
            if instrument_id and price > 0.0:
                latest_prices[instrument_id] = price
            if depth > 0.0 and price > 0.0:
                liquidity_caps.append(depth * price * min(1.0, participation))
        capacity = min([capital, *liquidity_caps]) if capital > 0.0 and liquidity_caps else 0.0
        current_notional = 0.0
        with self.engine.connect() as connection:
            position_rows = connection.execute(select(position.c.payload)).scalars()
            for payload in position_rows:
                if not isinstance(payload, Mapping) or payload.get(
                    "portfolio_id"
                ) != assignment.get("portfolio_id"):
                    continue
                position_metadata = payload.get("metadata")
                if assignment.get("id") and (
                    not isinstance(position_metadata, Mapping)
                    or str(position_metadata.get("assignment_id") or "")
                    != str(assignment.get("id"))
                ):
                    continue
                instrument_id = str(payload.get("instrument_id") or "")
                current_notional += abs(
                    _number(payload.get("quantity", 0.0), "position quantity")
                ) * latest_prices.get(instrument_id, 0.0)
        pending_notional = sum(
            _positive_or_zero(
                (row.get("payload") or {}).get("quantity")
                if isinstance(row.get("payload"), Mapping)
                else 0.0
            )
            * _positive_or_zero(
                (row.get("payload") or {}).get("price")
                if isinstance(row.get("payload"), Mapping)
                else 0.0
            )
            for row in orders
            if not self._order_filled(row["id"])
        )
        remaining = max(
            0.0,
            min(capacity, configured_budget or capacity) - current_notional - pending_notional,
        )
        return capacity, remaining

    @staticmethod
    def _expected_observations(
        *,
        window_start: str,
        evaluated_at: str,
        forecast: Mapping[str, Any],
        instrument_count: int = 1,
    ) -> int:
        duration = max(
            0.0,
            (
                dt.datetime.fromisoformat(evaluated_at) - dt.datetime.fromisoformat(window_start)
            ).total_seconds(),
        )
        cadence = _positive_or_zero(forecast.get("observation_interval_seconds")) or 300.0
        return max(1, int(math.ceil(duration / cadence))) * max(1, instrument_count)

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
        *,
        initial_equity: float | None = None,
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
        return (prices[-1] / prices[0] - 1.0) * (
            initial_equity
            if initial_equity is not None
            else _positive_or_zero(assignment.get("capital_limit"))
        )

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
        running = base
        peak = base
        worst = 0.0
        for row in rows:
            payload = row["payload"]
            metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
            if not isinstance(metadata, Mapping) or metadata.get("pnl_effect") is None:
                continue
            running += _number(metadata["pnl_effect"], "ledger pnl_effect")
            peak = max(peak, running)
            if peak > 0.0:
                worst = max(worst, (peak - running) / peak)
        return max(0.0, worst)

    @staticmethod
    def _nav_drawdown(rows: tuple[Mapping[str, Any], ...], *, start: str, at: str) -> float:
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
    def _data_uptime(observed: int, *, expected_observations: int) -> float:
        if expected_observations <= 0:
            return 0.0
        return max(0.0, min(1.0, observed / expected_observations))

    def _starting_equity(
        self,
        product_id: str,
        assignment: Mapping[str, Any],
        *,
        at: str | None = None,
        price: float | None = None,
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
                if product_id == "btc_accumulation" and price is not None and price > 0.0:
                    value = (
                        _positive_or_zero(balances.get("BTC"))
                        + _positive_or_zero(
                            balances.get("USDT", balances.get("USDC", balances.get("BUSD")))
                        )
                        / price
                    )
                else:
                    asset = "BTC" if product_id == "btc_accumulation" else "USDT"
                    value = _positive_or_zero(balances.get(asset))
                if value > 0:
                    cap = _positive_or_zero(assignment.get("capital_limit"))
                    return min(value, cap) if cap > 0.0 else value
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
        assignment_id: str | None = None,
        fill_rows: tuple[Mapping[str, Any], ...] = (),
    ) -> tuple[str, float | None, float | None, float | None, float | None, tuple[str, ...]]:
        nav_points = self._objective_nav_points(nav_rows, start=artefact_created_at)
        if product_id == "active_income":
            return self._active_income_objective(
                assignment=assignment,
                artefact_created_at=artefact_created_at,
                ledger_rows=ledger_rows,
                market_rows=market_rows,
                fill_rows=fill_rows,
            )
        if product_id != "btc_accumulation":
            return (product_id, None, None, None, None, ())
        return self._btc_objective(
            assignment=assignment,
            artefact_created_at=artefact_created_at,
            evaluation_time=evaluation_time,
            ledger_rows=ledger_rows,
            market_rows=market_rows,
            nav_rows=nav_rows,
            nav_points=nav_points,
            assignment_id=assignment_id,
            fill_rows=fill_rows,
        )

    def _active_income_objective(
        self,
        *,
        assignment: Mapping[str, Any],
        artefact_created_at: str,
        ledger_rows: tuple[Mapping[str, Any], ...],
        market_rows: tuple[Mapping[str, Any], ...],
        fill_rows: tuple[Mapping[str, Any], ...],
    ) -> tuple[str, float | None, float | None, float | None, float | None, tuple[str, ...]]:
        initial = self._starting_equity("active_income", assignment, at=artefact_created_at)
        exact = _futures_assignment_equity(
            initial_equity=initial,
            ledger_pnl=self._ledger_pnl(ledger_rows),
            fills=fill_rows,
            market_rows=market_rows,
        )
        if exact is not None:
            return (
                "USDT",
                exact,
                initial,
                exact - initial,
                (exact - initial) / initial if initial > 0.0 else None,
                (),
            )
        paper_state = _latest_paper_state(ledger_rows)
        final = (
            _paper_futures_equity(paper_state, market_rows=market_rows)
            if paper_state is not None
            else None
        )
        if final is None:
            final = initial + self._ledger_pnl(ledger_rows)
        excess = final - initial
        return (
            "USDT",
            final,
            initial,
            excess,
            excess / initial if initial > 0.0 else None,
            (),
        )

    def _btc_objective(
        self,
        *,
        assignment: Mapping[str, Any],
        artefact_created_at: str,
        evaluation_time: str,
        ledger_rows: tuple[Mapping[str, Any], ...],
        market_rows: tuple[Mapping[str, Any], ...],
        nav_rows: tuple[Mapping[str, Any], ...],
        nav_points: tuple[float, float, tuple[str, ...]] | None,
        assignment_id: str | None,
        fill_rows: tuple[Mapping[str, Any], ...],
    ) -> tuple[str, float | None, float | None, float | None, float | None, tuple[str, ...]]:
        all_prices = self._market_prices(market_rows)
        if len(all_prices) >= 2:
            return self._btc_paper_objective(
                assignment=assignment,
                artefact_created_at=artefact_created_at,
                ledger_rows=ledger_rows,
                market_rows=market_rows,
                prices=all_prices,
                fill_rows=fill_rows,
                evaluation_time=evaluation_time,
            )
        if nav_points is not None:
            return self._btc_nav_objective(
                nav_points=nav_points,
                nav_rows=nav_rows,
                artefact_created_at=artefact_created_at,
            )
        if assignment_id:
            account_objective = self._btc_account_snapshot_objective(
                assignment=assignment,
                artefact_created_at=artefact_created_at,
                evaluation_time=evaluation_time,
                ledger_rows=ledger_rows,
                market_rows=market_rows,
            )
            if account_objective is not None:
                return account_objective
        return self._btc_snapshot_objective(
            assignment=assignment,
            artefact_created_at=artefact_created_at,
            evaluation_time=evaluation_time,
            market_rows=market_rows,
        )

    def _btc_paper_objective(
        self,
        *,
        assignment: Mapping[str, Any],
        artefact_created_at: str,
        ledger_rows: tuple[Mapping[str, Any], ...],
        market_rows: tuple[Mapping[str, Any], ...],
        prices: tuple[float, ...],
        fill_rows: tuple[Mapping[str, Any], ...],
        evaluation_time: str,
    ) -> tuple[str, float | None, float | None, float | None, float | None, tuple[str, ...]]:
        balances = self._paper_starting_balances(
            assignment=assignment,
            product_id="btc_accumulation",
            at=artefact_created_at,
        )
        if balances is None:
            balances = self._paper_starting_balances(
                assignment=assignment,
                product_id="btc_accumulation",
                at=evaluation_time,
            )
        if not isinstance(balances, Mapping):
            return ("BTC", None, None, None, None, ())
        btc = _positive_or_zero(balances.get("BTC"))
        stable = _positive_or_zero(balances.get("USDT", balances.get("USDC", balances.get("BUSD"))))
        initial = btc + stable / prices[0]
        for row in fill_rows:
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                continue
            btc, stable = _apply_spot_fill_to_balances(btc, stable, payload)
        final = btc + stable / prices[-1]
        benchmark = initial * prices[-1] / prices[0]
        excess = final - benchmark
        source_ids = tuple(dict.fromkeys(str(row["id"]) for row in (*ledger_rows, *market_rows)))
        return (
            "BTC",
            final,
            benchmark,
            excess,
            excess / benchmark if benchmark else None,
            source_ids,
        )

    def _paper_starting_balances(
        self,
        *,
        assignment: Mapping[str, Any],
        product_id: str,
        at: str,
    ) -> Mapping[str, Any] | None:
        account_id = _assignment_account_id(assignment)
        if not account_id:
            payload = assignment.get("payload")
            if isinstance(payload, Mapping):
                account_id = str(payload.get("account_id") or "")
        if not account_id:
            return None
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(accounting_entry.c.payload)
                .where(
                    accounting_entry.c.product_id == product_id,
                    accounting_entry.c.created_at <= at,
                )
                .order_by(accounting_entry.c.created_at, accounting_entry.c.sequence)
            ).scalars()
        first_state: Mapping[str, Any] | None = None
        for payload in rows:
            if (
                isinstance(payload, Mapping)
                and isinstance(payload.get("metadata"), Mapping)
                and payload["metadata"].get("kind") == "paper_account_state"
                and str(payload["metadata"].get("account_id") or "") == account_id
            ):
                balances = payload["metadata"].get("balances")
                if isinstance(balances, Mapping):
                    first_state = first_state or balances
                    if payload["metadata"].get("seed") is True:
                        return balances
        return first_state

    def _btc_nav_objective(
        self,
        *,
        nav_points: tuple[float, float, tuple[str, ...]],
        nav_rows: tuple[Mapping[str, Any], ...],
        artefact_created_at: str,
    ) -> tuple[str, float | None, float | None, float | None, float | None, tuple[str, ...]]:
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
        ) or initial
        excess = final - benchmark
        return (
            "BTC",
            final,
            benchmark,
            excess,
            excess / benchmark if benchmark else None,
            source_ids,
        )

    def _btc_account_snapshot_objective(
        self,
        *,
        assignment: Mapping[str, Any],
        artefact_created_at: str,
        evaluation_time: str,
        ledger_rows: tuple[Mapping[str, Any], ...],
        market_rows: tuple[Mapping[str, Any], ...],
    ) -> tuple[str, float | None, float | None, float | None, float | None, tuple[str, ...]] | None:
        snapshots = self._account_snapshot_rows(
            account_id=_assignment_account_id(assignment), at=evaluation_time
        )
        prices = self._market_prices(market_rows)
        if not snapshots or not prices:
            return None
        first_snapshot = next(
            (row for row in snapshots if str(row["observed_at"]) <= artefact_created_at),
            snapshots[0],
        )
        initial = self._btc_nav(first_snapshot["payload"], prices[0])
        if initial is None or initial <= 0.0:
            return None
        benchmark = initial * prices[-1] / prices[0]
        final = initial + self._ledger_pnl(ledger_rows)
        excess = final - benchmark
        source_ids = tuple(str(row["id"]) for row in (*snapshots, *ledger_rows, *market_rows))
        return (
            "BTC",
            final,
            benchmark,
            excess,
            excess / benchmark,
            tuple(dict.fromkeys(source_ids)),
        )

    def _btc_snapshot_objective(
        self,
        *,
        assignment: Mapping[str, Any],
        artefact_created_at: str,
        evaluation_time: str,
        market_rows: tuple[Mapping[str, Any], ...],
    ) -> tuple[str, float | None, float | None, float | None, float | None, tuple[str, ...]]:
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
        initial = self._btc_nav(first_snapshot["payload"], prices[0])
        final = self._btc_nav(snapshots[-1]["payload"], prices[-1])
        if initial is None or final is None or initial <= 0.0:
            return ("BTC", None, None, None, None, source_ids)
        excess = final - initial
        return ("BTC", final, initial, excess, excess / initial, source_ids)

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
    def _model_drift(
        net_pnl: float,
        forecast: Mapping[str, Any],
        has_fills: bool,
        *,
        initial_equity: float,
    ) -> float:
        if not has_fills:
            return 0.0
        expected = _number(forecast.get("expected_return", 0.0), "forecast expected_return")
        realised = net_pnl / initial_equity if initial_equity > 0.0 else 0.0
        scale = max(abs(expected), 0.01)
        return min(1.0, abs(realised - expected) / scale)

    @staticmethod
    def _is_data_gap(row: Mapping[str, Any]) -> bool:
        payload = row["payload"]
        return (
            isinstance(payload, Mapping)
            and bool(payload.get("data_gap"))
            and not bool(payload.get("repaired"))
        )


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


def _utc_day(value: str) -> str:
    try:
        return dt.datetime.fromisoformat(timestamp(value, field="trading date")).date().isoformat()
    except ValueError:
        return value[:10]


def _forecast_return(forecast: Mapping[str, Any]) -> float:
    raw = forecast.get("expected_return", 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("forecast expected_return must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError("forecast expected_return must be finite")
    return value


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


def _nav_belongs_to_assignment(payload: Mapping[str, Any], *, assignment_id: str | None) -> bool:
    if assignment_id is None:
        return True
    declared = payload.get("assignment_id")
    components = payload.get("components")
    if declared is None and isinstance(components, Mapping):
        declared = components.get("assignment_id")
    return declared is not None and str(declared) == str(assignment_id)


def _payload_assignment_id(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    declared = payload.get("assignment_id")
    if declared is not None:
        return str(declared)
    observation = payload.get("observation")
    if isinstance(observation, Mapping) and observation.get("assignment_id") is not None:
        return str(observation["assignment_id"])
    return None


def _assignment_account_id(assignment: Mapping[str, Any]) -> str:
    account_id = str(assignment.get("account_id") or "").strip()
    payload = assignment.get("payload")
    if not account_id and isinstance(payload, Mapping):
        account_id = str(payload.get("account_id") or "").strip()
    return account_id


def _latest_paper_state(rows: tuple[Mapping[str, Any], ...]) -> Mapping[str, Any] | None:
    latest: Mapping[str, Any] | None = None
    for row in rows:
        payload = row.get("payload")
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        if isinstance(metadata, Mapping) and metadata.get("kind") == "paper_account_state":
            latest = metadata
    return latest


def _apply_spot_fill_to_balances(
    btc: float, stable: float, payload: Mapping[str, Any]
) -> tuple[float, float]:
    side = str(payload.get("side") or "").casefold()
    quantity = _positive_or_zero(payload.get("quantity"))
    price = _positive_or_zero(payload.get("price"))
    fee = _positive_or_zero(payload.get("fee"))
    if side not in {"buy", "sell"} or quantity <= 0.0 or price <= 0.0:
        return btc, stable
    fee_asset = str(payload.get("fee_asset") or "USDT").upper()
    if side == "buy":
        btc += quantity
        stable -= quantity * price
    else:
        btc -= quantity
        stable += quantity * price
    if fee_asset == "BTC":
        btc -= fee
    elif fee_asset in {"USDT", "USDC", "BUSD"}:
        stable -= fee
    if btc < -1e-12 or stable < -1e-12:
        raise ValueError("strategy-scoped BTC paper balance became negative")
    return btc, stable


def _paper_futures_equity(
    state: Mapping[str, Any], *, market_rows: tuple[Mapping[str, Any], ...]
) -> float | None:
    balances = state.get("balances")
    positions = state.get("positions")
    entry_prices = state.get("position_entry_prices", {})
    if not isinstance(balances, Mapping) or not isinstance(positions, Mapping):
        return None
    if not isinstance(entry_prices, Mapping):
        entry_prices = {}
    latest_prices: dict[str, float] = {}
    for row in market_rows:
        payload = row.get("payload")
        values = payload.get("values") if isinstance(payload, Mapping) else None
        instrument_id = (
            str(payload.get("instrument_id") or "") if isinstance(payload, Mapping) else ""
        )
        price = _positive_or_zero(values.get("close")) if isinstance(values, Mapping) else 0.0
        if instrument_id and price > 0.0:
            latest_prices[instrument_id] = price
    equity = _positive_or_zero(balances.get("USDT"))
    for instrument_id, raw_quantity in positions.items():
        quantity = _number(raw_quantity, "paper position quantity")
        if abs(quantity) <= 1e-12:
            continue
        entry = _positive_or_zero(entry_prices.get(instrument_id))
        mark = latest_prices.get(str(instrument_id))
        if entry <= 0.0 or mark is None:
            return None
        equity += quantity * (mark - entry)
    return equity if math.isfinite(equity) and equity >= 0.0 else None


def _futures_assignment_equity(
    *,
    initial_equity: float,
    ledger_pnl: float,
    fills: tuple[Mapping[str, Any], ...],
    market_rows: tuple[Mapping[str, Any], ...],
) -> float | None:
    """Value only one assignment's fills, excluding other paper strategies."""

    if not fills:
        return None
    quantity = 0.0
    entry_price = 0.0
    for row in fills:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        quantity, entry_price = _apply_assignment_fill(quantity, entry_price, payload)
    if abs(quantity) <= 1e-12:
        return initial_equity + ledger_pnl
    latest_price = _latest_market_price(market_rows)
    if latest_price is None or entry_price <= 0.0:
        return None
    result = initial_equity + ledger_pnl + quantity * (latest_price - entry_price)
    return result if math.isfinite(result) and result >= 0.0 else None


def _apply_assignment_fill(
    quantity: float, entry_price: float, payload: Mapping[str, Any]
) -> tuple[float, float]:
    side = str(payload.get("side") or "").casefold()
    if side not in {"buy", "sell"}:
        return quantity, entry_price
    amount = _positive_or_zero(payload.get("quantity"))
    price = _positive_or_zero(payload.get("price"))
    if amount <= 0.0 or price <= 0.0:
        return quantity, entry_price
    signed = amount if side == "buy" else -amount
    if abs(quantity) <= 1e-12 or quantity * signed >= 0.0:
        total = abs(quantity) + abs(signed)
        entry_price = (
            price if total <= 1e-12 else (abs(quantity) * entry_price + abs(signed) * price) / total
        )
    else:
        remaining = quantity + signed
        if abs(remaining) <= 1e-12 or quantity * remaining < 0.0:
            entry_price = 0.0 if abs(remaining) <= 1e-12 else price
    return quantity + signed, entry_price


def _latest_market_price(rows: tuple[Mapping[str, Any], ...]) -> float | None:
    prices = ForwardEvidenceCollector._market_prices(rows)
    return prices[-1] if prices else None


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
        declared_strategies = metadata.get("strategy_version_ids")
        if declared_strategy is None and isinstance(declared_strategies, list | tuple):
            if strategy_version_id not in {str(value) for value in declared_strategies}:
                return False
        elif declared_strategy is None or str(declared_strategy) != strategy_version_id:
            return False
    if instrument_id is not None:
        declared_instrument = metadata.get("instrument_id", metadata.get("symbol"))
        if declared_instrument is None or str(declared_instrument) != instrument_id:
            return False
    return True
