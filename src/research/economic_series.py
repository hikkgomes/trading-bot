"""Return statistics derived from timed, cash-flow-adjusted account marks."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.research.returns import PositionReturnReport

ECONOMIC_ENGINE_VERSION = "economic-engine/v4"


def completed_campaigns(accounting: Mapping[str, Any] | None) -> int | None:
    if accounting is None or accounting.get("event_receipts") is None:
        return None
    receipts = accounting["event_receipts"]
    if accounting["objective_unit"] == "BTC":
        return int(accounting["cycles"])
    previous: dict[str, float] = {}
    completed = 0
    for receipt in receipts:
        if "positions" not in receipt:
            return None
        current = {
            symbol: float(value["quantity"]) for symbol, value in receipt["positions"].items()
        }
        current = {
            symbol: quantity if abs(quantity) > 1e-12 else 0.0
            for symbol, quantity in current.items()
        }
        completed += sum(
            quantity != 0
            and (abs(current.get(symbol, 0)) <= 1e-12 or quantity * current.get(symbol, 0) < 0)
            for symbol, quantity in previous.items()
        )
        previous = current
    return completed


def account_return_series(accounting: Mapping[str, Any]) -> PositionReturnReport:
    initial = float(accounting["initial_value"])
    if initial <= 0:
        raise ValueError("economic return series requires positive initial resources")
    marks = accounting["nav_series"]
    if not isinstance(marks, Sequence) or not marks:
        raise ValueError("economic return series requires timed NAV marks")
    previous = initial
    values: list[float] = []
    peak = 1.0
    wealth = 1.0
    drawdown = 0.0
    for row in marks:
        nav = float(row["nav"])
        flow = float(row.get("external_flow", 0.0))
        if previous <= 0 or not math.isfinite(nav):
            raise ValueError("economic return series has invalid NAV")
        value = (nav - flow) / previous - 1.0
        values.append(value)
        previous = nav
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        drawdown = max(drawdown, (peak - wealth) / peak)
    fees = float(accounting["fees"]) / initial
    slippage = float(accounting.get("slippage_cost", 0.0)) / initial
    funding = float(accounting.get("funding_pnl", 0.0)) / initial
    return PositionReturnReport(
        positions=(),
        market_returns=(),
        gross_returns=(),
        net_returns=tuple(values),
        turnover=float(accounting.get("turnover_notional", 0.0)) / initial,
        fees=fees,
        slippage=slippage,
        funding_pnl=funding,
        gross_pnl=wealth - 1.0 + fees + slippage - funding,
        net_pnl=wealth - 1.0,
        maximum_drawdown=drawdown,
        terminal_exposure=float(accounting.get("terminal_exposure", 0.0)) / initial,
        terminal_liquidation_cost=float(accounting.get("terminal_liquidation_cost", 0.0)) / initial,
    )


def btc_nav_series(report: Any) -> list[dict[str, Any]]:
    previous_benchmark = report.initial_benchmark_btc_nav
    rows = []
    for mark in report.nav_series:
        rows.append(
            {
                "timestamp": mark.observed_at,
                "nav": mark.btc_nav,
                "benchmark_nav": mark.passive_btc_nav,
                "cash": mark.stablecoin_balance,
                "positions": {"BTC": mark.btc_balance},
                "mark": mark.stablecoin_per_btc,
                "external_flow": mark.passive_btc_nav - previous_benchmark,
            }
        )
        previous_benchmark = mark.passive_btc_nav
    return rows


def futures_nav_series(report: Any) -> list[dict[str, Any]]:
    mark_times = {
        event["occurred_at"] for event in report.event_receipts if event["event_type"] == "mark"
    }
    if mark_times and report.event_receipts:
        mark_times.add(report.event_receipts[-1]["occurred_at"])
    by_time = {}
    for event in report.event_receipts:
        if mark_times and event["occurred_at"] not in mark_times:
            continue
        by_time[event["occurred_at"]] = {
            "timestamp": event["occurred_at"],
            "nav": event["equity"],
            "cash": event["cash"],
            "positions": event["positions"],
        }
    rows = [by_time[key] for key in sorted(by_time)]
    if rows:
        rows[-1] = {**rows[-1], "nav": report.final_equity}
    return rows
