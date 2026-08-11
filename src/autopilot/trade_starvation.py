"""Durable rolling diagnosis of why configured products do not trade."""

from __future__ import annotations

import csv
import datetime as dt
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.io import append_json_line, write_json_atomic, write_text_atomic

CYCLE_SCHEMA = "autopilot.trade_starvation_cycle/v1"
REPORT_SCHEMA = "autopilot.trade_starvation_report/v1"
MAX_HISTORY_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_TRADE_LOG_BYTES = 64 * 1024 * 1024

SIGNAL_OUTCOMES = {
    "alpha_ensemble_conflict",
    "alpha_ensemble_not_selected",
    "alpha_ensemble_rejected",
    "entry_opened",
    "portfolio_rejected",
}
SIGNAL_EVALUATED_OUTCOMES = SIGNAL_OUTCOMES | {"signal_not_triggered"}
RISK_REJECTION_OUTCOMES = {
    "cooldown",
    "daily_stop",
    "daily_trade_limit",
    "drawdown_halted",
    "macro_data_unavailable",
    "macro_regime_blocked",
    "portfolio_rejected",
    "position_capacity_blocked",
    "strategy_inactive",
}
TRIGGER_FAILURE_STAGES = {
    "conditions",
    "frozen_ml_threshold",
    "score_threshold",
    "trigger",
}


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def _timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


def _counter_payload(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        normalized = _non_negative_int(count)
        if normalized:
            result[str(key)] = normalized
    return result


def cycle_record(report: Mapping[str, Any], *, generated_at: dt.datetime | None = None) -> dict:
    """Reduce a supervisor report to bounded starvation evidence."""
    timestamp = generated_at or _utc_now()
    products: list[dict[str, Any]] = []
    raw_products = report.get("products")
    if not isinstance(raw_products, list):
        raw_products = []
    for raw in raw_products:
        if not isinstance(raw, Mapping):
            continue
        identity = raw.get("product")
        if not isinstance(identity, Mapping):
            continue
        trace = raw.get("decision_trace")
        trace = trace if isinstance(trace, Mapping) else {}
        summary = trace.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        decisions = trace.get("strategies")
        decisions = decisions if isinstance(decisions, Mapping) else {}
        stages: Counter[str] = Counter()
        predicates: Counter[str] = Counter()
        regime_eligible = 0
        setup_matches = 0
        trigger_matches = 0
        last_signal_at: dt.datetime | None = None
        last_entry_at: dt.datetime | None = None
        for decision in decisions.values():
            if not isinstance(decision, Mapping):
                continue
            outcome = str(decision.get("outcome") or "")
            failed_stage = decision.get("failed_stage")
            if isinstance(failed_stage, str) and failed_stage:
                stages[failed_stage] += 1
            failed_predicate = decision.get("failed_predicate")
            if isinstance(failed_predicate, str) and failed_predicate:
                predicates[failed_predicate] += 1
            latest_bar = _timestamp(decision.get("latest_bar"))
            if outcome in SIGNAL_OUTCOMES and latest_bar is not None:
                last_signal_at = (
                    latest_bar
                    if last_signal_at is None or latest_bar > last_signal_at
                    else last_signal_at
                )
            if outcome == "entry_opened" and latest_bar is not None:
                last_entry_at = (
                    latest_bar
                    if last_entry_at is None or latest_bar > last_entry_at
                    else last_entry_at
                )
            if outcome not in SIGNAL_EVALUATED_OUTCOMES:
                continue
            if failed_stage != "regime":
                regime_eligible += 1
            if failed_stage not in {"regime", "setup"}:
                setup_matches += 1
            if failed_stage not in {"regime", "setup", *TRIGGER_FAILURE_STAGES}:
                trigger_matches += 1
        outcomes = _counter_payload(summary.get("outcomes"))
        market_bars: list[dict[str, str]] = []
        raw_market_bars = summary.get("market_bars")
        if isinstance(raw_market_bars, list):
            seen_market_bars: set[tuple[str, str]] = set()
            for item in raw_market_bars:
                if not isinstance(item, Mapping):
                    continue
                timeframe = str(item.get("timeframe") or "")
                bar_timestamp = _timestamp(item.get("timestamp"))
                if not timeframe or bar_timestamp is None:
                    continue
                key = (timeframe, bar_timestamp.isoformat())
                if key in seen_market_bars:
                    continue
                seen_market_bars.add(key)
                market_bars.append({"timeframe": key[0], "timestamp": key[1]})
        cycle_errors = raw.get("cycle_errors")
        products.append(
            {
                "name": str(identity.get("name") or ""),
                "objective": str(identity.get("objective") or ""),
                "market": str(identity.get("market") or ""),
                "symbol": str(identity.get("symbol") or "").upper(),
                "entries_allowed": raw.get("entries_allowed") is True,
                "entry_gate_reason": (
                    str(raw.get("entry_gate", {}).get("reason") or "")
                    if isinstance(raw.get("entry_gate"), Mapping)
                    else ""
                ),
                "strategies_loaded": _non_negative_int(summary.get("strategies")),
                "data_ready": _non_negative_int(summary.get("data_ready")),
                "market_bars": market_bars,
                "regime_eligible": regime_eligible,
                "setup_matches": setup_matches,
                "trigger_matches": trigger_matches,
                "signals": _non_negative_int(summary.get("signals")),
                "entries_opened": _non_negative_int(summary.get("entries_opened")),
                "positions_managed": _non_negative_int(summary.get("positions_managed")),
                "outcomes": outcomes,
                "failed_stages": dict(stages),
                "killer_predicates": dict(predicates),
                "cycle_errors": len(cycle_errors) if isinstance(cycle_errors, list) else 0,
                "last_signal_at": last_signal_at.isoformat() if last_signal_at else None,
                "last_entry_at": last_entry_at.isoformat() if last_entry_at else None,
            }
        )
    return {
        "schema": CYCLE_SCHEMA,
        "generated_at": timestamp.isoformat(),
        "products": products,
    }


def _load_history(path: Path, *, cutoff: dt.datetime) -> tuple[list[dict], bool]:
    if not path.exists():
        return [], False
    if path.is_symlink() or not path.is_file():
        raise ValueError("trade-starvation history must be a regular non-symlink file")
    if path.stat().st_size > MAX_HISTORY_BYTES:
        raise ValueError("trade-starvation history exceeds its size budget")
    records: list[dict] = []
    stale = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(line.encode()) > MAX_LINE_BYTES:
                raise ValueError(f"trade-starvation history line {line_number} is too large")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"trade-starvation history line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(payload, dict) or payload.get("schema") != CYCLE_SCHEMA:
                raise ValueError(f"trade-starvation history line {line_number} has invalid schema")
            generated_at = _timestamp(payload.get("generated_at"))
            if generated_at is None:
                raise ValueError(
                    f"trade-starvation history line {line_number} has invalid timestamp"
                )
            if generated_at < cutoff:
                stale = True
                continue
            records.append(payload)
    return records, stale


def _compact_history(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )
    write_text_atomic(path, content)


def _completed_trades(product: ProductConfig, cutoff: dt.datetime) -> tuple[int, str | None]:
    path = product.trade_log
    if not path.exists():
        return 0, None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{product.name} trade log must be a regular non-symlink file")
    if path.stat().st_size > MAX_TRADE_LOG_BYTES:
        raise ValueError(f"{product.name} trade log exceeds diagnostic size budget")
    count = 0
    last: dt.datetime | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            exited = _timestamp(row.get("exit_time"))
            if exited is None or exited < cutoff:
                continue
            count += 1
            last = exited if last is None or exited > last else last
    return count, last.isoformat() if last is not None else None


def _dominant(counter: Counter[str]) -> str | None:
    return counter.most_common(1)[0][0] if counter else None


def _starvation_point(summary: Mapping[str, Any]) -> str:
    if _non_negative_int(summary.get("entry_enabled_cycles")) == 0:
        return "entry_gate"
    if _non_negative_int(summary.get("data_ready")) == 0:
        return "market_data_or_feature_build"
    if _non_negative_int(summary.get("signals")) == 0:
        stage = summary.get("dominant_failed_stage")
        return f"signal_generation:{stage}" if stage else "signal_generation"
    if _non_negative_int(summary.get("entries_opened")) == 0:
        return "portfolio_risk_or_execution_gate"
    if _non_negative_int(summary.get("completed_trades")) == 0:
        return "open_position_or_exit_horizon"
    return "not_starved"


def build_report(
    records: Iterable[Mapping[str, Any]],
    products: Iterable[ProductConfig],
    *,
    cutoff: dt.datetime,
    generated_at: dt.datetime,
) -> dict[str, Any]:
    record_list = list(records)
    output = []
    for product in products:
        outcomes: Counter[str] = Counter()
        stages: Counter[str] = Counter()
        predicates: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        market_bars: set[tuple[str, str]] = set()
        last_signal: str | None = None
        last_entry: str | None = None
        for cycle in record_list:
            raw_products = cycle.get("products")
            if not isinstance(raw_products, list):
                continue
            for row in raw_products:
                if not isinstance(row, Mapping) or row.get("name") != product.name:
                    continue
                generated = str(cycle.get("generated_at") or "")
                outcomes.update(_counter_payload(row.get("outcomes")))
                stages.update(_counter_payload(row.get("failed_stages")))
                predicates.update(_counter_payload(row.get("killer_predicates")))
                totals["cycles"] += 1
                totals["entry_enabled_cycles"] += int(row.get("entries_allowed") is True)
                raw_market_bars = row.get("market_bars")
                if isinstance(raw_market_bars, list):
                    for item in raw_market_bars:
                        if not isinstance(item, Mapping):
                            continue
                        timeframe = str(item.get("timeframe") or "")
                        timestamp = _timestamp(item.get("timestamp"))
                        if timeframe and timestamp is not None:
                            market_bars.add((timeframe, timestamp.isoformat()))
                for key in (
                    "strategies_loaded",
                    "data_ready",
                    "regime_eligible",
                    "setup_matches",
                    "trigger_matches",
                    "signals",
                    "entries_opened",
                    "positions_managed",
                    "cycle_errors",
                ):
                    totals[key] += _non_negative_int(row.get(key))
                if _non_negative_int(row.get("signals")):
                    last_signal = str(row.get("last_signal_at") or generated)
                if _non_negative_int(row.get("entries_opened")):
                    last_entry = str(row.get("last_entry_at") or generated)
        completed_trades, last_trade = _completed_trades(product, cutoff)
        summary: dict[str, Any] = {
            **dict(totals),
            "market_bars_processed": len(market_bars),
            "completed_trades": completed_trades,
            "risk_rejected": sum(outcomes[name] for name in RISK_REJECTION_OUTCOMES),
            "execution_rejected": totals["cycle_errors"],
            "outcomes": dict(outcomes),
            "failed_stages": dict(stages),
            "killer_predicates": [
                {"predicate": name, "count": count} for name, count in predicates.most_common(10)
            ],
            "dominant_failed_stage": _dominant(stages),
            "last_signal": last_signal,
            "last_entry": last_entry,
            "last_trade": last_trade,
        }
        summary["starvation_point"] = _starvation_point(summary)
        output.append(
            {
                "product": product.name,
                "objective": product.objective,
                "market": product.market,
                "symbol": product.symbol.upper(),
                "summary": summary,
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "generated_at": generated_at.isoformat(),
        "window_start": cutoff.isoformat(),
        "window_end": generated_at.isoformat(),
        "products": output,
    }


def update_trade_starvation(
    config: AutopilotConfig,
    supervisor_report: Mapping[str, Any],
    *,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    now = (now or _utc_now()).astimezone(dt.UTC).replace(microsecond=0)
    cutoff = now - dt.timedelta(days=config.trade_starvation_window_days)
    record = cycle_record(supervisor_report, generated_at=now)
    append_json_line(config.trade_starvation_history_file, record)
    records, stale = _load_history(config.trade_starvation_history_file, cutoff=cutoff)
    if stale:
        _compact_history(config.trade_starvation_history_file, records)
    report = build_report(records, config.products, cutoff=cutoff, generated_at=now)
    write_json_atomic(config.trade_starvation_report_file, report)
    return report
