"""Stateful zero-money forward paper for autonomous relative-value forecasts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.autopilot.io import write_json_atomic
from src.autopilot.portfolio import ALPHA_FORECAST_SCHEMA
from src.config import PROJECT_ROOT, candle_data_dir

RESEARCH_SCHEMA = "autopilot.relative_value_research/v1"
MULTI_LEG_SCHEMA = "autopilot.multi_leg_alpha_forecast/v1"
REPORT_SCHEMA = "autopilot.relative_value_paper/v1"
STATE_SCHEMA = "autopilot.relative_value_paper_state/v1"
DEFAULT_INPUT = PROJECT_ROOT / "runtime" / "research" / "relative_value.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime" / "research" / "relative_value_paper.json"
DEFAULT_STATE = PROJECT_ROOT / "runtime" / "research" / "relative_value_paper_state.json"
ROUND_TRIP_COST_FRACTION = 0.0024
ANNUAL_BORROW_COST_FRACTION = 0.05


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _load_json(path: Path, *, required: bool) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"path must not be a symlink: {path}")
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _latest_price(symbol: str, market: str, timeframe: str) -> tuple[float, pd.Timestamp]:
    path = candle_data_dir(symbol, market, legacy_fallback=True) / f"{symbol}_{timeframe}.parquet"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["timestamp", "close"]).tail(1)
    if len(frame) != 1:
        raise ValueError(f"price history is empty: {path}")
    price = float(frame.iloc[0]["close"])
    timestamp = pd.Timestamp(frame.iloc[0]["timestamp"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"latest close is invalid: {path}")
    return price, timestamp


def _forecasts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema") != RESEARCH_SCHEMA or payload.get("ok") is not True:
        raise ValueError("relative-value research report is invalid")
    groups = payload.get("forecasts")
    if not isinstance(groups, dict):
        raise ValueError("relative-value forecast groups are invalid")
    forecasts = []
    for family in ("spot_perp_basis", "cross_sectional", "statistical_pairs"):
        values = groups.get(family, [])
        if not isinstance(values, list):
            raise ValueError(f"relative-value {family} forecasts must be a list")
        forecasts.extend(value for value in values if isinstance(value, dict))
    return forecasts


def _legs(forecast: dict[str, Any]) -> list[dict[str, Any]]:
    schema = forecast.get("schema")
    if schema == MULTI_LEG_SCHEMA:
        if (
            forecast.get("promotion_eligible") is not False
            or forecast.get("live_allowed") is not False
        ):
            raise ValueError("multi-leg forecast is not safely isolated")
        raw = forecast.get("legs")
        if not isinstance(raw, list) or len(raw) < 2:
            raise ValueError("multi-leg forecast has invalid legs")
        return [dict(leg) for leg in raw]
    if schema == ALPHA_FORECAST_SCHEMA:
        if (
            forecast.get("promotion_eligible") is not False
            or forecast.get("live_allowed") is not False
        ):
            raise ValueError("cross-sectional forecast is not safely isolated")
        return [
            {
                "market": str(forecast["market"]),
                "symbol": str(forecast["symbol"]),
                "side": "buy" if forecast["direction"] == "long" else "sell",
                "weight": 1.0,
            }
        ]
    raise ValueError("unsupported relative-value forecast schema")


def _prices(legs: list[dict[str, Any]], timeframe: str) -> tuple[dict[str, float], pd.Timestamp]:
    prices: dict[str, float] = {}
    timestamps = []
    for leg in legs:
        market = str(leg["market"])
        symbol = str(leg["symbol"]).upper()
        price, timestamp = _latest_price(symbol, market, timeframe)
        prices[f"{market}:{symbol}"] = price
        timestamps.append(timestamp)
    return prices, min(timestamps)


def _position_return(
    position: dict[str, Any], prices: dict[str, float], observed_at: pd.Timestamp
) -> tuple[float, dict[str, float]]:
    leg_returns: dict[str, float] = {}
    total = 0.0
    for leg in position["legs"]:
        key = f"{leg['market']}:{leg['symbol']}"
        entry = float(position["entry_prices"][key])
        signed = 1.0 if leg["side"] == "buy" else -1.0
        value = signed * (float(prices[key]) / entry - 1.0) * float(leg["weight"])
        leg_returns[key] = value
        total += value
    entry_at = pd.Timestamp(position["entry_observed_at"])
    holding_seconds = max(0.0, (observed_at - entry_at).total_seconds())
    metadata = position.get("metadata") if isinstance(position.get("metadata"), dict) else {}
    funding_rate = float(metadata.get("funding_rate_per_8h") or 0.0)
    if funding_rate:
        for leg in position["legs"]:
            if leg["market"] != "futures":
                continue
            signed = 1.0 if leg["side"] == "buy" else -1.0
            total += -signed * funding_rate * holding_seconds / (8 * 3600) * float(leg["weight"])
    if position.get("requires_borrow"):
        total -= ANNUAL_BORROW_COST_FRACTION * holding_seconds / (365.25 * 86400)
    return total - ROUND_TRIP_COST_FRACTION, leg_returns


def _close_existing_positions(
    sources: dict[str, Any],
    active_ids: set[str],
    *,
    timeframe: str,
    results: list[dict[str, Any]],
) -> None:
    for source_id, source_state in list(sources.items()):
        if not isinstance(source_state, dict) or not isinstance(source_state.get("position"), dict):
            continue
        position = source_state["position"]
        try:
            prices, observed_at = _prices(position["legs"], timeframe)
            matured = (
                observed_at - pd.Timestamp(position["entry_observed_at"])
            ).total_seconds() >= int(position["horizon_seconds"])
            disappeared = source_id not in active_ids
            if not (matured or disappeared):
                continue
            net_return, leg_returns = _position_return(position, prices, observed_at)
            source_state["equity"] = float(source_state.get("equity", 1.0)) * (1 + net_return)
            source_state["peak_equity"] = max(
                float(source_state.get("peak_equity", 1.0)), float(source_state["equity"])
            )
            source_state.setdefault("trades", []).append(
                {
                    "source_report": position["source_report"],
                    "entry_observed_at": position["entry_observed_at"],
                    "exit_observed_at": observed_at.isoformat(),
                    "net_return": net_return,
                    "leg_returns": leg_returns,
                    "reason": "horizon" if matured else "signal_disappeared",
                }
            )
            source_state["trades"] = source_state["trades"][-200:]
            source_state["position"] = None
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            results.append({"source_id": source_id, "status": "waiting", "detail": str(exc)})


def _new_relative_value_state(sources: dict[str, Any], source_id: str) -> dict[str, Any]:
    return sources.setdefault(
        source_id,
        {
            "equity": 1.0,
            "peak_equity": 1.0,
            "position": None,
            "trades": [],
            "last_opened_report": None,
        },
    )


def _open_forecast_position(
    source: dict[str, Any],
    source_state: dict[str, Any],
    forecast: dict[str, Any],
    *,
    source_id: str,
    timeframe: str,
) -> None:
    legs = _legs(forecast)
    prices, observed_at = _prices(legs, timeframe)
    source_state["position"] = {
        "source_report": source["generated_at"],
        "entry_observed_at": observed_at.isoformat(),
        "entry_prices": prices,
        "horizon_seconds": int(forecast["horizon_seconds"]),
        "requires_borrow": bool(forecast.get("requires_borrow")),
        "metadata": forecast.get("metadata") or {},
        "legs": legs,
    }
    source_state["last_opened_report"] = source["generated_at"]


def _open_relative_value_forecasts(
    source: dict[str, Any],
    forecasts: list[dict[str, Any]],
    sources: dict[str, Any],
    *,
    timeframe: str,
    results: list[dict[str, Any]],
) -> None:
    for forecast in forecasts:
        source_id = str(forecast.get("source_id") or "")
        source_state = _new_relative_value_state(sources, source_id)
        if source_state.get("position") is not None:
            results.append({"source_id": source_id, "status": "open"})
            continue
        if source_state.get("last_opened_report") == source["generated_at"]:
            results.append({"source_id": source_id, "status": "observed"})
            continue
        try:
            _open_forecast_position(
                source,
                source_state,
                forecast,
                source_id=source_id,
                timeframe=timeframe,
            )
            results.append({"source_id": source_id, "status": "opened"})
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            results.append({"source_id": source_id, "status": "waiting", "detail": str(exc)})


def run_cycle(
    *,
    input_path: Path = DEFAULT_INPUT,
    state_path: Path = DEFAULT_STATE,
    output_path: Path = DEFAULT_OUTPUT,
    timeframe: str = "1h",
    maximum_signal_age_seconds: int = 43_200,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    if not timeframe or maximum_signal_age_seconds <= 0:
        raise ValueError("relative-value paper timeframe/maximum age is invalid")
    now = now or _utc_now()
    generated_at = now.replace(microsecond=0).isoformat()
    if not input_path.exists():
        report = {
            "schema": REPORT_SCHEMA,
            "ok": True,
            "status": "waiting_for_forecasts",
            "generated_at": generated_at,
            "summary": {},
            "safety": _safety(),
        }
        write_json_atomic(output_path, report)
        return report
    source = _load_json(input_path, required=True)
    source_time = dt.datetime.fromisoformat(str(source["generated_at"]).replace("Z", "+00:00"))
    if source_time.tzinfo is None:
        source_time = source_time.replace(tzinfo=dt.UTC)
    source_age = (now - source_time).total_seconds()
    if not 0 <= source_age <= maximum_signal_age_seconds:
        raise ValueError("relative-value research report is stale or from the future")
    forecasts = _forecasts(source)
    state = _load_json(state_path, required=False)
    if state.get("schema") != STATE_SCHEMA:
        state = {"schema": STATE_SCHEMA, "sources": {}}
    sources = state.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("relative-value paper state sources must be an object")
    results = []
    active_ids = {str(forecast.get("source_id")) for forecast in forecasts}
    _close_existing_positions(sources, active_ids, timeframe=timeframe, results=results)
    _open_relative_value_forecasts(
        source,
        forecasts,
        sources,
        timeframe=timeframe,
        results=results,
    )
    state["updated_at"] = generated_at
    write_json_atomic(state_path, state)
    report = {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "status": "ready" if forecasts else "waiting_for_forecasts",
        "generated_at": generated_at,
        "source_report": source["generated_at"],
        "results": results,
        "summary": {
            "forecasts": len(forecasts),
            "open_positions": sum(
                isinstance(item, dict) and isinstance(item.get("position"), dict)
                for item in sources.values()
            ),
            "completed_trades": sum(
                len(item.get("trades") or []) for item in sources.values() if isinstance(item, dict)
            ),
            "waiting": sum(item["status"] == "waiting" for item in results),
        },
        "safety": _safety(),
    }
    write_json_atomic(output_path, report)
    return report


def _safety() -> dict[str, Any]:
    return {
        "zero_money": True,
        "promotion_evidence_allowed": False,
        "live_allowed": False,
        "atomic_orders_enabled": False,
        "blocked_reason": "atomic_multi_leg_execution_not_approved",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--maximum-signal-age-seconds", type=int, default=43_200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_cycle(
        input_path=args.input,
        state_path=args.state,
        output_path=args.output,
        timeframe=args.timeframe,
        maximum_signal_age_seconds=args.maximum_signal_age_seconds,
    )
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
