"""Build bounded research-only basis, cross-sectional, and pairs forecasts."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.alpha.relative_value import basis_forecast, cross_sectional_forecasts, pairs_forecast
from src.autopilot.io import write_json_atomic
from src.config import PROJECT_ROOT, candle_data_dir

CONFIG_SCHEMA = "autopilot.relative_value_config/v1"
REPORT_SCHEMA = "autopilot.relative_value_research/v1"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "relative_value.json"
MAX_CONFIG_BYTES = 64 * 1024


class RelativeValueConfigError(ValueError):
    pass


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RelativeValueConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _project_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RelativeValueConfigError(f"{label} must be a project-relative path")
    path = (PROJECT_ROOT / value).resolve(strict=False)
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise RelativeValueConfigError(f"{label} must stay inside the project") from exc
    return path


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise RelativeValueConfigError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RelativeValueConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise RelativeValueConfigError(f"{label} must be in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True)
class RelativeValueConfig:
    market_universe_report: Path
    output: Path
    timeframe: str
    maximum_symbols: int
    lookback_rows: int
    cross_sectional_lookback_rows: int
    cross_sectional_top_k: int
    basis_entry_threshold: float
    basis_funding_intervals: int
    pairs_entry_z: float
    maximum_pairs: int


def load_config(path: Path = DEFAULT_CONFIG) -> RelativeValueConfig:
    path = Path(path)
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.stat().st_mode):
        raise RelativeValueConfigError("relative-value config must be a regular file")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise RelativeValueConfigError("relative-value config is too large")

    def reject_constant(value: str) -> None:
        raise RelativeValueConfigError(f"non-standard JSON constant: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, Mapping):
        raise RelativeValueConfigError("relative-value config must be an object")
    allowed = {
        "schema",
        "market_universe_report",
        "output",
        "timeframe",
        "maximum_symbols",
        "lookback_rows",
        "cross_sectional_lookback_rows",
        "cross_sectional_top_k",
        "basis_entry_threshold",
        "basis_funding_intervals",
        "pairs_entry_z",
        "maximum_pairs",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RelativeValueConfigError(f"unknown relative-value fields: {', '.join(unknown)}")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise RelativeValueConfigError(f"schema must be {CONFIG_SCHEMA}")
    timeframe = payload.get("timeframe")
    if not isinstance(timeframe, str) or not timeframe:
        raise RelativeValueConfigError("timeframe must be non-empty")
    lookback = _integer(payload.get("lookback_rows"), "lookback_rows", 100, 100_000)
    cross_lookback = _integer(
        payload.get("cross_sectional_lookback_rows"),
        "cross_sectional_lookback_rows",
        2,
        lookback - 1,
    )
    maximum_symbols = _integer(payload.get("maximum_symbols"), "maximum_symbols", 4, 25)
    top_k = _integer(payload.get("cross_sectional_top_k"), "cross_sectional_top_k", 1, 10)
    if top_k * 2 > maximum_symbols:
        raise RelativeValueConfigError("cross_sectional_top_k requires at least 2 * top_k symbols")
    return RelativeValueConfig(
        market_universe_report=_project_path(
            payload.get("market_universe_report"), "market_universe_report"
        ),
        output=_project_path(payload.get("output"), "output"),
        timeframe=timeframe,
        maximum_symbols=maximum_symbols,
        lookback_rows=lookback,
        cross_sectional_lookback_rows=cross_lookback,
        cross_sectional_top_k=top_k,
        basis_entry_threshold=_number(
            payload.get("basis_entry_threshold"), "basis_entry_threshold", 0.0001, 0.1
        ),
        basis_funding_intervals=_integer(
            payload.get("basis_funding_intervals"), "basis_funding_intervals", 0, 30
        ),
        pairs_entry_z=_number(payload.get("pairs_entry_z"), "pairs_entry_z", 0.5, 10),
        maximum_pairs=_integer(payload.get("maximum_pairs"), "maximum_pairs", 1, 300),
    )


def _universe(config: RelativeValueConfig) -> tuple[list[str], dict[str, float]]:
    path = config.market_universe_report
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = payload.get("research_symbols") if isinstance(payload, dict) else None
    if not isinstance(symbols, list):
        raise ValueError("market universe has no research_symbols")
    selected = [
        symbol.upper()
        for symbol in symbols
        if isinstance(symbol, str) and symbol.upper().endswith("USDT")
    ][: config.maximum_symbols]
    funding: dict[str, float] = {}
    for row in payload.get("symbols", []):
        if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        value = row["metrics"].get("funding_rate")
        if symbol in selected and isinstance(value, int | float) and math.isfinite(float(value)):
            funding[symbol] = float(value)
    if len(selected) < 4:
        raise ValueError("relative-value research requires at least four selected symbols")
    return selected, funding


def _closes(symbol: str, market: str, config: RelativeValueConfig) -> pd.Series:
    path = (
        candle_data_dir(symbol, market, legacy_fallback=True)
        / f"{symbol}_{config.timeframe}.parquet"
    )
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["timestamp", "close"]).tail(config.lookback_rows)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    close = pd.Series(pd.to_numeric(frame["close"], errors="raise").to_numpy(), index=timestamps)
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError(f"{symbol} {market} history is not unique and chronological")
    if len(close) < config.lookback_rows or (close <= 0).any():
        raise ValueError(f"{symbol} {market} history is insufficient or invalid")
    return close


def build_report(config: RelativeValueConfig) -> dict[str, Any]:
    generated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    try:
        symbols, funding = _universe(config)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema": REPORT_SCHEMA,
            "ok": False,
            "status": "waiting_for_universe",
            "generated_at": generated_at,
            "reason": str(exc),
            "safety": _safety(),
        }
    futures: dict[str, pd.Series] = {}
    spot: dict[str, pd.Series] = {}
    waiting: list[dict[str, str]] = []
    for symbol in symbols:
        for market, target in (("futures", futures), ("spot", spot)):
            try:
                target[symbol] = _closes(symbol, market, config)
            except (FileNotFoundError, OSError, ValueError) as exc:
                waiting.append({"symbol": symbol, "market": market, "reason": str(exc)})
    basis = []
    for symbol in sorted(set(futures) & set(spot)):
        forecast = basis_forecast(
            symbol=symbol,
            spot_price=float(spot[symbol].iloc[-1]),
            perpetual_price=float(futures[symbol].iloc[-1]),
            funding_rate=funding.get(symbol, 0.0),
            expected_funding_intervals=config.basis_funding_intervals,
            entry_threshold=config.basis_entry_threshold,
            generated_at=generated_at,
        )
        if forecast is not None:
            basis.append(forecast.to_dict())
    relative_returns = {
        symbol: float(series.iloc[-1] / series.iloc[-1 - config.cross_sectional_lookback_rows] - 1)
        for symbol, series in futures.items()
    }
    cross_sectional = (
        [
            {
                **forecast.to_dict(),
                "research_only": True,
                "paper_trade_allowed": True,
                "live_allowed": False,
                "promotion_eligible": False,
            }
            for forecast in cross_sectional_forecasts(
                relative_returns,
                top_k=config.cross_sectional_top_k,
                generated_at=generated_at,
            )
        ]
        if len(relative_returns) >= config.cross_sectional_top_k * 2
        else []
    )
    pairs = []
    for first, second in itertools.islice(
        itertools.combinations(sorted(futures), 2), config.maximum_pairs
    ):
        aligned = pd.concat([futures[first], futures[second]], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            continue
        forecast = pairs_forecast(
            first_symbol=first,
            second_symbol=second,
            first_prices=aligned.iloc[:, 0].tolist(),
            second_prices=aligned.iloc[:, 1].tolist(),
            entry_z=config.pairs_entry_z,
            generated_at=generated_at,
        )
        if forecast is not None:
            pairs.append(forecast.to_dict())
    return {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "status": "ready",
        "generated_at": generated_at,
        "universe_symbols": symbols,
        "loaded_futures_symbols": sorted(futures),
        "loaded_spot_symbols": sorted(spot),
        "forecasts": {
            "spot_perp_basis": basis,
            "cross_sectional": cross_sectional,
            "statistical_pairs": pairs,
        },
        "summary": {
            "basis": len(basis),
            "cross_sectional": len(cross_sectional),
            "pairs": len(pairs),
            "waiting_inputs": len(waiting),
        },
        "waiting": waiting,
        "safety": _safety(),
    }


def _safety() -> dict[str, Any]:
    return {
        "research_only": True,
        "exploration_paper_allowed": True,
        "promotion_allowed": False,
        "live_allowed": False,
        "blocked_reason": "relative_value_forward_validation_and_atomic_execution_incomplete",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.validate:
        print(json.dumps({"ok": True, "schema": CONFIG_SCHEMA}, sort_keys=True))
        return
    report = build_report(config)
    write_json_atomic(args.output or config.output, report)
    print(json.dumps({"ok": report["ok"], "summary": report.get("summary", {})}, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
