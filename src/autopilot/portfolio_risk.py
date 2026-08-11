"""Build a bounded rolling correlation and benchmark-beta risk model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.autopilot.io import write_json_atomic
from src.autopilot.portfolio import PORTFOLIO_RISK_MODEL_SCHEMA
from src.config import PROJECT_ROOT, candle_data_dir

CONFIG_SCHEMA = "autopilot.portfolio_risk_config/v1"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "portfolio_risk.json"
MAX_CONFIG_BYTES = 64 * 1024


class PortfolioRiskConfigError(ValueError):
    pass


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise PortfolioRiskConfigError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _project_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PortfolioRiskConfigError(f"{label} must be a non-empty project path")
    path = (PROJECT_ROOT / value).resolve(strict=False)
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise PortfolioRiskConfigError(f"{label} must stay inside the project") from exc
    return path


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PortfolioRiskConfigError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class PortfolioRiskConfig:
    market: str
    benchmark_symbol: str
    timeframe: str
    lookback_rows: int
    minimum_overlap_rows: int
    maximum_symbols: int
    market_universe_report: Path
    output: Path


def load_config(path: Path = DEFAULT_CONFIG) -> PortfolioRiskConfig:
    path = Path(path)
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.stat().st_mode):
        raise PortfolioRiskConfigError("portfolio risk config must be a regular file")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise PortfolioRiskConfigError("portfolio risk config is too large")

    def reject_constant(value: str) -> None:
        raise PortfolioRiskConfigError(f"non-standard JSON constant: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, Mapping):
        raise PortfolioRiskConfigError("portfolio risk config must be an object")
    allowed = {
        "schema",
        "market",
        "benchmark_symbol",
        "timeframe",
        "lookback_rows",
        "minimum_overlap_rows",
        "maximum_symbols",
        "market_universe_report",
        "output",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise PortfolioRiskConfigError(f"unknown portfolio risk fields: {', '.join(unknown)}")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise PortfolioRiskConfigError(f"schema must be {CONFIG_SCHEMA}")
    market = payload.get("market")
    if market not in {"spot", "futures"}:
        raise PortfolioRiskConfigError("market must be spot or futures")
    benchmark = payload.get("benchmark_symbol")
    timeframe = payload.get("timeframe")
    if not isinstance(benchmark, str) or not benchmark.endswith("USDT"):
        raise PortfolioRiskConfigError("benchmark_symbol must be a USDT symbol")
    if not isinstance(timeframe, str) or not timeframe:
        raise PortfolioRiskConfigError("timeframe must be non-empty")
    lookback = _bounded_int(payload.get("lookback_rows"), "lookback_rows", 500, 100_000)
    overlap = _bounded_int(
        payload.get("minimum_overlap_rows"), "minimum_overlap_rows", 100, lookback
    )
    return PortfolioRiskConfig(
        market=market,
        benchmark_symbol=benchmark.upper(),
        timeframe=timeframe,
        lookback_rows=lookback,
        minimum_overlap_rows=overlap,
        maximum_symbols=_bounded_int(payload.get("maximum_symbols"), "maximum_symbols", 1, 100),
        market_universe_report=_project_path(
            payload.get("market_universe_report"), "market_universe_report"
        ),
        output=_project_path(payload.get("output"), "output"),
    )


def _symbols(config: PortfolioRiskConfig) -> list[str]:
    selected = [config.benchmark_symbol]
    path = config.market_universe_report
    if path.exists() and not path.is_symlink():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("research_symbols") if isinstance(payload, dict) else None
        if isinstance(raw, list):
            selected.extend(
                str(symbol).upper()
                for symbol in raw
                if isinstance(symbol, str) and symbol.upper().endswith("USDT")
            )
    return list(dict.fromkeys(selected))[: config.maximum_symbols]


def _close_series(config: PortfolioRiskConfig, symbol: str) -> tuple[pd.Series, Path]:
    path = (
        candle_data_dir(symbol, config.market, legacy_fallback=True)
        / f"{symbol}_{config.timeframe}.parquet"
    )
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["timestamp", "close"]).tail(config.lookback_rows)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    close = pd.Series(pd.to_numeric(frame["close"], errors="raise").to_numpy(), index=timestamps)
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError(f"{symbol} risk history is not unique and chronological")
    if len(close) < config.minimum_overlap_rows or (close <= 0).any():
        raise ValueError(f"{symbol} risk history is insufficient or invalid")
    return close.pct_change().dropna().rename(symbol), path


def build_risk_model(config: PortfolioRiskConfig) -> dict[str, Any]:
    returns: dict[str, pd.Series] = {}
    inputs: list[dict[str, Any]] = []
    waiting: list[dict[str, str]] = []
    for symbol in _symbols(config):
        try:
            series, path = _close_series(config, symbol)
        except (FileNotFoundError, ValueError, OSError) as exc:
            waiting.append({"symbol": symbol, "reason": str(exc)})
            continue
        returns[symbol] = series
        inputs.append(
            {
                "symbol": symbol,
                "path": str(path),
                "rows": len(series),
                "start": series.index[0].isoformat(),
                "end": series.index[-1].isoformat(),
            }
        )
    generated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    if config.benchmark_symbol not in returns:
        return {
            "schema": PORTFOLIO_RISK_MODEL_SCHEMA,
            "ok": False,
            "generated_at": generated_at,
            "reason": "benchmark_history_unavailable",
            "benchmark_symbol": config.benchmark_symbol,
            "inputs": inputs,
            "waiting": waiting,
        }
    matrix = pd.concat(returns.values(), axis=1, join="outer").sort_index()
    correlations: dict[str, dict[str, float]] = {symbol: {} for symbol in returns}
    beta: dict[str, float] = {}
    benchmark = matrix[config.benchmark_symbol]
    benchmark_variance = float(benchmark.var())
    if not math.isfinite(benchmark_variance) or benchmark_variance <= 0:
        raise ValueError("benchmark return variance is not positive")
    for first in returns:
        for second in returns:
            if first == second:
                correlations[first][second] = 1.0
                continue
            aligned = matrix[[first, second]].dropna()
            if len(aligned) < config.minimum_overlap_rows:
                continue
            correlation = float(aligned[first].corr(aligned[second]))
            if math.isfinite(correlation):
                correlations[first][second] = max(-1.0, min(1.0, correlation))
        if first == config.benchmark_symbol:
            beta[first] = 1.0
            continue
        aligned_benchmark = matrix[[first, config.benchmark_symbol]].dropna()
        if len(aligned_benchmark) >= config.minimum_overlap_rows:
            covariance = float(
                aligned_benchmark[first].cov(aligned_benchmark[config.benchmark_symbol])
            )
            value = covariance / float(aligned_benchmark[config.benchmark_symbol].var())
            if math.isfinite(value):
                beta[first] = value
    return {
        "schema": PORTFOLIO_RISK_MODEL_SCHEMA,
        "ok": True,
        "generated_at": generated_at,
        "benchmark_symbol": config.benchmark_symbol,
        "timeframe": config.timeframe,
        "lookback_rows": config.lookback_rows,
        "minimum_overlap_rows": config.minimum_overlap_rows,
        "correlations": correlations,
        "beta_by_symbol": beta,
        "inputs": inputs,
        "waiting": waiting,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.validate:
        print(json.dumps({"ok": True, "schema": CONFIG_SCHEMA}, sort_keys=True))
        return
    report = build_risk_model(config)
    write_json_atomic(args.output or config.output, report)
    print(json.dumps({"ok": report["ok"], "symbols": len(report.get("beta_by_symbol") or {})}))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
