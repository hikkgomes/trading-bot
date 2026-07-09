"""Lightweight strategy-framework smoke checks for the 24/7 server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.autopilot.io import write_json_atomic
from src.autopilot.reporting import utc_now
from src.run_backtest import _synthetic_ohlcv
from src.sweep import run_sweep

DEFAULT_SYNTHETIC_STRATEGIES = ["sma_cross", "macd_trend", "rsi_reversion", "ml_classifier"]


def _summarize_table(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {"rows": 0, "positive_return_rows": 0}
    sortable = table.copy()
    sort_column = "dsr" if "dsr" in sortable.columns else "total_return"
    sortable = sortable.sort_values(sort_column, ascending=False, na_position="last")
    best = sortable.iloc[0].to_dict()
    return {
        "rows": int(len(table)),
        "positive_return_rows": int((table.get("total_return", pd.Series(dtype=float)) > 0).sum()),
        "error_rows": int(table.get("error", pd.Series(dtype=object)).notna().sum()) if "error" in table else 0,
        "best_strategy": str(best.get("strategy", "")),
        "best_total_return": _finite_float(best.get("total_return")),
        "best_dsr": _finite_float(best.get("dsr")),
        "best_wf_pass_rate": _finite_float(best.get("wf_pass_rate")),
    }


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


def _synthetic_smoke(rows: int) -> dict[str, Any]:
    df = _synthetic_ohlcv(rows)
    table = run_sweep(
        df,
        DEFAULT_SYNTHETIC_STRATEGIES,
        train_fraction=0.7,
        grids={"ml_classifier": {"model": ["sklearn"], "max_features": [8]}},
        walk_forward_windows=3,
    )
    return {
        "ok": True,
        "name": "synthetic_strategy_sweep",
        "synthetic_rows": rows,
        **_summarize_table(table),
    }


def _regime_smoke(path: Path, max_rows: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "ok": True,
            "skipped": True,
            "name": "regime_filter_sweep",
            "reason": "missing_regime_input",
            "input": str(path),
        }
    columns = ["timestamp", "open", "high", "low", "close", "volume", "tf_1d_regime_id"]
    try:
        df = pd.read_parquet(path, columns=columns)
    except Exception:
        df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").set_index("timestamp")
    if max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows)
    table = run_sweep(
        df,
        ["regime_filter"],
        train_fraction=0.7,
        grids={"regime_filter": {"strategy": ["sma_cross"], "regime_ids": ["0", "1", "2", "3"]}},
        walk_forward_windows=3,
    )
    return {
        "ok": True,
        "skipped": False,
        "name": "regime_filter_sweep",
        "input": str(path),
        "scored_rows": int(len(df)),
        **_summarize_table(table),
    }


def run_strategy_smoke(
    *,
    synthetic_rows: int = 1600,
    regime_input: Path = Path("runtime/regime/futures_15m_regime.parquet"),
    max_regime_rows: int = 50_000,
) -> dict[str, Any]:
    if synthetic_rows < 500:
        raise ValueError("synthetic_rows must be at least 500")
    scenarios = []
    for name, runner in (
        ("synthetic_strategy_sweep", lambda: _synthetic_smoke(synthetic_rows)),
        ("regime_filter_sweep", lambda: _regime_smoke(regime_input, max_regime_rows)),
    ):
        try:
            scenarios.append(runner())
        except Exception as exc:
            scenarios.append(
                {
                    "ok": False,
                    "name": name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "ok": all(bool(scenario.get("ok")) for scenario in scenarios),
        "generated_at": utc_now(),
        "scenarios": scenarios,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight strategy framework smoke checks.")
    parser.add_argument("--synthetic-rows", type=int, default=1600)
    parser.add_argument("--regime-input", type=Path, default=Path("runtime/regime/futures_15m_regime.parquet"))
    parser.add_argument("--max-regime-rows", type=int, default=50_000)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_strategy_smoke(
        synthetic_rows=args.synthetic_rows,
        regime_input=args.regime_input,
        max_regime_rows=args.max_regime_rows,
    )
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
