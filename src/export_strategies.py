"""Export validated strategies from a search output directory to active_strategies.json.

This file is the contract between the research pipeline (heavy, runs on the
research machine) and the execution bot (light, runs 24/7). Only strategies
that passed every search filter and show positive expectancy are exported.
"""

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import PROJECT_ROOT

SCHEMA_VERSION = 1
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "active_strategies.json"
RISK_KEYS = ("risk_per_trade", "daily_stop_loss", "max_consecutive_losses", "cooldown_bars")
DEFAULT_RISK = {
    "risk_per_trade": 0.003,
    "daily_stop_loss": -0.02,
    "max_consecutive_losses": 3,
    "cooldown_bars": 24,
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


def _row_metric(row: pd.Series, name: str) -> Optional[float]:
    if name not in row or pd.isna(row[name]):
        return None
    return float(row[name])


def _baseline_win_rate(row: pd.Series) -> Optional[float]:
    # Prefer the untouched holdout; fall back to in-sample. A zero/missing
    # baseline disables the bot's drift kill-switch, so never export 0.0.
    for column in ("holdout_win_rate", "train_win_rate", "test_win_rate"):
        value = _row_metric(row, column)
        if value is not None and value > 0:
            return value
    return None


def build_payload(
    search_dir: Path,
    top_k: int = 3,
    min_dsr: Optional[float] = None,
) -> dict:
    config_path = search_dir / "config.json"
    ranked_path = search_dir / "ranked_strategies.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    if not ranked_path.exists():
        raise FileNotFoundError(f"Missing {ranked_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ranked = pd.read_csv(ranked_path)
    if ranked.empty:
        raise ValueError(f"{ranked_path} contains no strategies — nothing passed the search filters.")
    if "passes_filters" in ranked.columns:
        ranked = ranked[ranked["passes_filters"].astype(bool)]
    if not ranked.empty:
        if "wf_expectancy" in ranked.columns and ranked["wf_expectancy"].notna().any():
            expectancy_column = "wf_expectancy"
        elif "test_avg_net_return" in ranked.columns:
            expectancy_column = "test_avg_net_return"
        else:
            expectancy_column = None
        if expectancy_column is not None:
            ranked = ranked[ranked[expectancy_column] > 0]
    if min_dsr is not None and not ranked.empty and "dsr" in ranked.columns:
        ranked = ranked[ranked["dsr"] >= min_dsr]
    if ranked.empty:
        raise ValueError(
            f"No exportable strategies in {ranked_path}: all rows fail the "
            "passes_filters / positive-expectancy / min-dsr gates."
        )
    if "conditions_json" not in ranked.columns:
        raise ValueError(f"{ranked_path} is missing conditions_json — cannot reconstruct rules.")

    base_timeframe = config.get("base_timeframe", "15m")
    risk = {key: config.get(key, DEFAULT_RISK[key]) for key in RISK_KEYS}
    strategies = []
    for rank, (_, row) in enumerate(ranked.head(top_k).iterrows(), start=1):
        metrics = {
            name: _row_metric(row, name)
            for name in (
                "dsr", "wf_pass_rate", "wf_expectancy", "wf_avg_trades",
                "test_total_return", "test_avg_net_return", "test_win_rate",
                "holdout_total_return", "holdout_win_rate", "pool_pbo",
            )
        }
        strategies.append(
            {
                "id": f"{base_timeframe}_{row['direction']}_r{rank}",
                "rank": rank,
                "base_timeframe": base_timeframe,
                "direction": str(row["direction"]),
                "horizon_bars": int(row["horizon_bars"]),
                "take_profit": float(row["take_profit"]),
                "stop_loss": float(row["stop_loss"]),
                "use_atr_tp_sl": bool(config.get("use_atr_tp_sl", False)),
                "pnl_unit": config.get("pnl_unit", "usdt"),
                "conditions": json.loads(row["conditions_json"]),
                "rule": str(row["rule"]),
                "risk": risk,
                "fees": {
                    "fee_bps": config.get("fee_bps", 5.0),
                    "slippage_bps": config.get("slippage_bps", 2.0),
                },
                "metrics": {name: value for name, value in metrics.items() if value is not None},
                "baseline_win_rate": _baseline_win_rate(row),
            }
        )
    return {
        "version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "export_git_sha": _git_sha(),
        "source_dir": str(search_dir),
        "search_git_sha": config.get("git_sha", "unknown"),
        "search_timestamp": config.get("search_timestamp"),
        "strategies": strategies,
    }


def run(
    search_dir: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    top_k: int = 3,
    min_dsr: Optional[float] = None,
) -> Path:
    payload = build_payload(search_dir, top_k=top_k, min_dsr=min_dsr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export passing strategies to active_strategies.json for the execution bot."
    )
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-dsr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = run(args.search_dir, args.output, top_k=args.top_k, min_dsr=args.min_dsr)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"Wrote {path} ({len(payload['strategies'])} strategies)")


if __name__ == "__main__":
    main()
