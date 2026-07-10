"""Export validated strategies from a search output directory to active_strategies.json.

This file is the contract between the research pipeline (heavy, runs on the
research machine) and the execution bot (light, runs 24/7). Only strategies
that passed every search filter and show positive expectancy are exported.
"""

import argparse
import datetime
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

from src.autopilot.io import write_json_atomic
from src.config import PROJECT_ROOT

SCHEMA_VERSION = 1
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "active_strategies.json"
RISK_KEYS = (
    "risk_per_trade",
    "daily_stop_loss",
    "max_consecutive_losses",
    "cooldown_bars",
    "max_position_fraction",
    "max_trades_per_day",
)
DEFAULT_RISK_USDT = {
    "risk_per_trade": 0.003,
    "daily_stop_loss": -0.02,
    "max_consecutive_losses": 3,
    "cooldown_bars": 24,
    "max_position_fraction": 0.25,
    "max_trades_per_day": 4,
}
DEFAULT_RISK_BTC = {
    "risk_per_trade": 0.003,
    "daily_stop_loss": -0.01,
    "max_consecutive_losses": 3,
    "cooldown_bars": 24,
    "max_position_fraction": 0.35,
    "max_trades_per_day": 1,
}


def _with_benchmark_metrics(metrics: dict[str, float], pnl_unit: str) -> dict[str, float]:
    if pnl_unit.lower() != "btc" or "holdout_total_return" not in metrics:
        return metrics
    metrics = dict(metrics)
    # In BTC-denominated searches, total return is extra BTC accumulated versus
    # simply holding BTC. Make that benchmark explicit for promotion/live gates.
    metrics.setdefault("holdout_buy_hold_return", 0.0)
    metrics.setdefault("holdout_excess_return_vs_buy_hold", metrics["holdout_total_return"])
    return metrics


def _default_market(pnl_unit: str) -> str:
    return "spot" if pnl_unit.lower() == "btc" else "futures"


def _default_risk(pnl_unit: str) -> dict[str, float | int]:
    return dict(DEFAULT_RISK_BTC if pnl_unit.lower() == "btc" else DEFAULT_RISK_USDT)


def _normalize_non_empty_string(value, *, field: str, default: str | None = None) -> str:
    is_missing = value is None
    if not is_missing:
        try:
            missing_check = pd.isna(value)
        except (TypeError, ValueError):
            missing_check = False
        is_missing = isinstance(missing_check, bool) and missing_check
    if is_missing:
        if default is None:
            raise ValueError(f"{field} must be a non-empty string.")
        value = default
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string.")
    return normalized


def _normalize_pnl_unit(value) -> str:
    pnl_unit = _normalize_non_empty_string(value, field="pnl_unit", default="usdt").lower()
    if pnl_unit not in {"btc", "usdt"}:
        raise ValueError(f"pnl_unit must be 'btc' or 'usdt', got {value!r}.")
    return pnl_unit


def _normalize_market(value, *, pnl_unit: str) -> str:
    market = _normalize_non_empty_string(
        value, field="market", default=_default_market(pnl_unit)
    ).lower()
    if market not in {"spot", "futures"}:
        raise ValueError(f"market must be 'spot' or 'futures', got {value!r}.")
    if market == "spot" and pnl_unit != "btc":
        raise ValueError("spot strategy exports must use pnl_unit 'btc'.")
    if market == "futures" and pnl_unit != "usdt":
        raise ValueError("futures strategy exports must use pnl_unit 'usdt'.")
    return market


def _finite_config_float(config: dict, key: str, default: float | int) -> float:
    value = config.get(key, default)
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{key} must be finite.")
    return normalized


def _whole_number(value: float, *, key: str) -> int:
    if value != int(value):
        raise ValueError(f"{key} must be an integer.")
    return int(value)


def _normalize_risk(config: dict, default_risk: dict[str, float | int]) -> dict[str, float | int]:
    risk = {key: _finite_config_float(config, key, default_risk[key]) for key in RISK_KEYS}
    for key in ("max_consecutive_losses", "cooldown_bars", "max_trades_per_day"):
        risk[key] = _whole_number(float(risk[key]), key=key)
    if risk["risk_per_trade"] <= 0:
        raise ValueError("risk_per_trade must be positive.")
    if risk["max_position_fraction"] <= 0 or risk["max_position_fraction"] > 1:
        raise ValueError("max_position_fraction must be > 0 and <= 1.")
    if risk["daily_stop_loss"] >= 0:
        raise ValueError("daily_stop_loss must be negative.")
    if risk["max_consecutive_losses"] <= 0:
        raise ValueError("max_consecutive_losses must be positive.")
    if risk["cooldown_bars"] < 0:
        raise ValueError("cooldown_bars must be non-negative.")
    if risk["max_trades_per_day"] <= 0:
        raise ValueError("max_trades_per_day must be positive.")
    return risk


def _normalize_fees(config: dict) -> dict[str, float]:
    fees = {
        "fee_bps": _finite_config_float(config, "fee_bps", 5.0),
        "slippage_bps": _finite_config_float(config, "slippage_bps", 2.0),
    }
    if fees["fee_bps"] < 0:
        raise ValueError("fee_bps must be non-negative.")
    if fees["slippage_bps"] < 0:
        raise ValueError("slippage_bps must be non-negative.")
    return fees


def _positive_row_float(row: pd.Series, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite.")
    if value <= 0:
        raise ValueError(f"{key} must be positive.")
    return value


def _positive_row_int(row: pd.Series, key: str) -> int:
    value = _positive_row_float(row, key)
    if value != int(value):
        raise ValueError(f"{key} must be an integer.")
    return int(value)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


def _row_metric(row: pd.Series, name: str) -> float | None:
    if name not in row or pd.isna(row[name]):
        return None
    return float(row[name])


def _baseline_win_rate(row: pd.Series) -> float | None:
    # Prefer the untouched holdout; fall back to in-sample. A zero/missing
    # baseline disables the bot's drift kill-switch, so never export 0.0.
    for column in ("holdout_win_rate", "train_win_rate", "test_win_rate"):
        value = _row_metric(row, column)
        if value is not None and value > 0:
            return value
    return None


def _ranked_path(search_dir: Path, prefer_clustered: bool = True) -> Path:
    clustered = search_dir / "ranked_strategies_clustered.csv"
    if prefer_clustered and clustered.exists():
        return clustered
    return search_dir / "ranked_strategies.csv"


def build_payload(
    search_dir: Path,
    top_k: int = 3,
    min_dsr: float | None = None,
    min_holdout_return: float | None = 0.0,
    prefer_clustered: bool = True,
) -> dict:
    try:
        top_k_value = float(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be a positive integer.") from exc
    if not math.isfinite(top_k_value) or top_k_value != int(top_k_value) or top_k_value <= 0:
        raise ValueError("top_k must be a positive integer.")
    top_k = int(top_k_value)
    if min_dsr is not None and not math.isfinite(float(min_dsr)):
        raise ValueError("min_dsr must be finite.")
    if min_holdout_return is not None and not math.isfinite(float(min_holdout_return)):
        raise ValueError("min_holdout_return must be finite.")

    config_path = search_dir / "config.json"
    ranked_path = _ranked_path(search_dir, prefer_clustered=prefer_clustered)
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
    if min_holdout_return is not None and not ranked.empty:
        # The holdout GATES admission. A strategy that lost on its own untouched
        # holdout must never ship (the old report-only behaviour was the
        # documented flaw that put losing strategies in front of the bots).
        if "holdout_total_return" not in ranked.columns:
            raise ValueError(
                f"{ranked_path} has no holdout_total_return column — rerun the search "
                "with --holdout-fraction, or explicitly disable the gate with "
                "--no-holdout-gate if you accept exporting unvalidated strategies."
            )
        ranked = ranked[
            ranked["holdout_total_return"].notna()
            & (ranked["holdout_total_return"] > min_holdout_return)
        ]
    if ranked.empty:
        raise ValueError(
            f"No exportable strategies in {ranked_path}: all rows fail the "
            "passes_filters / positive-expectancy / min-dsr / positive-holdout gates."
        )
    if "conditions_json" not in ranked.columns:
        raise ValueError(f"{ranked_path} is missing conditions_json — cannot reconstruct rules.")

    base_timeframe = _normalize_non_empty_string(
        config.get("base_timeframe"), field="base_timeframe", default="15m"
    )
    pnl_unit = _normalize_pnl_unit(config.get("pnl_unit"))
    market = _normalize_market(config.get("market"), pnl_unit=pnl_unit)
    symbol = _normalize_non_empty_string(config.get("symbol"), field="symbol", default="BTCUSDT")
    default_risk = _default_risk(pnl_unit)
    risk = _normalize_risk(config, default_risk)
    fees = _normalize_fees(config)
    strategies = []
    for rank, (_, row) in enumerate(ranked.head(top_k).iterrows(), start=1):
        direction = _normalize_non_empty_string(row.get("direction"), field="direction")
        if direction not in {"long", "short"}:
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}.")
        rule = _normalize_non_empty_string(row.get("rule"), field="rule")
        horizon_bars = _positive_row_int(row, "horizon_bars")
        take_profit = _positive_row_float(row, "take_profit")
        stop_loss = _positive_row_float(row, "stop_loss")
        conditions = json.loads(row["conditions_json"])
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("conditions_json must decode to a non-empty list.")
        metrics = {
            name: _row_metric(row, name)
            for name in (
                "dsr", "wf_pass_rate", "wf_expectancy", "wf_avg_trades",
                "test_total_return", "test_avg_net_return", "test_win_rate",
                "holdout_total_return", "holdout_win_rate", "pool_pbo",
            )
        }
        exported_metrics = _with_benchmark_metrics(
            {name: value for name, value in metrics.items() if value is not None},
            pnl_unit,
        )
        strategies.append(
            {
                "id": f"{base_timeframe}_{direction}_r{rank}",
                "rank": rank,
                "market": market,
                "symbol": symbol,
                "base_timeframe": base_timeframe,
                "direction": direction,
                "horizon_bars": horizon_bars,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "use_atr_tp_sl": bool(config.get("use_atr_tp_sl", False)),
                "pnl_unit": pnl_unit,
                "conditions": conditions,
                "rule": rule,
                "risk": risk,
                "fees": fees,
                "metrics": exported_metrics,
                "baseline_win_rate": _baseline_win_rate(row),
            }
        )
    return {
        "version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "export_git_sha": _git_sha(),
        "source_dir": str(search_dir),
        "search_git_sha": config.get("git_sha", "unknown"),
        "search_timestamp": config.get("search_timestamp"),
        "source_ranked_file": ranked_path.name,
        "pnl_unit": pnl_unit,
        "market": market,
        "symbol": symbol,
        "paper_trade_allowed": True,
        "live_allowed": True,
        "promotion_eligible": True,
        "strategies": strategies,
    }


def run(
    search_dir: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    top_k: int = 3,
    min_dsr: float | None = None,
    min_holdout_return: float | None = 0.0,
    prefer_clustered: bool = True,
) -> Path:
    payload = build_payload(
        search_dir,
        top_k=top_k,
        min_dsr=min_dsr,
        min_holdout_return=min_holdout_return,
        prefer_clustered=prefer_clustered,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, payload)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export passing strategies to active_strategies.json for the execution bot."
    )
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-dsr", type=float, default=None)
    parser.add_argument(
        "--raw-ranked",
        action="store_true",
        help="Read ranked_strategies.csv even when ranked_strategies_clustered.csv exists.",
    )
    parser.add_argument(
        "--min-holdout-return", type=float, default=0.0,
        help="Require holdout_total_return strictly above this (default 0.0: the "
        "holdout gates admission).",
    )
    parser.add_argument(
        "--no-holdout-gate", action="store_true",
        help="Disable the holdout gate (NOT recommended — exports strategies that "
             "may have lost on their own holdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_holdout = None if args.no_holdout_gate else args.min_holdout_return
    path = run(
        args.search_dir,
        args.output,
        top_k=args.top_k,
        min_dsr=args.min_dsr,
        min_holdout_return=min_holdout,
        prefer_clustered=not args.raw_ranked,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"Wrote {path} ({len(payload['strategies'])} strategies)")


if __name__ == "__main__":
    main()
