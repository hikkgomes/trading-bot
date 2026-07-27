"""Refresh every configured active-income symbol with bounded, resumable jobs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.autopilot.history_bootstrap import run_history_bootstrap
from src.autopilot.io import write_json_atomic
from src.autopilot.research_factory import DEFAULT_CONFIG, load_factory_config
from src.config import PROJECT_ROOT

DEFAULT_MARKET_UNIVERSE_REPORT = PROJECT_ROOT / "runtime" / "market_universe.json"


def _eligible_symbols(report_path: Path) -> tuple[bool, set[str], dict[str, Any]]:
    if not report_path.exists() or report_path.is_symlink():
        return False, set(), {"reason": "market_universe_report_missing"}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        generated_at = datetime.fromisoformat(
            str(payload["generated_at"]).replace("Z", "+00:00")
        ).astimezone(UTC)
        age_seconds = (datetime.now(UTC) - generated_at).total_seconds()
        snapshot = payload.get("snapshot")
        valid = bool(
            isinstance(payload, dict)
            and payload.get("ok") is True
            and payload.get("schema") == "autopilot.market_universe/v2"
            and 0 <= age_seconds <= 48 * 3600
            and isinstance(snapshot, dict)
            and isinstance(snapshot.get("id"), str)
            and snapshot["id"].startswith("sha256:")
        )
    except Exception as exc:
        return (
            False,
            set(),
            {
                "reason": "market_universe_report_invalid",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
    symbols = {
        str(symbol).upper()
        for symbol in payload.get("eligible_research_symbols") or []
        if isinstance(symbol, str) and symbol.upper().endswith("USDT")
    }
    return (
        valid,
        symbols,
        {
            "reason": "ready" if valid else "market_universe_report_stale_or_invalid",
            "generated_at": payload.get("generated_at"),
            "age_seconds": round(age_seconds, 3),
            "snapshot_id": snapshot.get("id") if isinstance(snapshot, dict) else None,
        },
    )


def run_universe_history(
    *,
    config_path: Path = DEFAULT_CONFIG,
    market_universe_report: Path = DEFAULT_MARKET_UNIVERSE_REPORT,
    output_path: Path | None = None,
    timeframes: list[str] | None = None,
    exclude_timeframes: list[str] | None = None,
) -> dict[str, Any]:
    config = load_factory_config(config_path)
    configured_symbols = {
        space.symbol for space in config.search_spaces if space.product == "active_income"
    }
    universe_ok, eligible_symbols, universe = _eligible_symbols(market_universe_report)
    if config.dynamic_active_income_universe and not universe_ok:
        result = {
            "ok": False,
            "schema": "autopilot.universe_history/v1",
            "symbols": [],
            "completed": 0,
            "failed_symbols": [],
            "reports": [],
            "market_universe": universe,
        }
        if output_path is not None:
            write_json_atomic(output_path, result)
        return result
    symbols = sorted(
        (eligible_symbols if config.dynamic_active_income_universe else configured_symbols)
        - {"BTCUSDT"}
    )
    reports: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            report = run_history_bootstrap(
                config_path=config_path,
                markets=["futures"],
                timeframes=timeframes,
                exclude_timeframes=exclude_timeframes,
                symbol=symbol,
            )
        except Exception as exc:
            report = {
                "ok": False,
                "symbol": symbol,
                "error": f"{type(exc).__name__}: {exc}",
            }
        reports.append(report)
        if output_path is not None:
            write_json_atomic(
                output_path,
                {
                    "ok": False,
                    "schema": "autopilot.universe_history/v1",
                    "symbols": symbols,
                    "completed": len(reports),
                    "reports": reports,
                },
            )
    failed_symbols = [
        symbol for symbol, report in zip(symbols, reports, strict=True) if not report.get("ok")
    ]
    result = {
        "ok": len(reports) == len(symbols) and all(item.get("ok") for item in reports),
        "schema": "autopilot.universe_history/v1",
        "symbols": symbols,
        "completed": len(reports),
        "failed_symbols": failed_symbols,
        "market_universe": universe if config.dynamic_active_income_universe else None,
        "reports": reports,
    }
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the active-income research universe.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--market-universe-report",
        type=Path,
        default=DEFAULT_MARKET_UNIVERSE_REPORT,
    )
    parser.add_argument("--output", type=Path, default=Path("runtime/universe_history.json"))
    parser.add_argument("--timeframes", nargs="+")
    parser.add_argument("--exclude-timeframes", nargs="+")
    args = parser.parse_args()
    result = run_universe_history(
        config_path=args.config,
        market_universe_report=args.market_universe_report,
        output_path=args.output,
        timeframes=args.timeframes,
        exclude_timeframes=args.exclude_timeframes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
