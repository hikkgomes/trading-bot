"""Refresh every configured active-income symbol with bounded, resumable jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.autopilot.history_bootstrap import run_history_bootstrap
from src.autopilot.io import write_json_atomic
from src.autopilot.research_factory import DEFAULT_CONFIG, load_factory_config


def run_universe_history(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path | None = None,
    timeframes: list[str] | None = None,
    exclude_timeframes: list[str] | None = None,
) -> dict[str, Any]:
    config = load_factory_config(config_path)
    symbols = sorted(
        {space.symbol for space in config.search_spaces if space.product == "active_income"}
        - {"BTCUSDT"}
    )
    reports: list[dict[str, Any]] = []
    for symbol in symbols:
        report = run_history_bootstrap(
            config_path=config_path,
            markets=["futures"],
            timeframes=timeframes,
            exclude_timeframes=exclude_timeframes,
            symbol=symbol,
        )
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
        if not report.get("ok"):
            break
    result = {
        "ok": len(reports) == len(symbols) and all(item.get("ok") for item in reports),
        "schema": "autopilot.universe_history/v1",
        "symbols": symbols,
        "completed": len(reports),
        "reports": reports,
    }
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the active-income research universe.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("runtime/universe_history.json"))
    parser.add_argument("--timeframes", nargs="+")
    parser.add_argument("--exclude-timeframes", nargs="+")
    args = parser.parse_args()
    result = run_universe_history(
        config_path=args.config,
        output_path=args.output,
        timeframes=args.timeframes,
        exclude_timeframes=args.exclude_timeframes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
