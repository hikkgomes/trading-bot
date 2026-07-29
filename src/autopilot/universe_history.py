"""Refresh every configured active-income symbol with bounded, resumable jobs."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.autopilot.history_bootstrap import run_history_bootstrap
from src.autopilot.io import write_json_atomic
from src.autopilot.research_factory import (
    DEFAULT_CONFIG,
    load_factory_config,
    search_spaces_for_symbol,
)
from src.autopilot.research_history_contract import listing_history_compatibility
from src.config import PROJECT_ROOT

DEFAULT_MARKET_UNIVERSE_REPORT = PROJECT_ROOT / "runtime" / "market_universe.json"
DEFAULT_MAX_RUNTIME_SECONDS = 240.0


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
    listing_days = {
        str(item.get("symbol") or "").upper(): float(
            (item.get("metrics") or {}).get("listing_days")
        )
        for item in payload.get("symbols") or []
        if isinstance(item, dict)
        and isinstance(item.get("metrics"), dict)
        and isinstance((item.get("metrics") or {}).get("listing_days"), int | float)
    }
    return (
        valid,
        symbols,
        {
            "reason": "ready" if valid else "market_universe_report_stale_or_invalid",
            "generated_at": payload.get("generated_at"),
            "age_seconds": round(age_seconds, 3),
            "snapshot_id": snapshot.get("id") if isinstance(snapshot, dict) else None,
            "listing_days": listing_days,
        },
    )


def _partition(
    timeframes: list[str] | None,
    exclude_timeframes: list[str] | None,
) -> dict[str, list[str] | None]:
    return {
        "timeframes": sorted(timeframes) if timeframes is not None else None,
        "exclude_timeframes": (
            sorted(exclude_timeframes) if exclude_timeframes is not None else None
        ),
    }


def _load_progress(
    output_path: Path | None,
    *,
    symbols: list[str],
    partition: dict[str, list[str] | None],
    snapshot_id: Any,
) -> dict[str, Any]:
    if output_path is None or not output_path.exists() or output_path.is_symlink():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    output_universe = payload.get("market_universe")
    if not (
        isinstance(payload, dict)
        and payload.get("schema") == "autopilot.universe_history/v2"
        and payload.get("complete") is False
        and payload.get("symbols") == symbols
        and payload.get("partition") == partition
        and isinstance(output_universe, dict)
        and output_universe.get("snapshot_id") == snapshot_id
    ):
        return {}
    return payload


def _compatible_spaces(
    config: Any,
    symbol: str,
    *,
    universe: dict[str, Any],
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    spaces = search_spaces_for_symbol(config, symbol)
    if not config.dynamic_active_income_universe:
        return spaces, []
    listing_days = universe.get("listing_days")
    listing_days = listing_days if isinstance(listing_days, dict) else {}
    value = listing_days.get(symbol)
    if not isinstance(value, int | float):
        return (), [
            {
                "symbol": symbol,
                "search_space": space.name,
                "reason": "listing_history_unknown",
            }
            for space in spaces
        ]
    compatible = []
    excluded = []
    for space in spaces:
        detail = listing_history_compatibility(
            space,
            listing_days=float(value),
            as_of=str(universe.get("generated_at")),
        )
        if detail["ok"]:
            compatible.append(space)
        else:
            excluded.append(
                {
                    "symbol": symbol,
                    "search_space": space.name,
                    "opportunity_type": space.opportunity_type,
                    "reason": "listing_history_incompatible",
                    **detail,
                }
            )
    return tuple(compatible), excluded


def run_universe_history(
    *,
    config_path: Path = DEFAULT_CONFIG,
    market_universe_report: Path = DEFAULT_MARKET_UNIVERSE_REPORT,
    output_path: Path | None = None,
    timeframes: list[str] | None = None,
    exclude_timeframes: list[str] | None = None,
    max_runtime_seconds: float = DEFAULT_MAX_RUNTIME_SECONDS,
) -> dict[str, Any]:
    if max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")
    config = load_factory_config(config_path)
    configured_symbols = {
        space.symbol for space in config.search_spaces if space.product == "active_income"
    }
    universe_ok, eligible_symbols, universe = _eligible_symbols(market_universe_report)
    if config.dynamic_active_income_universe and not universe_ok:
        result = {
            "ok": False,
            "schema": "autopilot.universe_history/v2",
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
    partition = _partition(timeframes, exclude_timeframes)
    spaces_by_symbol: dict[str, tuple[Any, ...]] = {}
    excluded_spaces: list[dict[str, Any]] = []
    for symbol in symbols:
        compatible, excluded = _compatible_spaces(config, symbol, universe=universe)
        spaces_by_symbol[symbol] = compatible
        excluded_spaces.extend(excluded)
    work_symbols = [symbol for symbol in symbols if spaces_by_symbol[symbol]]
    progress = _load_progress(
        output_path,
        symbols=symbols,
        partition=partition,
        snapshot_id=universe.get("snapshot_id"),
    )
    reports = [
        item
        for item in progress.get("reports") or []
        if isinstance(item, dict) and item.get("ok") is True
    ]
    completed_symbols = {
        str(item.get("symbol"))
        for item in reports
        if isinstance(item.get("symbol"), str)
    }
    try:
        next_index = int(progress.get("next_index") or 0)
    except (TypeError, ValueError):
        next_index = 0
    if not 0 <= next_index <= len(work_symbols):
        next_index = 0
    deadline = time.monotonic() + float(max_runtime_seconds)
    failures: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    deferred = False
    for index in range(next_index, len(work_symbols)):
        symbol = work_symbols[index]
        if time.monotonic() >= deadline:
            next_index = index
            deferred = True
            break
        try:
            report = run_history_bootstrap(
                config_path=config_path,
                markets=["futures"],
                timeframes=timeframes,
                exclude_timeframes=exclude_timeframes,
                symbol=symbol,
                search_spaces=spaces_by_symbol[symbol],
                deadline_monotonic=deadline,
            )
        except Exception as exc:
            report = {
                "ok": False,
                "symbol": symbol,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if report.get("deferred"):
            current = report
            next_index = index
            deferred = True
            break
        reports = [item for item in reports if item.get("symbol") != symbol]
        reports.append(report)
        if report.get("ok"):
            completed_symbols.add(symbol)
            next_index = index + 1
        else:
            failures.append(report)
            next_index = index
        if output_path is not None:
            write_json_atomic(
                output_path,
                {
                    "ok": not failures,
                    "schema": "autopilot.universe_history/v2",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "symbols": symbols,
                    "work_symbols": work_symbols,
                    "partition": partition,
                    "complete": False,
                    "deferred": not failures,
                    "reason": (
                        "bootstrap_failed" if failures else "bootstrap_in_progress"
                    ),
                    "completed": len(completed_symbols),
                    "completed_symbols": sorted(completed_symbols),
                    "next_index": next_index,
                    "next_symbol": (
                        work_symbols[next_index] if next_index < len(work_symbols) else None
                    ),
                    "failed_symbols": sorted(
                        str(item.get("symbol")) for item in failures if item.get("symbol")
                    ),
                    "excluded_search_spaces": excluded_spaces,
                    "market_universe": universe if config.dynamic_active_income_universe else None,
                    "reports": reports,
                },
            )
        if failures:
            break
    complete = next_index >= len(work_symbols) and not deferred
    failed_symbols = sorted(
        str(item.get("symbol")) for item in reports if item.get("ok") is not True
    )
    is_deferred = not complete and not failed_symbols and not failures
    result = {
        "ok": not failed_symbols and not failures,
        "schema": "autopilot.universe_history/v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "symbols": symbols,
        "work_symbols": work_symbols,
        "partition": partition,
        "complete": complete,
        "deferred": is_deferred,
        "reason": (
            None
            if complete
            else ("bootstrap_in_progress" if is_deferred else "bootstrap_failed")
        ),
        "completed": len(completed_symbols),
        "completed_symbols": sorted(completed_symbols),
        "next_index": next_index,
        "next_symbol": work_symbols[next_index] if next_index < len(work_symbols) else None,
        "failed_symbols": failed_symbols,
        "excluded_search_spaces": excluded_spaces,
        "market_universe": universe if config.dynamic_active_income_universe else None,
        "reports": reports,
        **({"current": current} if current is not None else {}),
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
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=DEFAULT_MAX_RUNTIME_SECONDS,
    )
    args = parser.parse_args()
    result = run_universe_history(
        config_path=args.config,
        market_universe_report=args.market_universe_report,
        output_path=args.output,
        timeframes=args.timeframes,
        exclude_timeframes=args.exclude_timeframes,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
