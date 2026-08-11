"""Run bounded short-horizon alpha replay over the newest captured events."""

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

from src.alpha.microstructure import MicrostructureAlphaPolicy
from src.autopilot.event_capture import _dynamic_symbols, load_event_capture_config
from src.autopilot.event_replay import replay
from src.autopilot.io import write_json_atomic
from src.config import PROJECT_ROOT

CONFIG_SCHEMA = "autopilot.microstructure_research_config/v1"
REPORT_SCHEMA = "autopilot.microstructure_research/v1"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "microstructure_research.json"
MAX_CONFIG_BYTES = 64 * 1024


class MicrostructureResearchConfigError(ValueError):
    pass


def _project_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise MicrostructureResearchConfigError(f"{label} must be project-relative")
    path = (PROJECT_ROOT / value).resolve(strict=False)
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise MicrostructureResearchConfigError(f"{label} must stay inside the project") from exc
    return path


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MicrostructureResearchConfigError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MicrostructureResearchConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise MicrostructureResearchConfigError(f"{label} must be in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True)
class MicrostructureResearchConfig:
    event_capture_config: Path
    output: Path
    market: str
    maximum_symbols: int
    maximum_files: int
    maximum_events_per_symbol: int
    sample_every: int
    strategy_quantity: float
    policy: MicrostructureAlphaPolicy


def load_config(path: Path = DEFAULT_CONFIG) -> MicrostructureResearchConfig:
    path = Path(path)
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.stat().st_mode):
        raise MicrostructureResearchConfigError("microstructure config must be a regular file")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise MicrostructureResearchConfigError("microstructure config is too large")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != CONFIG_SCHEMA:
        raise MicrostructureResearchConfigError(f"schema must be {CONFIG_SCHEMA}")
    allowed = {
        "schema",
        "event_capture_config",
        "output",
        "market",
        "maximum_symbols",
        "maximum_files",
        "maximum_events_per_symbol",
        "sample_every",
        "strategy_quantity",
        "policy",
    }
    if unknown := sorted(set(payload) - allowed):
        raise MicrostructureResearchConfigError(f"unknown fields: {', '.join(unknown)}")
    market = payload.get("market")
    if market not in {"spot", "futures"}:
        raise MicrostructureResearchConfigError("market must be spot or futures")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise MicrostructureResearchConfigError("policy must be an object")
    policy_allowed = {
        "minimum_abs_score",
        "maximum_spread_bps",
        "minimum_total_depth",
        "minimum_liquidity_vacuum_ratio",
        "horizon_seconds",
    }
    if unknown := sorted(set(policy) - policy_allowed):
        raise MicrostructureResearchConfigError(f"unknown policy fields: {', '.join(unknown)}")
    return MicrostructureResearchConfig(
        event_capture_config=_project_path(
            payload.get("event_capture_config"), "event_capture_config"
        ),
        output=_project_path(payload.get("output"), "output"),
        market=str(market),
        maximum_symbols=_integer(payload.get("maximum_symbols"), "maximum_symbols", 1, 12),
        maximum_files=_integer(payload.get("maximum_files"), "maximum_files", 1, 24),
        maximum_events_per_symbol=_integer(
            payload.get("maximum_events_per_symbol"),
            "maximum_events_per_symbol",
            100,
            1_000_000,
        ),
        sample_every=_integer(payload.get("sample_every"), "sample_every", 1, 100_000),
        strategy_quantity=_number(
            payload.get("strategy_quantity"), "strategy_quantity", 0.000001, 1_000_000
        ),
        policy=MicrostructureAlphaPolicy(
            minimum_abs_score=_number(
                policy.get("minimum_abs_score"), "policy.minimum_abs_score", 0.01, 1
            ),
            maximum_spread_bps=_number(
                policy.get("maximum_spread_bps"), "policy.maximum_spread_bps", 0.01, 1_000
            ),
            minimum_total_depth=_number(
                policy.get("minimum_total_depth"), "policy.minimum_total_depth", 0.000001, 1e12
            ),
            minimum_liquidity_vacuum_ratio=_number(
                policy.get("minimum_liquidity_vacuum_ratio"),
                "policy.minimum_liquidity_vacuum_ratio",
                0.01,
                1,
            ),
            horizon_seconds=_integer(
                policy.get("horizon_seconds"), "policy.horizon_seconds", 1, 3600
            ),
        ),
    )


def _symbols(config: MicrostructureResearchConfig) -> tuple[str, ...]:
    capture = load_event_capture_config(config.event_capture_config)
    configured = [
        symbol
        for source in capture.sources
        if source.market == config.market
        for symbol in source.symbols
    ]
    dynamic = list(_dynamic_symbols(capture)) if config.market == "futures" else []
    return tuple(dict.fromkeys((*configured, *dynamic)))[: config.maximum_symbols]


def _files(config: MicrostructureResearchConfig) -> list[Path]:
    capture = load_event_capture_config(config.event_capture_config)
    if capture.root.is_symlink() or not capture.root.exists():
        return []
    candidates = [
        path
        for path in capture.root.glob(f"{config.market}_*.jsonl")
        if not path.is_symlink() and path.is_file()
    ]
    return sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))[
        -config.maximum_files :
    ]


def build_report(config: MicrostructureResearchConfig) -> dict[str, Any]:
    generated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    files = _files(config)
    if not files:
        return {
            "schema": REPORT_SCHEMA,
            "ok": True,
            "status": "waiting_for_events",
            "generated_at": generated_at,
            "symbols": [],
            "files": [],
            "safety": _safety(),
        }
    results = []
    for symbol in _symbols(config):
        try:
            item = replay(
                files,
                symbol=symbol,
                sample_every=config.sample_every,
                microstructure_policy=config.policy,
                strategy_quantity=config.strategy_quantity,
                max_events=config.maximum_events_per_symbol,
            )
            strategy = item.get("microstructure_strategy") or {}
            results.append(
                {
                    "symbol": symbol,
                    "ok": True,
                    "events": item["events"],
                    "first_received_ns": item["first_received_ns"],
                    "last_received_ns": item["last_received_ns"],
                    "final_features": item["final_features"],
                    "strategy": strategy,
                    "sampled_feature_count": len(item["sampled_features"]),
                }
            )
        except Exception as exc:
            results.append({"symbol": symbol, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "schema": REPORT_SCHEMA,
        "ok": all(item["ok"] for item in results),
        "status": "ready",
        "generated_at": generated_at,
        "files": [str(path) for path in files],
        "symbols": results,
        "summary": {
            "symbols": len(results),
            "errors": sum(item["ok"] is not True for item in results),
            "events": sum(int(item.get("events") or 0) for item in results),
            "signals": sum(
                int((item.get("strategy") or {}).get("signals") or 0) for item in results
            ),
            "completed_trades": sum(
                len((item.get("strategy") or {}).get("completed_trades") or []) for item in results
            ),
        },
        "safety": _safety(),
    }


def _safety() -> dict[str, Any]:
    return {
        "research_only": True,
        "promotion_eligible": False,
        "live_allowed": False,
        "order_api_available": False,
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
    print(json.dumps(report.get("summary") or {"status": report["status"]}, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
