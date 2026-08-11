"""Adaptive, non-promotable forward paper observations for research candidates."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from research_exploration.experiment_log import DEFAULT_LOG, load_log
from research_exploration.export import strategy_entry
from research_exploration.hypothesis_schema import Hypothesis
from src.autopilot.candidate_activation import product_identity
from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, ProductConfig, load_config
from src.autopilot.io import write_json_atomic
from src.autopilot.strategy_policy import (
    StrategyPolicyError,
    assert_loaded_strategy_artifact_allowed,
)
from src.config import PROJECT_ROOT
from src.run_bot import PaperTradingBot

SCHEMA = "autopilot.exploration_paper/v1"
MANIFEST_SCHEMA = "autopilot.exploration_paper_manifest/v1"
DEFAULT_INCUBATION = PROJECT_ROOT / "runtime" / "incubation_candidates.json"
DEFAULT_ROOT = PROJECT_ROOT / "runtime" / "exploration_paper"
DEFAULT_MANIFEST = DEFAULT_ROOT / "manifest.json"
DEFAULT_STATUS = DEFAULT_ROOT / "status.json"
DEFAULT_MAX_PER_PRODUCT = 12
MAX_TRADE_LOG_BYTES = 64 * 1024 * 1024
MAX_SIGNAL_TIMES = 512
MIN_DIAGNOSTIC_DATA_READY = 12
SIGNAL_OUTCOMES = {
    "alpha_ensemble_conflict",
    "alpha_ensemble_not_selected",
    "alpha_ensemble_rejected",
    "entry_opened",
    "portfolio_rejected",
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _safe_key(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return normalized.strip("_") or "candidate"


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    evaluation = (record.get("config") or {}).get("eval") or {}
    return (
        str(record.get("hypothesis_id") or ""),
        str(evaluation.get("market") or "").lower(),
        str(evaluation.get("symbol") or "BTCUSDT").upper(),
        str(evaluation.get("pnl_unit") or "usdt").lower(),
    )


def _latest_records(log_path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in load_log(log_path):
        if not isinstance(record, dict) or not isinstance(record.get("hypothesis"), dict):
            continue
        key = _record_key(record)
        if not all(key):
            continue
        previous = latest.get(key)
        if previous is None or str(record.get("timestamp") or "") > str(
            previous.get("timestamp") or ""
        ):
            latest[key] = record
    return latest


def _base_product(config: AutopilotConfig, product_name: str) -> ProductConfig:
    product = next((item for item in config.products if item.name == product_name), None)
    if product is None:
        raise ValueError(f"incubation queue references unknown product {product_name!r}")
    return product


def _paper_product(
    config: AutopilotConfig,
    *,
    product_name: str,
    market: str,
    symbol: str,
) -> ProductConfig:
    base = _base_product(config, product_name)
    if market != base.market:
        raise ValueError(f"{product_name}: exploration market does not match configured product")
    runtime_name = product_name
    if symbol != base.symbol.upper():
        if product_name != "active_income" or market != "futures" or not symbol.endswith("USDT"):
            raise ValueError(f"{product_name}: exploration symbol is not allowed: {symbol}")
        runtime_name = f"active_income__{symbol.lower()}"
    return dataclasses.replace(
        base,
        name=runtime_name,
        execution_mode="paper",
        symbol=symbol,
    )


def _finite_metadata(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite_metadata(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite_metadata(item) for item in value]
    return value


def _artifact_for_record(record: dict[str, Any], product: ProductConfig) -> dict[str, Any]:
    hypothesis = Hypothesis.from_dict(record["hypothesis"])
    entry = strategy_entry(record, 1, market=product.market)
    # Old experiment logs may contain NaN summary metrics. They are display
    # metadata, never behavior; executable hypothesis/risk/fee fields continue
    # through their strict schema validators above.
    entry["metrics"] = _finite_metadata(entry.get("metrics") or {})
    entry["exploration_evidence"] = {
        "adaptive": True,
        "promotion_eligible": False,
        "source_verdict": record.get("verdict"),
        "source_fingerprint": record.get("fingerprint"),
    }
    return {
        "version": 2,
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "market": product.market,
        "symbol": product.symbol,
        "pnl_unit": entry["pnl_unit"],
        "paper_trade_allowed": True,
        "live_allowed": False,
        "promotion_eligible": False,
        "exploration_only": True,
        "adaptive_evidence": True,
        "product": product_identity(product),
        "source": {
            "experiment_timestamp": record.get("timestamp"),
            "experiment_fingerprint": record.get("fingerprint"),
            "verdict": record.get("verdict"),
            "hypothesis_id": hypothesis.id,
        },
        "strategies": [entry],
    }


def _identity_digest(artifact: dict[str, Any]) -> str:
    identity = {
        key: artifact.get(key)
        for key in (
            "version",
            "schema",
            "market",
            "symbol",
            "pnl_unit",
            "paper_trade_allowed",
            "live_allowed",
            "promotion_eligible",
            "exploration_only",
            "adaptive_evidence",
            "product",
            "strategies",
        )
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def build_exploration_manifest(
    config: AutopilotConfig,
    *,
    incubation_path: Path = DEFAULT_INCUBATION,
    log_path: Path = DEFAULT_LOG,
    root: Path = DEFAULT_ROOT,
    max_per_product: int = DEFAULT_MAX_PER_PRODUCT,
) -> dict[str, Any]:
    """Compile the research-attention queue into isolated paper-only artifacts."""
    if max_per_product < 1 or max_per_product > 100:
        raise ValueError("max_per_product must be in [1, 100]")
    if not incubation_path.exists():
        return {
            "schema": MANIFEST_SCHEMA,
            "generated_at": utc_now(),
            "ok": True,
            "skipped": True,
            "reason": "waiting_for_incubation_queue",
            "candidates": [],
        }
    queue = _read_object(incubation_path, label="incubation queue")
    if (
        queue.get("schema") != "autopilot.incubation_candidates/v1"
        or queue.get("research_only") is not True
        or queue.get("executable") is not False
        or queue.get("paper_trade_allowed") is not False
        or queue.get("promotion_eligible") is not False
    ):
        raise ValueError("incubation queue safety contract is invalid")
    products = queue.get("products")
    if not isinstance(products, dict):
        raise ValueError("incubation queue products must be an object")
    records = _latest_records(log_path)
    artifacts_dir = root / "artifacts"
    states_dir = root / "states"
    trades_dir = root / "trades"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)
    trades_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    missing_records: list[dict[str, str]] = []
    policy_rejected_candidates: list[dict[str, str]] = []
    seen_digests: set[str] = set()
    for product_name, items in sorted(products.items()):
        if not isinstance(items, list):
            raise ValueError(f"incubation queue product {product_name!r} must be a list")
        for candidate in items[:max_per_product]:
            if not isinstance(candidate, dict):
                continue
            hypothesis_id = str(candidate.get("id") or "")
            market = str(candidate.get("market") or "").lower()
            symbol = str(candidate.get("symbol") or "BTCUSDT").upper()
            pnl_unit = str(candidate.get("pnl_unit") or "").lower()
            key = (hypothesis_id, market, symbol, pnl_unit)
            record = records.get(key)
            if record is None:
                missing_records.append(
                    {"product": str(product_name), "symbol": symbol, "id": hypothesis_id}
                )
                continue
            product = _paper_product(
                config,
                product_name=str(product_name),
                market=market,
                symbol=symbol,
            )
            artifact = _artifact_for_record(record, product)
            try:
                assert_loaded_strategy_artifact_allowed(
                    product,
                    artifact,
                    require_live_eligible=False,
                )
            except StrategyPolicyError as exc:
                policy_rejected_candidates.append(
                    {
                        "product": str(product_name),
                        "symbol": symbol,
                        "id": hypothesis_id,
                        "reason": "strategy_policy_rejected",
                        "detail": str(exc),
                    }
                )
                continue
            digest = _identity_digest(artifact)
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
            stem = f"{_safe_key(product.name)}__{digest[:16]}"
            artifact_path = artifacts_dir / f"{stem}.json"
            state_path = states_dir / f"{stem}.json"
            trade_path = trades_dir / f"{stem}.csv"
            write_json_atomic(artifact_path, artifact)
            candidates.append(
                {
                    "product": product.name,
                    "base_product": product_name,
                    "objective": product.objective,
                    "base_asset": product.base_asset,
                    "market": product.market,
                    "symbol": product.symbol,
                    "starting_equity": product.starting_equity,
                    "regime_guard": product.regime_guard,
                    "regime_mayer_top": product.regime_mayer_top,
                    "hypothesis_id": hypothesis_id,
                    "artifact_digest": f"sha256:{digest}",
                    "artifact": str(artifact_path),
                    "state": str(state_path),
                    "trade_log": str(trade_path),
                    "adaptive_evidence": True,
                    "promotion_eligible": False,
                }
            )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "source": {"incubation": str(incubation_path), "experiment_log": str(log_path)},
        "research_only": True,
        "paper_trade_allowed": True,
        "live_allowed": False,
        "promotion_eligible": False,
        "adaptive_evidence": True,
        "summary": {
            "candidates": len(candidates),
            "missing_experiment_records": len(missing_records),
            "policy_rejected_candidates": len(policy_rejected_candidates),
        },
        "missing_experiment_records": missing_records,
        "policy_rejected_candidates": policy_rejected_candidates,
        "candidates": candidates,
    }
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _manifest_product(config: AutopilotConfig, item: dict[str, Any]) -> ProductConfig:
    product = _paper_product(
        config,
        product_name=str(item["base_product"]),
        market=str(item["market"]),
        symbol=str(item["symbol"]),
    )
    if product.name != item.get("product"):
        raise ValueError("exploration manifest product identity mismatch")
    return product


def _aggregate_trace(aggregate: dict[str, Any], trace: dict[str, Any]) -> None:
    summary = trace.get("summary") if isinstance(trace, dict) else None
    if not isinstance(summary, dict):
        return
    for field in (
        "data_ready",
        "market_bars_processed",
        "signals",
        "entries_opened",
        "positions_managed",
    ):
        aggregate[field] = int(aggregate.get(field, 0)) + int(summary.get(field, 0) or 0)
    outcomes = aggregate.setdefault("outcomes", {})
    for outcome, count in (summary.get("outcomes") or {}).items():
        outcomes[str(outcome)] = int(outcomes.get(str(outcome), 0)) + int(count or 0)
    failed = Counter(aggregate.get("failed_predicates") or {})
    failed_stages = Counter(aggregate.get("failed_stages") or {})
    for strategy in (trace.get("strategies") or {}).values():
        if not isinstance(strategy, dict):
            continue
        predicate = strategy.get("failed_predicate")
        if predicate:
            failed[str(predicate)] += 1
        stage = strategy.get("failed_stage")
        if stage:
            failed_stages[str(stage)] += 1
    aggregate["failed_predicates"] = dict(failed.most_common(25))
    aggregate["failed_stages"] = dict(failed_stages.most_common(12))


def _trade_feedback(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_trades": 0, "net_return_sum": 0.0, "sized_return_sum": 0.0}
    if path.is_symlink() or not path.is_file():
        raise ValueError("exploration trade log must be a regular non-symlink file")
    if path.stat().st_size > MAX_TRADE_LOG_BYTES:
        raise ValueError("exploration trade log exceeds its feedback size budget")
    completed = 0
    net_return = 0.0
    sized_return = 0.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                net = float(row.get("net_return") or 0.0)
                sized = float(row.get("sized_return") or 0.0)
            except (TypeError, ValueError) as exc:
                raise ValueError("exploration trade log contains an invalid return") from exc
            if not math.isfinite(net) or not math.isfinite(sized):
                raise ValueError("exploration trade log contains a non-finite return")
            completed += 1
            net_return += net
            sized_return += sized
    return {
        "completed_trades": completed,
        "net_return_sum": round(net_return, 10),
        "sized_return_sum": round(sized_return, 10),
    }


def _signal_times(trace: dict[str, Any]) -> list[str]:
    times: set[str] = set()
    for decision in (trace.get("strategies") or {}).values():
        if not isinstance(decision, dict) or decision.get("outcome") not in SIGNAL_OUTCOMES:
            continue
        timestamp = decision.get("latest_bar")
        if isinstance(timestamp, str) and timestamp:
            times.add(timestamp)
    return sorted(times)


def _signal_frequency(feedback: dict[str, Any]) -> dict[str, Any]:
    raw_times = feedback.get("signal_times") or []
    parsed = []
    for value in raw_times:
        try:
            parsed.append(dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")))
        except ValueError:
            continue
    parsed = sorted(set(parsed))
    gaps = [
        (current - previous).total_seconds()
        for previous, current in zip(parsed, parsed[1:], strict=False)
    ]
    p95 = None
    if gaps:
        ordered = sorted(gaps)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    first = feedback.get("first_observed_at")
    last = feedback.get("last_observed_at")
    observed_days = None
    try:
        elapsed = dt.datetime.fromisoformat(str(last).replace("Z", "+00:00")) - dt.datetime.fromisoformat(
            str(first).replace("Z", "+00:00")
        )
        observed_days = elapsed.total_seconds() / 86_400
    except (TypeError, ValueError):
        pass
    signals = int(feedback.get("signals") or 0)
    return {
        "signals": signals,
        "signals_per_day": (
            round(signals / observed_days, 6) if observed_days is not None and observed_days > 0 else None
        ),
        "median_signal_gap_seconds": (
            round(float(statistics.median(gaps)), 3) if gaps else None
        ),
        "p95_signal_gap_seconds": round(float(p95), 3) if p95 is not None else None,
        "regime_coverage": round(
            max(
                0.0,
                min(
                    1.0,
                    1.0
                    - float((feedback.get("failed_stages") or {}).get("regime", 0))
                    / max(1, int(feedback.get("data_ready") or 0)),
                ),
            ),
            6,
        ),
    }


def _diagnosis(feedback: dict[str, Any]) -> tuple[str, str | None]:
    data_ready = int(feedback.get("data_ready") or 0)
    if data_ready < MIN_DIAGNOSTIC_DATA_READY:
        return "collecting_forward_observations", None
    if (
        int(feedback.get("completed_trades") or 0) >= 3
        and float(feedback.get("sized_return_sum") or 0.0) < 0
    ):
        return "trades_exist_but_negative_expectancy", "exit_risk"
    if int(feedback.get("signals") or 0) > 0:
        return "useful_signal_frequency_observed", None
    stages = Counter(feedback.get("failed_stages") or {})
    stage = stages.most_common(1)[0][0] if stages else None
    if stage == "regime":
        return "regime_never_fires", "regime"
    if stage == "setup":
        return "setup_never_fires", "setup"
    if stage in {
        "trigger",
        "conditions",
        "score_threshold",
        "volatility_filter",
        "frozen_ml_threshold",
    }:
        return "trigger_never_fires", "trigger"
    return "signal_generation_unclassified", None


def _candidate_feedback(
    previous: dict[str, Any],
    candidate: dict[str, Any],
    trace: dict[str, Any],
    *,
    observed_at: str,
) -> dict[str, Any]:
    feedback = {
        "artifact_digest": candidate.get("artifact_digest"),
        "hypothesis_id": candidate.get("hypothesis_id"),
        "product": candidate.get("base_product"),
        "symbol": candidate.get("symbol"),
        "cycles": max(0, int(previous.get("cycles") or 0)) + 1,
        "data_ready": max(0, int(previous.get("data_ready") or 0)),
        "market_bars_processed": max(0, int(previous.get("market_bars_processed") or 0)),
        "signals": max(0, int(previous.get("signals") or 0)),
        "entries_opened": max(0, int(previous.get("entries_opened") or 0)),
        "positions_managed": max(0, int(previous.get("positions_managed") or 0)),
        "outcomes": dict(previous.get("outcomes") or {}),
        "failed_predicates": dict(previous.get("failed_predicates") or {}),
        "failed_stages": dict(previous.get("failed_stages") or {}),
        "first_observed_at": previous.get("first_observed_at") or observed_at,
        "last_observed_at": observed_at,
    }
    _aggregate_trace(feedback, trace)
    signal_times = [
        *[str(item) for item in (previous.get("signal_times") or []) if item],
        *_signal_times(trace),
    ]
    feedback["signal_times"] = sorted(set(signal_times))[-MAX_SIGNAL_TIMES:]
    feedback.update(_trade_feedback(Path(str(candidate["trade_log"]))))
    feedback["signal_frequency"] = _signal_frequency(feedback)
    diagnosis, focus_stage = _diagnosis(feedback)
    feedback["diagnosis"] = diagnosis
    feedback["mutation_focus_stage"] = focus_stage
    return feedback


def run_exploration_paper(
    config: AutopilotConfig,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    previous_status_path: Path | None = DEFAULT_STATUS,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "adaptive_evidence": True,
        "promotion_eligible": False,
        "products": [],
        "candidate_feedback": {},
        "aggregate": {
            "cycles": 1,
            "data_ready": 0,
            "market_bars_processed": 0,
            "signals": 0,
            "entries_opened": 0,
            "positions_managed": 0,
            "outcomes": {},
            "failed_predicates": {},
            "failed_stages": {},
        },
    }
    if not manifest_path.exists():
        return {**report, "skipped": True, "reason": "waiting_for_exploration_manifest"}
    manifest = _read_object(manifest_path, label="exploration manifest")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("live_allowed") is not False
        or manifest.get("promotion_eligible") is not False
        or manifest.get("adaptive_evidence") is not True
    ):
        raise ValueError("exploration manifest safety contract is invalid")
    previous_feedback: dict[str, Any] = {}
    if previous_status_path is not None and previous_status_path.exists():
        previous = _read_object(previous_status_path, label="exploration status")
        if previous.get("schema") == SCHEMA and isinstance(previous.get("aggregate"), dict):
            report["aggregate"] = dict(previous["aggregate"])
            report["aggregate"]["cycles"] = int(report["aggregate"].get("cycles", 0)) + 1
            if isinstance(previous.get("candidate_feedback"), dict):
                previous_feedback = previous["candidate_feedback"]
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("exploration manifest candidates must be a list")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        item: dict[str, Any] = {
            "product": candidate.get("product"),
            "symbol": candidate.get("symbol"),
            "hypothesis_id": candidate.get("hypothesis_id"),
            "artifact_digest": candidate.get("artifact_digest"),
            "ok": True,
        }
        try:
            product = _manifest_product(config, candidate)
            artifact_path = Path(str(candidate["artifact"]))
            artifact = _read_object(artifact_path, label="exploration artifact")
            if (
                artifact.get("exploration_only") is not True
                or artifact.get("adaptive_evidence") is not True
                or artifact.get("live_allowed") is not False
                or artifact.get("promotion_eligible") is not False
            ):
                raise ValueError("exploration artifact safety contract is invalid")
            assert_loaded_strategy_artifact_allowed(
                product,
                artifact,
                artifact_path=artifact_path,
                require_live_eligible=False,
            )
            bot = PaperTradingBot(
                strategies_path=artifact_path,
                state_file=Path(str(candidate["state"])),
                trade_log=Path(str(candidate["trade_log"])),
                starting_equity=float(candidate["starting_equity"]),
                regime_guard=bool(candidate["regime_guard"]),
                regime_mayer_top=float(candidate["regime_mayer_top"]),
                symbol=product.symbol,
                market=product.market,
                objective=product.objective,
                base_asset=product.base_asset,
                artifact_payload=artifact,
            )
            bot.run_cycle()
            item.update(
                decision_trace=bot.decision_trace,
                equity=bot.state.get("equity"),
                open_positions=len(bot.state.get("open_positions", {})),
                drawdown_halted=bot.state.get("drawdown_halted"),
                trade_log=str(candidate["trade_log"]),
            )
            _aggregate_trace(report["aggregate"], bot.decision_trace)
            digest = str(candidate.get("artifact_digest") or "")
            report["candidate_feedback"][digest] = _candidate_feedback(
                previous_feedback.get(digest)
                if isinstance(previous_feedback.get(digest), dict)
                else {},
                candidate,
                bot.decision_trace,
                observed_at=report["generated_at"],
            )
        except Exception as exc:
            item.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            report["ok"] = False
        report["products"].append(item)
    diagnoses = Counter(
        str(item.get("diagnosis") or "unknown")
        for item in report["candidate_feedback"].values()
        if isinstance(item, dict)
    )
    report["summary"] = {
        "candidates": len(report["products"]),
        "healthy": sum(1 for item in report["products"] if item.get("ok")),
        "failed": sum(1 for item in report["products"] if not item.get("ok")),
        "diagnoses": dict(diagnoses.most_common()),
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or run adaptive, permanently non-promotable exploration paper."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--incubation", type=Path, default=DEFAULT_INCUBATION)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--build", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.build:
        report = build_exploration_manifest(
            config,
            incubation_path=args.incubation,
            log_path=args.log,
            root=args.root,
        )
    else:
        report = run_exploration_paper(
            config,
            manifest_path=args.manifest,
            previous_status_path=args.status,
        )
        write_json_atomic(args.status, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
