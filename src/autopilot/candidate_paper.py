"""Isolated forward-paper cycles for candidates staged beside a live product."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from src.autopilot.approvals import artifact_digest, load_artifact
from src.autopilot.candidate_activation import (
    DEFAULT_CANDIDATE_DIR,
    candidate_path_for_product,
    product_identity,
)
from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, ProductConfig, load_config
from src.autopilot.io import write_json_atomic
from src.autopilot.locking import acquire_runtime_lock
from src.autopilot.promotion import (
    PromotionThresholds,
    build_promotion_review,
    write_review,
)
from src.autopilot.reporting import utc_now
from src.autopilot.strategy_policy import assert_loaded_strategy_artifact_allowed
from src.config import PROJECT_ROOT
from src.run_bot import PaperTradingBot, configure_logging

DEFAULT_STATUS = PROJECT_ROOT / "runtime" / "candidate_paper_status.json"
DEFAULT_LOCK = PROJECT_ROOT / "runtime" / "candidate_paper.lock"


def candidate_paper_paths(
    product_name: str,
    candidate_digest: str,
    *,
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR,
) -> dict[str, Path]:
    digest_key = candidate_digest.removeprefix("sha256:")
    if len(digest_key) != 64 or any(char not in "0123456789abcdef" for char in digest_key):
        raise ValueError("candidate digest must be a sha256 digest")
    return {
        "state": candidate_dir / f"{product_name}_paper_state_{digest_key[:16]}.json",
        "trade_log": candidate_dir / f"{product_name}_paper_trades.csv",
        "review_json": candidate_dir / f"{product_name}_promotion_review.json",
        "review_markdown": candidate_dir / f"{product_name}_promotion_review.md",
    }


def _paper_result(
    product: ProductConfig,
    config: AutopilotConfig,
    *,
    candidate_dir: Path,
    allow_entries: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {"product": product.name, "ok": True}
    if product.execution_mode != "live":
        return {**result, "skipped": True, "reason": "product_not_live"}
    candidate_path = candidate_path_for_product(product.name, candidate_dir=candidate_dir)
    if not candidate_path.exists():
        return {
            **result,
            "skipped": True,
            "reason": "waiting_for_staged_candidate",
            "candidate": str(candidate_path),
        }
    candidate = load_artifact(candidate_path)
    if candidate.get("product") != product_identity(product):
        raise ValueError(f"{product.name}: staged candidate product identity mismatch")
    assert_loaded_strategy_artifact_allowed(
        product,
        candidate,
        artifact_path=candidate_path,
        require_live_eligible=True,
    )
    digest = artifact_digest(candidate)
    paths = candidate_paper_paths(
        product.name,
        digest,
        candidate_dir=candidate_dir,
    )
    bot = PaperTradingBot(
        strategies_path=candidate_path,
        state_file=paths["state"],
        trade_log=paths["trade_log"],
        starting_equity=product.starting_equity,
        regime_guard=product.regime_guard,
        regime_mayer_top=product.regime_mayer_top,
        symbol=product.symbol,
        market=product.market,
        objective=product.objective,
        base_asset=product.base_asset,
        allow_entries=allow_entries,
        artifact_payload=candidate,
    )
    replay = bot.run_candidate_replay_cycle(
        max_unseen_bars=config.candidate_paper_max_unseen_bars,
        max_observation_delay_seconds=(config.candidate_paper_cadence_seconds * 2),
    )
    review = build_promotion_review(
        artifact_path=candidate_path,
        trade_log=paths["trade_log"],
        # Candidate papering is research-only and must not read the live
        # approval ledger. Activation/final approval performs that comparison.
        ledger_path=candidate_dir / ".paper_approval_view.json",
        thresholds=PromotionThresholds(),
        product=product,
        require_candidate_paper_binding=True,
    )
    for item in review.get("strategies", []):
        if item.get("recommendation") == "needs_approval":
            item["recommendation"] = "ready_for_activation"
            item["reasons"] = [
                "exact-fingerprint forward-paper thresholds pass; activate the reviewed digest "
                "while paused, then rebuild the packet against the active path before approval"
            ]
            item["approval_command"] = None
    ready = bool(review.get("strategies")) and all(
        item.get("recommendation") in {"ready_for_activation", "already_approved"}
        for item in review.get("strategies", [])
    )
    review["candidate_activation_ready"] = ready
    write_review(review, paths["review_json"], paths["review_markdown"])
    state = bot.state
    return {
        **result,
        "candidate": str(candidate_path),
        "candidate_digest": digest,
        "state_file": str(paths["state"]),
        "trade_log": str(paths["trade_log"]),
        "promotion_review": str(paths["review_markdown"]),
        "candidate_activation_ready": ready,
        "execution_path": "paper_only_forward_observation_with_quarantined_replay",
        "replay": replay,
        "candidate_paper_execution_binding": review.get("candidate_paper_execution_binding"),
        "paper_evidence": [
            {
                "id": item.get("id"),
                "paper": item.get("paper"),
            }
            for item in review.get("strategies", [])
        ],
        "equity": state.get("equity"),
        "open_positions": len(state.get("open_positions", {})),
        "drawdown_halted": state.get("drawdown_halted"),
        "portfolio_entries_allowed": allow_entries,
    }


def _candidate_portfolio_status(
    candidate_dir: Path,
    *,
    max_open_positions: int,
) -> dict[str, Any]:
    states: dict[str, Any] = {}
    total = 0
    ok = True
    for path in sorted(candidate_dir.glob("*_paper_state_*.json")):
        item: dict[str, Any] = {"ok": False, "open_positions": 0}
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("candidate paper state must be a regular non-symlink file")
            payload = json.loads(path.read_text(encoding="utf-8"))
            positions = payload.get("open_positions") if isinstance(payload, dict) else None
            if not isinstance(positions, dict):
                raise ValueError("candidate paper state open_positions must be an object")
            item.update(ok=True, open_positions=len(positions))
            total += len(positions)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            ok = False
        states[str(path)] = item
    return {
        "ok": ok,
        "open_positions": total,
        "max_open_positions": max_open_positions,
        "entry_capacity_available": ok and total < max_open_positions,
        "states": states,
    }


def _discovered_symbol_products(
    config: AutopilotConfig,
    *,
    candidate_dir: Path,
) -> list[ProductConfig]:
    base = next(
        (product for product in config.products if product.name == "active_income"),
        None,
    )
    if base is None:
        return []
    products: list[ProductConfig] = []
    for path in sorted(candidate_dir.glob(f"{base.name}__*usdt.json")):
        if path.is_symlink() or not path.is_file():
            continue
        candidate = load_artifact(path)
        identity = candidate.get("product")
        if not isinstance(identity, dict):
            raise ValueError(f"{path}: staged symbol candidate has no product identity")
        name = str(identity.get("name") or "")
        symbol = str(identity.get("symbol") or "").upper()
        expected_name = f"{base.name}__{symbol.lower()}"
        if (
            name != expected_name
            or path != candidate_path_for_product(name, candidate_dir=candidate_dir)
            or identity.get("objective") != "active_income"
            or identity.get("market") != "futures"
            or not symbol.endswith("USDT")
            or symbol == base.symbol.upper()
        ):
            raise ValueError(f"{path}: invalid staged symbol candidate identity")
        product = dataclasses.replace(
            base,
            name=name,
            execution_mode="live",
            symbol=symbol,
            strategies_path=path,
            state_file=candidate_dir / f"{name}_state.json",
            trade_log=candidate_dir / f"{name}_paper_trades.csv",
            preflight_report=candidate_dir / f"{name}_preflight_report.json",
            testnet_rehearsal_report=(candidate_dir / f"{name}_testnet_rehearsal_report.json"),
        )
        if candidate.get("product") != product_identity(product):
            raise ValueError(f"{path}: staged symbol candidate product identity mismatch")
        products.append(product)
    return products


def run_candidate_paper(
    config: AutopilotConfig,
    *,
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "ok": True,
        "products": [],
        "portfolio": _candidate_portfolio_status(
            candidate_dir,
            max_open_positions=config.active_income_max_open_positions,
        ),
    }
    candidate_dir.mkdir(parents=True, exist_ok=True)
    if candidate_dir.is_symlink():
        raise ValueError(f"candidate paper directory must not be a symlink: {candidate_dir}")
    for product in config.products:
        try:
            portfolio = _candidate_portfolio_status(
                candidate_dir,
                max_open_positions=config.active_income_max_open_positions,
            )
            item = _paper_result(
                product,
                config,
                candidate_dir=candidate_dir,
                allow_entries=bool(portfolio["entry_capacity_available"]),
            )
        except Exception as exc:
            item = {
                "product": product.name,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["products"].append(item)
        report["ok"] = report["ok"] and bool(item.get("ok"))
    try:
        discovered = _discovered_symbol_products(config, candidate_dir=candidate_dir)
    except Exception as exc:
        report["products"].append(
            {
                "product": "active_income_symbol_discovery",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        report["ok"] = False
        discovered = []
    configured_names = {product.name for product in config.products}
    for product in discovered:
        if product.name in configured_names:
            continue
        try:
            portfolio = _candidate_portfolio_status(
                candidate_dir,
                max_open_positions=config.active_income_max_open_positions,
            )
            item = _paper_result(
                product,
                config,
                candidate_dir=candidate_dir,
                allow_entries=bool(portfolio["entry_capacity_available"]),
            )
            item["configured_for_execution"] = False
            item["activation_blocked_until_product_configured"] = True
        except Exception as exc:
            item = {
                "product": product.name,
                "ok": False,
                "configured_for_execution": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["products"].append(item)
        report["ok"] = report["ok"] and bool(item.get("ok"))
    report["portfolio"] = _candidate_portfolio_status(
        candidate_dir,
        max_open_positions=config.active_income_max_open_positions,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated paper cycles for staged live candidates."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    try:
        with acquire_runtime_lock(args.lock):
            try:
                report = run_candidate_paper(
                    load_config(args.config),
                    candidate_dir=args.candidate_dir,
                )
            except Exception as exc:
                report = {
                    "generated_at": utc_now(),
                    "ok": False,
                    "products": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            write_json_atomic(args.output, report)
    except RuntimeError as exc:
        if str(exc) == f"autopilot already running; lock is held: {args.lock}":
            report = {
                "generated_at": utc_now(),
                "ok": True,
                "products": [],
                "skipped": True,
                "reason": "candidate_paper_cycle_already_running",
                "lock": str(args.lock),
            }
        else:
            raise
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
