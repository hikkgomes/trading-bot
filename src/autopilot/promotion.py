"""Prepare human review packets for strategy promotion.

This module never approves strategies. It summarizes validation metadata,
paper-trading results, and approval status so the operator can decide whether
to run the explicit approval command.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.autopilot.approvals import (
    ApprovalError,
    ApprovalLedger,
    artifact_digest,
    is_valid_approval_actor,
    load_artifact,
    strategy_fingerprint,
)
from src.autopilot.config import DEFAULT_CONFIG_PATH, ProductConfig, load_config
from src.autopilot.execution_identity import execution_engine_digest
from src.autopilot.io import write_json_atomic, write_text_atomic
from src.autopilot.reporting import utc_now
from src.autopilot.strategy_policy import validate_strategy, validate_strategy_artifact
from src.config import PROJECT_ROOT

DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "runtime" / "promotion_review.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "runtime" / "promotion_review.md"


@dataclass
class PromotionThresholds:
    min_paper_trades: int = 20
    min_paper_sized_return: float = 0.0
    min_holdout_return: float = 0.0
    max_paper_drawdown: float = 0.05
    max_paper_consecutive_losses: int = 4
    min_paper_days: float = 7.0


def _finite_threshold(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _threshold_errors(thresholds: PromotionThresholds) -> list[str]:
    errors: list[str] = []
    min_trades = _finite_threshold(thresholds.min_paper_trades)
    if min_trades is None or int(min_trades) != min_trades or min_trades < 1:
        errors.append("min_paper_trades must be an integer >= 1")
    min_paper_return = _finite_threshold(thresholds.min_paper_sized_return)
    if min_paper_return is None or min_paper_return < 0:
        errors.append("min_paper_sized_return must be finite and non-negative")
    min_holdout = _finite_threshold(thresholds.min_holdout_return)
    if min_holdout is None or min_holdout < 0:
        errors.append("min_holdout_return must be finite and non-negative")
    max_drawdown = _finite_threshold(thresholds.max_paper_drawdown)
    if max_drawdown is None or max_drawdown < 0 or max_drawdown > 1:
        errors.append("max_paper_drawdown must be finite and between 0 and 1")
    max_losses = _finite_threshold(thresholds.max_paper_consecutive_losses)
    if max_losses is None or int(max_losses) != max_losses or max_losses < 0:
        errors.append("max_paper_consecutive_losses must be an integer >= 0")
    min_days = _finite_threshold(thresholds.min_paper_days)
    if min_days is None or min_days < 0:
        errors.append("min_paper_days must be finite and non-negative")
    return errors


def _load_trade_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _empty_paper_stats(
    *,
    unbound_trade_rows: int = 0,
    other_fingerprint_rows: int = 0,
) -> dict[str, Any]:
    return {
        "trades": 0,
        "win_rate": None,
        "total_sized_return": 0.0,
        "avg_net_return": None,
        "last_equity": None,
        "max_drawdown": None,
        "max_consecutive_losses": 0,
        "first_exit_time": None,
        "last_exit_time": None,
        "paper_days": 0.0,
        "invalid_return_rows": 0,
        "unbound_trade_rows": unbound_trade_rows,
        "other_fingerprint_rows": other_fingerprint_rows,
    }


def _paper_stats(
    trades: pd.DataFrame,
    strategy_id: str,
    fingerprint: str,
) -> dict[str, Any]:
    if trades.empty or "strategy_id" not in trades.columns:
        return _empty_paper_stats()
    id_subset = trades[trades["strategy_id"] == strategy_id].copy()
    if id_subset.empty:
        return _empty_paper_stats()
    if "strategy_fingerprint" not in id_subset.columns:
        return _empty_paper_stats(unbound_trade_rows=int(len(id_subset)))
    fingerprints = id_subset["strategy_fingerprint"]
    bound = fingerprints.map(lambda value: isinstance(value, str) and bool(value.strip()))
    exact = bound & (fingerprints == fingerprint)
    unbound_trade_rows = int((~bound).sum())
    other_fingerprint_rows = int((bound & ~exact).sum())
    subset = id_subset[exact].copy()
    if subset.empty:
        return _empty_paper_stats(
            unbound_trade_rows=unbound_trade_rows,
            other_fingerprint_rows=other_fingerprint_rows,
        )
    if "exit_time" in subset.columns:
        subset["_exit_time"] = pd.to_datetime(subset["exit_time"], utc=True, errors="coerce")
        subset = subset.sort_values("_exit_time", na_position="last")
    else:
        subset["_exit_time"] = pd.NaT
    net_raw = (
        subset["net_return"]
        if "net_return" in subset.columns
        else pd.Series([None] * len(subset), index=subset.index)
    )
    sized_raw = (
        subset["sized_return"]
        if "sized_return" in subset.columns
        else pd.Series([None] * len(subset), index=subset.index)
    )
    net = pd.to_numeric(net_raw, errors="coerce")
    sized = pd.to_numeric(sized_raw, errors="coerce")
    net_finite = net.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    sized_finite = sized.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    valid_return_rows = net_finite & sized_finite
    invalid_return_rows = int((~valid_return_rows).sum())
    valid_net = net[net_finite]
    sized_for_stats = sized.where(sized_finite, 0.0).fillna(0.0)
    equity = pd.to_numeric(subset.get("equity_after"), errors="coerce")
    if equity.notna().any():
        equity_curve = equity.dropna().astype(float)
    else:
        equity_curve = (1.0 + sized_for_stats).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0
    max_drawdown = abs(float(drawdown.min())) if len(drawdown) else None
    loss_flags = (net < 0).fillna(False).to_list()
    max_loss_streak = 0
    current_loss_streak = 0
    for is_loss in loss_flags:
        current_loss_streak = current_loss_streak + 1 if is_loss else 0
        max_loss_streak = max(max_loss_streak, current_loss_streak)
    exit_times = subset["_exit_time"].dropna()
    first_exit = exit_times.iloc[0] if not exit_times.empty else None
    last_exit = exit_times.iloc[-1] if not exit_times.empty else None
    paper_days = 0.0
    if first_exit is not None and last_exit is not None:
        paper_days = max(0.0, float((last_exit - first_exit).total_seconds() / 86_400.0))
    return {
        "trades": int(len(subset)),
        "win_rate": float((valid_net > 0).mean()) if not valid_net.empty else None,
        "total_sized_return": float(sized_for_stats.sum()),
        "avg_net_return": float(valid_net.mean()) if not valid_net.empty else None,
        "last_equity": float(equity.dropna().iloc[-1]) if equity.notna().any() else None,
        "max_drawdown": max_drawdown,
        "max_consecutive_losses": max_loss_streak,
        "first_exit_time": first_exit.isoformat() if first_exit is not None else None,
        "last_exit_time": last_exit.isoformat() if last_exit is not None else None,
        "paper_days": paper_days,
        "invalid_return_rows": invalid_return_rows,
        "unbound_trade_rows": unbound_trade_rows,
        "other_fingerprint_rows": other_fingerprint_rows,
    }


def _same_path(left: str | Path | None, right: Path) -> bool:
    if left is None:
        return False
    try:
        return Path(left).resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return False


def _approval_status(
    ledger_payload: dict[str, Any],
    fingerprint: str,
    *,
    artifact_path: Path,
    artifact_digest_value: str,
    execution_engine_digest_value: str,
    product: ProductConfig | None,
) -> str:
    approvals = ledger_payload.get("approvals", {})
    if not isinstance(approvals, dict):
        return "ledger_malformed"
    entry = approvals.get(fingerprint)
    if entry is None:
        return "missing"
    if not isinstance(entry, dict):
        return "malformed"
    status = str(entry.get("status", "unknown"))
    if status != "approved":
        return status
    if not is_valid_approval_actor(entry.get("approved_by")):
        return "invalid_actor"
    if entry.get("fingerprint") != fingerprint:
        return "fingerprint_mismatch"
    if not _same_path(entry.get("artifact_path"), artifact_path):
        return "artifact_mismatch"
    if entry.get("artifact_digest") != artifact_digest_value:
        return "artifact_content_mismatch"
    if entry.get("execution_engine_digest") != execution_engine_digest_value:
        return "execution_engine_mismatch"
    if product is not None:
        approved_product = entry.get("product")
        if not isinstance(approved_product, dict):
            return "product_mismatch"
        if (
            approved_product.get("name") != product.name
            or approved_product.get("objective") != product.objective
            or approved_product.get("market") != product.market
            or str(approved_product.get("symbol", "")).upper() != product.symbol.upper()
            or str(approved_product.get("base_asset", "")).upper() != product.base_asset.upper()
            or approved_product.get("starting_equity") != product.starting_equity
            or approved_product.get("regime_guard") is not product.regime_guard
            or approved_product.get("regime_mayer_top") != product.regime_mayer_top
        ):
            return "product_mismatch"
    return "approved"


def _holdout_return(strategy: dict[str, Any]) -> tuple[float | None, str | None]:
    metrics = strategy.get("metrics", {})
    value = metrics.get("holdout_total_return")
    if value is None:
        return None, "missing holdout_total_return metric"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, f"holdout_total_return metric must be numeric; got {value!r}"
    if not math.isfinite(parsed):
        return None, "holdout_total_return metric must be finite"
    return parsed, None


def _format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(parsed):
        return ""
    return f"{parsed:.4f}"


def _recommendation(
    strategy: dict[str, Any],
    paper: dict[str, Any],
    approval_status: str,
    thresholds: PromotionThresholds,
    policy_errors: list[str] | None = None,
    threshold_errors: list[str] | None = None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    policy_errors = policy_errors or []
    reasons.extend(policy_errors)
    threshold_errors = threshold_errors or []
    reasons.extend(threshold_errors)
    if approval_status == "ledger_error":
        reasons.append("approval ledger could not be read; suppressing promotion command")
    elif approval_status in {"ledger_malformed", "malformed"}:
        reasons.append("approval ledger entry is malformed; suppressing promotion command")
    holdout, holdout_error = _holdout_return(strategy)
    if holdout_error is not None:
        reasons.append(holdout_error)
    elif holdout <= thresholds.min_holdout_return:
        reasons.append(f"holdout_total_return {holdout:.6f} <= {thresholds.min_holdout_return:.6f}")
    if paper["trades"] < thresholds.min_paper_trades:
        reasons.append(
            "exact-fingerprint paper trades "
            f"{paper['trades']} < {thresholds.min_paper_trades}"
        )
    if paper.get("invalid_return_rows", 0) > 0:
        reasons.append(f"paper trade log has {paper['invalid_return_rows']} invalid return row(s)")
    if paper["total_sized_return"] <= thresholds.min_paper_sized_return:
        reasons.append(
            f"paper total_sized_return {paper['total_sized_return']:.6f} "
            f"<= {thresholds.min_paper_sized_return:.6f}"
        )
    if paper.get("max_drawdown") is not None and paper["max_drawdown"] > thresholds.max_paper_drawdown:
        reasons.append(
            f"paper max_drawdown {paper['max_drawdown']:.6f} "
            f"> {thresholds.max_paper_drawdown:.6f}"
        )
    if paper.get("max_consecutive_losses", 0) > thresholds.max_paper_consecutive_losses:
        reasons.append(
            f"paper max_consecutive_losses {paper['max_consecutive_losses']} "
            f"> {thresholds.max_paper_consecutive_losses}"
        )
    if paper.get("paper_days", 0.0) < thresholds.min_paper_days:
        reasons.append(f"paper days {paper.get('paper_days', 0.0):.2f} < {thresholds.min_paper_days:.2f}")
    if approval_status == "approved":
        if reasons:
            return "approved_review_failed", reasons
        return "already_approved", ["approved and passes configured review thresholds"]
    if reasons:
        return "not_ready", reasons
    return "needs_approval", ["passes configured review thresholds; human approval still required"]


def build_promotion_review(
    *,
    artifact_path: Path,
    trade_log: Path,
    ledger_path: Path,
    thresholds: PromotionThresholds | None = None,
    product: ProductConfig | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    thresholds = thresholds or PromotionThresholds()
    threshold_errors = _threshold_errors(thresholds)
    if not artifact_path.exists():
        return {
            "generated_at": utc_now(),
            "status": "waiting_for_strategy_artifact",
            "reason": f"Strategy artifact not found: {artifact_path}",
            "artifact_path": str(artifact_path),
            "trade_log": str(trade_log),
            "ledger_path": str(ledger_path),
            "product": None
            if product is None
            else {
                "name": product.name,
                "objective": product.objective,
                "market": product.market,
                "base_asset": product.base_asset,
                "symbol": product.symbol,
            },
            "thresholds": _threshold_payload(thresholds),
            "threshold_status": "fail" if threshold_errors else "pass",
            "threshold_errors": threshold_errors,
            "strategies": [],
        }
    artifact = load_artifact(artifact_path)
    current_artifact_digest = artifact_digest(artifact)
    current_execution_engine_digest = execution_engine_digest()
    trades = _load_trade_log(trade_log)
    ledger_error = None
    try:
        ledger_payload = ApprovalLedger(ledger_path).load()
    except (ApprovalError, OSError, json.JSONDecodeError) as exc:
        ledger_payload = {"version": 1, "approvals": {}}
        ledger_error = f"{type(exc).__name__}: {exc}"
    per_strategy_policy_errors = [
        validate_strategy(product, strategy, index) if product is not None else []
        for index, strategy in enumerate(artifact.get("strategies", []))
    ]
    if product is None:
        artifact_policy_errors: list[str] = []
    else:
        strategy_error_set = {
            error
            for errors in per_strategy_policy_errors
            for error in errors
        }
        artifact_policy_errors = [
            error
            for error in validate_strategy_artifact(product, artifact)
            if error not in strategy_error_set
        ]

    strategies = []
    for index, strategy in enumerate(artifact.get("strategies", [])):
        fingerprint = strategy_fingerprint(strategy)
        paper = _paper_stats(trades, strategy.get("id", ""), fingerprint)
        status = (
            "ledger_error"
            if ledger_error is not None
            else _approval_status(
                ledger_payload,
                fingerprint,
                artifact_path=artifact_path,
                artifact_digest_value=current_artifact_digest,
                execution_engine_digest_value=current_execution_engine_digest,
                product=product,
            )
        )
        policy_errors = per_strategy_policy_errors[index] + artifact_policy_errors
        recommendation, reasons = _recommendation(
            strategy,
            paper,
            status,
            thresholds,
            policy_errors,
            threshold_errors,
        )
        approval_command = None
        if recommendation == "needs_approval" and product is None:
            recommendation = "not_ready"
            reasons.append("product context is required before generating an approval command")
        if recommendation == "needs_approval":
            command = [
                "python",
                "-m",
                "src.autopilot.approvals",
                "--ledger",
                str(ledger_path),
                "approve",
            ]
            if product is not None:
                command.extend(["--config", str(config_path), "--product", product.name])
            command.extend(
                [
                    "--artifact",
                    str(artifact_path),
                    "--expected-artifact-digest",
                    current_artifact_digest,
                    "--strategy-id",
                    str(strategy.get("id")),
                    "--approved-by",
                    "<your-name>",
                    "--confirm-live",
                ]
            )
            approval_command = shlex.join(command)
        strategies.append(
            {
                "id": strategy.get("id"),
                "fingerprint": fingerprint,
                "approval_status": status,
                "recommendation": recommendation,
                "reasons": reasons,
                "direction": strategy.get("direction"),
                "base_timeframe": strategy.get("base_timeframe"),
                "pnl_unit": strategy.get("pnl_unit"),
                "risk": strategy.get("risk", {}),
                "metrics": strategy.get("metrics", {}),
                "paper": paper,
                "policy_status": "not_checked"
                if product is None
                else ("pass" if not policy_errors else "fail"),
                "policy_errors": policy_errors,
                "approval_command": approval_command,
            }
        )

    return {
        "generated_at": utc_now(),
        "artifact_path": str(artifact_path),
        "artifact_digest": current_artifact_digest,
        "execution_engine_digest": current_execution_engine_digest,
        "trade_log": str(trade_log),
        "ledger_path": str(ledger_path),
        "product": None
        if product is None
        else {
            "name": product.name,
            "objective": product.objective,
            "market": product.market,
            "base_asset": product.base_asset,
            "symbol": product.symbol,
        },
        "thresholds": {
            **_threshold_payload(thresholds),
        },
        "threshold_status": "fail" if threshold_errors else "pass",
        "threshold_errors": threshold_errors,
        "artifact_policy_status": "not_checked"
        if product is None
        else ("pass" if not artifact_policy_errors else "fail"),
        "artifact_policy_errors": artifact_policy_errors,
        "approval_ledger": {
            "path": str(ledger_path),
            "ok": ledger_error is None,
            "error": ledger_error,
        },
        "strategies": strategies,
    }


def _threshold_payload(thresholds: PromotionThresholds) -> dict[str, Any]:
    return {
        "min_paper_trades": thresholds.min_paper_trades,
        "min_paper_sized_return": thresholds.min_paper_sized_return,
        "min_holdout_return": thresholds.min_holdout_return,
        "max_paper_drawdown": thresholds.max_paper_drawdown,
        "max_paper_consecutive_losses": thresholds.max_paper_consecutive_losses,
        "min_paper_days": thresholds.min_paper_days,
    }


def render_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Strategy Promotion Review",
        "",
        f"Generated: `{review['generated_at']}`",
        f"Artifact: `{review['artifact_path']}`",
        f"Artifact digest: `{review.get('artifact_digest') or 'unavailable'}`",
        f"Execution engine digest: `{review.get('execution_engine_digest') or 'unavailable'}`",
        f"Trade log: `{review['trade_log']}`",
        "",
    ]
    approval_ledger = review.get("approval_ledger")
    if isinstance(approval_ledger, dict):
        lines.extend(
            [
                f"Approval ledger: `{'ok' if approval_ledger.get('ok') else 'error'}`",
            ]
        )
        if approval_ledger.get("error"):
            lines.append(f"Approval ledger error: `{approval_ledger['error']}`")
        lines.append("")
    if review.get("status") == "waiting_for_strategy_artifact":
        lines.extend(
            [
                f"Status: `{review['status']}`",
                "",
                str(review.get("reason", "Strategy artifact is not available yet.")),
            ]
        )
        return "\n".join(lines)
    threshold_status = review.get("threshold_status")
    if threshold_status:
        lines.append(f"Thresholds: `{threshold_status}`")
        for error in review.get("threshold_errors") or []:
            lines.append(f"- {error}")
        lines.append("")
    lines.extend(
        [
            "| Strategy | Recommendation | Approval | Paper trades | Paper return | Paper DD | Loss streak | Holdout |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in review["strategies"]:
        holdout_text = _format_optional_float(item["metrics"].get("holdout_total_return"))
        drawdown_text = _format_optional_float(item["paper"].get("max_drawdown"))
        lines.append(
            f"| `{item['id']}` | `{item['recommendation']}` | `{item['approval_status']}` | "
            f"{item['paper']['trades']} | {item['paper']['total_sized_return']:.4f} | "
            f"{drawdown_text} | {item['paper'].get('max_consecutive_losses', 0)} | {holdout_text} |"
        )
    lines.append("")
    for item in review["strategies"]:
        lines.extend(
            [
                f"## `{item['id']}`",
                "",
                f"Fingerprint: `{item['fingerprint']}`",
                f"Policy: `{item.get('policy_status', 'not_checked')}`",
                "",
                "Reasons:",
            ]
        )
        for reason in item["reasons"]:
            lines.append(f"- {reason}")
        lines.extend(["", "Approval command:", ""])
        if item.get("approval_command"):
            lines.extend([f"```bash\n{item['approval_command']}\n```", ""])
        else:
            lines.extend(["Not emitted because this strategy is not in `needs_approval` state.", ""])
    return "\n".join(lines)


def write_review(review: dict[str, Any], output_json: Path, output_md: Path | None = None) -> None:
    write_json_atomic(output_json, review)
    if output_md is not None:
        write_text_atomic(output_md, render_markdown(review))


def find_product_for_review(config_path: Path, product_name: str | None, artifact_path: Path) -> ProductConfig | None:
    if product_name is None and not config_path.exists():
        return None
    config = load_config(config_path)
    if product_name:
        for product in config.products:
            if product.name == product_name:
                return product
        raise ValueError(f"No product named {product_name!r} in {config_path}.")

    expected = artifact_path.resolve()
    matches = [product for product in config.products if product.strategies_path.resolve() == expected]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple products use artifact {artifact_path}; pass --product.")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strategy promotion review packet.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--trade-log", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=PROJECT_ROOT / "runtime" / "approvals.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--product", help="Product name for product-aware strategy policy checks.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--min-paper-trades", type=int, default=20)
    parser.add_argument("--min-paper-sized-return", type=float, default=0.0)
    parser.add_argument("--min-holdout-return", type=float, default=0.0)
    parser.add_argument("--max-paper-drawdown", type=float, default=0.05)
    parser.add_argument("--max-paper-consecutive-losses", type=int, default=4)
    parser.add_argument("--min-paper-days", type=float, default=7.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    product = find_product_for_review(args.config, args.product, args.artifact)
    review = build_promotion_review(
        artifact_path=args.artifact,
        trade_log=args.trade_log,
        ledger_path=args.ledger,
        product=product,
        config_path=args.config,
        thresholds=PromotionThresholds(
            min_paper_trades=args.min_paper_trades,
            min_paper_sized_return=args.min_paper_sized_return,
            min_holdout_return=args.min_holdout_return,
            max_paper_drawdown=args.max_paper_drawdown,
            max_paper_consecutive_losses=args.max_paper_consecutive_losses,
            min_paper_days=args.min_paper_days,
        ),
    )
    write_review(review, args.output_json, args.output_md)
    print(f"Wrote {args.output_json}")
    if args.output_md is not None:
        print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
