"""Build and explicitly stage executable frozen-ML candidate artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import build_binance_indicator_dataset as indicator_builder
from research_exploration.dsr import DSR_METHOD
from src.alpha.frozen_gradient_boosting import FrozenGradientBoostingModel
from src.autopilot.approvals import artifact_digest
from src.autopilot.candidate_activation import candidate_path_for_product, product_identity
from src.autopilot.config import DEFAULT_CONFIG_PATH, ProductConfig, load_config
from src.autopilot.io import write_json_atomic
from src.autopilot.strategy_policy import assert_loaded_strategy_artifact_allowed
from src.config import PROJECT_ROOT

DEFAULT_REVIEWABLE_DIR = PROJECT_ROOT / "runtime" / "research" / "ml_candidates"
DEFAULT_STAGED_DIR = PROJECT_ROOT / "runtime" / "candidates"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class MlCandidateArtifactError(ValueError):
    """Frozen-ML evidence cannot form a policy-compliant candidate."""


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MlCandidateArtifactError(f"{label} must be a regular non-symlink file")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise MlCandidateArtifactError(f"{label} repeats JSON key {key!r}")
            payload[key] = value
        return payload

    def reject_constant(value: str) -> None:
        raise MlCandidateArtifactError(f"{label} contains non-standard JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except MlCandidateArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MlCandidateArtifactError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MlCandidateArtifactError(f"{label} must be a JSON object")
    return payload


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MlCandidateArtifactError(f"{label} must be numeric") from exc
    if not (-float("inf") < result < float("inf")):
        raise MlCandidateArtifactError(f"{label} must be finite")
    return result


def _headline_metrics(metrics: Mapping[str, Any], product: ProductConfig) -> dict[str, Any]:
    total_return = _finite(metrics.get("total_return"), "holdout total_return")
    trades = metrics.get("trades")
    if isinstance(trades, bool) or not isinstance(trades, int) or trades <= 0:
        raise MlCandidateArtifactError("holdout trades must be a positive integer")
    headline: dict[str, Any] = {
        "holdout_total_return": total_return,
        "holdout_trades": trades,
        "holdout_max_drawdown": _finite(metrics.get("max_drawdown"), "holdout max_drawdown"),
    }
    for source, destination in (
        ("win_rate", "holdout_win_rate"),
        ("sharpe", "holdout_sharpe"),
        ("profit_factor", "holdout_profit_factor"),
        ("dsr_deflated", "dsr_deflated"),
        ("sr_std_trials", "sr_std_trials"),
        ("trial_sharpe_observed_std", "trial_sharpe_observed_std"),
        ("trial_sharpe_conservative_floor", "trial_sharpe_conservative_floor"),
    ):
        if metrics.get(source) is not None:
            headline[destination] = _finite(metrics[source], source)
    if metrics.get("dsr_method") != DSR_METHOD:
        raise MlCandidateArtifactError("protected holdout DSR method is not current")
    headline["dsr_method"] = DSR_METHOD
    for key in ("n_trials", "trial_sharpe_count"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MlCandidateArtifactError(f"{key} must be a non-negative integer")
        headline[key] = value
    if headline["n_trials"] < 1:
        raise MlCandidateArtifactError("n_trials must be positive")
    if product.objective == "btc_accumulation":
        # BTC-denominated backtests express the strategy's extra BTC versus
        # passively holding the starting BTC, whose BTC-unit return is zero.
        headline["holdout_buy_hold_return"] = 0.0
        headline["holdout_excess_return_vs_buy_hold"] = total_return
    return headline


def _risk(product: ProductConfig) -> dict[str, Any]:
    if product.objective == "btc_accumulation":
        return {
            "risk_per_trade": 0.003,
            "max_position_fraction": 0.35,
            "daily_stop_loss": -0.01,
            "max_consecutive_losses": 3,
            "cooldown_bars": 24,
            "max_trades_per_day": 2,
        }
    return {
        "risk_per_trade": 0.005,
        "max_position_fraction": 0.25,
        "daily_stop_loss": -0.03,
        "max_consecutive_losses": 4,
        "cooldown_bars": 12,
        "max_trades_per_day": 8,
    }


def _candidate_context(
    trial: Mapping[str, Any], product: ProductConfig
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, Any],
    FrozenGradientBoostingModel,
    Mapping[str, Any],
    dict[str, Any],
    str,
]:
    if trial.get("holdout_eligible") is not True:
        raise MlCandidateArtifactError("trial did not pass the protected holdout")
    protected = trial.get("protected_holdout")
    forward = trial.get("forward_paper_candidate")
    if not isinstance(protected, Mapping) or not isinstance(forward, Mapping):
        raise MlCandidateArtifactError("trial has no frozen forward-paper evidence")
    spec = forward.get("spec")
    frozen = forward.get("frozen_model")
    if not isinstance(spec, Mapping) or not isinstance(frozen, dict):
        raise MlCandidateArtifactError("frozen candidate payload is malformed")
    expected = (product.name, product.market, product.symbol.upper())
    observed = (spec.get("product"), spec.get("market"), str(spec.get("symbol") or "").upper())
    if observed != expected:
        raise MlCandidateArtifactError(
            f"candidate identity mismatch: expected {expected!r}, got {observed!r}"
        )
    model = FrozenGradientBoostingModel.from_dict(frozen)
    live_feature_names = {
        feature.split("_", 2)[2]
        if feature.startswith("tf_") and feature.count("_") >= 2
        else feature
        for feature in model.feature_names
    }
    unsupported = indicator_builder.unsupported_required_features(live_feature_names)
    if unsupported:
        raise MlCandidateArtifactError(
            "frozen model uses features unavailable to live inference: "
            + ", ".join(sorted(unsupported))
        )
    metrics_raw = protected.get("metrics")
    if not isinstance(metrics_raw, Mapping):
        raise MlCandidateArtifactError("protected holdout metrics are missing")
    metrics = _headline_metrics(metrics_raw, product)
    experiment_id = str(trial.get("experiment_id") or "")
    if not SAFE_ID_RE.fullmatch(experiment_id):
        raise MlCandidateArtifactError("experiment_id is unsafe")
    return protected, forward, spec, frozen, model, metrics_raw, metrics, experiment_id


def _build_strategies(
    product: ProductConfig,
    spec: Mapping[str, Any],
    frozen: dict[str, Any],
    model: FrozenGradientBoostingModel,
    metrics_raw: Mapping[str, Any],
    metrics: dict[str, Any],
    experiment_id: str,
) -> list[dict[str, Any]]:
    directions = ("short",) if product.objective == "btc_accumulation" else ("long", "short")
    strategies = []
    for rank, direction in enumerate(directions, start=1):
        strategy: dict[str, Any] = {
            "id": f"{experiment_id}_{direction}",
            "rank": rank,
            "market": product.market,
            "symbol": product.symbol.upper(),
            "entry_type": "frozen_ml",
            "frozen_model": frozen,
            "ml_regime": str(spec.get("regime") or "all"),
            "ml_regime_close_feature": str(spec.get("regime_close_feature") or "close"),
            "base_timeframe": str(spec.get("timeframe") or ""),
            "direction": direction,
            "horizon_bars": int(spec.get("horizon") or 0),
            "take_profit": 0.05,
            "stop_loss": 0.03,
            "use_atr_tp_sl": False,
            "pnl_unit": str(spec.get("pnl_unit") or ""),
            "risk": _risk(product),
            "fees": {"fee_bps": 10.0, "slippage_bps": 2.0},
            "metrics": metrics,
            "rule": (
                f"frozen {model.kind} {direction} threshold with {spec.get('regime', 'all')} regime"
            ),
        }
        if product.market == "futures":
            strategy.update(leverage=1, margin_mode="isolated")
        win_rate = metrics_raw.get("win_rate")
        if win_rate is not None and 0 < _finite(win_rate, "holdout win_rate") < 1:
            strategy["baseline_win_rate"] = float(win_rate)
        strategies.append(strategy)
    return strategies


def build_reviewable_artifact(
    trial: Mapping[str, Any],
    product: ProductConfig,
) -> dict[str, Any]:
    """Convert an eligible protected-holdout result to a reviewable artifact."""
    protected, forward, spec, frozen, model, metrics_raw, metrics, experiment_id = (
        _candidate_context(trial, product)
    )
    strategies = _build_strategies(
        product, spec, frozen, model, metrics_raw, metrics, experiment_id
    )
    artifact = {
        "version": 3,
        "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "market": product.market,
        "symbol": product.symbol.upper(),
        "pnl_unit": str(spec.get("pnl_unit") or ""),
        "paper_trade_allowed": True,
        "live_allowed": True,
        "promotion_eligible": True,
        "candidate_only": True,
        "product": product_identity(product),
        "research_evidence": {
            "experiment_id": experiment_id,
            "behavior_hash": forward.get("behavior_hash"),
            "snapshot_id": forward.get("snapshot_id"),
            "training_content_sha256": forward.get("training_content_sha256"),
            "training_start": forward.get("training_start"),
            "training_end": forward.get("training_end"),
            "protected_holdout_start": protected.get("holdout_start"),
            "protected_holdout_end": protected.get("holdout_end"),
            "model_format": frozen.get("schema"),
        },
        "strategies": strategies,
    }
    assert_loaded_strategy_artifact_allowed(
        product,
        artifact,
        artifact_path=Path(f"<generated:{experiment_id}>"),
        require_live_eligible=True,
    )
    return artifact


def export_reviewable_artifact(
    trial: Mapping[str, Any],
    product: ProductConfig,
    *,
    output_dir: Path = DEFAULT_REVIEWABLE_DIR,
) -> dict[str, Any]:
    artifact = build_reviewable_artifact(trial, product)
    output_dir = Path(output_dir)
    if output_dir.exists() and output_dir.is_symlink():
        raise MlCandidateArtifactError("reviewable output directory must not be a symlink")
    snapshot = str(artifact["research_evidence"].get("snapshot_id") or "")
    snapshot_key = re.sub(r"[^0-9a-f]", "", snapshot.lower())[-16:]
    if len(snapshot_key) != 16:
        raise MlCandidateArtifactError("snapshot_id has no stable digest suffix")
    experiment_id = str(artifact["research_evidence"]["experiment_id"])
    path = Path(output_dir) / f"{experiment_id}__{snapshot_key}.json"
    if path.is_symlink():
        raise MlCandidateArtifactError("reviewable artifact path must not be a symlink")
    write_json_atomic(path, artifact)
    return {
        "path": str(path),
        "artifact_digest": artifact_digest(artifact),
        "product": product.name,
        "experiment_id": experiment_id,
        "staged": False,
    }


def stage_reviewable_artifact(
    source: Path,
    product: ProductConfig,
    *,
    expected_digest: str,
    candidate_dir: Path = DEFAULT_STAGED_DIR,
    replace: bool = False,
) -> dict[str, Any]:
    if not DIGEST_RE.fullmatch(expected_digest):
        raise MlCandidateArtifactError("expected_digest must be a sha256 artifact digest")
    payload = _strict_json_object(source, "source artifact")
    actual = artifact_digest(payload)
    if actual != expected_digest:
        raise MlCandidateArtifactError(
            f"reviewed artifact digest changed: expected {expected_digest}, got {actual}"
        )
    if payload.get("product") != product_identity(product):
        raise MlCandidateArtifactError("reviewed artifact product identity mismatch")
    assert_loaded_strategy_artifact_allowed(
        product,
        payload,
        artifact_path=source,
        require_live_eligible=True,
    )
    destination = candidate_path_for_product(product.name, candidate_dir=candidate_dir)
    if candidate_dir.exists() and candidate_dir.is_symlink():
        raise MlCandidateArtifactError("candidate directory must not be a symlink")
    if destination.is_symlink():
        raise MlCandidateArtifactError("staged candidate path must not be a symlink")
    if destination.exists():
        current = _strict_json_object(destination, "staged candidate")
        current_digest = artifact_digest(current)
        if current_digest == actual:
            return {
                "ok": True,
                "staged": False,
                "reason": "already_staged",
                "candidate": str(destination),
                "artifact_digest": actual,
            }
        if not replace:
            raise MlCandidateArtifactError(
                f"staged candidate already exists with digest {current_digest}; use --replace"
            )
    write_json_atomic(destination, payload)
    return {
        "ok": True,
        "staged": True,
        "candidate": str(destination),
        "artifact_digest": actual,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--product", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_STAGED_DIR)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    products = [product for product in config.products if product.name == args.product]
    if len(products) != 1:
        raise MlCandidateArtifactError(f"product {args.product!r} is not uniquely configured")
    result = stage_reviewable_artifact(
        args.artifact,
        products[0],
        expected_digest=args.expected_digest,
        candidate_dir=args.candidate_dir,
        replace=args.replace,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
