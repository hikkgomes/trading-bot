"""Run isolated, non-promotable forward paper for protected-holdout ML candidates."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.alpha.frozen_gradient_boosting import FrozenGradientBoostingModel
from src.autopilot.io import write_json_atomic
from src.autopilot.ml_research import (
    DEFAULT_CONFIG,
    DatasetSpec,
    MlResearchConfig,
    _load_dataset,
    _regime_mask,
    frame_content_sha256,
    load_config,
)
from src.config import PROJECT_ROOT

REPORT_SCHEMA = "autopilot.ml_forward_paper_report/v1"
STATE_SCHEMA = "autopilot.ml_forward_paper_state/v1"
DEFAULT_CANDIDATES = PROJECT_ROOT / "runtime" / "research" / "ml_research.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime" / "research" / "ml_forward_paper.json"
DEFAULT_STATE = PROJECT_ROOT / "runtime" / "research" / "ml_forward_paper_state.json"
ROUND_TRIP_COST = 0.0024


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _load_json(path: Path, *, required: bool) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"JSON path must not be a symlink: {path}")
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _candidate_rows(payload: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
    if payload.get("schema") != "autopilot.ml_research_report/v1":
        raise ValueError("ML candidate report schema is invalid")
    candidates = [
        trial["forward_paper_candidate"]
        for trial in payload.get("trials", [])
        if isinstance(trial, dict) and isinstance(trial.get("forward_paper_candidate"), dict)
    ]
    return sorted(candidates, key=lambda item: str(item.get("experiment_id")))[:maximum]


def _dataset(config: MlResearchConfig, spec: dict[str, Any]) -> DatasetSpec:
    matches = [
        item
        for item in config.datasets
        if (item.product, item.market, item.symbol, item.timeframe, item.pnl_unit)
        == tuple(spec[key] for key in ("product", "market", "symbol", "timeframe", "pnl_unit"))
    ]
    if len(matches) != 1:
        raise ValueError("ML forward-paper candidate dataset is not uniquely configured")
    return matches[0]


def _close_column(frame: pd.DataFrame) -> str:
    candidates = [name for name in frame.columns if name == "close" or name.endswith("_close")]
    if not candidates:
        raise ValueError("ML forward paper requires a close column")
    return candidates[0]


def _initial_candidate_state(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "behavior_hash": candidate["behavior_hash"],
        "snapshot_id": candidate["snapshot_id"],
        "cursor": candidate["forward_start_after"],
        "equity": 1.0,
        "peak_equity": 1.0,
        "position": None,
        "trades": [],
    }


def _load_candidate_forward_context(
    config: MlResearchConfig, candidate: dict[str, Any], state: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if candidate.get("schema") != "autopilot.ml_forward_paper_candidate/v1":
        raise ValueError("ML forward-paper candidate schema is invalid")
    if (
        candidate.get("promotion_eligible") is not False
        or candidate.get("live_allowed") is not False
    ):
        raise ValueError("ML forward-paper candidate must be non-promotable and non-live")
    spec = candidate.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("ML forward-paper candidate spec is invalid")
    frame = _load_dataset(_dataset(config, spec), config.max_rows)
    training_start = pd.Timestamp(candidate["training_start"])
    training_end = pd.Timestamp(candidate["training_end"])
    training = frame.loc[(frame.index >= training_start) & (frame.index <= training_end)]
    if training.empty or frame_content_sha256(training) != candidate["training_content_sha256"]:
        raise ValueError("immutable ML training slice digest mismatch")
    cursor = pd.Timestamp(state.get("cursor") or candidate["forward_start_after"])
    forward = frame.loc[frame.index > cursor]
    return spec, training, forward


def _forward_signals(
    training: pd.DataFrame, forward: pd.DataFrame, spec: dict[str, Any], model
) -> pd.Series:
    context = pd.concat([training.tail(480), forward])
    regime = _regime_mask(
        context,
        str(spec["regime"]),
        str(spec.get("regime_close_feature") or "") or None,
    )
    signals = pd.Series(0, index=context.index, dtype=int)
    for timestamp, row in context.iterrows():
        values = row.to_dict()
        if model.triggered(values, "long"):
            signals.loc[timestamp] = 1
        elif model.triggered(values, "short"):
            signals.loc[timestamp] = -1
    return signals.where(regime, 0).reindex(forward.index).fillna(0).astype(int)


def _close_forward_position(
    state: dict[str, Any], spec: dict[str, Any], timestamp, price: float
) -> None:
    position = state.get("position")
    if not isinstance(position, dict):
        return
    position["bars_held"] = int(position.get("bars_held", 0)) + 1
    signed = 1.0 if position["direction"] == "long" else -1.0
    gross = signed * (price / float(position["entry_price"]) - 1)
    if not (gross >= 0.05 or gross <= -0.03 or position["bars_held"] >= int(spec["horizon"])):
        return
    net = gross - ROUND_TRIP_COST
    state["equity"] = float(state["equity"]) * (1 + net)
    state["peak_equity"] = max(float(state["peak_equity"]), float(state["equity"]))
    state.setdefault("trades", []).append(
        {
            "entry_time": position["entry_time"],
            "exit_time": timestamp.isoformat(),
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": price,
            "net_return": net,
        }
    )
    state["trades"] = state["trades"][-100:]
    state["position"] = None


def _open_forward_signal(state: dict[str, Any], signal: int, timestamp, price: float) -> None:
    if state.get("position") is not None or signal == 0:
        return
    state["position"] = {
        "direction": "long" if signal > 0 else "short",
        "entry_time": timestamp.isoformat(),
        "entry_price": price,
        "bars_held": 0,
    }


def _run_candidate(
    config: MlResearchConfig,
    candidate: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    spec, training, forward = _load_candidate_forward_context(config, candidate, state)
    if forward.empty:
        return {"status": "idle", "unseen_rows": 0}
    model = FrozenGradientBoostingModel.from_dict(candidate.get("frozen_model"))
    signals = _forward_signals(training, forward, spec, model)
    close_column = _close_column(forward)
    for timestamp, row in forward.iterrows():
        price = float(row[close_column])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("ML forward-paper close price is invalid")
        _close_forward_position(state, spec, timestamp, price)
        _open_forward_signal(state, int(signals.loc[timestamp]), timestamp, price)
        state["cursor"] = timestamp.isoformat()
    trades = state.get("trades") or []
    return {
        "status": "active",
        "unseen_rows": len(forward),
        "equity": state["equity"],
        "drawdown": float(state["equity"]) / float(state["peak_equity"]) - 1,
        "completed_trades": len(trades),
        "open_position": state.get("position") is not None,
        "last_cursor": state["cursor"],
    }


def run_cycle(
    config: MlResearchConfig,
    *,
    candidates_path: Path = DEFAULT_CANDIDATES,
    state_path: Path = DEFAULT_STATE,
    output_path: Path = DEFAULT_OUTPUT,
    maximum_candidates: int = 20,
) -> dict[str, Any]:
    if not 1 <= maximum_candidates <= 100:
        raise ValueError("maximum_candidates must be in [1, 100]")
    generated_at = _utc_now()
    if not candidates_path.exists():
        report = {
            "schema": REPORT_SCHEMA,
            "ok": True,
            "status": "waiting_for_candidates",
            "generated_at": generated_at,
            "candidates": [],
            "safety": _safety(),
        }
        write_json_atomic(output_path, report)
        return report
    candidates = _candidate_rows(_load_json(candidates_path, required=True), maximum_candidates)
    persisted = _load_json(state_path, required=False)
    states = persisted.get("candidates") if persisted.get("schema") == STATE_SCHEMA else {}
    if not isinstance(states, dict):
        raise ValueError("ML forward-paper state candidates must be an object")
    results = []
    for candidate in candidates:
        experiment_id = str(candidate.get("experiment_id") or "")
        state = states.get(experiment_id)
        if not isinstance(state, dict):
            state = _initial_candidate_state(candidate)
            states[experiment_id] = state
        if state.get("behavior_hash") != candidate.get("behavior_hash"):
            results.append(
                {"experiment_id": experiment_id, "status": "error", "error": "identity drift"}
            )
            continue
        try:
            result = _run_candidate(config, candidate, state)
        except (FileNotFoundError, OSError) as exc:
            result = {"status": "waiting_for_dataset", "detail": str(exc)}
        except Exception as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        results.append({"experiment_id": experiment_id, **result})
    write_json_atomic(
        state_path,
        {"schema": STATE_SCHEMA, "updated_at": generated_at, "candidates": states},
    )
    report = {
        "schema": REPORT_SCHEMA,
        "ok": not any(item["status"] == "error" for item in results),
        "status": "ready" if candidates else "waiting_for_candidates",
        "generated_at": generated_at,
        "candidates": results,
        "summary": {
            "configured": len(candidates),
            "active": sum(item["status"] == "active" for item in results),
            "waiting": sum(str(item["status"]).startswith("waiting") for item in results),
            "errors": sum(item["status"] == "error" for item in results),
            "completed_trades": sum(int(item.get("completed_trades") or 0) for item in results),
        },
        "safety": _safety(),
    }
    write_json_atomic(output_path, report)
    return report


def _safety() -> dict[str, Any]:
    return {
        "zero_money": True,
        "adaptive_feedback_allowed": False,
        "promotion_evidence_allowed": False,
        "live_allowed": False,
        "frozen_json_inference": True,
        "blocked_reason": "separate_reviewable_artifact_candidate_paper_and_approval_required",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-candidates", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_cycle(
        load_config(args.config),
        candidates_path=args.candidates,
        state_path=args.state,
        output_path=args.output,
        maximum_candidates=args.maximum_candidates,
    )
    print(json.dumps(report.get("summary") or {"status": report["status"]}, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
