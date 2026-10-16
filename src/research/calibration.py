"""Bounded calibration of selection inference and causal feature admission."""

import argparse
import datetime as dt
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.domain._codec import canonical_hash
from src.research.selection import METHOD, FrozenCohort, selection_adjusted_block_bootstrap

GENERATOR_VERSION = "dependent_return_calibration/v2"
CASES = ("null", "weak_edge", "cost_trap", "regime_limited", "leakage_trap", "flat")


def causal_prefix_probe(function: Callable[[np.ndarray], np.ndarray], values: np.ndarray) -> bool:
    original = np.asarray(function(values.copy()))
    if original.shape != values.shape:
        raise ValueError("causality probe requires an aligned feature sequence")
    for cut in (len(values) // 4, len(values) // 2, 3 * len(values) // 4):
        changed = values.copy()
        changed[cut + 1 :] += 1000
        if not np.allclose(original[: cut + 1], function(changed)[: cut + 1], rtol=0, atol=1e-12):
            return False
    return True


def _returns(case: str, seed: int, count: int, candidates: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.003, (count, 1))
    innovations = common * 0.7 + rng.normal(0, 0.003, (count, candidates)) * 0.3
    values = np.zeros((count, candidates))
    for index in range(1, count):
        values[index] = 0.4 * values[index - 1] + 0.6 * innovations[index]
    adjustments = {
        "null": 0.0,
        "weak_edge": 0.0003,
        "cost_trap": -0.0007,
        "regime_limited": np.where(np.arange(count)[:, None] < count // 2, 0.001, -0.001),
        "leakage_trap": 0.003,
        "flat": 0.0,
    }
    return (np.zeros_like(values) if case == "flat" else values + adjustments[case])[:, :, None]


def _causality(case: str, seed: int) -> bool:
    prices = np.cumsum(np.random.default_rng(seed).normal(size=100)) + 100
    function = (
        (lambda values: np.roll(values, -1) - values) if case == "leakage_trap" else np.cumsum
    )
    return causal_prefix_probe(function, prices)


def _calibration_trial(
    case: str, seed: int, count: int, alpha: float, candidates: int
) -> dict[str, Any]:
    development = _returns("null", seed * 2, count, candidates)
    evaluation = _returns(case, seed * 2 + 1, count, candidates)
    cohort = FrozenCohort(
        tuple(f"configuration-{index}" for index in range(candidates)),
        ("synthetic_leg",),
        "2026-01-01T00:00:00+00:00",
        canonical_hash(development.tolist()),
    )
    start = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    result = selection_adjusted_block_bootstrap(
        cohort,
        timestamps=[(start + dt.timedelta(hours=index)).isoformat() for index in range(count)],
        net_returns=evaluation,
        benchmark_returns=[0.0] * count,
        development_returns=development,
        opened_at="2026-01-02T00:00:00+00:00",
        seed=seed,
        draws=2000,
    )
    causal = _causality(case, seed)
    qualified = (
        causal
        and result["status"] == "measured"
        and min(result["adjusted_pvalues"].values()) <= alpha
    )
    return {
        "case": case,
        "seed": seed,
        "causal": causal,
        "qualified_in_simulation": qualified,
        "inference": result,
    }


def calibrate(
    *,
    seed: int,
    replicates: int,
    policy_hash: str,
    maximum_seconds: float,
    proposed_alpha: float = 0.1,
    test_alpha: float = 0.1,
    observations: int = 256,
    candidates: int = 3,
) -> dict[str, Any]:
    if (
        not 1 <= candidates <= 100
        or not 1 <= replicates <= 500
        or observations < 128
        or maximum_seconds <= 0
    ):
        raise ValueError("calibration resource bounds are invalid")
    if not 0 < proposed_alpha < 1:
        raise ValueError("proposed false-qualification budget must be in (0, 1)")
    if not 0 < test_alpha < 1:
        raise ValueError("nominal test cutoff must be in (0, 1)")
    started = time.monotonic()
    trials = []
    for case_index, case in enumerate(CASES):
        for replicate in range(replicates):
            if time.monotonic() - started > maximum_seconds:
                raise TimeoutError("calibration exceeded its declared time budget")
            trials.append(
                _calibration_trial(
                    case,
                    seed + case_index * 10000 + replicate,
                    observations,
                    test_alpha,
                    candidates,
                )
            )
    summary = {
        case: _case_summary([trial for trial in trials if trial["case"] == case]) for case in CASES
    }
    return {
        "schema": "research.selection_calibration/v1",
        "method": METHOD,
        "generator_version": GENERATOR_VERSION,
        "policy_hash": policy_hash,
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("selection.py", "calibration.py")
        },
        "seed": seed,
        "candidates": candidates,
        "calibration_scope": "selection_inference_kernel",
        "replicates": replicates,
        "observations": observations,
        "proposed_false_qualification_budget": proposed_alpha,
        "nominal_test_alpha": test_alpha,
        "summary": summary,
        "maximum_seconds": maximum_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "trials": trials,
        "qualification_authority": False,
        "policy_approval_required": True,
        "limitations": [
            "finite_seed_sample",
            "specified_synthetic_dependence_model",
            "not_a_probability_of_future_trading_loss",
            "economic_execution_and_lifecycle_require_separate_contract_validation",
        ],
    }


def _case_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(trials)
    qualified = sum(trial["qualified_in_simulation"] for trial in trials)
    deferred = sum(trial["inference"]["status"] == "inconclusive" for trial in trials)
    rate = qualified / count
    # Wilson interval exposes sampling uncertainty rather than hiding it behind
    # a point-estimate pass flag.
    z = 1.96
    centre = (rate + z * z / (2 * count)) / (1 + z * z / count)
    radius = (
        z * ((rate * (1 - rate) / count + z * z / (4 * count * count)) ** 0.5) / (1 + z * z / count)
    )
    return {
        "trials": count,
        "qualification_rate": rate,
        "deferral_rate": deferred / count,
        "qualification_rate_interval_95": [max(0, centre - radius), min(1, centre + radius)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate selection inference without enabling policy"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("config/research.json"))
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--maximum-seconds", type=float, default=7200)
    parser.add_argument("--test-alpha", type=float, default=0.05)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    result = calibrate(
        seed=args.seed,
        replicates=args.replicates,
        policy_hash=canonical_hash(policy),
        candidates=int(policy["search_budgets"]["maximum_candidates_per_epoch"]),
        maximum_seconds=args.maximum_seconds,
        test_alpha=args.test_alpha,
    )
    encoded = json.dumps(result, sort_keys=True, indent=2)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / (hashlib.sha256(encoded.encode()).hexdigest() + ".json")
    with path.open("x") as handle:
        handle.write(encoded)
    print(
        json.dumps(
            {
                "path": str(path.resolve()),
                "summary": result["summary"],
                "qualification_authority": False,
            }
        )
    )


if __name__ == "__main__":
    main()
