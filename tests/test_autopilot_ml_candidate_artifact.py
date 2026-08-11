import json
from pathlib import Path

import pytest

from research_exploration.dsr import DSR_METHOD
from src.autopilot.approvals import artifact_digest
from src.autopilot.config import ProductConfig
from src.autopilot.ml_candidate_artifact import (
    MlCandidateArtifactError,
    build_reviewable_artifact,
    export_reviewable_artifact,
    stage_reviewable_artifact,
)
from src.autopilot.strategy_policy import validate_strategy_artifact


def _product(tmp_path: Path) -> ProductConfig:
    return ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="paper",
        symbol="BTCUSDT",
        strategies_path=tmp_path / "active.json",
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
        starting_equity=1000.0,
    )


def _btc_product(tmp_path: Path) -> ProductConfig:
    return ProductConfig(
        name="btc_accumulation",
        enabled=True,
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="paper",
        symbol="BTCUSDT",
        strategies_path=tmp_path / "active-btc.json",
        state_file=tmp_path / "state-btc.json",
        trade_log=tmp_path / "trades-btc.csv",
        starting_equity=1.0,
        regime_guard=True,
    )


def _trial() -> dict:
    frozen = {
        "schema": "autopilot.frozen_gradient_boosting/v1",
        "kind": "classifier",
        "feature_names": ["tf_15m_close"],
        "learning_rate": 0.1,
        "initial_prediction": 0.0,
        "trees": [
            {
                "children_left": [-1],
                "children_right": [-1],
                "feature": [-2],
                "threshold": [-2.0],
                "value": [0.0],
            }
        ],
        "long_threshold": 0.55,
        "short_threshold": 0.45,
    }
    spec = {
        "product": "active_income",
        "market": "futures",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "pnl_unit": "usdt",
        "horizon": 12,
        "regime": "trend",
        "experiment_id": "ml-test",
    }
    return {
        **spec,
        "holdout_eligible": True,
        "protected_holdout": {
            "eligible": True,
            "holdout_start": "2026-01-01T00:00:00+00:00",
            "holdout_end": "2026-02-01T00:00:00+00:00",
            "metrics": {
                "trades": 24,
                "total_return": 0.08,
                "max_drawdown": -0.02,
                "win_rate": 0.58,
                "sharpe": 0.4,
                "dsr_deflated": 0.73,
                "dsr_method": DSR_METHOD,
                "n_trials": 4608,
                "sr_std_trials": 0.1,
                "trial_sharpe_count": 1,
                "trial_sharpe_observed_std": 0.0,
                "trial_sharpe_conservative_floor": 0.1,
            },
        },
        "forward_paper_candidate": {
            "experiment_id": "ml-test",
            "behavior_hash": "sha256:" + "b" * 64,
            "snapshot_id": "ml:" + "a" * 64,
            "training_content_sha256": "c" * 64,
            "training_start": "2025-01-01T00:00:00+00:00",
            "training_end": "2025-12-31T00:00:00+00:00",
            "spec": spec,
            "frozen_model": frozen,
        },
    }


def test_builds_policy_compliant_reviewable_frozen_candidate(tmp_path):
    product = _product(tmp_path)

    artifact = build_reviewable_artifact(_trial(), product)

    assert validate_strategy_artifact(product, artifact, require_live_eligible=True) == []
    assert {item["direction"] for item in artifact["strategies"]} == {"long", "short"}
    assert all(item["ml_regime"] == "trend" for item in artifact["strategies"])
    assert artifact["candidate_only"] is True


def test_builds_btc_denominated_step_aside_candidate(tmp_path):
    trial = _trial()
    trial.update(product="btc_accumulation", market="spot", timeframe="1h", pnl_unit="btc")
    spec = trial["forward_paper_candidate"]["spec"]
    spec.update(product="btc_accumulation", market="spot", timeframe="1h", pnl_unit="btc")
    spec["regime_close_feature"] = "close"
    trial["forward_paper_candidate"]["frozen_model"]["feature_names"] = ["tf_1h_close"]
    product = _btc_product(tmp_path)

    artifact = build_reviewable_artifact(trial, product)

    assert validate_strategy_artifact(product, artifact, require_live_eligible=True) == []
    assert [item["direction"] for item in artifact["strategies"]] == ["short"]
    metrics = artifact["strategies"][0]["metrics"]
    assert metrics["holdout_excess_return_vs_buy_hold"] == metrics["holdout_total_return"]


def test_exports_digest_named_evidence_then_stages_only_exact_review(tmp_path):
    product = _product(tmp_path)
    exported = export_reviewable_artifact(_trial(), product, output_dir=tmp_path / "review")
    source = Path(exported["path"])

    staged = stage_reviewable_artifact(
        source,
        product,
        expected_digest=exported["artifact_digest"],
        candidate_dir=tmp_path / "staged",
    )

    assert staged["staged"] is True
    candidate = Path(staged["candidate"])
    assert candidate.name == "active_income.json"
    assert artifact_digest(json.loads(candidate.read_text())) == exported["artifact_digest"]


def test_stage_rejects_digest_drift(tmp_path):
    product = _product(tmp_path)
    exported = export_reviewable_artifact(_trial(), product, output_dir=tmp_path / "review")

    with pytest.raises(MlCandidateArtifactError, match="digest changed"):
        stage_reviewable_artifact(
            Path(exported["path"]),
            product,
            expected_digest="sha256:" + "0" * 64,
            candidate_dir=tmp_path / "staged",
        )


def test_stage_rejects_non_standard_json_before_digest_check(tmp_path):
    source = tmp_path / "candidate.json"
    source.write_text('{"unsafe": NaN}', encoding="utf-8")

    with pytest.raises(MlCandidateArtifactError, match="non-standard JSON constant"):
        stage_reviewable_artifact(
            source,
            _product(tmp_path),
            expected_digest="sha256:" + "0" * 64,
            candidate_dir=tmp_path / "staged",
        )


def test_build_rejects_model_feature_live_engine_cannot_reproduce(tmp_path):
    trial = _trial()
    trial["forward_paper_candidate"]["frozen_model"]["feature_names"] = ["offline_only_feature"]

    with pytest.raises(MlCandidateArtifactError, match="unavailable to live inference"):
        build_reviewable_artifact(trial, _product(tmp_path))
