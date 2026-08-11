import json
from pathlib import Path

import pandas as pd

from src.autopilot import ml_forward_paper, ml_research


def _config(tmp_path: Path):
    payload = json.loads(Path("config/ml_research.json").read_text(encoding="utf-8"))
    path = tmp_path / "ml.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ml_research.load_config(path)


def _candidate(frame, spec):
    training = frame.iloc[:5]
    return {
        "schema": "autopilot.ml_forward_paper_candidate/v1",
        "experiment_id": spec["experiment_id"],
        "behavior_hash": "sha256:" + "1" * 64,
        "snapshot_id": "ml:test",
        "spec": spec,
        "training_content_sha256": ml_research.frame_content_sha256(training),
        "training_start": str(training.index[0]),
        "training_end": str(training.index[-1]),
        "forward_start_after": str(training.index[-1]),
        "frozen_model": {
            "schema": "autopilot.frozen_gradient_boosting/v1",
            "kind": "classifier",
            "feature_names": ["close"],
            "learning_rate": 0.1,
            "initial_prediction": 10.0,
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
        },
        "promotion_eligible": False,
        "live_allowed": False,
    }


def test_ml_forward_paper_refits_only_immutable_training_slice(tmp_path, monkeypatch):
    config = _config(tmp_path)
    spec = {**ml_research.experiment_grid(config)[0], "horizon": 2}
    index = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    frame = pd.DataFrame({"close": [100, 100, 100, 100, 100, 101, 102, 103]}, index=index)
    candidate = _candidate(frame, spec)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema": "autopilot.ml_research_report/v1",
                "trials": [{"forward_paper_candidate": candidate}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ml_forward_paper, "_load_dataset", lambda *args: frame)
    report = ml_forward_paper.run_cycle(
        config,
        candidates_path=candidates,
        state_path=tmp_path / "state.json",
        output_path=tmp_path / "output.json",
    )

    assert report["ok"] is True
    assert report["summary"]["completed_trades"] == 1
    assert report["safety"]["live_allowed"] is False


def test_ml_forward_paper_rejects_training_history_drift(tmp_path, monkeypatch):
    config = _config(tmp_path)
    spec = ml_research.experiment_grid(config)[0]
    index = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    frame = pd.DataFrame({"close": range(100, 108)}, index=index)
    candidate = _candidate(frame, spec)
    candidate["training_content_sha256"] = "0" * 64
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema": "autopilot.ml_research_report/v1",
                "trials": [{"forward_paper_candidate": candidate}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ml_forward_paper, "_load_dataset", lambda *args: frame)

    report = ml_forward_paper.run_cycle(
        config,
        candidates_path=candidates,
        state_path=tmp_path / "state.json",
        output_path=tmp_path / "output.json",
    )

    assert report["ok"] is False
    assert "digest mismatch" in report["candidates"][0]["error"]
