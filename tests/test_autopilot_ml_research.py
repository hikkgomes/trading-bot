import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.autopilot import ml_research
from src.autopilot.experiment_memory import ExperimentMemory


def _config(tmp_path: Path) -> ml_research.MlResearchConfig:
    payload = json.loads(Path("config/ml_research.json").read_text(encoding="utf-8"))
    path = tmp_path / "ml.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return dataclasses.replace(
        ml_research.load_config(path), memory_path=tmp_path / "memory.sqlite3"
    )


def test_config_builds_bounded_cross_product(tmp_path):
    config = _config(tmp_path)
    grid = ml_research.experiment_grid(config)

    assert len(grid) == 2 * 3 * 4 * 2 * 3 * 4 * 2 * 2 * 2
    assert len({item["experiment_id"] for item in grid}) == len(grid)
    assert config.max_trials_per_cycle == 2


def test_chronological_windows_purge_label_horizon_and_embargo(tmp_path):
    config = dataclasses.replace(
        _config(tmp_path),
        min_train_rows=200,
        validation_rows=100,
        step_rows=100,
        min_windows=3,
    )
    windows = ml_research.chronological_windows(700, config, horizon=24)

    assert len(windows) == 3
    for train, validation in windows:
        assert train.stop + 24 + config.embargo_bars == validation.start
        assert validation.stop - validation.start == 100


def test_ml_selects_capable_epoch_before_protected_history(tmp_path):
    config = dataclasses.replace(
        _config(tmp_path),
        min_train_rows=200,
        validation_rows=100,
        step_rows=100,
        min_windows=2,
    )
    index = pd.date_range("2024-01-01", periods=1_400, freq="15min", tz="UTC")
    frame = pd.DataFrame({"close": range(len(index))}, index=index)
    protected = (
        {
            "interval_key": "protected",
            "market": "futures",
            "symbol": "BTCUSDT",
            "start": str(index[900]),
            "end": str(index[1_000]),
        },
    )

    selected, detail = ml_research._select_unprotected_epoch(
        frame,
        config.datasets[0],
        config,
        protected,
    )

    assert len(selected) == 900
    assert selected.index[-1] < index[900]
    assert detail["policy"] == "largest_contiguous_unprotected_ml_epoch"
    assert detail["protected_rows_excluded"] == 101
    assert detail["feature_dependency_embargo_rows_excluded"] == 399


def test_ml_waits_when_no_unprotected_epoch_has_enough_rows(tmp_path, monkeypatch):
    config = dataclasses.replace(_config(tmp_path), max_trials_per_cycle=1)
    frame = pd.DataFrame(
        {"close": range(1, 101)},
        index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"),
    )
    monkeypatch.setattr(ml_research, "_load_dataset", lambda *args: frame)
    monkeypatch.setattr(
        ml_research,
        "_select_unprotected_epoch",
        lambda *args: (_ for _ in ()).throw(
            ml_research.UnprotectedMlEpochUnavailableError("insufficient safe history")
        ),
    )

    report = ml_research.run_cycle(
        config,
        output_path=tmp_path / "report.json",
        state_path=tmp_path / "state.json",
    )

    assert report["ok"] is True
    assert report["summary"]["waiting"] == 1
    assert report["summary"]["errors"] == 0
    assert report["trials"][0]["status"] == "waiting_for_unprotected_epoch"


def test_cycle_waits_safely_when_datasets_are_missing(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        ml_research.DatasetSpec,
        "path",
        property(lambda self: tmp_path / f"{self.product}.parquet"),
    )
    output, state = tmp_path / "report.json", tmp_path / "state.json"

    report = ml_research.run_cycle(config, output_path=output, state_path=state)

    assert report["ok"] is True
    assert report["summary"] == {
        "attempted": 2,
        "waiting": 2,
        "errors": 0,
        "pre_holdout_passes": 0,
        "protected_holdout_passes": 0,
        "protected_holdout_rejects": 0,
        "forward_paper_candidates": 0,
        "reviewable_candidate_artifacts": 0,
    }
    assert all(item["status"] == "waiting_for_dataset" for item in report["trials"])
    assert json.loads(state.read_text())["cursor"] == 2


def test_cycle_exports_holdout_winner_to_isolated_review_catalog(tmp_path, monkeypatch):
    config = dataclasses.replace(_config(tmp_path), max_trials_per_cycle=1)
    frame = pd.DataFrame(
        {"close": range(1, 101)},
        index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"),
    )

    def fake_evaluate(spec, dataset, observed, research_config):
        return {
            **spec,
            "status": "pre_holdout_pass",
            "reserved_holdout_rows": 20,
            "windows": [],
            "pre_holdout_eligible": True,
        }

    context = {
        "behavior_hash": "sha256:" + "b" * 64,
        "evaluation_key": "sha256:" + "e" * 64,
        "strategy_created": True,
        "evaluation_created": True,
        "snapshot_id": "ml:" + "a" * 64,
        "dataset": {},
        "protocol": {},
        "holdout_window": {},
    }

    def fake_holdout(research_config, candidates):
        for result, _, _, _ in candidates:
            result["holdout_eligible"] = True
            result["forward_paper_candidate"] = {"frozen_model": {}}

    product = SimpleNamespace(name=config.datasets[0].product)
    monkeypatch.setattr(ml_research, "_load_dataset", lambda *args: frame)
    monkeypatch.setattr(ml_research, "evaluate_experiment", fake_evaluate)
    monkeypatch.setattr(ml_research, "_remember_evaluation", lambda *args: context)
    monkeypatch.setattr(ml_research, "_evaluate_protected_cohort", fake_holdout)
    monkeypatch.setattr(
        ml_research,
        "load_autopilot_config",
        lambda path: SimpleNamespace(products=[product]),
    )
    monkeypatch.setattr(
        ml_research,
        "export_reviewable_artifact",
        lambda *args, **kwargs: {"path": "review.json", "artifact_digest": "sha256:test"},
    )

    report = ml_research.run_cycle(
        config,
        output_path=tmp_path / "report.json",
        state_path=tmp_path / "state.json",
        candidate_artifact_dir=tmp_path / "catalog",
    )

    assert report["ok"] is True
    assert report["summary"]["reviewable_candidate_artifacts"] == 1
    assert report["trials"][0]["candidate_artifact_status"] == "reviewable_not_staged"


def test_experiment_never_scores_reserved_holdout(tmp_path, monkeypatch):
    config = dataclasses.replace(
        _config(tmp_path),
        min_train_rows=200,
        validation_rows=100,
        step_rows=100,
        min_windows=2,
        minimum_total_trades=1,
    )
    index = pd.date_range("2024-01-01", periods=600, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "tf_15m_open": range(600),
            "tf_15m_high": range(1, 601),
            "tf_15m_low": range(600),
            "tf_15m_close": range(1, 601),
        },
        index=index,
    )
    seen_validation_ends = []

    class FakeModel:
        def fit(self, train):
            return self

        def generate_signals(self, validation):
            seen_validation_ends.append(validation.index[-1])
            return pd.Series(0, index=validation.index)

    class FakeResult:
        def summary(self):
            return {"trades": 0, "total_return": 0.0, "max_drawdown": 0.0}

    monkeypatch.setattr(ml_research, "_strategy", lambda spec, train: FakeModel())
    monkeypatch.setattr(ml_research, "run_backtest", lambda *args, **kwargs: FakeResult())
    spec = next(item for item in ml_research.experiment_grid(config) if item["regime"] == "all")
    result = ml_research.evaluate_experiment(spec, config.datasets[0], frame, config)

    assert result["reserved_holdout_rows"] == 120
    assert seen_validation_ends
    assert max(seen_validation_ends) < index[480]


def test_config_rejects_unknown_fields(tmp_path):
    payload = json.loads(Path("config/ml_research.json").read_text(encoding="utf-8"))
    payload["unsafe"] = True
    path = tmp_path / "ml.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ml_research.MlResearchConfigError, match="unknown fields"):
        ml_research.load_config(path)


def test_completed_trial_is_written_to_shared_experiment_memory(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = tmp_path / "indicators.parquet"
    source.write_bytes(b"immutable-test-snapshot")
    monkeypatch.setattr(ml_research.DatasetSpec, "path", property(lambda self: source))
    frame = pd.DataFrame(index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"))
    spec = ml_research.experiment_grid(config)[0]
    result = {
        "reserved_holdout_rows": 20,
        "pre_holdout_eligible": False,
        "aggregate": {"windows": 3, "trades": 10},
        "windows": [],
    }

    evidence = ml_research._remember_evaluation(config, spec, config.datasets[0], frame, result)

    assert evidence["behavior_hash"].startswith("sha256:")
    assert evidence["evaluation_key"].startswith("sha256:")
    assert config.memory_path.exists()


def test_protected_holdout_is_durably_claimed_before_scoring(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = tmp_path / "indicators.parquet"
    source.write_bytes(b"protected-test-snapshot")
    monkeypatch.setattr(ml_research.DatasetSpec, "path", property(lambda self: source))
    frame = pd.DataFrame(
        {"close": range(1, 101)},
        index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"),
    )
    spec = ml_research.experiment_grid(config)[0]
    result = {
        **spec,
        "reserved_holdout_rows": 20,
        "pre_holdout_eligible": True,
        "aggregate": {"windows": 3, "trades": 40},
        "windows": [],
    }
    context = ml_research._remember_evaluation(config, spec, config.datasets[0], frame, result)

    def score_after_claim(*args):
        with ExperimentMemory(config.memory_path) as memory:
            assert memory.holdout_claimed(
                context["behavior_hash"], snapshot_id=context["snapshot_id"]
            )
        return {
            "eligible": True,
            "metrics": {"trades": 5, "total_return": 0.01, "max_drawdown": -0.01},
            "train_start": str(frame.index[0]),
            "train_end": str(frame.index[79]),
            "holdout_start": str(frame.index[80]),
            "holdout_end": str(frame.index[-1]),
            "regime_close_feature": "close",
            "frozen_model": {
                "schema": "autopilot.frozen_gradient_boosting/v1",
                "kind": "classifier",
                "feature_names": ["close"],
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
            },
        }

    monkeypatch.setattr(ml_research, "_score_protected_holdout", score_after_claim)
    ml_research._evaluate_protected_cohort(config, [(result, config.datasets[0], frame, context)])

    assert result["holdout_status"] == "protected_holdout_pass"
    assert result["forward_paper_candidate"]["promotion_eligible"] is False


def test_protected_holdout_uses_conservative_multiple_trial_dsr(tmp_path, monkeypatch):
    config = dataclasses.replace(
        _config(tmp_path),
        minimum_holdout_trades=1,
        minimum_dsr=0.6,
    )
    frame = pd.DataFrame(
        {"close": range(1, 101)},
        index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"),
    )
    spec = next(
        item
        for item in ml_research.experiment_grid(config)
        if item["model"] == "lightgbm" and item["regime"] == "all"
    )

    class FakeModel:
        def fit(self, train):
            return self

        def generate_signals(self, holdout):
            return pd.Series(1, index=holdout.index)

    class FakeResult:
        returns = [0.01, -0.005] * 10

        def summary(self):
            return {"trades": 10, "total_return": 0.05, "max_drawdown": -0.01}

    captured = {}

    def fake_dsr(sharpe, **kwargs):
        captured.update(sharpe=sharpe, **kwargs)
        return 0.7

    monkeypatch.setattr(ml_research, "_strategy", lambda *args: FakeModel())
    monkeypatch.setattr(ml_research, "run_backtest", lambda *args, **kwargs: FakeResult())
    monkeypatch.setattr(ml_research.performance_metrics, "sharpe_ratio", lambda returns: 1.0)
    monkeypatch.setattr(ml_research.performance_metrics, "deflated_sharpe_ratio", fake_dsr)

    result = ml_research._score_protected_holdout(spec, config.datasets[0], frame, config)

    assert result["eligible"] is True
    assert captured["n_trials"] == len(ml_research.experiment_grid(config))
    assert captured["sr_std_trials"] >= 0.1
    assert result["metrics"]["trial_sharpe_count"] == 1
    assert result["metrics"]["trial_sharpe_observed_std"] == 0.0
    assert result["metrics"]["trial_sharpe_conservative_floor"] >= 0.1
