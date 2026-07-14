import pandas as pd
import pytest

from src import train_model as train_model_module
from src.build_dataset import TARGET_COLUMNS
from src.run_experiments import walk_forward_splits
from src.train_model import get_feature_matrix, target_horizon_bars, time_ordered_split


def test_get_feature_matrix_excludes_all_targets():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "feature_a": [1, 2, 3],
            "feature_b": [4, 5, 6],
            "target_return_next_1_bar": [0.1, 0.2, 0.3],
            "target_return_next_4_bars": [0.4, 0.5, 0.6],
            "target_direction_next_4_bars": [1.0, 0.0, 1.0],
        }
    )

    features, target = get_feature_matrix(data, target_column="target_return_next_1_bar")

    assert features.columns.tolist() == ["feature_a", "feature_b"]
    assert target.tolist() == [0.1, 0.2, 0.3]
    assert not set(TARGET_COLUMNS).intersection(features.columns)


def test_walk_forward_splits_use_recent_complete_windows():
    splits = walk_forward_splits(
        row_count=100,
        train_rows=40,
        test_rows=10,
        n_splits=3,
    )

    assert splits == [
        (30, 70, 70, 80),
        (40, 80, 80, 90),
        (50, 90, 90, 100),
    ]


def test_target_horizon_metadata_matches_forward_labels():
    assert target_horizon_bars("target_return_next_1_bar") == 1
    assert target_horizon_bars("target_return_next_4_bars") == 4
    assert target_horizon_bars("target_direction_next_4_bars") == 4

    with pytest.raises(ValueError, match="No target horizon metadata"):
        target_horizon_bars("unknown_target")


def test_time_ordered_split_purges_forward_label_overlap():
    features = pd.DataFrame({"feature": range(20)})
    target = pd.Series(range(20), name="target_return_next_4_bars")

    x_train, x_test, y_train, y_test = time_ordered_split(
        features,
        target,
        train_fraction=0.7,
        target_horizon_bars=4,
    )

    assert x_train.index.tolist() == list(range(10))
    assert y_train.index.tolist() == list(range(10))
    assert x_test.index.tolist() == list(range(14, 20))
    assert y_test.index.tolist() == list(range(14, 20))
    assert x_train.index[-1] + 4 < x_test.index[0]


def test_time_ordered_split_rejects_dataset_too_small_after_purge():
    features = pd.DataFrame({"feature": range(8)})
    target = pd.Series(range(8))

    with pytest.raises(ValueError, match="after purging"):
        time_ordered_split(
            features,
            target,
            train_fraction=0.5,
            target_horizon_bars=4,
        )


@pytest.mark.parametrize("horizon", [-1, True, None, 1.5])
def test_time_ordered_split_rejects_invalid_target_horizon(horizon):
    features = pd.DataFrame({"feature": range(20)})
    target = pd.Series(range(20))

    with pytest.raises(ValueError, match="non-negative integer"):
        time_ordered_split(
            features,
            target,
            train_fraction=0.7,
            target_horizon_bars=horizon,
        )


def test_train_model_purges_training_labels_before_early_stopping_validation(monkeypatch):
    captured = {}

    class DummyModel:
        best_iteration_ = 1

        def fit(self, x_train, y_train, *, eval_set, callbacks):
            captured["x_train"] = x_train
            captured["y_train"] = y_train
            captured["x_val"], captured["y_val"] = eval_set[0]
            captured["callbacks"] = callbacks

    monkeypatch.setattr(train_model_module.lgb, "LGBMRegressor", lambda **kwargs: DummyModel())
    monkeypatch.setattr(train_model_module.lgb, "early_stopping", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_model_module.lgb, "log_evaluation", lambda *args, **kwargs: None)
    features = pd.DataFrame({"feature": range(100)})
    target = pd.Series(range(100), dtype=float)

    train_model_module.train_model(
        features,
        target,
        target_horizon_bars=4,
    )

    assert captured["x_train"].index[-1] + 4 < captured["x_val"].index[0]
    assert captured["x_train"].index.equals(captured["y_train"].index)
    assert captured["x_val"].index.equals(captured["y_val"].index)


def test_walk_forward_splits_preserve_train_size_and_purge_before_test():
    splits = walk_forward_splits(
        row_count=100,
        train_rows=40,
        test_rows=10,
        n_splits=3,
        target_horizon_bars=4,
    )

    assert splits == [
        (26, 66, 70, 80),
        (36, 76, 80, 90),
        (46, 86, 90, 100),
    ]
    for train_start, train_end, test_start, _ in splits:
        assert train_end - train_start == 40
        assert train_end - 1 + 4 < test_start
