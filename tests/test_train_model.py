import pandas as pd

from src.build_dataset import TARGET_COLUMNS
from src.run_experiments import walk_forward_splits
from src.train_model import get_feature_matrix


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

    features, target = get_feature_matrix(
        data, target_column="target_return_next_1_bar"
    )

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
