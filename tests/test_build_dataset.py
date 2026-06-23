import pandas as pd

from src.build_dataset import (
    TARGET_COLUMN,
    TARGET_COLUMNS,
    build_feature_report,
    build_multitimeframe_dataset,
    drop_problem_columns,
)


def test_build_dataset_uses_last_closed_higher_timeframe_candle_and_target():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    frames = {
        "15m": pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "close": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
                "text_signal": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            }
        ),
        "60m": pd.DataFrame(
            {
                "timestamp": [timestamps[0]],
                "close": [200],
                "label": ["x"],
            }
        ),
    }

    dataset = build_multitimeframe_dataset(frames)

    assert len(dataset) == 6
    assert "timestamp" in dataset.columns
    assert "tf_15m_close" in dataset.columns
    assert "tf_1h_close" in dataset.columns
    assert "tf_15m_text_signal" not in dataset.columns
    assert "tf_1h_label" not in dataset.columns
    assert dataset[TARGET_COLUMN].iloc[0].round(6) == 0.4
    assert set(TARGET_COLUMNS).issubset(dataset.columns)
    assert dataset["target_return_next_1_bar"].iloc[0].round(6) == 0.1
    assert dataset["target_direction_next_4_bars"].isin([0.0, 1.0]).all()
    assert pd.isna(dataset["tf_1h_close"].iloc[0])
    assert dataset["tf_1h_close"].iloc[4] == 200.0


def test_feature_report_finds_constant_empty_and_duplicate_columns():
    dataset = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "a": [1, 1, 1, 1],
            "b": [1, None, None, None],
            "c": [1, 2, 3, 4],
            "d": [1, 2, 3, 4],
        }
    )

    report = build_feature_report(dataset, mostly_empty_threshold=0.75)

    assert report["number_of_rows"] == 4
    assert report["number_of_columns"] == 5
    assert report["constant_columns"] == ["a"]
    assert report["mostly_empty_columns"] == ["b"]
    assert report["duplicate_columns_by_exact_equality"] == [["c", "d"]]


def test_drop_problem_columns_keeps_protected_columns_and_first_duplicate():
    dataset = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "target_return_next_1_bar": [0.01, 0.02, 0.03, 0.04],
            "target_return_next_4_bars": [0.1, 0.2, 0.3, 0.4],
            "target_direction_next_4_bars": [1.0, 1.0, 1.0, 1.0],
            "constant": [1, 1, 1, 1],
            "mostly_empty": [1, None, None, None],
            "feature": [1, 2, 3, 4],
            "feature_duplicate": [1, 2, 3, 4],
        }
    )

    pruned = drop_problem_columns(dataset, mostly_empty_threshold=0.75)

    assert "timestamp" in pruned.columns
    assert set(TARGET_COLUMNS).issubset(pruned.columns)
    assert "constant" not in pruned.columns
    assert "mostly_empty" not in pruned.columns
    assert "feature" in pruned.columns
    assert "feature_duplicate" not in pruned.columns
