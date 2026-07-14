import pandas as pd

from src.load_data import clean_dataframe, parse_timestamp_series


def test_parse_timestamp_series_epoch_seconds_as_utc():
    parsed = parse_timestamp_series(pd.Series([1_714_521_600, 1_714_525_200]))

    assert parsed.iloc[0] == pd.Timestamp("2024-05-01T00:00:00Z")
    assert parsed.iloc[1] == pd.Timestamp("2024-05-01T01:00:00Z")


def test_parse_timestamp_series_iso_strings_as_utc():
    parsed = parse_timestamp_series(pd.Series(["2024-05-01 00:00:00", "2024-05-01T01:00:00Z"]))

    assert parsed.iloc[0] == pd.Timestamp("2024-05-01T00:00:00Z")
    assert parsed.iloc[1] == pd.Timestamp("2024-05-01T01:00:00Z")


def test_clean_dataframe_removes_exact_and_timestamp_duplicates():
    df = pd.DataFrame(
        {
            "time": [1_714_521_600, 1_714_521_600, 1_714_525_200, 1_714_525_200],
            "open": [1.0, 1.0, 2.0, 3.0],
            "close": [1.5, 1.5, 2.5, 3.5],
        }
    )

    cleaned = clean_dataframe(df)

    assert len(cleaned) == 2
    assert cleaned["timestamp"].is_unique
    row = cleaned["timestamp"] == pd.Timestamp("2024-05-01T01:00:00Z")
    assert cleaned.loc[row, "open"].iloc[0] == 3.0
