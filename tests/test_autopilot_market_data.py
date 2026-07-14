import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from src.autopilot.market_data import (
    bootstrap_command_for_market,
    build_indicator_feature_status,
    build_market_data_status,
    default_1m_candle_path,
    default_indicator_dir,
    required_indicator_features_by_market,
)


def candle_seed(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    rows = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [10.0 + i for i in range(rows)],
            "quote_asset_volume": [1000.0 + i for i in range(rows)],
            "number_of_trades": [20 + i for i in range(rows)],
            "taker_buy_base_volume": [5.0 + i for i in range(rows)],
            "taker_buy_quote_volume": [500.0 + i for i in range(rows)],
        }
    )


def test_market_data_default_paths_are_market_aware():
    spot_candles = default_1m_candle_path(market="spot")
    spot_indicators = default_indicator_dir(market="spot")

    assert spot_candles.as_posix().endswith("data/candles/spot/BTCUSDT/BTCUSDT_1m.parquet")
    assert spot_indicators.as_posix().endswith("data/candles/spot/BTCUSDT/indicators")


def test_market_data_status_reports_missing_seed_dataset(tmp_path):
    report = build_market_data_status(tmp_path / "missing.parquet", market="spot")

    assert report["ok"] is False
    assert report["exists"] is False
    assert report["reason"] == "missing_seed_dataset"
    assert report["remediation"]["action"] == "bootstrap_market_data"
    assert report["remediation"]["command"] == [
        ".venv/bin/python",
        "-m",
        "src.autopilot.history_bootstrap",
        "--config",
        "config/research_factory.json",
        "--market",
        "spot",
        "--report",
        "runtime/history_bootstrap_spot.json",
    ]


def test_bootstrap_command_for_market_uses_authoritative_factory_config():
    futures_command = bootstrap_command_for_market("futures")
    assert "--timeframes" not in futures_command
    assert futures_command[futures_command.index("--config") + 1] == "config/research_factory.json"
    assert bootstrap_command_for_market("spot") == [
        ".venv/bin/python",
        "-m",
        "src.autopilot.history_bootstrap",
        "--config",
        "config/research_factory.json",
        "--market",
        "spot",
        "--report",
        "runtime/history_bootstrap_spot.json",
    ]


def test_market_data_status_reports_fresh_dataset(tmp_path):
    path = tmp_path / "BTCUSDT_1m.parquet"
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min")
    candle_seed(timestamps).to_parquet(path)

    report = build_market_data_status(
        path,
        now=dt.datetime(2026, 1, 1, 0, 3, tzinfo=dt.UTC),
        max_age_seconds=300,
    )

    assert report["ok"] is True
    assert report["reason"] == "fresh"
    assert report["rows"] == 3
    assert report["last_timestamp"] == "2026-01-01T00:02:00+00:00"


def test_market_data_status_reports_stale_dataset(tmp_path):
    path = tmp_path / "BTCUSDT_1m.parquet"
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=1, freq="1min")
    candle_seed(timestamps).to_parquet(path)

    report = build_market_data_status(
        path,
        now=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
        max_age_seconds=3600,
    )

    assert report["ok"] is False
    assert report["reason"] == "stale"
    assert report["age_seconds"] == 86400.0


def test_market_data_status_reports_future_dataset_timestamp(tmp_path):
    path = tmp_path / "BTCUSDT_1m.parquet"
    timestamps = pd.date_range("2026-01-01T00:01:00Z", periods=1, freq="1min")
    candle_seed(timestamps).to_parquet(path)

    report = build_market_data_status(
        path,
        now=dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.UTC),
        max_age_seconds=300,
    )

    assert report["ok"] is False
    assert report["reason"] == "future_timestamp"
    assert report["last_timestamp"] == "2026-01-01T00:01:00+00:00"
    assert report["age_seconds"] == -60.0


def test_market_data_status_rejects_seed_missing_required_candle_columns(tmp_path):
    path = tmp_path / "BTCUSDT_1m.parquet"
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=1, freq="1min")
    pd.DataFrame({"timestamp": timestamps, "close": [1.0]}).to_parquet(path)

    report = build_market_data_status(path, market="futures")

    assert report["ok"] is False
    assert report["reason"] == "invalid_seed_dataset"
    assert "missing required 1m candle columns" in report["error"]


def test_market_data_status_rejects_non_positive_age_limit(tmp_path):
    with pytest.raises(ValueError, match="max_age_seconds must be positive"):
        build_market_data_status(tmp_path / "missing.parquet", max_age_seconds=0)


def test_indicator_feature_status_reports_ready_timeframes(tmp_path):
    indicator_dir = tmp_path / "indicators"
    indicator_dir.mkdir()
    pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")], "volume_z_20": [1.2]}
    ).to_parquet(indicator_dir / "BTCUSDT_1m_all_indicators.parquet")

    report = build_indicator_feature_status({"1m": ["volume_z_20"]}, indicator_dir=indicator_dir)

    assert report["ok"] is True
    assert report["timeframes"]["1m"]["ok"] is True
    assert report["timeframes"]["1m"]["reason"] == "ready"
    assert report["timeframes"]["1m"]["missing_features"] == []
    assert report["timeframes"]["1m"]["available_flow_features"] == ["volume_z_20"]


def test_indicator_feature_status_reports_missing_columns(tmp_path):
    indicator_dir = tmp_path / "indicators"
    indicator_dir.mkdir()
    pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")], "close": [1.0]}).to_parquet(
        indicator_dir / "BTCUSDT_1m_all_indicators.parquet"
    )

    report = build_indicator_feature_status({"1m": ["volume_z_20"]}, indicator_dir=indicator_dir)

    assert report["ok"] is False
    assert report["timeframes"]["1m"]["ok"] is False
    assert report["timeframes"]["1m"]["reason"] == "missing_required_features"
    assert report["timeframes"]["1m"]["missing_features"] == ["volume_z_20"]


def test_indicator_feature_status_reports_missing_parquet(tmp_path):
    report = build_indicator_feature_status({"1m": ["volume_z_20"]}, indicator_dir=tmp_path)

    assert report["ok"] is False
    assert report["timeframes"]["1m"]["exists"] is False
    assert report["timeframes"]["1m"]["reason"] == "missing_indicator_dataset"


def test_required_indicator_features_follow_enabled_market_data_jobs():
    class Job:
        def __init__(self, command, enabled=True):
            self.command = command
            self.enabled = enabled

    requirements = required_indicator_features_by_market(
        {"futures", "spot"},
        jobs=[
            Job(
                [
                    "python",
                    "-m",
                    "src.update_candles",
                    "--market",
                    "spot",
                    "--timeframes",
                    "1h",
                    "4h",
                    "1d",
                ]
            ),
            Job(
                [
                    "python",
                    "-m",
                    "src.update_candles",
                    "--market",
                    "futures",
                    "--timeframes",
                    "1m",
                    "5m",
                ]
            ),
            Job(
                ["python", "-m", "src.update_candles", "--market", "spot", "--timeframes", "1m"],
                enabled=False,
            ),
        ],
    )

    assert requirements["spot"] == {
        "1h": ["volume_z_20"],
        "4h": ["volume_z_20"],
        "1d": ["volume_z_20"],
    }
    assert requirements["futures"] == {
        "1m": ["volume_z_20"],
        "5m": ["volume_z_20"],
    }


def test_required_indicator_features_follow_implicit_history_factory_timeframes():
    class Job:
        enabled = True
        working_dir = Path.cwd()
        command = [
            "python",
            "-m",
            "src.autopilot.history_bootstrap",
            "--config",
            "config/research_factory.json",
            "--market",
            "spot",
        ]

    requirements = required_indicator_features_by_market({"spot"}, jobs=[Job()])

    assert requirements["spot"] == {
        "1m": ["volume_z_20"],
        "1h": ["volume_z_20"],
        "4h": ["volume_z_20"],
        "1d": ["volume_z_20"],
        "1w": ["volume_z_20"],
    }


def test_required_indicator_features_follow_inline_market_data_job_flags():
    class Job:
        def __init__(self, command, enabled=True):
            self.command = command
            self.enabled = enabled

    requirements = required_indicator_features_by_market(
        {"futures", "spot"},
        jobs=[
            Job(["python", "-m", "src.update_candles", "--market=spot", "--timeframes=1h"]),
            Job(
                [
                    "python",
                    "-m",
                    "src.update_candles",
                    "--market=futures",
                    "--timeframes=1m",
                    "--timeframes=5m",
                ]
            ),
        ],
    )

    assert requirements["spot"] == {"1h": ["volume_z_20"]}
    assert requirements["futures"] == {
        "1m": ["volume_z_20"],
        "5m": ["volume_z_20"],
    }
