import json

import numpy as np
import pandas as pd

from src.autopilot import relative_value_research


def _config(tmp_path):
    return relative_value_research.RelativeValueConfig(
        market_universe_report=tmp_path / "universe.json",
        output=tmp_path / "report.json",
        timeframe="1h",
        maximum_symbols=4,
        lookback_rows=120,
        cross_sectional_lookback_rows=12,
        cross_sectional_top_k=1,
        basis_entry_threshold=0.001,
        basis_funding_intervals=3,
        pairs_entry_z=1.0,
        maximum_pairs=6,
    )


def _history(root, symbol, market, returns):
    path = root / market / symbol
    path.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(returns), freq="h", tz="UTC"),
            "close": 100 * np.exp(np.cumsum(returns)),
        }
    )
    frame.to_parquet(path / f"{symbol}_1h.parquet", index=False)


def test_relative_value_job_builds_all_three_research_families(tmp_path, monkeypatch):
    config = _config(tmp_path)
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    config.market_universe_report.write_text(
        json.dumps(
            {
                "research_symbols": symbols,
                "symbols": [
                    {"symbol": symbol, "metrics": {"funding_rate": 0.001}} for symbol in symbols
                ],
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(9)
    base = rng.normal(0, 0.01, 120)
    for index, symbol in enumerate(symbols):
        futures_returns = base * (1 + index * 0.2)
        if symbol == "XRPUSDT":
            futures_returns = futures_returns.copy()
            futures_returns[-1] += 0.15
        spot_returns = futures_returns.copy()
        _history(tmp_path, symbol, "futures", futures_returns)
        _history(tmp_path, symbol, "spot", spot_returns)
    monkeypatch.setattr(
        relative_value_research,
        "candle_data_dir",
        lambda symbol, market, legacy_fallback=True: tmp_path / market / symbol,
    )

    report = relative_value_research.build_report(config)

    assert report["ok"] is True
    assert report["summary"]["basis"] == 4
    assert report["summary"]["cross_sectional"] == 2
    assert report["summary"]["pairs"] >= 1
    assert report["safety"]["live_allowed"] is False
    assert all(
        forecast["promotion_eligible"] is False
        for forecasts in report["forecasts"].values()
        for forecast in forecasts
    )


def test_relative_value_job_waits_safely_without_universe(tmp_path):
    report = relative_value_research.build_report(_config(tmp_path))

    assert report["ok"] is False
    assert report["status"] == "waiting_for_universe"
    assert report["safety"]["promotion_allowed"] is False
