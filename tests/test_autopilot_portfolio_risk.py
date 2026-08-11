import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.autopilot import portfolio_risk
from src.autopilot.portfolio import PORTFOLIO_RISK_MODEL_SCHEMA, PortfolioRiskModel


def _config(tmp_path: Path):
    return dataclasses.replace(
        portfolio_risk.load_config(),
        lookback_rows=600,
        minimum_overlap_rows=500,
        market_universe_report=tmp_path / "universe.json",
        output=tmp_path / "risk.json",
    )


def _write_history(root: Path, symbol: str, returns: np.ndarray, *, datetime_index: bool = False):
    close = 100 * np.exp(np.cumsum(returns))
    path = root / symbol / f"{symbol}_1h.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(close), freq="1h", tz="UTC"),
            "close": close,
        }
    )
    if datetime_index:
        frame = frame.set_index("timestamp")
    frame.to_parquet(path, index=datetime_index)
    return path


def test_build_portfolio_risk_model_computes_correlation_and_btc_beta(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.market_universe_report.write_text(
        json.dumps({"research_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}),
        encoding="utf-8",
    )
    rng = np.random.default_rng(42)
    btc = rng.normal(0, 0.01, 600)
    _write_history(tmp_path, "BTCUSDT", btc)
    _write_history(tmp_path, "ETHUSDT", btc * 1.5 + rng.normal(0, 0.001, 600))
    _write_history(tmp_path, "SOLUSDT", -btc + rng.normal(0, 0.002, 600))
    monkeypatch.setattr(
        portfolio_risk,
        "candle_data_dir",
        lambda symbol, market, legacy_fallback=True: tmp_path / symbol,
    )

    report = portfolio_risk.build_risk_model(config)
    model = PortfolioRiskModel.from_dict(report)

    assert report["schema"] == PORTFOLIO_RISK_MODEL_SCHEMA
    assert model.correlation("BTCUSDT", "ETHUSDT") > 0.9
    assert model.correlation("BTCUSDT", "SOLUSDT") < -0.9
    assert model.beta_by_symbol["ETHUSDT"] > 1
    assert model.beta_by_symbol["SOLUSDT"] < 0


def test_portfolio_risk_model_waits_when_benchmark_history_is_missing(tmp_path):
    report = portfolio_risk.build_risk_model(_config(tmp_path))

    assert report["ok"] is False
    assert report["reason"] == "benchmark_history_unavailable"


def test_portfolio_risk_model_supports_current_datetime_index_candles(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.market_universe_report.write_text(
        json.dumps({"research_symbols": ["BTCUSDT", "ETHUSDT"]}), encoding="utf-8"
    )
    rng = np.random.default_rng(7)
    btc = rng.normal(0, 0.01, 600)
    _write_history(tmp_path, "BTCUSDT", btc, datetime_index=True)
    _write_history(tmp_path, "ETHUSDT", btc * 1.2, datetime_index=True)
    monkeypatch.setattr(
        portfolio_risk,
        "candle_data_dir",
        lambda symbol, market, legacy_fallback=True: tmp_path / symbol,
    )

    report = portfolio_risk.build_risk_model(config)

    assert report["ok"] is True
    assert report["beta_by_symbol"]["ETHUSDT"] > 1
