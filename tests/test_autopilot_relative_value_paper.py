import json
from pathlib import Path

import pandas as pd

from src.autopilot import relative_value_paper


def _write_price(root: Path, market: str, symbol: str, timestamp: str, close: float):
    path = root / market / symbol
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": [pd.Timestamp(timestamp)], "close": [close]}).to_parquet(
        path / f"{symbol}_1h.parquet", index=False
    )


def _research(generated_at: str):
    return {
        "schema": "autopilot.relative_value_research/v1",
        "ok": True,
        "generated_at": generated_at,
        "forecasts": {
            "spot_perp_basis": [
                {
                    "schema": "autopilot.multi_leg_alpha_forecast/v1",
                    "source_id": "basis:BTCUSDT",
                    "family": "spot_perp_basis",
                    "legs": [
                        {"market": "spot", "symbol": "BTCUSDT", "side": "buy", "weight": 0.5},
                        {
                            "market": "futures",
                            "symbol": "BTCUSDT",
                            "side": "sell",
                            "weight": 0.5,
                        },
                    ],
                    "score": 1.0,
                    "expected_return": 0.01,
                    "confidence": 0.8,
                    "horizon_seconds": 3600,
                    "generated_at": generated_at,
                    "requires_borrow": False,
                    "metadata": {"funding_rate_per_8h": 0.0},
                    "research_only": True,
                    "paper_trade_allowed": True,
                    "live_allowed": False,
                    "promotion_eligible": False,
                }
            ],
            "cross_sectional": [],
            "statistical_pairs": [],
        },
    }


def test_relative_value_paper_opens_and_closes_hedged_forward_position(tmp_path, monkeypatch):
    now = pd.Timestamp("2026-08-10T00:00:00Z")
    input_path = tmp_path / "research.json"
    input_path.write_text(json.dumps(_research(now.isoformat())), encoding="utf-8")
    _write_price(tmp_path, "spot", "BTCUSDT", now.isoformat(), 100)
    _write_price(tmp_path, "futures", "BTCUSDT", now.isoformat(), 102)
    monkeypatch.setattr(
        relative_value_paper,
        "candle_data_dir",
        lambda symbol, market, legacy_fallback=True: tmp_path / market / symbol,
    )
    state_path, output_path = tmp_path / "state.json", tmp_path / "output.json"

    first = relative_value_paper.run_cycle(
        input_path=input_path,
        state_path=state_path,
        output_path=output_path,
        now=now.to_pydatetime(),
    )
    assert first["summary"]["open_positions"] == 1

    later = now + pd.Timedelta(hours=2)
    _write_price(tmp_path, "spot", "BTCUSDT", later.isoformat(), 101)
    _write_price(tmp_path, "futures", "BTCUSDT", later.isoformat(), 101)
    second = relative_value_paper.run_cycle(
        input_path=input_path,
        state_path=state_path,
        output_path=output_path,
        now=later.to_pydatetime(),
    )

    assert second["summary"]["open_positions"] == 0
    assert second["summary"]["completed_trades"] == 1
    assert second["safety"]["atomic_orders_enabled"] is False


def test_relative_value_paper_rejects_promotable_forecast(tmp_path, monkeypatch):
    now = pd.Timestamp("2026-08-10T00:00:00Z")
    payload = _research(now.isoformat())
    payload["forecasts"]["spot_perp_basis"][0]["promotion_eligible"] = True
    input_path = tmp_path / "research.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_price(tmp_path, "spot", "BTCUSDT", now.isoformat(), 100)
    _write_price(tmp_path, "futures", "BTCUSDT", now.isoformat(), 102)
    monkeypatch.setattr(
        relative_value_paper,
        "candle_data_dir",
        lambda symbol, market, legacy_fallback=True: tmp_path / market / symbol,
    )

    report = relative_value_paper.run_cycle(
        input_path=input_path,
        state_path=tmp_path / "state.json",
        output_path=tmp_path / "output.json",
        now=now.to_pydatetime(),
    )

    assert report["summary"]["open_positions"] == 0
    assert report["summary"]["waiting"] == 1
