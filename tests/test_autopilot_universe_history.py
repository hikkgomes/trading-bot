from __future__ import annotations

import json
from datetime import UTC, datetime

from src.autopilot.research_factory import load_factory_config
from src.autopilot.universe_history import run_universe_history


def test_universe_history_uses_screened_symbols_and_isolates_failures(
    monkeypatch,
    tmp_path,
):
    report_path = tmp_path / "market_universe.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "schema": "autopilot.market_universe/v2",
                "generated_at": datetime.now(UTC).isoformat(),
                "snapshot": {"id": "sha256:" + "2" * 64},
                "eligible_research_symbols": [
                    "BTCUSDT",
                    "ETHUSDT",
                    "DOGEUSDT",
                    "SOLUSDT",
                ],
                "symbols": [
                    {"symbol": symbol, "metrics": {"listing_days": 2000}}
                    for symbol in ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT")
                ],
            }
        ),
        encoding="utf-8",
    )
    called: list[str] = []

    def fake_bootstrap(**kwargs):
        symbol = kwargs["symbol"]
        called.append(symbol)
        if symbol == "ETHUSDT":
            return {"ok": False, "symbol": symbol, "error": "temporary failure"}
        return {"ok": True, "symbol": symbol}

    monkeypatch.setattr(
        "src.autopilot.universe_history.run_history_bootstrap",
        fake_bootstrap,
    )

    result = run_universe_history(
        config_path=load_factory_config().path,
        market_universe_report=report_path,
        output_path=tmp_path / "history.json",
    )

    assert called == ["DOGEUSDT", "ETHUSDT"]
    assert result["ok"] is False
    assert result["completed"] == 1
    assert result["deferred"] is False
    assert result["failed_symbols"] == ["ETHUSDT"]


def test_universe_history_resumes_from_the_deferred_symbol(monkeypatch, tmp_path):
    report_path = tmp_path / "market_universe.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "schema": "autopilot.market_universe/v2",
                "generated_at": datetime.now(UTC).isoformat(),
                "snapshot": {"id": "sha256:" + "4" * 64},
                "eligible_research_symbols": ["BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT"],
                "symbols": [
                    {"symbol": symbol, "metrics": {"listing_days": 2000}}
                    for symbol in ("BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT")
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "history.json"
    called: list[str] = []
    eth_attempts = 0

    def fake_bootstrap(**kwargs):
        nonlocal eth_attempts
        symbol = kwargs["symbol"]
        called.append(symbol)
        if symbol == "ETHUSDT":
            eth_attempts += 1
            if eth_attempts == 1:
                return {
                    "ok": True,
                    "deferred": True,
                    "symbol": symbol,
                    "reason": "time_budget_exhausted",
                }
        return {"ok": True, "complete": True, "symbol": symbol}

    monkeypatch.setattr(
        "src.autopilot.universe_history.run_history_bootstrap",
        fake_bootstrap,
    )

    first = run_universe_history(
        config_path=load_factory_config().path,
        market_universe_report=report_path,
        output_path=output,
    )
    second = run_universe_history(
        config_path=load_factory_config().path,
        market_universe_report=report_path,
        output_path=output,
    )

    assert first["ok"] is True
    assert first["deferred"] is True
    assert first["next_symbol"] == "ETHUSDT"
    assert second["ok"] is True
    assert second["complete"] is True
    assert called == ["DOGEUSDT", "ETHUSDT", "ETHUSDT", "SOLUSDT"]
