from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.autopilot.market_universe import run_screen


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def get(self, url, *, params=None, timeout=None):
        symbol = (params or {}).get("symbol", "BTCUSDT")
        if url.endswith("exchangeInfo"):
            return _Response(
                {
                    "symbols": [
                        {
                            "symbol": "BTCUSDT",
                            "status": "TRADING",
                            "contractType": "PERPETUAL",
                            "onboardDate": 1_500_000_000_000,
                        }
                    ]
                }
            )
        if url.endswith("ticker/24hr"):
            return _Response(
                [
                    {
                        "symbol": "BTCUSDT",
                        "quoteVolume": "1000000000",
                        "count": 500000,
                        "lastPrice": "50000",
                    }
                ]
            )
        if url.endswith("ticker/bookTicker"):
            return _Response([{"symbol": "BTCUSDT", "bidPrice": "49999", "askPrice": "50001"}])
        if url.endswith("premiumIndex"):
            return _Response(
                [{"symbol": "BTCUSDT", "markPrice": "50000", "lastFundingRate": "0.0001"}]
            )
        if url.endswith("openInterest"):
            assert symbol == "BTCUSDT"
            return _Response({"openInterest": "10000"})
        raise AssertionError(url)


def test_market_universe_screens_bounded_watchlist(tmp_path: Path):
    config = tmp_path / "universe.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "quote_asset": "USDT",
                "research_symbols": ["BTCUSDT"],
                "watchlist_symbols": ["BTCUSDT"],
                "criteria": {
                    "minimum_listing_days": 1000,
                    "minimum_24h_quote_volume_usdt": 50000000,
                    "minimum_24h_trades": 50000,
                    "maximum_spread_bps": 5.0,
                    "minimum_open_interest_usdt": 25000000,
                    "maximum_absolute_funding_rate": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )
    report = run_screen(
        config_path=config,
        output_path=None,
        session=_Session(),
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )
    assert report["ok"] is True
    assert report["eligible_research_symbols"] == ["BTCUSDT"]
    assert report["symbols"][0]["checks"]["tight_spread"] is True
    assert report["note"].endswith("never authorizes live trading.")
