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


class _DynamicSession:
    def get(self, url, *, params=None, timeout=None):
        symbols = ("BTCUSDT", "DOGEUSDT", "ILLIQUSDT")
        if url.endswith("exchangeInfo"):
            return _Response(
                {
                    "symbols": [
                        {
                            "symbol": symbol,
                            "quoteAsset": "USDT",
                            "status": "TRADING",
                            "contractType": "PERPETUAL",
                            "onboardDate": 1_500_000_000_000,
                        }
                        for symbol in symbols
                    ]
                }
            )
        if url.endswith("ticker/24hr"):
            return _Response(
                [
                    {
                        "symbol": symbol,
                        "quoteVolume": ("1000000000" if symbol == "BTCUSDT" else "100000000"),
                        "count": 500000 if symbol != "ILLIQUSDT" else 1,
                        "lastPrice": "1",
                    }
                    for symbol in symbols
                ]
            )
        if url.endswith("ticker/bookTicker"):
            return _Response(
                [
                    {"symbol": symbol, "bidPrice": "0.9999", "askPrice": "1.0001"}
                    for symbol in symbols
                ]
            )
        if url.endswith("premiumIndex"):
            return _Response(
                [
                    {
                        "symbol": symbol,
                        "markPrice": "1",
                        "lastFundingRate": "0.0001",
                    }
                    for symbol in symbols
                ]
            )
        if url.endswith("openInterest"):
            assert (params or {})["symbol"] != "ILLIQUSDT"
            return _Response({"openInterest": "50000000"})
        raise AssertionError(url)


def test_market_universe_discovers_all_contracts_and_writes_snapshot(tmp_path: Path):
    config = tmp_path / "universe.json"
    output = tmp_path / "latest.json"
    snapshots = tmp_path / "snapshots"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "quote_asset": "USDT",
                "discovery": {
                    "mode": "all_trading_usdt_perpetuals",
                    "maximum_research_symbols": 2,
                    "always_include": ["BTCUSDT"],
                    "exclude": [],
                },
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
        output_path=output,
        snapshot_dir=snapshots,
        session=_DynamicSession(),
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )

    assert report["ok"] is True
    assert report["evaluated_symbols"] == 3
    assert report["eligible_research_symbols"] == ["BTCUSDT", "DOGEUSDT"]
    illiquid = next(item for item in report["symbols"] if item["symbol"] == "ILLIQUSDT")
    assert illiquid["metrics"]["open_interest_queried"] is False
    assert report["snapshot"]["id"].startswith("sha256:")
    assert len(list(snapshots.glob("*.json"))) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["snapshot"]["path"]


def test_market_universe_without_explicit_cap_keeps_every_eligible_symbol(tmp_path: Path):
    config = tmp_path / "universe.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "quote_asset": "USDT",
                "discovery": {
                    "mode": "all_trading_usdt_perpetuals",
                    "always_include": ["BTCUSDT"],
                    "exclude": [],
                },
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
        session=_DynamicSession(),
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )

    assert report["maximum_research_symbols"] is None
    assert report["eligible_research_symbols"] == ["BTCUSDT", "DOGEUSDT"]
