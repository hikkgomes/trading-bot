"""Daily, read-only Binance futures universe quality screen."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from src.autopilot.io import write_json_atomic
from src.config import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "market_universe.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime" / "market_universe.json"
API_ROOT = "https://fapi.binance.com"


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("market universe config must be a version 1 object")
    allowed = {
        "version",
        "market",
        "quote_asset",
        "research_symbols",
        "watchlist_symbols",
        "criteria",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"market universe config has unknown fields: {', '.join(unknown)}")
    research = payload.get("research_symbols")
    watchlist = payload.get("watchlist_symbols")
    if not isinstance(research, list) or not research or not isinstance(watchlist, list):
        raise ValueError("research_symbols and watchlist_symbols must be non-empty lists")
    if not set(research).issubset(set(watchlist)):
        raise ValueError("every research symbol must be on the watchlist")
    if any(not isinstance(item, str) or not item.endswith("USDT") for item in watchlist):
        raise ValueError("watchlist entries must be USDT symbols")
    if payload.get("market") != "futures" or payload.get("quote_asset") != "USDT":
        raise ValueError("this screen is restricted to USDT futures")
    return payload


def _get(session: requests.Session, path: str, **params: str) -> Any:
    response = session.get(f"{API_ROOT}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def run_screen(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path | None = DEFAULT_OUTPUT,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    generated = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    try:
        config = _load_config(config_path)
        client = session or requests.Session()
        exchange = _get(client, "/fapi/v1/exchangeInfo")
        tickers = _get(client, "/fapi/v1/ticker/24hr")
        books = _get(client, "/fapi/v1/ticker/bookTicker")
        premiums = _get(client, "/fapi/v1/premiumIndex")
        by_symbol = {
            "exchange": {item["symbol"]: item for item in exchange.get("symbols", [])},
            "ticker": {item["symbol"]: item for item in tickers},
            "book": {item["symbol"]: item for item in books},
            "premium": {item["symbol"]: item for item in premiums},
        }
        criteria = config["criteria"]
        rows: list[dict[str, Any]] = []
        for symbol in config["watchlist_symbols"]:
            contract = by_symbol["exchange"].get(symbol, {})
            ticker = by_symbol["ticker"].get(symbol, {})
            book = by_symbol["book"].get(symbol, {})
            premium = by_symbol["premium"].get(symbol, {})
            bid = float(book.get("bidPrice") or 0)
            ask = float(book.get("askPrice") or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            spread_bps = ((ask - bid) / mid * 10_000) if mid else float("inf")
            mark = float(premium.get("markPrice") or ticker.get("lastPrice") or 0)
            open_interest = _get(client, "/fapi/v1/openInterest", symbol=symbol)
            oi_usdt = float(open_interest.get("openInterest") or 0) * mark
            onboard_ms = int(contract.get("onboardDate") or 0)
            listing_days = (
                (generated.timestamp() * 1000 - onboard_ms) / 86_400_000 if onboard_ms else 0
            )
            metrics = {
                "listing_days": round(listing_days, 1),
                "quote_volume_24h_usdt": round(float(ticker.get("quoteVolume") or 0), 2),
                "trades_24h": int(ticker.get("count") or 0),
                "spread_bps": round(spread_bps, 4) if spread_bps != float("inf") else None,
                "open_interest_usdt": round(oi_usdt, 2),
                "funding_rate": float(premium.get("lastFundingRate") or 0),
            }
            checks = {
                "tradable_perpetual": contract.get("status") == "TRADING"
                and contract.get("contractType") == "PERPETUAL",
                "mature_history": listing_days >= float(criteria["minimum_listing_days"]),
                "liquid_volume": metrics["quote_volume_24h_usdt"]
                >= float(criteria["minimum_24h_quote_volume_usdt"]),
                "active_market": metrics["trades_24h"] >= int(criteria["minimum_24h_trades"]),
                "tight_spread": spread_bps <= float(criteria["maximum_spread_bps"]),
                "sufficient_open_interest": oi_usdt
                >= float(criteria["minimum_open_interest_usdt"]),
                "normal_funding": abs(metrics["funding_rate"])
                <= float(criteria["maximum_absolute_funding_rate"]),
            }
            rows.append(
                {
                    "symbol": symbol,
                    "research_enabled": symbol in config["research_symbols"],
                    "eligible": all(checks.values()),
                    "checks": checks,
                    "metrics": metrics,
                }
            )
        report = {
            "ok": True,
            "schema": "autopilot.market_universe/v1",
            "generated_at": generated.isoformat(),
            "research_symbols": list(config["research_symbols"]),
            "eligible_research_symbols": [
                row["symbol"] for row in rows if row["research_enabled"] and row["eligible"]
            ],
            "criteria": criteria,
            "symbols": rows,
            "note": "Eligibility is a research/paper gate only; it never authorizes live trading.",
        }
    except Exception as exc:
        report = {
            "ok": False,
            "schema": "autopilot.market_universe/v1",
            "generated_at": generated.isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen a bounded Binance futures watchlist.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_screen(config_path=args.config, output_path=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
