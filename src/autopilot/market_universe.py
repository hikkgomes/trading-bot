"""Daily, read-only Binance futures universe quality screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from src.autopilot.io import write_json_atomic
from src.config import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "market_universe.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime" / "market_universe.json"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "runtime" / "market_universe_snapshots"
API_ROOT = "https://fapi.binance.com"


def _validate_legacy_config(payload: dict[str, Any]) -> None:
    research = payload.get("research_symbols")
    watchlist = payload.get("watchlist_symbols")
    if not isinstance(research, list) or not research or not isinstance(watchlist, list):
        raise ValueError("research_symbols and watchlist_symbols must be non-empty lists")
    if not set(research).issubset(set(watchlist)):
        raise ValueError("every research symbol must be on the watchlist")
    if any(not isinstance(item, str) or not item.endswith("USDT") for item in watchlist):
        raise ValueError("watchlist entries must be USDT symbols")


def _validate_discovery_config(discovery: Any) -> None:
    if not isinstance(discovery, dict):
        raise ValueError("discovery must be an object")
    allowed_discovery = {
        "mode",
        "maximum_research_symbols",
        "always_include",
        "exclude",
    }
    unknown_discovery = sorted(set(discovery) - allowed_discovery)
    if unknown_discovery:
        raise ValueError(
            "market universe discovery has unknown fields: " + ", ".join(unknown_discovery)
        )
    if discovery.get("mode") != "all_trading_usdt_perpetuals":
        raise ValueError("discovery.mode must be all_trading_usdt_perpetuals")
    maximum = discovery.get("maximum_research_symbols")
    if maximum is not None and (
        not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1
    ):
        raise ValueError("discovery.maximum_research_symbols must be a positive integer when set")
    for field in ("always_include", "exclude"):
        values = discovery.get(field, [])
        if not isinstance(values, list) or any(
            not isinstance(item, str) or not item or not item.endswith("USDT") for item in values
        ):
            raise ValueError(f"discovery.{field} must be a list of USDT symbols")
    if set(discovery.get("always_include", [])) & set(discovery.get("exclude", [])):
        raise ValueError("discovery always_include and exclude must not overlap")


def _validate_market_config(payload: dict[str, Any]) -> None:
    allowed = {
        "version",
        "market",
        "quote_asset",
        "research_symbols",
        "watchlist_symbols",
        "discovery",
        "criteria",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"market universe config has unknown fields: {', '.join(unknown)}")
    discovery = payload.get("discovery")
    if discovery is None:
        _validate_legacy_config(payload)
    else:
        _validate_discovery_config(discovery)
    if payload.get("market") != "futures" or payload.get("quote_asset") != "USDT":
        raise ValueError("this screen is restricted to USDT futures")


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("market universe config must be a version 1 object")
    _validate_market_config(payload)
    return payload


def _get(session: requests.Session, path: str, **params: str) -> Any:
    response = session.get(f"{API_ROOT}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def run_screen(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path | None = DEFAULT_OUTPUT,
    snapshot_dir: Path | None = None,
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
        discovery = config.get("discovery")
        if discovery is None:
            candidate_symbols = list(config["watchlist_symbols"])
            maximum_research_symbols = len(config["research_symbols"])
            always_include = set(config["research_symbols"])
            excluded: set[str] = set()
            selection_mode = "bounded_watchlist"
        else:
            excluded = {str(item).upper() for item in discovery.get("exclude", [])}
            candidate_symbols = sorted(
                str(item.get("symbol"))
                for item in exchange.get("symbols", [])
                if item.get("quoteAsset") == config["quote_asset"]
                and item.get("status") == "TRADING"
                and item.get("contractType") == "PERPETUAL"
                and isinstance(item.get("symbol"), str)
                and item.get("symbol") not in excluded
            )
            # A strategy selects its own dynamic universe from the full
            # eligible list. A configured cap remains available only for an
            # explicitly bounded operational deployment, never as a platform
            # restriction.
            maximum_research_symbols = discovery.get("maximum_research_symbols")
            always_include = {str(item).upper() for item in discovery.get("always_include", [])}
            selection_mode = str(discovery["mode"])
        rows: list[dict[str, Any]] = []
        for symbol in candidate_symbols:
            contract = by_symbol["exchange"].get(symbol, {})
            ticker = by_symbol["ticker"].get(symbol, {})
            book = by_symbol["book"].get(symbol, {})
            premium = by_symbol["premium"].get(symbol, {})
            bid = float(book.get("bidPrice") or 0)
            ask = float(book.get("askPrice") or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            spread_bps = ((ask - bid) / mid * 10_000) if mid else float("inf")
            mark = float(premium.get("markPrice") or ticker.get("lastPrice") or 0)
            onboard_ms = int(contract.get("onboardDate") or 0)
            listing_days = (
                (generated.timestamp() * 1000 - onboard_ms) / 86_400_000 if onboard_ms else 0
            )
            prechecks = {
                "tradable_perpetual": contract.get("status") == "TRADING"
                and contract.get("contractType") == "PERPETUAL",
                "mature_history": listing_days >= float(criteria["minimum_listing_days"]),
                "liquid_volume": float(ticker.get("quoteVolume") or 0)
                >= float(criteria["minimum_24h_quote_volume_usdt"]),
                "active_market": int(ticker.get("count") or 0)
                >= int(criteria["minimum_24h_trades"]),
                "tight_spread": spread_bps <= float(criteria["maximum_spread_bps"]),
                "normal_funding": abs(float(premium.get("lastFundingRate") or 0))
                <= float(criteria["maximum_absolute_funding_rate"]),
            }
            oi_usdt = 0.0
            open_interest_queried = all(prechecks.values())
            if open_interest_queried:
                open_interest = _get(client, "/fapi/v1/openInterest", symbol=symbol)
                oi_usdt = float(open_interest.get("openInterest") or 0) * mark
            metrics = {
                "listing_days": round(listing_days, 1),
                "quote_volume_24h_usdt": round(float(ticker.get("quoteVolume") or 0), 2),
                "trades_24h": int(ticker.get("count") or 0),
                "spread_bps": round(spread_bps, 4) if spread_bps != float("inf") else None,
                "open_interest_usdt": round(oi_usdt, 2),
                "funding_rate": float(premium.get("lastFundingRate") or 0),
                "open_interest_queried": open_interest_queried,
            }
            checks = {
                **prechecks,
                "sufficient_open_interest": oi_usdt
                >= float(criteria["minimum_open_interest_usdt"]),
            }
            rows.append(
                {
                    "symbol": symbol,
                    "research_enabled": False,
                    "eligible": all(checks.values()),
                    "checks": checks,
                    "metrics": metrics,
                }
            )
        eligible_rows = sorted(
            (row for row in rows if row["eligible"]),
            key=lambda row: (
                row["symbol"] not in always_include,
                -float(row["metrics"]["quote_volume_24h_usdt"]),
                row["symbol"],
            ),
        )
        selected_rows = (
            eligible_rows
            if maximum_research_symbols is None
            else eligible_rows[: int(maximum_research_symbols)]
        )
        research_universe_symbols = [str(row["symbol"]) for row in eligible_rows]
        selected_symbols = [str(row["symbol"]) for row in selected_rows]
        selected = set(selected_symbols)
        for row in rows:
            row["research_enabled"] = row["symbol"] in selected
        rows.sort(
            key=lambda row: (
                not row["research_enabled"],
                not row["eligible"],
                -float(row["metrics"]["quote_volume_24h_usdt"]),
                row["symbol"],
            )
        )
        report = {
            "ok": True,
            "schema": "autopilot.market_universe/v2",
            "generated_at": generated.isoformat(),
            "selection_mode": selection_mode,
            "evaluated_symbols": len(rows),
            "eligible_symbols": [str(row["symbol"]) for row in eligible_rows],
            "research_symbols": selected_symbols,
            "eligible_research_symbols": selected_symbols,
            "research_universe_symbols": research_universe_symbols,
            "collection_symbols": selected_symbols,
            "maximum_research_symbols": maximum_research_symbols,
            "excluded_symbols": sorted(excluded),
            "criteria": criteria,
            "symbols": rows,
            "note": "Eligibility is a research/paper gate only; it never authorizes live trading.",
        }
        snapshot_payload = {key: value for key, value in report.items() if key != "snapshot"}
        snapshot_digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(snapshot_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        report["snapshot"] = {
            "id": snapshot_digest,
            "append_only": snapshot_dir is not None,
        }
        if snapshot_dir is not None:
            stamp = generated.strftime("%Y%m%dT%H%M%SZ")
            snapshot_path = snapshot_dir / (
                f"{stamp}_{snapshot_digest.removeprefix('sha256:')[:12]}.json"
            )
            if snapshot_path.exists():
                existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if existing != report:
                    raise ValueError(f"market universe snapshot collision: {snapshot_path}")
            else:
                write_json_atomic(snapshot_path, report)
            report["snapshot"]["path"] = str(snapshot_path)
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
    parser = argparse.ArgumentParser(description="Screen the Binance USDT futures universe.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    args = parser.parse_args()
    report = run_screen(
        config_path=args.config,
        output_path=args.output,
        snapshot_dir=args.snapshot_dir,
    )
    console_report = {
        key: report.get(key)
        for key in (
            "ok",
            "schema",
            "generated_at",
            "selection_mode",
            "evaluated_symbols",
            "eligible_research_symbols",
            "snapshot",
            "error",
        )
        if report.get(key) is not None
    }
    print(json.dumps(console_report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
