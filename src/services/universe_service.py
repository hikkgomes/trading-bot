"""Point-in-time instrument eligibility service."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable
from typing import Any

import requests

from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.domain.instruments import Instrument, MarketType
from src.services.scheduler import DatabaseJobQueue


class DatabaseUniverseService:
    """Expose canonical universe snapshots to market and strategy workers."""

    def __init__(
        self,
        *,
        store: SqlUniverseStore,
        queue: DatabaseJobQueue | None = None,
        worker_id: str | None = None,
        binance: BinanceUniverseClient | None = None,
    ) -> None:
        self.store = store
        self.queue = queue
        self.worker_id = worker_id
        self.binance = binance or BinanceUniverseClient()

    def eligible_symbols(self, *, universe_id: str, at: str | None = None) -> tuple[str, ...]:
        observed_at = at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        return tuple(
            member.instrument.exchange_symbol
            for member in self.store.members_at(
                universe_id=universe_id, observed_at=observed_at, eligible_only=True
            )
        )

    def eligible_symbols_for_capture(self, *, at: str | None = None) -> tuple[str, ...]:
        observed_at = at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        return self.store.eligible_exchange_symbols(observed_at=observed_at)

    def record_snapshot(
        self,
        *,
        universe_id: str,
        observed_at: str,
        observations: Iterable[InstrumentObservation],
        policy: UniverseEligibilityPolicy,
    ) -> str:
        return self.store.record_snapshot(
            universe_id=universe_id,
            observed_at=observed_at,
            observations=observations,
            policy=policy,
        )

    def run_once(self, *, now: str) -> dict[str, Any]:
        if self.queue is None or self.worker_id is None:
            return {"reason_code": "universe_service_read_only"}
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=60,
            names=("universe_refresh",),
        )
        if claimed is None:
            return {"reason_code": "universe_queue_empty"}
        try:
            payload = claimed.payload
            policy_values = payload.get("policy", {})
            if not isinstance(policy_values, dict):
                raise ValueError("universe policy must be an object")
            policy = UniverseEligibilityPolicy(**policy_values)
            observations = self.binance.observations(
                observed_at=now,
                maximum_symbols=int(payload.get("maximum_symbols", 100)),
            )
            snapshot_id = self.record_snapshot(
                universe_id=str(payload["universe_id"]),
                observed_at=now,
                observations=observations,
                policy=policy,
            )
        except Exception as exc:
            retry_at = (dt.datetime.fromisoformat(now) + dt.timedelta(minutes=5)).isoformat()
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=retry_at,
            )
            return {"reason_code": "universe_refresh_failed", "job_id": claimed.job_id}
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "universe_refresh_completed",
            "job_id": claimed.job_id,
            "snapshot_id": snapshot_id,
            "observations": len(observations),
        }


class BinanceUniverseClient:
    """Build point-in-time futures observations from Binance public REST data."""

    def __init__(self, *, base_url: str = "https://fapi.binance.com", session=None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _get(self, path: str, **params: object) -> Any:
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    def observations(
        self, *, observed_at: str, maximum_symbols: int
    ) -> tuple[InstrumentObservation, ...]:
        if maximum_symbols < 1:
            raise ValueError("maximum_symbols must be positive")
        now_ms = int(dt.datetime.fromisoformat(observed_at).timestamp() * 1000)
        exchange = self._get("/fapi/v1/exchangeInfo")
        tickers = {item["symbol"]: item for item in self._get("/fapi/v1/ticker/24hr")}
        books = {item["symbol"]: item for item in self._get("/fapi/v1/ticker/bookTicker")}
        premiums = {
            item["symbol"]: item
            for item in self._get("/fapi/v1/premiumIndex")
            if isinstance(item, dict) and "symbol" in item
        }
        symbols = sorted(
            (
                item
                for item in exchange["symbols"]
                if item.get("contractType") == "PERPETUAL" and item.get("quoteAsset") == "USDT"
            ),
            key=lambda item: float(tickers.get(item["symbol"], {}).get("quoteVolume", 0)),
            reverse=True,
        )[:maximum_symbols]
        result: list[InstrumentObservation] = []
        for raw in symbols:
            symbol = str(raw["symbol"])
            ticker = tickers.get(symbol, {})
            book = books.get(symbol, {})
            premium = premiums.get(symbol, {})
            filters = {item["filterType"]: item for item in raw.get("filters", [])}
            lot = filters.get("LOT_SIZE", {})
            notional = filters.get("MIN_NOTIONAL", {})
            bid = float(book.get("bidPrice") or 0)
            ask = float(book.get("askPrice") or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > bid else 0
            spread = ((ask - bid) / mid * 10_000) if mid else math.inf
            depth = self._get("/fapi/v1/depth", symbol=symbol, limit=20)
            depth_notional = sum(float(price) * float(quantity) for price, quantity in depth["bids"])
            depth_notional += sum(float(price) * float(quantity) for price, quantity in depth["asks"])
            open_interest = float(
                self._get("/fapi/v1/openInterest", symbol=symbol)["openInterest"]
            ) * max(mid, float(premium.get("markPrice") or 0))
            klines = self._get("/fapi/v1/klines", symbol=symbol, interval="1h", limit=168)
            closes = [float(item[4]) for item in klines]
            returns = [
                math.log(right / left)
                for left, right in zip(closes, closes[1:], strict=False)
                if left > 0
            ]
            mean = sum(returns) / len(returns) if returns else 0.0
            variance = (
                sum((value - mean) ** 2 for value in returns) / max(1, len(returns) - 1)
            )
            instrument = Instrument(
                venue="binance",
                market_type=MarketType.FUTURES,
                base_asset=str(raw["baseAsset"]),
                quote_asset=str(raw["quoteAsset"]),
                settlement_asset=str(raw.get("marginAsset") or "USDT"),
                exchange_symbol=symbol,
                price_precision=int(raw["pricePrecision"]),
                quantity_precision=int(raw["quantityPrecision"]),
                minimum_quantity=float(lot.get("minQty") or 0),
                minimum_notional=float(notional.get("notional") or 0),
                status="trading" if raw.get("status") == "TRADING" else str(raw.get("status", "unknown")),
            )
            result.append(
                InstrumentObservation(
                    instrument=instrument,
                    listing_age_days=max(0.0, (now_ms - int(raw.get("onboardDate", now_ms))) / 86_400_000),
                    quote_volume=float(ticker.get("quoteVolume") or 0),
                    trade_count=int(ticker.get("count") or 0),
                    spread_bps=spread,
                    open_interest=open_interest,
                    funding_rate=float(premium.get("lastFundingRate") or 0),
                    realised_volatility=math.sqrt(variance * 24 * 365),
                    depth_notional=depth_notional,
                    data_completeness=min(1.0, len(klines) / 168),
                )
            )
        if not result:
            raise RuntimeError("Binance returned no eligible futures observations")
        return tuple(result)
