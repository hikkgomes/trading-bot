"""Point-in-time instrument eligibility service."""

from __future__ import annotations

import datetime as dt
import decimal
import math
import threading
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

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
                market_type=str(payload.get("market_type") or "futures"),
            )
            if str(payload.get("product_id") or "") == "btc_accumulation":
                observations = tuple(
                    item for item in observations if item.instrument.exchange_symbol == "BTCUSDT"
                )
                if not observations:
                    raise RuntimeError("BTC accumulation universe has no BTCUSDT observation")
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
            "snapshot_status": dict(self.binance.last_status),
        }


class BinanceUniverseClient:
    """Build point-in-time spot or futures observations from Binance REST data."""

    WEIGHTS = {
        "/fapi/v1/exchangeInfo": 1,
        "/fapi/v1/ticker/24hr": 40,
        "/fapi/v1/ticker/bookTicker": 5,
        "/fapi/v1/premiumIndex": 10,
        "/fapi/v1/depth": 5,
        "/fapi/v1/openInterest": 1,
        "/fapi/v1/klines": 2,
        "/api/v3/exchangeInfo": 20,
        "/api/v3/ticker/24hr": 40,
        "/api/v3/ticker/bookTicker": 5,
        "/api/v3/depth": 5,
        "/api/v3/klines": 2,
    }

    def __init__(
        self,
        *,
        base_url: str = "https://fapi.binance.com",
        session=None,
        request_weight_budget: int = 1_000,
        maximum_retries: int = 3,
        maximum_concurrency: int = 4,
    ) -> None:
        if request_weight_budget < 1 or maximum_retries < 1 or maximum_concurrency < 1:
            raise ValueError("universe client budgets must be positive")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.request_weight_budget = request_weight_budget
        self.maximum_retries = maximum_retries
        self.maximum_concurrency = maximum_concurrency
        self._weight_used = 0
        self._weight_lock = threading.Lock()
        self.last_status: dict[str, Any] = {}

    def _get(self, path: str, **params: object) -> Any:
        weight = self.WEIGHTS.get(path, 1)
        with self._weight_lock:
            if self._weight_used + weight > self.request_weight_budget:
                raise RuntimeError("Binance request-weight budget is exhausted")
            self._weight_used += weight
        for attempt in range(self.maximum_retries):
            try:
                base_url = self.base_url
                if path.startswith("/api/") and base_url == "https://fapi.binance.com":
                    base_url = "https://api.binance.com"
                response = self.session.get(f"{base_url}{path}", params=params, timeout=20)
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt + 1 >= self.maximum_retries:
                    raise
                time.sleep(min(2**attempt, 4))
        raise RuntimeError("Binance request failed")

    def observations(
        self, *, observed_at: str, maximum_symbols: int, market_type: str = "futures"
    ) -> tuple[InstrumentObservation, ...]:
        if maximum_symbols < 1:
            raise ValueError("maximum_symbols must be positive")
        if market_type not in {"spot", "futures"}:
            raise ValueError("market_type must be spot or futures")
        self._weight_used = 0
        now_ms = int(dt.datetime.fromisoformat(observed_at).timestamp() * 1000)
        prefix = "/api/v3" if market_type == "spot" else "/fapi/v1"
        exchange = self._get(f"{prefix}/exchangeInfo")
        tickers = {item["symbol"]: item for item in self._get(f"{prefix}/ticker/24hr")}
        books = {item["symbol"]: item for item in self._get(f"{prefix}/ticker/bookTicker")}
        premiums = {
            item["symbol"]: item
            for item in (self._get("/fapi/v1/premiumIndex") if market_type == "futures" else ())
            if isinstance(item, dict) and "symbol" in item
        }
        symbols = sorted(
            (
                item
                for item in exchange["symbols"]
                if (
                    item.get("contractType") == "PERPETUAL"
                    if market_type == "futures"
                    else item.get("status") == "TRADING"
                    and item.get("isSpotTradingAllowed", True) is True
                )
                and item.get("quoteAsset") == "USDT"
            ),
            key=lambda item: float(tickers.get(item["symbol"], {}).get("quoteVolume", 0)),
            reverse=True,
        )[:maximum_symbols]
        if not symbols:
            raise RuntimeError(f"Binance returned no eligible {market_type} symbols")
        result: list[InstrumentObservation] = []
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=min(self.maximum_concurrency, len(symbols))) as pool:
            futures = {
                pool.submit(
                    self._observation,
                    raw,
                    tickers.get(str(raw["symbol"]), {}),
                    books.get(str(raw["symbol"]), {}),
                    premiums.get(str(raw["symbol"]), {}),
                    now_ms,
                    market_type,
                ): str(raw["symbol"])
                for raw in symbols
            }
            for future in as_completed(futures):
                try:
                    result.append(future.result())
                except (requests.RequestException, RuntimeError, KeyError, ValueError, TypeError):
                    failures.append(futures[future])
        if not result:
            raise RuntimeError(f"Binance returned no {market_type} observations")
        self.last_status = {
            "complete": not failures,
            "requested_symbols": len(symbols),
            "observed_symbols": len(result),
            "failed_symbols": sorted(failures),
            "request_weight_used": self._weight_used,
            "request_weight_budget": self.request_weight_budget,
            "maximum_concurrency": self.maximum_concurrency,
        }
        return tuple(sorted(result, key=lambda item: item.instrument.exchange_symbol))

    def _observation(
        self,
        raw: Mapping[str, Any],
        ticker: Mapping[str, Any],
        book: Mapping[str, Any],
        premium: Mapping[str, Any],
        now_ms: int,
        market_type: str,
    ) -> InstrumentObservation:
        symbol = str(raw["symbol"])
        filters = {item["filterType"]: item for item in raw.get("filters", [])}
        price_filter = filters.get("PRICE_FILTER", {})
        lot = filters.get("LOT_SIZE", {})
        notional = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
        bid = float(book.get("bidPrice") or 0)
        ask = float(book.get("askPrice") or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > bid else 0
        spread = ((ask - bid) / mid * 10_000) if mid else math.inf
        prefix = "/api/v3" if market_type == "spot" else "/fapi/v1"
        depth = self._get(f"{prefix}/depth", symbol=symbol, limit=20)
        open_interest_response = (
            self._get("/fapi/v1/openInterest", symbol=symbol) if market_type == "futures" else {}
        )
        klines = self._get(f"{prefix}/klines", symbol=symbol, interval="1h", limit=168)
        onboard_ms = raw.get("onboardDate")
        if onboard_ms is None and market_type == "spot":
            first_klines = self._get(
                f"{prefix}/klines", symbol=symbol, interval="1d", startTime=0, limit=1
            )
            if not isinstance(first_klines, list) or not first_klines:
                raise RuntimeError(f"Binance returned no listing history for {symbol}")
            onboard_ms = first_klines[0][0]
        depth_notional = sum(float(price) * float(quantity) for price, quantity in depth["bids"])
        depth_notional += sum(float(price) * float(quantity) for price, quantity in depth["asks"])
        open_interest = (
            float(open_interest_response["openInterest"])
            * max(mid, float(premium.get("markPrice") or 0))
            if market_type == "futures"
            else 0.0
        )
        closes = [float(item[4]) for item in klines]
        returns = [
            math.log(right / left)
            for left, right in zip(closes, closes[1:], strict=False)
            if left > 0
        ]
        mean = sum(returns) / len(returns) if returns else 0.0
        variance = sum((value - mean) ** 2 for value in returns) / max(1, len(returns) - 1)
        instrument = Instrument(
            venue="binance",
            market_type=MarketType.FUTURES if market_type == "futures" else MarketType.SPOT,
            base_asset=str(raw["baseAsset"]),
            quote_asset=str(raw["quoteAsset"]),
            settlement_asset=(
                str(raw.get("marginAsset") or "USDT") if market_type == "futures" else None
            ),
            exchange_symbol=symbol,
            price_precision=(
                int(raw["pricePrecision"])
                if raw.get("pricePrecision") is not None
                else _precision(price_filter.get("tickSize"))
            ),
            quantity_precision=(
                int(raw["quantityPrecision"])
                if raw.get("quantityPrecision") is not None
                else _precision(lot.get("stepSize"))
            ),
            minimum_quantity=float(lot.get("minQty") or 0),
            minimum_notional=float(notional.get("notional") or notional.get("minNotional") or 0),
            status="trading"
            if raw.get("status") == "TRADING"
            else str(raw.get("status", "unknown")),
        )
        return InstrumentObservation(
            instrument=instrument,
            listing_age_days=max(0.0, (now_ms - int(onboard_ms or now_ms)) / 86_400_000),
            quote_volume=float(ticker.get("quoteVolume") or 0),
            trade_count=int(ticker.get("count") or 0),
            spread_bps=spread,
            open_interest=open_interest,
            funding_rate=float(premium.get("lastFundingRate") or 0)
            if market_type == "futures"
            else 0.0,
            realised_volatility=math.sqrt(variance * 24 * 365),
            depth_notional=depth_notional,
            data_completeness=min(1.0, len(klines) / 168),
        )


def _precision(value: object) -> int:
    if value is None:
        raise ValueError("Binance instrument precision is missing")
    increment = decimal.Decimal(str(value))
    if not increment.is_finite() or increment <= 0:
        raise ValueError("Binance instrument precision is invalid")
    exponent = cast(int, increment.normalize().as_tuple().exponent)
    return max(0, -exponent)
