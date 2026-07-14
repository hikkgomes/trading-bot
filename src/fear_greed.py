"""Crypto Fear & Greed Index — external sentiment data for regime/contrarian use.

The blueprint's core mantra ("extreme fear precedes opportunity, extreme greed
precedes risk") quantified. Source: alternative.me's free Fear & Greed API
(daily values 0-100, history back to 2018-02). Uses stdlib ``urllib`` so it adds
no dependency.

Typical use:

    from src.fear_greed import fetch_fear_greed, add_fear_greed_column
    fng = fetch_fear_greed()                  # cached to data/processed/fear_greed.parquet
    df = add_fear_greed_column(df, fng)       # merges a daily `fear_greed` column by date

Then backtest the ``fear_greed_contrarian`` strategy on ``df``.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from src.config import PROCESSED_DATA_DIR

LOGGER = logging.getLogger(__name__)

_API_URL = "https://api.alternative.me/fng/?limit=0&format=json"
_API_HOST = "api.alternative.me"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_CACHE = Path(PROCESSED_DATA_DIR) / "fear_greed.parquet"


def fetch_fear_greed(use_cache: bool = True, timeout: float = 20.0) -> pd.DataFrame:
    """Return a DataFrame indexed by UTC date with `fear_greed` (0-100) + label.

    Caches to ``data/processed/fear_greed.parquet``. Pass ``use_cache=False`` to
    force a refresh.
    """
    if use_cache and _CACHE.exists():
        return pd.read_parquet(_CACHE)

    LOGGER.info("Fetching Fear & Greed Index from %s", _API_URL)
    request = urllib.request.Request(
        _API_URL,
        headers={"Accept": "application/json", "User-Agent": "trading-bot-research/1"},
        method="GET",
    )
    # The URL is fixed HTTPS and the post-redirect host is revalidated.
    with urllib.request.urlopen(  # nosec B310
        request,
        timeout=timeout,
    ) as resp:
        final_url = urlsplit(str(resp.geturl()))
        if final_url.scheme != "https" or final_url.hostname != _API_HOST:
            raise RuntimeError("Fear & Greed API redirected outside its approved HTTPS host.")
        content_length = resp.headers.get("Content-Length")
        if content_length is not None and int(content_length) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Fear & Greed API response exceeds the size limit.")
        body = resp.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Fear & Greed API response exceeds the size limit.")
        payload = json.loads(body.decode("utf-8"))

    rows = payload.get("data", [])
    if not rows:
        raise RuntimeError("Fear & Greed API returned no data.")
    df = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [int(r["timestamp"]) for r in rows], unit="s", utc=True
                ).normalize(),
                "fear_greed": [int(r["value"]) for r in rows],
                "fear_greed_label": [r.get("value_classification", "") for r in rows],
            }
        )
        .sort_values("date")
        .set_index("date")
    )
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE)
    LOGGER.info(
        "Fetched %d daily Fear & Greed values (%s -> %s)",
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def add_fear_greed_column(df: pd.DataFrame, fng: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge a daily ``fear_greed`` column onto ``df`` by calendar date (as-of).

    ``df`` must have a DatetimeIndex (any intraday resolution). Each row gets the
    Fear & Greed value for its UTC date, forward-filled. If ``fng`` is None it is
    fetched (cached).
    """
    if fng is None:
        fng = fetch_fear_greed()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("add_fear_greed_column requires a DatetimeIndex on df.")

    idx = df.index
    dates = (
        idx.tz_localize("UTC").normalize() if idx.tz is None else idx.tz_convert("UTC").normalize()
    )
    daily = fng["fear_greed"].reindex(fng.index.union(dates.unique())).ffill()
    out = df.copy()
    out["fear_greed"] = daily.reindex(dates).to_numpy()
    return out
