import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_SYMBOL = "BTCUSDT"
MARKETS = ("spot", "futures")


def normalize_market(market: str | None = None) -> str:
    selected = (market or os.environ.get("TRADING_DATA_MARKET", "futures")).strip().lower()
    if selected not in MARKETS:
        raise ValueError("market must be 'spot' or 'futures'")
    return selected


def canonical_candle_data_dir(symbol: str = DEFAULT_SYMBOL, market: str | None = None) -> Path:
    return PROJECT_ROOT / "data" / "candles" / normalize_market(market) / symbol


def legacy_candle_data_dir(symbol: str = DEFAULT_SYMBOL) -> Path:
    return PROJECT_ROOT / "data" / "candles" / symbol


def candle_data_dir(
    symbol: str = DEFAULT_SYMBOL,
    market: str | None = None,
    *,
    legacy_fallback: bool = True,
) -> Path:
    selected_market = normalize_market(market)
    canonical = canonical_candle_data_dir(symbol, selected_market)
    legacy = legacy_candle_data_dir(symbol)
    if legacy_fallback and selected_market == "futures" and legacy.exists() and not canonical.exists():
        return legacy
    return canonical


def indicator_data_dir(
    symbol: str = DEFAULT_SYMBOL,
    market: str | None = None,
    *,
    legacy_fallback: bool = True,
) -> Path:
    return candle_data_dir(symbol, market, legacy_fallback=legacy_fallback) / "indicators"


INDICATOR_DATA_DIR = indicator_data_dir(DEFAULT_SYMBOL, legacy_fallback=True)

TIMEFRAMES = ("1m", "5m", "15m", "30m", "60m", "240m", "1d", "1w")
INDICATOR_TIMEFRAMES = ("15m", "30m", "1h", "4h", "1d", "1w")
BASE_TIMEFRAME = "15m"

DAY_TRADE_HIGHER_TFS = {
    "1m": ("5m", "15m", "30m"),
    "5m": ("15m", "30m", "1h"),
}

WALK_FORWARD_DEFAULTS = {
    "1m": {"train_bars": 1_051_200, "test_bars": 262_800, "step_bars": 262_800},
    "5m": {"train_bars": 210_240, "test_bars": 52_560, "step_bars": 52_560},
    "15m": {"train_bars": 70_080, "test_bars": 17_520, "step_bars": 17_520},
}
