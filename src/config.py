from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INDICATOR_DATA_DIR = PROJECT_ROOT / "data" / "candles" / "BTCUSDT" / "indicators"

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
