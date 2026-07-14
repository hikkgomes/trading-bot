"""Lightweight feature inventory across every available timeframe.

Goal: make the research agent aware of the *full* feature universe — across
``1m 5m 15m 30m 1h 4h 1d 1w`` — without loading the multi-GB parquets.

How it stays cheap: everything here is read from the parquet **footer
metadata** (schema + per-row-group column statistics). Null fractions and
constant-column detection come from ``null_count`` / ``min==max`` statistics
that Arrow writes into the file footer, so no column data is ever materialised.

Outputs ``outputs/research_exploration/feature_inventory.json`` plus a printed
human summary. Importable by the report generator and the hypothesis tooling.

Run:  python -m research_exploration.feature_inventory
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from src.config import INDICATOR_DATA_DIR

# All timeframes we care about, finest -> coarsest. Indicator parquets are named
# BTCUSDT_{tf}_all_indicators.parquet (note: 1h/4h, not 60m/240m).
ALL_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w")

# Approx bars-per-day, used only to translate horizons into wall-clock for the report.
BARS_PER_DAY = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "30m": 48,
    "1h": 24,
    "4h": 6,
    "1d": 1,
    "1w": 1 / 7,
}

RAW_OHLCV = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
}

# Pure element-wise TA-Lib math operators: little strategy value, flagged low-priority.
MATH_SCALAR = {
    "add",
    "div",
    "mult",
    "sub",
    "acos",
    "asin",
    "atan",
    "ceil",
    "cos",
    "cosh",
    "exp",
    "floor",
    "ln",
    "log10",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
}

# Ordered (feature_family, regex) rules. First match wins, so order matters:
# order-flow before volume, candlesticks before everything, etc.
_FAMILY_RULES: list[tuple[str, re.Pattern]] = [
    (
        "orderflow",
        re.compile(
            r"^(cvd|taker_imbalance|taker_buy_ratio|trades_z|avg_trade_size|volume_z|delta)(_|$)"
        ),
    ),
    ("candlestick", re.compile(r"^cdl")),
    ("cycle_hilbert", re.compile(r"^ht_")),
    ("price_transform", re.compile(r"^(avgprice|medprice|typprice|wclprice)$")),
    # Rolling extrema / Donchian boundaries — directly usable for breakout & sweep families.
    ("range_extrema", re.compile(r"^(max|min|sum|maxindex|minindex|minmax|minmaxindex)(_|$)")),
    ("volatility", re.compile(r"^(atr|natr|trange|true_range|stddev|stdev|var|bbands)(_|$)")),
    (
        "statistic",
        re.compile(
            r"^(beta|correl|linearreg|linearreg_angle|linearreg_intercept|linearreg_slope|tsf)(_|$)"
        ),
    ),
    (
        "trend_ma",
        re.compile(
            r"^(sma|ema|dema|tema|trima|wma|kama|t3|mama|fama|ma|midpoint|midprice|sar|sarext|trendline)(_|$)"
        ),
    ),
    (
        "trend_dmi",
        re.compile(r"^(adx|adxr|dx|plus_di|minus_di|plus_dm|minus_dm|aroon|aroonosc)(_|$)"),
    ),
    (
        "momentum",
        re.compile(
            r"^(rsi|stoch|stochf|stochrsi|macd|macdext|macdfix|mom|roc|rocp|rocr|rocr100|cci|cmo|ppo|apo|willr|ultosc|trix|bop|mfi)(_|$)"
        ),
    ),
    ("volume", re.compile(r"^(ad|adosc|obv)(_|$)")),
]

# Short human descriptions of each family, used in the report.
FAMILY_DESCRIPTIONS = {
    "raw_ohlcv": "Raw candle fields (price, volume, trade counts, taker buy volume).",
    "orderflow": "Order-flow: CVD, taker imbalance, volume/trade z-scores, avg trade size.",
    "trend_ma": "Trend / overlap: moving averages (sma/ema/...), SAR, MAMA, midprice.",
    "trend_dmi": "Directional movement & trend strength: ADX, DI+/-, Aroon.",
    "momentum": "Momentum oscillators: RSI, MACD, stoch, CCI, ROC, Williams %R, MFI...",
    "volatility": "Volatility & bands: ATR/NATR, true range, stddev/var, Bollinger bands.",
    "range_extrema": "Rolling highs/lows & sums (Donchian boundaries; bars-since-extreme).",
    "statistic": "Linear-regression family: slope, angle, forecast (tsf), beta, correl.",
    "volume": "Classic volume indicators: OBV, A/D, A/D oscillator.",
    "candlestick": "TA-Lib candlestick patterns (-100/0/+100 signals).",
    "cycle_hilbert": "Hilbert-transform cycle features: dominant period/phase, trendmode.",
    "price_transform": "Price transforms: average/median/typical/weighted-close price.",
    "math_scalar": "Element-wise math operators (sin/cos/sqrt/...) — low strategy value.",
    "time": "Timestamp.",
    "unclassified": "Columns not matched by any family rule.",
}

# Families a research hypothesis would actually reach for, grouped by role.
TRADING_FAMILIES = {
    "regime_context": ["trend_ma", "trend_dmi", "statistic"],
    "setup": ["momentum", "volatility", "range_extrema", "price_transform"],
    "trigger": ["momentum", "orderflow", "candlestick", "range_extrema"],
    "filter": ["volatility", "orderflow", "cycle_hilbert"],
}


def classify_column(col: str) -> str:
    """Return the feature family for a column name."""
    if col == "timestamp":
        return "time"
    if col in RAW_OHLCV:
        return "raw_ohlcv"
    if col in MATH_SCALAR:
        return "math_scalar"
    for family, rule in _FAMILY_RULES:
        if rule.match(col):
            return family
    return "unclassified"


def feature_root(col: str) -> str:
    """Strip a trailing ``_<period>`` to group e.g. rsi_14/rsi_50 under ``rsi``."""
    return re.sub(r"_\d+$", "", col)


@dataclass
class ColumnStat:
    name: str
    family: str
    null_fraction: float
    is_constant: bool


@dataclass
class TimeframeInventory:
    timeframe: str
    path: str
    exists: bool
    num_rows: int = 0
    num_columns: int = 0
    timestamp_min: str | None = None
    timestamp_max: str | None = None
    family_counts: dict[str, int] = field(default_factory=dict)
    family_roots: dict[str, list[str]] = field(default_factory=dict)
    orderflow_available: bool = False
    high_null_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)


def _column_stats_from_metadata(
    pf: pq.ParquetFile,
) -> tuple[dict[str, float], dict[str, bool], str | None, str | None]:
    """Null fractions + constant flags + timestamp range, all from the footer."""
    md = pf.metadata
    schema = pf.schema_arrow
    n_rows = md.num_rows or 1
    null_counts = {name: 0 for name in schema.names}
    has_stats = {name: False for name in schema.names}
    g_min = {name: None for name in schema.names}
    g_max = {name: None for name in schema.names}
    name_to_idx = {name: i for i, name in enumerate(schema.names)}

    for rg in range(md.num_row_groups):
        row_group = md.row_group(rg)
        for name, ci in name_to_idx.items():
            col = row_group.column(ci)
            stats = col.statistics
            if stats is None:
                continue
            has_stats[name] = True
            if stats.has_null_count:
                null_counts[name] += stats.null_count
            if stats.has_min_max:
                lo, hi = stats.min, stats.max
                g_min[name] = lo if g_min[name] is None else min(g_min[name], lo)
                g_max[name] = hi if g_max[name] is None else max(g_max[name], hi)

    null_fraction = {
        name: (null_counts[name] / n_rows) if has_stats[name] else float("nan")
        for name in schema.names
    }
    is_constant = {
        name: (has_stats[name] and g_min[name] is not None and g_min[name] == g_max[name])
        for name in schema.names
    }

    ts_min = ts_max = None
    if "timestamp" in g_min and g_min["timestamp"] is not None:
        ts_min, ts_max = str(g_min["timestamp"]), str(g_max["timestamp"])
    return null_fraction, is_constant, ts_min, ts_max


def inventory_timeframe(
    timeframe: str, indicator_dir: Path = INDICATOR_DATA_DIR, high_null_threshold: float = 0.5
) -> TimeframeInventory:
    path = indicator_dir / f"BTCUSDT_{timeframe}_all_indicators.parquet"
    if not path.exists():
        return TimeframeInventory(timeframe=timeframe, path=str(path), exists=False)

    pf = pq.ParquetFile(path)
    columns = list(pf.schema_arrow.names)
    null_fraction, is_constant, ts_min, ts_max = _column_stats_from_metadata(pf)

    family_counts: dict[str, int] = {}
    family_roots: dict[str, set] = {}
    for col in columns:
        fam = classify_column(col)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        family_roots.setdefault(fam, set()).add(feature_root(col))

    high_null = sorted(
        c
        for c in columns
        if null_fraction.get(c) == null_fraction.get(c)  # not NaN
        and null_fraction.get(c, 0) >= high_null_threshold
    )
    constants = sorted(c for c in columns if is_constant.get(c))

    return TimeframeInventory(
        timeframe=timeframe,
        path=str(path),
        exists=True,
        num_rows=pf.metadata.num_rows,
        num_columns=len(columns),
        timestamp_min=ts_min,
        timestamp_max=ts_max,
        family_counts=dict(sorted(family_counts.items(), key=lambda kv: -kv[1])),
        family_roots={k: sorted(v) for k, v in sorted(family_roots.items())},
        orderflow_available=family_counts.get("orderflow", 0) > 0,
        high_null_columns=high_null,
        constant_columns=constants,
        columns=columns,
    )


def build_inventory(
    timeframes: tuple[str, ...] = ALL_TIMEFRAMES, indicator_dir: Path = INDICATOR_DATA_DIR
) -> dict[str, TimeframeInventory]:
    return {tf: inventory_timeframe(tf, indicator_dir) for tf in timeframes}


def to_serializable(inv: dict[str, TimeframeInventory], include_columns: bool = False) -> dict:
    out = {}
    for tf, ti in inv.items():
        d = {
            "timeframe": ti.timeframe,
            "exists": ti.exists,
            "num_rows": ti.num_rows,
            "num_columns": ti.num_columns,
            "timestamp_min": ti.timestamp_min,
            "timestamp_max": ti.timestamp_max,
            "orderflow_available": ti.orderflow_available,
            "family_counts": ti.family_counts,
            "family_roots": ti.family_roots,
            "n_high_null_columns": len(ti.high_null_columns),
            "high_null_columns": ti.high_null_columns[:40],
            "n_constant_columns": len(ti.constant_columns),
            "constant_columns": ti.constant_columns[:40],
        }
        if include_columns:
            d["columns"] = ti.columns
        out[tf] = d
    return out


def print_summary(inv: dict[str, TimeframeInventory]) -> None:
    print("\n=== FEATURE INVENTORY (BTCUSDT indicator parquets) ===\n")
    header = f"{'tf':>4} {'rows':>12} {'cols':>5} {'flow':>5}  range"
    print(header)
    print("-" * len(header))
    for tf, ti in inv.items():
        if not ti.exists:
            print(f"{tf:>4}  MISSING  ({ti.path})")
            continue
        rng = f"{(ti.timestamp_min or '?')[:10]} -> {(ti.timestamp_max or '?')[:10]}"
        print(
            f"{tf:>4} {ti.num_rows:>12,} {ti.num_columns:>5} {('yes' if ti.orderflow_available else 'no'):>5}  {rng}"
        )

    # Families present (union across timeframes), with the per-tf count.
    print("\nFeature families (count of columns per timeframe):")
    fams = sorted({f for ti in inv.values() if ti.exists for f in ti.family_counts})
    cols = [tf for tf, ti in inv.items() if ti.exists]
    print(f"  {'family':16} " + " ".join(f"{tf:>5}" for tf in cols))
    for fam in fams:
        row = " ".join(f"{inv[tf].family_counts.get(fam, 0):>5}" for tf in cols)
        print(f"  {fam:16} {row}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory features across all timeframes (metadata only)."
    )
    parser.add_argument("--indicator-dir", type=Path, default=INDICATOR_DATA_DIR)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/research_exploration/feature_inventory.json")
    )
    parser.add_argument(
        "--include-columns",
        action="store_true",
        help="Embed the full column list per timeframe in the JSON (large).",
    )
    parser.add_argument("--high-null-threshold", type=float, default=0.5)
    args = parser.parse_args()

    inv = build_inventory(indicator_dir=args.indicator_dir)
    print_summary(inv)

    payload = to_serializable(inv, include_columns=args.include_columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
