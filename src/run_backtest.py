"""CLI to backtest any registered strategy with the unified engine.

Examples
--------
List everything available:
    python -m src.run_backtest --list

Backtest a rule strategy on the aligned dataset:
    python -m src.run_backtest --strategy sma_cross \
        --input data/processed/train_15m_indicators.parquet --param fast=10 --param slow=40

Smoke-test on synthetic data (no dataset needed):
    python -m src.run_backtest --strategy donchian_breakout --synthetic 5000

ML strategies are fit on the first --train-fraction of the data and scored on
the rest (chronological, no leakage):
    python -m src.run_backtest --strategy ml_classifier \
        --input data/processed/train_15m_indicators.parquet --train-fraction 0.7
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src.strategies import BacktestConfig, Strategy, available, describe, get, run_backtest

LOGGER = logging.getLogger("run_backtest")


def _needs_fit(strategy_cls) -> bool:
    """True for fittable (ML) strategies — those that override ``Strategy.fit``."""
    return strategy_cls.fit is not Strategy.fit


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_params(pairs):
    params = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got {pair!r}")
        key, _, val = pair.partition("=")
        params[key.strip()] = _coerce(val.strip())
    return params


def _synthetic_ohlcv(n: int, seed: int = 7, with_features: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Trending random walk with volatility, so trend + reversion both appear.
    rets = rng.normal(0.0002, 0.01, size=n) + 0.0008 * np.sin(np.arange(n) / 250.0)
    close = 30_000.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2020-01-01", periods=n, freq="15min", name="timestamp")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(1, 100, n)}, index=idx,
    )
    if with_features:
        # A few derived feature columns so ML strategies have something to fit
        # when smoke-testing on synthetic data (no real dataset required).
        from src.strategies import indicators as ind

        c, h, lo = df["close"], df["high"], df["low"]
        df["feat_rsi_14"] = ind.rsi(c, 14)
        df["feat_roc_12"] = ind.roc(c, 12)
        df["feat_zscore_50"] = ind.zscore(c, 50)
        df["feat_atr_14"] = ind.atr(h, lo, c, 14) / c
        df["feat_ema_ratio"] = ind.ema(c, 12) / ind.ema(c, 48) - 1.0
    return df


def _load_input(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def _build_config(args, strategy_cls) -> BacktestConfig:
    cfg = strategy_cls.default_config()
    overrides = {
        "fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps,
        "take_profit": args.tp, "stop_loss": args.sl,
        "horizon_bars": args.horizon, "pnl_unit": args.pnl_unit,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(cfg, key, val)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List registered strategies and exit.")
    parser.add_argument("--strategy", help="Registered strategy name.")
    parser.add_argument("--input", help="Parquet/CSV with OHLCV (+ features).")
    parser.add_argument("--synthetic", type=int, metavar="N", help="Use N bars of synthetic OHLCV instead of --input.")
    parser.add_argument("--param", action="append", help="Strategy param override key=value (repeatable).")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Train split for fittable strategies.")
    parser.add_argument("--base-tf", default=None, help="Base timeframe for tf_{tf}_ column resolution.")
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--tp", type=float, default=None, help="Take-profit (fractional).")
    parser.add_argument("--sl", type=float, default=None, help="Stop-loss (fractional).")
    parser.add_argument("--horizon", type=int, default=None, help="Max holding bars (time stop).")
    parser.add_argument("--pnl-unit", choices=["usdt", "btc"], default=None)
    parser.add_argument("--save-trades", help="Write the trade ledger to this CSV path.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.list or not args.strategy:
        print("Registered strategies:")
        for name, desc in describe().items():
            print(f"  {name:<22} {desc}")
        if not args.strategy:
            return
        print()

    if args.strategy not in available():
        raise SystemExit(f"Unknown strategy {args.strategy!r}. Use --list.")

    if not args.input and not args.synthetic:
        raise SystemExit("Provide --input PATH or --synthetic N.")

    df = _synthetic_ohlcv(args.synthetic) if args.synthetic else _load_input(args.input)
    LOGGER.info("Loaded %d rows", len(df))

    strategy_cls = get(args.strategy)
    strategy = strategy_cls(**_parse_params(args.param))
    cfg = _build_config(args, strategy_cls)

    score_df = df
    if _needs_fit(strategy_cls):
        split = int(len(df) * args.train_fraction)
        train_df, score_df = df.iloc[:split], df.iloc[split:]
        LOGGER.info("Fitting on %d train rows, scoring on %d", len(train_df), len(score_df))
        strategy.fit(train_df)

    result = run_backtest(strategy, score_df, config=cfg, base_tf=args.base_tf)
    summary = result.summary()
    print(json.dumps(summary, indent=2, default=str))

    if args.save_trades and not result.trades.empty:
        result.trades.to_csv(args.save_trades, index=False)
        LOGGER.info("Wrote %d trades to %s", len(result.trades), args.save_trades)


if __name__ == "__main__":
    main()
