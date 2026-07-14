"""Batch backtest + compare many strategies on identical out-of-sample data.

This is the framework's head-to-head harness: it runs every (or a chosen
subset of) registered strategy through the *same* engine over the *same*
chronological holdout, then ranks them and benchmarks each against simple
buy-and-hold. Use it to triage which paradigm is worth a full walk-forward
search before spending the expensive compute.

Discipline: all strategies are scored on the last ``1 - train-fraction`` of the
data (the holdout). Fittable (ML) strategies are trained on the earlier part;
rule strategies ignore the train part but are still scored on the same holdout,
so the comparison is apples-to-apples and leakage-free.

Examples
--------
Compare everything on synthetic data:
    python -m src.sweep --all --synthetic 8000

Compare a subset on a real dataset and save the table:
    python -m src.sweep --strategies sma_cross,macd_trend,supertrend,ml_classifier \
        --input data/processed/train_15m_indicators.parquet --base-tf 15m --out outputs/sweep_15m.csv

Sweep one strategy across a parameter grid:
    python -m src.sweep --strategy rsi_reversion --grid period=7,14,21 --grid oversold=20,30 \
        --synthetic 8000
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from src import metrics
from src.run_backtest import _coerce, _load_input, _synthetic_ohlcv
from src.strategies import BacktestConfig, Strategy, available, get, run_backtest

LOGGER = logging.getLogger("sweep")


def _needs_fit(strategy_cls) -> bool:
    return strategy_cls.fit is not Strategy.fit


def _apply_overrides(cfg: BacktestConfig, overrides: dict) -> BacktestConfig:
    changes = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **changes) if changes else cfg


def _default_config(strategy) -> BacktestConfig:
    resolver = getattr(strategy, "resolved_default_config", None)
    return resolver() if callable(resolver) else strategy.default_config()


def _buy_and_hold(score_df: pd.DataFrame, base_tf: str | None, pnl_unit: str) -> float:
    """Total return of simply holding over the scored window.

    For ``pnl_unit='btc'`` the benchmark is holding BTC, which by definition
    returns 0 in BTC terms — the bar a position strategy must clear.
    """
    if pnl_unit == "btc":
        return 0.0
    from src.strategies.base import extract_ohlcv

    close = extract_ohlcv(score_df, base_tf=base_tf).close
    if close.size < 2 or close[0] == 0:
        return float("nan")
    return float(close[-1] / close[0] - 1.0)


def _parse_grid(grid_args: list[str] | None) -> dict[str, list]:
    """Parse repeated ``--grid key=v1,v2,..`` into {key: [coerced values]}."""
    grid: dict[str, list] = {}
    for item in grid_args or []:
        if "=" not in item:
            raise SystemExit(f"--grid expects key=v1,v2,.. got {item!r}")
        key, _, vals = item.partition("=")
        grid[key.strip()] = [_coerce(v.strip()) for v in vals.split(",") if v.strip()]
    return grid


def _param_combos(grid: dict[str, list]) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [
        dict(zip(keys, combo, strict=False))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def _deflated_sharpe(returns: np.ndarray, *, n_trials: int) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size <= 3:
        return 0.0
    sharpe = float(metrics.sharpe_ratio(returns))
    series = pd.Series(returns)
    return float(
        metrics.deflated_sharpe_ratio(
            sharpe,
            n_trials=max(1, int(n_trials)),
            skew=float(series.skew()),
            kurt=float(series.kurt() + 3.0),
            n_obs=int(returns.size),
        )
    )


def _walk_forward_stats(
    strategy_cls,
    params: dict,
    cfg: BacktestConfig,
    df: pd.DataFrame,
    *,
    base_tf: str | None,
    start_index: int,
    windows: int,
) -> dict:
    if windows <= 0:
        return {}
    boundaries = np.linspace(start_index, len(df), windows + 1).astype(int)
    rows = []
    returns: list[float] = []
    for index in range(windows):
        lo, hi = int(boundaries[index]), int(boundaries[index + 1])
        if hi <= lo:
            continue
        train_df = df.iloc[:lo]
        score_df = df.iloc[lo:hi]
        strat = strategy_cls(**params)
        if _needs_fit(strategy_cls):
            strat.base_tf = base_tf
            strat.fit(train_df)
        result = run_backtest(strat, score_df, config=cfg, base_tf=base_tf)
        summary = result.summary()
        window_return = float(summary.get("total_return") or 0.0)
        trades = int(summary.get("trades") or 0)
        returns.extend(float(value) for value in result.returns)
        rows.append(
            {
                "window": index + 1,
                "rows": len(score_df),
                "trades": trades,
                "total_return": window_return,
                "passed": bool(trades > 0 and window_return > 0),
            }
        )
    passed = sum(1 for row in rows if row["passed"])
    return {
        "wf_windows": len(rows),
        "wf_pass_rate": float(passed / len(rows)) if rows else 0.0,
        "wf_expectancy": float(np.mean([row["total_return"] for row in rows])) if rows else 0.0,
        "wf_trades": int(sum(row["trades"] for row in rows)),
        "wf_window_returns": json_dumps_compact(
            [round(float(row["total_return"]), 8) for row in rows]
        ),
        "wf_returns": np.asarray(returns, dtype=float),
    }


def json_dumps_compact(value) -> str:
    import json

    return json.dumps(value, separators=(",", ":"))


def run_sweep(
    df: pd.DataFrame,
    strategy_names: list[str],
    *,
    base_tf: str | None = None,
    train_fraction: float = 0.7,
    overrides: dict | None = None,
    grids: dict[str, dict[str, list]] | None = None,
    walk_forward_windows: int = 0,
) -> pd.DataFrame:
    """Backtest each strategy on the holdout and return a ranked summary frame.

    ``grids`` maps a strategy name to a param grid; each grid point becomes its
    own row (labelled ``name[k=v,..]``).
    """
    overrides = overrides or {}
    grids = grids or {}
    split = int(len(df) * train_fraction)
    train_df, score_df = df.iloc[:split], df.iloc[split:]
    if len(score_df) < 50:
        raise SystemExit(
            f"Holdout too small ({len(score_df)} rows). Lower --train-fraction or add data."
        )

    rows: list[dict] = []
    n_trials = sum(len(_param_combos(grids.get(name, {}))) for name in strategy_names)
    for name in strategy_names:
        cls = get(name)
        for params in _param_combos(grids.get(name, {})):
            label = (
                name if not params else f"{name}[{','.join(f'{k}={v}' for k, v in params.items())}]"
            )
            try:
                strat = cls(**params)
                cfg = _apply_overrides(_default_config(strat), overrides)
                if _needs_fit(cls):
                    strat.base_tf = base_tf
                    strat.fit(train_df)
                result = run_backtest(strat, score_df, config=cfg, base_tf=base_tf)
                summary = result.summary()
                summary["strategy"] = label
                summary["n_trials"] = n_trials
                summary["dsr"] = _deflated_sharpe(result.returns, n_trials=n_trials)
                if walk_forward_windows:
                    wf = _walk_forward_stats(
                        cls,
                        params,
                        cfg,
                        df,
                        base_tf=base_tf,
                        start_index=split,
                        windows=walk_forward_windows,
                    )
                    wf_returns = wf.pop("wf_returns", np.array([], dtype=float))
                    summary.update(wf)
                    summary["wf_dsr"] = _deflated_sharpe(wf_returns, n_trials=n_trials)
                rows.append(summary)
            except Exception as exc:  # keep the sweep going if one strategy fails
                LOGGER.warning("Strategy %s failed: %s", label, exc)
                rows.append(
                    {
                        "strategy": label,
                        "trades": 0,
                        "n_trials": n_trials,
                        "dsr": 0.0,
                        "error": str(exc),
                    }
                )

    pnl_unit = overrides.get("pnl_unit") or "usdt"
    bh = _buy_and_hold(score_df, base_tf, pnl_unit)
    out = pd.DataFrame(rows)
    if "total_return" in out.columns:
        out["vs_buy_hold"] = out["total_return"] - bh
    out.attrs["buy_and_hold"] = bh
    out.attrs["holdout_rows"] = len(score_df)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true", help="Sweep every registered strategy.")
    grp.add_argument("--strategies", help="Comma-separated subset of strategy names.")
    grp.add_argument("--strategy", help="A single strategy (use with --grid to sweep params).")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Parquet/CSV with OHLCV (+ features).")
    src.add_argument("--synthetic", type=int, metavar="N", help="Use N bars of synthetic OHLCV.")

    parser.add_argument(
        "--grid", action="append", help="Param grid key=v1,v2,.. (repeatable; needs --strategy)."
    )
    parser.add_argument(
        "--base-tf", default=None, help="Base timeframe for tf_{tf}_ column resolution."
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Chronological train split (rest is the scored holdout).",
    )
    parser.add_argument(
        "--walk-forward-windows",
        type=int,
        default=0,
        help="Optional number of chronological post-train windows to score.",
    )
    parser.add_argument(
        "--min-dsr",
        type=float,
        default=None,
        help="Filter displayed/output rows to deflated Sharpe >= this value.",
    )
    parser.add_argument(
        "--min-wf-pass-rate",
        type=float,
        default=None,
        help="With --walk-forward-windows, filter rows to wf_pass_rate >= this value.",
    )
    parser.add_argument(
        "--sort-by", default="sharpe", help="Metric column to sort the table by (default: sharpe)."
    )
    parser.add_argument("--out", help="Write the ranked table to this CSV path.")
    # Trade-model overrides (apply to every strategy in the sweep).
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--tp", type=float, default=None, help="Take-profit (fractional).")
    parser.add_argument("--sl", type=float, default=None, help="Stop-loss (fractional).")
    parser.add_argument("--horizon", type=int, default=None, help="Max holding bars.")
    parser.add_argument("--pnl-unit", choices=["usdt", "btc"], default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.all:
        names = available()
    elif args.strategies:
        names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    else:
        names = [args.strategy]
    # These need externally-supplied inputs (conditions / a fear_greed column);
    # skip them in a blind sweep.
    _needs_extra_input = {"condition_grid", "fear_greed_contrarian"}
    names = [n for n in names if n not in _needs_extra_input]

    grids = {}
    if args.grid:
        if not args.strategy:
            raise SystemExit("--grid requires --strategy (a single strategy to sweep).")
        grids[args.strategy] = _parse_grid(args.grid)

    df = _synthetic_ohlcv(args.synthetic) if args.synthetic else _load_input(args.input)
    LOGGER.info("Loaded %d rows; %d strategies", len(df), len(names))

    overrides = {
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "take_profit": args.tp,
        "stop_loss": args.sl,
        "horizon_bars": args.horizon,
        "pnl_unit": args.pnl_unit,
    }
    table = run_sweep(
        df,
        names,
        base_tf=args.base_tf,
        train_fraction=args.train_fraction,
        overrides=overrides,
        grids=grids,
        walk_forward_windows=max(0, int(args.walk_forward_windows)),
    )
    if args.min_dsr is not None and "dsr" in table.columns:
        table = table[table["dsr"] >= args.min_dsr]
    if args.min_wf_pass_rate is not None and "wf_pass_rate" in table.columns:
        table = table[table["wf_pass_rate"] >= args.min_wf_pass_rate]

    sort_col = args.sort_by if args.sort_by in table.columns else "total_return"
    table = table.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

    display_cols = [
        c
        for c in [
            "strategy",
            "trades",
            "win_rate",
            "total_return",
            "vs_buy_hold",
            "max_drawdown",
            "profit_factor",
            "sharpe",
            "psr",
            "dsr",
            "wf_pass_rate",
            "wf_expectancy",
            "wf_dsr",
            "error",
        ]
        if c in table.columns
    ]
    bh = table.attrs.get("buy_and_hold", float("nan"))
    print(f"\nHoldout rows: {table.attrs.get('holdout_rows')}  |  buy & hold return: {bh:+.4f}\n")
    with pd.option_context(
        "display.max_rows", None, "display.width", 160, "display.float_format", lambda v: f"{v:.4f}"
    ):
        print(table[display_cols].to_string(index=False))

    if args.out:
        table.to_csv(args.out, index=False)
        LOGGER.info("Wrote ranked table to %s", args.out)


if __name__ == "__main__":
    main()
