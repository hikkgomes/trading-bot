"""Vectorized event backtester for the strategy framework.

The trade model is intentionally identical to ``src.strategy_search.simulate_trades``
so that a strategy run through this engine is directly comparable to a
condition-grid candidate from the search:

* an entry signal at bar ``i`` opens at the **open of bar i+1**;
* exits are taken intrabar on TP/SL (stop checked before take-profit — the
  pessimistic assumption) and otherwise at the close after ``horizon_bars``;
* trades are **non-overlapping** (a new entry is only allowed after the prior
  trade has exited);
* round-trip cost is ``2 * (fee_bps + slippage_bps) / 10_000``;
* with ``pnl_unit='btc'`` only shorts realise a return (longs == holding BTC).

The result carries the per-trade net returns, the trade ledger, an equity
curve, and a summary dict that reuses ``src.metrics`` for Sharpe / DSR inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import metrics
from src.strategies.base import BacktestConfig, Strategy, extract_ohlcv
from src.trade_utils import gross_return_for_pnl_unit


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    config: BacktestConfig
    strategy_name: str

    @property
    def returns(self) -> np.ndarray:
        if self.trades.empty:
            return np.array([], dtype=float)
        return self.trades["net_return"].to_numpy(dtype=float)

    def summary(self) -> dict[str, float]:
        r = self.returns
        n = int(r.size)
        if n == 0:
            return {
                "strategy": self.strategy_name,
                "trades": 0,
                "win_rate": float("nan"),
                "avg_net_return": float("nan"),
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": float("nan"),
                "sharpe": 0.0,
                "psr": 0.0,
            }
        equity = np.cumprod(1.0 + r)
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        gains = r[r > 0].sum()
        losses = -r[r < 0].sum()
        skew = float(pd.Series(r).skew()) if n > 2 else 0.0
        kurt = float(pd.Series(r).kurt() + 3.0) if n > 3 else 3.0
        sharpe = metrics.sharpe_ratio(r)
        # Probabilistic Sharpe (DSR with a single trial == no deflation).
        psr = metrics.deflated_sharpe_ratio(sharpe, n_trials=1, skew=skew, kurt=kurt, n_obs=n)
        return {
            "strategy": self.strategy_name,
            "trades": n,
            "win_rate": float((r > 0).mean()),
            "avg_net_return": float(r.mean()),
            "total_return": float(equity[-1] - 1.0),
            "max_drawdown": float(drawdown.min()),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
            "sharpe": float(sharpe),
            "psr": float(psr),
        }


def _simulate(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: np.ndarray,  # int in {-1, 0, +1} per bar
    index: pd.Index,
    cfg: BacktestConfig,
) -> list[dict]:
    n = len(close)
    horizon = int(cfg.horizon_bars)
    tp, sl = float(cfg.take_profit), float(cfg.stop_loss)
    cost = cfg.round_trip_cost
    trades: list[dict] = []
    next_allowed_entry = 0
    max_entry_index = n - horizon - 1

    for i in range(n):
        sig = int(direction[i])
        if sig == 0:
            continue
        entry_index = i + 1
        if entry_index < next_allowed_entry or entry_index > max_entry_index:
            continue
        is_long = sig > 0
        entry = open_[entry_index]
        exit_index = entry_index + horizon
        exit_price = close[exit_index]
        exit_reason = "time"

        for j in range(entry_index, entry_index + horizon + 1):
            if is_long:
                if low[j] <= entry * (1 - sl):
                    exit_index, exit_price, exit_reason = j, entry * (1 - sl), "stop"
                    break
                if high[j] >= entry * (1 + tp):
                    exit_index, exit_price, exit_reason = j, entry * (1 + tp), "take_profit"
                    break
            else:
                if high[j] >= entry * (1 + sl):
                    exit_index, exit_price, exit_reason = j, entry * (1 + sl), "stop"
                    break
                if low[j] <= entry * (1 - tp):
                    exit_index, exit_price, exit_reason = j, entry * (1 - tp), "take_profit"
                    break

        gross = gross_return_for_pnl_unit(
            entry,
            exit_price,
            is_long=is_long,
            pnl_unit=cfg.pnl_unit,
        )
        net = gross - cost
        trades.append(
            {
                "signal_time": index[i],
                "entry_time": index[entry_index],
                "exit_time": index[exit_index],
                "direction": "long" if is_long else "short",
                "entry": float(entry),
                "exit": float(exit_price),
                "exit_reason": exit_reason,
                "gross_return": float(gross),
                "net_return": float(net),
                "holding_bars": int(exit_index - entry_index),
            }
        )
        next_allowed_entry = exit_index + 1

    return trades


def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
    base_tf: str | None = None,
) -> BacktestResult:
    """Generate signals from ``strategy`` and simulate trades over ``df``."""
    cfg = config or strategy.default_config()
    ohlcv = extract_ohlcv(df, base_tf=base_tf)

    strategy.base_tf = base_tf
    signals = strategy.generate_signals(df)
    if not isinstance(signals, pd.Series):
        signals = pd.Series(signals, index=df.index)
    direction = signals.reindex(df.index).fillna(0).clip(-1, 1).round().to_numpy(dtype=int)

    trades = _simulate(ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, direction, df.index, cfg)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        equity = pd.Series([cfg.initial_equity], index=df.index[:1] if len(df.index) else None)
    else:
        eq = cfg.initial_equity * np.cumprod(1.0 + trades_df["net_return"].to_numpy())
        equity = pd.Series(eq, index=pd.Index(trades_df["exit_time"], name="exit_time"))
    return BacktestResult(
        trades=trades_df,
        equity_curve=equity,
        config=cfg,
        strategy_name=strategy.name,
    )
