# Strategy Framework

A single contract for **every** strategy paradigm — simple rules, multi-timeframe
filters, ML classifiers, and the existing condition-grid search rules — so they
all share one backtester, one set of metrics, and one CLI. This sits alongside
the heavy search engines (`strategy_search`, `day_trade_search`); it's the place
to prototype, compare and stress-test individual strategy ideas cheaply.

## The contract

Every strategy subclasses `src.strategies.base.Strategy` and implements one
method:

```python
def generate_signals(self, df: pd.DataFrame) -> pd.Series:
    """Return an int Series in {-1, 0, +1} aligned to df.index."""
```

* `+1` at bar *i* → open a **long** at the open of bar *i+1*
* `-1` → open a **short**
* `0` → do nothing

The backtester turns those entry signals into **non-overlapping** trades with
fees, slippage, and TP/SL/time exits. The trade model is byte-for-byte the same
as `strategy_search.simulate_trades` (verified by
`tests/test_strategies.py::test_backtester_matches_search_engine`), so framework
results are directly comparable to search results.

ML strategies additionally override `fit(self, df)` — train on one slice, score
on a later one. The CLI does the chronological split for you.

## Bundled strategies

| Name | Paradigm | Idea |
|---|---|---|
| `sma_cross` | trend | Fast SMA crosses slow SMA |
| `macd_trend` | trend | MACD line crosses its signal line |
| `supertrend` | trend | ATR trailing-stop trend flip |
| `adx_trend` | trend | +DI/-DI cross, gated by ADX trend strength |
| `multi_tf_trend` | multi-timeframe | Higher-TF EMA trend gates base-TF entries |
| `momentum_roc` | momentum | N-bar rate-of-change breakout, trend-gated |
| `donchian_breakout` | breakout | Break of prior N-bar high/low |
| `keltner_breakout` | breakout / volatility | Close beyond the ATR-based Keltner band |
| `atr_channel_breakout` | breakout / volatility | Close beyond prior close ± k·ATR |
| `bollinger_squeeze` | breakout / volatility | Break out of a low-volatility coil |
| `rsi_reversion` | mean-reversion | Buy oversold RSI, sell overbought |
| `bollinger_reversion` | mean-reversion | Fade a pierce of the outer Bollinger band |
| `zscore_reversion` | mean-reversion | Fade extreme price z-scores |
| `stochastic_reversion` | mean-reversion | %K/%D cross from oversold/overbought |
| `candlestick_reversal` | pattern / reversal | Multi-candle engulfing / hammer / shooting-star reversals |
| `rsi_divergence` | pattern / momentum | Price/RSI divergence at swing pivots (bullish & bearish) |
| `swing_structure` | trend / structure | Break of the last swing high/low (HH-HL vs LH-LL market structure) |
| `regression_channel` | channel | Rolling linear-regression channel — fade the bands or follow breaks |
| `fear_greed_contrarian` | sentiment | Buy extreme fear, sell extreme greed (needs a `fear_greed` column) |
| `btc_cycle_guard` | regime / BTC-accumulation | Step aside (short) near cycle tops (Mayer/Pi-Cycle) or on a trend-EMA break |
| `regime_filter` | regime wrapper | Runs any registered strategy only inside selected `tf_1d_regime_id` states |
| `ml_classifier` | machine learning | Gradient boosting predicts next-horizon direction or TP-before-SL triple-barrier labels |
| `ml_regressor` | machine learning | Gradient boosting predicts forward or TP/SL-capped barrier return; trade above an edge |
| `condition_grid` | bridge | Runs a `discover_patterns` rule (e.g. from `active_strategies.json`) |

Regime IDs are causal and stable: `-1=unknown`, `0=range`, `1=bull_trend`,
`2=bear_trend`, and `3=high_volatility`. A daily close can affect intraday
labels only from the following UTC day, and appending future observations does
not relabel historical rows.

All indicators they use are pure-pandas (no TA-Lib) and live in
`src/strategies/indicators.py` — `sma/ema/rsi/atr`, `macd`, `bollinger_bands`,
`keltner_channels`, `stochastic`, `adx`, `supertrend`, `roc`, `williams_r`,
`zscore`, candlestick patterns (`bullish_engulfing`, `hammer`, `shooting_star`,
`doji`, …), swing pivots (`last_swing_high`/`last_swing_low`), BTC-macro
(`mayer_multiple`, `pi_cycle_top`), plus `crossover`/`crossunder` helpers — so
every strategy runs self-contained on synthetic data on any machine.

> **`btc_cycle_guard` is daily/weekly-scaled.** Mayer Multiple and Pi-Cycle are
> macro (daily) indicators; on 15m bars they degrade into noisy short-MA churn.
> Resample to daily before testing it. It also exposes a design limit: a macro
> "step aside" is really a *position-state regime overlay* (held vs. flat), which
> the fixed-TP/SL trade model approximates poorly. That overlay now ships in the
> **bot itself**: run the position bot with `python -m src.run_bot --regime-guard`
> to block new long entries when the daily macro regime is risk-off (trend break /
> Mayer overheat / Pi-Cycle top). The gate logic is `run_bot.compute_macro_step_aside`.

### Sentiment & execution add-ons

- **Fear & Greed Index** (`src/fear_greed.py`): `fetch_fear_greed()` pulls the
  free alternative.me daily index (cached); `add_fear_greed_column(df)` merges a
  `fear_greed` column so `fear_greed_contrarian` (and any feature pipeline) can
  use it. No new dependency (stdlib `urllib`).
- **Position tactics** (`src/execution/position_plan.py`): `dca_buy_plan`,
  `scaled_exit_plan` (the 50/30/20 "into FOMO wicks" ladder), and `stink_bid_plan`
  return broker-agnostic order legs (`PlanLeg.to_order(...)`) for any `Broker`.

## CLI

```bash
python -m src.run_backtest --list                       # discover strategies
python -m src.run_backtest --strategy donchian_breakout --synthetic 5000   # no dataset needed
python -m src.run_backtest --strategy sma_cross \
    --input data/processed/train_15m_indicators.parquet --param fast=10 --param slow=40
python -m src.run_backtest --strategy ml_classifier \
    --input data/processed/train_15m_indicators.parquet --train-fraction 0.7
python -m src.run_backtest --strategy ml_classifier \
    --input data/processed/train_5m_indicators.parquet --base-tf 5m \
    --param label_mode=triple_barrier --param label_tp=0.005 --param label_sl=0.003 \
    --param feature_screen=spearman --train-fraction 0.7
python -m src.regime --input data/processed/train_15m_indicators.parquet \
    --output outputs/train_15m_regime.parquet
make regime-tag-futures
python -m src.run_backtest --strategy regime_filter --input outputs/train_15m_regime.parquet \
    --base-tf 15m --param strategy=sma_cross --param regime_ids=1 \
    --param child_params='{"fast":5,"slow":20}'
```

Override the trade model with `--fee-bps --slippage-bps --tp --sl --horizon --pnl-unit`.
`--pnl-unit btc` switches to the BTC-accumulation convention (only shorts realise
a return — being long is just holding BTC), for the **position bot**.
ML strategies select numeric feature columns with a lightweight Spearman screen
by default. Use `--param feature_screen=none` for raw first-N feature behavior,
or pass explicit `feature_cols` from Python when running controlled experiments.
`regime_filter` preserves the wrapped strategy's trade-model defaults and masks
signals outside the selected regime ids, which makes it useful in `src.sweep`
grids such as `--grid strategy=sma_cross,macd_trend --grid regime_ids=0,1,2`.
Use child strategy defaults in grids; pass `child_params` in direct
`run_backtest` calls or separate controlled runs when needed.

## Sweeping & comparing (`src.sweep`)

To triage *which* paradigm is worth an expensive walk-forward search, run the
batch harness. It scores every (or a chosen) strategy on the **same chronological
holdout**, ranks them, and benchmarks each against buy-and-hold. Fittable (ML)
strategies are trained on the earlier slice; rule strategies are scored on the
identical holdout, and every row gets a DSR deflated by the number of tried
strategy/grid rows. Optional walk-forward windows split the post-train region
into chronological slices and report pass rate, expectancy, and windowed DSR.

```bash
# Compare everything on synthetic data (no dataset needed)
python -m src.sweep --all --synthetic 8000

# Compare a subset on a real dataset, save the ranked table
python -m src.sweep --strategies sma_cross,macd_trend,supertrend,ml_classifier \
    --input data/processed/train_15m_indicators.parquet --base-tf 15m --out outputs/sweep_15m.csv

# Sweep one strategy across a parameter grid
python -m src.sweep --strategy rsi_reversion --grid period=7,14,21 --grid oversold=20,30 --synthetic 8000

# Require repeated-window robustness and DSR evidence before saving candidates
python -m src.sweep --all --input data/processed/train_15m_indicators.parquet \
    --base-tf 15m --walk-forward-windows 6 --min-wf-pass-rate 0.5 --min-dsr 0.6 \
    --out outputs/sweep_15m_wf.csv
```

The `vs_buy_hold` column is the key number for the **position bot**: a strategy
that can't beat holding has no business in `active_strategies.json`. Sweep output
is still triage; executable artifacts must pass the product policy and explicit
approval path. Same trade-model overrides as `run_backtest` apply to the whole
sweep.

## Adding a strategy

1. Drop a module in `src/strategies/library/` defining a `Strategy` subclass
   decorated with `@register`.
2. Give it a unique class-level `name` and a `description`.
3. Implement `generate_signals` (and `default_params` / `default_config`).
4. Import it in `src/strategies/library/__init__.py`.
5. Add a test in `tests/test_strategies.py`.

```python
from src.strategies import indicators as ind
from src.strategies.base import Strategy
from src.strategies.registry import register

@register
class MyStrategy(Strategy):
    name = "my_strategy"
    description = "One-line description."

    @classmethod
    def default_params(cls):
        return {"window": 20}

    def generate_signals(self, df):
        # self.ohlcv(df) resolves OHLCV using the base_tf the runner set, so the
        # same strategy works on plain OHLCV and on tf_{tf}_-prefixed datasets.
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        sig = self._empty_signals(df)
        sig[close > ind.sma(close, self.params["window"])] = 1
        return sig
```

## How it maps to the two products

* **Position / BTC-accumulation bot** — backtest with `--pnl-unit btc`. The goal
  is more BTC than buy-and-hold; `sma_cross` / `multi_tf_trend` / `donchian_breakout`
  on high timeframes are natural starting points.
* **Day-trade / algo bot** — `--pnl-unit usdt`, lower timeframes, `ml_classifier`
  and `condition_grid` (flow features), executed through `src.execution` on any
  ccxt-supported futures venue. See [EXECUTION.md](EXECUTION.md).
