# Multi-Timeframe BTCUSDT Strategy Research & Execution

Downloads BTCUSDT candle/indicator data, searches for repeatable long/short
patterns across timeframes with walk-forward validation, and runs the surviving
strategies in a paper-trading executor.

The system targets **two products**:

1. **Position / BTC-accumulation bot** — BTC-denominated (`pnl_unit: btc`). Goal: end with **more BTC than buy-and-hold** by stepping aside during pullbacks (not to grow a USDT balance).
2. **Day-trade / flow bot** — USDT-denominated income, driven by order-flow features (CVD, taker imbalance).

Each product is fed by its own search and its own `active_strategies*.json`. The
research pipeline is heavy and runs on a workstation; the executor is light and
runs 24/7 on a server.

---

## Current Status (2026-06-22)

**Both bots run as paper trading only. No search has yet produced a deployable edge.**

The research instrumentation is solid (walk-forward, Deflated Sharpe Ratio,
CSCV PBO, an untouched holdout, strategy clustering). What it is telling us is
that, so far, the patterns found do **not** generalize:

| Search | Candidates | Pass WF gate | Top DSR | Holdout return |
|---|---|---|---|---|
| `search_v4_btc` (position) | 105,849 | 39 | ≈ 0.03–0.04 | −0.5% to −3.5% |
| `search_v6_15m_flowonly` (flow) | 342,304 | 3,199 | ≈ 0.005–0.007 | −1.6% to −2.1% |
| `search_v4_usdt` | 270,160 | 5 | — | — |

DSR is `P(true Sharpe > 0)` after deflating for the number of trials — values
near zero mean the result is statistically indistinguishable from noise. The
strategies currently in `active_strategies*.json` all **lost money on the
untouched holdout**.

**Known gap:** `export_strategies.py` ranks by DSR and admits on
`passes_filters` + positive `wf_expectancy`, but `--min-dsr` defaults to off and
the holdout is **report-only** — it never gates admission. So holdout-losers got
promoted to the live (paper) bots. Adding a holdout + DSR gate would currently
export **nothing**, which is the honest result. Treat `active_strategies*.json`
as not validated until that changes.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TA-Lib (the C library) must be installed separately from pip (`brew install ta-lib`).
The environment is Python 3.14 / numpy 2.x; `requirements.txt` is pinned to match
the working venv. Dev tooling is in `requirements-dev.txt`, the execution-only
set in `requirements-bot.txt`.

## Pipeline

Each step depends on the previous one.

```bash
# 1. Download Binance monthly 1m klines, rebuild candle timeframes, compute indicators.
#    Edit the config block at the top to change SYMBOL, MARKET, START_MONTH, END_MONTH, timeframes.
python build_binance_indicator_dataset.py

# 2. Clean raw CSV exports → data/processed/ parquets.
python -m src.load_data

# 3. Join all timeframes onto the 15m base (closed-candle as-of alignment) → aligned training table.
python -m src.build_dataset
```

Expected raw inputs (one CSV per timeframe) in `data/raw/`: `btcusdt_1m.csv`,
`btcusdt_5m.csv`, `btcusdt_15m.csv`, `btcusdt_30m.csv`, `btcusdt_60m.csv`,
`btcusdt_240m.csv`, `btcusdt_1d.csv`, `btcusdt_1w.csv`.

Timeframe columns use the prefix pattern `tf_{timeframe}_{indicator}` (e.g.
`tf_15m_close`, `tf_4h_rsi_14`). Order-flow columns (`cvd_*`,
`taker_imbalance_ma_*`, `volume_z_*`, `trades_z_*`, `avg_trade_size_z_*`) feed
the flow search.

## Strategy Search (walk-forward is the production path)

In `--walk-forward` mode, candidates are generated from the first train window
only, thresholds are refit per window, ranking uses walk-forward out-of-sample
stats + DSR (never single-split test metrics), and the final
`--holdout-fraction` of data is scored **report-only**. Runs are checkpointed —
rerun with `--resume` after an interruption.

```bash
# Position / BTC search (15m base, pnl in BTC)
python -m src.strategy_search --walk-forward --n-jobs 7 --resume \
  --output-dir outputs/search_v4_btc ...

# Day-trade / flow search (15m base, flow features, pnl in USDT)
python -m src.day_trade_search --base-tf 5m --walk-forward --n-jobs 7 --resume \
  --output-dir outputs/search_v6_15m_flowonly ...
```

> Searches are expensive (hours to days) — do not launch without intent.

Each search dir contains: `config.json`, `filter_summary.json`,
`scored_strategies_all.csv`, `ranked_strategies.csv`,
`ranked_strategies_clustered.csv`, and `report.md`. `report.md` is the place to
read DSR, holdout return, walk-forward pass rate, and PBO at a glance.

## Research → Execution Handoff

```bash
# Export the top strategies from a search dir into a bot contract.
# --min-dsr is optional and OFF by default; the holdout is NOT yet a gate (see Known gap).
python -m src.export_strategies --search-dir outputs/search_v4_btc            # → outputs/active_strategies.json
python -m src.export_strategies --search-dir outputs/search_v6_15m_flowonly \
  --output outputs/active_strategies_flow.json                                # → flow bot contract

# Refresh candles/indicators before a cycle (server pre-step).
python -m src.update_candles

# Run one paper-trading cycle. Defaults serve the position bot:
python -m src.run_bot

# The flow bot is a second invocation with its own contract + state file:
python -m src.run_bot \
  --strategies outputs/active_strategies_flow.json \
  --state-file outputs/bot_state_flow.json \
  --starting-equity 1000
```

`run_bot` evaluates **closed candles only**, sizes positions by equity risk, and
enforces daily-stop / cooldown circuit breakers plus a per-strategy win-rate
drift kill switch. On the server it runs from cron at the base-timeframe cadence;
logs go to `outputs/cron.log` and `outputs/cron_flow.log`.

## Strategy Framework (prototype any strategy paradigm cheaply)

Alongside the heavy condition-grid searches, `src/strategies/` provides one
contract for **every** strategy type — simple rules, multi-timeframe filters, ML
models, and a bridge that runs exported search rules through the same engine. Its
backtester reproduces the search's trade model exactly, so results are
comparable. No dataset needed to smoke-test:

```bash
python -m src.run_backtest --list
python -m src.run_backtest --strategy donchian_breakout --synthetic 5000
python -m src.run_backtest --strategy ml_classifier \
  --input data/processed/train_15m_indicators.parquet --train-fraction 0.7

# Compare every strategy on the same holdout vs buy-and-hold (param grids too):
python -m src.sweep --all --synthetic 8000
python -m src.sweep --all --input data/processed/train_15m_indicators.parquet --base-tf 15m
```

23 bundled strategies across every paradigm — trend (`sma_cross`, `macd_trend`,
`supertrend`, `adx_trend`, `multi_tf_trend`, `swing_structure`), momentum
(`momentum_roc`, `rsi_divergence`), breakout/volatility (`donchian_breakout`,
`keltner_breakout`, `atr_channel_breakout`, `bollinger_squeeze`), channel
(`regression_channel`), mean-reversion (`rsi_reversion`, `bollinger_reversion`,
`zscore_reversion`, `stochastic_reversion`), patterns (`candlestick_reversal`),
sentiment (`fear_greed_contrarian`), BTC-macro regime (`btc_cycle_guard`), ML
(`ml_classifier`, `ml_regressor`), and the `condition_grid` bridge. Full guide:
[docs/STRATEGIES.md](docs/STRATEGIES.md).

The position bot can run with a macro **regime guard** (`python -m src.run_bot
--regime-guard`) that blocks new long entries when BTC is risk-off on the daily
(trend break / Mayer overheat / Pi-Cycle top) — the accumulation overlay.

> Reality check: on the 2020–2026 15m holdout, none of the rule strategies beat
> buy-and-hold at default params/fees (`src.sweep`). The framework is for
> *finding* an edge, not proof one exists — same honest verdict as the searches.

## Execution Layer (paper + live futures)

`src/execution/` is a broker abstraction so the algo bot can target any
ccxt-supported futures venue with one interface. `PaperBroker` simulates fills
(default, safe); `CcxtBroker` trades live/testnet behind hard safety rails (a
real order needs `TRADING_LIVE=1` **and** notional ≤ `MAX_NOTIONAL_USD`). Copy
`.env.example` → `.env` to configure. Full guide: [docs/EXECUTION.md](docs/EXECUTION.md).

## Tooling

```bash
make setup     # venv + research + dev deps        make test      # full suite
make lint      # ruff check                         make format    # ruff format + autofix
make strategies   # list registered strategies      make sweep-synth   # compare all on synthetic
make sweep INPUT=data/processed/train_15m_indicators.parquet BASE_TF=15m   # compare on real data
make help      # all targets (heavy ones guarded behind CONFIRM=1)
```

## Tests

```bash
python -m pytest                         # all
python -m pytest tests/test_strategies.py tests/test_execution.py   # new framework + execution
python -m pytest -k "test_name"
```

## Legacy / auxiliary

Not on the production search→export→bot path: `train_model.py`,
`run_experiments.py`, `mine_patterns.py`, plus research helpers
(`feature_screener.py`, `feature_ranking.py`, `label_trades.py`,
`model_signals.py`, `meta_labeling.py`, `optimize_topk.py`, `regime.py`,
`audit_indicator_data.py`).

## Next Steps

Full plan in [docs/ROADMAP.md](docs/ROADMAP.md). The integrity-critical ones:

- Make the holdout a hard export gate (require `holdout_total_return > 0` and a meaningful `--min-dsr`, e.g. ≥ 0.9). Expect this to currently export nothing.
- Benchmark the position bot in **BTC terms vs. buy-and-hold**, not USDT P&L.
- Treat the flow/day-trade edge as unproven: more indicators/pairs multiply false positives (3,199 "passing" candidates at DSR ≈ 0 is the symptom), not edge.
