# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-timeframe BTCUSDT strategy research system. Downloads Binance kline data, builds indicator datasets across timeframes, and searches for repeatable long/short trading patterns using combinatorial condition grids scored as non-overlapping trades.

Two intended products, each fed by its own search and its own active-strategies file:
1. **Position / BTC-accumulation bot** — BTC-denominated (`pnl_unit: btc`); goal is to end with more BTC than buy-and-hold by dodging pullbacks. Source: `outputs/search_v4_btc` → `outputs/active_strategies.json`.
2. **Day-trade / flow bot** — USDT-denominated income; uses order-flow features (CVD, taker imbalance). Source: `outputs/search_v6_15m_flowonly` → `outputs/active_strategies_flow.json`.

## Current Status (as of 2026-06-22)

Both bots run as **paper trading only** from cron on the server (`outputs/cron.log`, `outputs/cron_flow.log`); state lives in `outputs/bot_state.json` (equity 10000) and `outputs/bot_state_flow.json` (equity 1000).

**No search has yet produced a deployable edge.** Across the latest runs, the top-ranked strategies score near-zero DSR (`P(true Sharpe > 0)`) and lose money on the untouched holdout:
- `search_v4_btc`: 105,849 candidates, 39 pass the walk-forward gate; top-3 (exported) DSR ≈ 0.03–0.04, holdout returns −0.5% to −3.5%.
- `search_v6_15m_flowonly`: 342,304 candidates, 3,199 pass the gate; top-2 (exported) DSR ≈ 0.005–0.007, holdout −1.6% to −2.1%.
- `search_v4_usdt`: only 5 candidates pass the gate.

**Known gap:** `export_strategies.py` ranks by DSR and gates on `wf_expectancy > 0` / `passes_filters`, but `--min-dsr` defaults to `None` and the holdout is **report-only** — it never gates admission. So the live paper strategies are ones that lost on their own holdout. Treat current `active_strategies*.json` as **not validated**.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Tests
python -m pytest                    # all tests
python -m pytest tests/test_backtest.py  # single file
python -m pytest -k "test_name"     # single test by name

# Data pipeline (each step depends on prior)
python build_binance_indicator_dataset.py   # download Binance 1m klines, rebuild candles + indicators
python -m src.load_data                      # clean raw CSVs → data/processed/ parquets
python -m src.build_dataset                  # join timeframes into aligned training table

# Strategy search (main workflow)
python -m src.strategy_search --walk-forward --n-jobs 7 --resume ...   # checkpointed; rerun with --resume after interruption
python -m src.day_trade_search --base-tf 5m --walk-forward --n-jobs 7 ...   # same engine: checkpointed, --resume, --holdout-fraction

# Research → execution handoff
python -m src.export_strategies --search-dir outputs/<search dir> [--min-dsr 0.9]   # writes active_strategies.json (passing + positive wf_expectancy; --min-dsr optional, default off)
python -m src.update_candles                                        # incremental 1m candle update + indicator rebuild
python -m src.run_bot                                               # position bot cycle: defaults to active_strategies.json + bot_state.json
python -m src.run_bot --strategies outputs/active_strategies_flow.json \
  --state-file outputs/bot_state_flow.json --starting-equity 1000   # day-trade/flow bot cycle (second cron job)

# Strategy framework (cheap; runs alongside the heavy searches)
python -m src.run_backtest --list                                  # list registered strategies
python -m src.run_backtest --strategy donchian_breakout --synthetic 5000   # smoke test, no dataset
python -m src.run_backtest --strategy sma_cross --input data/processed/train_15m_indicators.parquet --param fast=10
make test            # full suite     make lint / make format   # ruff      make strategies   # list

# Legacy model experiments
python -m src.train_model
python -m src.run_experiments
```

## Architecture

**Data flow:** Raw Binance CSVs → cleaned parquets → multi-timeframe aligned training table → strategy search.

Key modules in `src/`:

- **config.py** — Central paths (`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `INDICATOR_DATA_DIR`) and timeframe constants. All data paths derive from `PROJECT_ROOT`.
- **load_data.py** — CSV cleaning: standardizes column names to lowercase snake_case, detects timestamps, deduplicates. Also provides `configure_logging()` used across all modules.
- **build_dataset.py** — Joins indicator parquets across timeframes onto a base timeframe using closed-candle as-of alignment (`merge_asof`). Columns are prefixed with `tf_{timeframe}_`. Exports feature quality report JSON.
- **discover_patterns.py** — Core pattern engine. Defines the `Condition` dataclass (feature + kind + threshold) and `build_all_conditions()` which generates quantile/cross/boolean conditions. Provides `condition_mask()`, `split_train_test()`, and cross-feature pair detection. Used by both search modules.
- **strategy_search.py** — Position trading search (15m base). Combines 1/2/3-condition rules, scores as non-overlapping trades with fees/slippage/TP/SL. In `--walk-forward` mode (the production path): candidates are generated from the first train window only, thresholds are refit per window, ranking uses walk-forward OOS stats + DSR (never single-split test metrics), and the final `--holdout-fraction` of data is scored report-only. Supports `--n-jobs` parallel scoring and `--resume` from a per-candidate checkpoint. Outputs ranked CSVs and markdown report to `outputs/strategy_search/`.
- **day_trade_search.py** — Day trading search (1m/5m base). Same pattern engine but adds daily risk limits (daily stop-loss, max consecutive losses, cooldown bars). `--walk-forward` mode uses the same window-cached engine as strategy_search (candidates from first window, per-window threshold refits, `--n-jobs`, checkpoint/`--resume`, report-only `--holdout-fraction`). With `--use-atr-tp-sl`, TP/SL values are ATR MULTIPLES (defaults switch to 1.0–3.0 scale), not fractional returns. Outputs to `outputs/day_trade_search/`.
- **walk_forward.py** — `WalkForwardConfig` + `generate_windows()`: anchored/rolling train→test window slices used by both search engines. Defaults per base TF live in `config.py` (`WALK_FORWARD_DEFAULTS`).
- **export_strategies.py** — Writes an active-strategies file (versioned: git SHA, timestamp, per-strategy base timeframe, conditions, risk, fees, drift baseline) from a search output dir. Gates on `passes_filters` and positive `wf_expectancy`; `--min-dsr` is optional (default `None`). NOTE: it does **not** gate on `holdout_total_return` — the holdout is report-only — so a strategy that lost on the holdout can still be exported. This artifact is the only contract between research and execution.
- **run_bot.py** — Paper-trading executor. Reads an active-strategies file (multi-strategy; `--strategies`/`--state-file` select which bot), evaluates closed candles only, sizes positions by equity risk, enforces daily stop/cooldown circuit breakers and a per-strategy win-rate drift kill switch.
- **metrics.py** — Deflated Sharpe Ratio (Bailey/López de Prado: `P(true SR > 0)` after deflation by cross-trial SR dispersion), CSCV PBO, block-bootstrap Sharpe CI, Jaccard strategy clustering.
- **regime.py** — `add_regime_column()`: tags bull/bear/range regime off `tf_1d_close` for regime-aware analysis.
- **train_model.py** — LightGBM regressor baseline with chronological split and early stopping.

Research helpers (not on the production search→export→bot path):

- **feature_screener.py** / **feature_ranking.py** — Screen/prune features and rank them by model importance.
- **label_trades.py** — Triple-barrier-style TP/SL trade labels (`compute_tp_sl_labels`).
- **model_signals.py** — Train a model that emits entry signals from features.
- **meta_labeling.py** — López de Prado meta-labeling: a secondary model to size/filter primary signals.
- **optimize_topk.py** — Optimize top-k strategy selection from a scored pool.
- **mine_patterns.py** — Older standalone pattern-mining experiment (predates the search engines).
- **update_candles.py** — Incremental 1m candle update + indicator rebuild (imports from `build_binance_indicator_dataset.py`); the server's pre-cycle data refresh.
- **audit_indicator_data.py** — Audits indicator parquets for null/quality issues.

**Strategy framework (`src/strategies/`)** — A pluggable layer that runs *any* strategy paradigm through one engine, alongside (not replacing) the condition-grid searches. `base.py` defines the `Strategy` ABC (`generate_signals(df) -> Series in {-1,0,1}`, optional `fit` for ML) + `BacktestConfig` + OHLCV column resolution; `backtester.py` is a vectorized event backtester whose trade model is identical to `strategy_search.simulate_trades` (next-bar-open entry, SL-before-TP intrabar, non-overlapping, `2*(fee+slip)/1e4` round-trip cost, BTC pnl-unit shorts-only) — enforced by `tests/test_strategies.py::test_backtester_matches_search_engine`; `registry.py` is the `@register`/`get`/`available` map; `library/` holds bundled strategies (`sma_cross`, `rsi_reversion`, `donchian_breakout`, `multi_tf_trend`, `ml_classifier`, and `condition_grid` which bridges `active_strategies*.json` rules into the engine); `indicators.py` is pure-pandas TA (no TA-Lib). CLI: `python -m src.run_backtest --list | --strategy NAME [--input PARQUET | --synthetic N]`. See `docs/STRATEGIES.md`.

**Execution layer (`src/execution/`)** — Broker abstraction for futures so the algo bot can target any venue with one interface. `broker.py` = `Broker` ABC + `Order`/`Fill`/`Position`/`OrderSide` (signed positions, quote-currency balances); `paper.py` = `PaperBroker` (simulated fills, fees/slippage, injectable price source, equity/PnL) and `binance_mark_price`; `ccxt_broker.py` = `CcxtBroker` over optional `ccxt` (Binance USDM/Bybit/OKX/…) with hard safety rails — a live order requires `TRADING_LIVE=1` **and** notional ≤ `MAX_NOTIONAL_USD`, else it raises; `config.py` loads `.env` (`ExchangeConfig.from_env`, defaults to paper/testnet). See `docs/EXECUTION.md`. ccxt is NOT in requirements (optional `pip install ccxt`).

**Flow features:** order-flow columns (`cvd_*`, `taker_imbalance_ma_*`, `volume_z_*`, `trades_z_*`, `avg_trade_size_z_*`) are produced by the indicator build and drive the day-trade/flow search (`search_v*_flow*`). Covered by `tests/test_flow_features.py`.

**build_binance_indicator_dataset.py** (root) — Standalone script that downloads Binance monthly kline ZIPs, rebuilds higher-timeframe candles from 1m data, computes TA-Lib indicators with multiple period variants, and writes per-timeframe indicator parquets. Config block at the top of the file controls symbol, market, date range, and timeframes.

## Data Layout

```
data/raw/              — raw CSVs (per timeframe)
data/processed/        — cleaned parquets + aligned training tables + feature reports (~21G)
data/candles/BTCUSDT/  — rebuilt candle parquets from Binance downloads (~13G)
  indicators/          — per-timeframe indicator parquets
outputs/               — strategy search results + execution artifacts:
  search_v*/           — one dir per search (config.json, ranked_strategies.csv, report.md, …)
  active_strategies.json        — position/BTC bot contract (from search_v4_btc)
  active_strategies_flow.json   — day-trade/flow bot contract (from search_v6_15m_flowonly)
  bot_state.json / bot_state_flow.json   — per-bot equity, open positions, circuit-breaker state
  cron.log / cron_flow.log               — per-bot cron run logs
```

Data dirs are large and gitignored; the search dirs and `active_strategies*.json` are the artifacts that matter.

## Conventions

- All modules are run as `python -m src.<module>` from the project root.
- Timeframe columns in training data use the prefix pattern `tf_{timeframe}_{indicator}` (e.g., `tf_15m_close`, `tf_4h_rsi_14`).
- Target columns follow the pattern `future_return_{N}_bars` where N is the holding horizon.
- Chronological train/test splits only — never random splits (leaks future regimes).
- Strategy scoring always accounts for fees (`--fee-bps`), slippage (`--slippage-bps`), take-profit, and stop-loss as non-overlapping trades.

## Important Constraints

- **Dataset builds, searches, and training are expensive** — never run `build_binance_indicator_dataset.py`, `strategy_search`, `day_trade_search`, or `train_model` without explicit user approval.
- The Binance download script uses TA-Lib (C library) which must be installed separately from pip.
- The working environment is Python 3.14 / numpy 2.x / pandas 3.x; `requirements.txt` is pinned to match it. (The old numpy-1.23.x/vectorbt pin is obsolete — vectorbt is no longer used.)
