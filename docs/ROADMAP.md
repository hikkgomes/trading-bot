# Roadmap

Two products, each with its own search, contract file, and bot:

1. **Position / BTC-accumulation bot** — BTC-denominated; beat buy-and-hold by
   dodging pullbacks. Source: `outputs/search_v4_btc` → `active_strategies.json`.
2. **Day-trade / algo bot** — USDT income on any automatable futures venue.
   Source: flow searches → `active_strategies_flow.json`, executed via
   `src/execution`.

## Where things stand (2026-06-22)

The research instrumentation is solid (walk-forward, Deflated Sharpe, CSCV PBO,
untouched holdout, clustering). The honest result so far: **no search has
produced a deployable edge** — top strategies score DSR ≈ 0 and lose on the
holdout. The live paper bots run strategies that lost on their own holdout
because the export step never gated on it.

## Now shipped (this setup pass)

- Unified **strategy framework** (`src/strategies`): one `Strategy` contract +
  vectorized backtester (matches the search trade model) + registry + CLI
  (`python -m src.run_backtest`). See [STRATEGIES.md](STRATEGIES.md).
- **17 bundled strategies across every paradigm** — trend (`sma_cross`,
  `macd_trend`, `supertrend`, `adx_trend`, `multi_tf_trend`), momentum
  (`momentum_roc`), breakout/volatility (`donchian_breakout`, `keltner_breakout`,
  `atr_channel_breakout`, `bollinger_squeeze`), mean-reversion (`rsi_reversion`,
  `bollinger_reversion`, `zscore_reversion`, `stochastic_reversion`), ML
  (`ml_classifier` direction, `ml_regressor` magnitude), and the `condition_grid`
  bridge — on a pure-pandas indicator toolkit (no TA-Lib).
- **Sweep/compare harness** (`python -m src.sweep`): ranks any set of strategies
  on the same leakage-free holdout with a buy-and-hold benchmark + param grids.
- **Execution layer** (`src/execution`): `PaperBroker` + ccxt live/testnet
  adapter with notional + live-mode safety rails. See [EXECUTION.md](EXECUTION.md).
- Repo hygiene: real code under version control, `.venv`/caches untracked,
  `pyproject.toml` (ruff + pytest), `Makefile`, split requirements.

## Next — research integrity (do these before trusting any live signal)

- [ ] **Make the holdout a hard export gate.** Require `holdout_total_return > 0`
      and a meaningful `--min-dsr` (≥ 0.9). Expect this to export *nothing* today
      — that's the correct, honest outcome.
- [ ] Benchmark the position bot in **BTC terms vs. buy-and-hold**, not USDT P&L.
- [ ] Treat the flow edge as unproven: 3,199 "passing" candidates at DSR ≈ 0 is
      a false-positive symptom, not edge. Reduce the trial count, don't add features.

## Next — new strategy generation (the framework makes these cheap)

- [x] Broad strategy library across paradigms + a sweep/compare harness
      (`src.sweep`) that benchmarks them on a holdout vs buy-and-hold.
- [ ] Wire `src.sweep` into walk-forward + DSR (it currently uses a single
      chronological holdout); keep only what clears a real holdout gate.
- [ ] Expand `ml_classifier`/`ml_regressor`: triple-barrier labels, meta-labeling
      (`src/meta_labeling.py`), purged/embargoed CV, feature screening.
- [ ] Add regime-conditional variants (`src/regime.py`) — bull/bear/range.
- [ ] Portfolio layer: combine low-correlation surviving strategies (Jaccard
      clustering already exists in `src/metrics.py`).

## Next — execution

- [ ] **Position-state regime overlay for the BTC bot.** `btc_cycle_guard`
      (Mayer/Pi-Cycle/trend-break, daily-scaled) catches the real cycle tops
      (Dec'21, May'22) but the fixed-TP/SL short model whipsaws and trails hold.
      Re-express macro "step aside" as a held-vs-flat state in `run_bot`, not a
      backtester short — that's the natural fit for accumulation.
- [ ] Route `src/run_bot.py` fills through a `Broker` so paper/live is a swap.
- [ ] Wire `CcxtBroker` to a testnet end-to-end (sandbox keys, tiny notional).
- [ ] Add position reconciliation + a kill switch on the live path.
