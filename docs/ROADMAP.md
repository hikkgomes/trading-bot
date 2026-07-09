# Roadmap

Two products, each with its own research stream, execution artifact, and
autopilot product:

1. **Position / BTC-accumulation bot** — BTC-denominated; beat buy-and-hold by
   dodging pullbacks. Artifact: `outputs/active_strategies_position.json`.
2. **Day-trade / algo bot** — USDT income on any automatable futures venue.
   Artifact: `outputs/active_strategies_flow.json`, executed via `src/execution`.

## Where things stand (2026-07-08)

The research instrumentation is solid (walk-forward, Deflated Sharpe, CSCV PBO,
untouched holdout, clustering). The honest result so far: **no search has
produced a deployable edge**. The export path now treats holdout as a hard gate
by default, so a weak research run should export no live candidate rather than
quietly promoting noise.

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
- **Autopilot runtime** (`src/autopilot`): config validation, status reporting,
  file-based pause/flatten control, approval ledger, promotion review, preflight,
  alerts, and offline rehearsal.
- **Broker-routed executor** (`src/run_bot.py`): paper by default, live broker
  injection for approved products, broker position reconciliation, and spot/futures
  routing split by product objective.
- Repo hygiene: real code under version control, `.venv`/caches untracked,
  `pyproject.toml` (ruff + pytest), `Makefile`, split requirements.

## Next — research integrity (do these before trusting any live signal)

- [x] **Make the holdout a hard export gate.** Export requires positive holdout
      return by default; use a meaningful `--min-dsr` once enough comparable
      candidates exist.
- [x] Benchmark the position bot in **BTC terms vs. buy-and-hold**, not USDT P&L.
      BTC artifacts now export `holdout_excess_return_vs_buy_hold`, and the
      BTC-accumulation policy requires it to be positive before paper/live use.
- [x] Treat the flow edge as unproven: 3,199 "passing" candidates at DSR ≈ 0 is
      a false-positive symptom, not edge. The active-income autopilot validates
      bounded rotating slices, deflates DSR by the full available scenario
      universe, and requires DSR >= 0.60 before an artifact can be exported or
      executed.

## Next — new strategy generation (the framework makes these cheap)

- [x] Broad strategy library across paradigms + a sweep/compare harness
      (`src.sweep`) that benchmarks them on a holdout vs buy-and-hold.
- [x] Wire `src.sweep` into walk-forward + DSR. It now reports DSR deflated by
      the tried strategy/grid rows, can score repeated post-train windows, and
      exposes `--min-dsr` / `--min-wf-pass-rate` triage filters before heavier
      validation.
- [x] Expand ML research tooling: unified `ml_classifier`/`ml_regressor` now
      support triple-barrier/capped-return targets plus Spearman feature
      screening; standalone `src/meta_labeling.py` and purged/embargoed
      walk-forward model signals remain available for heavier research passes.
- [x] Add regime-conditional variants: `src.regime` tags market states and
      `regime_filter` can wrap any registered strategy so sweeps can test the
      same idea only inside selected `tf_1d_regime_id` regimes.
- [x] Portfolio handoff: export now prefers `ranked_strategies_clustered.csv`
      low-overlap representatives when present, with `--raw-ranked` reserved
      for deliberate inspection runs.

## Next — execution

- [x] **Position-state regime overlay for the BTC bot.** `btc_cycle_guard`
      (Mayer/Pi-Cycle/trend-break, daily-scaled) catches the real cycle tops
      (Dec'21, May'22) but the fixed-TP/SL short model whipsaws and trails hold.
      Re-express macro "step aside" as a held-vs-flat state in `run_bot`, not a
      backtester short — that's the natural fit for accumulation.
- [x] Route `src/run_bot.py` fills through a `Broker` so paper/live is a swap.
- [x] Wire `CcxtBroker` to a testnet end-to-end command
      (`make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5`) with approval,
      preflight, testnet-only, flat-position, notional-cap, entry, and immediate
      close gates.
- [x] Add position reconciliation + an emergency futures flatten control on the live path.
- [ ] Run a real exchange testnet rehearsal with sandbox keys and inspect resulting
      balances, positions, fills, status, alerts, and preflight artifacts.
