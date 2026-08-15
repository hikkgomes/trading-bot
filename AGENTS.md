# AGENTS.md

Guidance for coding agents working in this repository.

## Current System

This repo is now an autonomous crypto trading framework for a light Linux
server. The production supervisor is `src.autopilot.runtime` with configuration
in `config/autopilot.json`.

Products:

- `btc_accumulation`: BTC base asset, spot `BTCUSDT`, conservative BTC
  accumulation, no leverage. Strategy artifacts must use BTC PnL and beat
  buy-and-hold on holdout.
- `active_income`: USDT base asset, Binance USDT futures, active
  day/swing/scalp income with tight per-symbol and portfolio risk limits. The
  checked-in executable product is `BTCUSDT`; a dynamic liquidity screen
  researches eligible altcoins into symbol-isolated candidate artifacts that
  cannot activate until the exact product/symbol is explicitly configured.

The mandatory live gate is enforced in code: no new or changed strategy may run
live unless its behavior fingerprint is explicitly approved in
`runtime/approvals.json`. Approval is also bound to the artifact path and product
identity, including symbol. Live products require a fresh matching preflight
report by default.

## Safe Commands

```bash
make test
make lint-autopilot
make autopilot-validate
make readiness
make autopilot-once
make report
make artifact-hygiene
make control ARGS="status"
```

For 24/7 Linux operation use the user-level systemd installer:

```bash
bash scripts/install_autopilot_service.sh
```

Do not reintroduce the old cron deployment scripts. Historical search outputs
belong in `runtime/quarantine`, not beside active runtime files in `outputs`.

## Research Workflow

Preferred research path is `research_exploration/`, documented in
`docs/RESEARCH_WORKFLOW.md`. The autopilot can run cheap synthetic research
smokes and bounded real-data research cycles. Exported candidates still remain
paper-only until review and explicit approval.

The legacy search modules (`src.strategy_search`, `src.day_trade_search`) remain
available for manual experiments, but heavy searches, dataset rebuilds, and model
training are expensive. Do not run them without explicit user approval.

## Execution Notes

`src/run_bot.py` is the closed-candle executor used by paper and live broker
paths. It validates artifact risk and fee blocks at load time, enforces daily
trade limits, one open position per product/symbol, daily loss stops,
consecutive-loss cooldowns, drift checks, and product-aware spot/futures routing.

Runtime state, reports, approval ledgers, alerts, and quarantined files live
under `runtime/`. Strategy artifacts, when present, are:

- `outputs/active_strategies_position.json` for BTC accumulation.
- `outputs/active_strategies_flow.json` for active income.

Missing paper artifacts are a safe waiting state. Missing live artifacts are a
hard failure.

## Constraints

- Use chronological validation only; never random train/test splits.
- Keep fees, slippage, TP/SL, risk, and holdout gates in all execution artifacts.
- Keep generated data and old research outputs out of git.
- Prefer small, tested changes. Run focused tests first, then `make
  lint-autopilot`, `make autopilot-validate`, full `pytest`, and
  `make autopilot-once` for runtime changes.
