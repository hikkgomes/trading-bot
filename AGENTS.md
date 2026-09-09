# AGENTS.md

Guidance for coding agents working in this repository.

## Current System

This repo is now an autonomous crypto trading framework for a light Linux
server. The production authority is the PostgreSQL platform supervisor in
`src.services.supervisor`, configured by `config/platform.json`. The file-based
autopilot and old search/execution entry points have been retired.

Products:

- `btc_accumulation`: BTC base asset, spot `BTCUSDT`, conservative BTC
  accumulation, no leverage. Strategy artifacts must use BTC PnL and beat
  buy-and-hold on holdout.
- `active_income`: USDT base asset, Binance USDT futures, active
  day/swing/scalp income with tight per-symbol and portfolio risk limits. The
  checked-in executable product is `BTCUSDT`; a dynamic liquidity screen
  researches eligible altcoins into symbol-isolated candidate artifacts that
  cannot activate until the exact product/symbol is explicitly configured.

Live authority is held in PostgreSQL, with explicit human approval bound to the
artefact, account, environment, instrument scope, capital, code and configuration.
A current matching preflight is required. Research and paper progression never
grant live approval.

## Safe Commands

```bash
make test
make lint
make platform-validate
make platform-readiness
make platform-process-once
make platform-report
make control ARGS="status"
```

For 24/7 Linux operation use the system-level Linux platform installer:

```bash
bash scripts/install_platform_services.sh
```

Do not reintroduce the old cron deployment scripts. Historical search outputs
belong in `runtime/quarantine`, not beside active runtime files in `outputs`.

## Research Workflow

Use the canonical research workflow in `docs/RESEARCH_WORKFLOW.md`. Adaptive
research and frozen evaluation run under separate restricted service roles. Historical
imports are read-only evidence, not promotable strategies. Heavy searches,
dataset rebuilds and model training require explicit user approval.

## Execution Notes

`src.services.order_execution` routes approved intents to paper or live venues.
`src.services.financial_application` applies fills, positions, fees and PnL as
one transaction. Native protection and emergency recovery use these records.

PostgreSQL owns assignments, approvals, orders, financial state and reports.
Market and strategy files live in the configured `data/` stores. `runtime/`
contains local operational state and quarantine. Missing paper artefacts are a
safe waiting state; missing live authority is a hard failure.

## Constraints

- Use chronological validation only; never random train/test splits.
- Keep fees, slippage, TP/SL, risk, and holdout gates in all execution artifacts.
- Keep generated data and old research outputs out of git.
- Prefer small, tested changes. Run focused tests first, then `make
  lint`, `make platform-validate`, full `pytest`, and
  `make platform-smoke` for platform changes.
