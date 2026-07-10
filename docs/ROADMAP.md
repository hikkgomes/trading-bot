# Roadmap and honest status

The repository now implements the autonomous research platform; it does not
claim to contain a profitable strategy. The two independent objectives are:

1. BTC-denominated accumulation through bounded BTC/USDT spot step-aside and
   rebuy behavior, without leverage.
2. USDT-denominated active-income research for Binance futures across scalping,
   day, and swing horizons, with strict risk limits.

## Implemented

- Resumable native-timeframe Binance history, atomic parquet publication,
  feature contracts, coverage/freshness checks, and regime data.
- A typed compositional strategy grammar that emits data specifications rather
  than code, with fresh generation, recursive failure-aware mutation, crossover,
  dynamic safe features, and hard complexity/risk/time/resource limits.
- Persistent SQLite experiment memory with canonical behavioral identity,
  global deduplication, lineage, novelty, pending-work recovery, engine-scoped
  development evidence, dataset/protocol identity, and consistent backups.
- Chronological train/validation/OOS/sensitivity/final-holdout validation with
  cumulative multiple-testing penalties and per-candidate crash checkpoints.
- Durable lineage-scoped holdout claims committed before protected data is
  read. Protected outcomes cannot influence generation or OpenClaw.
- Continuous development-only adaptation of primitive weights, method mix,
  parent selection, and mutation choices while preserving fresh exploration.
- Paper-by-default execution for both products, isolated paper evidence for a
  staged replacement of an already-live product, and explicit candidate
  activation.
- Exact artifact/strategy/product/engine-bound human approval, production
  preflight, futures testnet rehearsal, native protective stops, reconciliation,
  circuit breakers, pause/panic/flatten controls, and recovery intents.
- Separate lightweight systemd services for position supervision, bounded jobs,
  and healthchecks; resource limits, monitoring, alerts, maintenance, verified
  backups, offline rehearsal, and recovery documentation.
- Optional Telegram status/alerts/pause-only control and an isolated OpenClaw
  development-research proposal bridge. Neither can approve or execute.

## Still required from the operator

- Deploy on the target Linux server and let the initial history bootstrap
  complete; this can take time and network bandwidth.
- Verify continuous paper operation, restart/reboot recovery, alert delivery,
  resource use, daily backups, and an off-host restore rehearsal.
- Supply Telegram/OpenClaw configuration only if those optional edges are
  wanted, using the isolation instructions in `COMMUNICATIONS.md`.
- Evaluate whatever evidence the system produces. "No keeper" is a valid and
  expected research result; do not weaken gates to force a strategy.
- Before risking money, accumulate adequate exact-fingerprint forward-paper
  evidence, run the connected futures testnet rehearsal where applicable,
  inspect the promotion packet, and explicitly approve the exact active
  artifact. Start with one product and tiny caps.

## Not promised

Backtests, holdouts, paper trading, and adaptive search reduce avoidable errors;
they do not establish future profitability. The platform is ready to conduct
research and enforce governance, not to guarantee an edge.
