# Platform status and remaining evidence

The repository has one production architecture: PostgreSQL owns lifecycle,
assignments, approvals, orders, fills, positions, risk, accounting, controls,
and reports. Immutable Parquet owns market and research data.

## Implemented

- Linux platform services with separate runtime, research, agent, control, and migration domains.
- Point-in-time universes and content-addressed screening, development, robustness, and protected-holdout bundles.
- Parquet-to-bundle production with explicit waiting and invalid-data outcomes.
- Typed candidate plans, immutable stage identities, lineage, deduplication, bounded generation, and failure feedback.
- Shared strategy behaviour across research, paper, testnet, and live execution with parity receipts.
- BTC-denominated accumulation accounting against passive BTC holding.
- USDT futures accounting with signed funding, leverage, margin, liquidation, capacity, and shortfall checks.
- Family-specific evidence profiles with applicable, diagnostic, deferred, and fatal outcomes.
- Forward-paper observations, immutable summaries, drift and drawdown checks, and `live_ready` decisions.
- Human-only live authority with exact artefact, account, instrument, approval, preflight, and risk binding.
- Durable protective stops, Algo Order updates, REST recovery, market-gap repair, emergency management-only controls, and alerts.
- PostgreSQL operator reports with funnel, queue, recovery, authority, data, backup, and health SLIs.
- Ruff, C90, mypy, focused tests, full tests, smoke, and testnet rehearsal targets.

## Not claimed

- No checked-in strategy is claimed to be profitable.
- Synthetic bootstrap data is diagnostic only and cannot satisfy production readiness.
- Local tests do not prove the external OptiPlex, PostgreSQL, exchange, alert, or connected-testnet state.

## Operator evidence still required

- Deploy with [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) on the Linux authority host.
- Run the non-secret audit commands in the plan and retain the output outside the repository.
- Confirm systemd state, PostgreSQL funnel counts, Parquet coverage, report freshness, and backup restore.
- Run the connected testnet rehearsal with a dedicated flat account when exchange credentials are available.
- Review every live-ready packet before any explicit live assignment.

Legacy `src.autopilot`, JSONL, SQLite, and file artefacts remain migration or
offline research inputs only. They are not runtime, approval, position, or
execution authority.
