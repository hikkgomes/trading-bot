# Autonomous Crypto Trading Platform

This repository contains one production architecture: a PostgreSQL-authoritative
platform with immutable Parquet market data. The platform supports two products:

- `btc_accumulation`: Binance BTCUSDT spot, BTC-denominated accumulation against passive BTC holding, no leverage.
- `active_income`: Binance USDT-margined futures, USDT-denominated day, swing, and scalp research, with bounded long and short risk.

Both checked-in products run in paper mode. Live authority requires an exact
artefact, accepted forward evidence, a current account snapshot, a fresh
preflight, explicit human approval, and a live assignment. Agents and
schedulers cannot approve or submit live orders.

## Architecture

```text
Binance market data
        -> PostgreSQL events and immutable Parquet
        -> canonical features and strategy evaluator
        -> forecast -> portfolio -> six-scope risk
        -> paper or authorised live orders
        -> user stream and REST reconciliation
        -> fills -> accounting -> NAV, reports, alerts, and recovery

immutable research bundles
        -> thesis and bounded generator
        -> screening -> development -> robustness -> protected holdout
        -> sealed strategy artefact -> forward paper -> live_ready
        -> human approval and preflight -> live_canary
```

The same sealed strategy behaviour is used by research, paper, testnet, and
live execution. Behaviour identity and deployment identity are separate.
Research agents receive only an allowlisted context. They cannot see protected
holdout data, credentials, approvals, or live execution state.

## Setup

For local development:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
make test
```

For the Linux platform, follow [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).
The installer uses separate runtime, research, agent, and migration users and
installs the grouped units:

- `trading-platform-runtime.service`
- `trading-platform-research.service`
- `trading-platform-agent.service`
- `trading-platform-control.service`
- `trading-platform-migration.service`

## Developer commands

```bash
make test
make lint
make lint-complexity
make typecheck-platform
make platform-validate
make platform-readiness
make platform-smoke
make platform-testnet-rehearsal
make platform-ci
```

`platform-testnet-connected` is a manual command. It places real testnet
orders only when `CONFIRM=1` is supplied. Never run it with production
credentials.

## Research lifecycle

Each candidate receives a content-addressed dataset bundle containing distinct
screening, development, robustness, and protected-holdout snapshots. Forward
data can be used only after the artefact is sealed. Missing data creates an
observable waiting state. Evaluation uses net returns, product-specific
accounting, chronological validation, family-specific evidence policies, and
bounded mutation, crossover, and lineage budgets.

BTC acceptance is based on terminal BTC NAV excess versus passive holding after
fees and re-entry costs. Active-income acceptance is based on net USDT results
with signed long and short funding, leverage, margin, drawdown, capacity, and
liquidation checks.

## Operations

```bash
make db-alembic
make db-migration-check
make platform-backup-postgresql
make platform-backup-parquet
make platform-backup-verify
make platform-live-authority ARGS="status"
```

The control API provides authenticated status, reports, metrics, pause and
resume, block-new-risk, management-only mode, strategy suspension, entry-order
cancellation, emergency reduction, emergency flatten, and sanitised agent
review access. Reduce-only execution and reconciliation remain active during a
pause.

Legacy SQLite memories and historical JSONL artefacts may be imported once with
`make sqlite-import`. They are retained as provenance and archive material, not
as active lifecycle, approval, position, or execution authority. Do not install
legacy autopilot units beside the platform.
