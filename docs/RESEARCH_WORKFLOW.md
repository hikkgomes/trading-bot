# Canonical research workflow

Research runs on the Linux research service. PostgreSQL is the lifecycle
authority. Parquet is immutable input. Queue payloads contain identities and
commands, not results or raw datasets.

## Data supply chain

1. `data-writer` stores closed market events and bars in partitioned Parquet.
2. `universe-service` records a point-in-time eligible universe.
3. The scheduled catalogue job resolves the latest eligible universe and builds a canonical bundle from available bars.
4. The bundle records source partition hashes, intervals, availability times, feature manifest, cost model, and parameter-set identity.
5. Missing universe, manifests, or complete history returns `waiting_for_dataset`; invalid partitions return `blocked_dataset`.
6. Diagnostic or synthetic bootstrap bundles are excluded from catalogue and readiness authority.

The four pre-artefact roles are `screening`, `development`, `robustness`, and
`protected_holdout`. Their information intervals cannot overlap. Forward data
is created only after an artefact is sealed and is never available to adaptive
research.

## Candidate lifecycle

```text
typed thesis
  -> candidate identity and lineage
  -> screening
  -> development
  -> robustness
  -> protected holdout
  -> sealed artefact
  -> forward paper
  -> forward summary and decision
  -> live_ready
```

Every candidate carries a `CandidateDatasetPlan` with the exact stage
snapshots, product, universe, feature, cost, and parameter identities. The
scheduler records the next stage or a durable waiting reason. It does not
silently skip a candidate.

## Evidence policy

Common fatal checks cover data integrity, causality, semantic parity, realistic
costs, adequate evidence, and executable risk. Product and family profiles
then select applicable evidence:

- BTC accumulation uses BTC NAV excess versus passive BTC holding, complete tactical cycles, drawdown, re-entry, and bull-market protection.
- Futures income uses net USDT PnL, signed funding, effective trades, drawdown, capacity, margin, liquidation buffer, and shortfall.
- Family profiles cover directional, mean-reversion, cross-sectional, relative-value, microstructure, and ML methods.

Non-applicable diagnostics are recorded as `not_applicable`. Insufficient
forward evidence is deferred. Missing or contradictory authority data fails
closed. Statistical diagnostics such as PBO, bootstrap intervals, and regime
breadth are not universal substitutes for economic evidence.

## Generation and agents

The scheduled generator uses typed campaigns, exact and near-duplicate
identity, lineage budgets, bounded mutation and crossover, and failure
feedback. OpenClaw proposals enter an untrusted PostgreSQL proposal boundary,
are schema-checked and compiled to a typed thesis, then use the same queue and
evidence path. Agents cannot read credentials, protected data, approvals, or
live state, and cannot submit orders or grant authority.

## Forward paper and promotion

Raw forward observations contain facts only. A summary worker aggregates
elapsed time, independent decisions, net PnL, drawdown, drift, capacity,
risk-budget availability, data gaps, fill quality, benchmark results, and
strategy decay. One immutable decision consumes that summary.

Research may advance to `forward_paper` and `live_ready` automatically when
configured evidence passes. Live capital requires an explicit human approval,
fresh preflight, current account authority, and a manual live assignment.
Automatic live-canary promotion is disabled.

## Commands

```bash
make platform-validate
make platform-readiness
make platform-report
make platform-ci
```

Use `make sqlite-import SOURCE=...` only for one-time migration of legacy
research memory. Imported rows retain provenance and cannot create active
assignments. Legacy modules are offline migration or research libraries, not
part of the production research workflow.
