# Canonical research workflow

Adaptive research runs on the Linux research service. The separate evaluator
seals datasets and runs frozen evaluations, including protected holdouts.
PostgreSQL is the lifecycle
authority. Frozen datasets are immutable input; daily market partitions are published atomically. Queue payloads contain identities and
commands, not results or raw datasets.

## Data supply chain

1. `data-writer` stores closed market events and bars in partitioned Parquet.
2. `universe-service` records a point-in-time eligible universe.
3. The scheduler pins BTC spot or up to 12 eligible futures markets with sufficient listing history. The catalogue first builds the frozen dataset, then admits bounded candidate pages.
4. The bundle records source partition hashes, intervals, availability times, feature manifest, cost model, and parameter-set identity.
5. Missing universe, manifests, or complete history returns `waiting_for_dataset`; invalid partitions return `blocked_dataset`.
6. Diagnostic or synthetic bootstrap bundles are excluded from catalogue and readiness authority.

The four pre-artefact roles are `screening`, `development`, `robustness`, and
`protected_holdout`. Their information intervals cannot overlap. Forward data
is created only after an artefact is sealed. Adaptive feedback uses development
evidence. Protected outcomes are hidden from research and agent database roles.
Candidate definitions are read through `adaptive_candidate`; protected status
and mutable deferral details remain available only to trusted operational readers.

Fixed-market studies use `SqlUniverseStore.record_fixed_research_scope`. These record
the selection time and specification hash without asserting historical liquidity
eligibility. Dynamic universe studies still require historical membership.
The selected history cutoff is separate from download and bundle creation time.
Retries retain the original bar interval while accepting evidence downloaded later.
Fixed research scopes never become live capture universes. Successor epochs must
have a non-overlapping protected interval after an opening. If a completed
epoch rejected every candidate without opening protected data, a new daily
epoch can continue within the shared admission budget.

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
  -> research_qualified (approved confidence policy)
  -> live_ready (fresh technical preflight)
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

Screening computes the executable contract and costed account path. Development
adds its stability checks. Robustness owns perturbations and statistical
diagnostics. Protected evaluation runs only the frozen strategy and its economic
checks. It does not search parameters on the protected sample. A cross-sectional
portfolio is evaluated as a portfolio; raw market returns are not evidence that
its individual legs have independent edges.

Research confidence approval binds the inference source and its numerical
dependencies, not unrelated deployment tools. Live approval separately binds the
full execution source and installed environment.

Protected data stays unopened until the current research confidence policy has
reviewed calibration approval. Calibration covers the configured maximum cohort,
reports uncertainty on negative-control error rates, and is labelled as selection
inference calibration. The joint bootstrap-t recalculates Bartlett HAC uncertainty
for every resampled panel, retaining cross-candidate dependence. Economic execution and lifecycle contracts require their
own validation. Neither calibration nor paper progress authorises live capital.

## Generation and agents

Native generation persists an epoch and round cursor. Retries use the same
round request and feedback watermark; only new registrations consume another
trial. For an explicitly finite campaign, the coordinator calls
`ResearchRoundCoordinator.close_finite` with the complete registered candidate
set before protected evaluation. A missing epoch leaves protected access
pending. Closure cannot replace an adaptive epoch or admit another candidate.
An admission-created epoch with no started rounds can be sealed as a finite
campaign when its complete declaration matches all reserved trials.

All providers share atomic daily and epoch trial limits. Catalogue pages contain
at most 12 candidates and consume at most 80% of an epoch, leaving capacity for
adaptive proposals. Family/symbol rotation survives restarts and epoch changes.
Directional futures candidates have one symbol each. BTC tactical allocations
use the smallest applicable product, instrument, strategy and sleeve cap.

The separate `research_exploration` evaluator, grammar and file memory have
been removed. Use the canonical generator, dataset and evaluation services
below. Current strategy indicator libraries remain available.

The scheduled generator uses typed campaigns, exact and near-duplicate
identity, lineage budgets, bounded mutation and crossover, and failure
feedback. OpenClaw proposals enter an untrusted PostgreSQL proposal boundary,
are schema-checked and compiled to a typed thesis, then use the same queue and
evidence path. Agents cannot read credentials, protected data, approvals, or
live state, and cannot submit orders or grant authority.

The directional DSL supports time-series, mean-reversion and advanced directional
feature rules. Cross-sectional, relative-value, microstructure, ML, execution and
allocation contracts require their specialised executors. A scalar rule cannot
claim hedged or market-neutral exposure. `typed_rule/v2` honours the declared
comparison: `gt`/`ge` follow deviations and `lt`/`le` reverse them, with mirrored
thresholds. The corrected contract does not reuse v1 behaviour fingerprints.

Scheduled reviews have a sealed context, a one-hour response deadline and a
leased `agent_delivery` job. The scoped transport polls `POST /agent/requests/next`
and submits HMAC-authenticated envelopes to `POST /agent/responses`. Configure
`TRADING_PLATFORM_AGENT_TRANSPORT_KEY` as the same private hex-encoded key of at
least 32 random bytes in the control service and transport process. It must differ
from the operator API token. Keep it out of the OpenClaw configuration and prompt.
The transport credential cannot use operator routes. An accepted reply is stored
before compilation intake; retries and expired-lease recovery finish that same
reply without creating another proposal. New replies after the deadline fail.

The `agent-runtime` process polls the scoped API and sends the sealed request to
`/run/trading-platform-model-broker/model.sock`. It never receives a model or
operator credential. The local broker admits only the `trading-agent` peer,
validates the request and its expiry, limits new attempts to four per hour, and
caches successful request-bound proposals for idempotent delivery.

The optional `trading-platform-model-broker.service` runs as the existing OpenClaw
owner. Set its trusted executable and separate configuration paths in
`/etc/trading-platform/model-broker/model-broker.env`. It uses the original
OpenClaw authentication and shared refresh locks in place. Global state and
session lookup are mounted into private service storage, so a research turn
cannot restore or prune the main agent’s session registry;
model credentials are not copied into the trading service. The separate research
configuration must declare `tools.deny: ["*"]`, `skills: []`, sandbox mode `off`,
workspace access `none`, and disabled hooks. Only the bundled native OpenAI
provider plugin is allowed, with the embedded runtime selected explicitly.
Installed provider overrides, additional plugin paths and source overlays are
rejected; the native package recovery directory is hidden from the service. Its workspace and
`agentDir` are separate directories under the existing OpenClaw state root.
The broker command, agent, prompt and model configuration are fixed server-side.
It cannot deliver messages or select another agent from a client request.

The installer installs the broker unit but leaves it disabled until its paths,
permissions and native authentication are verified. The broker group receives
traversal-only access to the platform configuration parent; other service
credentials and Docker sockets remain inaccessible. Trusted executable, prompt
and configuration paths must not be group- or world-writable. The research
process owns proposal compilation and code checks. The model-facing transport
has no Git metadata access and does not commit changes.

Installed database-cycle checks require three separate service URLs:
`TRADING_PLATFORM_INSTALLED_RUNTIME_DATABASE_URL`,
`TRADING_PLATFORM_INSTALLED_RESEARCH_DATABASE_URL`, and
`TRADING_PLATFORM_INSTALLED_EVALUATOR_DATABASE_URL`. All three must target one
isolated test database, with a distinct service-specific login for each domain.
The verifier rejects the production database name before running a cycle. It
does not load production service environment files into test processes, and it
forces `TRADING_LIVE=0`.
The installed agent ACL check verifies that the agent cannot read research
state, Git metadata or other service environment files.

## Forward paper and promotion

Promotion reads current evidence at worker execution time. Forward observation
starts after the later of artefact creation and actual paper activation; queue
waiting time is not paper history. Restarting reuses the existing paper-entry
outbox job instead of creating another command for the same artefact.

Each paper activation opens an immutable operational qualification window with
its policy, family evidence schema and engine hash. Observations, summaries and
decisions carry that window ID. Summaries cannot combine windows, and an old
window cannot supply current qualification after a repair. Window creation does
not grant statistical confidence; the approved independent research gate remains.

A suspended paper-only strategy resumes into provisional paper observation after
its research artefact remains valid. It does not require an already accepted
forward window or become live-ready on resume. A new window starts at execution
time. Financial ownership follows verified assignment revisions with matching
product, portfolio, sleeve, strategy version, artefact, instrument scope and mode.
Cash, entry basis and carried positions are preserved. Window PnL subtracts the
opening unrealised value, so closing an old position cannot rebook prior gains.
Resuming a strategy with live history additionally requires accepted forward
evidence, current preflight and human live approval.

Assignment admission declares `capital_limit_currency: USDT` and a
`risk_budget_unit` of `quote_notional` or `equity_fraction`. Operator live
assignments retain their existing USDT amount semantics; canonical paper
allocations declare their equity fraction. The exposure guard converts only the
declared unit and rejects missing or unsupported units for new risk. Verified
reductions remain possible when this metadata is absent. Existing numeric caps
have not been increased or inferred from their magnitude.

`config/risk.json` uses `risk/v2`: each scope declares its limit units and the
notional currency is USDT. Allocation, drawdown, margin and depth fractions are
bounded by one; gross/net exposure and turnover multiples are not. Invalid
units, non-finite values, fractional trade counts, unknown limits and conflicting
gross-exposure aliases fail configuration validation. The explicit migration
helper `src.risk.policy.convert_legacy_risk_policy` preserves values and requires
the operator to supply the notional currency. The changed contract is part of the
live-authority configuration hash, so old approvals cannot carry over.

Installed risk-policy records use content-based IDs bound to the logical policy,
account and complete risk/v2 configuration. Portfolio-state requests reference
those IDs. The stored contract retains the original configuration and units;
reading or writing a compiled record checks that its limits match the canonical
compiler. Configuration changes create new records and preserve old policy
history, rather than overwriting an existing policy ID.
Execution rejects legacy records without the verified unit contract. Legacy
limits remain readable only through an explicit historical lookup; diagnostic
smokes also install and use canonical, account-bound policy records.
Risk requests must use the snapshot kinds, product, canonical state and policy
IDs that belong together. Decisions retain the request's evaluation time, so a
retry after partial persistence produces the same records instead of a wall-clock
identity collision.

Risk-input exposure is marked quantity times price divided by equity. Order
notional and depth participation use the proposed quantity change. Projected
funding expense sums paying legs using signed notional and funding rate; funding
credits do not cancel that expense limit. These are separate from allocation
fractions and from the realised funding entries in the accounting ledger.
The final pre-trade guard also applies the configured absolute instrument and
order notional limits. Instrument exposure includes its pending orders; other
symbols do not consume that instrument's cap. The proposed order's gross
reservation uses the largest of its reference, current mark and limit price.
Signed net exposure and account equity use current marks. Non-positive marks
cannot hide an existing position's exposure.
Unfilled opposite orders do not offset reserved net risk: the cap uses the
largest absolute exposure if either side fills first, including the proposed
order. The projected signed net remains a separate reported value.

Risk beta, correlation and derived volatility use up to 256 consecutive closed
candles at the product's dataset timeframe. Pairwise estimates match the exact
start and end of each return interval; trades, other timeframes and unavailable
bars cannot supply samples. At least two matching returns are needed to compute
an estimate. This is an operational point estimate, not statistical confidence.
Each estimate retains its sample count, interval bounds, source snapshot IDs and
availability time. Estimates expire after two candle intervals; conflicting
closes require resolution. The selected BTC benchmark's beta of one is labelled
as an identity, not an empirical estimate. Supplied volatility retains its own
source identity and expires independently of later price updates.

Operators can inspect the latest 100 window records with `GET /paper/windows`.
`POST /paper/restart-forward-window` accepts request_id, strategy_version_id,
product_id, artefact_hash, previous_window_id, requested_by, repair_reference and
reason_code. Supported reasons are market_data_repaired,
forecast_dependency_repaired and execution_reconciliation_repaired. This records
an operator repair attestation and starts a prospective paper-only window using
the frozen policy. It does not change cash, positions or risk controls, or remove
old observations and incidents. Request replay returns the original window;
conflicting requests and stale predecessor IDs fail. The agent credential cannot
use this route. Engine changes require a new compatible research artefact.

Raw forward observations contain facts only. A summary worker aggregates
elapsed time, independent decisions, net PnL, drawdown, drift, capacity,
risk-budget availability, data gaps, fill quality, benchmark results, and
strategy decay. One immutable decision consumes that summary.

Paper admission requires each applicable data, economic and risk control to
pass. Confidence diagnostics cannot be combined into an admission score.
Research qualification requires a sealed cohort result and a matching operator
policy approval recorded before protected access. Later approval cannot change
an existing cohort opening. Code changes, configuration changes and revocation
invalidate use of the approval.

Live capital requires a separate explicit human approval,
fresh preflight, current account authority, and a manual live assignment.
Automatic live-canary promotion is disabled.

## Evidence units and configuration migration

Return observations, completed episodes and statistical effective sample size
are different quantities. Research completion counts come from accounting.
Marks and partial futures reductions do not close a campaign. A BTC cycle closes
when its sold quantity is restored or its quote proceeds are spent; external
transfers are excluded from cycle profit. Completion counts do not assert
statistical independence. The sealed inference receipt reports effective sample
diagnostics separately.

Promotion configuration uses `promotion/v2` and
`minimum_forward_completed_episodes`. Rename the previous
`minimum_forward_effective_episodes` key while preserving its numeric value.
The checked-in policy revisions are `btc-accumulation-v3` and `active-income-v3`;
update product references with the corresponding policy revision. Existing
immutable policy records remain stored, and records with the previous count
contract are rejected for current use.

## Historical execution study

Keep source exports and study outputs outside the checkout. With an authorised
read-only Sheets OAuth token in `GOOGLE_ACCESS_TOKEN`, export the three order
tabs and position-history tab:

```bash
python -m src.research.historical.export \
  --spreadsheet-id <authorised-workbook-id> \
  --output /private/research/export
```

The command prints the content-addressed values and native ID-cell paths. It
uses bounded cell reads and preserves blank row offsets, exact strings and
numeric cell types. `--order-sheet` (repeatable) and `--position-sheet` declare
different source tab names when needed. Export a stable workbook snapshot;
Google's range reads do not provide a transaction across all tabs. Rerunning
the command against unchanged cells gives identical file hashes.

Import the printed values snapshot and native ID-cell metadata paths with:

```bash
python -m src.research.historical.study \
  --source /private/research/study-values.json \
  --id-cells /private/research/study-id-cells.json \
  --timezone Asia/Dubai \
  --output /private/research/reconstruction
```

The source owner confirmed Dubai time for this export. For another export,
supply `--timezone` only after confirming its timezone. Numeric IDs
whose precision was lost remain unresolved. Campaign reconstruction retains
missing inventory and separates spot from swaps; no strategy or assignment is
created by the import. Both input hashes, export schema and source row records
are retained in the private result. The workbook display timezone is not used
to infer the execution export timezone. Unreconciled campaign differences stay
visible at the existing tolerance.

`src.data.okx_history.download_history` acquires bounded native trade, mark and
index candles. `download_funding_history` reads the official daily funding
archives, with at most 93 archive days per invocation. It retains source URLs,
checksums and signed funding rates, resumes completed downloads and reports
missing instruments. Complete archive coverage does not certify the historical
funding schedule or replace private account bills.

Add `--market-history /private/research/native-bars.json` for each native
traded-price file produced by `download_history`. The command records a separate
immutable study for each market, splits directions, links campaign IDs to the
last strictly earlier closed bar and retains excluded campaigns with reasons.
The declared defaults are a four-bar outcome horizon and four-bar embargo;
use `--structure-horizon-bars` and `--structure-embargo-bars` to specify another
study. Gaps and unknown exposure cannot supply controls. Each comparison uses
the recorded exposure strictly before its actual execution, with separate
specifications for each exposure state. All actual entry bars are excluded from
the control pool. Entries within one bar retain their campaign references and
share a market-bar cluster; they are not independent price observations.
The study includes 14 July and does not interpret bar returns as obtainable
entry fills or stop prices.

Run `study_entry_structure` separately for each market and trade direction.
Pass closed-bar availability timestamps and an as-of exposure state for every
bar: `flat`, `long`, `short`, `hedged`, or `None` when unresolved. Pseudo-entries
match time, trend, volatility and exposure; unknown exposure cannot supply a
control. Persist every specification with `write_study`. These are descriptive
price-response studies, not fee-adjusted strategy returns. Stop counterfactuals
require intraperiod execution data and an explicit stop rule. A later candidate
still needs a complete frozen strategy specification and independent validation.

## Commands

```bash
make platform-validate
make platform-readiness
make platform-report
make platform-ci
```

Use `make sqlite-import SOURCE=...` only for one-time migration of legacy
research memory. Imported rows retain provenance and cannot create active
assignments. Deployment uses the platform supervisor.

Before Linux handover, back up the existing installation and inventory both
system and user service managers. The installer refuses enabled or active
legacy user units before changing system services. Retire those user units only
after the backup and replacement checks; a system-only unit listing is not
sufficient. The local install dry run checks paths and shell syntax, not remote
host authority or readiness.
