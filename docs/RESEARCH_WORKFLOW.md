# Autonomous Research Workflow

This is the operating model for the strategy-creation system. The system does
not ship with a supposedly profitable strategy. It runs continuously, creates
testable strategy specifications, rejects weak or duplicate ideas, learns from
development results, and sends only validated candidates into paper trading.
No backtest or paper result is a promise of future profit.

The two research objectives are:

- `btc_accumulation`: BTC/USDT spot, performance measured in BTC, no leverage.
  A generated `short` signal means stepping aside from BTC into USDT and later
  re-entering; it is not a borrowed or leveraged spot short.
- `active_income`: BTC/USDT USDT-margined futures, performance measured in
  USDT, with scalp, day-trade, and swing search spaces. The deployment policy
  applies isolated margin, one-way mode, strict position/risk limits, and at
  most 1x leverage.

The default configuration keeps both products in paper mode. A new or changed
strategy cannot become live through this research workflow. Live use requires
the separate human activation, review, approval, preflight, and (for futures)
testnet sequence documented in [DEPLOYMENT.md](DEPLOYMENT.md).

## The continuous loop

```text
native market history -> indicator inventory -> typed strategy factory
                                                |       ^
                       optional OpenClaw ideas --+       |
                                                |  development-only feedback
                                                v       |
                                      canonical SQLite memory
                                                |
                                      generated research batch
                                                |
                            train -> validation -> robustness checks
                                                |
                                  durable holdout claim -> final holdout
                                                |                X
                                                |       no adaptive feedback
                                                v
                         paper artifact or isolated staged-candidate paper
                                                |
                                      explicit human live gate
```

The installed job worker performs this loop 24/7. By default, data refreshes
run every six hours (with futures 1m refreshed daily), the strategy factory and
real-data research run daily, and staged candidates for already-live products
receive an isolated paper cycle every five minutes. A newly generated batch
also makes the research cycle due, so it does not wait for a second full daily
interval. Jobs run separately from trading supervision and only one due job is
started per worker cycle, preventing research from delaying position
management.

History downloads are also bounded: Binance requests use 1,000-candle pages, a
0.2-second inter-page delay, a 5,000-page per-dataset ceiling, periodic atomic
checkpoints, and the job's system timeout. Re-running after a page-budget or
network stop resumes the checkpoint instead of restarting the download.

Each generation is deliberately bounded for a light server. The checked-in
defaults emit at most 15 candidates, at most 3 per search space, stop after 800
attempts or 25 seconds, and cap parent-pool and lineage sizes. At 80% of the
explicit 48 MiB SQLite budget, the factory performs at most 5,000 rows of
evidence-preserving compaction and an integrity-checked vacuum. Repeated cycles
provide the ongoing exploration; one process is never allowed to consume the
server indefinitely. If valid compacted memory still exceeds the hard ceiling,
generation pauses fail closed instead of deleting safety evidence.

## What the factory creates

The trusted strategy language is a typed, serializable specification, not
Python code. Every idea contains:

- market context and direction;
- multi-timeframe `regime`, `setup`, and `trigger` predicates;
- bounded take-profit, stop-loss, and time exit;
- bounded risk per trade, maximum position fraction, daily trade cap,
  cooldown, and optional volatility limits.

The grammar combines causal closed-candle primitives into fresh compositions.
It can discover additional eligible indicator columns from the local parquet
schemas, but those columns may only enter through approved operations such as
rolling ranks, slopes, price/average relations, range breaks, or candlestick
conditions. Unknown operations, arbitrary code, missing stages, invalid
directions, unsafe exits, excessive risk, unavailable features, and structures
outside the configured search space are rejected.

The factory uses three native generation methods:

- fresh grammar samples, which preserve a mandatory exploration floor;
- recursive mutation of eligible pre-holdout parents;
- crossover between compatible eligible parents.

Mutation can respond to development failures: for example, low trade counts
can lead to simpler or less restrictive entries, while instability or
fragility can lead to different filters, exits, or structure. This is
programmatic hypothesis generation inside a safety grammar, not a fixed list of
handwritten strategies and not arbitrary self-modifying code.

The authoritative search spaces and limits live in
`config/research_factory.json`. They cover active-income scalping, day trading,
and swing trading plus two BTC-accumulation horizons. The config loader refuses
to start if either objective or required horizon is missing or mismatched,
search-space fields/types/timeframe ordering are invalid, budgets are unsafe,
the exploration floor is removed, or the holdout policy is changed. The history
plan is derived from these exact search-space roles rather than a separate fixed
scenario list.

## Persistent experiment memory

`runtime/research/experiment_memory.sqlite3` is the durable memory of the
research system. It is transactional SQLite rather than an append-only report.
It records:

- the canonical behavioral identity of every generated strategy;
- human-readable IDs mapped to that identity;
- generation method, primitives, novelty, parent hashes, and ancestry;
- immutable evaluation context: data-snapshot manifest, chronological window,
  protocol, and phase;
- development outcomes and rejection reasons;
- durable protected-holdout claims and retirement state.

Canonical identity ignores display IDs, prose, tags, research-horizon labels,
search-space names, and ordering of commutative conditions while retaining
executable rules and product, market, and PnL-unit context. Renaming a campaign
or moving identical behavior between research taxonomies therefore does not
make it a new experiment. Exact duplicates are rejected; fresh roots are also
compared against the complete product history, including retired and
holdout-exposed work. Parameter-only descendants remain attached to their
lineage so legitimate adaptation cannot disguise itself as a fresh root.

Every new behavior is registered before it appears in a generated batch. If a
batch has not yet been evaluated, the next factory cycle resumes those pending
records instead of creating an ever-growing queue. A strategy already tested
against the same snapshot, window, and protocol is not tested again. Generated
ideas that cannot compile against the real feature inventory are retired so an
unsupported pending item cannot stall future generations.

Stored JSON may be transparently compressed by automatic maintenance. The
decoded canonical spec, immutable test context, outcomes, lineage, human IDs,
and protected-holdout claims remain exact and are re-hashed by the deep
integrity check. Compaction does not summarize away or delete old tests. Do not
attempt to decode SQLite text columns with external JSON SQL functions; use the
documented memory APIs and reports.

Do not edit, delete, replace, or vacuum this database manually. Losing it loses
duplicate history and protected-data claims. Use the backup and recovery
procedure below.

## How adaptation works

The factory may learn only from non-protected development evidence. It uses
that evidence to adjust:

- the relative weight of safe grammar primitives;
- the fresh/mutation/crossover method mix;
- which eligible parents are sampled;
- which failure-aware mutation operators are tried next.

Smoothed weights, novelty, duplicate rate, and a minimum fresh-exploration
share prevent one early result from taking over the search. Rejections remain
useful evidence; a run with no keepers is a normal research result, not a
runtime failure.

Adaptive evidence is scoped to the current research-engine digest (source,
Python, and relevant dependency versions). After evaluator code or dependency
changes, old results remain in the audit memory and duplicate history, but they
cannot weight current primitives, operators, or parent selection until the
behavior is evaluated under the new protocol.

The factory automatically emits the existing canonical behavior for gradual
revalidation, bypassing exact/near-dedup registration because this is not a new
idea. Never-evaluated crash-recovery work retains first priority. Old-engine
revalidation is limited to one candidate per search space per cycle and cannot
consume that space's final slot, so a deploy cannot crowd out genuinely fresh
research. Holdout-exposed or retired ancestry is never revalidated. A
taxonomy-only search-space rename resumes matching product/opportunity/timeframe
behavior under the new label rather than stranding or duplicating it.

Final holdout outcomes are never returned to these adaptive selectors.
Holdout-exposed or retired strategies and their tainted ancestry are excluded
from the parent pool. Forward-paper outcomes are also a final, adaptive-free
gate: they support an operator's promotion decision but do not teach the
generator to optimize against the paper period.

The optional OpenClaw edge supplies untrusted idea proposals only. The trusted
factory either maps a high-level thesis into the native grammar or validates a
complete suggestion against the exact typed schema, search space, features,
and limits. Invalid proposals are rejected and native generation fills the
batch. OpenClaw is never required for progress and has no execution or approval
path. See [COMMUNICATIONS.md](COMMUNICATIONS.md).

## Validation and protected holdout

The real-data research cycle refuses to validate until each generated scenario
meets its native-timeframe history contract: earliest and latest observations,
minimum span, minimum rows, cadence, and required features. Insufficient
history blocks validation and export before any holdout is consumed.

Candidates then pass these chronological stages in order:

1. Train (first 60%): enough trades and a positive training edge.
2. Validation (next 20%): enough trades and positive out-of-sample behavior.
3. OOS windows: the result must not depend on one lucky subperiod.
4. Sensitivity: nearby entry thresholds, exits, and horizons must remain viable.
5. Final holdout (last 20%): enough trades and positive return.

Fees and slippage are included, entry uses the next bar, higher-timeframe joins
use closed candles, and stop handling is conservative. Deflated Sharpe Ratio
accounts for the tested population; active-income export currently requires a
minimum deflated score in addition to a positive holdout.

Immediately before any final-holdout read, the cycle first records that the
candidate passed development, then commits a claim for that data snapshot and
every root of the candidate's lineage. Only after that transaction succeeds may
the validator read the holdout. If the process crashes one instruction later,
the claim remains consumed and a retry fails closed with
`holdout_already_consumed`. Do not clear such a claim to get another look.

The protected result is stored for audit and export gating, but
`generator_feedback()` excludes all protected phases. The adaptive feedback
embedded in the generated batch and exported to OpenClaw uses that
development-only API and never receives final-holdout outcomes or metrics.
Human audit artifacts retain protected results because the operator must be
able to review the gate; those results are not routed back into generation.

## First run and manual smoke checks

From the repository root with `.venv` installed:

```bash
make autopilot-validate
make research-factory-validate
make research-history-plan
make research-history-bootstrap
make research-smoke
```

`research-smoke` generates and validates synthetic ideas across both products.
It proves wiring, not an edge.

Run one bounded real-data iteration:

```bash
make research-once
```

That target runs `research-generate` followed by `research-cycle`. They can also
be run separately while diagnosing a handoff:

```bash
make research-generate
make research-cycle
```

Inspect the result without relying on log text:

```bash
jq '{ok, generated_at, budget, summary}' runtime/research/generated_hypotheses.json
jq '{ok, skipped, error, generated_batch, summary}' runtime/research_cycle.json
make report
jq '.experiment_memory, .generated_batch, .research_cycle.summary.generative_search' \
  runtime/operator_report.json
make healthcheck
```

Healthy operation means the factory report is safe and bounded, experiment
memory passes integrity checks, the research cycle either evaluates the new
batch or explicitly reports why it skipped, and the job worker continues to
advance. It does not mean a keeper must exist. `no_exportable_strategies` is an
honest successful outcome.

## Paper handoff and the mandatory live gate

For a product configured `paper`, a keeper may atomically replace its configured
paper strategy artifact only while the product has no open position. The paper
bot then accumulates exact-fingerprint evidence. Changing behavior starts that
evidence again.

For a product already configured `live`, research never changes the active
artifact. It writes a separate candidate to
`runtime/candidates/<product>.json`, and `candidate_paper_cycle` runs that exact
digest in isolated paper state. Wait until
`runtime/candidate_paper_status.json` reports
`candidate_activation_ready: true` after the configured trade-count, time-span,
return, drawdown, and loss-streak requirements.

Activation is itself an explicit local maintenance action and grants no live
approval. The product and jobs must be paused, state must be flat and
reconciled, the services must be stopped, and the reviewed digest must match:

```bash
make candidate-paper-once
jq '.products[] | select(.candidate_activation_ready == true)' \
  runtime/candidate_paper_status.json

make control ARGS="pause-product active_income --reason 'candidate activation review'"
make control ARGS="pause-jobs --reason 'candidate activation review'"
systemctl --user stop \
  trading-bot-autopilot.service trading-bot-autopilot-jobs.service

CANDIDATE_DIGEST=$(jq -r \
  '.products[] | select(.product == "active_income" and .candidate_activation_ready == true) | .candidate_digest' \
  runtime/candidate_paper_status.json)
make activate-candidate \
  PRODUCT=active_income CANDIDATE_DIGEST="$CANDIDATE_DIGEST" \
  CONFIRM=1 OPERATOR="$USER"
```

After activation, rebuild the promotion review against the active path and the
candidate's exact-fingerprint paper log. Only the human operator may run the
approval command printed by that packet:

```bash
make promotion-review \
  PRODUCT=active_income \
  ARTIFACT=outputs/active_strategies_flow.json \
  TRADE_LOG=runtime/candidates/active_income_paper_trades.csv
```

Approval binds the exact artifact digest, every selected strategy fingerprint,
product identity, and execution-engine identity. Any new strategy or artifact,
or a relevant code/dependency/product change, invalidates it. Live entry also
requires fresh production preflight evidence and, for `active_income`, a
matching testnet rehearsal. Telegram and OpenClaw cannot approve, activate,
resume, enable live mode, or place orders. Follow the complete sequence in
[DEPLOYMENT.md](DEPLOYMENT.md); do not use this shortened overview as a live
cutover checklist.

## Important artifacts

| Path | Purpose |
|---|---|
| `config/research_factory.json` | Trusted search spaces, budgets, and immutable holdout policy. |
| `runtime/research/experiment_memory.sqlite3` | Canonical strategy, lineage, evaluation, and holdout-claim memory. |
| `runtime/research/generated_hypotheses.json` | Latest bounded, research-only generated batch and safe development feedback. |
| `runtime/research_cycle.json` | Latest real-data validation, export, coverage, and generative-search summary. |
| `runtime/research_cycle_state.json` | Market/batch markers and cycle cursor/recovery state; not a substitute for SQLite memory. |
| `outputs/research_exploration/experiment_log.jsonl` | Detailed append-only validation audit, including protected results. Never expose it to OpenClaw. |
| `runtime/incubation_candidates.json` | Ranked research-attention queue; explicitly non-executable and non-promotable. |
| `outputs/active_strategies_position.json` | BTC-accumulation active paper/live artifact, according to product mode. |
| `outputs/active_strategies_flow.json` | Active-income active paper/live artifact, according to product mode. |
| `runtime/candidates/<product>.json` | Inert staged replacement for a product already configured live. |
| `runtime/candidate_paper_status.json` | Isolated staged-candidate paper and activation readiness. |
| `runtime/operator_report.json` | Compact generated-batch, development-memory, research, product, and job status. |
| `runtime/healthcheck.json` | Machine-readable blockers and warnings for the external watchdog. |

All of these runtime/output artifacts are machine-specific and should remain
out of git.

## Failure and recovery

Use `make report`, `make healthcheck`, and the job-worker journal before changing
state:

```bash
systemctl --user status trading-bot-autopilot-jobs.service --no-pager
journalctl --user -u trading-bot-autopilot-jobs.service -n 200 --no-pager
make report
make healthcheck
```

Common outcomes:

- `generated_batch_not_ready`, `missing`, or `read_error`: run
  `make research-factory-validate`, then `make research-generate`. The research
  cycle refuses malformed or unsafe batches.
- `insufficient_history_coverage`: rerun
  `make research-history-bootstrap MARKET=futures` and/or `MARKET=spot`.
  Do not lower coverage thresholds to spend a partial holdout.
- `unsupported_features`: refresh/rebuild the indicated indicator files. The
  generated behavior is retired rather than left pending forever.
- `already_evaluated_on_snapshot`: expected deduplication, not a failure.
- `holdout_already_consumed`: expected fail-closed behavior after a prior claim
  or crash. Never delete the claim and retry.
- zero keepers: normal evidence. Inspect development rejection reasons and let
  later generations adapt; do not weaken gates merely to force an export.
- OpenClaw unavailable or invalid: no recovery is needed for the core loop;
  native grammar generation continues.

The daily backup uses SQLite's online backup API, deeply verifies the snapshot,
and stores `runtime/research/experiment_memory.backup.sqlite3` in the recovery
archive. Archives are forced to `0600`; restore staging directories and files
are forced to `0700`/`0600` regardless of the caller's umask. Verify and
transfer backups off-host:

```bash
make backup
BACKUP=$(find runtime/backups -type f -name 'autopilot_state_*.zip' | sort | tail -n 1)
make backup-verify BACKUP="$BACKUP"
```

If experiment memory is corrupt or lost, stop the job worker first. Preserve
the failed database, restore a verified archive into a separate directory, copy
its validated snapshot into the live memory path, and validate before restart:

```bash
systemctl --user stop trading-bot-autopilot-jobs.service
RESTORE_DIR="/tmp/trading-bot-research-restore-$(date +%s)"
make backup-restore BACKUP="$BACKUP" RESTORE_DIR="$RESTORE_DIR"

if [ -f runtime/research/experiment_memory.sqlite3 ]; then
  cp -p runtime/research/experiment_memory.sqlite3 \
    "runtime/research/experiment_memory.corrupt.$(date +%s).sqlite3"
fi
cp "$RESTORE_DIR/runtime/research/experiment_memory.backup.sqlite3" \
  runtime/research/.experiment_memory.restore.sqlite3
chmod 600 runtime/research/.experiment_memory.restore.sqlite3
mv runtime/research/.experiment_memory.restore.sqlite3 \
  runtime/research/experiment_memory.sqlite3
make research-factory-validate
make healthcheck
systemctl --user start trading-bot-autopilot-jobs.service
```

Do not initialize an empty database over the same historical data when no
trusted backup exists: that would forget duplicate experiments and holdout
exposure. Keep research paused and require an operator recovery decision with a
new unseen-data plan.

## Safe tuning

Change generation scope only in `config/research_factory.json`, then run:

```bash
make research-factory-validate
make research-history-plan
make research-history-bootstrap
make research-smoke
make research-once
make report
make healthcheck
```

Keep resource budgets appropriate for the server and preserve the exploration
floor. Do not bypass the typed compiler, canonical memory, data coverage,
multiple-testing adjustment, durable holdout claim, paper evidence, or human
approval boundary. Expanding the grammar increases what can be tested; it does
not establish that any resulting strategy is profitable.
