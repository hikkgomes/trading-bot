# Autonomous Crypto Trading System

Safety-first research, paper trading, and execution framework for a light Linux
server. The repo has two separate products:

1. **BTC accumulation**: conservative, BTC-denominated, spot-only. It may trade
   BTC/USDT to increase total BTC holdings by stepping aside from a bounded
   slice of existing BTC and rebuying with the same quote proceeds. No leverage,
   no futures, and no quote-funded long BTC strategies.
2. **Active income**: USDT-denominated day/swing/scalp research and paper trading
   for Binance USDT futures. Risk is capped per trade and per day.

The mandatory production rule is enforced in code:

**No new or changed strategy artifact can enter live mode unless every strategy
fingerprint is present in `runtime/approvals.json`.**

Environment flags such as `TRADING_LIVE=1` are not enough. The approval ledger is
checked separately by `src.autopilot.approvals`. Live products also require a
fresh automated preflight report by default, so approval alone cannot bypass
environment and broker readiness checks.

## Architecture

| Layer | Modules | Purpose |
|---|---|---|
| Data | `build_binance_indicator_dataset.py`, `src/update_candles.py`, `src/load_data.py`, `src/build_dataset.py` | Refresh Binance candles and derived features. |
| Research | `research_exploration/`, `src/strategy_search.py`, `src/day_trade_search.py`, `src/sweep.py`, `src/walk_forward.py` | Generate hypotheses, test strategies, run walk-forward validation, holdout gates, and comparison sweeps. |
| Strategy contract | `src/export_strategies.py`, `outputs/active_strategies*.json` | Converts validated research results into execution artifacts. Holdout is a hard gate by default. |
| Runtime | `src/autopilot/` | Lightweight 24/7 orchestration, file-based pause control, status reporting, and live approval enforcement. |
| Execution | `src/run_bot.py`, `src/execution/` | Closed-candle paper execution by default; approved active-income live mode can route orders through the ccxt broker behind safety rails. |

Generated datasets, search results, runtime state, and trade logs are ignored by
git. The repo should stay small; regenerate artifacts when needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

For a small Linux execution server:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-bot.txt
```

Copy `.env.example` to `.env` for exchange credentials. Keep `TRADING_LIVE=0`
until the approval ledger and live adapter path are deliberately enabled.

## Autopilot

Default config is `config/autopilot.json`. It defines:

| Product | Base asset | Market | Default mode |
|---|---:|---|---|
| `btc_accumulation` | BTC | spot | paper |
| `active_income` | USDT | futures | paper |

Config validation fails closed if BTC accumulation is not BTC/spot, if active
income is not USDT/futures, if a product uses an unknown objective or market, or
if products share runtime-owned strategy, state, trade-log, preflight, or required
testnet-rehearsal evidence paths. Scheduled jobs are command lists, never shell
strings, and validation rejects duplicate job output/report paths so autonomous
jobs cannot silently overwrite each other's evidence. Config booleans must be
JSON booleans (`true` or `false`), and numeric fields such as cadences,
freshness windows, disk limits, and starting equity must be JSON numbers, not
quoted strings or boolean stand-ins. Product routing, runtime paths, and
environment-variable names must be non-empty strings, so hand-edited safety gates
fail closed on typos. Unknown top-level, product, or scheduled-job config keys
are rejected instead of being silently ignored, and duplicate JSON keys are
rejected instead of letting the last value silently win. Non-standard JSON
constants such as `NaN` and `Infinity` are also rejected.
Production runtime and readiness validation also require both core products,
`btc_accumulation` and `active_income`, to be present and enabled, and require
the core autonomous jobs for data updates, regime tagging, research, mutation
testing, promotion review, maintenance, backup, and artifact hygiene to be
present, enabled, pointed at their expected Python modules, and wired with the
expected market/product/reporting arguments.
Scheduled-job execution is capped by `max_jobs_per_cycle` (`1` by default) so a
backlog of data, research, backup, and hygiene jobs cannot monopolize a light
server cycle before product supervision runs. The scheduler rotates the starting
job after each execution, so a backlog drains over multiple cycles instead of
always favoring the first configured job. Deferred due jobs remain due for a later
cycle, are persisted in job state as `cycle_job_limit`, and are surfaced as
healthcheck warnings. A job that reaches `max_consecutive_job_deferrals` (`3` by
default) becomes a healthcheck failure so scheduler starvation cannot stay silent.

Commands:

```bash
make autopilot-validate
make bootstrap-strategies
make readiness
make autopilot-once
make report
make healthcheck
make backup
make backup-verify
make backup-restore RESTORE_DIR=/tmp/trading-bot-restore
make maintenance
make data-update
make data-update-1m-flow
make regime-tag-futures
make strategy-smoke
make research-smoke
python -m src.autopilot.runtime --config config/autopilot.json
```

For 24/7 operation on Linux with user-level systemd:

```bash
make service-dry-run
DRY_RUN=1 bash scripts/install_autopilot_service.sh
bash scripts/install_autopilot_service.sh
```

The installer runs `src.autopilot.runtime --validate` and
`src.autopilot.readiness` before enabling the unit, and the generated systemd
service repeats both checks in `ExecStartPre` before every start. Readiness
blockers stop the service from starting instead of letting a bad live
configuration churn through failed cycles. It also installs a companion
`trading-bot-autopilot-healthcheck.timer`, which runs the machine-readable
healthcheck every five minutes. Blocking healthcheck issues emit the configured
critical alert through the same JSONL/webhook channel as runtime alerts, with
cooldown applied. The units load `$REPO/.env`, write stdout/stderr to the user
journal, restart the supervisor after any process exit, rate-limit restart
storms, and use a restrictive umask plus lightweight sandboxing options. They
also cap common BLAS/OpenMP/joblib thread pools to
`AUTOPILOT_THREADS=2` by default; override that environment variable when
installing if the server has more or fewer cores available. The generated
units also set conservative cgroup limits by default:
`AUTOPILOT_MEMORY_MAX=1G`, `AUTOPILOT_CPU_QUOTA=75%`, and
`AUTOPILOT_TASKS_MAX=128`; override them during install if your server budget is
different. The generated units also enable additional systemd sandboxing
(`PrivateDevices`, kernel/control-group protections, SUID/SGID restrictions, and
locked personality) while still allowing network access and runtime file writes.
Custom service and timer names supplied through environment variables are
validated as plain `.service` or `.timer` unit names before any unit files are
written, and raw resource/timer overrides are rejected if they are empty or
contain control characters. `DRY_RUN` must be exactly `0` or `1`, so ambiguous
values such as `true` fail before the installer can touch live user units.
`make service-dry-run` writes the generated unit files under
`runtime/systemd-dry-run` for inspection instead of touching the live user
systemd directory.
Operate it with:

```bash
systemctl --user status trading-bot-autopilot.service --no-pager
systemctl --user list-timers trading-bot-autopilot-healthcheck.timer --no-pager
journalctl --user -u trading-bot-autopilot.service -f
journalctl --user -u trading-bot-autopilot-healthcheck.service -f
systemctl --user restart trading-bot-autopilot.service
```

The runtime writes status to `runtime/status.json` and takes an exclusive
`runtime/autopilot.lock` while running so accidental duplicate supervisors cannot
evaluate or trade the same products concurrently. Duplicate startup attempts do
not release the existing holder's lock; the file content is only holder metadata
for inspection. Pause control is a tiny JSON file at `runtime/control.json`, with
an atomic CLI wrapper for safer operation:

```bash
make control ARGS="status"
make control ARGS="pause --reason maintenance"
make control ARGS="resume --reason done"
```

Mutating default control commands append durable JSONL audit events to
`runtime/control_audit.jsonl`, including the command, reason, operator, and
before/after control payload. For alternate control files, pass
`--audit path/to/control_audit.jsonl` before the command. If the audit append
fails after the control file is written, the command still prints the applied
control state with `audit_error` so emergency pause/flatten requests are not
lost.

If the control file is malformed, not a JSON object, uses invalid selector
lists, or has typoed boolean controls such as `paused`, `pause_jobs`, or
`flatten_all`, the runtime fails closed: products and scheduled jobs are paused,
the cycle is marked failed, and the control parse error is written into
status/reporting. The CLI can repair a bad file with
`make control ARGS="clear --reason repaired"`; when a mutating control command
recovers from malformed existing state, its JSON output and audit event include a
transient `recovered_control_error` while the repaired control file itself stays
clean. The default
`make control` path validates product/job names against `config/autopilot.json`
before writing, and `make control ARGS="status"` prints selector validation for
the current file. Unknown product or job names in manually edited
`paused_products`, `paused_jobs`, or `flatten_products` still fail closed at
runtime so typoed emergency controls are not ignored silently. `make readiness`
also parses the current control file and blocks service preflight when it is
malformed or references unknown products/jobs.

Pause one product:

```bash
make control ARGS="pause-product active_income --reason 'risk review'"
make control ARGS="resume-product active_income --reason done"
```

Pause one autonomous job without pausing trading supervision:

```bash
make control ARGS="pause-job market_data_update_futures --reason 'network maintenance'"
make control ARGS="resume-job market_data_update_futures --reason done"
```

or pause all scheduled jobs while products continue:

```bash
make control ARGS="pause-jobs --reason 'server load'"
make control ARGS="resume-jobs --reason done"
```

To emergency-close live exposure for a product, request flatten. Futures flatten
runs even while paused, uses the broker reduce-only close path, and clears local
open-position state only after the broker is flat. If the local state file is
corrupt after the broker is flat, the runtime writes a minimal recovered state
with a `last_flatten` audit marker instead of leaving the account unreconciled.
BTC spot accumulation is never flattened by selling the BTC base stack; when
local state contains one tracked spot step-aside position, flatten buys BTC back
with the recorded quote-reinvest budget and clears state only after a full buy
fill is accepted:

```bash
make control ARGS="panic --reason 'exchange incident'"
make control ARGS="pause --reason 'emergency close'"
make control ARGS="flatten active_income --reason 'emergency close'"
make control ARGS="clear-flatten active_income --reason done"
```

`panic` is the one-command fail-safe: it pauses all products, pauses scheduled
jobs, and requests flatten for every live product in one audited atomic
control-file write. BTC spot flatten fails closed without placing an order when
local state is unreadable, has multiple open positions, lacks broker metadata, or
does not contain a quote-reinvest step-aside position. A spot product with no
local step-aside position is reported as skipped and left untouched.

Manual JSON edits are still supported for dependency-free recovery; the CLI
writes the same schema atomically.

If a paper product has no exported strategy artifact yet, the runtime reports it
as `waiting_for_strategy_artifact` instead of failing the whole service. Live
products still fail closed until their strategy artifact exists, is explicitly
approved, and has a fresh matching preflight report.

For a fresh server, `make bootstrap-strategies` writes deterministic paper-only
artifacts for missing paper products. These probes are marked
`paper_trade_allowed: true`, `live_allowed: false`, and
`promotion_eligible: false`; they can exercise data, sizing, state, and paper
order flow while research looks for validated edges, but the approval and live
policy paths reject them.

Product symbols are policy-checked at startup. The current system is scoped to
BTC/USDT data and execution: `btc_accumulation` must be spot BTC/USDT, while
`active_income` must be BTC/USDT on USDT-margined futures. Binance compact
symbols (`BTCUSDT`) and ccxt forms (`BTC/USDT`, `BTC/USDT:USDT`) are accepted
where they describe the same instrument; data fetches use compact Binance REST
symbols and ccxt broker calls are normalized internally.

Before any artifact is executed, the runtime also applies a product-aware
strategy policy: positive finite holdout evidence, bounded finite risk per
trade, `max_position_fraction`, daily stop, per-strategy daily trade cap,
consecutive-loss limit, cooldown, sane finite TP/SL and fee inputs, positive
integer horizons, valid condition or hypothesis-entry payloads, and the correct
BTC/USDT PnL unit for the product. Explicit paper-only bootstrap artifacts may
omit performance metrics only while `live_allowed` and `promotion_eligible` are
both `false`; ordinary paper, promotion, and live-eligible artifacts still need
the same positive evidence gates. Promotion/live review also requires
`paper_trade_allowed`, `live_allowed`, and `promotion_eligible` to be
explicitly `true`; missing flags are treated as not live-eligible. Runtime
execution caps stop-distance sizing
by `max_position_fraction` and also prevents
multi-strategy artifacts from stacking simultaneous positions on the same
product/symbol. Active-income artifacts must show finite deflated-Sharpe
evidence (`dsr_deflated` or legacy `dsr`) of at least `0.60`; DSR-near-zero flow
candidates are treated as false positives and blocked before paper/live
execution. BTC-accumulation artifacts must also show positive
finite `holdout_excess_return_vs_buy_hold`, so a strategy is judged by extra BTC
accumulated versus simply holding BTC. Active-income artifacts may risk at most
`0.5%` per trade, `25%` notional exposure, `3%` daily loss, and `8` entries per
strategy per day; BTC-accumulation artifacts may risk at most `0.3%` per trade,
`35%` step-aside exposure, `1%` daily BTC drawdown, and `2` entries per strategy
per day. Older ignored artifacts that no longer pass these rules are blocked
even in paper mode. The same basic executable artifact checks run inside
`src.run_bot` for direct manual invocations. If an artifact or individual
strategy declares `market` or `symbol`, direct bot runs require those fields to
match the configured bot market and symbol. New exports stamp both fields at the
artifact and strategy level. Direct manual product runs
should pass `--objective active_income --base-asset USDT` or
`--objective btc_accumulation --base-asset BTC` so product-specific DSR,
holdout, and step-aside-only guards are enabled outside the autopilot wrapper.
When the BTC accumulation regime guard is enabled, daily macro-regime data
failures block new entries for that cycle instead of assuming risk-on.
Persisted bot state is validated on load as well: equity must be finite and
positive, cooldown timestamps and loss/trade counters must be finite and
non-negative, daily trade counters must be integers, and open-position records
must reference known strategies with valid timestamps, directions, prices,
position sizes, and broker metadata.
Malformed strategy artifacts, including non-object JSON or non-object strategy
entries, are blocked by policy, approval, preflight, and live gates.

Autonomous data, maintenance, backup, and research jobs live in the same config
under `jobs`. They are command arrays, not shell strings, and each has a cadence
and timeout. The default config enables six-hour incremental market-data updates,
daily runtime maintenance, daily runtime-state backups, and a daily synthetic
research smoke check for both products. Data updates are split by market and use `--skip-if-missing`, so a
fresh server can bootstrap a bounded recent seed with `--bootstrap-days` instead
of trying to download years of history. The updater validates 1m candle
timestamps, OHLC consistency, and volume/trade fields before fetched or merged
data can be written or used to rebuild indicators. Market-data readiness also
rejects candle datasets whose newest 1m candle is timestamped in the future.
Expensive real-data research examples stay disabled so a fresh server will not
launch costly work by accident:

```json
{
  "name": "market_data_update_futures",
  "enabled": true,
  "command": [
    ".venv/bin/python", "-m", "src.update_candles",
    "--market", "futures",
    "--bootstrap-days", "90",
    "--skip-if-missing",
    "--timeframes", "5m", "15m", "30m", "1h", "4h", "1d"
  ],
  "cadence_seconds": 21600,
  "timeout_seconds": 1800
}
```

Job state is persisted in `runtime/job_state.json` and recent stdout/stderr tails
are included in `runtime/status.json`. Malformed job state is surfaced in
operator reports and healthcheck as a runtime file issue, and non-object
job-state payloads fail the scheduler closed with a clear error; malformed or
future per-job timestamps are treated as due, surfaced as healthcheck issues for
enabled jobs, and repaired by the next successful run. Malformed job counters and
durations are sanitized in reports and shown as healthcheck job-state issues.
Jobs may log progress to stdout and then print a final JSON object; the scheduler
captures that final structured status for reports, failure reasons, and backoff
state, including pretty-printed JSON objects after progress logs. Large
structured reports are summarized before being stored in runtime status, but list
counts and the first few `errors` are retained so failed maintenance or hygiene
jobs keep actionable failure detail.
`make autopilot-validate` verifies that job working directories and executables
exist, rejects shell-wrapper jobs, and checks configured Python `-m` modules are
importable before the service starts.
Failed jobs retry sooner than their normal cadence, then back off exponentially
up to six hours as consecutive failures accumulate, so a broken data or research
command does not spin a light server.
`make research-smoke` runs the same cheap synthetic validation job on the
active-income and BTC-accumulation hypothesis families and writes
`runtime/research_smoke.json`. It is a wiring check, not evidence of a live edge.
`make strategy-smoke` runs lightweight representative `src.sweep` checks over
synthetic data and, when available, the latest regime-tagged futures parquet; it
writes `runtime/strategy_framework_smoke.json` and is also scheduled by the
autopilot as `strategy_framework_smoke`.
`make research-cycle` runs the bounded real-data research loop used by the
autopilot: validate the configured active-income and BTC-accumulation candidate
batches, append results to `outputs/research_exploration/experiment_log.jsonl`,
and export only candidates that already passed the positive-holdout gate. If no
candidate qualifies, the cycle exits successfully and records
`no_exportable_strategies` in `runtime/research_cycle.json`. Active-income
research rotates through deterministic slices of the full curated batch,
including a recent 1m scalping slice, so the server expands coverage over time
without launching an unbounded grid search. Deflated Sharpe accounting still
pays for the full available scenario universe, not only the small slice validated
in the current cycle, so rotating slices does not quietly weaken the
multiple-testing penalty. If a candidate needs a feature that is missing from the
current local parquets, it is listed under `unsupported_hypotheses` and the rest
of the slice still runs. Rejected-but-informative candidates are also written to
`runtime/incubation_candidates.json` as a non-executable research queue; that
file is explicitly blocked from paper trading, promotion, and live execution.
The lightweight strategy framework can also triage ML ideas with
triple-barrier/capped-return targets and Spearman feature screening through
`ml_classifier` and `ml_regressor`; heavier meta-labeling and purged walk-forward
model-signal tooling remain under `src/meta_labeling.py` and
`src/model_signals.py`.
For regime-specific variants, tag a dataset with `python -m src.regime` and
sweep `regime_filter` across `strategy` and `regime_ids`; it masks child
strategy signals outside the selected market states while keeping the child
strategy's TP/SL/horizon defaults. The autopilot also runs
`regime_tag_futures_15m`, which writes `runtime/regime/futures_15m_regime.parquet`
and `runtime/regime_tag_futures_15m.json` from the market-aware 15m and 1d
indicator files when they exist.
The normal `make data-update` target skips 1m indicator rebuilds because they
are the slow path. If reports show missing 1m scalping flow features such as
`volume_z_20`, run `make data-update-1m-flow` during a maintenance window.

Alerts are enabled by default and written to `runtime/alerts.jsonl` with a
fingerprint cooldown stored in `runtime/alert_state.json`. Set
`AUTOPILOT_WEBHOOK_URL` to send the same alert payload to an HTTP webhook.
Webhook delivery is best-effort: failures are recorded in the local alert JSONL
but do not crash trading supervision or bypass the alert cooldown. If local alert
file or cooldown writes fail after readiness has passed, the runtime records that
alert error in status and keeps supervising products. If the JSONL alert is
written but cooldown-state persistence fails, the alert result still reports
`sent: true` with `state_error`, and operator reports plus healthcheck warning
details surface that field, so operators know the alert was durable but may
repeat until state storage is fixed. Runtime alerts cover failed
cycles, market-data/feature/disk-space readiness warnings, research cycles that
keep generating hypotheses without exportable candidates while paper products
wait for artifacts, missing/stale required testnet rehearsals, and promotion
reviews where an already-approved strategy later fails the configured
paper-review thresholds.
Failed-cycle alerts include product mode, market, cycle errors, and local
`state_errors` so a paging payload can distinguish paper faults from live faults.
If the cooldown state file is malformed, alerting recovers to a fresh cooldown
map and records the recovery reason in the emitted alert payload.
Alert cooldown fingerprints ignore volatile fields such as timestamps and
free-byte counters, while the full detail is still written to
`runtime/alerts.jsonl`.
`make maintenance` runs bounded housekeeping: it compacts `runtime/alerts.jsonl`
to the most recent 1000 records, prunes `runtime/alert_state.json` to the most
recent 1000 cooldown fingerprints, and rotates older
`outputs/research_exploration/experiment_log.jsonl` and
`runtime/control_audit.jsonl` lines into compressed archives while keeping each
hot log at 5000 records. It does not touch trade logs, approvals, strategy
artifacts, or `data/`, and it refuses symlinked log/state inputs instead of
following them during unattended compaction. Independent maintenance tasks keep
running after one task fails; the command prints `ok: false` with per-task
`errors` and exits nonzero so the scheduler records the failed housekeeping
evidence without losing successful task results. To deliberately prune old files under
`runtime/quarantine`, pass a byte budget, for example
`make maintenance QUARANTINE_BYTES=1073741824`; it deletes the oldest generated
quarantine files until the directory is under that budget.
`make artifact-hygiene` writes a dry-run report of configured artifacts,
policy-blocked active artifacts, unreferenced `active_strategies*.json` files,
and historical search-output directories. Add `APPLY=1` only when you want to
move policy-blocked paper artifacts into `runtime/quarantine`; add
`UNREFERENCED=1` as well to also move unreferenced active-strategy artifacts.
It never deletes files and it will not quarantine live-product artifacts. If one
artifact move fails, the report keeps inspecting the rest, records per-artifact
`errors`, writes `ok: false`, and exits nonzero so the scheduled job preserves
the failure evidence. Operator reports summarize the first hygiene error, and
healthcheck surfaces failed hygiene reports as warnings.
`make healthcheck` writes `runtime/healthcheck.json` and exits nonzero if the
last status heartbeat is stale, missing, or timestamped in the future, the latest
cycle failed, market-data status is missing, stale, invalid, or timestamped in
the future, readiness has blocking failures, an enabled scheduled job is still
marked failing, an enabled job is more than two effective cadences overdue, an
enabled scheduled job reports invalid scheduler state, or the latest verified
backup is stale. Live open positions also fail healthcheck when their
counts/detail rows, risk metadata, or stale-position monitoring metadata are
missing, invalid, or timestamped in the future.
Paper open-position visibility, risk, and monitoring metadata gaps are
warning-level so paper execution issues remain visible without failing the
watchdog.
Invalid trade-log numeric audit fields are surfaced the same way: live product
trade-log corruption is a healthcheck issue, while paper product trade-log
corruption is a warning.
Warning-level readiness checks, such as optional paper-mode `.env` absence or
approval-ledger actor audit warnings, are also copied into healthcheck warnings
without changing the exit code.
Runtime report-refresh failures are also warning-level in healthcheck: they
include the failed report path and any successfully refreshed outputs, but do not
fail the trading watchdog unless the cycle itself failed.
Malformed product state snapshots fail the product cycle, are preserved in
operator reports under `state_errors`, shown in the product table issue column,
and are surfaced directly by healthcheck: live state errors are issues, while
paper state errors are warnings. Malformed open-position state is also
classified by the live/paper visibility checks above.
Backup staleness defaults to twice the enabled backup job cadence, and verified
backup reports timestamped in the future also fail healthcheck instead of being
treated as fresh. Use `--max-backup-age-hours` for stricter external watchdogs. Use
`--ignore-job-overdue` only when an external scheduler intentionally owns job
cadence. Enabled scheduled jobs that are due but have never run are healthcheck
issues too; this catches a supervisor that starts but never launches its
background data, research, or maintenance jobs. Warning-level alerts such as
readiness warnings, stale/unsafe/failed research handoffs, promotion-review
warnings, enabled paper products waiting for an exported strategy artifact,
research cycles that found no exportable candidates, or a missing/stale required
testnet rehearsal for a paper product are also included in the JSON under
`warnings` without changing the exit code. A
missing/stale rehearsal required by a live product is a healthcheck issue and
exits nonzero. Blocking healthcheck issues also emit a cooldown-controlled
`autopilot healthcheck failed` alert unless you pass `--no-alert`; if that alert
write fails, the error is kept in `healthcheck_alert` so the original healthcheck
JSON still prints and writes. Config/report build failures print
`healthcheck_build_failed`; if the output file itself cannot be written, the
command still prints JSON with `healthcheck_output_write_failed` and exits
nonzero. It is intended for a cron, systemd timer, or external uptime monitor
that can alert when the main supervisor has stopped or a background workflow is
stuck.
`make backup` creates a small timestamped zip under `runtime/backups/` and writes
`runtime/backup_report.json`. It includes the autopilot config, approvals,
control/audit files, product state, trade logs, active strategy artifacts, and
runtime reports, including the non-executable incubation queue, with a per-file
size cap so market datasets and large research outputs are not pulled into
recovery bundles. Symlink sources are skipped instead of followed, so a linked
runtime path cannot pull unintended target contents into the archive, and
symlink backup output paths are refused before writing. It does not include
`.env` or API credentials. The default scheduled backup job keeps the
latest 30 generated backup zips.
`make backup-verify` validates the latest backup zip against its manifest; pass
`BACKUP=runtime/backups/name.zip` to verify a specific archive. Verification
fails closed for unsupported manifest versions, so restore will not accept a
backup written with an unknown archive schema.
`make backup-restore RESTORE_DIR=/tmp/trading-bot-restore` verifies a backup and
extracts it into a separate directory without overwriting existing files, so you
can rehearse recovery before copying state back into a live runtime directory.
Restore refuses archive path traversal, symlink restore roots, and symlink
escapes from the restore directory, including when overwrite mode is enabled.
The operator report summarizes the latest backup report, and `make healthcheck`
fails if an existing backup report says creation or verification failed, or if
the last verified backup is older than the configured freshness window.
When `auto_report_enabled` is true, every runtime cycle refreshes
`runtime/operator_report.md/json` and `runtime/readiness_report.md/json` after
writing the latest status. Report-rendering errors are recorded in status but do
not stop trading supervision; partial runtime report output-write failures are
recorded with the affected path and successful outputs continue to refresh.
Scheduled job subprocess output is captured through temporary files with bounded
in-memory reads; runtime status keeps short stdout/stderr tails and truncation
metadata instead of loading unbounded logs into the supervisor. Truncation flags
and byte counts are also persisted in `runtime/job_state.json` and shown in the
operator report; `make healthcheck` surfaces them as warning-level watchdog
events.
`make readiness` and `make report` run the same renderers manually. If readiness
cannot load config, build its report, or write one of its output files, it exits
nonzero with a readiness-shaped failure report.
If the operator report CLI cannot load config or build the report, it exits
nonzero with a compact operator failure report; if an output write fails, it
prints the structured report JSON to stdout.
Readiness checks local server paths including alert log and cooldown-state
writability, config, artifacts, approval/preflight requirements for live
products, environment switches, 1m candle freshness, required indicator feature
columns, and runtime filesystem free space using
`min_runtime_free_bytes` from `config/autopilot.json`. Existing
approval ledgers must be readable JSON objects; malformed ledgers block
readiness because they would also block live execution. When
`src.regime` jobs are configured, readiness also warns if regime-tagged research
data is missing or skipped. It also warns when the strategy-framework smoke
report is missing or failing. The service-installer readiness check verifies the
script still has strict shell mode, config validation, readiness pre-start,
healthcheck timer installation, and fail-closed unit override validation. The
operator report summarizes status heartbeat age, control state, market-data
freshness, indicator feature readiness, regime-data rows/counts,
strategy-framework smoke status, synthetic and real-data research status,
approval ledger health including revoked fingerprints and the latest approval
event, configured scheduled-job state, latest-cycle job output, and product
trade logs.

## Live Approval Gate

Approve strategies only after reviewing the exported artifact and validation
report. Use a non-empty human operator identifier for `--approved-by` and
`--revoked-by`, include `--confirm-live` when approving, and include a non-empty
`--reason` when revoking. Blank approval actors are rejected and existing
blank-actor approvals do not satisfy the live gate. Obvious automation
identities such as `autopilot`, `system`, `cron`, and `github-actions[bot]` are
also rejected for approval/revocation actor fields and are treated as invalid if
found in an existing ledger. Revocation reasons are free-form required text:

```bash
python -m src.autopilot.approvals approve \
  --config config/autopilot.json \
  --product active_income \
  --artifact outputs/active_strategies_flow.json \
  --all \
  --approved-by henrique \
  --confirm-live \
  --notes "Reviewed holdout, paper stats, and risk limits"

python -m src.autopilot.approvals check \
  --config config/autopilot.json \
  --product active_income \
  --artifact outputs/active_strategies_flow.json

python -m src.autopilot.approvals list

python -m src.autopilot.approvals revoke \
  --fingerprint sha256:<fingerprint> \
  --revoked-by henrique \
  --reason "paper drawdown breached"
```

`list` shows the relevant actor for the current state: `approved_by` for active
approvals and `revoked_by` plus the revocation reason for revoked fingerprints.

Changing a strategy’s behavior changes its fingerprint and invalidates the old
approval, including entry/exit logic, risk, fees, market/symbol routing, and the
drift baseline used by the kill switch. Strategy-level `leverage` and
`margin_mode`, when present, are also behavior fields; policy rejects leverage
above `1`, spot margin metadata, and non-isolated futures margin. Metrics-only
changes do not affect the fingerprint. Live checks also require the approval to
match the current artifact path and product metadata, and then re-apply the
product-aware strategy policy, so an approval for one product, symbol, copied
artifact, or now-policy-failing artifact cannot be reused silently. Revoked
fingerprints are retained in the ledger for auditability and block live
execution. Each approval entry must also carry the same fingerprint as its ledger
key; missing or mismatched embedded fingerprints are treated as invalid approval
evidence. A malformed approval ledger or malformed approval entry also blocks
live execution rather than being ignored.

## Live/Testnet Preflight

Before enabling a live/testnet product, run the preflight. It never places
orders. It checks product config, artifact existence, approval status, exchange
environment, broker construction, and read-only exchange access.
Config validation and normal live execution both require `require_preflight=true`,
a configured `preflight_report`, and a fresh passing report that points at the
current product and strategy artifact. The saved report must include successful
check entries for config, strategy artifact, policy, approval, exchange
environment, broker construction, read-only connectivity, and the product-specific
position check (`broker_position_flat` for active-income futures or
`broker_spot_position_non_negative` for BTC accumulation spot).
The live gate also validates the saved read-only connectivity evidence itself:
price and balance must be finite and positive, position quantity must be finite,
position average price must be finite and non-negative, and active-income
connectivity evidence must show a flat futures position.
Preflight reports are also bound to the exact strategy fingerprints inside the
artifact, so editing a strategy after preflight forces a fresh preflight before
live execution. Active-income live execution also always requires
`require_testnet_rehearsal=true` and a recent successful testnet rehearsal.
Malformed preflight reports block readiness and live execution instead of being
treated as missing approval evidence.
If the preflight CLI cannot load config, build the report, or write the output
file, it exits nonzero with structured JSON so automation can record the failed
gate instead of losing the reason in a traceback.
Separately, every live cycle still checks `TRADING_LIVE=1`, exchange
credentials, a positive `MAX_NOTIONAL_USD`, a positive
`MAX_FILL_SLIPPAGE_BPS`, and `MAX_FUTURES_LEVERAGE=1` for the active-income
futures product before constructing a broker.

```bash
make preflight PRODUCT=active_income

python -m src.autopilot.preflight \
  --config config/autopilot.json \
  --product btc_accumulation \
  --assume-live \
  --connect \
  --output runtime/btc_accumulation_preflight_report.json
```

The Makefile preflight target always passes `--connect`, which only fetches
ticker, balance, and current position. For
`active_income` futures, a connected preflight also requires the broker position
to be flat before live/testnet enablement; use the flatten control or reconcile
manually if it is not. BTC accumulation spot preflight allows existing BTC
holdings because the product starts from the BTC base stack. Use testnet keys
first, with `TRADING_LIVE=1`, `EXCHANGE_TESTNET=1`, and a tiny
`MAX_NOTIONAL_USD`. Keep `MAX_FILL_SLIPPAGE_BPS` tight enough to catch bad
fills, and keep `MAX_FUTURES_LEVERAGE=1`; active-income autopilot gates reject
higher leverage.

After an `active_income` artifact has been explicitly approved and preflight is
green, the next exchange-facing step is a tiny futures testnet rehearsal:

```bash
make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5
make testnet-status
```

This command is intentionally separate from preflight because it places testnet
orders. It requires `TRADING_LIVE=1`, `EXCHANGE_TESTNET=1`, Binance USDT-M
futures routing, approval, successful preflight checks, a flat starting futures
position, and `NOTIONAL_USD <= MAX_NOTIONAL_USD`. It buys a tiny BTCUSDT futures
quantity and immediately closes it with the broker close path, then writes
`runtime/testnet_rehearsal_report.json`. `make testnet-status` reads that saved
artifact without placing orders and prints the same structural status used by the
live gate. The operator report summarizes the file as `Testnet rehearsal`,
including missing, malformed, failed, stale, or successful status. A usable
report must have a finite timestamp, positive notional, positive order quantity,
product-symbol-matched buy entry fill, product-symbol-matched sell close fill,
entry/close quantities matching the order quantity, testnet routing, and a flat
final position. If an entry fills but the normal close/readback step fails, the command
attempts one best-effort cleanup close and records the recovery result, but the
rehearsal still fails and must be rerun cleanly before live execution. Rehearsal
CLI failures are reported as structured JSON, including
config-load or output-write failures. Missing, malformed, stale, or failed
rehearsal warnings include the next preflight/rehearsal commands and required
testnet environment checklist, plus the read-only `make testnet-status`
inspection command. The default active-income config also
requires a recent successful
testnet rehearsal report before any live cycle can construct a broker. The live
gate compares the rehearsal's embedded preflight product metadata, artifact path,
fingerprints, and the recorded testnet exchange with the current product,
strategy artifact, and Binance USDT-M policy, so a changed strategy or routing
config must be approved, preflighted, and rehearsed again before it can reach live
execution. Malformed embedded preflight payloads also invalidate the rehearsal.

## Promotion Review

The system can prepare review packets, but it never approves strategies by
itself. A promotion review combines the strategy artifact, validation metrics,
paper-trade results, paper drawdown/loss-streak evidence, approval status, and
the exact approval command:

```bash
make promotion-review \
  PRODUCT=active_income \
  ARTIFACT=outputs/active_strategies_flow.json \
  TRADE_LOG=runtime/active_income_trades.csv
```

This writes `runtime/promotion_review.json` and `runtime/promotion_review.md`.
Only run the printed approval command after reviewing the packet. When a product
is supplied or inferred from `config/autopilot.json`, the packet also applies
the same product-aware strategy policy as runtime and suppresses approval
commands for policy-failing artifacts. It also withholds approval commands when
paper evidence fails the configured trade-count, return, drawdown, consecutive
loss, minimum-duration, or trade-log return-field validity gates. Invalid review
thresholds, such as negative paper-return requirements or zero required paper
trades, are reported as threshold failures and also suppress approval commands.
If an
already-approved strategy later fails
those paper-review thresholds, the packet marks it `approved_review_failed` so
you can revoke the fingerprint before any further live use. If the approval
ledger is malformed, the packet records the ledger error and emits no approval
commands until the ledger is repaired.
Operator reports and healthcheck warnings treat stale, missing-timestamp, and
future-dated promotion review packets as not fresh, so approval review evidence
cannot be made to look current by a bad clock.

Per-product promotion-review jobs are enabled in `config/autopilot.json` so the
operator report stays current as paper evidence accumulates. The manual target is
still useful for ad hoc review packets.

## Offline Rehearsal

Run a deterministic, no-network rehearsal of the full safety workflow:

```bash
make rehearse
```

It creates synthetic strategy and paper-trade artifacts for both products under
`runtime/rehearsal`, produces promotion reviews, approves only a local rehearsal
ledger, and runs preflight through a fake read-only broker. This does not touch
real approvals, credentials, or exchange APIs. `make readiness` reports a
warning until this offline workflow rehearsal has produced a passing
`runtime/rehearsal/rehearsal_summary.json` for both products.

## Research Workflow

Use the existing research paths, but keep their output out of git:

```bash
# Strategy framework smoke checks
python -m src.run_backtest --list
python -m src.sweep --all --synthetic 8000

# Triage ideas with repeated-window robustness and DSR filters
python -m src.sweep --all --input data/processed/train_15m_indicators.parquet \
  --base-tf 15m --walk-forward-windows 6 --min-wf-pass-rate 0.5 --min-dsr 0.6 \
  --out outputs/sweep_15m_wf.csv

# Export only low-overlap representatives that pass filters, positive expectancy, and holdout
python -m src.export_strategies --search-dir outputs/search_dir \
  --output outputs/active_strategies_flow.json
```

When a search writes `ranked_strategies_clustered.csv`, export uses it by
default so highly overlapping candidates do not crowd the active artifact. Use
`--raw-ranked` only for deliberate inspection runs.
Validated exports explicitly stamp `paper_trade_allowed: true`,
`live_allowed: true`, and `promotion_eligible: true`; bootstrap probes and
research-only mutation/incubation artifacts stamp the opposite and are rejected
by promotion/live gates. Live review fails closed if any of the three
eligibility flags is missing instead of explicitly `true`. Mutation batches also
fail closed if a generated or manually edited mutation plan is marked executable,
paper-trade allowed, promotion allowed, or live allowed; unsafe individual
proposals are skipped before hypothesis generation. Unsafe or failed mutation
artifacts are surfaced as research handoff warnings in runtime alerts and
healthcheck.

For hypothesis-driven research, use `docs/RESEARCH_WORKFLOW.md`.
The autonomous path is `make research-cycle`; it uses the same staged validation
and export gates, but keeps the run bounded and skips repeated validation only
when the market-data readiness marker is unchanged. Changes in spot/futures
freshness, row counts, timestamps, or missing-data reasons trigger a fresh cycle
so recovered or degraded data feeds are not hidden by an old timestamp.

## Tests And Quality

```bash
make test
make lint-autopilot
```

Focused smoke checks:

```bash
python -m pytest tests/test_autopilot_approvals.py tests/test_autopilot_runtime.py
python -m pytest tests/test_execution.py tests/test_run_bot.py
```

## Current Safety Posture

Paper trading and research orchestration are the default runtime path. Approved
`active_income` live mode injects a futures ccxt broker into `src.run_bot`.
Approved `btc_accumulation` live mode injects a spot ccxt broker, so it cannot
route through futures or leverage. `src.run_bot` refuses a live broker unless
the autopilot path has already passed approval, preflight, and testnet rehearsal
gates. The bot also uses each product's configured
market for closed-candle data, so BTC accumulation evaluates spot candles while
active income evaluates futures candles.

Broker-routed execution sizes orders from broker balances, records broker fill
details, and reconciles local open positions against broker positions before
managing exits. BTC spot step-aside exits reinvest the original sell proceeds
into the buyback, so profitable sell-lower/buy-lower cycles can increase BTC
instead of leaving gains idle in USDT. The ccxt adapter still requires
`TRADING_LIVE=1` and enforces `MAX_NOTIONAL_USD` plus
`MAX_FILL_SLIPPAGE_BPS` for entries and position increases. Futures reduce-only
closes may exceed `MAX_NOTIONAL_USD` so an emergency flatten can reduce existing
exposure. For futures, it also sets `MAX_FUTURES_LEVERAGE` before opening
entries and refuses to open if the exchange adapter cannot set leverage.
