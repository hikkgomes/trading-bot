# Deployment and Operations Runbook

This is the authoritative runbook for a fresh light-Linux deployment. The
repository defaults to paper trading. Keep it that way until the promotion
sequence below has completed for one specific strategy artifact and a human has
approved it.

For the optional isolated Telegram and OpenClaw edges, follow
[`COMMUNICATIONS.md`](COMMUNICATIONS.md). Neither integration is required for
trading/research operation, and neither is an approval or execution channel.

## Safety model

- `btc_accumulation` trades BTC/USDT spot, uses BTC as its accounting asset, and
  never uses leverage.
- `active_income` trades BTC/USDT on Binance USDT-margined futures, uses USDT as
  its accounting asset, and is restricted to the `binanceusdm` API, isolated
  margin, one-way position mode (`positionSide=BOTH`), and 1x leverage.
- Environment variables alone cannot authorize live trading. Live entry also
  requires an eligible artifact, a matching human approval, fresh connected
  preflight evidence, and, for active income, a matching testnet rehearsal.
- Approval is bound to the exact artifact digest/path, strategy fingerprints,
  every canonical product field/path, an execution-engine digest covering
  Python, installed pinned dependencies and execution-capable source, and the
  stable account/venue/risk-cap manifest from a reviewed successful production
  preflight. Equivalent timestamp refreshes preserve approval; account, venue,
  market/testnet routing, quote asset, notional/slippage, leverage, or margin
  drift invalidates it. Runtime separately requires fresh connected evidence.
- Live positions and durable recovery/flatten intents retain that non-secret
  account fingerprint. Exit/recovery/flatten refuses a different configured
  account instead of sending a risk-reducing order to the wrong account.
- Immediately before a live order intent, the supervisor rechecks current
  panic/pause/flatten control, approval, production preflight, and environment.
  The futures broker then re-fetches the signed position immediately before
  `create_order` and requires flatness, closing races during feature evaluation.
- Every live futures entry requires a verified exchange-native reduce-only
  stop-market order. Protection is placed and fetched back before the entry
  intent is cleared; an unverifiable protection attempt triggers a deterministic
  reduce-only recovery close and a blocking durable incident. Native take-profit
  is not installed. Paper trading and BTC spot exits remain software-managed
  from closed candles, so a host, network, process, or exchange outage can leave
  those exposures unmanaged. Start live operation at tiny size and account for
  these boundaries.

## 1. Prepare a fresh server

Python 3.11 or newer and user-level systemd are required. Changing the Python
version changes the execution identity and therefore requires a deliberate
revalidation. On Debian/Ubuntu, install the distribution's Python 3 package and
verify it is new enough:

```bash
sudo apt-get update
sudo apt-get install -y acl git jq make python3 python3-venv rsync
python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version; print(sys.version)'
timedatectl status
test "$(timedatectl show --property=NTPSynchronized --value)" = "yes"
```

Binance signed requests depend on a correct system clock. If NTP is not
synchronized, enable the distribution's `systemd-timesyncd` or `chrony`
service and do not proceed to connected preflight until the check returns
`yes`.

Clone into the installer's default location:

```bash
REPOSITORY_URL="ssh://git.example/trading-bot.git"
git clone "$REPOSITORY_URL" "$HOME/trading-bot"
cd "$HOME/trading-bot"
git status --short
```

Replace `REPOSITORY_URL` with the real remote. Use `git clone`; do not copy a
workstation checkout. In particular, do not copy local `data/`, `runtime/`, or
`outputs/`. They are ignored, machine-specific, and can contain stale exchange
state or very large research data. Regenerate market data and paper artifacts on
the server. Restore runtime state only via the reviewed recovery procedure below.

Create the virtual environment and install the lightweight autopilot dependency
set. It covers live execution and every enabled default scheduled job; optional
heavy research packages are intentionally omitted.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pip==26.1.2
.venv/bin/pip install -r requirements-bot.txt
.venv/bin/pip check
.venv/bin/python -c 'import ccxt, numpy, pandas, pyarrow, scipy, sklearn, talib; print("dependencies OK")'
```

`requirements-bot.txt` is a fully pinned, lean lock for execution and every
enabled default job, including transitive networking and crypto dependencies.
In particular, `ccxt==4.5.64` is the validated Binance conditional Algo
Order/native-stop path; do not upgrade it independently. The live identity
hashes the complete installed package environment, so adding an unreviewed
package also invalidates approval and preflight evidence. If a
TA-Lib wheel is unavailable for the server architecture, install the
distribution's TA-Lib C library and build prerequisites, then repeat the pip
install. Do not silently omit TA-Lib.

## 2. Configure paper mode and credentials

```bash
cp .env.example .env
chmod 600 .env
test ! -L .env
test "$(stat -c '%a' .env)" = "600"
cp config/alerts.env.example runtime/alerts.env
chmod 600 runtime/alerts.env
```

Edit `.env` with one assignment per line and comments on their own lines. For
the initial paper deployment, retain:

```dotenv
TRADING_LIVE=0
EXCHANGE_TESTNET=1
FUTURES_MARGIN_MODE=isolated
MAX_FUTURES_LEVERAGE=1
```

Use exchange keys with trading/read permissions only: disable withdrawals and
restrict source IPs where Binance supports it. Keep `MAX_NOTIONAL_USD` tiny when
testing and set a deliberate `MAX_FILL_SLIPPAGE_BPS`. Configure the Binance
USDT-M account for one-way position mode; hedge mode is unsupported. `.env` is
ignored by git, must be owned by the trading-service user with no group/world
permissions, must not be a symlink, and is intentionally absent from runtime
backups. `make readiness` and the service installer reject an existing `.env`
that is not a regular, readable, owner-matched `0600` file.
`runtime/alerts.env` contains only optional webhook routing and the path to the
separate Telegram settings file. Never put exchange credentials in it.

## 3. Bootstrap and verify paper operation

Validate before downloading data or running the supervisor:

```bash
make autopilot-validate
make research-factory-validate
make bootstrap-strategies
make rehearse
```

`bootstrap-strategies` creates deterministic paper probes only. They are useful
for testing the pipeline but are ineligible for approval or live trading.

Inspect the generative search spaces' native-timeframe plan, then bootstrap it.
Both commands strictly load `config/research_factory.json`; every declared base,
trigger, setup, and regime timeframe is included, while undeclared legacy
timeframes are not downloaded. Pagination is resumable; rerunning continues from
atomic checkpoints. The downloader fetches coarse Binance klines directly
instead of constructing years of them from 1m data, and scheduled updates use
this same config-driven path:

```bash
make research-history-plan
make research-history-bootstrap
```

For an interrupted or later market-specific refresh:

```bash
make research-history-bootstrap MARKET=futures
make research-history-bootstrap MARKET=spot
```

Each dataset run has an explicit 5,000-request-page ceiling and checkpointed
resume behavior in addition to the job timeout and request delay. Do not remove
the ceiling to force a first bootstrap through in one process; rerun it.

Build the compact regime artifact and exercise the research/runtime wiring:

```bash
make regime-tag-futures
make research-smoke
make strategy-smoke
make research-generate
make research-cycle
make autopilot-once
make jobs-once
make backup
make backup-verify
make readiness
make report
make healthcheck
```

`make research-generate` creates the first bounded, non-executable population;
`make research-cycle` consumes it. Later scheduled runs perform those same two
stages separately. `make research-once` is the combined manual shortcut.

Read `runtime/readiness_report.md`, `runtime/operator_report.md`, and any
healthcheck details. A first healthcheck can report scheduled jobs that have not
yet run; the separate job worker will drain the due queue one bounded job per
cycle. Do not treat the deployment as healthy until `make healthcheck` exits
zero after that queue has progressed.

`make autopilot-once` is supervision-only: it manages products and flatten
requests but never launches data/research jobs. `make jobs-once` executes at
most one due job under the separate job-worker lock. The enabled history jobs
refresh futures coarse data every six hours, futures 1m daily, and spot every
six hours; other bounded jobs handle typed hypothesis generation, real-data
validation, review, maintenance, and hygiene. Dedicated credential-free timers
run forward candidate papering and verified backup independently.

## 4. Install the user-systemd deployment

Generate units under `runtime/systemd-dry-run` first and inspect them:

```bash
make service-dry-run
make autopilot-validate
make readiness
REPO="$PWD" bash scripts/install_autopilot_service.sh
```

The installer creates eight user units:

| Unit | Role |
|---|---|
| `trading-bot-autopilot.service` | Trading supervision and emergency flattening. It runs `src.autopilot.runtime --skip-jobs`, so a slow research/data task cannot block a position-management cycle. |
| `trading-bot-autopilot-jobs.service` | Independent bounded scheduled-job loop (`src.autopilot.job_worker`). |
| `trading-bot-candidate-paper.service` | Credential-free, approval-ledger-inaccessible, resource-limited one-shot candidate paper cycle with a nonblocking process lock. |
| `trading-bot-candidate-paper.timer` | Starts candidate papering every 45 seconds. A digest-specific closed-bar cursor recovers every unseen bar, while downtime/backfill rows are quarantined from promotion evidence. |
| `trading-bot-autopilot-backup.service` | Credential-free daily backup with the live approval ledger mounted read-only and bounded runtime writes. |
| `trading-bot-autopilot-backup.timer` | Starts the verified backup workflow every 24 hours. |
| `trading-bot-autopilot-healthcheck.service` | One-shot machine-readable watchdog. It reads `runtime/alerts.env` through a strict two-key owner-private parser (never as a systemd `EnvironmentFile`), strips inherited exchange/operations variables, skips credential-aware readiness, and performs a bounded drain of queued webhook/Telegram alerts before exit. |
| `trading-bot-autopilot-healthcheck.timer` | Starts the healthcheck every five minutes by default. |

The installer validates config and full readiness before initial enablement,
writes restrictive units with resource limits, enables both long-running
services and all three timers, and verifies user lingering. The 45-second candidate
cadence adds public market-data requests plus bounded CPU/disk activity; its
separate unit retains the configured memory, CPU, task and 240-second timeout
limits. The backup timer is represented by `backup_enabled`,
`backup_cadence_seconds`, and `backup_timeout_seconds` in the autopilot config so
healthcheck uses the same daily freshness contract even though backup is no
longer a generic job. Later supervisor restarts
repeat strict product/config validation but do not use stale research/data
readiness as a start gate: the process must be able to resume management of
existing exposure and report the readiness fault. Independent live-entry gates
still fail closed. The installed watchdog evaluates durable runtime, research,
candidate, backup, and job health without reading `.env`; run `make readiness`
or `make healthcheck` manually for a full credential-aware readiness check. If
lingering cannot be enabled without administrator access:

```bash
sudo loginctl enable-linger "$(id -un)"
loginctl show-user "$(id -un)" --property=Linger
REPO="$PWD" bash scripts/install_autopilot_service.sh
```

Inspect every installed role and its outputs:

```bash
systemctl --user status trading-bot-autopilot.service trading-bot-autopilot-jobs.service --no-pager
systemctl --user list-timers trading-bot-candidate-paper.timer \
  trading-bot-autopilot-backup.timer trading-bot-autopilot-healthcheck.timer --all --no-pager
journalctl --user -u trading-bot-autopilot.service -n 100 --no-pager
journalctl --user -u trading-bot-autopilot-jobs.service -n 100 --no-pager
journalctl --user -u trading-bot-autopilot-healthcheck.service -n 100 --no-pager
.venv/bin/python -m json.tool runtime/status.json
.venv/bin/python -m json.tool runtime/job_worker_status.json
make healthcheck
```

The supervisor and job worker have separate locks and status files. Manual
`make autopilot-once` and `make jobs-once` preserve the same separation as the
installed services, so research/download latency cannot delay position
management.

Optional communications are installed as separate, least-privilege units only
after the core healthcheck is clean. Telegram uses its own token file; the
OpenClaw bridge never loads `.env` or launches OpenClaw:

```bash
cp config/telegram.env.example runtime/telegram.env
chmod 600 runtime/telegram.env
# edit runtime/telegram.env, then:
REPO="$PWD" bash scripts/install_communications_service.sh

# If OpenClaw runs under a dedicated Unix user, first create the shared bridge
# group exactly as documented in COMMUNICATIONS.md. The real install applies
# exact-user deny ACLs around the two narrow bridge paths:
REPO="$PWD" OPENCLAW_GROUP=trading-research-bridge OPENCLAW_USER=openclaw \
  bash scripts/install_openclaw_bridge_timer.sh
```

The Telegram polling unit sees the checkout read-only and receives write access
only to `runtime/operator-control/` (atomic control/audit updates and their
sibling locks) and `runtime/telegram/` (atomic poll-offset state). Its
`runtime/telegram.env` settings file is explicitly mounted read-only. On an
upgrade from the older flat paths, the real installer copies existing
`runtime/control.json`, `runtime/control_audit.jsonl`, and
`runtime/telegram_poll_state.json` into those dedicated directories without
deleting the originals. The control and poll readers also retain a one-time
legacy fallback until the narrowed files are created. Stop or pause the core
services and compare both copies before removing any legacy file.

See [COMMUNICATIONS.md](COMMUNICATIONS.md) before running either installer;
neither channel can approve, activate, resume, alter risk, or place orders.
Shared-user mode preserves unrelated modes/ACLs, refuses symlinked or
unexpected-owner boundary entries, and denies the OpenClaw identity even when a
private subtree still contains older world-readable files.

## 5. Paper soak

Run continuously in paper mode before considering promotion. Default promotion
and candidate-activation review require exact-fingerprint paper evidence with at
least 20 valid trades spanning at least seven days. Treat that as a floor, not a
promise of safety. During the soak, verify:

- clean restarts and reboot recovery with lingering enabled;
- data freshness, research handoffs, and bounded CPU/memory use;
- alert delivery and an external alert when the five-minute healthcheck fails;
- pause, per-product pause, and resume behavior;
- daily backup creation, verification, off-host transfer, and a staged restore;
- recovery after temporary network and exchange API failures.

The default review also requires positive sized paper return, positive holdout
return, no more than 5% paper drawdown, and no more than four consecutive paper
losses. A row counts only when its recorded strategy fingerprint exactly matches
the current strategy; changing behavior while retaining an ID starts the
evidence clock/count again.

## 6. Routine monitoring and control

```bash
make report
make healthcheck
make control ARGS="status"
systemctl --user status trading-bot-autopilot.service trading-bot-autopilot-jobs.service --no-pager
journalctl --user -u trading-bot-autopilot.service -f
```

Important files are:

- `runtime/operator_report.md` and `.json`: compact operator view;
- `runtime/status.json`: trading supervisor heartbeat and product results;
- `runtime/job_worker_status.json`: latest independent job-worker cycle;
- `runtime/job_state.json`: job cadence/backoff history;
- `runtime/research/generated_hypotheses.json`: latest bounded research-only population;
- `runtime/research/experiment_memory.sqlite3`: canonical identities, lineage, evaluation context, and holdout claims;
- `runtime/healthcheck.json`: watchdog result;
- `runtime/alerts.jsonl`: durable local alerts;
- `runtime/operator-control/control_audit.jsonl`: operator control audit.

The product rows expose native-stop identity/trigger evidence plus every durable blocker:
`pending_order`, `pending_entry_recovery`, `risk_recovery_incident`,
`flatten_intent`, and `exit_accounting_intent`. Healthcheck treats these as
blocking on live products. Do not clear one merely to quiet the watchdog.

Set `AUTOPILOT_WEBHOOK_URL` in owner-private `runtime/alerts.env` (not `.env`)
and restart the supervisor and watchdog units to deliver alerts to a lightweight
external channel. Local alert writes remain the audit source.

Pause new entries and jobs for maintenance:

```bash
make control ARGS="pause --reason 'maintenance'"
make control ARGS="resume --reason 'maintenance complete'"
```

Pause only one product or only scheduled jobs:

```bash
make control ARGS="pause-product active_income --reason 'risk review'"
make control ARGS="resume-product active_income --reason 'risk review complete'"
make control ARGS="pause-jobs --reason 'server maintenance'"
make control ARGS="resume-jobs --reason 'server maintenance complete'"
```

A pause blocks new entries. If local state contains a tracked open position, the
trading supervisor continues management-only cycles and can still submit its
risk-reducing exit. A live futures position retains its exchange-native stop;
paper and BTC spot still depend on the supervisor. An unresolved `pending_order`
blocks the cycle and requires the procedure in section 10.

For an emergency, `panic` atomically pauses products and jobs and requests
flattening of every live product:

```bash
make control ARGS="panic --reason 'exchange or risk incident'"
make control ARGS="status"
journalctl --user -u trading-bot-autopilot.service -f
```

Confirm the actual exchange position/balances are safe before resuming. Futures
flatten uses a reduce-only close; also verify its conditional protective order
is terminal rather than orphaned. BTC spot flatten never sells the base BTC
stack; it only buys back a single locally tracked step-aside position and
otherwise fails closed. Unreadable/corrupt futures state is not replaced with a
guessed minimal state. It blocks for manual exchange/state reconciliation.

After a successful flatten, the runtime atomically clears that flatten request
but deliberately leaves the affected product paused (and `panic` leaves the
global pause/jobs pause in place). If the request remains, recovery is not
complete; do not use `clear-flatten` to force it away.

## 7. Backups and off-host copies

The dedicated backup timer keeps 30 small local archives. These include config, approval
and control audit, runtime state/trades/reports, active/staged strategy artifacts,
candidate-paper state/log/review files, the research-factory config/latest batch,
and a transactionally consistent SQLite experiment-memory snapshot. The live
SQLite file is never copied directly. Archives exclude `.env`, credentials,
large market data, and large research outputs. Before the default experiment
memory reaches its explicit 48 MiB ceiling, the factory automatically compresses
a bounded number of rows without deleting canonical identities, lineage,
evaluation context, engine scope, or holdout claims, then deeply verifies and
vacuums the database. If that safe maintenance cannot bring an anomalously large
database below the ceiling, generation pauses and archive size checks fail
visibly instead of silently omitting that recovery-critical file.
Archive creation verifies its own manifest and contents before reporting success;
failed verification returns nonzero and skips retention pruning. The separate
verify command remains an operator check before transfer or restore. Archive
files are forced to `0600` regardless of the invoking shell's umask.
The manifest records configured-but-not-yet-created state as optional missing
files. That is expected on a new deployment. By contrast, every configured
recovery file that exists at backup time is required: if it is too large, a
symlink, or not a regular file, `critical_skipped_files` is nonzero, verification
fails, and healthcheck reports `backup_incomplete`. Do not transfer or rely on
that archive; correct the path/size problem and create a new verified backup.

Create, verify, and copy a backup over an authenticated channel:

```bash
make backup
BACKUP="$(find runtime/backups -type f -name 'autopilot_state_*.zip' | sort | tail -n 1)"
make backup-verify BACKUP="$BACKUP"
sha256sum "$BACKUP"
rsync -av "$BACKUP" backup-host:/secure/trading-bot/
```

Protect the remote archive because it contains trading history and strategy
artifacts. Store exchange credentials separately in a password manager or
secrets service. Rehearse restoration without touching the live tree:

```bash
RESTORE_DIR="/tmp/trading-bot-restore-$(date +%s)"
make backup-restore BACKUP="$BACKUP" RESTORE_DIR="$RESTORE_DIR"
.venv/bin/python -m json.tool "$RESTORE_DIR/RESTORE_REPORT.json"
test "$(stat -c '%a' "$RESTORE_DIR")" = "700"
```

Restore forces the staging root and nested directories to `0700` and every
restored file, including `RESTORE_REPORT.json`, to `0600`, even when the caller
has a permissive umask or overwrites an older staging file.

## 8. Mandatory strategy approval and live sequence

Use this mapping:

| Product | Active artifact | Initial paper log | Testnet rehearsal |
|---|---|---|---|
| `btc_accumulation` | `outputs/active_strategies_position.json` | `runtime/btc_accumulation_trades.csv` | Not required |
| `active_income` | `outputs/active_strategies_flow.json` | `runtime/active_income_trades.csv` | Required |

Do this separately for each product. No automation may run the approval command.

### 8.1 Freeze exact paper evidence

For an initial paper-to-live promotion, pause the target and all jobs so research
cannot replace the artifact during review:

```bash
make control ARGS="pause-product active_income --reason 'promotion review'"
make control ARGS="pause-jobs --reason 'promotion review'"
make report
make backup
make backup-verify
```

Wait for `open_positions` to become empty. Confirm there is no `pending_order`,
`pending_entry_recovery`, `risk_recovery_incident`, `flatten_intent`, or
`exit_accounting_intent`. A pause still permits management-only exits. Never
carry unresolved broker/accounting state into activation or live mode.

Build and read the packet against the exact active artifact and initial paper
log:

```bash
make promotion-review \
  PRODUCT=active_income \
  ARTIFACT=outputs/active_strategies_flow.json \
  TRADE_LOG=runtime/active_income_trades.csv
```

The defaults require at least 20 valid exact-fingerprint trades spanning seven
days, positive return/holdout, and bounded drawdown/loss streak. Read
`runtime/promotion_review.md`, its JSON, and the artifact. A passing packet is
evidence for a decision, not approval.

For a new candidate beside an already-live product, research never touches the
active artifact. It stages `runtime/candidates/<product>.json`; the scheduled
dedicated 45-second candidate-paper timer uses digest-isolated paper state and writes:

- `runtime/candidates/<product>_paper_trades.csv`;
- `runtime/candidates/<product>_promotion_review.md`;
- `runtime/candidate_paper_status.json`.

The state contains one event-time cursor per strategy plus durable historical
next-open recovery entries. Events are ordered by when their bar closes and
becomes knowable; exact-close ties process the shorter timeframe first, then
artifact order. Thus a `10:00` one-hour bar cannot influence any five-minute bar
that opened from `10:05` through `10:55`.

Only the newest closed signal observed within two candidate-timer cadences can
produce promotable evidence. It enters at the credential-free public quote and
the timestamp captured after that quote response—not at a historical price or
an earlier cycle timestamp. Because that observation can occur inside the next
forming bar, the entry-overlapping bar is excluded from exit evaluation; the
first eligible OHLC exit bar must start at or after the recorded entry time.

Within the bounded outage window, unseen bars still advance the cursor and
manage existing positions so state can recover safely. Historical signals may
use deterministic next-open replay, but those entries are tagged
non-promotable. Any catch-up event while a position is open permanently
quarantines the eventual trade, even if the entry was originally observed
forward. `candidate_paper_max_unseen_bars` (240 by default) remains a hard
limit: exceeding it, or detecting a candle gap, leaves the cursor unchanged and
makes `runtime/candidate_paper_status.json` unhealthy. Raise the limit only
after sizing public API, memory, CPU, and service timeout capacity for the
longest enabled base timeframe. This unit never loads exchange credentials or
constructs a broker.

Every trade row and digest-isolated state records
`autopilot.candidate_paper.forward_observation/v2`, a candidate-paper engine
digest, observation/fill provenance, and explicit promotion eligibility.
Promotion and activation count only genuine forward rows matching the current
schema, engine digest, candidate artifact digest, and strategy fingerprint.
Legacy, different-engine, invalid-provenance, and downtime/backfill rows remain
visible as quarantined audit evidence. A code or dependency change starts a
clean flat candidate-paper account; an execution identity change while a paper
position is open fails closed for explicit operator reconciliation.

Wait until the exact candidate digest reports `candidate_activation_ready: true`.
Then pause the product and all jobs, verify the same flat/reconciled state, stop
both services, and activate exactly that reviewed digest:

```bash
make candidate-paper-once
jq '.products[] | select(.product == "active_income")' runtime/candidate_paper_status.json
make control ARGS="pause-product active_income --reason 'candidate activation review'"
make control ARGS="pause-jobs --reason 'candidate activation review'"
systemctl --user stop trading-bot-autopilot.service trading-bot-autopilot-jobs.service \
  trading-bot-candidate-paper.timer trading-bot-candidate-paper.service
CANDIDATE_DIGEST=$(jq -r '.products[] | select(.product == "active_income" and .candidate_activation_ready == true) | .candidate_digest' runtime/candidate_paper_status.json)
make activate-candidate PRODUCT=active_income CANDIDATE_DIGEST="$CANDIDATE_DIGEST" CONFIRM=1 OPERATOR="$USER"
```

Activation rechecks the exact-fingerprint 20-trade/seven-day review under the
runtime/control locks, atomically replaces the active artifact, and records
write-ahead audit events. It grants no approval and adds unique activation
provenance, so the active digest is new. Rebuild the packet against the active
path and candidate paper log:

```bash
make promotion-review \
  PRODUCT=active_income \
  ARTIFACT=outputs/active_strategies_flow.json \
  TRADE_LOG=runtime/candidates/active_income_paper_trades.csv
```

### 8.2 Record explicit human approval

Keep the product configured `live` and paused. First use production credentials
with `EXCHANGE_TESTNET=0` to create the final connected read-only production
preflight. Only the human operator then runs the approval command. If the earlier
packet still contains the preflight-digest placeholder, rebuild it after this
preflight or use the explicit form below. The two
mandatory expected digests prevent approving either a replaced artifact or an
unreviewed preflight environment manifest. The equivalent explicit form is:

```bash
HUMAN_OPERATOR="your-name"
ARTIFACT_DIGEST=$(jq -r '.artifact_digest' runtime/promotion_review.json)
make preflight PRODUCT=active_income
PREFLIGHT_DIGEST="sha256:$(sha256sum runtime/active_income_preflight_report.json | awk '{print $1}')"
.venv/bin/python -m src.autopilot.approvals \
  --ledger runtime/approvals.json \
  approve \
  --config config/autopilot.json \
  --product active_income \
  --artifact outputs/active_strategies_flow.json \
  --expected-artifact-digest "$ARTIFACT_DIGEST" \
  --expected-preflight-digest "$PREFLIGHT_DIGEST" \
  --all \
  --approved-by "$HUMAN_OPERATOR" \
  --confirm-live \
  --notes "Reviewed artifact, holdout, paper results, and risk limits"

.venv/bin/python -m src.autopilot.approvals \
  --ledger runtime/approvals.json \
  check \
  --config config/autopilot.json \
  --product active_income \
  --artifact outputs/active_strategies_flow.json

make backup
make backup-verify
```

For BTC accumulation, replace the product and artifact with
`btc_accumulation` and `outputs/active_strategies_position.json`. `--all` means
every strategy in that artifact was reviewed; do not use it as a shortcut.

Do not approve an automation identity. Approval captures product identity and
the execution-engine digest (Python, installed pinned packages, source, and
`requirements-bot.txt`) plus the stable production exchange/account/risk-cap
manifest. A later equivalent preflight refresh changes its report timestamp and
file digest without invalidating approval; manifest drift does invalidate it.
Any artifact/product/code/Python/dependency change blocks live entry until
review and approval are repeated. Do not deploy code between approval and first
live entry.

### 8.3 Active-income testnet rehearsal

This subsection applies only to `active_income`. Keep the product paused. Put
Binance USDT-M testnet credentials in `.env`, set
`TRADING_LIVE=1`, retain `EXCHANGE_TESTNET=1`, and keep the notional cap tiny.
Then run:

```bash
make preflight PRODUCT=active_income REQUIRE_TESTNET=1
make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100
make testnet-status
```

The broker normalizes amount and trigger price to Binance market precision and
rejects an order before submission if it is below the venue's current amount or
notional minimum. If Binance changes those filters, raise the rehearsal amount
and `MAX_NOTIONAL_USD` only enough to clear the reported minimum; do not bypass
the check.

The `REQUIRE_TESTNET=1` preflight writes
`runtime/active_income_testnet_preflight_report.json`, leaving the approved
production report untouched. The rehearsal intentionally places and closes a
testnet order. Verify the
report proves the native stop place/read/cancel lifecycle and says the ending
futures position is flat; the embedded preflight must also show empty starting
regular/conditional order inventories. If not, reconcile the testnet account
before proceeding.

### 8.4 Refresh production preflight

Use the production trading key, keep withdrawals disabled, set
`EXCHANGE_TESTNET=0`, retain `TRADING_LIVE=1`, isolated margin and 1x leverage,
and set a deliberately tiny `MAX_NOTIONAL_USD`. Do not restart the trading
service yet. Refresh the connected, read-only preflight against the real venue:

```bash
make preflight PRODUCT=active_income
```

For BTC accumulation use:

```bash
make preflight PRODUCT=btc_accumulation
```

The active-income preflight requires the entire dedicated USD-M account to be
flat, including positions in symbols other than the configured BTC product. BTC
spot preflight permits the existing BTC base balance. Futures also proves
one-way mode (`positionSide=BOTH`), native-stop capability, and empty account-wide
regular and conditional order inventories across every symbol. An unsupported or
malformed inventory response fails closed. The report binds the non-secret account
fingerprint and exact current venue/market/testnet/quote/notional/slippage/
leverage/margin settings. Live runtime requires an exact match and requires the
production report to record `EXCHANGE_TESTNET=0`. Preflight expires after one
hour by default, so perform the remaining steps promptly.

### 8.5 Enable exactly one product, verify, then resume

The target product must already have been set to `live` while paused before the
production preflight and final approval (a changed-candidate activation is
already live-mode). Confirm the intended setting in `config/autopilot.json`:

```json
"execution_mode": "live"
```

Leave the target and jobs paused, then validate and start/restart both services
so both processes use the same config and `.env`:

```bash
git diff -- config/autopilot.json
make autopilot-validate
make readiness
systemctl --user restart trading-bot-autopilot.service trading-bot-autopilot-jobs.service
systemctl --user status trading-bot-autopilot.service trading-bot-autopilot-jobs.service --no-pager
make report
make healthcheck
make control ARGS="status"
```

Confirm readiness and the production preflight show the intended product,
venue, live mode, tiny caps, no local or broker exposure, and satisfied entry
gates. Resume the product only then; resume jobs only after the live product is
healthy:

```bash
make control ARGS="resume-product active_income --reason 'human-approved live enablement'"
make control ARGS="resume-jobs --reason 'promotion complete'"
```

Use `btc_accumulation` in that final command when enabling the spot product.
Monitor the first cycles and exchange account directly. Each accepted live
futures position must have `broker_stop_order_id`, `broker_stop_client_id`, and
`broker_stop_trigger_price` in local state and a matching open reduce-only
conditional order at Binance. A new or changed strategy must repeat sections 8.1
through 8.5; the autonomous workflow never approves it.

## 9. Rollback and recovery

### Safe rollback to known-good code

1. Pause the affected product to block new entries.
2. Prefer `panic` and verify the exchange is flat before stopping or replacing
   code. BTC spot and paper exits depend on software supervision. A live futures
   position has a native stop, but rollback is safest only after the position and
   its conditional protection are reconciled or flat.
3. Reconcile any `pending_order` before rollback.
4. Create, verify, and copy an off-host backup.
5. Stop both long-running services, deploy the reviewed known-good commit, rebuild
   the venv if dependencies changed, and validate before restart.

Typical commands after the account is confirmed safe and flat are:

```bash
make control ARGS="pause --reason 'rollback'"
make backup
make backup-verify
systemctl --user stop trading-bot-autopilot-healthcheck.timer
systemctl --user stop trading-bot-autopilot.service trading-bot-autopilot-jobs.service
git status --short
git fetch --tags --prune
KNOWN_GOOD_COMMIT="replace-with-reviewed-commit"
git checkout --detach "$KNOWN_GOOD_COMMIT"
.venv/bin/pip install -r requirements-bot.txt
.venv/bin/pip check
make autopilot-validate
make readiness
systemctl --user start trading-bot-autopilot.service trading-bot-autopilot-jobs.service
systemctl --user start trading-bot-autopilot-healthcheck.timer
make healthcheck
```

Do not use a destructive git reset to bypass a locally modified live config.
Preserve and review that config, or deploy the known-good commit into a fresh
clone and rerun the installer. Keep control paused until the restored version,
environment, strategy artifact, approvals, and current exchange state agree.
Because source and installed versions are part of the engine digest, a code or
dependency rollback intentionally invalidates approval/preflight/rehearsal
recorded under another engine. Repeat the human gate for the restored engine.

### Recover from an archive

Stop live services only after exposure is safe. Verify and extract into staging,
never directly over the live repository:

```bash
BACKUP=/path/to/autopilot_state_TIMESTAMP.zip
make backup-verify BACKUP="$BACKUP"
make backup-restore BACKUP="$BACKUP" RESTORE_DIR=/tmp/trading-bot-recovery
.venv/bin/python -m json.tool /tmp/trading-bot-recovery/RESTORE_REPORT.json
```

Use a fresh clone at the matching known-good commit, recreate `.env` from the
secret store, and compare staged config/state/artifacts with the actual exchange
position, balances, orders, and fills. Copy back only reviewed files. A backup is
a local-state snapshot, not proof of current broker state; never restore stale
`open_positions` or order intent blindly. Finish with `make autopilot-validate`,
`make readiness`, a paused service start, direct exchange reconciliation,
`make report`, and `make healthcheck` before resuming entries.

Experiment memory is the exception to piecemeal JSON recovery: restoring it is
necessary to retain duplicate history and consumed holdout claims. With both
services stopped, verify that the restore report proves a validated
`experiment_memory_snapshot`, then install the snapshot at the configured live
path and validate it before starting anything. (`MANIFEST.json` remains inside
the archive; the restore report carries its verified snapshot result.)

```bash
jq -e '.verification.ok == true and
  .verification.experiment_memory_snapshot.snapshot_integrity.ok == true' \
  /tmp/trading-bot-recovery/RESTORE_REPORT.json
test -f /tmp/trading-bot-recovery/runtime/research/experiment_memory.backup.sqlite3
mkdir -p runtime/research
install -m 600 \
  /tmp/trading-bot-recovery/runtime/research/experiment_memory.backup.sqlite3 \
  runtime/research/experiment_memory.sqlite3
make research-factory-validate
make readiness
```

Never restore only `research_cycle_state.json` without the matching experiment
memory; that could forget canonical deduplication and protected-data claims.

## 10. Unresolved `pending_order` procedure

Before each broker submission, the bot durably records an order intent with a
deterministic `client_id`. If submission, fill parsing, or state persistence is
ambiguous, that `pending_order` remains to prevent a duplicate order. A futures
entry that may have been accepted receives special risk-reduction treatment:
the same cycle and every restart inspect the actual signed position and, when it
matches the intended direction, persist and submit one deterministic reduce-only
close for the full actual quantity. `pending_entry_recovery` remains latched even
after flat proof, so recovery cannot silently become permission to trade again.

1. Pause the product and leave it paused.
2. Inspect the target state file and copy the exact `stage`, `symbol`, `side`,
   `qty`, `reduce_only`, and `client_id`.
3. In Binance order and trade history, search that client ID. Check regular and
   conditional orders, fills, the actual futures position or spot balances, and
   whether an order can still fill later. For an established futures position,
   also reconcile its `broker_stop_order_id` and `broker_stop_client_id`.
4. Cancel an open order or reconcile a fill at the venue. Do not delete the
   `pending_order` field merely to unblock the bot.

For active-income futures, first let the supervisor perform the automatic entry
recovery and inspect both durable markers. If the direction is contradictory,
the close/readback is ambiguous, or the intent is a pending exit, automatic
recovery refuses to guess. After order status is unambiguous and no order can
later fill, use the audited flatten path when a tracked position remains:

```bash
make control ARGS="flatten active_income --reason 'reconcile pending broker order'"
make control ARGS="status"
journalctl --user -u trading-bot-autopilot.service -f
```

Verify Binance is flat and the conditional protective order is terminal before
resuming. A successful flatten request clears itself and leaves the pause set.

For an established futures position, a canceled/expired/rejected or partially
filled native stop with residual exposure,
extra or missing same-direction contracts, opposite exposure, or partial
external close invalidates local protection. The bot persists a deterministic
close intent, closes the full actual signed broker quantity, proves flat, cleans
up the tracked stop, and latches `risk_recovery_incident`. Reconcile the incident,
fills, fees/funding, and local accounting even when the close succeeded; it is a
blocking audit record, not an automatically cleared warning.

A normal/native-stop exit uses a separate `exit_accounting_intent` WAL after
broker-flat proof. It contains a deterministic exit event ID, pre/post state, and
trade row. On restart the bot verifies broker flat, atomically inserts or verifies
the keyed CSV row, and commits state without sending another close. If this
intent remains visible, keep the product paused and fix the storage/evidence
fault; never delete it or append a substitute row manually.

BTC spot is different: automatic flatten can only buy back one fully tracked
step-aside position. A pending-only ambiguous spot fill has insufficient local
metadata for safe automatic recovery. Keep the product paused, reconcile the
sell/buy and BTC/USDT balances manually from exchange records, and reconstruct or
close the state deliberately. Escalate rather than guessing or blindly clearing
the pending intent. A normal step-aside sell persists the exact observed free
USDT increase across the fill and uses that as the later buyback budget; a
missing/implausible delta intentionally leaves the intent unresolved. Keep this
product in a dedicated account with no transfers/manual trades during the
measurement window.

### BTC spot panic-flatten intent reconciliation

The spot panic-flatten path uses a separate `flatten_intent` write-ahead record.
It is atomically stored in `runtime/btc_accumulation_state.json` before the
buyback is submitted and includes the normalized BTC quantity, original USDT
quote budget, pre-order BTC balance evidence, timestamp, and a deterministic
Binance-safe client ID. The intent remains alongside the tracked step-aside
position until both the fill and BTC balance increase are proven and the local
state commit succeeds.

If a submission response, fill, balance read, or local commit is ambiguous, the
runtime reports `unresolved_flatten_intent`. Every later flatten pass validates
that record and refuses to submit another buy. A later BTC balance increase alone
does not contain the missing execution price and fee, so it is not sufficient to
silently clear or account the position. Recovery never recalculates quantity or
price and never places a replacement order.

1. Leave `btc_accumulation` and jobs paused and keep the flatten request active.
2. Stop manual trading on the account. Back up the state file, then copy the
   exact `client_id`, `qty`, `quote_budget`, `position_before`, and `created_ts`
   from `flatten_intent`.
3. In Binance spot order and trade history, search the exact client ID. Reconcile
   every fill or partial fill, whether an accepted order can still execute, and
   the current BTC and USDT balances. Account for deposits, withdrawals, fees,
   conversions, and any other balance-changing activity after `created_ts`.
4. If the order filled, preserve the exact exchange order/fill price, quantity,
   commission asset/amount, and BTC/USDT balance history. The runtime will not
   guess these missing accounting fields from the balance delta; complete a
   reviewed state/trade-ledger reconciliation while the request remains paused.
5. If the order is open, partial, absent, rejected, or the balance history is
   ambiguous, keep the state and flatten request intact. Cancel any still-open
   order only after recording its history, then resolve the remaining BTC/USDT
   exposure explicitly from exchange evidence. A reviewed state repair may be
   performed only after the venue proves no order can fill later.
6. A malformed intent also fails before broker construction. Restore it only
   from a known-good backup and matching exchange evidence; otherwise escalate.

Never delete or alter `flatten_intent` merely to unblock the service. A manual
clear destroys the duplicate-order guard, while changing its pre-order balance
can make later reconciliation unsafe. Resume only after Binance history, current
balances, the local state, the control status, and the operator report all agree.

See the [README](../README.md), [research workflow](RESEARCH_WORKFLOW.md), and
[execution guide](EXECUTION.md) for deeper subsystem behavior.
