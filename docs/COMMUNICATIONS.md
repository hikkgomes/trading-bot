# Telegram and OpenClaw boundaries

These integrations are optional. The trading and research system does not depend
on either service and continues operating when they are offline.

## Telegram

Telegram is a deterministic notification and restricted operator edge. It is
not an approval or execution channel.

Available inbound commands are:

- `/status` — sanitized supervisor, product, worker, and research status.
- `/help` — show the commands enabled for this bot.
- `/pause` — pause all products and scheduled jobs, when explicitly enabled.
- `/pause_jobs` — pause scheduled data/research jobs, when explicitly enabled.
- `/pause_product NAME` — pause one configured product, when explicitly enabled.

There are deliberately no Telegram commands for strategy approval, candidate
activation, live enablement, resume, flattening, risk changes, configuration,
credentials, or order placement. Resuming always requires the local control CLI
on the server. A broadcast channel works for outbound alerts; pause commands
require a private chat or group message with a real user ID.

### Configure

Create a bot with BotFather, add it only to the private chat/group/channel you
control, and create `runtime/telegram.env`:

```bash
cp config/telegram.env.example runtime/telegram.env
```

```dotenv
AUTOPILOT_TELEGRAM_BOT_TOKEN=replace-with-botfather-token
AUTOPILOT_TELEGRAM_CHAT_ID=-1001234567890
# Pause commands stay off unless both settings below are present.
AUTOPILOT_TELEGRAM_PAUSE_COMMANDS=0
AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS=123456789
```

Protect the file and never put exchange credentials in it:

```bash
chmod 600 runtime/telegram.env
```

Create the separate alert-routing settings file as well. It contains no
exchange credentials and is the only operations settings file read by the
watchdog:

```bash
cp config/alerts.env.example runtime/alerts.env
chmod 600 runtime/alerts.env
```

```dotenv
AUTOPILOT_WEBHOOK_URL=
AUTOPILOT_TELEGRAM_SETTINGS_FILE=runtime/telegram.env
```

`runtime/telegram.env` accepts exactly the four `AUTOPILOT_TELEGRAM_*` keys in
the first example. `runtime/alerts.env` accepts exactly the two alert-routing
keys in the second example. Unknown, duplicate, malformed, symlinked,
oversized, or group/world-readable settings in either file fail closed without
printing any setting value. The dedicated systemd units do not load either file
as an `EnvironmentFile`; each consumer reads its settings file directly, so an
accidental assignment cannot enter the network-facing process environment. The
token does not need to be duplicated in the trading `.env`.

Telegram payloads are a second allowlisted boundary. Secret-bearing keys and
string patterns (bearer/JWT/Telegram tokens, credential query parameters,
signed URLs, and embedded credential assignments) are redacted. Raw diagnostic
fields are replaced with a marker, and protected holdout/final-test keys or
content are recursively omitted. Detailed errors and protected research results
remain in the private local reports and logs. The outbound path is added
alongside the existing local JSONL and webhook alert destinations. Local alerts
and cooldown state are committed before outbound delivery is queued; webhook
and Telegram outcomes are appended later under the same fingerprint. A slow or
unavailable remote endpoint therefore cannot stop supervision or erase local
alerts. The long-running supervisor delivers asynchronously. The systemd
healthcheck is a short-lived oneshot, so it performs a bounded drain of its
queued remote alert before exiting; delivery success or failure is appended to
the local alert log. Its unit passes only the path to `runtime/alerts.env`; the
application opens a non-symlink owner-matched `0600` file and accepts exactly
the webhook URL and Telegram-settings path keys. Systemd never imports its
assignments into the process environment. The unit explicitly removes inherited
exchange and direct operations variables, cannot read `.env`, and runs
readiness-free watchdog checks. Full credential-aware readiness remains an
install/preflight operation. Generic research, candidate-paper, and backup units
both unset inherited operations variables and cannot read either private
operations settings file.

Check the sanitized view without using the network, send a one-off status, then
install the isolated polling service:

```bash
.venv/bin/python -m src.autopilot.telegram_edge \
  --settings-file runtime/telegram.env --validate-settings
.venv/bin/python -m src.autopilot.telegram_edge --status
.venv/bin/python -m src.autopilot.telegram_edge --send-status

REPO="$PWD" DRY_RUN=1 UNIT_DIR="$PWD/runtime/systemd-communications-dry-run" \
  bash scripts/install_communications_service.sh
REPO="$PWD" bash scripts/install_communications_service.sh
```

Inspect it with:

```bash
systemctl --user status \
  trading-bot-telegram.service \
  trading-bot-telegram-report.timer --no-pager
journalctl --user -u trading-bot-telegram.service -n 100 --no-pager
```

The installer also creates a deterministic sanitized status report timer. It
runs every 24 hours by default; set `REPORT_INTERVAL=12h` (or another systemd
duration) while running the installer to change the cadence.

The polling unit mounts the repository read-only. It can write only the two
dedicated state directories required for crash-safe atomic replacement:
`runtime/operator-control/` for `control.json`, `control_audit.jsonl`, their
sibling locks and temporary replacements, and `runtime/telegram/` for
`telegram_poll_state.json`. The settings file `runtime/telegram.env` is an
explicit read-only path; `.env`, approvals, outputs, and market data are
inaccessible. File-only write mounts are intentionally not used because atomic
`os.replace` updates require write permission on the target parent directory.

For upgrades from the older flat layout, a real installer run copies existing
`runtime/control.json`, `runtime/control_audit.jsonl`, and
`runtime/telegram_poll_state.json` into the dedicated directories with mode
`0600`, while retaining the originals for review. Until the first narrowed
control or poll-state write, the readers can fall back to the legacy file. Pause
or stop the core services during the upgrade, verify the new copies, and remove
legacy files only after all installed units use the new config.

Start with `AUTOPILOT_TELEGRAM_PAUSE_COMMANDS=0`. After status and alerts have
been verified, set it to `1` only if the numeric user allowlist is correct, then
restart the service. Both the exact chat ID and the sender user ID must match.

## OpenClaw

OpenClaw is an optional idea and reporting assistant, not part of the trusted
trading process. Do not run OpenClaw from the autopilot job worker: that trusted
worker has broad research/runtime write access and is not a sandbox for an
untrusted assistant. Its installed unit strips trading and operations variables,
but that does not replace process and filesystem isolation. Keep the existing
OpenClaw installation in a separate service, container, or Unix account with
access only to:

- read: `runtime/openclaw/research_context.json`
- write: `runtime/research_inbox/openclaw/incoming/`

It must not receive access to `.env`, `runtime/approvals.json`, control files,
product state, trade logs, strategy artifacts, `data/`, or exchange credentials.
If the existing OpenClaw process runs as the same unrestricted Unix user as the
trader, filesystem isolation is not real; move it to a separate account or
container before enabling the bridge.

### One-way sanitized context

Export context for OpenClaw:

```bash
.venv/bin/python -m src.autopilot.openclaw_bridge export
```

The output contains aggregate development/train/validation research progress,
failure categories, generation methods, feature/primitive performance, coverage,
and novelty counts. Its sources are the latest trusted generated batch at
`runtime/research/generated_hypotheses.json` and the latest research-cycle
summary. The generated batch carries a bounded snapshot of
`ExperimentMemory.generator_feedback()`, whose contract includes development
phases only. OpenClaw receives neither direct SQLite access nor the detailed
experiment log.

The context excludes credentials, approvals, live state, control state, account
information, raw market data, and locked/final holdout outcomes and metrics.
Final holdout decisions are deliberately one-way: they may gate export, stage a
candidate, or retire a lineage inside the trusted core but never become
feedback that OpenClaw can optimize against. Forward-paper evidence is also
excluded from the adaptive research context.

### Proposal contract

OpenClaw may write one JSON file per idea to
`runtime/research_inbox/openclaw/incoming/`. It should write a non-`.json`
temporary file completely and then atomically rename it to `.json`, so the
ingester never sees a partial record. Example:

```json
{
  "schema": "research_proposal/v1",
  "source": "openclaw",
  "created_at": "2026-07-10T12:00:00Z",
  "objective": "active_income",
  "opportunity_type": "day",
  "base_timeframe": "5m",
  "thesis": "A volatility expansion after a quiet regime may carry short-term continuation.",
  "suggested_primitives": ["volatility percentile", "range expansion"],
  "constraints": ["avoid unusually high funding windows"],
  "provenance": {
    "agent": "openclaw-researcher",
    "model": "local-model"
  },
  "source_proposal_id": "session-42:idea-7"
}
```

Valid objectives are `btc_accumulation` and `active_income`. Valid opportunity
types are `scalp`, `day`, `swing`, and `position`.

Ingest proposals with:

```bash
.venv/bin/python -m src.autopilot.openclaw_bridge ingest
```

The bridge bounds and validates input, rejects duplicate/unsafe records, assigns
a canonical digest and proposal ID, archives the raw file, and writes accepted
records to `runtime/research_inbox/openclaw/accepted/`. Rejection metadata goes
to `rejected/`; it does not echo the untrusted thesis.

Incoming files may be owned by the separate OpenClaw account. The bridge never
moves such a file into its private archive and then assumes it can change the
owner's permissions. It reads at most 64 KiB plus one detection byte through a
non-following descriptor, copies the exact bounded bytes into a newly created
bridge-owned `0600` archive file, fsyncs it, and only then unlinks the unchanged
source. Oversized inputs are fingerprinted from bounded metadata/prefix bytes,
rejected, and removed without copying their unbounded body into the archive.

`accepted/` is a transient hand-off spool, not an archive. After the factory
commits an accepted/rejected disposition to its atomic proposal-state receipt,
it removes the corresponding spool file. A crash between those steps is repaired
on the next cycle without reprocessing the proposal. The bridge's private
`index.json` retains canonical IDs and content digests independently of the
spool, so removing a file cannot make the same semantic proposal new again. The
index is bounded at 50,000 identities; at capacity the optional bridge rejects
new submissions fail closed rather than evicting old duplicate memory or
starving native generation. The ingest report sets `degraded: true`, includes
`dedup_index_capacity`, and explicitly reports that native generation is
unaffected.

Raw and rejection audit files use deterministic rolling retention after the
dedup index has been durably updated. `archive/` keeps at most 2,000 files and
128 MiB; `rejected/` keeps at most 5,000 files and 32 MiB. The oldest records by
modification time and name are removed first, with at most 5,000 removals per
bridge cycle. Both directories and every retained/pruned record must be owned by
the bridge user, non-symlink, regular, and inaccessible to group/other users. An
unsafe path stops the bridge visibly instead of being followed or silently
skipped. The ingest report exposes initial, pruned, and retained file/byte counts
under `retention`.

The shared `incoming/` spool is also best-effort bounded to 2,000 observed files
and 128 MiB. Each bridge cycle scans at most 20,000 entries, removes temporary
non-JSON files stale for at least one hour, and can remove at most 5,000 oldest
inert entries to reduce a producer backlog. A truncated scan or remaining
backlog is reported as degraded, while trusted native generation continues.
This cleanup is not a security quota: for a separate user or container, apply a
filesystem/project quota to `incoming/`. A malicious or much faster writer can
otherwise outpace any periodic userspace cleanup.

Every accepted record is stamped as research-only, non-executable, not eligible
for paper trading, promotion, or live trading, and requiring trusted compilation
and full validation. The optional `suggested_spec` is renamed
`untrusted_suggested_spec`; it is never passed directly to an executor. On the
next factory cycle, the trusted compiler either maps the thesis to a native
grammar motif or accepts a complete suggestion only after strict field, typed
schema, search-space, feature-inventory, and risk-limit validation. The compiled
behavior is then canonicalized, deduplicated, recorded in experiment memory,
and subjected to the same real-data and protected-holdout gates as a native
idea. Invalid suggestions are rejected and the native generator continues to
fill its bounded batch.

For a separate Unix account, use a dedicated trading checkout. The checkout and
the immediate entries on each ACL boundary (`REPO`, `runtime`,
`runtime/research_inbox`, and the private OpenClaw inbox root) must be owned by
the trading-service user and must not be symlinks. Create a dedicated group once
and add both the trading-service user and OpenClaw user to it (replace the names
to match the server). Log out and back in, and restart the existing OpenClaw
service, after changing group membership:

```bash
sudo groupadd --force trading-research-bridge
sudo usermod -aG trading-research-bridge trader
sudo usermod -aG trading-research-bridge openclaw
id -nG trader
id -nG openclaw
# Restart the existing OpenClaw service/process so it receives the new group.
```

The installer enforces this boundary with narrow named-user ACLs while
preserving unrelated checkout modes and ACLs. It grants the exact OpenClaw user
execute-only traversal on the minimum trading-user-owned ancestors, explicitly
denies that user on every other existing immediate child at each boundary, and
adds a default named-user denial for future children. An exact named-user ACL is
evaluated before group/world mode bits, so even an existing `0644` data file is
unreachable below its denied directory. Only the sanitized context directory
is readable, and only the incoming spool is setgid/group-writable. Accepted,
rejected, archive, approval, control, trade-log, data, output, config, source,
and credential paths remain private to the OpenClaw identity. OpenClaw should
use `umask 0007` when atomically writing proposal files so the bridge user can
read them.

This mode requires the Linux `setfacl` and `getfacl` commands (usually provided
by the `acl` package), both `OPENCLAW_USER` and `OPENCLAW_GROUP`, and a real non-symlink
checkout. The installer never tries to edit root-owned ancestors such as
`/home` or `/`; it verifies that those foreign-owned parents already permit the
OpenClaw account to traverse them. If a hardened parent does not, have an
administrator grant execute-only traversal to that one account before rerunning.
Because execute-only access to a service home permits access to any independently
world/group-readable path whose exact name is known, keep other contents of the
trading-service home owner-private as well. Prefer a dedicated service home.

Install the credential-free five-minute export/ingest timer:

```bash
REPO="$PWD" DRY_RUN=1 UNIT_DIR="$PWD/runtime/systemd-openclaw-dry-run" \
  bash scripts/install_openclaw_bridge_timer.sh
REPO="$PWD" OPENCLAW_GROUP=trading-research-bridge OPENCLAW_USER=openclaw \
  bash scripts/install_openclaw_bridge_timer.sh
```

The dry run writes units only; it deliberately does not change checkout modes,
ownership, or ACLs. The real shared-user install is the step that validates the
accounts and applies the narrow named-user ACL boundary. Inspect `git status`
first and back up intentional local state before that one-time ACL operation.

If OpenClaw runs in a container, mount only
`runtime/openclaw/research_context.json` read-only plus
`runtime/research_inbox/openclaw/incoming/` read-write. Do not mount the repo or
the trading `.env`. A different container UID cannot read the default `0600` /
`0700` non-shared files: either map a dedicated host OpenClaw identity into the
container and run the shared-user install above, or use an administrator-managed
idmapped/bind-mount boundary that grants equivalent read/write access only to
those two paths. Do not solve container permissions by making the repo or
runtime tree world-readable.

Inspect it with:

```bash
systemctl --user status \
  trading-bot-openclaw-bridge.service \
  trading-bot-openclaw-bridge.timer --no-pager
journalctl --user -u trading-bot-openclaw-bridge.service -n 100 --no-pager
```

OpenClaw being unavailable, returning invalid JSON, or proposing unsafe content
cannot stop the trading supervisor. Its proposal remains inert or is rejected;
there is no direct OpenClaw-to-execution path.
