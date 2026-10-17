# Control, alerts, and research agents

Communication is an optional edge around the PostgreSQL platform. Trading,
research, accounting, reconciliation, and recovery continue without Telegram,
webhooks, or OpenClaw.

## Control API

`control-api` listens on the configured local bind. Operator routes require the
bearer token from the private service environment. They read and write control
state through PostgreSQL.

`GET /healthz` accepts only actual loopback peers without a token. It returns
fixed health categories and service names, without account data or stored
heartbeat payloads. It checks the configured node's current heartbeats and
published operational report. Missing or stale evidence returns HTTP 503.

Read-only routes:

- `GET /health` or `GET /status`
- `GET /configuration`
- `GET /reports`
- `GET /agent/reviews`
- `GET /paper/windows`

Audited routes:

- `POST /pause`, `/resume`, `/block-new-risk`, or `/management-only`
- `POST /suspend-strategy`
- `POST /cancel-all-entry-orders`
- `POST /emergency-flatten`
- `POST /agent/proposals`
- `POST /paper/restart-forward-window`

Requests must include the target, reason, requesting actor, and timestamp when
the route requires them. Live approval, live assignment, credential changes,
and discretionary orders are not API operations.

Control modes are ordered from least to most restrictive: `run`,
`block_new_risk`, `management_only`, `suspended`, and `emergency_flatten`.
Risk-reducing actions, account reconciliation, and recovery remain available
when new entries are blocked.

## Alerts

Every alert is written to PostgreSQL before remote delivery. Dedupe, cooldown,
acknowledgement, and delivery-failure records are durable. An optional
`TRADING_PLATFORM_ALERT_WEBHOOK_URL` adds outbound webhook delivery; the URL
is never persisted or printed. The report worker alerts on stalled research,
missing datasets, unstarted jobs, stale account or market authority, missing
risk inputs, unresolved recovery, conflicting execution authority, backup
failure, and remote delivery failure.

Current delivery failures count the latest unacknowledged attempt for each
alert and destination. A successful retry or acknowledgement clears that count;
the failure history remains in PostgreSQL.

## OpenClaw

OpenClaw has no order, approval, credential, or live-risk access. Scheduled
reviews receive only a sanitised PostgreSQL research context. Proposals enter
an untrusted proposal record, pass schema and security validation, and are
compiled into the normal typed research queue. Protected holdout results are
never exposed to adaptive agents.

The isolated consumer polls `POST /agent/requests/next` and submits signed
responses to `POST /agent/responses`. Its scoped transport credential is
distinct from the operator token and cannot access operator routes. Delivery,
response, compilation and evaluation remain durable, separate states.

Agents may request bounded research, revise a thesis, or retire a lineage.
They cannot grant `live_ready`, create a live assignment, or alter execution
controls.

The main OpenClaw profile can run `python3 scripts/check_platform_health.py`
as a command job. This reads only the loopback health endpoint and needs no
operator or exchange credential. It prints `NO_REPLY` when healthy and exits
with a fixed failure description otherwise. Native job failure alerts can use
the existing Telegram destination after the operator approves delivery. A
monitor on the OptiPlex cannot report a power or network loss of that host;
server-down notification needs a monitor elsewhere.

## Verification

```bash
make control ARGS="status"
curl --fail http://127.0.0.1:9108/metrics
make platform-report
make platform-readiness
```

Use [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) for service installation and
incident recovery. The obsolete file-based Telegram and autopilot bridges have been removed.
