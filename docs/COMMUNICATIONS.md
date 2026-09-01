# Control, alerts, and research agents

Communication is an optional edge around the PostgreSQL platform. Trading,
research, accounting, reconciliation, and recovery continue without Telegram,
webhooks, or OpenClaw.

## Control API

`control-api` listens on the configured local bind and requires the bearer
token from the private service environment. It reads and writes control state
through PostgreSQL.

Read-only routes:

- `GET /health` or `GET /status`
- `GET /configuration`
- `GET /reports`
- `GET /agent/reviews`

Audited routes:

- `POST /pause`, `/resume`, `/block-new-risk`, or `/management-only`
- `POST /suspend-strategy`
- `POST /cancel-all-entry-orders`
- `POST /emergency-flatten`
- `POST /agent/proposals`

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

## OpenClaw

OpenClaw has no order, approval, credential, or live-risk access. Scheduled
reviews receive only a sanitised PostgreSQL research context. Proposals enter
an untrusted proposal record, pass schema and security validation, and are
compiled into the normal typed research queue. Protected holdout results are
not exposed before final disposition.

Agents may request bounded research, revise a thesis, or retire a lineage.
They cannot grant `live_ready`, create a live assignment, or alter execution
controls.

## Verification

```bash
curl --fail http://127.0.0.1:8088/status
curl --fail http://127.0.0.1:9108/metrics
make platform-report
make platform-readiness
```

Use [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) for service installation and
incident recovery. The old file-based Telegram and autopilot bridges are
offline migration or research code only.
