# Trading platform operator instructions

Use `/home/alfred/trading-bot` on the Linux authority host. PostgreSQL is the
only source of lifecycle, assignment, approval, order, position, risk,
accounting, control, and report state.

## Inspect

```bash
cd /home/alfred/trading-bot
make platform-report
make platform-readiness
curl --fail http://127.0.0.1:8088/status
```

Inspect the resulting PostgreSQL report and system-level service state before
claiming that a process recovered, a strategy progressed, or an account is
safe. Never infer state from a stale file.

## Controls

Use the authenticated local control API for explicit operator actions:

- block new risk or set management-only mode during an incident;
- cancel all entry orders;
- suspend one strategy;
- reduce or flatten selected exposure;
- resume only after reconciliation and an explicit confirmation.

Keep the bearer token private. Do not use chat messages as live approval.
Emergency actions must be idempotent and must be followed by account and
stop reconciliation.

## Research

Review candidates, stage states, waiting reasons, forward summaries, and
generation feedback from PostgreSQL. Synthetic diagnostics are not production
evidence. Agents may submit bounded research proposals, but cannot approve,
promote, place orders, or change risk.
Agents must not autonomously promote or approve any strategy.

If optional Telegram is deployed, OpenClaw is the sole inbound Telegram poller.
Use `config/openclaw_daily_review_prompt.md` and
`config/openclaw_weekly_deep_review_prompt.md` for research reviews.

## Deployment and recovery

Follow [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) for installation,
backups, restores, testnet rehearsal, and incident recovery. Use the grouped
system-level `trading-platform-*` units only.

Legacy autopilot services, SQLite memory, JSON approvals, and active-strategy
files are migration or archive inputs. Do not start them or use them as
runtime authority.
