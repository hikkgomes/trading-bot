#!/bin/bash
set -euo pipefail

REPO="${REPO:-/opt/trading-bot}"
DATABASE_URL="${TRADING_PLATFORM_DATABASE_URL:-}"

if [[ "$(id -u)" != "0" ]]; then
  echo "platform service verification requires root" >&2
  exit 1
fi
if [[ -z "$DATABASE_URL" ]]; then
  echo "TRADING_PLATFORM_DATABASE_URL is required" >&2
  exit 1
fi
command -v runuser >/dev/null 2>&1 || {
  echo "runuser is required" >&2
  exit 1
}

probe_write() {
  local user="$1"
  local directory="$2"
  local probe="$directory/.platform-permission-probe.$$"
  runuser -u "$user" -- sh -c ': > "$1" && rm -f "$1"' \
    platform-permission-probe "$probe"
}

for user in trading-runtime trading-research trading-agent; do
  id "$user" >/dev/null 2>&1 || {
    echo "missing service user: $user" >&2
    exit 1
  }
done

for path in \
  "$REPO/data/raw" \
  "$REPO/data/bars" \
  "$REPO/data/features" \
  "$REPO/runtime"; do
  probe_write trading-runtime "$path"
done
for path in \
  "$REPO/data/research" \
  "$REPO/data/artefacts" \
  "$REPO/data/reports" \
  "$REPO/runtime/research"; do
  probe_write trading-research "$path"
done
probe_write trading-agent "$REPO/runtime/agent-worktrees"

run_cycle() {
  local user="$1"
  local python="$2"
  local service="$3"
  shift 3
  [[ -x "$python" ]] || {
    echo "missing service interpreter: $python" >&2
    exit 1
  }
  runuser -u "$user" -- env \
    TRADING_PLATFORM_DATABASE_URL="$DATABASE_URL" \
    "$@" \
    "$python" -m src.services.supervisor \
    --config "$REPO/config/platform.json" \
    --node linux-optiplex \
    --service "$service" \
    --once
}

run_cycle trading-runtime "$REPO/.venv-runtime/bin/python" product-supervisor
run_cycle trading-research "$REPO/.venv-research/bin/python" research-worker
run_cycle trading-agent "$REPO/.venv-agent/bin/python" agent-sandbox \
  TRADING_PLATFORM_AGENT_WORKTREE_ROOT="$REPO/runtime/agent-worktrees"

echo "verified service-user traversal, writes, and one cycle per service domain"
