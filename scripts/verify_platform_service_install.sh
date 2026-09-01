#!/bin/bash
set -euo pipefail

REPO="${REPO:-/home/alfred/trading-bot}"
DATABASE_URL="${TRADING_PLATFORM_DATABASE_URL:-}"
PLATFORM_CONFIG="${TRADING_PLATFORM_CONFIG:-$REPO/config/platform.json}"

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

probe_json() {
  local user="$1"
  local directory="$2"
  local probe="$directory/.platform-json-probe.$$"
  runuser -u "$user" -- sh -c 'printf "%s\n" "{}" > "$1" && rm -f "$1"' \
    platform-json-probe "$probe"
}

probe_parquet() {
  local user="$1"
  local python="$2"
  local directory="$3"
  local probe="$directory/.platform-parquet-probe.$$.parquet"
  runuser -u "$user" -- "$python" -c \
    'import pathlib, sys; import pyarrow as pa; import pyarrow.parquet as pq; path = pathlib.Path(sys.argv[1]); pq.write_table(pa.table({"probe": [1]}), path); path.unlink()' \
    "$probe"
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
probe_parquet trading-runtime "$REPO/.venv-runtime/bin/python" "$REPO/data/bars"
probe_json trading-runtime "$REPO/runtime"
for path in \
  "$REPO/data/research" \
  "$REPO/data/artefacts" \
  "$REPO/runtime/research"; do
  probe_write trading-research "$path"
done
probe_parquet trading-research "$REPO/.venv-research/bin/python" "$REPO/data/research"
probe_json trading-research "$REPO/data/artefacts"
probe_json trading-research "$REPO/runtime/research"
probe_write trading-agent "$REPO/runtime/agent-worktrees"
probe_json trading-agent "$REPO/runtime/agent-worktrees"
runuser -u trading-agent -- env \
  GIT_CONFIG_COUNT=1 \
  GIT_CONFIG_KEY_0=safe.directory \
  GIT_CONFIG_VALUE_0="$REPO" \
  git -C "$REPO" cat-file -e 'HEAD^{commit}'

run_cycle() {
  local user="$1"
  local python="$2"
  local service="$3"
  shift 3
  [[ -x "$python" ]] || {
    echo "missing service interpreter: $python" >&2
    exit 1
  }
  runuser -u "$user" -- sh -c 'cd "$1" && shift && exec "$@"' \
    platform-service-cycle "$REPO" env \
    TRADING_PLATFORM_DATABASE_URL="$DATABASE_URL" \
    "$@" \
    "$python" -m src.services.supervisor \
    --config "$PLATFORM_CONFIG" \
    --node linux-optiplex \
    --service "$service" \
    --once
}

run_cycle trading-runtime "$REPO/.venv-runtime/bin/python" product-supervisor
run_cycle trading-research "$REPO/.venv-research/bin/python" research-worker \
  NUMBA_CACHE_DIR="$REPO/runtime/research/numba-cache"
run_cycle trading-agent "$REPO/.venv-agent/bin/python" agent-sandbox \
  TRADING_PLATFORM_AGENT_WORKTREE_ROOT="$REPO/runtime/agent-worktrees" \
  NUMBA_CACHE_DIR="$REPO/runtime/agent-worktrees/numba-cache"

echo "verified service-user traversal, writes, and one cycle per service domain"
