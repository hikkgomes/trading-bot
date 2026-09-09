#!/bin/bash
set -euo pipefail

REPO="${REPO:-/home/alfred/trading-bot}"
PLATFORM_CONFIG="${TRADING_PLATFORM_CONFIG:-$REPO/config/platform.json}"

if [[ "$(id -u)" != "0" ]]; then
  echo "platform service verification requires root" >&2
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

for user in trading-runtime trading-research trading-evaluator trading-agent; do
  id "$user" >/dev/null 2>&1 || {
    echo "missing service user: $user" >&2
    exit 1
  }
done

for path in \
  "$REPO/data/raw" \
  "$REPO/data/bars" \
  "$REPO/data/features" \
  "$REPO/runtime" \
  "$REPO/runtime/numba-cache"; do
  probe_write trading-runtime "$path"
done
probe_parquet trading-runtime "$REPO/.venv-runtime/bin/python" "$REPO/data/bars"
probe_json trading-runtime "$REPO/runtime"
for path in \
  "$REPO/data/research" \
  "$REPO/runtime/research"; do
  probe_write trading-research "$path"
done
probe_parquet trading-research "$REPO/.venv-research/bin/python" "$REPO/data/research"
probe_json trading-evaluator "$REPO/data/artefacts"
probe_write trading-evaluator "$REPO/runtime/evaluation"
probe_json trading-research "$REPO/runtime/research"
probe_write trading-agent "$REPO/runtime/agent-worktrees"
probe_json trading-agent "$REPO/runtime/agent-worktrees"
if runuser -u trading-agent -- test -r "$REPO/.git/config"; then
  echo "agent can read source Git metadata" >&2
  exit 1
fi
if probe_write trading-research "$REPO/data/artefacts" 2>/dev/null; then
  echo "adaptive research can write sealed artefacts" >&2
  exit 1
fi
for user in trading-research trading-evaluator trading-agent; do
  if runuser -u "$user" -- test -r "$REPO/runtime/backups"; then
    echo "$user can read private backups" >&2
    exit 1
  fi
done

run_cycle() {
  local user="$1"
  local python="$2"
  local service="$3"
  local environment="$4"
  shift 4
  [[ -x "$python" ]] || {
    echo "missing service interpreter: $python" >&2
    exit 1
  }
  runuser -u "$user" -- sh -c 'set -a; . "$1"; set +a; cd "$2"; shift 2; exec "$@"' \
    platform-service-cycle "$environment" "$REPO" env \
    "$@" \
    "$python" -m src.services.supervisor \
    --config "$PLATFORM_CONFIG" --node linux-optiplex --service "$service" --once
}

run_cycle trading-runtime "$REPO/.venv-runtime/bin/python" product-supervisor /etc/trading-platform/runtime.env \
  NUMBA_CACHE_DIR="$REPO/runtime/numba-cache"
run_cycle trading-research "$REPO/.venv-research/bin/python" research-worker /etc/trading-platform/research.env \
  NUMBA_CACHE_DIR="$REPO/runtime/research/numba-cache"
run_cycle trading-evaluator "$REPO/.venv-research/bin/python" evaluation-worker /etc/trading-platform/evaluator.env \
  NUMBA_CACHE_DIR="$REPO/runtime/evaluation/numba-cache"
run_cycle trading-agent "$REPO/.venv-agent/bin/python" agent-sandbox /etc/trading-platform/agent.env \
  TRADING_PLATFORM_AGENT_WORKTREE_ROOT="$REPO/runtime/agent-worktrees" \
  NUMBA_CACHE_DIR="$REPO/runtime/agent-worktrees/numba-cache"

echo "verified service-user traversal, writes, and one cycle per service domain"
