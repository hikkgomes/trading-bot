#!/bin/bash
set -euo pipefail

if [[ "$#" != "4" ]]; then
  echo "Usage: run_platform_service.sh ENV_FILE REPOSITORY NODE SERVICE" >&2
  exit 2
fi

environment_file="$1"
repository="$2"
node_id="$3"
service_name="$4"

if [[ -L "$environment_file" || ! -f "$environment_file" ]]; then
  echo "Platform environment must be a regular non-symlink file: $environment_file" >&2
  exit 1
fi
if [[ "$(uname -s)" == "Darwin" ]]; then
  environment_owner="$(stat -f '%u' "$environment_file")"
  environment_mode="$(stat -f '%Lp' "$environment_file")"
else
  environment_owner="$(stat -c '%u' "$environment_file")"
  environment_mode="$(stat -c '%a' "$environment_file")"
fi
if [[ "$environment_owner" != "0" ]]; then
  echo "Platform environment must be owned by root: $environment_file" >&2
  exit 1
fi
environment_mode="000${environment_mode: -3}"
if [[ "${environment_mode: -2:1}" =~ [2367] || "${environment_mode: -1}" =~ [2367] ]]; then
  echo "Platform environment must not be group- or world-writable: $environment_file" >&2
  exit 1
fi
if [[ -L "$repository" || ! -d "$repository" ]]; then
  echo "Platform repository must be a regular directory: $repository" >&2
  exit 1
fi

set -a
# The installer requires this root-owned file. It contains shell-style KEY=VALUE entries.
source "$environment_file"
set +a

cd "$repository"
python_root="$repository/.venv-runtime"
case "$service_name" in
  research-worker|ml-worker|event-replay-worker|feature-build-worker)
    python_root="$repository/.venv-research"
    ;;
  agent-sandbox)
    python_root="$repository/.venv-agent"
    ;;
esac
exec "$python_root/bin/python" -m src.services.supervisor \
  --config "$repository/config/platform.json" \
  --node "$node_id" \
  --service "$service_name"
