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
if [[ -L "$repository" || ! -d "$repository" ]]; then
  echo "Platform repository must be a regular directory: $repository" >&2
  exit 1
fi

set -a
# The installer requires this root-owned file. It contains shell-style KEY=VALUE entries.
source "$environment_file"
set +a

cd "$repository"
exec "$repository/.venv/bin/python" -m src.services.supervisor \
  --config "$repository/config/platform.json" \
  --node "$node_id" \
  --service "$service_name"
