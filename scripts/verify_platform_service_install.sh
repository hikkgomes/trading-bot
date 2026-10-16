#!/bin/bash
set -euo pipefail

REPO="${REPO:-/home/alfred/trading-bot}"
PYTHON="$REPO/.venv-research/bin/python"

if [[ "$(id -u)" != "0" ]]; then
  echo "platform service verification requires root" >&2
  exit 1
fi
command -v runuser >/dev/null 2>&1 || {
  echo "runuser is required" >&2
  exit 1
}
[[ -x "$PYTHON" ]] || {
  echo "missing research test interpreter: $PYTHON" >&2
  exit 1
}
[[ -f "$REPO/tests/test_platform_linux_deployment.py" ]] || {
  echo "installed deployment tests are missing" >&2
  exit 1
}

for domain in RUNTIME RESEARCH EVALUATOR; do
  variable="TRADING_PLATFORM_INSTALLED_${domain}_DATABASE_URL"
  database_url="${!variable:-}"
  [[ "$database_url" == postgresql* ]] || {
    echo "$variable must point to the isolated PostgreSQL test database" >&2
    exit 1
  }
done

cd "$REPO"
env -i \
  "HOME=${HOME:-/root}" \
  "PATH=${PATH:-/usr/bin:/bin}" \
  "TRADING_PLATFORM_INSTALL_ROOT=$REPO" \
  "TRADING_PLATFORM_INSTALLED_RUNTIME_DATABASE_URL=$TRADING_PLATFORM_INSTALLED_RUNTIME_DATABASE_URL" \
  "TRADING_PLATFORM_INSTALLED_RESEARCH_DATABASE_URL=$TRADING_PLATFORM_INSTALLED_RESEARCH_DATABASE_URL" \
  "TRADING_PLATFORM_INSTALLED_EVALUATOR_DATABASE_URL=$TRADING_PLATFORM_INSTALLED_EVALUATOR_DATABASE_URL" \
  TRADING_LIVE=0 \
  EXCHANGE_TESTNET=1 \
  "$PYTHON" -m pytest -q "$REPO/tests/test_platform_linux_deployment.py" \
  -k 'installed_service_users_can_traverse_and_write_their_real_paths or installed_database_service_users_can_run_one_real_service_cycle'
