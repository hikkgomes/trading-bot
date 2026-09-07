#!/bin/bash
set -euo pipefail

REPO="${REPO:-/home/alfred/trading-bot}"
PYTHON="${PYTHON:-$REPO/.venv-runtime/bin/python}"
CONFIG="${PLATFORM_CONFIG:-$REPO/config/platform.json}"
ENV_FILE="${PLATFORM_ENV_FILE:-/etc/trading-platform/migration.env}"

if [[ "$REPO" != /* || "$REPO" =~ [[:space:]\|] ]]; then
  echo "REPO must be an absolute path without whitespace or pipes." >&2
  exit 1
fi
if [[ "$(uname -s)" != "Linux" || "$(id -u)" != "0" ]]; then
  echo "platform upgrade requires a Linux root shell" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "platform upgrade interpreter is missing: $PYTHON" >&2
  exit 1
fi
if [[ -f "$ENV_FILE" && ! -L "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
if [[ -z "${TRADING_PLATFORM_DATABASE_URL:-}" ]]; then
  echo "TRADING_PLATFORM_DATABASE_URL is required for the backup" >&2
  exit 1
fi

stop_active_units() {
  local unit
  local -a units
  units=(
    trading-platform-runtime.service
    trading-platform-market-gateway.service
    trading-platform-data-writer.service
    trading-platform-operations.service
    trading-platform-research.service
    trading-platform-agent.service
    trading-platform-control.service
    trading-platform-migration.service
  )
  while read -r unit _rest; do
    [[ -n "$unit" ]] && units+=("$unit")
  done < <(
    systemctl list-units --all --no-legend --plain 2>/dev/null |
      awk '$1 ~ /^(trading-platform(@|-research@|-bot-)|trading-bot-)/ && $1 ~ /\.(service|timer)$/ {print $1}'
  )
  for unit in "${units[@]}"; do
    systemctl stop "$unit" >/dev/null 2>&1 || true
    if systemctl is-active --quiet "$unit"; then
      echo "platform unit did not stop: $unit" >&2
      exit 1
    fi
  done
}

cd "$REPO"
stop_active_units
"$PYTHON" -m src.services.backup_service --config "$CONFIG" --mode postgresql
"$PYTHON" -m src.services.platform_upgrade
REPO="$REPO" SKIP_SYSTEMD="${SKIP_SYSTEMD:-0}" bash "$REPO/scripts/install_platform_services.sh"
