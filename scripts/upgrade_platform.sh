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
    trading-platform-evaluator.service
    trading-platform-agent.service
    trading-platform-control.service
    trading-platform-migration.service
  )
  local inventory state
  inventory="$(systemctl list-units --all --no-legend --plain)" || {
    echo "Cannot inspect trading services before upgrade." >&2
    exit 1
  }
  while read -r unit _rest; do
    [[ -n "$unit" ]] && units+=("$unit")
  done < <(awk '$1 ~ /^(trading-platform|trading-bot-)/ && $1 ~ /\.(service|timer)$/ {print $1}' <<< "$inventory")
  # One systemd transaction also stops timers which could restart a writer.
  systemctl stop "${units[@]}" >/dev/null 2>&1 || true
  for unit in "${units[@]}"; do
    state="$(systemctl show "$unit" --property=ActiveState --value)" || {
      echo "Cannot verify stopped trading unit: $unit" >&2
      exit 1
    }
    case "$state" in
      inactive|failed) ;;
      *) echo "platform unit did not stop: $unit ($state)" >&2; exit 1 ;;
    esac
  done
}

cd "$REPO"
stop_active_units
"$PYTHON" -m src.services.backup_service --config "$CONFIG" --mode postgresql
"$PYTHON" -m src.services.backup_service --config "$CONFIG" --mode parquet
"$PYTHON" -m src.services.backup_service --config "$CONFIG" --mode verify
"$PYTHON" -m src.services.platform_upgrade
REPO="$REPO" SKIP_SYSTEMD="${SKIP_SYSTEMD:-0}" bash "$REPO/scripts/install_platform_services.sh"
