#!/bin/bash
set -euo pipefail

REPO="${REPO:-/home/alfred/trading-bot}"
NODE="${NODE:-linux-optiplex}"
SKIP_SYSTEMD="${SKIP_SYSTEMD:-0}"

if [[ "$REPO" != /* || "$REPO" =~ [[:space:]\|] ]]; then
  echo "REPO must be an absolute path without whitespace or pipes." >&2
  exit 1
fi

PROTECT_HOME=true
if [[ "$REPO" == /home/*/* ]]; then
  PROTECT_HOME=read-only
fi

install_platform_unit() {
  local source="$1"
  local destination="$2"
  sed \
    -e "s|/opt/trading-bot|$REPO|g" \
    -e "s|ProtectHome=true|ProtectHome=$PROTECT_HOME|g" \
    "$source" | install -m 0644 /dev/stdin "$destination"
}

refuse_duplicate_authorities() {
  local current_active legacy_active
  current_active="$({
    systemctl list-units --state=active --no-legend --plain 2>/dev/null || true
  } | awk '$1 ~ /^trading-platform-(runtime|market-gateway|data-writer|operations|research|evaluator|agent|control|migration)\.service$/ {print $1}')"
  legacy_active="$({
    systemctl list-units --state=active --no-legend --plain 2>/dev/null || true
  } | awk '$1 ~ /^(trading-platform(@|\-research@|\-agent@|\-bot\-)|trading-bot\-)/ && $1 ~ /\.(service|timer)$/ {print $1}')"
  if [[ -n "$current_active" && -n "$legacy_active" ]]; then
    echo "Refusing platform startup while replaced and current authorities are both active." >&2
    printf '%s\n' "$current_active" "$legacy_active" >&2
    exit 1
  fi
}

enabled_user_links() {
  local directory="$1"
  [[ -d "$directory" ]] || return 0
  find "$directory" -type l \
    \( -path '*.wants/trading-bot-*' -o -path '*.requires/trading-bot-*' \
       -o -path '*.wants/trading-platform*' -o -path '*.requires/trading-platform*' \) \
    \( -name '*.service' -o -name '*.timer' \) -print
}

refuse_user_authorities() {
  local accounts account_user account_uid account_home unit_links user_units runtime_dir
  unit_links="$(enabled_user_links /etc/systemd/user)"
  if [[ -n "$unit_links" ]]; then
    echo "Retire globally enabled legacy user units after backup before installing the platform." >&2
    printf '%s\n' "$unit_links" >&2
    exit 1
  fi
  accounts="$(getent passwd)" || { echo "Cannot inspect user service owners." >&2; exit 1; }
  while IFS=: read -r account_user _ account_uid _ _ account_home _; do
    [[ -n "$account_user" ]] || continue
    runtime_dir="/run/user/$account_uid"
    unit_links="$(enabled_user_links "$account_home/.config/systemd/user")"
    if [[ -n "$unit_links" ]]; then
      echo "Retire legacy user units for $account_user after backup before installing the platform." >&2
      printf '%s\n' "$unit_links" >&2
      exit 1
    fi
    [[ -d "$account_home/.config/systemd/user" || -S "$runtime_dir/bus" ]] || continue
    if ! user_units="$(runuser -u "$account_user" -- env \
      XDG_RUNTIME_DIR="$runtime_dir" DBUS_SESSION_BUS_ADDRESS="unix:path=$runtime_dir/bus" \
      systemctl --user list-units --all --no-legend --plain 'trading*' 2>/dev/null)"; then
      if [[ -S "$runtime_dir/bus" ]] || systemctl is-active --quiet "user@$account_uid.service"; then
        echo "Cannot inspect the active user service manager for $account_user." >&2
        exit 1
      fi
      continue
    fi
    user_units="$(awk '$1 ~ /^(trading-bot-|trading-platform)/ && $3 ~ /^(active|activating|reloading|deactivating)$/ {print $1}' <<< "$user_units")"
    if [[ -n "$user_units" ]]; then
      echo "Retire active legacy user units for $account_user after backup before installing the platform." >&2
      printf '%s\n' "$user_units" >&2
      exit 1
    fi
  done <<< "$accounts"
}

stop_current_authorities() {
  local unit
  local -a current_units
  current_units=(
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
  for unit in "${current_units[@]}"; do
    if systemctl is-active --quiet "$unit"; then
      systemctl stop "$unit" || {
        echo "Refusing platform upgrade because the current unit did not stop: $unit" >&2
        exit 1
      }
    fi
    systemctl disable "$unit" >/dev/null 2>&1 || true
    if systemctl is-active --quiet "$unit"; then
      echo "Refusing platform upgrade while current unit remains active: $unit" >&2
      exit 1
    fi
  done
}

retire_legacy_units() {
  local unit file
  local -a legacy_units legacy_instances legacy_files
  legacy_units=(
    trading-platform@.service
    trading-platform-research@.service
    trading-platform-agent@.service
    trading-bot-autopilot.service
    trading-bot-autopilot-jobs.service
    trading-bot-candidate-paper.service
    trading-bot-candidate-paper.timer
    trading-bot-event-capture.service
    trading-bot-autopilot-backup.service
    trading-bot-autopilot-backup.timer
    trading-bot-autopilot-healthcheck.service
    trading-bot-autopilot-healthcheck.timer
    trading-bot-openclaw-bridge.service
    trading-bot-openclaw-bridge.timer
    trading-bot-telegram.service
    trading-bot-telegram-report.service
    trading-bot-telegram-report.timer
  )
  for unit in "${legacy_units[@]}"; do
    systemctl disable --now "$unit" >/dev/null 2>&1 || true
    if systemctl is-active --quiet "$unit"; then
      echo "Refusing to enable the PostgreSQL platform while legacy unit is active: $unit" >&2
      exit 1
    fi
  done

  legacy_instances=()
  while read -r unit _rest; do
    [[ -n "$unit" ]] && legacy_instances+=("$unit")
  done < <(
    systemctl list-units --all --no-legend --plain 2>/dev/null |
      awk '$1 ~ /^trading-platform(@|-research@|-agent@|-bot-)/ && $1 ~ /\.(service|timer)$/ {print $1}'
  )
  for unit in "${legacy_instances[@]}"; do
    systemctl disable --now "$unit" >/dev/null 2>&1 || true
    if systemctl is-active --quiet "$unit"; then
      echo "Refusing to enable the PostgreSQL platform while replaced unit is active: $unit" >&2
      exit 1
    fi
  done

  shopt -s nullglob
  legacy_files=(
    /etc/systemd/system/trading-platform@*.service
    /etc/systemd/system/trading-platform-research@*.service
    /etc/systemd/system/trading-platform-agent@*.service
    /etc/systemd/system/trading-bot-*.service
    /etc/systemd/system/trading-bot-*.timer
  )
  shopt -u nullglob
  for file in "${legacy_files[@]}"; do
    unit="${file##*/}"
    systemctl disable --now "$unit" >/dev/null 2>&1 || true
    if systemctl is-active --quiet "$unit"; then
      echo "Refusing to enable the PostgreSQL platform while replaced unit is active: $unit" >&2
      exit 1
    fi
  done
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  [[ "$NODE" == "linux-optiplex" ]] || { echo "NODE must be linux-optiplex." >&2; exit 1; }
  [[ -f "$REPO/config/platform.json" ]] || { echo "platform config is missing" >&2; exit 1; }
  bash -n "$REPO/scripts/run_platform_service.sh"
  echo "dry-run: validated Linux platform installation for $REPO"
  exit 0
fi

if [[ "$NODE" == "linux-optiplex" ]]; then
  if [[ "$(id -u)" != "0" ]]; then
    echo "Linux platform service installation requires root." >&2
    exit 1
  fi
  refuse_user_authorities
  stop_current_authorities
  for group in trading-runtime trading-research trading-evaluator trading-agent trading-platform-owner trading-platform; do
    if ! getent group "$group" >/dev/null; then
      groupadd --system "$group"
    fi
  done
  for user in trading-runtime trading-research trading-evaluator trading-agent trading-platform-owner; do
    if ! id "$user" >/dev/null 2>&1; then
      useradd --system --gid "$user" --home-dir /nonexistent \
        --shell /usr/sbin/nologin "$user"
    fi
  done
  for user in trading-runtime trading-research trading-evaluator trading-agent trading-platform-owner; do
    usermod --append --groups trading-platform "$user"
  done
  command -v setfacl >/dev/null 2>&1 || {
    echo "Linux platform installation requires setfacl for service-specific write access." >&2
    exit 1
  }
  # Service accounts traverse the source tree and shared data parents. The
  # agent clones into its own writable root and only reads the source tree.
  chgrp trading-platform "$REPO"
  chmod 0750 "$REPO"
  if [[ "$REPO" == /home/*/* ]]; then
    repository_owner="${REPO#/home/}"
    repository_owner="${repository_owner%%/*}"
    repository_home="/home/$repository_owner"
    [[ -d "$repository_home" ]] || {
      echo "repository home does not exist: $repository_home" >&2
      exit 1
    }
    setfacl -m g:trading-platform:--x "$repository_home"
  fi
  setfacl -R -m g:trading-platform:r-X "$REPO/src" "$REPO/config" "$REPO/alembic"
  setfacl -m g:trading-platform:r "$REPO/alembic.ini"
  setfacl -R -m u:trading-research:r-X,u:trading-evaluator:r-X,u:trading-agent:--- "$REPO/.git"
  setfacl -R -m u:trading-runtime:r-X,u:trading-platform-owner:r-X \
    "$REPO/.venv-runtime"
  setfacl -R -m u:trading-research:r-X,u:trading-evaluator:r-X "$REPO/.venv-research"
  setfacl -R -m u:trading-agent:r-X "$REPO/.venv-agent"
  install -d -m 0750 -o root -g trading-platform /etc/trading-platform
  for environment in \
    runtime:trading-runtime \
    research:trading-research \
    evaluator:trading-evaluator \
    agent:trading-agent \
    migration:trading-platform-owner \
    backup:trading-runtime; do
    name="${environment%%:*}"
    group="${environment#*:}"
    if [[ ! -e "/etc/trading-platform/$name.env" ]]; then
      install -m 0640 -o root -g "$group" /dev/null "/etc/trading-platform/$name.env"
    fi
  done
  install -d -m 0750 -o root -g trading-platform "$REPO/data"
  install -d -m 0750 -o root -g trading-platform "$REPO/runtime"
  install -d -m 0750 -o trading-runtime -g trading-runtime "$REPO/runtime/numba-cache"
  install -d -m 0750 -o trading-platform-owner -g trading-platform-owner "$REPO/runtime/migration"
  install -d -m 0750 -o trading-platform-owner -g trading-platform-owner "$REPO/runtime/migration/numba-cache"
  for directory in raw bars features; do
    install -d -m 2770 -o trading-runtime -g trading-runtime "$REPO/data/$directory"
  done
  for directory in funding metadata; do
    install -d -m 2770 -o trading-runtime -g trading-runtime "$REPO/data/$directory"
  done
  install -d -m 2770 -o trading-runtime -g trading-runtime "$REPO/runtime/backups"
  setfacl -m u:trading-research:---,u:trading-evaluator:--- "$REPO/runtime/backups"
  for directory in research; do
    install -d -m 2770 -o trading-research -g trading-research "$REPO/data/$directory"
  done
  install -d -m 2770 -o trading-evaluator -g trading-evaluator "$REPO/data/artefacts"
  install -d -m 2770 -o trading-evaluator -g trading-evaluator "$REPO/runtime/evaluation"
  install -d -m 2770 -o trading-evaluator -g trading-evaluator "$REPO/runtime/evaluation/numba-cache"
  install -d -m 2770 -o trading-research -g trading-research "$REPO/runtime/research"
  install -d -m 2770 -o trading-research -g trading-research "$REPO/runtime/research/agent-worktrees"
  install -d -m 2770 -o trading-research -g trading-research \
    "$REPO/runtime/research/numba-cache"
  install -d -m 2770 -o trading-agent -g trading-agent "$REPO/runtime/agent-worktrees"
  install -d -m 2770 -o trading-agent -g trading-agent \
    "$REPO/runtime/agent-worktrees/numba-cache"
  # Common-group access is limited to traversal. ACLs grant research read
  # access to market data and keep writes owned by the producing service.
  setfacl -m u:trading-runtime:rwx,u:trading-research:rx,u:trading-evaluator:rx,u:trading-agent:--x "$REPO/data"
  setfacl -m u:trading-runtime:rwx,u:trading-research:rx,u:trading-evaluator:rx,u:trading-agent:--x "$REPO/runtime"
  for directory in raw bars features; do
    setfacl -m u:trading-research:rx,u:trading-evaluator:rx "$REPO/data/$directory"
    find "$REPO/data/$directory" -type d -exec setfacl -m d:u:trading-research:rx,d:u:trading-evaluator:rx {} +
    # Default ACLs cover future files. Repair existing partitions as well so
    # an installation upgrade does not leave historical Parquet unreadable.
    setfacl -R -m u:trading-research:r-X,u:trading-evaluator:r-X "$REPO/data/$directory"
  done
  for directory in funding metadata; do
    setfacl -m u:trading-research:rx,u:trading-evaluator:rx "$REPO/data/$directory"
    find "$REPO/data/$directory" -type d -exec setfacl -m d:u:trading-research:rx,d:u:trading-evaluator:rx {} +
    setfacl -R -m u:trading-research:r-X,u:trading-evaluator:r-X "$REPO/data/$directory"
  done
  setfacl -R -m u:trading-evaluator:r-X "$REPO/data/research"
  setfacl -m d:u:trading-evaluator:rx "$REPO/data/research"
  chown -R trading-evaluator:trading-evaluator "$REPO/data/artefacts"
  setfacl -R -m u:trading-research:---,u:trading-runtime:r-X "$REPO/data/artefacts"
  setfacl -m u:trading-runtime:rx "$REPO/data/research" "$REPO/data/artefacts"
  setfacl -m d:u:trading-runtime:rx "$REPO/data/research" "$REPO/data/artefacts"
  setfacl -m u:trading-runtime:rx "$REPO/runtime/research"
  setfacl -m d:u:trading-runtime:rx "$REPO/runtime/research"
  setfacl -m u:trading-agent:rwx "$REPO/runtime/agent-worktrees"
  setfacl -m u:trading-agent:--- \
    "$REPO/data/raw" "$REPO/data/bars" "$REPO/data/features" \
    "$REPO/data/funding" "$REPO/data/metadata" "$REPO/data/research" \
    "$REPO/data/artefacts" "$REPO/runtime/research" "$REPO/runtime/backups"
  install_platform_unit "$REPO/deploy/systemd/trading-platform-migration.service" \
    /etc/systemd/system/trading-platform-migration.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-runtime.service" \
    /etc/systemd/system/trading-platform-runtime.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-market-gateway.service" \
    /etc/systemd/system/trading-platform-market-gateway.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-data-writer.service" \
    /etc/systemd/system/trading-platform-data-writer.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-operations.service" \
    /etc/systemd/system/trading-platform-operations.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-research.service" \
    /etc/systemd/system/trading-platform-research.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-evaluator.service" \
    /etc/systemd/system/trading-platform-evaluator.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-agent.service" \
    /etc/systemd/system/trading-platform-agent.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-control.service" \
    /etc/systemd/system/trading-platform-control.service
  for slice in critical background agent; do
    install -m 0644 "$REPO/deploy/systemd/trading-platform-${slice}.slice" \
      "/etc/systemd/system/trading-platform-${slice}.slice"
  done
  install_platform_unit "$REPO/deploy/systemd/trading-platform-backup@.service" \
    /etc/systemd/system/trading-platform-backup@.service
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup-postgresql.timer" \
    /etc/systemd/system/trading-platform-backup-postgresql.timer
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup-parquet.timer" \
    /etc/systemd/system/trading-platform-backup-parquet.timer
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup-verify.timer" \
    /etc/systemd/system/trading-platform-backup-verify.timer
  systemctl daemon-reload
  retire_legacy_units
  systemctl daemon-reload
  # Retire old authorities before checking for duplicate ownership.
  refuse_duplicate_authorities
  if [[ "$SKIP_SYSTEMD" == "1" ]]; then
    echo "Installed Linux service users, permissions, and units without enabling systemd."
    exit 0
  fi
  systemctl enable trading-platform-runtime.service
  systemctl enable trading-platform-market-gateway.service
  systemctl enable trading-platform-data-writer.service
  systemctl enable trading-platform-operations.service
  systemctl enable trading-platform-research.service
  systemctl enable trading-platform-evaluator.service
  systemctl enable trading-platform-agent.service
  systemctl enable trading-platform-control.service
  systemctl enable trading-platform-migration.service
  systemctl enable trading-platform-backup-postgresql.timer
  systemctl enable trading-platform-backup-parquet.timer
  systemctl enable trading-platform-backup-verify.timer
  echo "Installed Linux service units. Add service-specific files under /etc/trading-platform/, then start them."
  exit 0
fi

echo "NODE must be linux-optiplex." >&2
exit 1
