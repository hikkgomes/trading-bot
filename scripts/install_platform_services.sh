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
  for group in trading-runtime trading-research trading-agent trading-platform-owner trading-platform; do
    if ! getent group "$group" >/dev/null; then
      groupadd --system "$group"
    fi
  done
  for user in trading-runtime trading-research trading-agent trading-platform-owner; do
    if ! id "$user" >/dev/null 2>&1; then
      useradd --system --gid "$user" --home-dir /nonexistent \
        --shell /usr/sbin/nologin "$user"
    fi
  done
  for user in trading-runtime trading-research trading-agent trading-platform-owner; do
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
  setfacl -R -m u:trading-agent:r-X "$REPO/.git"
  setfacl -R -m u:trading-runtime:r-X,u:trading-platform-owner:r-X \
    "$REPO/.venv-runtime"
  setfacl -R -m u:trading-research:r-X "$REPO/.venv-research"
  setfacl -R -m u:trading-agent:r-X "$REPO/.venv-agent"
  install -d -m 0750 -o root -g trading-platform /etc/trading-platform
  for environment in \
    runtime:trading-runtime \
    research:trading-research \
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
  for directory in raw bars features; do
    install -d -m 2770 -o trading-runtime -g trading-runtime "$REPO/data/$directory"
  done
  install -d -m 2770 -o trading-runtime -g trading-platform "$REPO/runtime/backups"
  for directory in research artefacts reports; do
    install -d -m 2770 -o trading-research -g trading-research "$REPO/data/$directory"
  done
  install -d -m 2770 -o trading-research -g trading-research "$REPO/runtime/research"
  install -d -m 2770 -o trading-research -g trading-research \
    "$REPO/runtime/research/numba-cache"
  install -d -m 2770 -o trading-agent -g trading-agent "$REPO/runtime/agent-worktrees"
  install -d -m 2770 -o trading-agent -g trading-agent \
    "$REPO/runtime/agent-worktrees/numba-cache"
  # Common-group access is limited to traversal. ACLs grant research read
  # access to market data and keep writes owned by the producing service.
  setfacl -m u:trading-runtime:rwx,u:trading-research:rx,u:trading-agent:--x "$REPO/data"
  setfacl -m u:trading-runtime:rwx,u:trading-research:rx,u:trading-agent:--x "$REPO/runtime"
  for directory in raw bars features; do
    setfacl -m u:trading-research:rx "$REPO/data/$directory"
    setfacl -m d:u:trading-research:rx "$REPO/data/$directory"
    # Default ACLs cover future files. Repair existing partitions as well so
    # an installation upgrade does not leave historical Parquet unreadable.
    setfacl -R -m u:trading-research:r-X "$REPO/data/$directory"
  done
  setfacl -m u:trading-runtime:rx "$REPO/data/research" "$REPO/data/artefacts" "$REPO/data/reports"
  setfacl -m d:u:trading-runtime:rx "$REPO/data/research" "$REPO/data/artefacts" "$REPO/data/reports"
  setfacl -m u:trading-runtime:rx "$REPO/runtime/research"
  setfacl -m d:u:trading-runtime:rx "$REPO/runtime/research"
  setfacl -m u:trading-agent:rwx "$REPO/runtime/agent-worktrees"
  install_platform_unit "$REPO/deploy/systemd/trading-platform-migration.service" \
    /etc/systemd/system/trading-platform-migration.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-runtime.service" \
    /etc/systemd/system/trading-platform-runtime.service
  install_platform_unit "$REPO/deploy/systemd/trading-platform-research.service" \
    /etc/systemd/system/trading-platform-research.service
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
  if [[ "$SKIP_SYSTEMD" == "1" ]]; then
    echo "Installed Linux service users, permissions, and units without enabling systemd."
    exit 0
  fi
  systemctl daemon-reload
  legacy_units=(
    trading-bot-autopilot.service
    trading-bot-autopilot-jobs.service
    trading-bot-candidate-paper.service
    trading-bot-event-capture.service
    trading-bot-autopilot-backup.timer
    trading-bot-autopilot-healthcheck.timer
    trading-bot-openclaw-bridge.service
    trading-bot-telegram.service
  )
  for unit in "${legacy_units[@]}"; do
    systemctl disable --now "$unit" >/dev/null 2>&1 || true
    if systemctl is-active --quiet "$unit"; then
      echo "Refusing to enable the PostgreSQL platform while legacy unit is active: $unit" >&2
      exit 1
    fi
  done
  systemctl enable trading-platform-runtime.service
  systemctl enable trading-platform-research.service
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
