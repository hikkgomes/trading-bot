#!/bin/bash
set -euo pipefail

REPO="${REPO:-/opt/trading-bot}"
NODE="${NODE:-linux-optiplex}"

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
  install -d -m 0750 -o root -g trading-runtime /etc/trading-platform
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
  install -d -m 2770 -o trading-agent -g trading-agent "$REPO/runtime/agent-worktrees"
  # Common-group access is limited to traversal. ACLs grant research read
  # access to market data and keep writes owned by the producing service.
  setfacl -m u:trading-runtime:rwx,u:trading-research:rx,u:trading-agent:--x "$REPO/data"
  setfacl -m u:trading-runtime:rwx,u:trading-research:rx "$REPO/runtime"
  for directory in raw bars features; do
    setfacl -m u:trading-research:rx "$REPO/data/$directory"
    setfacl -m d:u:trading-research:rx "$REPO/data/$directory"
  done
  setfacl -m u:trading-runtime:rx "$REPO/data/research" "$REPO/data/artefacts" "$REPO/data/reports"
  setfacl -m d:u:trading-runtime:rx "$REPO/data/research" "$REPO/data/artefacts" "$REPO/data/reports"
  setfacl -m u:trading-runtime:rx "$REPO/runtime/research"
  setfacl -m d:u:trading-runtime:rx "$REPO/runtime/research"
  setfacl -m u:trading-agent:rwx "$REPO/runtime/agent-worktrees"
  install -m 0644 "$REPO/deploy/systemd/trading-platform@.service" \
    /etc/systemd/system/trading-platform@.service
  install -m 0644 "$REPO/deploy/systemd/trading-platform-research@.service" \
    /etc/systemd/system/trading-platform-research@.service
  install -m 0644 "$REPO/deploy/systemd/trading-platform-agent@.service" \
    /etc/systemd/system/trading-platform-agent@.service
  install -m 0644 "$REPO/deploy/systemd/trading-platform-migration.service" \
    /etc/systemd/system/trading-platform-migration.service
  for slice in critical background agent; do
    install -m 0644 "$REPO/deploy/systemd/trading-platform-${slice}.slice" \
      "/etc/systemd/system/trading-platform-${slice}.slice"
  done
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup@.service" \
    /etc/systemd/system/trading-platform-backup@.service
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup-postgresql.timer" \
    /etc/systemd/system/trading-platform-backup-postgresql.timer
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup-parquet.timer" \
    /etc/systemd/system/trading-platform-backup-parquet.timer
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup-verify.timer" \
    /etc/systemd/system/trading-platform-backup-verify.timer
  systemctl daemon-reload
  critical_services=(market-gateway data-writer feature-service strategy-evaluator portfolio-engine portfolio-state-service risk-engine execution-engine paper-engine product-supervisor accounting-service promotion-engine control-api universe-service)
  research_services=(research-worker ml-worker event-replay-worker feature-build-worker report-worker)
  for service in "${critical_services[@]}"; do
    systemctl enable "trading-platform@${service}.service"
  done
  for service in "${research_services[@]}"; do
    systemctl enable "trading-platform-research@${service}.service"
  done
  systemctl enable trading-platform-agent@agent-sandbox.service
  systemctl enable trading-platform-migration.service
  systemctl enable trading-platform-backup-postgresql.timer
  systemctl enable trading-platform-backup-parquet.timer
  systemctl enable trading-platform-backup-verify.timer
  echo "Installed Linux service units. Add service-specific files under /etc/trading-platform/, then start them."
  exit 0
fi

echo "NODE must be linux-optiplex." >&2
exit 1
