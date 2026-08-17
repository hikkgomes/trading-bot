#!/bin/bash
set -euo pipefail

REPO="${REPO:-/opt/trading-bot}"
NODE="${NODE:-}"

if [[ "$NODE" == "linux-optiplex" ]]; then
  if [[ "$(id -u)" != "0" ]]; then
    echo "Linux platform service installation requires root." >&2
    exit 1
  fi
  if ! getent group trading-platform >/dev/null; then
    groupadd --system trading-platform
  fi
  if ! id trading-platform >/dev/null 2>&1; then
    useradd --system --gid trading-platform --home-dir /nonexistent \
      --shell /usr/sbin/nologin trading-platform
  fi
  install -d -m 0750 -o trading-platform -g trading-platform /etc/trading-platform
  for directory in data runtime runtime/backups; do
    install -d -m 0750 -o trading-platform -g trading-platform "$REPO/$directory"
  done
  install -m 0644 "$REPO/deploy/systemd/trading-platform@.service" \
    /etc/systemd/system/trading-platform@.service
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup@.service" \
    /etc/systemd/system/trading-platform-backup@.service
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup-postgresql.timer" \
    /etc/systemd/system/trading-platform-backup-postgresql.timer
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup-parquet.timer" \
    /etc/systemd/system/trading-platform-backup-parquet.timer
  install -m 0644 "$REPO/deploy/systemd/trading-platform-backup-verify.timer" \
    /etc/systemd/system/trading-platform-backup-verify.timer
  systemctl daemon-reload
  services=(market-gateway data-writer feature-service portfolio-engine risk-engine execution-engine paper-engine product-supervisor accounting-service promotion-engine control-api)
  for service in "${services[@]}"; do
    systemctl enable "trading-platform@${service}.service"
  done
  systemctl enable trading-platform-backup-postgresql.timer
  systemctl enable trading-platform-backup-parquet.timer
  systemctl enable trading-platform-backup-verify.timer
  echo "Installed Linux service units. Add /etc/trading-platform/platform.env, then start them."
  exit 0
fi

if [[ "$NODE" == "macbook-research" ]]; then
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "The research service installer must run on macOS." >&2
    exit 1
  fi
  target_dir="/Library/LaunchDaemons"
  template="$REPO/deploy/launchd/trading-platform-worker.plist.template"
  environment_dir="/Library/Application Support/TradingPlatform"
  if ! id trading-research >/dev/null 2>&1; then
    echo "Create the non-login trading-research service account before installation." >&2
    exit 1
  fi
  install -d -m 0750 -o root -g wheel "$environment_dir"
  chmod 0755 "$REPO/scripts/run_platform_service.sh"
  services=(research-worker ml-worker event-replay-worker agent-sandbox feature-build-worker report-worker)
  for service in "${services[@]}"; do
    target="$target_dir/com.trading-platform.${service}.plist"
    sed -e "s/__SERVICE__/${service}/g" -e "s|__REPOSITORY__|${REPO}|g" \
      "$template" > "$target"
    chown root:wheel "$target"
    chmod 0644 "$target"
  done
  echo "Installed macOS research services. Add the root-owned platform.env file, then load them."
  exit 0
fi

echo "Set NODE to linux-optiplex or macbook-research." >&2
exit 1
