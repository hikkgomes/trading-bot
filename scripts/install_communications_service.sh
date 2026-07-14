#!/bin/bash
# Install the restricted Telegram polling edge as a separate user service.
set -euo pipefail
umask 077

REPO="${REPO:-$HOME/trading-bot}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
CONFIG="${CONFIG:-$REPO/config/autopilot.json}"
TELEGRAM_ENV="${TELEGRAM_ENV:-$REPO/runtime/telegram.env}"
CONTROL_STATE_DIR="$REPO/runtime/operator-control"
CONTROL_FILE="$CONTROL_STATE_DIR/control.json"
CONTROL_AUDIT_FILE="$CONTROL_STATE_DIR/control_audit.jsonl"
TELEGRAM_STATE_DIR="$REPO/runtime/telegram"
POLL_STATE_FILE="$TELEGRAM_STATE_DIR/telegram_poll_state.json"
LEGACY_CONTROL_FILE="$REPO/runtime/control.json"
LEGACY_CONTROL_AUDIT_FILE="$REPO/runtime/control_audit.jsonl"
LEGACY_POLL_STATE_FILE="$REPO/runtime/telegram_poll_state.json"
SERVICE_NAME="${SERVICE_NAME:-trading-bot-telegram.service}"
REPORT_SERVICE_NAME="${REPORT_SERVICE_NAME:-trading-bot-telegram-report.service}"
REPORT_TIMER_NAME="${REPORT_TIMER_NAME:-trading-bot-telegram-report.timer}"
REPORT_INTERVAL="${REPORT_INTERVAL:-24h}"
UNIT_DIR="${UNIT_DIR:-$HOME/.config/systemd/user}"
DRY_RUN="${DRY_RUN:-0}"

for value in "$SERVICE_NAME" "$REPORT_SERVICE_NAME" "$REPORT_TIMER_NAME" "$REPORT_INTERVAL"; do
  if [[ -z "$value" || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "Unit settings must be non-empty and contain no control characters" >&2
    exit 1
  fi
done
if [[ "$SERVICE_NAME" == */* || "$SERVICE_NAME" != *.service ]]; then
  echo "SERVICE_NAME must be a .service unit name without slashes: $SERVICE_NAME" >&2
  exit 1
fi
if [[ "$REPORT_SERVICE_NAME" == */* || "$REPORT_SERVICE_NAME" != *.service ]]; then
  echo "REPORT_SERVICE_NAME must be a .service unit name without slashes" >&2
  exit 1
fi
if [[ "$REPORT_TIMER_NAME" == */* || "$REPORT_TIMER_NAME" != *.timer ]]; then
  echo "REPORT_TIMER_NAME must be a .timer unit name without slashes" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 1
fi
if [ ! -x "$PYTHON" ]; then
  echo "Missing Python executable: $PYTHON" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "Missing autopilot config: $CONFIG" >&2
  exit 1
fi
if [ ! -f "$TELEGRAM_ENV" ]; then
  echo "Missing Telegram-only environment file: $TELEGRAM_ENV" >&2
  echo "Create it with mode 600; do not put exchange credentials in it." >&2
  exit 1
fi
if [ -L "$TELEGRAM_ENV" ]; then
  echo "Telegram environment file must not be a symlink: $TELEGRAM_ENV" >&2
  exit 1
fi
chmod 600 "$TELEGRAM_ENV"
mkdir -p "$UNIT_DIR" "$CONTROL_STATE_DIR" "$TELEGRAM_STATE_DIR"
for directory in "$CONTROL_STATE_DIR" "$TELEGRAM_STATE_DIR"; do
  if [ -L "$directory" ] || [ ! -d "$directory" ]; then
    echo "Telegram writable state directory must be a non-symlink directory: $directory" >&2
    exit 1
  fi
done
chmod 700 "$CONTROL_STATE_DIR" "$TELEGRAM_STATE_DIR"

migrate_legacy_file() {
  local source="$1"
  local destination="$2"
  local label="$3"
  if [ -L "$destination" ]; then
    echo "Refusing symlinked dedicated $label path: $destination" >&2
    exit 1
  fi
  if [ -e "$destination" ]; then
    if [ ! -f "$destination" ]; then
      echo "Dedicated $label path must be a regular file: $destination" >&2
      exit 1
    fi
    return
  fi
  if [ -L "$source" ]; then
    echo "Refusing unsafe legacy $label path: $source" >&2
    exit 1
  fi
  if [ ! -e "$source" ]; then
    return
  fi
  if [ ! -f "$source" ]; then
    echo "Refusing unsafe legacy $label path: $source" >&2
    exit 1
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run: would copy legacy $label into the dedicated state directory"
    return
  fi
  install -m 600 "$source" "$destination"
  echo "Copied legacy $label into the dedicated state directory; the legacy file was retained"
}

migrate_legacy_file "$LEGACY_CONTROL_FILE" "$CONTROL_FILE" "control file"
migrate_legacy_file "$LEGACY_CONTROL_AUDIT_FILE" "$CONTROL_AUDIT_FILE" "control audit"
migrate_legacy_file "$LEGACY_POLL_STATE_FILE" "$POLL_STATE_FILE" "Telegram poll state"

# Fail closed before writing any unit. The validator accepts only the four
# Telegram-specific keys and never includes their values in diagnostics.
"$PYTHON" -m src.autopilot.telegram_edge \
  --settings-file "$TELEGRAM_ENV" --validate-settings >/dev/null
"$PYTHON" -m src.autopilot.telegram_edge \
  --config "$CONFIG" \
  --poll-state "$POLL_STATE_FILE" \
  --expected-control-file "$CONTROL_FILE" \
  --expected-control-audit "$CONTROL_AUDIT_FILE" \
  --validate-service-paths >/dev/null

systemd_quote() {
  local value="$1"
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "Systemd unit values must not contain control characters" >&2
    exit 1
  fi
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//%/%%}"
  printf '"%s"' "$value"
}

REPO_UNIT="$(systemd_quote "$REPO")"
PYTHON_UNIT="$(systemd_quote "$PYTHON")"
CONFIG_UNIT="$(systemd_quote "$CONFIG")"
TELEGRAM_ENV_UNIT="$(systemd_quote "$TELEGRAM_ENV")"
CONTROL_STATE_DIR_UNIT="$(systemd_quote "$CONTROL_STATE_DIR")"
CONTROL_FILE_UNIT="$(systemd_quote "$CONTROL_FILE")"
CONTROL_AUDIT_FILE_UNIT="$(systemd_quote "$CONTROL_AUDIT_FILE")"
TELEGRAM_STATE_DIR_UNIT="$(systemd_quote "$TELEGRAM_STATE_DIR")"
POLL_STATE_FILE_UNIT="$(systemd_quote "$POLL_STATE_FILE")"
TRADING_ENV_UNIT="$(systemd_quote "$REPO/.env")"
APPROVALS_UNIT="$(systemd_quote "$REPO/runtime/approvals.json")"
OUTPUTS_UNIT="$(systemd_quote "$REPO/outputs")"
DATA_UNIT="$(systemd_quote "$REPO/data")"
UNIT_FILE="$UNIT_DIR/$SERVICE_NAME"
REPORT_SERVICE_FILE="$UNIT_DIR/$REPORT_SERVICE_NAME"
REPORT_TIMER_FILE="$UNIT_DIR/$REPORT_TIMER_NAME"

"$PYTHON" -m src.autopilot.telegram_edge --config "$CONFIG" --status >/dev/null

cat > "$UNIT_FILE" <<UNIT
[Unit]
Description=Trading Bot Restricted Telegram Edge
After=network-online.target trading-bot-autopilot.service
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory=$REPO_UNIT
Environment=PYTHONUNBUFFERED=1
ExecStartPre=$PYTHON_UNIT -m src.autopilot.telegram_edge --settings-file $TELEGRAM_ENV_UNIT --validate-settings
ExecStartPre=$PYTHON_UNIT -m src.autopilot.telegram_edge --config $CONFIG_UNIT --poll-state $POLL_STATE_FILE_UNIT --expected-control-file $CONTROL_FILE_UNIT --expected-control-audit $CONTROL_AUDIT_FILE_UNIT --validate-service-paths
ExecStartPre=$PYTHON_UNIT -m src.autopilot.telegram_edge --config $CONFIG_UNIT --status
ExecStart=$PYTHON_UNIT -m src.autopilot.telegram_edge --config $CONFIG_UNIT --settings-file $TELEGRAM_ENV_UNIT --poll-state $POLL_STATE_FILE_UNIT
Restart=always
RestartSec=10
TimeoutStopSec=20
KillSignal=SIGINT
UMask=0077
Nice=10
MemoryAccounting=true
MemoryMax=192M
CPUAccounting=true
CPUQuota=10%
TasksAccounting=true
TasksMax=32
NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectClock=true
ProtectControlGroups=true
ProtectKernelLogs=true
ProtectKernelModules=true
ProtectKernelTunables=true
ProtectSystem=strict
ReadOnlyPaths=$REPO_UNIT
ReadOnlyPaths=$TELEGRAM_ENV_UNIT
ReadWritePaths=$CONTROL_STATE_DIR_UNIT
ReadWritePaths=$TELEGRAM_STATE_DIR_UNIT
InaccessiblePaths=$TRADING_ENV_UNIT
InaccessiblePaths=-$APPROVALS_UNIT
InaccessiblePaths=-$OUTPUTS_UNIT
InaccessiblePaths=-$DATA_UNIT
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
RestrictSUIDSGID=true
LockPersonality=true
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
UNIT

cat > "$REPORT_SERVICE_FILE" <<UNIT
[Unit]
Description=Trading Bot Daily Telegram Status Report
After=network-online.target trading-bot-autopilot.service
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$REPO_UNIT
Environment=PYTHONUNBUFFERED=1
ExecStartPre=$PYTHON_UNIT -m src.autopilot.telegram_edge --settings-file $TELEGRAM_ENV_UNIT --validate-settings
ExecStart=$PYTHON_UNIT -m src.autopilot.telegram_edge --config $CONFIG_UNIT --settings-file $TELEGRAM_ENV_UNIT --send-status
UMask=0077
Nice=15
MemoryAccounting=true
MemoryMax=192M
CPUAccounting=true
CPUQuota=10%
TasksAccounting=true
TasksMax=32
NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectClock=true
ProtectControlGroups=true
ProtectKernelLogs=true
ProtectKernelModules=true
ProtectKernelTunables=true
ProtectSystem=strict
ReadOnlyPaths=$REPO_UNIT
ReadOnlyPaths=$TELEGRAM_ENV_UNIT
InaccessiblePaths=$TRADING_ENV_UNIT
InaccessiblePaths=-$APPROVALS_UNIT
InaccessiblePaths=-$OUTPUTS_UNIT
InaccessiblePaths=-$DATA_UNIT
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
RestrictSUIDSGID=true
LockPersonality=true
StandardOutput=journal
StandardError=journal
UNIT

cat > "$REPORT_TIMER_FILE" <<UNIT
[Unit]
Description=Send the daily sanitized trading-bot Telegram status

[Timer]
OnBootSec=5min
OnUnitActiveSec=$REPORT_INTERVAL
AccuracySec=5min
Persistent=true
Unit=$REPORT_SERVICE_NAME

[Install]
WantedBy=timers.target
UNIT

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete: wrote $UNIT_FILE, $REPORT_SERVICE_FILE, and $REPORT_TIMER_FILE"
  exit 0
fi

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"
systemctl --user enable --now "$REPORT_TIMER_NAME"
systemctl --user status "$SERVICE_NAME" "$REPORT_TIMER_NAME" --no-pager
