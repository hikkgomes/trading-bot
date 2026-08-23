#!/bin/bash
# Install the autopilot as a user-level systemd service on a Linux server.
set -euo pipefail
umask 077

REPO="${REPO:-$HOME/trading-bot}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
CONFIG="${CONFIG:-$REPO/config/autopilot.json}"
EVENT_CAPTURE_CONFIG="${EVENT_CAPTURE_CONFIG:-$REPO/config/event_capture.json}"
ALERT_ENV_FILE="${ALERT_ENV_FILE:-$REPO/runtime/alerts.env}"
TELEGRAM_ENV_FILE="${TELEGRAM_ENV_FILE:-$REPO/runtime/telegram.env}"
SERVICE_NAME="${SERVICE_NAME:-trading-bot-autopilot.service}"
JOB_SERVICE_NAME="${JOB_SERVICE_NAME:-trading-bot-autopilot-jobs.service}"
CANDIDATE_PAPER_SERVICE_NAME="${CANDIDATE_PAPER_SERVICE_NAME:-trading-bot-candidate-paper.service}"
CANDIDATE_PAPER_TIMER_NAME="${CANDIDATE_PAPER_TIMER_NAME:-trading-bot-candidate-paper.timer}"
EVENT_CAPTURE_SERVICE_NAME="${EVENT_CAPTURE_SERVICE_NAME:-trading-bot-event-capture.service}"
BACKUP_SERVICE_NAME="${BACKUP_SERVICE_NAME:-trading-bot-autopilot-backup.service}"
BACKUP_TIMER_NAME="${BACKUP_TIMER_NAME:-trading-bot-autopilot-backup.timer}"
HEALTHCHECK_SERVICE_NAME="${HEALTHCHECK_SERVICE_NAME:-trading-bot-autopilot-healthcheck.service}"
HEALTHCHECK_TIMER_NAME="${HEALTHCHECK_TIMER_NAME:-trading-bot-autopilot-healthcheck.timer}"
HEALTHCHECK_ON_BOOT="${HEALTHCHECK_ON_BOOT:-2min}"
HEALTHCHECK_INTERVAL="${HEALTHCHECK_INTERVAL:-5min}"
CANDIDATE_PAPER_ON_BOOT="${CANDIDATE_PAPER_ON_BOOT:-45s}"
CANDIDATE_PAPER_INTERVAL="${CANDIDATE_PAPER_INTERVAL:-45s}"
CANDIDATE_PAPER_TIMEOUT="${CANDIDATE_PAPER_TIMEOUT:-240}"
BACKUP_ON_BOOT="${BACKUP_ON_BOOT:-15min}"
BACKUP_INTERVAL="${BACKUP_INTERVAL:-24h}"
BACKUP_TIMEOUT="${BACKUP_TIMEOUT:-60}"
AUTOPILOT_THREADS="${AUTOPILOT_THREADS:-2}"
AUTOPILOT_MEMORY_MAX="${AUTOPILOT_MEMORY_MAX:-1G}"
AUTOPILOT_CPU_QUOTA="${AUTOPILOT_CPU_QUOTA:-75%}"
AUTOPILOT_TASKS_MAX="${AUTOPILOT_TASKS_MAX:-128}"
DRY_RUN="${DRY_RUN:-0}"
UNIT_DIR="${UNIT_DIR:-$HOME/.config/systemd/user}"

validate_unit_name() {
  local name="$1"
  local suffix="$2"
  local label="$3"
  if [[ -z "$name" || "$name" == */* || "$name" == *$'\n'* || "$name" == *$'\r'* || "$name" != *"$suffix" ]]; then
    echo "$label must be a systemd $suffix unit name without slashes or control characters: $name" >&2
    exit 1
  fi
}

validate_unit_value() {
  local value="$1"
  local label="$2"
  if [[ -z "$value" || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "$label must be non-empty and must not contain control characters" >&2
    exit 1
  fi
}

validate_positive_integer() {
  local value="$1"
  local label="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$label must be a positive integer: $value" >&2
    exit 1
  fi
}

validate_zero_or_one() {
  local value="$1"
  local label="$2"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$label must be 0 or 1: $value" >&2
    exit 1
  fi
}

ensure_user_linger() {
  local target_user="$1"
  local linger_status=""

  if ! command -v loginctl >/dev/null 2>&1; then
    echo "Cannot verify user lingering because loginctl is unavailable." >&2
    echo "Install systemd-login support, then run: sudo loginctl enable-linger $target_user" >&2
    return 1
  fi

  if ! linger_status="$(loginctl show-user "$target_user" --property=Linger 2>/dev/null)"; then
    linger_status=""
  fi
  if [ "$linger_status" = "Linger=yes" ]; then
    return 0
  fi

  if ! loginctl enable-linger "$target_user"; then
    echo "Could not enable user lingering for $target_user." >&2
    echo "Run as an administrator: sudo loginctl enable-linger $target_user" >&2
    echo "Then rerun this installer." >&2
    return 1
  fi

  if ! linger_status="$(loginctl show-user "$target_user" --property=Linger 2>/dev/null)"; then
    linger_status=""
  fi
  if [ "$linger_status" != "Linger=yes" ]; then
    echo "User lingering could not be verified for $target_user." >&2
    echo "Run as an administrator: sudo loginctl enable-linger $target_user" >&2
    echo "Verify with: loginctl show-user $target_user --property=Linger" >&2
    return 1
  fi
}

systemd_quote() {
  local value="$1"
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "Systemd unit values cannot contain newline or carriage return characters" >&2
    exit 1
  fi
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//%/%%}"
  printf '"%s"' "$value"
}

systemd_scalar() {
  local value="$1"
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "Systemd scalar values cannot contain newline or carriage return characters" >&2
    exit 1
  fi
  value="${value//%/%%}"
  printf '%s' "$value"
}

verify_unit_files() {
  if ! command -v systemd-analyze >/dev/null 2>&1; then
    if [ "$DRY_RUN" = "1" ]; then
      echo "Dry run: systemd-analyze is unavailable; skipped unit-file verification."
      return 0
    fi
    echo "systemd-analyze is required to verify generated unit files before installation." >&2
    exit 1
  fi
  systemd-analyze --user verify "$@"
}

prepare_unit_staging() {
  mkdir -p "$TARGET_UNIT_DIR"
  if [ -L "$TARGET_UNIT_DIR" ] || [ ! -d "$TARGET_UNIT_DIR" ]; then
    echo "Systemd unit directory must be a real directory: $TARGET_UNIT_DIR" >&2
    exit 1
  fi
  STAGING_UNIT_DIR="$(mktemp -d "$TARGET_UNIT_DIR/.trading-bot-units.XXXXXX")"
  trap 'if [[ -n "${STAGING_UNIT_DIR:-}" && -d "$STAGING_UNIT_DIR" ]]; then rm -rf -- "$STAGING_UNIT_DIR"; fi' EXIT
  UNIT_DIR="$STAGING_UNIT_DIR"
}

publish_unit_files() {
  local file destination
  for file in "$@"; do
    chmod 600 "$file"
    destination="$TARGET_UNIT_DIR/${file##*/}"
    mv -f -- "$file" "$destination"
  done
  rmdir "$STAGING_UNIT_DIR"
  STAGING_UNIT_DIR=""
  UNIT_DIR="$TARGET_UNIT_DIR"
}

configured_project_path() {
  local key="$1"
  local default_value="$2"
  "$PYTHON" -c 'import json, sys; from pathlib import Path; root = Path(sys.argv[1]); payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")); value = Path(payload.get(sys.argv[3], sys.argv[4])); print((value if value.is_absolute() else root / value).resolve(strict=False))' "$REPO" "$CONFIG" "$key" "$default_value"
}

validate_unit_name "$SERVICE_NAME" ".service" "SERVICE_NAME"
validate_unit_name "$JOB_SERVICE_NAME" ".service" "JOB_SERVICE_NAME"
validate_unit_name "$CANDIDATE_PAPER_SERVICE_NAME" ".service" "CANDIDATE_PAPER_SERVICE_NAME"
validate_unit_name "$CANDIDATE_PAPER_TIMER_NAME" ".timer" "CANDIDATE_PAPER_TIMER_NAME"
validate_unit_name "$EVENT_CAPTURE_SERVICE_NAME" ".service" "EVENT_CAPTURE_SERVICE_NAME"
validate_unit_name "$BACKUP_SERVICE_NAME" ".service" "BACKUP_SERVICE_NAME"
validate_unit_name "$BACKUP_TIMER_NAME" ".timer" "BACKUP_TIMER_NAME"
validate_unit_name "$HEALTHCHECK_SERVICE_NAME" ".service" "HEALTHCHECK_SERVICE_NAME"
validate_unit_name "$HEALTHCHECK_TIMER_NAME" ".timer" "HEALTHCHECK_TIMER_NAME"
validate_positive_integer "$AUTOPILOT_THREADS" "AUTOPILOT_THREADS"
validate_positive_integer "$AUTOPILOT_TASKS_MAX" "AUTOPILOT_TASKS_MAX"
validate_unit_value "$AUTOPILOT_MEMORY_MAX" "AUTOPILOT_MEMORY_MAX"
validate_unit_value "$AUTOPILOT_CPU_QUOTA" "AUTOPILOT_CPU_QUOTA"
validate_unit_value "$ALERT_ENV_FILE" "ALERT_ENV_FILE"
validate_unit_value "$TELEGRAM_ENV_FILE" "TELEGRAM_ENV_FILE"
validate_unit_value "$HEALTHCHECK_ON_BOOT" "HEALTHCHECK_ON_BOOT"
validate_unit_value "$HEALTHCHECK_INTERVAL" "HEALTHCHECK_INTERVAL"
validate_unit_value "$CANDIDATE_PAPER_ON_BOOT" "CANDIDATE_PAPER_ON_BOOT"
validate_unit_value "$CANDIDATE_PAPER_INTERVAL" "CANDIDATE_PAPER_INTERVAL"
validate_positive_integer "$CANDIDATE_PAPER_TIMEOUT" "CANDIDATE_PAPER_TIMEOUT"
validate_unit_value "$BACKUP_ON_BOOT" "BACKUP_ON_BOOT"
validate_unit_value "$BACKUP_INTERVAL" "BACKUP_INTERVAL"
validate_positive_integer "$BACKUP_TIMEOUT" "BACKUP_TIMEOUT"
validate_zero_or_one "$DRY_RUN" "DRY_RUN"

TARGET_UNIT_DIR="$UNIT_DIR"

if [ ! -x "$PYTHON" ]; then
  echo "Missing Python executable: $PYTHON" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "Missing autopilot config: $CONFIG" >&2
  exit 1
fi
if [ ! -f "$EVENT_CAPTURE_CONFIG" ]; then
  echo "Missing event capture config: $EVENT_CAPTURE_CONFIG" >&2
  exit 1
fi

mkdir -p "$REPO/runtime" "$REPO/data" "$REPO/outputs"
prepare_unit_staging
UNIT_FILE="$UNIT_DIR/$SERVICE_NAME"
JOB_SERVICE_FILE="$UNIT_DIR/$JOB_SERVICE_NAME"
CANDIDATE_PAPER_SERVICE_FILE="$UNIT_DIR/$CANDIDATE_PAPER_SERVICE_NAME"
CANDIDATE_PAPER_TIMER_FILE="$UNIT_DIR/$CANDIDATE_PAPER_TIMER_NAME"
EVENT_CAPTURE_SERVICE_FILE="$UNIT_DIR/$EVENT_CAPTURE_SERVICE_NAME"
BACKUP_SERVICE_FILE="$UNIT_DIR/$BACKUP_SERVICE_NAME"
BACKUP_TIMER_FILE="$UNIT_DIR/$BACKUP_TIMER_NAME"
HEALTHCHECK_SERVICE_FILE="$UNIT_DIR/$HEALTHCHECK_SERVICE_NAME"
HEALTHCHECK_TIMER_FILE="$UNIT_DIR/$HEALTHCHECK_TIMER_NAME"

REPO_UNIT="$(systemd_quote "$REPO")"
REPO_WORKING_DIRECTORY="$(systemd_scalar "$REPO")"
PYTHON_UNIT="$(systemd_quote "$PYTHON")"
CONFIG_UNIT="$(systemd_quote "$CONFIG")"
EVENT_CAPTURE_CONFIG_UNIT="$(systemd_quote "$EVENT_CAPTURE_CONFIG")"
ENV_FILE_SCALAR="$(systemd_scalar "-$REPO/.env")"
ALERT_ENV_FILE_UNIT="$(systemd_quote "-$ALERT_ENV_FILE")"
ALERT_ENV_PATH_UNIT="$(systemd_quote "$ALERT_ENV_FILE")"
ALERT_ENV_ASSIGNMENT_UNIT="$(systemd_quote "AUTOPILOT_ALERT_SETTINGS_FILE=$ALERT_ENV_FILE")"
TELEGRAM_ENV_FILE_UNIT="$(systemd_quote "-$TELEGRAM_ENV_FILE")"
HEALTHCHECK_JSON_UNIT="$(systemd_quote "$REPO/runtime/healthcheck.json")"
CANDIDATE_PAPER_LOCK_UNIT="$(systemd_quote "$REPO/runtime/candidate_paper.lock")"
EVENT_CAPTURE_STATUS_UNIT="$(systemd_quote "$REPO/runtime/event_capture_status.json")"
RUNTIME_UNIT="$(systemd_quote "$REPO/runtime")"
DATA_UNIT="$(systemd_quote "$REPO/data")"
OUTPUTS_UNIT="$(systemd_quote "$REPO/outputs")"
JOB_ENV_INACCESSIBLE_UNIT="$(systemd_quote "-$REPO/.env")"

echo "Validating autopilot config..."
if [ "$DRY_RUN" = "1" ]; then
  "$PYTHON" -m src.autopilot.runtime --config "$CONFIG" --validate --skip-jobs
else
  "$PYTHON" -m src.autopilot.runtime --config "$CONFIG" --validate
fi
echo "Validating event capture config..."
"$PYTHON" -m src.autopilot.event_capture --config "$EVENT_CAPTURE_CONFIG" --validate
echo "Validating operations-only alert settings..."
"$PYTHON" -m src.autopilot.alert_settings --file "$ALERT_ENV_FILE"
echo "Checking autopilot readiness..."
"$PYTHON" -m src.autopilot.readiness --config "$CONFIG" \
  --allow-data-bootstrap \
  --output "$REPO/runtime/readiness_report.md" \
  --json-output "$REPO/runtime/readiness_report.json"
APPROVAL_LEDGER="$(configured_project_path "approval_ledger" "runtime/approvals.json")"
CANDIDATE_PAPER_STATUS="$(configured_project_path "candidate_paper_status_file" "runtime/candidate_paper_status.json")"
BACKUP_REPORT="$(configured_project_path "backup_report_file" "runtime/backup_report.json")"
CANDIDATE_PAPER_STATUS_UNIT="$(systemd_quote "$CANDIDATE_PAPER_STATUS")"
BACKUP_REPORT_UNIT="$(systemd_quote "$BACKUP_REPORT")"
JOB_APPROVALS_INACCESSIBLE_UNIT="$(systemd_quote "-$APPROVAL_LEDGER")"
BACKUP_APPROVALS_READ_ONLY_UNIT="$(systemd_quote "-$APPROVAL_LEDGER")"

cat > "$UNIT_FILE" <<UNIT
[Unit]
Description=Trading Bot Autopilot
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory=$REPO_WORKING_DIRECTORY
EnvironmentFile=$ENV_FILE_SCALAR
Environment=$ALERT_ENV_ASSIGNMENT_UNIT
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=$AUTOPILOT_THREADS
Environment=OPENBLAS_NUM_THREADS=$AUTOPILOT_THREADS
Environment=MKL_NUM_THREADS=$AUTOPILOT_THREADS
Environment=NUMEXPR_NUM_THREADS=$AUTOPILOT_THREADS
Environment=LOKY_MAX_CPU_COUNT=$AUTOPILOT_THREADS
ExecStartPre=$PYTHON_UNIT -m src.autopilot.alert_settings --file $ALERT_ENV_PATH_UNIT
ExecStartPre=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT --validate --skip-jobs
ExecStart=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT --skip-jobs
Restart=always
RestartSec=10
TimeoutStopSec=30
KillSignal=SIGINT
KillMode=mixed
Nice=5
IOSchedulingClass=best-effort
IOSchedulingPriority=7
MemoryAccounting=true
MemoryMax=$AUTOPILOT_MEMORY_MAX
CPUAccounting=true
CPUQuota=$AUTOPILOT_CPU_QUOTA
TasksAccounting=true
TasksMax=$AUTOPILOT_TASKS_MAX
NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectClock=true
ProtectControlGroups=true
ProtectKernelLogs=true
ProtectKernelModules=true
ProtectKernelTunables=true
RestrictSUIDSGID=true
LockPersonality=true
UMask=0077
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
UNIT

cat > "$JOB_SERVICE_FILE" <<UNIT
[Unit]
Description=Trading Bot Autopilot Scheduled Jobs
After=network-online.target $SERVICE_NAME
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory=$REPO_WORKING_DIRECTORY
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=OMP_NUM_THREADS=$AUTOPILOT_THREADS
Environment=OPENBLAS_NUM_THREADS=$AUTOPILOT_THREADS
Environment=MKL_NUM_THREADS=$AUTOPILOT_THREADS
Environment=NUMEXPR_NUM_THREADS=$AUTOPILOT_THREADS
Environment=LOKY_MAX_CPU_COUNT=$AUTOPILOT_THREADS
UnsetEnvironment=EXCHANGE_API_KEY EXCHANGE_API_SECRET EXCHANGE_API_PASSWORD TRADING_LIVE EXCHANGE_TESTNET AUTOPILOT_WEBHOOK_URL AUTOPILOT_TELEGRAM_BOT_TOKEN AUTOPILOT_TELEGRAM_CHAT_ID AUTOPILOT_TELEGRAM_PAUSE_COMMANDS AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS AUTOPILOT_TELEGRAM_SETTINGS_FILE AUTOPILOT_ALERT_SETTINGS_FILE
ExecStartPre=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT --validate
ExecStart=$PYTHON_UNIT -m src.autopilot.job_worker --config $CONFIG_UNIT
Restart=always
RestartSec=10
TimeoutStopSec=30
KillSignal=SIGINT
KillMode=mixed
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=7
MemoryAccounting=true
MemoryMax=$AUTOPILOT_MEMORY_MAX
CPUAccounting=true
CPUQuota=$AUTOPILOT_CPU_QUOTA
TasksAccounting=true
TasksMax=$AUTOPILOT_TASKS_MAX
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
ReadWritePaths=$RUNTIME_UNIT
ReadWritePaths=$DATA_UNIT
ReadWritePaths=$OUTPUTS_UNIT
InaccessiblePaths=$JOB_ENV_INACCESSIBLE_UNIT
InaccessiblePaths=$JOB_APPROVALS_INACCESSIBLE_UNIT
InaccessiblePaths=$ALERT_ENV_FILE_UNIT
InaccessiblePaths=$TELEGRAM_ENV_FILE_UNIT
RestrictSUIDSGID=true
LockPersonality=true
UMask=0077
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
UNIT

cat > "$HEALTHCHECK_SERVICE_FILE" <<UNIT
[Unit]
Description=Trading Bot Autopilot Healthcheck
After=$SERVICE_NAME

[Service]
Type=oneshot
WorkingDirectory=$REPO_WORKING_DIRECTORY
Environment=$ALERT_ENV_ASSIGNMENT_UNIT
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=$AUTOPILOT_THREADS
Environment=OPENBLAS_NUM_THREADS=$AUTOPILOT_THREADS
Environment=MKL_NUM_THREADS=$AUTOPILOT_THREADS
Environment=NUMEXPR_NUM_THREADS=$AUTOPILOT_THREADS
Environment=LOKY_MAX_CPU_COUNT=$AUTOPILOT_THREADS
UnsetEnvironment=EXCHANGE_API_KEY EXCHANGE_API_SECRET EXCHANGE_API_PASSWORD TRADING_LIVE EXCHANGE_TESTNET AUTOPILOT_WEBHOOK_URL AUTOPILOT_TELEGRAM_BOT_TOKEN AUTOPILOT_TELEGRAM_CHAT_ID AUTOPILOT_TELEGRAM_PAUSE_COMMANDS AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS AUTOPILOT_TELEGRAM_SETTINGS_FILE
ExecStartPre=$PYTHON_UNIT -m src.autopilot.alert_settings --file $ALERT_ENV_PATH_UNIT
ExecStart=$PYTHON_UNIT -m src.autopilot.healthcheck --config $CONFIG_UNIT --output $HEALTHCHECK_JSON_UNIT --skip-readiness
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=7
MemoryAccounting=true
MemoryMax=$AUTOPILOT_MEMORY_MAX
CPUAccounting=true
CPUQuota=$AUTOPILOT_CPU_QUOTA
TasksAccounting=true
TasksMax=$AUTOPILOT_TASKS_MAX
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
ReadWritePaths=$RUNTIME_UNIT
ReadOnlyPaths=$ALERT_ENV_FILE_UNIT
ReadOnlyPaths=$TELEGRAM_ENV_FILE_UNIT
ReadOnlyPaths=$BACKUP_APPROVALS_READ_ONLY_UNIT
InaccessiblePaths=$JOB_ENV_INACCESSIBLE_UNIT
RestrictSUIDSGID=true
LockPersonality=true
UMask=0077
StandardOutput=journal
StandardError=journal
UNIT

cat > "$HEALTHCHECK_TIMER_FILE" <<UNIT
[Unit]
Description=Run Trading Bot Autopilot Healthcheck

[Timer]
OnBootSec=$HEALTHCHECK_ON_BOOT
OnUnitActiveSec=$HEALTHCHECK_INTERVAL
AccuracySec=60s
Persistent=true
Unit=$HEALTHCHECK_SERVICE_NAME

[Install]
WantedBy=timers.target
UNIT

cat > "$CANDIDATE_PAPER_SERVICE_FILE" <<UNIT
[Unit]
Description=Trading Bot Isolated Candidate Paper Cycle
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$REPO_WORKING_DIRECTORY
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=OMP_NUM_THREADS=$AUTOPILOT_THREADS
Environment=OPENBLAS_NUM_THREADS=$AUTOPILOT_THREADS
Environment=MKL_NUM_THREADS=$AUTOPILOT_THREADS
Environment=NUMEXPR_NUM_THREADS=$AUTOPILOT_THREADS
Environment=LOKY_MAX_CPU_COUNT=$AUTOPILOT_THREADS
UnsetEnvironment=EXCHANGE_API_KEY EXCHANGE_API_SECRET EXCHANGE_API_PASSWORD TRADING_LIVE EXCHANGE_TESTNET AUTOPILOT_WEBHOOK_URL AUTOPILOT_TELEGRAM_BOT_TOKEN AUTOPILOT_TELEGRAM_CHAT_ID AUTOPILOT_TELEGRAM_PAUSE_COMMANDS AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS AUTOPILOT_TELEGRAM_SETTINGS_FILE AUTOPILOT_ALERT_SETTINGS_FILE
ExecStart=$PYTHON_UNIT -m src.autopilot.candidate_paper --config $CONFIG_UNIT --output $CANDIDATE_PAPER_STATUS_UNIT --lock $CANDIDATE_PAPER_LOCK_UNIT
TimeoutStartSec=$CANDIDATE_PAPER_TIMEOUT
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=7
MemoryAccounting=true
MemoryMax=$AUTOPILOT_MEMORY_MAX
CPUAccounting=true
CPUQuota=$AUTOPILOT_CPU_QUOTA
TasksAccounting=true
TasksMax=$AUTOPILOT_TASKS_MAX
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
ReadWritePaths=$RUNTIME_UNIT
ReadWritePaths=$DATA_UNIT
ReadWritePaths=$OUTPUTS_UNIT
InaccessiblePaths=$JOB_ENV_INACCESSIBLE_UNIT
InaccessiblePaths=$JOB_APPROVALS_INACCESSIBLE_UNIT
InaccessiblePaths=$ALERT_ENV_FILE_UNIT
InaccessiblePaths=$TELEGRAM_ENV_FILE_UNIT
RestrictSUIDSGID=true
LockPersonality=true
UMask=0077
StandardOutput=journal
StandardError=journal
UNIT

cat > "$CANDIDATE_PAPER_TIMER_FILE" <<UNIT
[Unit]
Description=Run Trading Bot Candidate Paper Cycle at Sub-Minute Cadence

[Timer]
OnBootSec=$CANDIDATE_PAPER_ON_BOOT
OnUnitActiveSec=$CANDIDATE_PAPER_INTERVAL
AccuracySec=5s
Persistent=true
Unit=$CANDIDATE_PAPER_SERVICE_NAME

[Install]
WantedBy=timers.target
UNIT

cat > "$EVENT_CAPTURE_SERVICE_FILE" <<UNIT
[Unit]
Description=Trading Bot Bounded Public Market Event Capture
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=10

[Service]
Type=simple
WorkingDirectory=$REPO_WORKING_DIRECTORY
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
UnsetEnvironment=EXCHANGE_API_KEY EXCHANGE_API_SECRET EXCHANGE_API_PASSWORD TRADING_LIVE EXCHANGE_TESTNET AUTOPILOT_WEBHOOK_URL AUTOPILOT_TELEGRAM_BOT_TOKEN AUTOPILOT_TELEGRAM_CHAT_ID AUTOPILOT_TELEGRAM_PAUSE_COMMANDS AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS AUTOPILOT_TELEGRAM_SETTINGS_FILE AUTOPILOT_ALERT_SETTINGS_FILE
ExecStartPre=$PYTHON_UNIT -m src.autopilot.event_capture --config $EVENT_CAPTURE_CONFIG_UNIT --validate
ExecStart=$PYTHON_UNIT -m src.autopilot.event_capture --config $EVENT_CAPTURE_CONFIG_UNIT --status $EVENT_CAPTURE_STATUS_UNIT
Restart=always
RestartSec=5
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=7
MemoryAccounting=true
MemoryMax=512M
CPUAccounting=true
CPUQuota=35%
TasksAccounting=true
TasksMax=64
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
ReadWritePaths=$RUNTIME_UNIT
InaccessiblePaths=$JOB_ENV_INACCESSIBLE_UNIT
InaccessiblePaths=$JOB_APPROVALS_INACCESSIBLE_UNIT
InaccessiblePaths=$ALERT_ENV_FILE_UNIT
InaccessiblePaths=$TELEGRAM_ENV_FILE_UNIT
RestrictSUIDSGID=true
LockPersonality=true
UMask=0077
StandardOutput=journal
StandardError=journal
UNIT

cat > "$BACKUP_SERVICE_FILE" <<UNIT
[Unit]
Description=Trading Bot Trusted Runtime Backup
After=$SERVICE_NAME

[Service]
Type=oneshot
WorkingDirectory=$REPO_WORKING_DIRECTORY
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
UnsetEnvironment=EXCHANGE_API_KEY EXCHANGE_API_SECRET EXCHANGE_API_PASSWORD TRADING_LIVE EXCHANGE_TESTNET AUTOPILOT_WEBHOOK_URL AUTOPILOT_TELEGRAM_BOT_TOKEN AUTOPILOT_TELEGRAM_CHAT_ID AUTOPILOT_TELEGRAM_PAUSE_COMMANDS AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS AUTOPILOT_TELEGRAM_SETTINGS_FILE AUTOPILOT_ALERT_SETTINGS_FILE
ExecStart=$PYTHON_UNIT -m src.autopilot.backup --config $CONFIG_UNIT --report $BACKUP_REPORT_UNIT --max-file-bytes 52428800 --max-backups 30
TimeoutStartSec=$BACKUP_TIMEOUT
Nice=15
IOSchedulingClass=best-effort
IOSchedulingPriority=7
MemoryAccounting=true
MemoryMax=$AUTOPILOT_MEMORY_MAX
CPUAccounting=true
CPUQuota=$AUTOPILOT_CPU_QUOTA
TasksAccounting=true
TasksMax=$AUTOPILOT_TASKS_MAX
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
ReadWritePaths=$RUNTIME_UNIT
ReadOnlyPaths=$BACKUP_APPROVALS_READ_ONLY_UNIT
InaccessiblePaths=$JOB_ENV_INACCESSIBLE_UNIT
InaccessiblePaths=$ALERT_ENV_FILE_UNIT
InaccessiblePaths=$TELEGRAM_ENV_FILE_UNIT
RestrictSUIDSGID=true
LockPersonality=true
UMask=0077
StandardOutput=journal
StandardError=journal
UNIT

cat > "$BACKUP_TIMER_FILE" <<UNIT
[Unit]
Description=Run Trading Bot Trusted Runtime Backup Daily

[Timer]
OnBootSec=$BACKUP_ON_BOOT
OnUnitActiveSec=$BACKUP_INTERVAL
AccuracySec=5min
Persistent=true
Unit=$BACKUP_SERVICE_NAME

[Install]
WantedBy=timers.target
UNIT

verify_unit_files \
  "$UNIT_FILE" \
  "$JOB_SERVICE_FILE" \
  "$CANDIDATE_PAPER_SERVICE_FILE" \
  "$CANDIDATE_PAPER_TIMER_FILE" \
  "$EVENT_CAPTURE_SERVICE_FILE" \
  "$BACKUP_SERVICE_FILE" \
  "$BACKUP_TIMER_FILE" \
  "$HEALTHCHECK_SERVICE_FILE" \
  "$HEALTHCHECK_TIMER_FILE"

publish_unit_files \
  "$UNIT_FILE" \
  "$JOB_SERVICE_FILE" \
  "$CANDIDATE_PAPER_SERVICE_FILE" \
  "$CANDIDATE_PAPER_TIMER_FILE" \
  "$EVENT_CAPTURE_SERVICE_FILE" \
  "$BACKUP_SERVICE_FILE" \
  "$BACKUP_TIMER_FILE" \
  "$HEALTHCHECK_SERVICE_FILE" \
  "$HEALTHCHECK_TIMER_FILE"
UNIT_FILE="$UNIT_DIR/$SERVICE_NAME"
JOB_SERVICE_FILE="$UNIT_DIR/$JOB_SERVICE_NAME"
CANDIDATE_PAPER_SERVICE_FILE="$UNIT_DIR/$CANDIDATE_PAPER_SERVICE_NAME"
CANDIDATE_PAPER_TIMER_FILE="$UNIT_DIR/$CANDIDATE_PAPER_TIMER_NAME"
EVENT_CAPTURE_SERVICE_FILE="$UNIT_DIR/$EVENT_CAPTURE_SERVICE_NAME"
BACKUP_SERVICE_FILE="$UNIT_DIR/$BACKUP_SERVICE_NAME"
BACKUP_TIMER_FILE="$UNIT_DIR/$BACKUP_TIMER_NAME"
HEALTHCHECK_SERVICE_FILE="$UNIT_DIR/$HEALTHCHECK_SERVICE_NAME"
HEALTHCHECK_TIMER_FILE="$UNIT_DIR/$HEALTHCHECK_TIMER_NAME"

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete. Wrote unit files:"
  echo "  $UNIT_FILE"
  echo "  $JOB_SERVICE_FILE"
  echo "  $CANDIDATE_PAPER_SERVICE_FILE"
  echo "  $CANDIDATE_PAPER_TIMER_FILE"
  echo "  $EVENT_CAPTURE_SERVICE_FILE"
  echo "  $BACKUP_SERVICE_FILE"
  echo "  $BACKUP_TIMER_FILE"
  echo "  $HEALTHCHECK_SERVICE_FILE"
  echo "  $HEALTHCHECK_TIMER_FILE"
  exit 0
fi

TARGET_USER="$(id -un)"
ensure_user_linger "$TARGET_USER"

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"
systemctl --user enable --now "$JOB_SERVICE_NAME"
systemctl --user enable --now "$CANDIDATE_PAPER_TIMER_NAME"
systemctl --user enable --now "$EVENT_CAPTURE_SERVICE_NAME"
systemctl --user enable --now "$BACKUP_TIMER_NAME"
systemctl --user enable --now "$HEALTHCHECK_TIMER_NAME"

systemctl --user status "$SERVICE_NAME" --no-pager
systemctl --user status "$JOB_SERVICE_NAME" --no-pager
systemctl --user list-timers "$CANDIDATE_PAPER_TIMER_NAME" --no-pager
systemctl --user status "$EVENT_CAPTURE_SERVICE_NAME" --no-pager
systemctl --user list-timers "$BACKUP_TIMER_NAME" --no-pager
systemctl --user list-timers "$HEALTHCHECK_TIMER_NAME" --no-pager
