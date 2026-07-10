#!/bin/bash
# Install the autopilot as a user-level systemd service on a Linux server.
set -euo pipefail
umask 077

REPO="${REPO:-$HOME/trading-bot}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
CONFIG="${CONFIG:-$REPO/config/autopilot.json}"
SERVICE_NAME="${SERVICE_NAME:-trading-bot-autopilot.service}"
JOB_SERVICE_NAME="${JOB_SERVICE_NAME:-trading-bot-autopilot-jobs.service}"
HEALTHCHECK_SERVICE_NAME="${HEALTHCHECK_SERVICE_NAME:-trading-bot-autopilot-healthcheck.service}"
HEALTHCHECK_TIMER_NAME="${HEALTHCHECK_TIMER_NAME:-trading-bot-autopilot-healthcheck.timer}"
HEALTHCHECK_ON_BOOT="${HEALTHCHECK_ON_BOOT:-2min}"
HEALTHCHECK_INTERVAL="${HEALTHCHECK_INTERVAL:-5min}"
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

validate_unit_name "$SERVICE_NAME" ".service" "SERVICE_NAME"
validate_unit_name "$JOB_SERVICE_NAME" ".service" "JOB_SERVICE_NAME"
validate_unit_name "$HEALTHCHECK_SERVICE_NAME" ".service" "HEALTHCHECK_SERVICE_NAME"
validate_unit_name "$HEALTHCHECK_TIMER_NAME" ".timer" "HEALTHCHECK_TIMER_NAME"
validate_positive_integer "$AUTOPILOT_THREADS" "AUTOPILOT_THREADS"
validate_positive_integer "$AUTOPILOT_TASKS_MAX" "AUTOPILOT_TASKS_MAX"
validate_unit_value "$AUTOPILOT_MEMORY_MAX" "AUTOPILOT_MEMORY_MAX"
validate_unit_value "$AUTOPILOT_CPU_QUOTA" "AUTOPILOT_CPU_QUOTA"
validate_unit_value "$HEALTHCHECK_ON_BOOT" "HEALTHCHECK_ON_BOOT"
validate_unit_value "$HEALTHCHECK_INTERVAL" "HEALTHCHECK_INTERVAL"
validate_zero_or_one "$DRY_RUN" "DRY_RUN"

UNIT_FILE="$UNIT_DIR/$SERVICE_NAME"
JOB_SERVICE_FILE="$UNIT_DIR/$JOB_SERVICE_NAME"
HEALTHCHECK_SERVICE_FILE="$UNIT_DIR/$HEALTHCHECK_SERVICE_NAME"
HEALTHCHECK_TIMER_FILE="$UNIT_DIR/$HEALTHCHECK_TIMER_NAME"

if [ ! -x "$PYTHON" ]; then
  echo "Missing Python executable: $PYTHON" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "Missing autopilot config: $CONFIG" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR" "$REPO/runtime"

REPO_UNIT="$(systemd_quote "$REPO")"
PYTHON_UNIT="$(systemd_quote "$PYTHON")"
CONFIG_UNIT="$(systemd_quote "$CONFIG")"
ENV_FILE_UNIT="$(systemd_quote "-$REPO/.env")"
HEALTHCHECK_JSON_UNIT="$(systemd_quote "$REPO/runtime/healthcheck.json")"

echo "Validating autopilot config..."
"$PYTHON" -m src.autopilot.runtime --config "$CONFIG" --validate
echo "Checking autopilot readiness..."
"$PYTHON" -m src.autopilot.readiness --config "$CONFIG" \
  --output "$REPO/runtime/readiness_report.md" \
  --json-output "$REPO/runtime/readiness_report.json"

cat > "$UNIT_FILE" <<UNIT
[Unit]
Description=Trading Bot Autopilot
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory=$REPO_UNIT
EnvironmentFile=$ENV_FILE_UNIT
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=$AUTOPILOT_THREADS
Environment=OPENBLAS_NUM_THREADS=$AUTOPILOT_THREADS
Environment=MKL_NUM_THREADS=$AUTOPILOT_THREADS
Environment=NUMEXPR_NUM_THREADS=$AUTOPILOT_THREADS
Environment=LOKY_MAX_CPU_COUNT=$AUTOPILOT_THREADS
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
WorkingDirectory=$REPO_UNIT
EnvironmentFile=$ENV_FILE_UNIT
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=$AUTOPILOT_THREADS
Environment=OPENBLAS_NUM_THREADS=$AUTOPILOT_THREADS
Environment=MKL_NUM_THREADS=$AUTOPILOT_THREADS
Environment=NUMEXPR_NUM_THREADS=$AUTOPILOT_THREADS
Environment=LOKY_MAX_CPU_COUNT=$AUTOPILOT_THREADS
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
WorkingDirectory=$REPO_UNIT
EnvironmentFile=$ENV_FILE_UNIT
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=$AUTOPILOT_THREADS
Environment=OPENBLAS_NUM_THREADS=$AUTOPILOT_THREADS
Environment=MKL_NUM_THREADS=$AUTOPILOT_THREADS
Environment=NUMEXPR_NUM_THREADS=$AUTOPILOT_THREADS
Environment=LOKY_MAX_CPU_COUNT=$AUTOPILOT_THREADS
ExecStart=$PYTHON_UNIT -m src.autopilot.healthcheck --config $CONFIG_UNIT --output $HEALTHCHECK_JSON_UNIT
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

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete. Wrote unit files:"
  echo "  $UNIT_FILE"
  echo "  $JOB_SERVICE_FILE"
  echo "  $HEALTHCHECK_SERVICE_FILE"
  echo "  $HEALTHCHECK_TIMER_FILE"
  exit 0
fi

TARGET_USER="$(id -un)"
ensure_user_linger "$TARGET_USER"

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"
systemctl --user enable --now "$JOB_SERVICE_NAME"
systemctl --user enable --now "$HEALTHCHECK_TIMER_NAME"

systemctl --user status "$SERVICE_NAME" --no-pager
systemctl --user status "$JOB_SERVICE_NAME" --no-pager
systemctl --user list-timers "$HEALTHCHECK_TIMER_NAME" --no-pager
