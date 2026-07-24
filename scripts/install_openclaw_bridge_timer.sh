#!/bin/bash
# Install a credential-free timer that exports context and ingests proposals.
set -euo pipefail
umask 077

REPO="${REPO:-$HOME/trading-bot}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
SERVICE_NAME="${SERVICE_NAME:-trading-bot-openclaw-bridge.service}"
TIMER_NAME="${TIMER_NAME:-trading-bot-openclaw-bridge.timer}"
INTERVAL="${INTERVAL:-5min}"
UNIT_DIR="${UNIT_DIR:-$HOME/.config/systemd/user}"
DRY_RUN="${DRY_RUN:-0}"
OPENCLAW_GROUP="${OPENCLAW_GROUP:-}"
OPENCLAW_USER="${OPENCLAW_USER:-}"
TARGET_UNIT_DIR="$UNIT_DIR"

for value in "$SERVICE_NAME" "$TIMER_NAME" "$INTERVAL"; do
  if [[ -z "$value" || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "Unit settings must be non-empty and contain no control characters" >&2
    exit 1
  fi
done
if [[ "$SERVICE_NAME" == */* || "$SERVICE_NAME" != *.service ]]; then
  echo "SERVICE_NAME must be a .service unit name without slashes" >&2
  exit 1
fi
if [[ "$TIMER_NAME" == */* || "$TIMER_NAME" != *.timer ]]; then
  echo "TIMER_NAME must be a .timer unit name without slashes" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ -n "$OPENCLAW_GROUP" && ! "$OPENCLAW_GROUP" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "OPENCLAW_GROUP must be a plain Unix group name" >&2
  exit 1
fi
if [[ -n "$OPENCLAW_USER" && ! "$OPENCLAW_USER" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "OPENCLAW_USER must be a plain Unix user name" >&2
  exit 1
fi
if [[ -n "$OPENCLAW_USER" && -z "$OPENCLAW_GROUP" ]] || \
   [[ -n "$OPENCLAW_GROUP" && -z "$OPENCLAW_USER" ]]; then
  echo "Shared-user mode requires both OPENCLAW_USER and OPENCLAW_GROUP" >&2
  exit 1
fi
if [ ! -x "$PYTHON" ]; then
  echo "Missing Python executable: $PYTHON" >&2
  exit 1
fi
if [ ! -d "$REPO" ] || [ -L "$REPO" ]; then
  echo "REPO must be a real directory, not a symlink: $REPO" >&2
  exit 1
fi

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

systemd_scalar() {
  local value="$1"
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "Systemd scalar values must not contain control characters" >&2
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

group_list_contains() {
  local expected="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$expected" ]]; then return 0; fi
  done
  return 1
}

foreign_path_is_traversable() {
  local path="$1"
  local permissions owner owning_group execute_bit acl_permissions acl_mask
  acl_permissions="$(
    getfacl -cp -- "$path" | awk -F: -v user="$OPENCLAW_USER" \
      '$1 == "user" && $2 == user {print $3; exit}'
  )"
  if [[ -n "$acl_permissions" ]]; then
    acl_permissions="${acl_permissions%%[[:space:]]*}"
    acl_mask="$(getfacl -cp -- "$path" | awk -F: '$1 == "mask" {print $3; exit}')"
    acl_mask="${acl_mask%%[[:space:]]*}"
    [[ "$acl_permissions" == *x* && ( -z "$acl_mask" || "$acl_mask" == *x* ) ]]
    return
  fi
  permissions="$(stat -Lc '%A' "$path")"
  owner="$(stat -Lc '%U' "$path")"
  owning_group="$(stat -Lc '%G' "$path")"
  if [[ "$owner" == "$OPENCLAW_USER" ]]; then
    execute_bit="${permissions:3:1}"
  elif group_list_contains "$owning_group" "${OPENCLAW_GROUPS[@]}"; then
    execute_bit="${permissions:6:1}"
  else
    execute_bit="${permissions:9:1}"
  fi
  [[ "$execute_bit" == "x" || "$execute_bit" == "s" || "$execute_bit" == "t" ]]
}

assert_owned_real_path() {
  local path="$1"
  local current_user="$2"
  if [[ -L "$path" || ! -e "$path" ]]; then
    echo "Shared-user ACL target must exist and must not be a symlink: $path" >&2
    exit 1
  fi
  if [[ "$(stat -Lc '%U' "$path")" != "$current_user" ]]; then
    echo "Shared-user ACL target must be owned by $current_user: $path" >&2
    exit 1
  fi
}

deny_immediate_children() {
  local parent="$1"
  local current_user="$2"
  shift 2
  local child allowed exception
  setfacl -m "d:u:$OPENCLAW_USER:---" -- "$parent"
  while IFS= read -r -d '' child; do
    allowed=0
    for exception in "$@"; do
      if [[ "$child" == "$exception" ]]; then allowed=1; break; fi
    done
    if [[ "$allowed" == "1" ]]; then continue; fi
    assert_owned_real_path "$child" "$current_user"
    setfacl -m "u:$OPENCLAW_USER:---" -- "$child"
  done < <(find "$parent" -mindepth 1 -maxdepth 1 -print0)
}

grant_minimal_parent_traversal() {
  local current_user="$1"
  local traverse_path owner
  traverse_path="$REPO"
  while true; do
    owner="$(stat -Lc '%U' "$traverse_path")"
    if [[ "$owner" == "$current_user" ]]; then
      setfacl -m "u:$OPENCLAW_USER:--x" -- "$traverse_path"
    elif ! foreign_path_is_traversable "$traverse_path"; then
      echo "OpenClaw cannot traverse foreign-owned parent: $traverse_path" >&2
      echo "Ask an administrator to grant execute-only traversal to $OPENCLAW_USER, then rerun." >&2
      exit 1
    fi
    if [[ "$traverse_path" == "/" ]]; then break; fi
    traverse_path="$(dirname "$traverse_path")"
  done
}

mkdir -p "$TARGET_UNIT_DIR"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p \
    "$REPO/runtime/openclaw" \
    "$REPO/runtime/research_inbox/openclaw/incoming" \
    "$REPO/runtime/research_inbox/openclaw/accepted" \
    "$REPO/runtime/research_inbox/openclaw/rejected" \
    "$REPO/runtime/research_inbox/openclaw/archive"
  for path in \
    "$REPO/runtime" \
    "$REPO/runtime/openclaw" \
    "$REPO/runtime/research_inbox" \
    "$REPO/runtime/research_inbox/openclaw" \
    "$REPO/runtime/research_inbox/openclaw/incoming" \
    "$REPO/runtime/research_inbox/openclaw/accepted" \
    "$REPO/runtime/research_inbox/openclaw/rejected" \
    "$REPO/runtime/research_inbox/openclaw/archive"; do
    if [[ -L "$path" || ! -d "$path" ]]; then
      echo "Bridge path must be a real directory, not a symlink: $path" >&2
      exit 1
    fi
  done
  chmod 700 "$REPO/runtime/openclaw" "$REPO/runtime/research_inbox/openclaw"/*
fi

SHARED_GROUP_FLAG=0
if [[ -n "$OPENCLAW_GROUP" ]]; then
  SHARED_GROUP_FLAG=1
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run: shared-user ownership, private-mode, and ACL changes were not applied."
  else
    if ! command -v setfacl >/dev/null 2>&1 || ! command -v getfacl >/dev/null 2>&1; then
      echo "OPENCLAW_GROUP mode requires setfacl/getfacl (install the Linux acl package)" >&2
      exit 1
    fi
    if ! getent group "$OPENCLAW_GROUP" >/dev/null 2>&1; then
      echo "OPENCLAW_GROUP does not exist: $OPENCLAW_GROUP" >&2
      exit 1
    fi
    if ! id "$OPENCLAW_USER" >/dev/null 2>&1; then
      echo "OPENCLAW_USER does not exist: $OPENCLAW_USER" >&2
      exit 1
    fi
    CURRENT_USER="$(id -un)"
    if [[ "$CURRENT_USER" == "$OPENCLAW_USER" ]]; then
      echo "OPENCLAW_USER must be separate from the trading-service user" >&2
      exit 1
    fi
    read -r -a BRIDGE_GROUPS <<< "$(id -nG)"
    read -r -a OPENCLAW_GROUPS <<< "$(id -nG "$OPENCLAW_USER")"
    if ! group_list_contains "$OPENCLAW_GROUP" "${BRIDGE_GROUPS[@]}"; then
      echo "Current bridge user must belong to OPENCLAW_GROUP=$OPENCLAW_GROUP" >&2
      exit 1
    fi
    if ! group_list_contains "$OPENCLAW_GROUP" "${OPENCLAW_GROUPS[@]}"; then
      echo "$OPENCLAW_USER must belong to OPENCLAW_GROUP=$OPENCLAW_GROUP" >&2
      exit 1
    fi

    grant_minimal_parent_traversal "$CURRENT_USER"
    assert_owned_real_path "$REPO/runtime" "$CURRENT_USER"
    assert_owned_real_path "$REPO/runtime/openclaw" "$CURRENT_USER"
    assert_owned_real_path "$REPO/runtime/research_inbox" "$CURRENT_USER"
    assert_owned_real_path "$REPO/runtime/research_inbox/openclaw" "$CURRENT_USER"
    assert_owned_real_path "$REPO/runtime/research_inbox/openclaw/incoming" "$CURRENT_USER"

    deny_immediate_children "$REPO" "$CURRENT_USER" "$REPO/runtime"
    setfacl -m "u:$OPENCLAW_USER:--x" -- "$REPO/runtime"
    deny_immediate_children "$REPO/runtime" "$CURRENT_USER" \
      "$REPO/runtime/openclaw" "$REPO/runtime/research_inbox"
    setfacl -m "u:$OPENCLAW_USER:--x" -- "$REPO/runtime/research_inbox"
    deny_immediate_children "$REPO/runtime/research_inbox" "$CURRENT_USER" \
      "$REPO/runtime/research_inbox/openclaw"
    setfacl -m "u:$OPENCLAW_USER:--x" -- \
      "$REPO/runtime/research_inbox/openclaw"
    deny_immediate_children "$REPO/runtime/research_inbox/openclaw" "$CURRENT_USER" \
      "$REPO/runtime/research_inbox/openclaw/incoming"

    chgrp "$OPENCLAW_GROUP" \
      "$REPO/runtime/openclaw" \
      "$REPO/runtime/research_inbox/openclaw" \
      "$REPO/runtime/research_inbox/openclaw/incoming"
    chmod 2750 "$REPO/runtime/openclaw"
    chmod 2710 "$REPO/runtime/research_inbox/openclaw"
    chmod 2770 "$REPO/runtime/research_inbox/openclaw/incoming"
    chmod 700 \
      "$REPO/runtime/research_inbox/openclaw/accepted" \
      "$REPO/runtime/research_inbox/openclaw/rejected" \
      "$REPO/runtime/research_inbox/openclaw/archive"
    setfacl -m "u:$OPENCLAW_USER:r-x" -- "$REPO/runtime/openclaw"
    if [[ -e "$REPO/runtime/openclaw/research_context.json" ]]; then
      assert_owned_real_path "$REPO/runtime/openclaw/research_context.json" "$CURRENT_USER"
      setfacl -m "u:$OPENCLAW_USER:r--" -- \
        "$REPO/runtime/openclaw/research_context.json"
    fi
    setfacl -m "u:$OPENCLAW_USER:rwx" -- \
      "$REPO/runtime/research_inbox/openclaw/incoming"
    # Do not rely on a long-running user-systemd manager refreshing its
    # supplementary groups after account provisioning. Files created by the
    # separate OpenClaw identity inherit an explicit ACL for the bridge user.
    setfacl -m "d:u:$CURRENT_USER:rwx" -- \
      "$REPO/runtime/research_inbox/openclaw/incoming"
    setfacl -m 'd:g::r-x,d:m::r-x,d:o::---' -- "$REPO/runtime/openclaw"
    setfacl -m 'd:g::rwx,d:m::rwx,d:o::---' -- \
      "$REPO/runtime/research_inbox/openclaw/incoming"
  fi
fi

REPO_UNIT="$(systemd_quote "$REPO")"
REPO_WORKING_DIRECTORY="$(systemd_scalar "$REPO")"
PYTHON_UNIT="$(systemd_quote "$PYTHON")"
OPENCLAW_UNIT="$(systemd_quote "$REPO/runtime/openclaw")"
INBOX_UNIT="$(systemd_quote "$REPO/runtime/research_inbox/openclaw")"
TRADING_ENV_UNIT="$(systemd_quote "$REPO/.env")"
APPROVALS_UNIT="$(systemd_quote "$REPO/runtime/approvals.json")"
CONTROL_UNIT="$(systemd_quote "$REPO/runtime/operator-control/control.json")"
OUTPUTS_UNIT="$(systemd_quote "$REPO/outputs")"
DATA_UNIT="$(systemd_quote "$REPO/data")"
prepare_unit_staging
SERVICE_FILE="$UNIT_DIR/$SERVICE_NAME"
TIMER_FILE="$UNIT_DIR/$TIMER_NAME"

cat > "$SERVICE_FILE" <<UNIT
[Unit]
Description=Trading Bot Sanitized OpenClaw Research Bridge
After=trading-bot-autopilot-jobs.service

[Service]
Type=oneshot
WorkingDirectory=$REPO_WORKING_DIRECTORY
Environment=PYTHONUNBUFFERED=1
Environment=OPENCLAW_SHARED_GROUP=$SHARED_GROUP_FLAG
ExecStart=$PYTHON_UNIT -m src.autopilot.openclaw_bridge export
ExecStart=$PYTHON_UNIT -m src.autopilot.openclaw_bridge ingest
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
ReadWritePaths=$OPENCLAW_UNIT $INBOX_UNIT
InaccessiblePaths=$TRADING_ENV_UNIT
InaccessiblePaths=-$APPROVALS_UNIT
InaccessiblePaths=-$CONTROL_UNIT
InaccessiblePaths=-$OUTPUTS_UNIT
InaccessiblePaths=-$DATA_UNIT
RestrictAddressFamilies=AF_UNIX
RestrictSUIDSGID=true
LockPersonality=true
StandardOutput=journal
StandardError=journal
UNIT

cat > "$TIMER_FILE" <<UNIT
[Unit]
Description=Refresh the sanitized OpenClaw bridge

[Timer]
OnBootSec=2min
OnUnitActiveSec=$INTERVAL
AccuracySec=30s
Persistent=true
Unit=$SERVICE_NAME

[Install]
WantedBy=timers.target
UNIT

verify_unit_files "$SERVICE_FILE" "$TIMER_FILE"

publish_unit_files "$SERVICE_FILE" "$TIMER_FILE"
SERVICE_FILE="$UNIT_DIR/$SERVICE_NAME"
TIMER_FILE="$UNIT_DIR/$TIMER_NAME"

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete: wrote $SERVICE_FILE and $TIMER_FILE"
  exit 0
fi

systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"
systemctl --user start "$SERVICE_NAME"
systemctl --user status "$SERVICE_NAME" "$TIMER_NAME" --no-pager
