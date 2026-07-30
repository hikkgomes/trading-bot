#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/alfred/trading-bot}"
OPENCLAW_BIN="${OPENCLAW_BIN:-/home/alfred/.npm-global/bin/openclaw}"
OPENCLAW_WORKSPACE="${OPENCLAW_WORKSPACE:-/home/alfred/.openclaw/workspace}"
MODEL="${MODEL:-openai/gpt-5.5}"
TIMEZONE="${TIMEZONE:-Europe/Madrid}"
REVIEW_CRON="${REVIEW_CRON:-45 0,6,12,18 * * *}"
EVENT_INTERVAL="${EVENT_INTERVAL:-15m}"
TELEGRAM_TO="${TELEGRAM_TO:-}"
DRY_RUN="${DRY_RUN:-0}"
REVIEW_NAME="Trading Research Supervisor"
EVENT_NAME="Trading Research Event Watcher"
MANAGED_START="<!-- trading-bot-alfred:start -->"
MANAGED_END="<!-- trading-bot-alfred:end -->"

if [[ "$REPO" != /* || "$OPENCLAW_WORKSPACE" != /* || "$OPENCLAW_BIN" != /* ]]; then
  echo "REPO, OPENCLAW_WORKSPACE, and OPENCLAW_BIN must be absolute paths" >&2
  exit 1
fi
if [[ "$REPO" == "/" || "$OPENCLAW_WORKSPACE" == "/" ]]; then
  echo "refusing broad root path" >&2
  exit 1
fi
if [[ ! -f "$REPO/config/openclaw_daily_review_prompt.md" ||
      ! -f "$REPO/config/alfred_trading_operator.md" ]]; then
  echo "Alfred integration files are missing from $REPO" >&2
  exit 1
fi

MESSAGE="From $REPO, read and follow $REPO/config/openclaw_daily_review_prompt.md exactly. This is an autonomous research-supervisor cycle and must always end with an audited review receipt."
if [[ -z "$TELEGRAM_TO" && -f "$OPENCLAW_WORKSPACE/USER_IDS.md" ]]; then
  TELEGRAM_TO="$(sed -nE 's/^- \\*\\*Henrique\\*\\*: ([0-9]+)$/\\1/p' \
    "$OPENCLAW_WORKSPACE/USER_IDS.md" | head -n 1)"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Would install managed Alfred instructions in $OPENCLAW_WORKSPACE/AGENTS.md"
  echo "Would schedule $REVIEW_NAME at $REVIEW_CRON ($TIMEZONE)"
  echo "Would schedule $EVENT_NAME every $EVENT_INTERVAL"
  echo "Would deliver meaningful findings to ${TELEGRAM_TO:-the last active channel}"
  exit 0
fi

if [[ ! -x "$OPENCLAW_BIN" ]]; then
  echo "OpenClaw executable is missing or not executable: $OPENCLAW_BIN" >&2
  exit 1
fi
mkdir -p "$OPENCLAW_WORKSPACE"
AGENTS_FILE="$OPENCLAW_WORKSPACE/AGENTS.md"
touch "$AGENTS_FILE"

AGENTS_FILE="$AGENTS_FILE" INSTRUCTIONS_FILE="$REPO/config/alfred_trading_operator.md" \
MANAGED_START="$MANAGED_START" MANAGED_END="$MANAGED_END" python3 - <<'PY'
import os
from pathlib import Path

target = Path(os.environ["AGENTS_FILE"])
instructions = Path(os.environ["INSTRUCTIONS_FILE"]).read_text(encoding="utf-8").strip()
start = os.environ["MANAGED_START"]
end = os.environ["MANAGED_END"]
current = target.read_text(encoding="utf-8") if target.exists() else ""
block = f"{start}\n{instructions}\n{end}"
if start in current or end in current:
    if current.count(start) != 1 or current.count(end) != 1:
        raise SystemExit("managed Alfred instruction markers are malformed")
    prefix, remainder = current.split(start, 1)
    _, suffix = remainder.split(end, 1)
    updated = prefix.rstrip() + "\n\n" + block + suffix
else:
    updated = current.rstrip() + "\n\n" + block + "\n"
temporary = target.with_name(f".{target.name}.trading-bot.tmp")
temporary.write_text(updated, encoding="utf-8")
temporary.chmod(0o600)
temporary.replace(target)
PY

cron_json="$(mktemp)"
trap 'rm -f "$cron_json"' EXIT
delivery_args=(--announce --channel last --best-effort-deliver)
if [[ -n "$TELEGRAM_TO" ]]; then
  delivery_args=(--announce --channel telegram --to "$TELEGRAM_TO" --best-effort-deliver)
fi
"$OPENCLAW_BIN" cron list --json > "$cron_json"
review_id="$(python3 - "$cron_json" "$REVIEW_NAME" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [item["id"] for item in payload.get("jobs", []) if item.get("name") == sys.argv[2]]
if len(matches) > 1:
    raise SystemExit("multiple Alfred research supervisor cron jobs exist")
print(matches[0] if matches else "")
PY
)"

if [[ -n "$review_id" ]]; then
  "$OPENCLAW_BIN" cron edit "$review_id" \
    --name "$REVIEW_NAME" --cron "$REVIEW_CRON" --tz "$TIMEZONE" \
    --session isolated --wake now --message "$MESSAGE" --model "$MODEL" \
    --thinking medium --timeout-seconds 900 "${delivery_args[@]}" --enable
else
  review_json="$("$OPENCLAW_BIN" cron add --json \
    --name "$REVIEW_NAME" --cron "$REVIEW_CRON" --tz "$TIMEZONE" \
    --session isolated --wake now --message "$MESSAGE" --model "$MODEL" \
    --thinking medium --timeout-seconds 900 "${delivery_args[@]}")"
  review_id="$(python3 -c 'import json,sys; p=json.load(sys.stdin); print(p.get("id") or (p.get("job") or {}).get("id") or "")' <<<"$review_json")"
  if [[ -z "$review_id" ]]; then
    echo "OpenClaw did not return the created review job id" >&2
    exit 1
  fi
fi

"$OPENCLAW_BIN" cron list --json > "$cron_json"
event_id="$(python3 - "$cron_json" "$EVENT_NAME" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [item["id"] for item in payload.get("jobs", []) if item.get("name") == sys.argv[2]]
if len(matches) > 1:
    raise SystemExit("multiple Alfred event watcher cron jobs exist")
print(matches[0] if matches else "")
PY
)"
event_command="cd $REPO && .venv/bin/python -m src.autopilot.openclaw_bridge claim-event && $OPENCLAW_BIN cron run $review_id"
if [[ -n "$event_id" ]]; then
  "$OPENCLAW_BIN" cron edit "$event_id" \
    --name "$EVENT_NAME" --every "$EVENT_INTERVAL" \
    --command "$event_command" --command-cwd "$REPO" \
    --timeout-seconds 120 --no-output-timeout-seconds 90 \
    --output-max-bytes 16384 --no-deliver --enable
else
  "$OPENCLAW_BIN" cron add \
    --name "$EVENT_NAME" --every "$EVENT_INTERVAL" \
    --command "$event_command" --command-cwd "$REPO" \
    --timeout-seconds 120 --no-output-timeout-seconds 90 \
    --output-max-bytes 16384 --no-deliver
fi

# Disable the superseded once-daily job if it still exists.
"$OPENCLAW_BIN" cron list --json > "$cron_json"
while IFS= read -r legacy_id; do
  [[ -n "$legacy_id" ]] && "$OPENCLAW_BIN" cron disable "$legacy_id"
done < <(python3 - "$cron_json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for item in payload.get("jobs", []):
    if item.get("name") == "Trading Research Daily Review":
        print(item["id"])
PY
)

"$OPENCLAW_BIN" cron list --json
