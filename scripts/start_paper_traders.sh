#!/bin/bash
# Sets up BOTH paper traders end-to-end:
#   1) exports the CVD swing family to its own artifact (sandboxed from the BTC book)
#   2) installs both 15-minute cron lines (idempotent — safe to rerun)
#   3) runs one immediate cycle of each bot as a sanity check
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== 1/3: exporting CVD swing strategies ==="
python -m src.export_strategies \
  --search-dir outputs/search_v6_15m_flowonly \
  --top-k 2 \
  --output outputs/active_strategies_flow.json

echo "=== 2/3: installing cron lines ==="
REPO="/Users/henriquegomes/Git/trading-bot"
BTC_LINE="*/15 * * * * cd $REPO && .venv/bin/python -m src.run_bot >> outputs/cron.log 2>&1"
FLOW_LINE="*/15 * * * * cd $REPO && .venv/bin/python -m src.run_bot --strategies outputs/active_strategies_flow.json --state-file outputs/bot_state_flow.json --trade-log outputs/paper_trades_flow.csv >> outputs/cron_flow.log 2>&1"
current="$(crontab -l 2>/dev/null || true)"
echo "$current" | grep -Fq "$BTC_LINE" || current="${current}"$'\n'"$BTC_LINE"
echo "$current" | grep -Fq "$FLOW_LINE" || current="${current}"$'\n'"$FLOW_LINE"
printf '%s\n' "$current" | sed '/^$/d' | crontab -
crontab -l

echo "=== 3/3: one immediate cycle of each bot ==="
python -m src.run_bot
python -m src.run_bot \
  --strategies outputs/active_strategies_flow.json \
  --state-file outputs/bot_state_flow.json \
  --trade-log outputs/paper_trades_flow.csv

echo "=== Both paper traders are live. ==="
