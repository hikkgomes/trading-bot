#!/bin/bash
# Run FROM the Mac: ships the execution side to the Linux server and sets it
# up end-to-end, then removes the Mac cron lines so the server owns the
# paper traders. Usage:
#   bash scripts/deploy_to_server.sh user@server-address
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: $0 user@server-address   (e.g. henrique@192.168.1.50)"
  exit 1
fi
TARGET="$1"
cd "$(dirname "$0")/.."
# Mirror everything to a log file so failures are diagnosable after the fact.
exec > >(tee outputs/deploy.log) 2>&1

echo "=== 1/4: copying code + strategy artifacts to $TARGET ==="
ssh "$TARGET" 'mkdir -p ~/trading-bot/outputs ~/trading-bot/scripts'
rsync -az --delete src "$TARGET":~/trading-bot/
rsync -az build_binance_indicator_dataset.py requirements-bot.txt "$TARGET":~/trading-bot/
rsync -az scripts/setup_server.sh "$TARGET":~/trading-bot/scripts/
rsync -az outputs/active_strategies.json outputs/active_strategies_flow.json "$TARGET":~/trading-bot/outputs/

echo "=== 2/4: running server setup (you may be asked for the server's sudo password) ==="
ssh -t "$TARGET" 'bash ~/trading-bot/scripts/setup_server.sh'

echo "=== 3/4: removing Mac cron lines (server owns the bots now) ==="
crontab -l 2>/dev/null | grep -v 'src.run_bot' | crontab - || true
crontab -l 2>/dev/null || echo "(mac crontab now empty)"

echo "=== 4/4: done ==="
echo "Check on it any time with:"
echo "  ssh $TARGET 'tail -5 ~/trading-bot/outputs/cron.log ~/trading-bot/outputs/cron_flow.log'"
