#!/bin/bash
# Runs ON the Linux server (the deploy script invokes it over ssh).
# Installs system deps + TA-Lib C library, creates the venv, installs the
# minimal bot requirements, schedules both paper traders via cron, and runs
# one sanity cycle of each. Idempotent — safe to rerun.
set -euo pipefail
REPO="$HOME/trading-bot"
cd "$REPO"

echo "=== System packages ==="
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-dev build-essential wget cron

echo "=== TA-Lib C library ==="
if ! ldconfig -p | grep -qi "libta[_-]lib"; then
  TMP="$(mktemp -d)"
  cd "$TMP"
  wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
  tar xzf ta-lib-0.4.0-src.tar.gz
  cd ta-lib
  ./configure --prefix=/usr
  # TA-Lib 0.4.0's automake setup is NOT parallel-safe (gen_code race); build single-threaded.
  make
  sudo make install
  sudo ldconfig
  cd "$REPO"
  rm -rf "$TMP"
else
  echo "TA-Lib C library already installed."
fi

echo "=== Python environment ==="
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "TA-Lib<0.5" || .venv/bin/pip install TA-Lib
.venv/bin/pip install -r requirements-bot.txt

echo "=== Cron (both paper traders, every 15 min) ==="
BTC_LINE="*/15 * * * * cd $REPO && .venv/bin/python -m src.run_bot >> outputs/cron.log 2>&1"
FLOW_LINE="*/15 * * * * cd $REPO && .venv/bin/python -m src.run_bot --strategies outputs/active_strategies_flow.json --state-file outputs/bot_state_flow.json --trade-log outputs/paper_trades_flow.csv >> outputs/cron_flow.log 2>&1"
current="$(crontab -l 2>/dev/null || true)"
echo "$current" | grep -Fq "$BTC_LINE" || current="${current}"$'\n'"$BTC_LINE"
echo "$current" | grep -Fq "$FLOW_LINE" || current="${current}"$'\n'"$FLOW_LINE"
printf '%s\n' "$current" | sed '/^$/d' | crontab -
crontab -l

echo "=== Sanity cycle of each bot ==="
mkdir -p outputs
.venv/bin/python -m src.run_bot
.venv/bin/python -m src.run_bot \
  --strategies outputs/active_strategies_flow.json \
  --state-file outputs/bot_state_flow.json \
  --trade-log outputs/paper_trades_flow.csv

echo "=== Server setup complete. Paper traders run 24/7 from here. ==="
