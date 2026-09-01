#!/bin/sh

set +e

echo '=== UTC TIME ==='
date -u --iso-8601=seconds
echo '=== KERNEL ==='
uname -a
echo '=== UPTIME ==='
uptime
echo '=== MEMORY ==='
free -h
echo '=== FILESYSTEMS ==='
df -hT
echo '=== SYSTEM UNIT FILES ==='
systemctl list-unit-files --type=service --no-pager | grep -E 'trading-platform|trading-bot' || true
echo '=== SYSTEM SERVICES ==='
systemctl list-units --all --type=service --no-pager | grep -E 'trading-platform|trading-bot' || true
echo '=== SYSTEM TIMERS ==='
systemctl list-timers --all --no-pager | grep -E 'trading-platform|trading-bot' || true
echo '=== SERVICE DETAILS ==='
for unit in $(systemctl list-units --all --type=service --plain --no-legend | awk '$1 ~ /^(trading-platform|trading-bot)/ {print $1}'); do
    echo "### $unit"
    systemctl show "$unit" \
        -p Id \
        -p LoadState \
        -p ActiveState \
        -p SubState \
        -p UnitFileState \
        -p FragmentPath \
        -p ExecMainStartTimestamp \
        -p ExecMainExitTimestamp \
        -p ExecMainCode \
        -p ExecMainStatus \
        -p NRestarts \
        -p MemoryCurrent \
        -p CPUUsageNSec
done
echo '=== RECENT PLATFORM WARNINGS ==='
journalctl \
    --since '2026-08-30 00:00:00' \
    --no-pager \
    -p warning..alert \
    -u 'trading-platform@*.service' \
    -u 'trading-platform-research@*.service' \
    -u 'trading-platform-agent@*.service' \
    -u trading-platform-migration.service \
    -n 300 2>&1 || true
echo '=== ENVIRONMENT FILE METADATA ==='
find /etc/trading-platform \
    -maxdepth 1 \
    -type f \
    -name '*.env' \
    -printf '%p %u:%g mode=%m bytes=%s\n' 2>/dev/null | sort
echo '=== ACTIVE PROCESS INTERPRETERS ==='
ps -eo user=,pid=,stat=,etime=,args= |
    grep -E '/home/alfred/trading-bot/\.venv-(runtime|research|agent)/bin/python' |
    grep -v grep || true
