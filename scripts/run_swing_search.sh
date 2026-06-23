#!/bin/bash
# 15m swing-trade campaign, the last experiment within existing data:
#   1) refresh candles + rebuild the 4h/1d/1w indicator parquets (the only
#      ones still lacking flow features; 15m/30m/1h were rebuilt already)
#   2) retire the stale 15m training table (predates flow features); the
#      first search auto-rebuilds it from the fresh parquets
#   3) search A: flow-only feature universe (direct hypothesis test)
#   4) search B: full universe (everything competes)
# Both searches: futures costs, swing grids (TP 0.5-2%, holds up to 1 day),
# walk-forward + holdout, checkpointed. Total ~1.5h. Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Step 1/4: rebuilding 4h/1d/1w indicators with flow features ==="
python -m src.update_candles --timeframes 4h 1d 1w 2>&1 | tee outputs/rebuild_swing.log

echo "=== Step 2/4: retiring stale 15m training table ==="
if [ -f data/processed/train_15m_indicators.parquet ]; then
  mv data/processed/train_15m_indicators.parquet data/processed/train_15m_indicators.parquet.bak
  echo "Retired -> train_15m_indicators.parquet.bak"
fi

SWING_ARGS=(
  --base-tf 15m
  --fee-bps 5 --slippage-bps 2
  --horizon 16 --horizon 32 --horizon 96
  --take-profit 0.005 --take-profit 0.008 --take-profit 0.012 --take-profit 0.02
  --stop-loss 0.004 --stop-loss 0.006 --stop-loss 0.01 --stop-loss 0.015
  --walk-forward --n-jobs 5
)

echo "=== Step 3/4: search A — flow-only universe ==="
mkdir -p outputs/search_v6_15m_flowonly
python -m src.day_trade_search \
  --output-dir outputs/search_v6_15m_flowonly \
  --feature-pattern 'cvd_|taker_imbalance|taker_buy_ratio|volume_z_|trades_z_|avg_trade_size' \
  "${SWING_ARGS[@]}" \
  2>&1 | tee outputs/search_v6_15m_flowonly/search.log

echo "=== Step 4/4: search B — full universe ==="
mkdir -p outputs/search_v6_15m_full
python -m src.day_trade_search \
  --output-dir outputs/search_v6_15m_full \
  "${SWING_ARGS[@]}" \
  2>&1 | tee outputs/search_v6_15m_full/search.log

echo "=== Swing campaign complete — both result sets ready for review ==="
