#!/bin/bash
# Flow-ONLY day-trade search: restricts the mined feature universe to the
# order-flow features (CVD, taker imbalance, volume/trades/size z-scores),
# so the flow hypothesis is tested directly instead of competing against
# ~2,300 TA-Lib columns in the feature-ranking funnel.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p outputs/search_v5_5m_flowonly
python -m src.day_trade_search \
  --output-dir outputs/search_v5_5m_flowonly \
  --fee-bps 5 --slippage-bps 2 \
  --feature-pattern 'cvd_|taker_imbalance|taker_buy_ratio|volume_z_|trades_z_|avg_trade_size' \
  --walk-forward --n-jobs 5 \
  2>&1 | tee outputs/search_v5_5m_flowonly/search.log
