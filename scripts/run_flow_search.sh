#!/bin/bash
# Flow-features day-trade search: retires the stale 5m training table (the
# search rebuilds it from the freshly built indicator parquets), then runs the
# walk-forward search at futures costs. Safe to rerun; add --resume by hand if
# a run was interrupted mid-scoring.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ -f data/processed/train_5m_indicators.parquet ]; then
  mv data/processed/train_5m_indicators.parquet data/processed/train_5m_indicators.parquet.bak
  echo "Retired old training table -> train_5m_indicators.parquet.bak"
fi

mkdir -p outputs/search_v5_5m_flow
python -m src.day_trade_search \
  --output-dir outputs/search_v5_5m_flow \
  --fee-bps 5 --slippage-bps 2 \
  --walk-forward --n-jobs 5 \
  2>&1 | tee outputs/search_v5_5m_flow/search.log
