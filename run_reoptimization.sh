#!/bin/bash
# Exit immediately if any command exits with a non-zero status
set -e

# Load python environment (assuming virtualenv is named .venv)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "$(date) === Starting Re-optimization Pipeline ==="

echo "$(date) [1/4] Fetching recent candles and building indicators..."
python -m src.update_candles

echo "$(date) [2/4] Merging indicators into base 5m timeframe dataset..."
python -m src.build_dataset --input-kind indicators --base-timeframe 5m --output-path data/processed/train_5m_indicators.parquet --skip-duplicate-scan

echo "$(date) [3/4] Labeling trades with TP/SL targets..."
python -m src.label_trades --input-path data/processed/train_5m_indicators.parquet --output-path data/processed/train_5m_indicators_labels.parquet --base-timeframe 5m

echo "$(date) [4/4] Running strategy grid search and updating active rules..."
python -m src.day_trade_search --base-tf 5m --ranking-method spearman --skip-duplicate-scan --rank-sample-rows 25000 --output-dir outputs/test_day_trade_search

echo "$(date) === Re-optimization Completed Successfully! ==="
