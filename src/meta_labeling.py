import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.day_trade_search import combined_mask, numeric_feature_columns
from src.discover_patterns import Condition
from src.model_signals import walk_forward_model_signals
from src.walk_forward import WalkForwardConfig

LOGGER = logging.getLogger(__name__)


def _infer_params(row: pd.Series) -> tuple[str, int, float, float]:
    return (
        str(row["direction"]),
        int(row["horizon_bars"]),
        float(row["take_profit"]),
        float(row["stop_loss"]),
    )


def _label_column(direction: str, horizon: int, tp: float, sl: float) -> str:
    return f"label_{direction}_tp{int(round(tp * 10_000))}_sl{int(round(sl * 10_000))}_h{horizon}"


def _conditions(row: pd.Series) -> Sequence[Condition]:
    return tuple(Condition(**payload) for payload in json.loads(row["conditions_json"]))


def run(
    rules_path: Path,
    labels_path: Path,
    output_path: Path,
    top_k: int = 50,
    min_rule_rows: int = 500,
    fee_bps: float = 5.0,
    slippage_bps: float = 1.0,
    base_timeframe: str = "5m",
) -> Path:
    rules = pd.read_csv(rules_path).head(top_k)
    if "conditions_json" not in rules.columns:
        raise ValueError(
            "rules file must include conditions_json; use ranked_strategies.csv or scored_strategies_all.csv"
        )
    data = pd.read_parquet(labels_path).sort_values("timestamp").reset_index(drop=True)
    features = numeric_feature_columns(data, base_prefix=f"tf_{base_timeframe}_")
    fee_cost = 2 * ((fee_bps + slippage_bps) / 10_000)
    parts = []
    for rule_id, row in rules.iterrows():
        direction, horizon, tp, sl = _infer_params(row)
        label_column = _label_column(direction, horizon, tp, sl)
        if label_column not in data.columns:
            # Horizons are denominated in BASE-TF bars: a 15m-search rule's h4
            # means 1 hour, while the 5m label table's h4 means 20 minutes.
            # Mixing them silently would train the filter on the wrong trade
            # definition, so refuse loudly instead of skipping quietly.
            LOGGER.warning(
                "Skipping rule %s: label column %s not in %s. Make sure the rules "
                "file and the labels table come from the SAME base timeframe.",
                rule_id,
                label_column,
                labels_path,
            )
            continue
        masked = data.loc[combined_mask(data, _conditions(row))].copy()
        if len(masked) < min_rule_rows:
            continue
        wf = WalkForwardConfig(
            train_bars=max(100, len(masked) // 3),
            test_bars=max(50, len(masked) // 10),
            step_bars=max(50, len(masked) // 10),
            min_windows=2,
            embargo_bars=horizon,
        )
        signals = walk_forward_model_signals(
            masked.reset_index(drop=True),
            features,
            label_column,
            wf,
            tp,
            sl,
            fee_cost=fee_cost,
        )
        signals["rule_id"] = int(rule_id)
        signals["accept"] = signals["signal"].astype(bool)
        parts.append(signals)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = (
        pd.concat(parts, axis=0).reset_index(drop=True)
        if parts
        else pd.DataFrame(columns=["rule_id", "timestamp", "prob", "ev", "accept"])
    )
    out.to_parquet(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train meta-label filters for top strategy rules.")
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "day_trade_search" / "ranked_strategies.csv",
        help="Ranked rules CSV. Must come from the same base timeframe as --labels.",
    )
    parser.add_argument(
        "--labels", type=Path, default=PROCESSED_DATA_DIR / "train_5m_indicators_labels.parquet"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "meta_labeling" / "meta_signals.parquet",
    )
    parser.add_argument("--base-timeframe", default="5m")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Wrote {run(args.rules, args.labels, args.output, top_k=args.top_k, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps, base_timeframe=args.base_timeframe)}"
    )


if __name__ == "__main__":
    main()
