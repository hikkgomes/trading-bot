import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.discover_patterns import Condition, split_train_test
from src.strategy_search import StrategyCandidate, TradeConfig, score_candidate_with_config


def _candidate_from_row(row: pd.Series, horizon_bars: int) -> StrategyCandidate:
    if "conditions_json" not in row or pd.isna(row["conditions_json"]):
        raise ValueError("rules file must include conditions_json")
    conditions = tuple(Condition(**payload) for payload in json.loads(row["conditions_json"]))
    return StrategyCandidate(str(row["direction"]), int(horizon_bars), conditions)


def run(
    rules_path: Path,
    output_path: Path,
    top_k: int = 25,
    trials: int = 50,
    input_path: Path = PROCESSED_DATA_DIR / "train_15m_indicators.parquet",
    train_fraction: float = 0.7,
    fee_bps: float = 5.0,
    slippage_bps: float = 1.0,
) -> Path:
    import optuna

    rules = pd.read_csv(rules_path).head(top_k).copy()
    data = pd.read_parquet(input_path).sort_values("timestamp").reset_index(drop=True)
    train, test = split_train_test(data, train_fraction)
    rows = []
    for _, row in rules.iterrows():
        def objective(trial: optuna.Trial) -> float:
            tp = trial.suggest_float("take_profit", 0.001, 0.03, log=True)
            sl = trial.suggest_float("stop_loss", 0.001, 0.03, log=True)
            horizon = trial.suggest_int("horizon_bars", 2, 64)
            candidate = _candidate_from_row(row, horizon)
            config = TradeConfig(
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                take_profit=tp,
                stop_loss=sl,
            )
            scored = score_candidate_with_config(train, test, candidate, config)
            return float(scored["test_total_return"])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        out = row.to_dict()
        out.update(study.best_params)
        out["optuna_score"] = float(study.best_value)
        rows.append(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune top strategy hyperparameters with Optuna.")
    parser.add_argument("--rules", type=Path, default=PROJECT_ROOT / "outputs" / "strategy_search" / "ranked_strategies.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "strategy_search" / "ranked_strategies_optuna.csv")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--input-path", type=Path, default=PROCESSED_DATA_DIR / "train_15m_indicators.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Wrote {run(args.rules, args.output, args.top_k, args.trials, input_path=args.input_path)}")


if __name__ == "__main__":
    main()
