"""Offline end-to-end rehearsal of the promotion and preflight workflow.

No exchange calls are made and no real approval ledger is touched by default.
The rehearsal creates synthetic strategy artifacts and paper-trade logs for both
products, promotion reviews, a temporary approval ledger, and an offline
preflight report using a fake read-only broker.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.autopilot.approvals import ApprovalLedger, strategy_fingerprint
from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.experiment_memory import ExperimentMemory
from src.autopilot.io import write_json_atomic, write_text_atomic
from src.autopilot.preflight import run_preflight
from src.autopilot.promotion import (
    PromotionThresholds,
    build_promotion_review,
    write_review,
)
from src.autopilot.research_factory import build_generation, load_factory_config
from src.autopilot.research_smoke import run_research_smoke
from src.config import PROJECT_ROOT
from src.execution.broker import Position

DEFAULT_REHEARSAL_DIR = PROJECT_ROOT / "runtime" / "rehearsal"


def synthetic_active_income_strategy() -> dict[str, Any]:
    return {
        "id": "rehearsal_active_income",
        "rank": 1,
        "market": "futures",
        "symbol": "BTCUSDT",
        "base_timeframe": "5m",
        "direction": "long",
        "horizon_bars": 12,
        "take_profit": 0.02,
        "stop_loss": 0.01,
        "use_atr_tp_sl": False,
        "pnl_unit": "usdt",
        "conditions": [
            {
                "feature": "tf_5m_rsi_14",
                "kind": "value_ge",
                "threshold": 50.0,
                "description": "tf_5m_rsi_14 >= 50.0",
            }
        ],
        "risk": {
            "risk_per_trade": 0.003,
            "max_position_fraction": 0.25,
            "daily_stop_loss": -0.02,
            "max_consecutive_losses": 3,
            "cooldown_bars": 24,
            "max_trades_per_day": 4,
        },
        "fees": {"fee_bps": 5.0, "slippage_bps": 2.0},
        "metrics": {
            "holdout_total_return": 0.04,
            "holdout_win_rate": 0.58,
            "dsr": 0.95,
        },
    }


def synthetic_btc_accumulation_strategy() -> dict[str, Any]:
    return {
        "id": "rehearsal_btc_accumulation",
        "rank": 1,
        "market": "spot",
        "symbol": "BTCUSDT",
        "base_timeframe": "4h",
        "direction": "short",
        "horizon_bars": 18,
        "take_profit": 0.03,
        "stop_loss": 0.02,
        "use_atr_tp_sl": False,
        "pnl_unit": "btc",
        "conditions": [
            {
                "feature": "tf_4h_rsi_14",
                "kind": "value_le",
                "threshold": 45.0,
                "description": "tf_4h_rsi_14 <= 45.0",
            }
        ],
        "risk": {
            "risk_per_trade": 0.002,
            "max_position_fraction": 0.35,
            "daily_stop_loss": -0.008,
            "max_consecutive_losses": 3,
            "cooldown_bars": 24,
            "max_trades_per_day": 1,
        },
        "fees": {"fee_bps": 5.0, "slippage_bps": 2.0},
        "metrics": {
            "holdout_total_return": 0.025,
            "holdout_excess_return_vs_buy_hold": 0.025,
            "holdout_buy_hold_return": 0.0,
            "holdout_win_rate": 0.55,
            "dsr": 0.92,
        },
    }


def _product_config(
    work_dir: Path,
    *,
    name: str,
    strategy: dict[str, Any],
    artifact_path: Path,
    trade_log: Path,
) -> ProductConfig:
    if name == "btc_accumulation":
        return ProductConfig(
            name="btc_accumulation",
            enabled=True,
            objective="btc_accumulation",
            base_asset="BTC",
            market="spot",
            execution_mode="paper",
            symbol="BTCUSDT",
            strategies_path=artifact_path,
            state_file=work_dir / "btc_accumulation_state.json",
            trade_log=trade_log,
            preflight_report=work_dir / "btc_accumulation_preflight_report.json",
            starting_equity=1.0,
            regime_guard=True,
        )
    return ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="paper",
        symbol="BTCUSDT",
        strategies_path=artifact_path,
        state_file=work_dir / "active_income_state.json",
        trade_log=trade_log,
        preflight_report=work_dir / "active_income_preflight_report.json",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=work_dir / "testnet_rehearsal_report.json",
        starting_equity=1000.0,
    )


def write_synthetic_inputs(
    work_dir: Path,
    *,
    name: str = "active_income",
) -> tuple[Path, Path, dict[str, Any], ProductConfig]:
    work_dir.mkdir(parents=True, exist_ok=True)
    strategy = (
        synthetic_btc_accumulation_strategy()
        if name == "btc_accumulation"
        else synthetic_active_income_strategy()
    )
    artifact_path = work_dir / f"{name}_active_strategies_rehearsal.json"
    write_json_atomic(
        artifact_path,
        {
            "version": 1,
            "source": "offline_rehearsal",
            "market": strategy["market"],
            "symbol": strategy["symbol"],
            "pnl_unit": strategy["pnl_unit"],
            "paper_trade_allowed": True,
            "live_allowed": True,
            "promotion_eligible": True,
            "strategies": [strategy],
        },
    )
    trade_log = work_dir / f"{name}_paper_trades_rehearsal.csv"
    trades = pd.DataFrame(
        [
            {
                "strategy_id": strategy["id"],
                "strategy_fingerprint": strategy_fingerprint(strategy),
                "exit_time": str(pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i)),
                "net_return": 0.012 if i % 3 else -0.004,
                "sized_return": 0.001,
                "equity_after": (1.0 + (i * 0.0005)) if name == "btc_accumulation" else 1000.0 + i,
            }
            for i in range(25)
        ]
    )
    write_text_atomic(trade_log, trades.to_csv(index=False))
    product = _product_config(
        work_dir,
        name=name,
        strategy=strategy,
        artifact_path=artifact_path,
        trade_log=trade_log,
    )
    return artifact_path, trade_log, strategy, product


class FakeReadOnlyBroker:
    name = "fake-read-only"

    def verify_one_way_position_mode(self, symbol: str) -> bool:
        return True

    def supports_native_protective_stops(self) -> bool:
        return True

    def list_open_orders(self, symbol: str, *, conditional: bool) -> tuple:
        return ()

    def get_price(self, symbol: str) -> float:
        return 100.0

    def get_balance(self) -> float:
        return 1000.0

    def get_position(self, symbol: str) -> Position:
        return Position(symbol=symbol, qty=0.0, avg_price=0.0)


def run_rehearsal(work_dir: Path = DEFAULT_REHEARSAL_DIR) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    research_memory_path = work_dir / "experiment_memory.sqlite3"
    research_batch_path = work_dir / "generated_hypotheses.json"
    research_proposal_state_path = work_dir / "openclaw_proposal_state.json"
    for path in (research_memory_path, research_batch_path, research_proposal_state_path):
        if path.is_symlink():
            raise ValueError(f"rehearsal artifact must not be a symlink: {path}")
        if path.exists():
            path.unlink()

    factory_template = load_factory_config()
    rehearsal_factory = dataclasses.replace(
        factory_template,
        memory_path=research_memory_path,
        generated_batch_path=research_batch_path,
        openclaw_accepted_dir=work_dir / "openclaw" / "accepted",
        proposal_state_path=research_proposal_state_path,
    )
    first_generation = build_generation(
        rehearsal_factory,
        seed=101,
        now="2026-01-01T00:00:00+00:00",
    )
    first_hashes = {
        str(item["strategy_hash"]) for item in first_generation["generation_metadata"]
    }
    with ExperimentMemory(research_memory_path) as memory:
        for item in first_generation["generation_metadata"]:
            memory.record_outcome(
                str(item["strategy_hash"]),
                dataset={"snapshot_id": "offline-rehearsal-development-v1"},
                window={"start": "2024-01-01", "end": "2025-01-01"},
                protocol={
                    "name": "offline_rehearsal",
                    "version": 1,
                    "research_engine_digest": first_generation["source"][
                        "research_engine_digest"
                    ],
                },
                phase="development",
                outcome="reject",
                rejection_reasons=("synthetic_rehearsal_rejection",),
                details={"synthetic_only": True, "holdout_feedback_allowed": False},
            )
    second_generation = build_generation(
        rehearsal_factory,
        seed=202,
        now="2026-01-02T00:00:00+00:00",
    )
    write_json_atomic(research_batch_path, second_generation)
    second_hashes = {
        str(item["strategy_hash"]) for item in second_generation["generation_metadata"]
    }
    research_smoke = run_research_smoke(synthetic_rows=700)
    research_rehearsal = {
        "ok": bool(first_generation.get("ok"))
        and bool(second_generation.get("ok"))
        and bool(research_smoke.get("ok"))
        and bool(first_hashes)
        and first_hashes.isdisjoint(second_hashes),
        "synthetic_only": True,
        "first_generation": len(first_hashes),
        "second_generation": len(second_hashes),
        "new_behavioral_specs_after_feedback": len(second_hashes - first_hashes),
        "products": sorted(
            {
                str(item.get("product"))
                for item in second_generation.get("generation_metadata", [])
            }
        ),
        "opportunity_types": sorted(
            {
                str(item.get("opportunity_type"))
                for item in second_generation.get("generation_metadata", [])
            }
        ),
        "experiment_memory": str(research_memory_path),
        "generated_batch": str(research_batch_path),
        "smoke": research_smoke,
    }

    product_names = ("active_income", "btc_accumulation")
    ledger_path = work_dir / "approvals_rehearsal.json"
    if ledger_path.exists():
        ledger_path.unlink()

    thresholds = PromotionThresholds(
        min_paper_trades=20, min_paper_sized_return=0.0, min_holdout_return=0.0
    )
    products: list[ProductConfig] = []
    product_reports: dict[str, Any] = {}
    ledger = ApprovalLedger(ledger_path)
    for name in product_names:
        artifact_path, trade_log, strategy, product = write_synthetic_inputs(work_dir, name=name)
        products.append(product)
        review_json = work_dir / f"{name}_promotion_review.json"
        review_md = work_dir / f"{name}_promotion_review.md"
        review_before = build_promotion_review(
            artifact_path=artifact_path,
            trade_log=trade_log,
            ledger_path=ledger_path,
            thresholds=thresholds,
            product=product,
        )
        write_review(review_before, review_json, review_md)
        before_recommendation = review_before["strategies"][0]["recommendation"]
        ledger.approve(
            strategy,
            artifact_path=artifact_path,
            approved_by="offline-rehearsal",
            product=product,
        )
        review_after = build_promotion_review(
            artifact_path=artifact_path,
            trade_log=trade_log,
            ledger_path=ledger_path,
            thresholds=thresholds,
            product=product,
        )
        product_reports[name] = {
            "artifact": str(artifact_path),
            "trade_log": str(trade_log),
            "promotion_review_json": str(review_json),
            "promotion_review_md": str(review_md),
            "before_recommendation": before_recommendation,
            "after_recommendation": review_after["strategies"][0]["recommendation"],
        }

    config = AutopilotConfig(
        approval_ledger=ledger_path,
        products=products,
    )
    preflight = run_preflight(
        config,
        assume_live=True,
        connect=True,
        exchange_env_checker=lambda product: [],
        broker_builder=lambda product: FakeReadOnlyBroker(),
    )
    preflight_path = work_dir / "preflight_report.json"
    write_json_atomic(preflight_path, preflight)

    active_report = product_reports["active_income"]
    report = {
        "ok": all(
            item["before_recommendation"] == "needs_approval"
            and item["after_recommendation"] == "already_approved"
            for item in product_reports.values()
        )
        and preflight["ok"]
        and research_rehearsal["ok"],
        "work_dir": str(work_dir),
        "products": product_reports,
        "preflight_report": str(preflight_path),
        "preflight_products": [
            item.get("product", {}).get("name") for item in preflight.get("products", [])
        ],
        "preflight_ok": preflight["ok"],
        "research_rehearsal": research_rehearsal,
        # Backwards-compatible summary fields for older smoke checks.
        "artifact": active_report["artifact"],
        "trade_log": active_report["trade_log"],
        "promotion_review_json": active_report["promotion_review_json"],
        "promotion_review_md": active_report["promotion_review_md"],
        "before_recommendation": active_report["before_recommendation"],
        "after_recommendation": active_report["after_recommendation"],
    }
    summary_path = work_dir / "rehearsal_summary.json"
    write_json_atomic(summary_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an offline end-to-end autopilot workflow rehearsal."
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_REHEARSAL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_rehearsal(args.work_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
