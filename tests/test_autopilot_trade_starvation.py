import csv
import datetime as dt
from pathlib import Path

from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.trade_starvation import update_trade_starvation


def _product(tmp_path: Path) -> ProductConfig:
    return ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="paper",
        symbol="BTCUSDT",
        strategies_path=tmp_path / "strategies.json",
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
        starting_equity=1000.0,
    )


def _config(tmp_path: Path, product: ProductConfig) -> AutopilotConfig:
    return AutopilotConfig(
        products=[product],
        trade_starvation_enabled=True,
        trade_starvation_history_file=tmp_path / "starvation.jsonl",
        trade_starvation_report_file=tmp_path / "starvation.json",
        trade_starvation_window_days=30,
    )


def _supervisor_product(
    *, outcome: str, signals: int, entries: int, failed_stage=None, bar="2026-08-09T10:00:00Z"
):
    decision = {
        "strategy_id": "candidate",
        "outcome": outcome,
    }
    if failed_stage:
        decision.update(failed_stage=failed_stage, failed_predicate="volume confirmation")
    return {
        "product": {
            "name": "active_income",
            "objective": "active_income",
            "market": "futures",
            "symbol": "BTCUSDT",
        },
        "ok": True,
        "entries_allowed": True,
        "cycle_errors": [],
        "decision_trace": {
            "strategies": {"candidate": decision},
            "summary": {
                "strategies": 1,
                "data_ready": 1,
                "market_bars_processed": 1,
                "market_bars": [{"timeframe": "5m", "timestamp": bar}],
                "signals": signals,
                "entries_opened": entries,
                "positions_managed": 0,
                "outcomes": {outcome: 1},
            },
        },
    }


def test_rolling_diagnostic_finds_funnel_and_completed_trades(tmp_path):
    product = _product(tmp_path)
    config = _config(tmp_path, product)
    with product.trade_log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["exit_time"])
        writer.writeheader()
        writer.writerow({"exit_time": "2026-08-09T12:00:00+00:00"})
    first = {
        "products": [
            _supervisor_product(
                outcome="signal_not_triggered",
                signals=0,
                entries=0,
                failed_stage="trigger",
                bar="2026-08-09T10:00:00Z",
            )
        ]
    }
    second = {
        "products": [
            _supervisor_product(
                outcome="entry_opened",
                signals=1,
                entries=1,
                bar="2026-08-10T10:00:00Z",
            )
        ]
    }

    update_trade_starvation(
        config,
        first,
        now=dt.datetime(2026, 8, 9, 10, tzinfo=dt.UTC),
    )
    report = update_trade_starvation(
        config,
        second,
        now=dt.datetime(2026, 8, 10, 10, tzinfo=dt.UTC),
    )

    summary = report["products"][0]["summary"]
    assert summary["cycles"] == 2
    assert summary["data_ready"] == 2
    assert summary["market_bars_processed"] == 2
    assert summary["regime_eligible"] == 2
    assert summary["setup_matches"] == 2
    assert summary["trigger_matches"] == 1
    assert summary["signals"] == 1
    assert summary["entries_opened"] == 1
    assert summary["completed_trades"] == 1
    assert summary["killer_predicates"] == [{"predicate": "volume confirmation", "count": 1}]
    assert summary["starvation_point"] == "not_starved"


def test_diagnostic_identifies_management_only_entry_gate(tmp_path):
    product = _product(tmp_path)
    config = _config(tmp_path, product)
    raw = _supervisor_product(outcome="entry_disabled", signals=0, entries=0)
    raw["entries_allowed"] = False
    raw["entry_gate"] = {"reason": "unvalidated_bootstrap_artifact"}
    raw["decision_trace"]["summary"]["data_ready"] = 0

    report = update_trade_starvation(
        config,
        {"products": [raw]},
        now=dt.datetime(2026, 8, 10, 10, tzinfo=dt.UTC),
    )

    summary = report["products"][0]["summary"]
    assert summary["entry_enabled_cycles"] == 0
    assert summary["outcomes"] == {"entry_disabled": 1}
    assert summary["starvation_point"] == "entry_gate"
