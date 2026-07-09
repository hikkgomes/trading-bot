import json
import shlex

import pandas as pd

from src.autopilot.approvals import ApprovalLedger, strategy_fingerprint
from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.promotion import (
    PromotionThresholds,
    build_promotion_review,
    find_product_for_review,
    render_markdown,
)


def strategy(strategy_id="s1", holdout=0.05):
    return {
        "id": strategy_id,
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
                "threshold": 50,
                "description": "rsi >= 50",
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
        "fees": {"fee_bps": 5, "slippage_bps": 2},
        "metrics": {"holdout_total_return": holdout, "dsr_deflated": 0.72},
    }


def write_artifact(path, strategies):
    symbol = strategies[0].get("symbol", "BTCUSDT") if strategies and isinstance(strategies[0], dict) else "BTCUSDT"
    pnl_unit = strategies[0].get("pnl_unit", "usdt") if strategies and isinstance(strategies[0], dict) else "usdt"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "symbol": symbol,
                "pnl_unit": pnl_unit,
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": strategies,
            }
        ),
        encoding="utf-8",
    )


def write_trades(path, strategy_id="s1", n=3, sized_return=0.01, start="2026-01-01"):
    rows = [
        {
            "strategy_id": strategy_id,
            "exit_time": str(pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=i)),
            "net_return": 0.02 if i % 2 == 0 else -0.01,
            "sized_return": sized_return,
            "equity_after": 1000 + i,
        }
        for i in range(n)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_config(path, artifact):
    path.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "name": "active_income",
                        "enabled": True,
                        "objective": "active_income",
                        "base_asset": "USDT",
                        "market": "futures",
                        "execution_mode": "paper",
                        "symbol": "BTCUSDT",
                        "strategies_path": str(artifact),
                        "state_file": str(path.parent / "state.json"),
                        "trade_log": str(path.parent / "trades.csv"),
                        "starting_equity": 1000.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def product(tmp_path, **overrides):
    payload = {
        "name": "active_income",
        "enabled": True,
        "objective": "active_income",
        "base_asset": "USDT",
        "market": "futures",
        "execution_mode": "paper",
        "symbol": "BTCUSDT",
        "strategies_path": tmp_path / "active.json",
        "state_file": tmp_path / "state.json",
        "trade_log": tmp_path / "trades.csv",
        "starting_equity": 1000.0,
    }
    payload.update(overrides)
    return ProductConfig(**payload)


def test_promotion_review_not_ready_without_paper_trades(tmp_path):
    artifact = tmp_path / "active.json"
    write_artifact(artifact, [strategy()])

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=tmp_path / "missing.csv",
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=1),
    )

    item = review["strategies"][0]
    assert item["recommendation"] == "not_ready"
    assert item["paper"]["trades"] == 0
    assert "paper trades 0 < 1" in item["reasons"]
    assert item["approval_command"] is None


def test_promotion_review_waits_for_missing_strategy_artifact(tmp_path):
    artifact = tmp_path / "missing.json"
    active_product = product(tmp_path, strategies_path=artifact)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=tmp_path / "missing_trades.csv",
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=1),
        product=active_product,
    )
    markdown = render_markdown(review)

    assert review["status"] == "waiting_for_strategy_artifact"
    assert review["strategies"] == []
    assert "Strategy artifact not found" in review["reason"]
    assert "Status: `waiting_for_strategy_artifact`" in markdown
    assert "Strategy artifact not found" in markdown
    assert "python -m src.autopilot.approvals approve" not in markdown


def test_promotion_review_needs_approval_when_thresholds_pass(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy()])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
    )

    item = review["strategies"][0]
    assert item["recommendation"] == "not_ready"
    assert item["approval_status"] == "missing"
    assert item["paper"]["trades"] == 3
    assert item["paper"]["total_sized_return"] == 0.03
    assert item["paper"]["max_drawdown"] == 0.0
    assert item["paper"]["max_consecutive_losses"] == 1
    assert item["paper"]["paper_days"] == 2.0
    assert any("product context is required" in reason for reason in item["reasons"])
    assert item["approval_command"] is None


def test_promotion_review_blocks_invalid_paper_return_rows(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy()])
    pd.DataFrame(
        [
            {
                "strategy_id": "s1",
                "exit_time": "2026-01-01T00:00:00Z",
                "net_return": 0.02,
                "sized_return": 0.01,
                "equity_after": 1010.0,
            },
            {
                "strategy_id": "s1",
                "exit_time": "2026-01-02T00:00:00Z",
                "net_return": "bad",
                "sized_return": 0.01,
                "equity_after": 1020.0,
            },
            {
                "strategy_id": "s1",
                "exit_time": "2026-01-03T00:00:00Z",
                "net_return": 0.01,
                "equity_after": 1030.0,
            },
        ]
    ).to_csv(trade_log, index=False)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=product(tmp_path, strategies_path=artifact, trade_log=trade_log),
    )

    item = review["strategies"][0]
    assert item["recommendation"] == "not_ready"
    assert item["approval_command"] is None
    assert item["paper"]["invalid_return_rows"] == 2
    assert "paper trade log has 2 invalid return row(s)" in item["reasons"]


def test_promotion_review_blocks_non_numeric_holdout_metric(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy(holdout="bad")])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=product(tmp_path, strategies_path=artifact),
    )
    markdown = render_markdown(review)

    item = review["strategies"][0]
    assert item["recommendation"] == "not_ready"
    assert item["approval_command"] is None
    assert any("holdout_total_return metric must be numeric" in reason for reason in item["reasons"])
    assert "python -m src.autopilot.approvals approve" not in markdown


def test_promotion_review_blocks_non_finite_holdout_metric(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy(holdout=float("nan"))])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=product(tmp_path, strategies_path=artifact),
    )
    markdown = render_markdown(review)

    item = review["strategies"][0]
    assert item["recommendation"] == "not_ready"
    assert item["approval_command"] is None
    assert "holdout_total_return metric must be finite" in item["reasons"]
    assert "nan" not in markdown.lower()


def test_promotion_review_emits_approval_command_when_product_is_inferred_from_config(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    config = tmp_path / "autopilot.json"
    write_artifact(artifact, [strategy()])
    write_trades(trade_log, n=3, sized_return=0.01)
    write_config(config, artifact)
    active_product = find_product_for_review(config, None, artifact)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=active_product,
        config_path=config,
    )

    item = review["strategies"][0]
    assert item["recommendation"] == "needs_approval"
    assert "--strategy-id s1" in item["approval_command"]
    assert "--product active_income" in item["approval_command"]
    assert "--confirm-live" in item["approval_command"]


def test_promotion_review_suppresses_commands_for_invalid_thresholds(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy()])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(
            min_paper_trades=0,
            min_paper_sized_return=-0.10,
            min_holdout_return=-0.01,
            max_paper_drawdown=1.5,
            max_paper_consecutive_losses=-1,
            min_paper_days=-7.0,
        ),
        product=product(tmp_path, strategies_path=artifact),
    )
    markdown = render_markdown(review)

    item = review["strategies"][0]
    assert review["threshold_status"] == "fail"
    assert review["threshold_errors"] == [
        "min_paper_trades must be an integer >= 1",
        "min_paper_sized_return must be finite and non-negative",
        "min_holdout_return must be finite and non-negative",
        "max_paper_drawdown must be finite and between 0 and 1",
        "max_paper_consecutive_losses must be an integer >= 0",
        "min_paper_days must be finite and non-negative",
    ]
    assert item["recommendation"] == "not_ready"
    assert item["approval_command"] is None
    assert set(review["threshold_errors"]).issubset(set(item["reasons"]))
    assert "Thresholds: `fail`" in markdown
    assert "python -m src.autopilot.approvals approve" not in markdown


def test_promotion_review_suppresses_commands_when_approval_ledger_is_malformed(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    ledger = tmp_path / "approvals.json"
    write_artifact(artifact, [strategy()])
    write_trades(trade_log, n=3, sized_return=0.01)
    ledger.write_text("[]", encoding="utf-8")

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
    )
    markdown = render_markdown(review)

    item = review["strategies"][0]
    assert review["approval_ledger"]["ok"] is False
    assert "Approval ledger must be a JSON object" in review["approval_ledger"]["error"]
    assert item["approval_status"] == "ledger_error"
    assert item["recommendation"] == "not_ready"
    assert item["approval_command"] is None
    assert any("approval ledger could not be read" in reason for reason in item["reasons"])
    assert "Approval ledger: `error`" in markdown
    assert "python -m src.autopilot.approvals approve" not in markdown


def test_promotion_review_suppresses_commands_for_malformed_approval_entry(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_trades(trade_log, n=3, sized_return=0.01)
    ledger = tmp_path / "approvals.json"
    ledger.write_text(
        json.dumps({"version": 1, "approvals": {strategy_fingerprint(strat): "approved"}}),
        encoding="utf-8",
    )

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
    )

    item = review["strategies"][0]
    assert review["approval_ledger"]["ok"] is True
    assert item["approval_status"] == "malformed"
    assert item["recommendation"] == "not_ready"
    assert item["approval_command"] is None
    assert any("approval ledger entry is malformed" in reason for reason in item["reasons"])


def test_promotion_review_marks_blank_actor_approval_as_needing_approval(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_trades(trade_log, n=3, sized_return=0.01)
    fingerprint = ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="test")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["approved_by"] = " "
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
    )

    item = review["strategies"][0]
    assert item["approval_status"] == "invalid_actor"
    assert item["recommendation"] == "not_ready"
    assert any("product context is required" in reason for reason in item["reasons"])
    assert item["approval_command"] is None


def test_promotion_review_blocks_large_paper_drawdown(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy()])
    pd.DataFrame(
        [
            {
                "strategy_id": "s1",
                "exit_time": "2026-01-01T00:00:00Z",
                "net_return": 0.03,
                "sized_return": 0.03,
                "equity_after": 1030.0,
            },
            {
                "strategy_id": "s1",
                "exit_time": "2026-01-02T00:00:00Z",
                "net_return": -0.02,
                "sized_return": -0.02,
                "equity_after": 900.0,
            },
            {
                "strategy_id": "s1",
                "exit_time": "2026-01-03T00:00:00Z",
                "net_return": 0.01,
                "sized_return": 0.01,
                "equity_after": 910.0,
            },
        ]
    ).to_csv(trade_log, index=False)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, max_paper_drawdown=0.05),
    )

    item = review["strategies"][0]
    assert item["recommendation"] == "not_ready"
    assert item["approval_command"] is None
    assert item["paper"]["max_drawdown"] > 0.05
    assert any("paper max_drawdown" in reason for reason in item["reasons"])


def test_promotion_review_blocks_excessive_loss_streak(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy()])
    pd.DataFrame(
        [
            {
                "strategy_id": "s1",
                "exit_time": f"2026-01-0{i + 1}T00:00:00Z",
                "net_return": -0.001,
                "sized_return": 0.001,
                "equity_after": 1000.0 + i,
            }
            for i in range(5)
        ]
    ).to_csv(trade_log, index=False)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(
            min_paper_trades=5,
            min_paper_sized_return=0.0,
            max_paper_consecutive_losses=4,
        ),
    )

    item = review["strategies"][0]
    assert item["recommendation"] == "not_ready"
    assert item["paper"]["max_consecutive_losses"] == 5
    assert any("paper max_consecutive_losses 5 > 4" in reason for reason in item["reasons"])


def test_promotion_review_blocks_short_paper_duration_when_required(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy()])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_days=7.0),
    )

    item = review["strategies"][0]
    assert item["recommendation"] == "not_ready"
    assert any("paper days 2.00 < 7.00" in reason for reason in item["reasons"])


def test_promotion_review_emits_policy_aware_approval_command(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    config = tmp_path / "autopilot.json"
    write_artifact(artifact, [strategy()])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=product(tmp_path, strategies_path=artifact),
        config_path=config,
    )

    command = review["strategies"][0]["approval_command"]
    assert f"--config {config}" in command
    assert "--product active_income" in command


def test_promotion_review_shell_quotes_approval_command_paths_and_placeholder(tmp_path):
    artifact_dir = tmp_path / "artifact dir"
    config_dir = tmp_path / "config dir"
    artifact_dir.mkdir()
    config_dir.mkdir()
    artifact = artifact_dir / "active strategies.json"
    trade_log = tmp_path / "trades.csv"
    config = config_dir / "autopilot config.json"
    ledger = tmp_path / "approval ledger.json"
    write_artifact(artifact, [strategy()])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=product(tmp_path, strategies_path=artifact),
        config_path=config,
    )

    command = review["strategies"][0]["approval_command"]

    assert "'<your-name>'" in command
    assert shlex.split(command) == [
        "python",
        "-m",
        "src.autopilot.approvals",
        "--ledger",
        str(ledger),
        "approve",
        "--config",
        str(config),
        "--product",
        "active_income",
        "--artifact",
        str(artifact),
        "--strategy-id",
        "s1",
        "--approved-by",
        "<your-name>",
        "--confirm-live",
    ]


def test_promotion_review_marks_approval_product_mismatch_for_different_symbol(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    strat = strategy()
    strat["symbol"] = "ETHUSDT"
    write_artifact(artifact, [strat])
    write_trades(trade_log, n=3, sized_return=0.01)
    ledger = tmp_path / "approvals.json"
    approved_product = product(tmp_path, strategies_path=artifact, symbol="BTCUSDT")
    reviewed_product = product(tmp_path, strategies_path=artifact, symbol="ETHUSDT")
    ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="test", product=approved_product)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=reviewed_product,
    )

    assert review["strategies"][0]["approval_status"] == "product_mismatch"
    assert review["strategies"][0]["recommendation"] == "needs_approval"


def test_promotion_review_blocks_approval_command_when_policy_fails(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy(holdout=-0.01)])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=product(tmp_path, strategies_path=artifact),
    )

    item = review["strategies"][0]
    assert item["recommendation"] == "not_ready"
    assert item["policy_status"] == "fail"
    assert item["approval_command"] is None
    assert any("holdout_total_return -0.010000 must be positive" in reason for reason in item["reasons"])


def test_promotion_review_blocks_approval_command_when_artifact_policy_fails(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy(), strategy()])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=product(tmp_path, strategies_path=artifact),
    )

    assert review["artifact_policy_status"] == "fail"
    assert review["artifact_policy_errors"] == ["active_income: duplicate strategy id 's1'."]
    assert {item["recommendation"] for item in review["strategies"]} == {"not_ready"}
    assert all(item["policy_status"] == "fail" for item in review["strategies"])
    assert all(item["approval_command"] is None for item in review["strategies"])


def test_promotion_review_already_approved(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_trades(trade_log, n=3, sized_return=0.01)
    ledger = tmp_path / "approvals.json"
    ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="test")

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3),
    )

    assert review["strategies"][0]["recommendation"] == "already_approved"
    assert review["strategies"][0]["reasons"] == ["approved and passes configured review thresholds"]


def test_promotion_review_marks_content_mismatched_approval(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_trades(trade_log, n=3, sized_return=0.01)
    ledger = tmp_path / "approvals.json"
    ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="test")
    changed_payload = json.loads(artifact.read_text(encoding="utf-8"))
    changed_payload["strategies"][0]["metrics"]["holdout_total_return"] = 0.07
    artifact.write_text(json.dumps(changed_payload), encoding="utf-8")

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3),
    )

    item = review["strategies"][0]
    assert item["approval_status"] == "artifact_content_mismatch"
    assert item["recommendation"] == "not_ready"
    assert any("product context is required" in reason for reason in item["reasons"])
    assert item["approval_command"] is None


def test_promotion_review_marks_fingerprint_mismatched_approval(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_trades(trade_log, n=3, sized_return=0.01)
    ledger = tmp_path / "approvals.json"
    fingerprint = ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="test")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["fingerprint"] = "sha256:wrong"
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3),
    )

    item = review["strategies"][0]
    assert item["approval_status"] == "fingerprint_mismatch"
    assert item["recommendation"] == "not_ready"
    assert any("product context is required" in reason for reason in item["reasons"])
    assert item["approval_command"] is None


def test_promotion_review_marks_missing_entry_fingerprint_as_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_trades(trade_log, n=3, sized_return=0.01)
    ledger = tmp_path / "approvals.json"
    fingerprint = ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="test")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    del payload["approvals"][fingerprint]["fingerprint"]
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3),
    )

    item = review["strategies"][0]
    assert item["approval_status"] == "fingerprint_mismatch"
    assert item["recommendation"] == "not_ready"
    assert item["approval_command"] is None


def test_promotion_review_marks_approved_strategy_that_later_fails_paper_thresholds(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_trades(trade_log, n=3, sized_return=-0.01)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path, strategies_path=artifact)
    ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="test", product=active_product)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3, min_paper_sized_return=0.0),
        product=active_product,
    )

    item = review["strategies"][0]
    assert item["approval_status"] == "approved"
    assert item["recommendation"] == "approved_review_failed"
    assert item["approval_command"] is None
    assert any("paper total_sized_return" in reason for reason in item["reasons"])


def test_promotion_review_product_payload_includes_symbol(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    active_product = product(tmp_path, strategies_path=artifact, symbol="ETHUSDT")
    write_artifact(artifact, [strategy()])
    write_trades(trade_log, n=3, sized_return=0.01)

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3),
        product=active_product,
    )

    assert review["product"]["symbol"] == "ETHUSDT"


def test_promotion_review_distinguishes_product_mismatched_approval(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_trades(trade_log, n=3, sized_return=0.01)
    ledger = tmp_path / "approvals.json"
    other_product = product(tmp_path, name="other_income", strategies_path=artifact)
    active_product = product(tmp_path, strategies_path=artifact)
    ApprovalLedger(ledger).approve(
        strat,
        artifact_path=artifact,
        approved_by="test",
        product=other_product,
    )

    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=ledger,
        thresholds=PromotionThresholds(min_paper_trades=3),
        product=active_product,
    )

    item = review["strategies"][0]
    assert item["approval_status"] == "product_mismatch"
    assert item["recommendation"] == "needs_approval"
    assert item["approval_command"] is not None


def test_render_markdown_includes_fingerprint_and_command(tmp_path):
    artifact = tmp_path / "active.json"
    trade_log = tmp_path / "trades.csv"
    write_artifact(artifact, [strategy()])
    write_trades(trade_log, n=3)
    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=trade_log,
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=3),
    )

    markdown = render_markdown(review)

    assert "# Strategy Promotion Review" in markdown
    assert "Fingerprint:" in markdown
    assert "product context is required before generating an approval command" in markdown
    assert "python -m src.autopilot.approvals approve" not in markdown


def test_render_markdown_omits_command_when_not_ready(tmp_path):
    artifact = tmp_path / "active.json"
    write_artifact(artifact, [strategy(holdout=-0.01)])
    review = build_promotion_review(
        artifact_path=artifact,
        trade_log=tmp_path / "missing.csv",
        ledger_path=tmp_path / "approvals.json",
        thresholds=PromotionThresholds(min_paper_trades=1),
    )

    markdown = render_markdown(review)

    assert "python -m src.autopilot.approvals approve" not in markdown
    assert "Not emitted because this strategy is not in `needs_approval` state." in markdown


def test_find_product_for_review_infers_product_from_artifact_path(tmp_path):
    artifact = tmp_path / "active.json"
    config_path = tmp_path / "autopilot.json"
    cfg = AutopilotConfig(products=[product(tmp_path, strategies_path=artifact)])
    config_path.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "name": cfg.products[0].name,
                        "enabled": True,
                        "objective": cfg.products[0].objective,
                        "base_asset": cfg.products[0].base_asset,
                        "market": cfg.products[0].market,
                        "execution_mode": cfg.products[0].execution_mode,
                        "symbol": cfg.products[0].symbol,
                        "strategies_path": str(cfg.products[0].strategies_path),
                        "state_file": str(cfg.products[0].state_file),
                        "trade_log": str(cfg.products[0].trade_log),
                        "starting_equity": cfg.products[0].starting_equity,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    found = find_product_for_review(config_path, None, artifact)

    assert found is not None
    assert found.name == "active_income"
