import json

import pytest

from src.autopilot.approvals import assert_artifact_policy_allowed_for_product
from src.autopilot.bootstrap_strategies import build_bootstrap_artifact, write_bootstrap_artifacts
from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.strategy_policy import StrategyPolicyError, assert_strategy_artifact_allowed


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


def test_build_bootstrap_artifact_is_paper_only_and_policy_valid_for_paper(tmp_path):
    active_product = product(tmp_path)
    artifact = build_bootstrap_artifact(active_product)
    active_product.strategies_path.write_text(json.dumps(artifact), encoding="utf-8")

    assert artifact["paper_trade_allowed"] is True
    assert artifact["live_allowed"] is False
    assert artifact["promotion_eligible"] is False
    assert len(artifact["strategies"]) == 2
    assert all(strategy["metrics"] == {} for strategy in artifact["strategies"])
    assert_strategy_artifact_allowed(active_product, require_live_eligible=False)
    with pytest.raises(StrategyPolicyError, match="not allowed for live trading"):
        assert_strategy_artifact_allowed(active_product, require_live_eligible=True)


def test_build_bootstrap_artifact_handles_btc_accumulation(tmp_path):
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        strategies_path=tmp_path / "btc.json",
        starting_equity=1.0,
    )

    artifact = build_bootstrap_artifact(btc_product)

    assert artifact["market"] == "spot"
    assert artifact["pnl_unit"] == "btc"
    assert artifact["strategies"][0]["direction"] == "short"
    assert artifact["strategies"][0]["risk"]["max_trades_per_day"] == 1


def test_write_bootstrap_artifacts_skips_existing_and_overwrites_when_requested(tmp_path):
    active_product = product(tmp_path)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[active_product],
    )
    active_product.strategies_path.write_text('{"existing": true}', encoding="utf-8")

    skipped = write_bootstrap_artifacts(cfg)
    assert skipped["artifacts"][0]["action"] == "skipped_existing"
    assert json.loads(active_product.strategies_path.read_text(encoding="utf-8")) == {"existing": True}

    written = write_bootstrap_artifacts(cfg, overwrite=True)
    payload = json.loads(active_product.strategies_path.read_text(encoding="utf-8"))
    assert written["artifacts"][0]["action"] == "written"
    assert payload["schema"] == "autopilot.paper_bootstrap/v1"


def test_paper_only_bootstrap_cannot_be_approved_for_live(tmp_path):
    active_product = product(tmp_path)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[active_product],
    )
    write_bootstrap_artifacts(cfg)
    artifact = json.loads(active_product.strategies_path.read_text(encoding="utf-8"))

    with pytest.raises(StrategyPolicyError, match="not allowed for live trading"):
        assert_artifact_policy_allowed_for_product(artifact, active_product)
    assert not cfg.approval_ledger.exists()
