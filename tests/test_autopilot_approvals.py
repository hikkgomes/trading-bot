import json
import sys
import time
from dataclasses import fields, replace

import pytest

from research_exploration.dsr import DSR_METHOD
from src.autopilot.approvals import (
    ApprovalError,
    ApprovalLedger,
    artifact_digest,
    assert_artifact_live_approved,
    main,
    preflight_report_digest,
    strategy_fingerprint,
)
from src.autopilot.config import ProductConfig, canonical_product_config, load_config
from src.autopilot.execution_identity import execution_engine_digest
from src.execution.config import ExchangeConfig


def strategy(strategy_id="s1", take_profit=0.02):
    return {
        "id": strategy_id,
        "market": "futures",
        "symbol": "BTCUSDT",
        "base_timeframe": "5m",
        "direction": "long",
        "horizon_bars": 12,
        "take_profit": take_profit,
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
        "metrics": {
            "holdout_total_return": 0.03,
            "dsr_deflated": 0.72,
            "dsr_method": DSR_METHOD,
            "n_trials": 8,
            "sr_std_trials": 0.20,
            "trial_sharpe_count": 8,
            "trial_sharpe_observed_std": 0.15,
            "trial_sharpe_conservative_floor": 0.10,
        },
    }


def write_artifact(path, strategies):
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "symbol": "BTCUSDT",
                "pnl_unit": "usdt",
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": strategies,
            }
        ),
        encoding="utf-8",
    )


def write_config(path, artifact, *, execution_mode="paper"):
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
                        "execution_mode": execution_mode,
                        "symbol": "BTCUSDT",
                        "strategies_path": str(artifact),
                        "state_file": str(path.parent / "state.json"),
                        "trade_log": str(path.parent / "trades.csv"),
                        "preflight_report": str(path.parent / "preflight.json"),
                        "starting_equity": 1000.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def write_successful_production_preflight(path, configured_product, artifact):
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    engine_digest = execution_engine_digest()
    exchange = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="key",
        testnet=False,
    )
    check_names = [
        "product_config",
        "execution_engine_identity",
        "strategy_artifact_exists",
        "strategy_fingerprints",
        "strategy_policy",
        "exchange_environment",
        "broker_constructed",
        "exchange_read_connectivity",
        "broker_position_mode_one_way",
        "broker_native_protective_stops",
        "broker_open_orders_empty",
        "broker_position_flat",
    ]
    checks = [{"name": name, "ok": True} for name in check_names]
    next(item for item in checks if item["name"] == "exchange_environment")["detail"] = {
        "exchange": "binanceusdm",
        "market_type": "futures",
        "testnet": False,
        "require_testnet": False,
        "quote_asset": "USDT",
        "account_fingerprint": exchange.account_fingerprint,
        "max_notional_usd": 100.0,
        "max_fill_slippage_bps": 100.0,
        "max_futures_leverage": 1,
        "futures_margin_mode": "isolated",
    }
    generated_ts = time.time()
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": generated_ts,
                "ok": True,
                "products": [
                    {
                        "ok": True,
                        "product": canonical_product_config(configured_product),
                        "checks": checks,
                        "errors": [],
                        "execution_engine_digest": engine_digest,
                        "artifact_fingerprints": [
                            strategy_fingerprint(artifact_payload["strategies"][0])
                        ],
                        "artifact_digest": artifact_digest(artifact_payload),
                    }
                ],
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


def test_strategy_fingerprint_ignores_metrics_but_tracks_behavior():
    base = strategy()
    same_behavior = dict(base)
    same_behavior["metrics"] = {"holdout_total_return": 99}
    changed_behavior = dict(base)
    changed_behavior["take_profit"] = 0.03
    changed_symbol = dict(base)
    changed_symbol["symbol"] = "BTC/USDT:USDT"
    changed_baseline = dict(base)
    changed_baseline["baseline_win_rate"] = 0.61
    changed_leverage = dict(base)
    changed_leverage["leverage"] = 2
    changed_margin = dict(base)
    changed_margin["margin_mode"] = "cross"

    assert strategy_fingerprint(base) == strategy_fingerprint(same_behavior)
    assert strategy_fingerprint(base) != strategy_fingerprint(changed_behavior)
    assert strategy_fingerprint(base) != strategy_fingerprint(changed_symbol)
    assert strategy_fingerprint(base) != strategy_fingerprint(changed_baseline)
    assert strategy_fingerprint(base) != strategy_fingerprint(changed_leverage)
    assert strategy_fingerprint(base) != strategy_fingerprint(changed_margin)


def test_live_check_blocks_until_strategy_is_approved(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])

    with pytest.raises(ApprovalError, match="missing approval"):
        assert_artifact_live_approved(artifact, ledger)

    ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="henrique")
    assert_artifact_live_approved(artifact, ledger)


def test_approval_entry_records_fingerprint(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])

    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["approvals"][fingerprint]["fingerprint"] == fingerprint


def test_approval_actor_is_required_and_trimmed(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    approval_ledger = ApprovalLedger(ledger)

    with pytest.raises(ApprovalError, match="approved_by must be a non-empty"):
        approval_ledger.approve(strat, artifact_path=artifact, approved_by="   ")

    fingerprint = approval_ledger.approve(strat, artifact_path=artifact, approved_by="  henrique  ")
    approval_ledger.revoke(fingerprint, revoked_by="  henrique  ", reason="  manual risk review  ")

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    entry = payload["approvals"][fingerprint]
    assert entry["approved_by"] == "henrique"
    assert entry["revoked_by"] == "henrique"
    assert entry["revocation_reason"] == "manual risk review"


@pytest.mark.parametrize("actor", ["autopilot", "system", "github-actions[bot]", "trading_bot"])
def test_approval_actor_must_be_human_operator(tmp_path, actor):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])

    with pytest.raises(ApprovalError, match="must identify a human operator"):
        ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by=actor)


def test_revoke_actor_is_required(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )

    with pytest.raises(ApprovalError, match="revoked_by must be a non-empty"):
        ApprovalLedger(ledger).revoke(fingerprint, revoked_by="\t")


@pytest.mark.parametrize("actor", ["autopilot", "system", "github-actions[bot]", "trading_bot"])
def test_revoke_actor_must_be_human_operator(tmp_path, actor):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )

    with pytest.raises(ApprovalError, match="must identify a human operator"):
        ApprovalLedger(ledger).revoke(fingerprint, revoked_by=actor, reason="manual risk review")


def test_revoke_reason_is_required(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )

    with pytest.raises(ApprovalError, match="revocation reason must be non-empty"):
        ApprovalLedger(ledger).revoke(fingerprint, revoked_by="henrique", reason="  ")


def test_live_check_fails_closed_for_blank_approval_actor(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["approved_by"] = " "
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ApprovalError, match="invalid approval actor"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_automation_approval_actor(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["approved_by"] = "autopilot"
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ApprovalError, match="invalid approval actor"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_non_object_strategy_artifact(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    artifact.write_text("[]", encoding="utf-8")

    with pytest.raises(ApprovalError, match="must be a JSON object"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_invalid_json_strategy_artifact(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    artifact.write_text('{"version": 1,', encoding="utf-8")

    with pytest.raises(ApprovalError, match="must be valid JSON"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_symlink_strategy_artifact(tmp_path):
    target = tmp_path / "target.json"
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(target, [strat])
    ApprovalLedger(ledger).approve(strat, artifact_path=target, approved_by="henrique")
    artifact.symlink_to(target)

    with pytest.raises(ApprovalError, match="Strategy artifact must not be a symlink"):
        assert_artifact_live_approved(artifact, ledger)


def test_approval_write_refuses_symlink_strategy_artifact(tmp_path):
    target = tmp_path / "target.json"
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(target, [strat])
    artifact.symlink_to(target)

    with pytest.raises(ApprovalError, match="Strategy artifact must not be a symlink"):
        ApprovalLedger(ledger).approve(
            strat,
            artifact_path=artifact,
            artifact=json.loads(target.read_text(encoding="utf-8")),
            approved_by="henrique",
        )

    assert not ledger.exists()
    assert artifact.is_symlink()


def test_live_check_fails_closed_for_malformed_strategy_entries(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    artifact.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": ["bad"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ApprovalError, match="strategies must be JSON objects"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_non_object_approval_ledger(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    ledger.write_text("[]", encoding="utf-8")

    with pytest.raises(ApprovalError, match="Approval ledger must be a JSON object"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_symlink_approval_ledger(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    target = tmp_path / "external_approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    ApprovalLedger(target).approve(strat, artifact_path=artifact, approved_by="henrique")
    ledger.symlink_to(target)

    with pytest.raises(ApprovalError, match="Approval ledger must not be a symlink"):
        assert_artifact_live_approved(artifact, ledger)


def test_approval_write_refuses_symlink_ledger_without_touching_target(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    target = tmp_path / "external_approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    target.write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    ledger.symlink_to(target)

    with pytest.raises(ApprovalError, match="Approval ledger must not be a symlink"):
        ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="henrique")

    assert ledger.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"version": 1, "approvals": {}}\n'


def test_live_check_fails_closed_for_invalid_json_approval_ledger(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    ledger.write_text('{"version": 1,', encoding="utf-8")

    with pytest.raises(ApprovalError, match="Approval ledger must be valid JSON"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_non_object_approvals_map(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    ledger.write_text(json.dumps({"version": 1, "approvals": []}), encoding="utf-8")

    with pytest.raises(ApprovalError, match="Approval ledger approvals must be a JSON object"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_malformed_approval_entry(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    ledger.write_text(
        json.dumps({"version": 1, "approvals": {strategy_fingerprint(strat): "approved"}}),
        encoding="utf-8",
    )

    with pytest.raises(ApprovalError, match="malformed approval"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_entry_fingerprint_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["fingerprint"] = "sha256:other"
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ApprovalError, match="approval fingerprint mismatch"):
        assert_artifact_live_approved(artifact, ledger)


def test_live_check_fails_closed_for_missing_entry_fingerprint(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    del payload["approvals"][fingerprint]["fingerprint"]
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ApprovalError, match="approval fingerprint mismatch"):
        assert_artifact_live_approved(artifact, ledger)


def test_approval_is_bound_to_artifact_path(tmp_path):
    artifact = tmp_path / "active.json"
    copied_artifact = tmp_path / "copied.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_artifact(copied_artifact, [strat])
    ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="henrique")

    with pytest.raises(ApprovalError, match="approval artifact mismatch"):
        assert_artifact_live_approved(copied_artifact, ledger)


def test_approval_is_bound_to_artifact_content(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="henrique")
    approved_digest = artifact_digest(json.loads(artifact.read_text(encoding="utf-8")))

    changed_payload = json.loads(artifact.read_text(encoding="utf-8"))
    changed_payload["strategies"][0]["metrics"]["holdout_total_return"] = 0.99
    artifact.write_text(json.dumps(changed_payload), encoding="utf-8")
    current_digest = artifact_digest(changed_payload)

    with pytest.raises(ApprovalError) as excinfo:
        assert_artifact_live_approved(artifact, ledger)
    message = str(excinfo.value)
    assert "approval artifact content mismatch" in message
    assert f"approved={approved_digest}" in message
    assert f"current={current_digest}" in message


def test_product_bound_approval_cannot_be_reused_for_another_product(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    active_product = product(tmp_path, strategies_path=artifact)
    other_product = product(
        tmp_path,
        name="other_income",
        strategies_path=artifact,
    )
    ApprovalLedger(ledger).approve(
        strat,
        artifact_path=artifact,
        approved_by="henrique",
        product=active_product,
    )

    assert_artifact_live_approved(artifact, ledger, product=active_product)
    with pytest.raises(ApprovalError, match="approval product mismatch"):
        assert_artifact_live_approved(artifact, ledger, product=other_product)


def test_product_bound_approval_cannot_be_reused_for_another_symbol(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    active_product = product(tmp_path, strategies_path=artifact, symbol="BTCUSDT")
    other_symbol = product(tmp_path, strategies_path=artifact, symbol="ETHUSDT")
    ApprovalLedger(ledger).approve(
        strat,
        artifact_path=artifact,
        approved_by="henrique",
        product=active_product,
    )

    assert_artifact_live_approved(artifact, ledger, product=active_product)
    with pytest.raises(ApprovalError, match="approval product mismatch"):
        assert_artifact_live_approved(artifact, ledger, product=other_symbol)


def test_every_product_config_field_is_part_of_the_approval_identity(tmp_path):
    artifact = tmp_path / "active.json"
    ledger_path = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    baseline = product(tmp_path, strategies_path=artifact)
    ledger = ApprovalLedger(ledger_path)
    ledger.approve(
        strat,
        artifact_path=artifact,
        approved_by="henrique",
        product=baseline,
    )
    mutations = {
        "name": "other_income",
        "enabled": False,
        "objective": "btc_accumulation",
        "base_asset": "BTC",
        "market": "spot",
        "execution_mode": "live",
        "symbol": "ETHUSDT",
        "strategies_path": tmp_path / "other-artifact.json",
        "state_file": tmp_path / "other-state.json",
        "trade_log": tmp_path / "other-trades.csv",
        "starting_equity": 1001.0,
        "regime_guard": True,
        "regime_mayer_top": 2.5,
        "require_preflight": False,
        "preflight_report": tmp_path / "other-preflight.json",
        "preflight_max_age_seconds": 3601,
        "require_testnet_rehearsal": True,
        "testnet_rehearsal_report": tmp_path / "other-rehearsal.json",
        "testnet_rehearsal_max_age_seconds": 2592001,
    }
    assert set(mutations) == {descriptor.name for descriptor in fields(ProductConfig)}

    for field, value in mutations.items():
        with pytest.raises(ApprovalError):
            ledger.assert_approved(
                [strat],
                artifact_path=artifact,
                product=replace(baseline, **{field: value}),
            )


def test_live_approval_check_rejects_product_policy_failing_artifact(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    bad = strategy()
    bad["metrics"] = {"holdout_total_return": -0.01}
    write_artifact(artifact, [bad])
    active_product = product(tmp_path, strategies_path=artifact)
    ApprovalLedger(ledger).approve(
        bad,
        artifact_path=artifact,
        approved_by="henrique",
        product=active_product,
    )

    with pytest.raises(ApprovalError, match="strategy artifact violates policy"):
        assert_artifact_live_approved(artifact, ledger, product=active_product)


def test_changed_strategy_requires_new_approval(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    original = strategy(take_profit=0.02)
    changed = strategy(take_profit=0.03)

    ApprovalLedger(ledger).approve(original, artifact_path=artifact, approved_by="henrique")
    write_artifact(artifact, [changed])

    with pytest.raises(ApprovalError, match="missing approval"):
        assert_artifact_live_approved(artifact, ledger)


def test_approval_cli_rejects_policy_failing_product_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    config = tmp_path / "autopilot.json"
    bad = strategy()
    bad["metrics"] = {"holdout_total_return": -0.01}
    write_artifact(artifact, [bad])
    write_config(config, artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "approve",
            "--config",
            str(config),
            "--product",
            "active_income",
            "--artifact",
            str(artifact),
            "--expected-artifact-digest",
            artifact_digest(json.loads(artifact.read_text(encoding="utf-8"))),
            "--all",
            "--approved-by",
            "test",
            "--confirm-live",
        ],
    )

    with pytest.raises(SystemExit, match="violates policy"):
        main()

    assert not ledger.exists()


def test_approval_cli_rejects_duplicate_strategy_ids(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    config = tmp_path / "autopilot.json"
    write_artifact(artifact, [strategy(), strategy()])
    write_config(config, artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "approve",
            "--config",
            str(config),
            "--product",
            "active_income",
            "--artifact",
            str(artifact),
            "--expected-artifact-digest",
            artifact_digest(json.loads(artifact.read_text(encoding="utf-8"))),
            "--strategy-id",
            "s1",
            "--approved-by",
            "test",
            "--confirm-live",
        ],
    )

    with pytest.raises(SystemExit, match="duplicate strategy id 's1'"):
        main()

    assert not ledger.exists()


def test_approval_cli_rejects_ambiguous_all_and_strategy_id(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    config = tmp_path / "autopilot.json"
    write_artifact(artifact, [strategy("s1"), strategy("s2")])
    write_config(config, artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "approve",
            "--config",
            str(config),
            "--product",
            "active_income",
            "--artifact",
            str(artifact),
            "--expected-artifact-digest",
            artifact_digest(json.loads(artifact.read_text(encoding="utf-8"))),
            "--all",
            "--strategy-id",
            "s1",
            "--approved-by",
            "test",
            "--confirm-live",
        ],
    )

    with pytest.raises(SystemExit):
        main()

    assert not ledger.exists()


def test_approval_cli_requires_product_context_for_approve(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    write_artifact(artifact, [strategy()])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "approve",
            "--config",
            str(tmp_path / "missing_config.json"),
            "--artifact",
            str(artifact),
            "--expected-artifact-digest",
            artifact_digest(json.loads(artifact.read_text(encoding="utf-8"))),
            "--all",
            "--approved-by",
            "test",
            "--confirm-live",
        ],
    )

    with pytest.raises(ApprovalError, match="requires a product context"):
        main()

    assert not ledger.exists()


def test_approval_cli_requires_explicit_live_confirmation(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    config = tmp_path / "autopilot.json"
    write_artifact(artifact, [strategy()])
    write_config(config, artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "approve",
            "--config",
            str(config),
            "--product",
            "active_income",
            "--artifact",
            str(artifact),
            "--expected-artifact-digest",
            artifact_digest(json.loads(artifact.read_text(encoding="utf-8"))),
            "--all",
            "--approved-by",
            "test",
        ],
    )

    with pytest.raises(SystemExit, match="requires --confirm-live"):
        main()

    assert not ledger.exists()


def test_approval_cli_rejects_named_product_artifact_mismatch(monkeypatch, tmp_path):
    configured_artifact = tmp_path / "active.json"
    other_artifact = tmp_path / "other.json"
    ledger = tmp_path / "approvals.json"
    config = tmp_path / "autopilot.json"
    write_artifact(configured_artifact, [strategy()])
    write_artifact(other_artifact, [strategy("other")])
    write_config(config, configured_artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "approve",
            "--config",
            str(config),
            "--product",
            "active_income",
            "--artifact",
            str(other_artifact),
            "--expected-artifact-digest",
            artifact_digest(json.loads(other_artifact.read_text(encoding="utf-8"))),
            "--all",
            "--approved-by",
            "test",
            "--confirm-live",
        ],
    )

    with pytest.raises(ApprovalError, match="does not match product active_income strategies_path"):
        main()

    assert not ledger.exists()


def test_approval_cli_accepts_policy_passing_product_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    config = tmp_path / "autopilot.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_config(config, artifact, execution_mode="live")
    configured_product = load_config(config).products[0]
    write_successful_production_preflight(
        configured_product.preflight_report,
        configured_product,
        artifact,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "approve",
            "--config",
            str(config),
            "--product",
            "active_income",
            "--artifact",
            str(artifact),
            "--expected-artifact-digest",
            artifact_digest(json.loads(artifact.read_text(encoding="utf-8"))),
            "--expected-preflight-digest",
            preflight_report_digest(configured_product.preflight_report),
            "--all",
            "--approved-by",
            "test",
            "--confirm-live",
        ],
    )

    main()

    ApprovalLedger(ledger).assert_approved(
        [strat],
        artifact_path=artifact,
        artifact_digest_value=artifact_digest(json.loads(artifact.read_text(encoding="utf-8"))),
        product=configured_product,
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    fingerprint = strategy_fingerprint(strat)
    approved_artifact = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["approvals"][fingerprint]["artifact_digest"] == artifact_digest(
        approved_artifact
    )
    evidence = payload["approvals"][fingerprint]["production_preflight"]
    assert evidence["source_report_path"] == str(configured_product.preflight_report.resolve())
    assert evidence["source_report_digest"] == preflight_report_digest(
        configured_product.preflight_report
    )
    assert evidence["source_generated_at"] == "2026-01-01T00:00:00+00:00"
    assert evidence["manifest"] == {
        "exchange_environment": {
            "exchange": "binanceusdm",
            "market_type": "futures",
            "testnet": False,
            "require_testnet": False,
            "quote_asset": "USDT",
            "account_fingerprint": ExchangeConfig(
                exchange="binanceusdm",
                market_type="futures",
                api_key="key",
                testnet=False,
            ).account_fingerprint,
            "max_notional_usd": 100.0,
            "max_fill_slippage_bps": 100.0,
            "max_futures_leverage": 1,
            "futures_margin_mode": "isolated",
        },
    }
    assert evidence["manifest_digest"].startswith("sha256:")


def test_live_approval_is_invalidated_when_production_preflight_evidence_changes(tmp_path):
    artifact = tmp_path / "active.json"
    ledger_path = tmp_path / "approvals.json"
    config_path = tmp_path / "autopilot.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_config(config_path, artifact, execution_mode="live")
    configured_product = load_config(config_path).products[0]
    write_successful_production_preflight(
        configured_product.preflight_report,
        configured_product,
        artifact,
    )
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    digest = artifact_digest(artifact_payload)
    ledger = ApprovalLedger(ledger_path)
    ledger.approve(
        strat,
        artifact_path=artifact,
        approved_by="henrique",
        product=configured_product,
        artifact=artifact_payload,
    )

    report = json.loads(configured_product.preflight_report.read_text(encoding="utf-8"))
    exchange_check = next(
        item for item in report["products"][0]["checks"] if item["name"] == "exchange_environment"
    )
    exchange_check["detail"]["max_notional_usd"] = 101.0
    configured_product.preflight_report.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ApprovalError, match="approval production preflight mismatch"):
        ledger.assert_approved(
            [strat],
            artifact_path=artifact,
            artifact_digest_value=digest,
            product=configured_product,
        )


def test_equivalent_fresh_preflight_preserves_live_approval(tmp_path):
    artifact = tmp_path / "active.json"
    ledger_path = tmp_path / "approvals.json"
    config_path = tmp_path / "autopilot.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_config(config_path, artifact, execution_mode="live")
    configured_product = load_config(config_path).products[0]
    write_successful_production_preflight(
        configured_product.preflight_report,
        configured_product,
        artifact,
    )
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    digest = artifact_digest(artifact_payload)
    ledger = ApprovalLedger(ledger_path)
    ledger.approve(
        strat,
        artifact_path=artifact,
        approved_by="henrique",
        product=configured_product,
        artifact=artifact_payload,
    )
    approved_report_digest = preflight_report_digest(configured_product.preflight_report)

    refreshed = json.loads(configured_product.preflight_report.read_text(encoding="utf-8"))
    refreshed["generated_at"] = "2026-01-01T01:00:00+00:00"
    refreshed["generated_ts"] = time.time()
    configured_product.preflight_report.write_text(json.dumps(refreshed), encoding="utf-8")
    assert preflight_report_digest(configured_product.preflight_report) != approved_report_digest

    ledger.assert_approved(
        [strat],
        artifact_path=artifact,
        artifact_digest_value=digest,
        product=configured_product,
    )


def test_live_approval_rejects_boolean_futures_leverage(tmp_path):
    artifact = tmp_path / "active.json"
    ledger_path = tmp_path / "approvals.json"
    config_path = tmp_path / "autopilot.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    write_config(config_path, artifact, execution_mode="live")
    configured_product = load_config(config_path).products[0]
    write_successful_production_preflight(
        configured_product.preflight_report,
        configured_product,
        artifact,
    )
    report = json.loads(configured_product.preflight_report.read_text(encoding="utf-8"))
    exchange_check = next(
        item for item in report["products"][0]["checks"] if item["name"] == "exchange_environment"
    )
    exchange_check["detail"]["max_futures_leverage"] = True
    configured_product.preflight_report.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ApprovalError, match="incomplete or unsafe"):
        ApprovalLedger(ledger_path).approve(
            strat,
            artifact_path=artifact,
            approved_by="henrique",
            product=configured_product,
            artifact=json.loads(artifact.read_text(encoding="utf-8")),
        )


def test_approval_cli_requires_product_context_for_check(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    ApprovalLedger(ledger).approve(strat, artifact_path=artifact, approved_by="test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "check",
            "--config",
            str(tmp_path / "missing_config.json"),
            "--artifact",
            str(artifact),
        ],
    )

    with pytest.raises(ApprovalError, match="requires a product context"):
        main()


def test_approval_cli_revokes_fingerprint(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "revoke",
            "--fingerprint",
            fingerprint,
            "--revoked-by",
            "henrique",
            "--reason",
            "paper drawdown breached",
        ],
    )

    main()

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["approvals"][fingerprint]["status"] == "revoked"
    assert payload["approvals"][fingerprint]["revoked_by"] == "henrique"
    assert payload["approvals"][fingerprint]["revocation_reason"] == "paper drawdown breached"
    with pytest.raises(ApprovalError, match="revoked/not approved"):
        assert_artifact_live_approved(artifact, ledger)


def test_approval_cli_revoke_requires_reason(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="henrique"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "revoke",
            "--fingerprint",
            fingerprint,
            "--revoked-by",
            "henrique",
        ],
    )

    with pytest.raises(SystemExit):
        main()

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["approvals"][fingerprint]["status"] == "approved"


def test_approval_cli_list_surfaces_malformed_entry(monkeypatch, tmp_path, capsys):
    ledger = tmp_path / "approvals.json"
    ledger.write_text(
        json.dumps({"version": 1, "approvals": {"sha256:bad": "approved"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "list",
        ],
    )

    main()

    assert capsys.readouterr().out.strip() == "malformed <invalid> sha256:bad by=-"


def test_approval_cli_list_uses_revocation_actor_and_reason(monkeypatch, tmp_path, capsys):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="approver"
    )
    ApprovalLedger(ledger).revoke(
        fingerprint, revoked_by="reviewer", reason="paper drawdown breached"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "list",
        ],
    )

    main()

    line = capsys.readouterr().out.strip()
    assert line.startswith(f"revoked                  s1 {fingerprint} by=reviewer")
    assert "reason=paper drawdown breached" in line


def test_approval_cli_list_allows_revocation_reason_to_mention_automation(
    monkeypatch, tmp_path, capsys
):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="approver"
    )
    ApprovalLedger(ledger).revoke(fingerprint, revoked_by="reviewer", reason="system outage")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "list",
        ],
    )

    main()

    line = capsys.readouterr().out.strip()
    assert line.startswith(f"revoked                  s1 {fingerprint} by=reviewer")
    assert "reason=system outage" in line


def test_approval_cli_list_marks_automation_actor_invalid(monkeypatch, tmp_path, capsys):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="approver"
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["approved_by"] = "autopilot"
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "list",
        ],
    )

    main()

    line = capsys.readouterr().out.strip()
    assert line.startswith(f"invalid_actor            s1 {fingerprint} by=autopilot")


def test_approval_cli_list_marks_invalid_revocation_audit(monkeypatch, tmp_path, capsys):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    fingerprint = ApprovalLedger(ledger).approve(
        strat, artifact_path=artifact, approved_by="approver"
    )
    ApprovalLedger(ledger).revoke(
        fingerprint, revoked_by="reviewer", reason="paper drawdown breached"
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["revoked_by"] = " "
    payload["approvals"][fingerprint]["revocation_reason"] = ""
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "list",
        ],
    )

    main()

    line = capsys.readouterr().out.strip()
    assert line.startswith(f"invalid_revocation_audit s1 {fingerprint} by= ")
    assert "reason=-" in line
    assert "audit=invalid_revoked_by,missing_revocation_reason" in line


def test_approval_cli_revoke_fails_closed_for_malformed_entry(monkeypatch, tmp_path):
    ledger = tmp_path / "approvals.json"
    ledger.write_text(
        json.dumps({"version": 1, "approvals": {"sha256:bad": "approved"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "--ledger",
            str(ledger),
            "revoke",
            "--fingerprint",
            "sha256:bad",
            "--revoked-by",
            "henrique",
            "--reason",
            "malformed ledger repair",
        ],
    )

    with pytest.raises(ApprovalError, match="Malformed approval entry"):
        main()


def test_reapproval_preserves_prior_approval_and_revocation_history(tmp_path):
    artifact = tmp_path / "active.json"
    ledger_path = tmp_path / "approvals.json"
    strat = strategy()
    write_artifact(artifact, [strat])
    ledger = ApprovalLedger(ledger_path)
    fingerprint = ledger.approve(strat, artifact_path=artifact, approved_by="henrique")
    ledger.revoke(fingerprint, revoked_by="henrique", reason="paper drawdown breached")

    ledger.approve(strat, artifact_path=artifact, approved_by="henrique", notes="re-reviewed")

    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    entry = payload["approvals"][fingerprint]
    assert entry["status"] == "approved"
    assert entry["notes"] == "re-reviewed"
    assert [event["event"] for event in entry["history"]] == ["approved", "revoked"]
    assert entry["history"][1]["revocation_reason"] == "paper drawdown breached"
    assert_artifact_live_approved(artifact, ledger_path)
