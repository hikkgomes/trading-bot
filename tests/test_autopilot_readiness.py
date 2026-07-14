import json
import time
from collections import namedtuple
from pathlib import Path

import pytest

import src.autopilot.readiness as readiness
from research_exploration.dsr import DSR_METHOD
from src.autopilot.approvals import ApprovalLedger, artifact_digest, strategy_fingerprint
from src.autopilot.config import (
    AutopilotConfig,
    JobConfig,
    ProductConfig,
    canonical_product_config,
)
from src.autopilot.execution_identity import execution_engine_digest
from src.autopilot.experiment_memory import ExperimentMemory
from src.autopilot.readiness import build_readiness_report, main, render_readiness_markdown
from src.autopilot.research_factory import build_generation, load_factory_config
from src.execution.config import ExchangeConfig

VALID_SERVICE_INSTALLER = """#!/bin/bash
set -euo pipefail
validate_unit_name() { :; }
validate_unit_value() { :; }
validate_positive_integer() { :; }
validate_zero_or_one() { :; }
"$PYTHON" -m src.autopilot.runtime --config "$CONFIG" --validate
"$PYTHON" -m src.autopilot.readiness --config "$CONFIG"
ExecStart=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT --skip-jobs
HEALTHCHECK_TIMER_NAME="${HEALTHCHECK_TIMER_NAME:-trading-bot-autopilot-healthcheck.timer}"
systemctl --user enable --now "$HEALTHCHECK_TIMER_NAME"
"""


def product(tmp_path, **overrides):
    payload = {
        "name": "active_income",
        "enabled": True,
        "objective": "active_income",
        "base_asset": "USDT",
        "market": "futures",
        "execution_mode": "paper",
        "symbol": "BTCUSDT",
        "strategies_path": tmp_path / "missing.json",
        "state_file": tmp_path / "state.json",
        "trade_log": tmp_path / "trades.csv",
        "preflight_report": tmp_path / "preflight.json",
        "testnet_rehearsal_report": tmp_path / "testnet.json",
        "starting_equity": 1000.0,
    }
    payload.update(overrides)
    return ProductConfig(**payload)


def write_research_factory_config(tmp_path, monkeypatch):
    import src.autopilot.research_factory as research_factory

    monkeypatch.setattr(research_factory, "PROJECT_ROOT", tmp_path)
    payload = json.loads(Path("config/research_factory.json").read_text(encoding="utf-8"))
    payload.update(
        {
            "memory_path": "runtime/research/memory.sqlite3",
            "generated_batch_path": "runtime/research/batch.json",
            "openclaw_accepted_dir": "runtime/openclaw/accepted",
            "openclaw_proposal_state_path": "runtime/research/proposals.json",
        }
    )
    payload["budgets"].update(
        {
            "max_candidates_per_cycle": 5,
            "max_candidates_per_space": 1,
            "max_generation_attempts": 300,
            "max_generation_seconds": 5,
        }
    )
    path = tmp_path / "config" / "research_factory.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def strategy_artifact(path):
    strategy = {
        "id": "ready_r1",
        "market": "futures",
        "symbol": "BTCUSDT",
        "base_timeframe": "5m",
        "direction": "long",
        "horizon_bars": 4,
        "take_profit": 0.02,
        "stop_loss": 0.01,
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
                "strategies": [strategy],
            }
        ),
        encoding="utf-8",
    )
    return strategy


def test_research_factory_readiness_validates_goals_horizons_and_path_wiring(tmp_path, monkeypatch):
    factory_path = write_research_factory_config(tmp_path, monkeypatch)
    cfg = AutopilotConfig(
        research_factory_config_file=factory_path,
        generated_batch_file=tmp_path / "runtime" / "research" / "batch.json",
        experiment_memory_file=tmp_path / "runtime" / "research" / "memory.sqlite3",
    )

    status, factory = readiness._research_factory_status(cfg)

    assert factory is not None
    assert status["ok"] is True
    assert status["active_income_opportunities"] == [
        "day_trading",
        "scalping",
        "swing_trading",
    ]
    assert status["btc_accumulation_timeframes"] == ["1h", "4h"]
    assert all(status["goal_contract"].values())
    assert all(status["path_contract"].values())


def test_research_factory_readiness_rejects_autopilot_memory_path_mismatch(tmp_path, monkeypatch):
    factory_path = write_research_factory_config(tmp_path, monkeypatch)
    cfg = AutopilotConfig(
        research_factory_config_file=factory_path,
        generated_batch_file=tmp_path / "runtime" / "research" / "batch.json",
        experiment_memory_file=tmp_path / "runtime" / "other-memory.sqlite3",
    )

    status, factory = readiness._research_factory_status(cfg)

    assert factory is not None
    assert status["ok"] is False
    assert status["reason"] == "unsafe_search_contract"
    assert status["problems"] == ["path_mismatch:memory"]


def test_experiment_memory_readiness_handles_clean_first_boot_and_detects_corruption(tmp_path):
    memory_path = tmp_path / "research" / "memory.sqlite3"
    backup_path = tmp_path / "research" / "memory.backup.sqlite3"

    missing = readiness._experiment_memory_status(memory_path, backup_path=backup_path)
    assert missing["ok"] is True
    assert missing["reason"] == "initializes_on_first_factory_run"
    assert memory_path.exists() is False

    with ExperimentMemory(memory_path):
        pass
    ready = readiness._experiment_memory_status(memory_path, backup_path=backup_path)
    assert ready["ok"] is True
    assert ready["integrity"]["deep"] is True

    memory_path.write_bytes(b"not a sqlite database")
    corrupt = readiness._experiment_memory_status(memory_path, backup_path=backup_path)
    assert corrupt["ok"] is False
    assert corrupt["reason"] == "integrity_failed"


def test_generated_batch_readiness_accepts_only_canonical_safe_factory_output(
    tmp_path, monkeypatch
):
    factory_path = write_research_factory_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    factory = load_factory_config(factory_path)
    payload = build_generation(factory, seed=42, now="2026-07-10T00:00:00+00:00")
    factory.generated_batch_path.parent.mkdir(parents=True, exist_ok=True)
    factory.generated_batch_path.write_text(json.dumps(payload), encoding="utf-8")

    ready = readiness._generated_batch_status(factory.generated_batch_path, factory=factory)
    assert ready["ok"] is True
    assert ready["hypotheses"] == 5
    assert ready["unique_behavioral_specs"] == 5
    assert ready["products"] == ["active_income", "btc_accumulation"]

    payload["live_allowed"] = True
    factory.generated_batch_path.write_text(json.dumps(payload), encoding="utf-8")
    unsafe = readiness._generated_batch_status(factory.generated_batch_path, factory=factory)
    assert unsafe["ok"] is False
    assert unsafe["level"] == "error"
    assert unsafe["reason"] == "failed_safety_contract"


def test_generated_batch_readiness_warns_without_creating_a_first_boot_batch(tmp_path, monkeypatch):
    factory_path = write_research_factory_config(tmp_path, monkeypatch)
    factory = load_factory_config(factory_path)

    status = readiness._generated_batch_status(factory.generated_batch_path, factory=factory)

    assert status["ok"] is False
    assert status["level"] == "warning"
    assert status["next_action"] == "make research-generate"
    assert factory.generated_batch_path.exists() is False


def test_service_installer_status_accepts_readable_valid_bash_script_without_executable_bit(
    tmp_path,
):
    installer = tmp_path / "install_autopilot_service.sh"
    installer.write_text(VALID_SERVICE_INSTALLER, encoding="utf-8")
    installer.chmod(0o644)

    status = readiness._service_installer_status(installer)

    assert status["ok"] is True
    assert status["readable"] is True
    assert status["non_empty"] is True
    assert status["has_shell_shebang"] is True
    assert all(status["required_markers"].values())


def test_service_installer_status_rejects_malformed_script(tmp_path):
    installer = tmp_path / "install_autopilot_service.sh"
    installer.write_text("echo missing shell header\n", encoding="utf-8")

    status = readiness._service_installer_status(installer)

    assert status["ok"] is False
    assert status["exists"] is True
    assert status["has_shell_shebang"] is False


def test_service_installer_status_rejects_script_missing_startup_contract(tmp_path):
    installer = tmp_path / "install_autopilot_service.sh"
    installer.write_text("#!/bin/bash\nset -euo pipefail\n", encoding="utf-8")

    status = readiness._service_installer_status(installer)

    assert status["ok"] is False
    assert status["has_shell_shebang"] is True
    assert status["required_markers"]["strict_shell"] is True
    assert "config_validation" in status["missing_markers"]
    assert "readiness_install_gate" in status["missing_markers"]
    assert "healthcheck_timer" in status["missing_markers"]
    assert "unit_name_validation" in status["missing_markers"]
    assert "raw_unit_value_validation" in status["missing_markers"]


def test_offline_rehearsal_status_reports_missing_summary(tmp_path):
    status = readiness._offline_rehearsal_status(tmp_path / "missing.json")

    assert status["ok"] is False
    assert status["reason"] == "missing_report"
    assert status["next_action"] == "make rehearse"


def test_offline_rehearsal_status_rejects_non_object_summary(tmp_path):
    summary = tmp_path / "rehearsal_summary.json"
    summary.write_text("[]", encoding="utf-8")

    status = readiness._offline_rehearsal_status(summary)

    assert status["ok"] is False
    assert status["reason"] == "invalid_report"
    assert status["error"] == "expected JSON object, got list"


def test_offline_rehearsal_status_rejects_incomplete_workflow(tmp_path):
    summary = tmp_path / "rehearsal_summary.json"
    summary.write_text(
        json.dumps(
            {
                "ok": True,
                "work_dir": str(tmp_path),
                "products": {
                    "active_income": {
                        "artifact": str(tmp_path / "active.json"),
                        "trade_log": str(tmp_path / "active.csv"),
                        "promotion_review_json": str(tmp_path / "active_review.json"),
                        "before_recommendation": "needs_approval",
                        "after_recommendation": "already_approved",
                    },
                    "btc_accumulation": {
                        "artifact": str(tmp_path / "btc.json"),
                        "trade_log": str(tmp_path / "btc.csv"),
                        "promotion_review_json": str(tmp_path / "btc_review.json"),
                        "before_recommendation": "already_approved",
                        "after_recommendation": "already_approved",
                    },
                },
                "preflight_ok": True,
                "preflight_products": ["active_income"],
            }
        ),
        encoding="utf-8",
    )

    status = readiness._offline_rehearsal_status(summary)

    assert status["ok"] is False
    assert status["reason"] == "failed"
    assert status["invalid_products"] == ["btc_accumulation"]
    assert status["missing_preflight_products"] == ["btc_accumulation"]
    assert status["reasons"] == ["invalid_product_recommendations", "missing_preflight_products"]
    assert status["next_action"] == "make rehearse"


def test_offline_rehearsal_status_accepts_complete_workflow(tmp_path):
    summary = tmp_path / "rehearsal_summary.json"
    summary.write_text(
        json.dumps(
            {
                "ok": True,
                "work_dir": str(tmp_path),
                "products": {
                    "active_income": {
                        "artifact": str(tmp_path / "active.json"),
                        "trade_log": str(tmp_path / "active.csv"),
                        "promotion_review_json": str(tmp_path / "active_review.json"),
                        "before_recommendation": "needs_approval",
                        "after_recommendation": "already_approved",
                    },
                    "btc_accumulation": {
                        "artifact": str(tmp_path / "btc.json"),
                        "trade_log": str(tmp_path / "btc.csv"),
                        "promotion_review_json": str(tmp_path / "btc_review.json"),
                        "before_recommendation": "needs_approval",
                        "after_recommendation": "already_approved",
                    },
                },
                "preflight_report": str(tmp_path / "preflight.json"),
                "preflight_ok": True,
                "preflight_products": ["active_income", "btc_accumulation"],
            }
        ),
        encoding="utf-8",
    )

    status = readiness._offline_rehearsal_status(summary)

    assert status["ok"] is True
    assert status["reason"] == "ready"
    assert status["products"]["active_income"]["ok"] is True
    assert status["products"]["btc_accumulation"]["ok"] is True
    assert status["missing_preflight_products"] == []
    assert status["reasons"] == []


def passing_preflight_checks(product_config):
    market_type = "spot" if product_config.objective == "btc_accumulation" else "futures"
    exchange = "binance" if market_type == "spot" else "binanceusdm"
    exchange_detail = {
        "exchange": exchange,
        "market_type": market_type,
        "testnet": False,
        "require_testnet": False,
        "quote_asset": "USDT",
        "account_fingerprint": ExchangeConfig(
            exchange=exchange,
            market_type=market_type,
            api_key="key",
            testnet=False,
        ).account_fingerprint,
        "max_notional_usd": 100.0,
        "max_fill_slippage_bps": 100.0,
    }
    if market_type == "futures":
        exchange_detail["max_futures_leverage"] = 1
        exchange_detail["futures_margin_mode"] = "isolated"
    checks = [
        {"name": "product_config", "ok": True},
        {"name": "execution_engine_identity", "ok": True},
        {"name": "strategy_artifact_exists", "ok": True},
        {"name": "strategy_fingerprints", "ok": True},
        {"name": "strategy_policy", "ok": True},
        {"name": "exchange_environment", "ok": True, "detail": exchange_detail},
        {"name": "broker_constructed", "ok": True},
        {
            "name": "exchange_read_connectivity",
            "ok": True,
            "detail": {
                "price": 100.0,
                "balance": 1000.0,
                "position_qty": 0.0 if product_config.objective == "active_income" else 0.5,
                "position_avg_price": 0.0,
                "position_is_flat": product_config.objective == "active_income",
            },
        },
    ]
    if product_config.objective == "active_income" and product_config.market == "futures":
        checks.append(
            {
                "name": "broker_position_mode_one_way",
                "ok": True,
                "detail": {"symbol": product_config.symbol, "one_way": True},
            }
        )
        checks.append(
            {
                "name": "broker_native_protective_stops",
                "ok": True,
                "detail": {"supported": True},
            }
        )
        checks.append(
            {
                "name": "broker_open_orders_empty",
                "ok": True,
                "detail": {
                    "scope": "whole_account",
                    "configured_symbol": product_config.symbol,
                    "regular": {"count": 0, "orders": []},
                    "conditional": {"count": 0, "orders": []},
                },
            }
        )
        checks.append(
            {
                "name": "broker_position_flat",
                "ok": True,
                "detail": {
                    "scope": "whole_account",
                    "configured_symbol": product_config.symbol,
                    "count": 0,
                    "positions": [],
                },
            }
        )
    if product_config.objective == "btc_accumulation" and product_config.market == "spot":
        checks.append({"name": "broker_spot_position_non_negative", "ok": True})
    return checks


def mark_preflight_testnet_required(preflight):
    payload = json.loads(json.dumps(preflight))
    products = payload.get("products")
    if not isinstance(products, list):
        return payload
    for item in products:
        if not isinstance(item, dict):
            continue
        product_payload = item.get("product") if isinstance(item.get("product"), dict) else {}
        market_type = product_payload.get("market") or "futures"
        checks = item.get("checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if isinstance(check, dict) and check.get("name") == "exchange_environment":
                detail = check.get("detail") if isinstance(check.get("detail"), dict) else {}
                detail.update(
                    {
                        "exchange": "binance" if market_type == "spot" else "binanceusdm",
                        "market_type": market_type,
                        "testnet": True,
                        "require_testnet": True,
                        "quote_asset": "USDT",
                        "account_fingerprint": ExchangeConfig(
                            exchange="binance" if market_type == "spot" else "binanceusdm",
                            market_type=market_type,
                            api_key="key",
                            testnet=True,
                        ).account_fingerprint,
                        "max_notional_usd": 100.0,
                        "max_fill_slippage_bps": 100.0,
                    }
                )
                if market_type == "futures":
                    detail["max_futures_leverage"] = 1
                    detail["futures_margin_mode"] = "isolated"
                check["detail"] = detail
    return payload


def write_preflight(path, product_config, *, generated_ts=None):
    artifact = json.loads(product_config.strategies_path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "generated_ts": time.time() if generated_ts is None else generated_ts,
                "ok": True,
                "products": [
                    {
                        "artifact_fingerprints": [
                            strategy_fingerprint(strategy)
                            for strategy in artifact.get("strategies", [])
                        ],
                        "artifact_digest": artifact_digest(artifact),
                        "execution_engine_digest": execution_engine_digest(),
                        "ok": True,
                        "product": canonical_product_config(product_config),
                        "checks": passing_preflight_checks(product_config),
                        "errors": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def write_testnet_rehearsal(path, *, generated_ts=None, preflight=None):
    product_payload = {
        "name": "active_income",
        "objective": "active_income",
        "base_asset": "USDT",
        "market": "futures",
        "symbol": "BTCUSDT",
    }
    if isinstance(preflight, dict):
        products = preflight.get("products")
        if isinstance(products, list) and products and isinstance(products[0], dict):
            embedded = products[0].get("product")
            if isinstance(embedded, dict):
                product_payload = dict(embedded)
    payload = {
        "ok": True,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "generated_ts": time.time() if generated_ts is None else generated_ts,
        "product": product_payload,
        "exchange": "binanceusdm",
        "testnet": True,
        "risk_controls": {
            "max_futures_leverage": 1,
            "futures_margin_mode": "isolated",
            "max_notional_usd": 100.0,
            "max_fill_slippage_bps": 100.0,
        },
        "notional_usd": 5.0,
        "order_qty": 0.05,
        "entry_fill": {
            "symbol": "BTCUSDT",
            "side": "buy",
            "qty": 0.05,
            "price": 100.0,
            "fee": 0.01,
            "timestamp": 1000.0,
        },
        "close_fill": {
            "symbol": "BTCUSDT",
            "side": "sell",
            "qty": 0.05,
            "price": 100.0,
            "fee": 0.01,
            "timestamp": 1001.0,
        },
        "native_protective_stop": {
            "capability_supported": True,
            "native": True,
            "reduce_only": True,
            "trigger_distance_fraction": 0.05,
            "trigger_reference_price": 100.0,
            "raw_trigger_price": 95.0,
            "normalized_trigger_price": 95.0,
            "open_verified": True,
            "canceled_verified": True,
            "placed": {
                "symbol": "BTCUSDT",
                "side": "sell",
                "qty": 0.05,
                "trigger_price": 95.0,
                "status": "open",
                "order_id": "stop-1",
                "client_id": "testnet-stop-1",
                "filled_qty": 0.0,
                "average_price": None,
                "fee": 0.0,
            },
            "fetched_open": {
                "symbol": "BTCUSDT",
                "side": "sell",
                "qty": 0.05,
                "trigger_price": 95.0,
                "status": "open",
                "order_id": "stop-1",
                "client_id": "testnet-stop-1",
                "filled_qty": 0.0,
                "average_price": None,
                "fee": 0.0,
            },
            "cancel_result": {
                "symbol": "BTCUSDT",
                "side": "sell",
                "qty": 0.05,
                "trigger_price": 95.0,
                "status": "canceled",
                "order_id": "stop-1",
                "client_id": "testnet-stop-1",
                "filled_qty": 0.0,
                "average_price": None,
                "fee": 0.0,
            },
            "fetched_terminal": {
                "symbol": "BTCUSDT",
                "side": "sell",
                "qty": 0.05,
                "trigger_price": 95.0,
                "status": "canceled",
                "order_id": "stop-1",
                "client_id": "testnet-stop-1",
                "filled_qty": 0.0,
                "average_price": None,
                "fee": 0.0,
            },
        },
        "final_position_qty": 0.0,
    }
    if preflight is not None:
        payload["preflight"] = mark_preflight_testnet_required(preflight)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_readiness_allows_paper_product_waiting_for_artifact(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path)],
    )

    report = build_readiness_report(cfg, env={})
    markdown = render_readiness_markdown(report)

    assert report["ok"] is True
    assert report["blocking_count"] == 0
    check = next(
        item
        for item in report["checks"]
        if item["name"] == "active_income: paper strategy artifact"
    )
    assert check["ok"] is False
    assert check["level"] == "info"
    assert "product will wait" in check["detail"]
    assert "Autopilot Readiness" in markdown
    assert (
        next(item for item in report["checks"] if item["name"] == "alert log path writable")["ok"]
        is True
    )
    assert (
        next(
            item
            for item in report["checks"]
            if item["name"] == "alert cooldown state path writable"
        )["ok"]
        is True
    )


def test_readiness_blocks_missing_core_products_in_production_mode(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path)],
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True, require_core_products=True)

    check = next(item for item in report["checks"] if item["name"] == "autopilot config valid")
    assert check["ok"] is False
    assert "missing required product: btc_accumulation" in check["detail"]
    assert report["ok"] is False


def test_readiness_blocks_missing_core_jobs_in_production_mode(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[
            product(tmp_path),
            product(
                tmp_path,
                name="btc_accumulation",
                objective="btc_accumulation",
                base_asset="BTC",
                market="spot",
                strategies_path=tmp_path / "btc_strategies.json",
                state_file=tmp_path / "btc_state.json",
                trade_log=tmp_path / "btc_trades.csv",
                preflight_report=tmp_path / "btc_preflight.json",
                testnet_rehearsal_report=tmp_path / "btc_testnet.json",
            ),
        ],
    )

    report = build_readiness_report(
        cfg,
        env={},
        ccxt_available=True,
        require_core_products=True,
        require_core_jobs=True,
    )

    check = next(item for item in report["checks"] if item["name"] == "autopilot config valid")
    assert check["ok"] is False
    assert "missing required job: market_data_update_futures" in check["detail"]
    assert report["ok"] is False


def test_readiness_warns_when_offline_workflow_rehearsal_has_not_run(tmp_path, monkeypatch):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "install_autopilot_service.sh").write_text(
        VALID_SERVICE_INSTALLER, encoding="utf-8"
    )
    monkeypatch.setattr(readiness, "PROJECT_ROOT", tmp_path)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path)],
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    check = next(item for item in report["checks"] if item["name"] == "offline workflow rehearsal")
    assert report["ok"] is True
    assert check["ok"] is False
    assert check["level"] == "warning"
    assert check["detail"]["reason"] == "missing_report"
    assert check["detail"]["next_action"] == "make rehearse"


def test_readiness_blocks_malformed_control_file(tmp_path):
    control_file = tmp_path / "control.json"
    control_file.write_text(json.dumps({"paused": "treu"}), encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=control_file,
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path)],
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    check = next(item for item in report["checks"] if item["name"] == "control file valid")
    assert report["ok"] is False
    assert check["ok"] is False
    assert check["level"] == "error"
    assert check["detail"]["reason"] == "invalid_control_file"
    assert "paused must be a boolean" in check["detail"]["control_error"]


def test_readiness_blocks_symlink_control_file(tmp_path):
    control_file = tmp_path / "control.json"
    target = tmp_path / "external_control.json"
    target.write_text(json.dumps({"paused": False}), encoding="utf-8")
    control_file.symlink_to(target)
    cfg = AutopilotConfig(
        control_file=control_file,
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path)],
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    writable = next(item for item in report["checks"] if item["name"] == "control path writable")
    valid = next(item for item in report["checks"] if item["name"] == "control file valid")
    assert report["ok"] is False
    assert writable["ok"] is False
    assert valid["ok"] is False
    assert valid["detail"]["reason"] == "invalid_control_file"
    assert "control file must not be a symlink" in valid["detail"]["control_error"]


def test_readiness_blocks_unknown_control_selectors(tmp_path):
    control_file = tmp_path / "control.json"
    control_file.write_text(
        json.dumps(
            {
                "paused_products": ["active-incme"],
                "paused_jobs": ["market-data-update"],
                "flatten_products": ["active-incme"],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=control_file,
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        jobs=[
            JobConfig(
                name="market_data_update_futures",
                enabled=True,
                command=["python", "-m", "src.update_candles"],
                cadence_seconds=60,
            )
        ],
        products=[product(tmp_path)],
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    check = next(item for item in report["checks"] if item["name"] == "control file valid")
    assert report["ok"] is False
    assert check["ok"] is False
    assert check["detail"]["reason"] == "unknown_control_selectors"
    assert check["detail"]["unknown_selectors"] == {
        "flatten_products": ["active-incme"],
        "paused_jobs": ["market-data-update"],
        "paused_products": ["active-incme"],
    }


def test_readiness_blocks_unwritable_alert_paths(tmp_path, monkeypatch):
    alert_file = tmp_path / "readonly" / "alerts.jsonl"
    alert_state_file = tmp_path / "readonly" / "alert_state.json"
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=alert_file,
        alert_state_file=alert_state_file,
        products=[product(tmp_path)],
    )

    original_path_writable = readiness._path_writable

    def fake_path_writable(path):
        if path in {alert_file, alert_state_file}:
            return False
        return original_path_writable(path)

    monkeypatch.setattr(readiness, "_path_writable", fake_path_writable)

    report = build_readiness_report(cfg, env={})

    assert report["ok"] is False
    failed = {
        item["name"] for item in report["checks"] if item["level"] == "error" and not item["ok"]
    }
    assert "alert log path writable" in failed
    assert "alert cooldown state path writable" in failed


def test_readiness_blocks_symlink_runtime_lock_path(tmp_path):
    lock_path = tmp_path / "autopilot.lock"
    target = tmp_path / "external.lock"
    target.write_text("external\n", encoding="utf-8")
    lock_path.symlink_to(target)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        lock_file=lock_path,
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path)],
    )

    report = build_readiness_report(cfg, env={})

    check = next(item for item in report["checks"] if item["name"] == "runtime lock path writable")
    assert report["ok"] is False
    assert check["ok"] is False
    assert check["detail"] == str(lock_path)
    assert target.read_text(encoding="utf-8") == "external\n"


def test_readiness_blocks_symlink_scheduled_job_state_path(tmp_path):
    job_state = tmp_path / "job_state.json"
    target = tmp_path / "external_job_state.json"
    target.write_text('{"version": 1, "jobs": {}}\n', encoding="utf-8")
    job_state.symlink_to(target)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        job_state_file=job_state,
        products=[product(tmp_path)],
    )

    report = build_readiness_report(cfg, env={})

    check = next(
        item for item in report["checks"] if item["name"] == "scheduled job state path writable"
    )
    assert report["ok"] is False
    assert check["ok"] is False
    assert check["detail"] == str(job_state)
    assert target.read_text(encoding="utf-8") == '{"version": 1, "jobs": {}}\n'


def test_readiness_blocks_live_product_without_required_gates(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(cfg, env={"TRADING_LIVE": "0"}, ccxt_available=True)

    assert report["ok"] is False
    failed = {
        item["name"] for item in report["checks"] if item["level"] == "error" and not item["ok"]
    }
    assert "active_income: strategy artifact exists" in failed
    assert "active_income: preflight report exists" in failed
    assert "active_income: TRADING_LIVE=1" in failed
    assert "active_income: exchange API credentials" in failed


def test_readiness_warns_for_invalid_paper_artifact_without_blocking(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["strategies"][0]["metrics"]["holdout_total_return"] = -0.01
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, strategies_path=artifact)],
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    assert report["ok"] is True
    warning = next(
        item for item in report["checks"] if item["name"] == "active_income: strategy policy"
    )
    assert warning["ok"] is False
    assert warning["level"] == "warning"
    assert "holdout_total_return -0.010000 must be positive" in warning["detail"]


def test_readiness_passes_live_product_with_artifact_approval_preflight_and_env(tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    preflight = tmp_path / "preflight.json"
    testnet = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=testnet,
    )
    write_preflight(preflight, live_product)
    write_testnet_rehearsal(testnet, preflight=json.loads(preflight.read_text(encoding="utf-8")))
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
        products=[live_product],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_TESTNET": "0",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "isolated",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    assert report["ok"] is True
    failed_errors = [
        item for item in report["checks"] if item["level"] == "error" and not item["ok"]
    ]
    assert failed_errors == []


def test_readiness_loads_dotenv_when_env_is_not_explicit(tmp_path, monkeypatch):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    preflight = tmp_path / "preflight.json"
    testnet = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=testnet,
    )
    write_preflight(preflight, live_product)
    write_testnet_rehearsal(testnet, preflight=json.loads(preflight.read_text(encoding="utf-8")))
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TRADING_LIVE=1",
                "EXCHANGE_TESTNET=0",
                "EXCHANGE_API_KEY=dotenv-key",
                "EXCHANGE_API_SECRET=dotenv-secret",
                "MAX_NOTIONAL_USD=25",
                "FUTURES_MARGIN_MODE=isolated",
                "MAX_FUTURES_LEVERAGE=1",
                "FUTURES_EXCHANGE=binanceusdm",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env").chmod(0o600)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "install_autopilot_service.sh").write_text(
        VALID_SERVICE_INSTALLER, encoding="utf-8"
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
        products=[live_product],
    )
    for key in (
        "TRADING_LIVE",
        "EXCHANGE_TESTNET",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "MAX_NOTIONAL_USD",
        "FUTURES_MARGIN_MODE",
        "MAX_FUTURES_LEVERAGE",
        "FUTURES_EXCHANGE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(readiness, "PROJECT_ROOT", tmp_path)

    report = build_readiness_report(cfg, ccxt_available=True)

    assert report["ok"] is True
    failed_errors = [
        item for item in report["checks"] if item["level"] == "error" and not item["ok"]
    ]
    assert failed_errors == []


def test_readiness_dotenv_reader_handles_inline_comments_and_quotes(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TRADING_LIVE=1 # live flag",
                'AUTOPILOT_WEBHOOK_URL="https://example.test/hook#alerts" # channel',
                "TOKEN=abc#123",
                "SPACED='  quoted value  ' # comment",
            ]
        ),
        encoding="utf-8",
    )

    values = readiness._read_dotenv(env_file)

    assert values["TRADING_LIVE"] == "1"
    assert values["AUTOPILOT_WEBHOOK_URL"] == "https://example.test/hook#alerts"
    assert values["TOKEN"] == "abc#123"
    assert values["SPACED"] == "  quoted value  "


def test_readiness_dotenv_reader_defers_unreadable_file_to_status_check(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("EXCHANGE_API_KEY=secret\n", encoding="utf-8")
    original_read_text = Path.read_text

    def deny_env_read(path, *args, **kwargs):
        if path == env_file:
            raise PermissionError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_env_read)

    assert readiness._read_dotenv(env_file) == {}


def test_environment_file_status_requires_current_owner_and_private_mode(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("EXCHANGE_API_KEY=secret\n", encoding="utf-8")
    env_file.chmod(0o640)

    insecure = readiness._environment_file_status(env_file)

    assert insecure["ok"] is False
    assert insecure["reason"] == "credential_file_not_owner_private"
    assert insecure["mode"] == "0640"
    assert insecure["group_world_permissions"] == 0o40

    env_file.chmod(0o600)
    secure = readiness._environment_file_status(env_file)
    assert secure["ok"] is True
    assert secure["reason"] == "ready"


def test_readiness_parses_shipped_dotenv_example():
    values = readiness._read_dotenv(Path(".env.example"))

    assert readiness._env_bool(values, "TRADING_LIVE", True) is False
    assert readiness._env_bool(values, "EXCHANGE_TESTNET", False) is True
    assert readiness._env_float(values, "MAX_NOTIONAL_USD", 0.0) == 250.0
    assert readiness._env_float(values, "MAX_FILL_SLIPPAGE_BPS", 0.0) == 100.0
    assert values["FUTURES_MARGIN_MODE"] == "isolated"
    assert values["MAX_FUTURES_LEVERAGE"] == "1"


def test_readiness_ignores_symlink_dotenv_for_live_credentials(tmp_path, monkeypatch):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    preflight = tmp_path / "preflight.json"
    testnet = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=testnet,
    )
    write_preflight(preflight, live_product)
    write_testnet_rehearsal(testnet, preflight=json.loads(preflight.read_text(encoding="utf-8")))
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    target = tmp_path / "external.env"
    target.write_text(
        "\n".join(
            [
                "TRADING_LIVE=1",
                "EXCHANGE_TESTNET=1",
                "EXCHANGE_API_KEY=dotenv-key",
                "EXCHANGE_API_SECRET=dotenv-secret",
                "MAX_NOTIONAL_USD=25",
                "FUTURES_MARGIN_MODE=isolated",
                "MAX_FUTURES_LEVERAGE=1",
                "FUTURES_EXCHANGE=binanceusdm",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env").symlink_to(target)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "install_autopilot_service.sh").write_text(
        VALID_SERVICE_INSTALLER, encoding="utf-8"
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
        products=[live_product],
    )
    for key in (
        "TRADING_LIVE",
        "EXCHANGE_TESTNET",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "MAX_NOTIONAL_USD",
        "FUTURES_MARGIN_MODE",
        "MAX_FUTURES_LEVERAGE",
        "FUTURES_EXCHANGE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(readiness, "PROJECT_ROOT", tmp_path)

    report = build_readiness_report(cfg, ccxt_available=True)

    failed = {
        item["name"] for item in report["checks"] if item["level"] == "error" and not item["ok"]
    }
    assert report["ok"] is False
    assert "environment file not symlink" in failed
    assert "active_income: TRADING_LIVE=1" in failed
    assert "active_income: exchange API credentials" in failed
    assert (tmp_path / ".env").is_symlink()
    assert "EXCHANGE_API_KEY=dotenv-key" in target.read_text(encoding="utf-8")


def test_readiness_blocks_live_product_missing_required_testnet_rehearsal(tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    preflight = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=tmp_path / "missing_testnet.json",
    )
    write_preflight(preflight, live_product)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
        products=[live_product],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_TESTNET": "0",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "isolated",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    failed = {
        item["name"] for item in report["checks"] if item["level"] == "error" and not item["ok"]
    }
    assert "active_income: testnet rehearsal report exists" in failed


def test_readiness_blocks_invalid_exchange_boolean_env(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_TESTNET": "treu",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "isolated",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    check = next(
        item
        for item in report["checks"]
        if item["name"] == "active_income: exchange environment values"
    )
    assert report["ok"] is False
    assert check["ok"] is False
    assert check["detail"] == [
        "EXCHANGE_TESTNET must be a boolean flag: 1/0, true/false, yes/no, or on/off."
    ]


def test_readiness_blocks_blank_exchange_selector(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_TESTNET": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "isolated",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": " ",
        },
        ccxt_available=True,
    )

    check = next(
        item
        for item in report["checks"]
        if item["name"] == "active_income: exchange environment values"
    )
    assert report["ok"] is False
    assert check["ok"] is False
    assert check["detail"] == ["FUTURES_EXCHANGE must be non-empty."]


def test_readiness_blocks_blank_live_credentials(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_TESTNET": "1",
            "EXCHANGE_API_KEY": " ",
            "EXCHANGE_API_SECRET": " ",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "isolated",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    check = next(
        item
        for item in report["checks"]
        if item["name"] == "active_income: exchange API credentials"
    )
    assert report["ok"] is False
    assert check["ok"] is False


def test_readiness_passes_live_product_with_required_testnet_rehearsal(tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    preflight = tmp_path / "preflight.json"
    testnet = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=testnet,
    )
    write_preflight(preflight, live_product)
    write_testnet_rehearsal(testnet, preflight=json.loads(preflight.read_text(encoding="utf-8")))
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
        products=[live_product],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_TESTNET": "0",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "isolated",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    assert report["ok"] is True
    check = next(
        item
        for item in report["checks"]
        if item["name"] == "active_income: testnet rehearsal current"
    )
    assert check["ok"] is True
    assert check["detail"]["notional_usd"] == 5.0


def test_readiness_blocks_testnet_routing_for_configured_live_product(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_TESTNET": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "isolated",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    check = next(
        item
        for item in report["checks"]
        if item["name"] == "active_income: production exchange routing"
    )
    assert check["ok"] is False
    assert check["level"] == "error"


def test_readiness_blocks_stale_live_preflight_report(tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    preflight = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight,
        preflight_max_age_seconds=60,
    )
    write_preflight(preflight, live_product)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    stale_report = json.loads(preflight.read_text(encoding="utf-8"))
    stale_report["generated_ts"] = time.time() - 120
    preflight.write_text(json.dumps(stale_report), encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
        products=[live_product],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_TESTNET": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    check = next(
        item
        for item in report["checks"]
        if item["name"] == "active_income: preflight report current"
    )
    assert report["ok"] is False
    assert check["ok"] is False
    assert "preflight report is stale" in check["detail"]


def test_readiness_blocks_malformed_live_preflight_report(tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    preflight = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight,
    )
    write_preflight(preflight, live_product)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    preflight.write_text("[]", encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
        products=[live_product],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_TESTNET": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    check = next(
        item
        for item in report["checks"]
        if item["name"] == "active_income: preflight report current"
    )
    assert report["ok"] is False
    assert check["ok"] is False
    assert "preflight report must be a JSON object" in check["detail"]


def test_readiness_blocks_unsafe_futures_leverage(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "MAX_FUTURES_LEVERAGE": "10",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    failed = {
        item["name"] for item in report["checks"] if item["level"] == "error" and not item["ok"]
    }
    assert "active_income: max futures leverage" in failed


def test_readiness_blocks_active_income_leverage_above_one(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "MAX_FUTURES_LEVERAGE": "2",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    check = next(
        item for item in report["checks"] if item["name"] == "active_income: max futures leverage"
    )
    assert check["ok"] is False
    assert check["detail"] == {"value": 2.0, "required": 1}


def test_readiness_blocks_non_isolated_futures_margin(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "cross",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
        },
        ccxt_available=True,
    )

    failed = {
        item["name"] for item in report["checks"] if item["level"] == "error" and not item["ok"]
    }
    assert "active_income: futures margin mode" in failed


def test_readiness_blocks_non_binance_active_income_futures_exchange(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "isolated",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "bybit",
        },
        ccxt_available=True,
    )

    failed = {
        item["name"] for item in report["checks"] if item["level"] == "error" and not item["ok"]
    }
    assert "active_income: active-income futures exchange" in failed


def test_readiness_blocks_non_binance_btc_accumulation_spot_exchange(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[
            product(
                tmp_path,
                name="btc_accumulation",
                objective="btc_accumulation",
                base_asset="BTC",
                market="spot",
                execution_mode="live",
            )
        ],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "SPOT_EXCHANGE": "kraken",
        },
        ccxt_available=True,
    )

    failed = {
        item["name"] for item in report["checks"] if item["level"] == "error" and not item["ok"]
    }
    assert "btc_accumulation: BTC accumulation spot exchange" in failed


def test_readiness_accepts_binance_btc_accumulation_spot_exchange(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[
            product(
                tmp_path,
                name="btc_accumulation",
                objective="btc_accumulation",
                base_asset="BTC",
                market="spot",
                execution_mode="live",
            )
        ],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "SPOT_EXCHANGE": "binance",
        },
        ccxt_available=True,
    )

    check = next(
        item
        for item in report["checks"]
        if item["name"] == "btc_accumulation: BTC accumulation spot exchange"
    )
    assert check["ok"] is True
    assert check["detail"] == "binance"


def test_readiness_blocks_non_usdt_quote_asset(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_readiness_report(
        cfg,
        env={
            "TRADING_LIVE": "1",
            "EXCHANGE_API_KEY": "key",
            "EXCHANGE_API_SECRET": "secret",
            "MAX_NOTIONAL_USD": "25",
            "FUTURES_MARGIN_MODE": "isolated",
            "MAX_FUTURES_LEVERAGE": "1",
            "FUTURES_EXCHANGE": "binanceusdm",
            "QUOTE_ASSET": "USDC",
        },
        ccxt_available=True,
    )

    failed = {
        item["name"] for item in report["checks"] if item["level"] == "error" and not item["ok"]
    }
    assert "active_income: quote asset" in failed


def test_readiness_warns_when_required_indicator_features_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.autopilot.readiness.build_indicator_feature_statuses",
        lambda markets, **kwargs: {
            "futures": {
                "ok": False,
                "timeframes": {
                    "1m": {
                        "ok": False,
                        "reason": "missing_required_features",
                        "missing_features": ["volume_z_20"],
                    }
                },
            },
        },
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    check = next(item for item in report["checks"] if item["name"] == "indicator feature readiness")
    assert report["ok"] is True
    assert check["ok"] is False
    assert check["level"] == "warning"
    assert check["detail"]["futures"]["timeframes"]["1m"]["missing_features"] == ["volume_z_20"]


def test_readiness_warns_when_regime_data_report_is_missing(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        jobs=[
            JobConfig(
                name="regime_tag_futures_15m",
                enabled=True,
                command=[
                    ".venv/bin/python",
                    "-m",
                    "src.regime",
                    "--output",
                    str(tmp_path / "regime.parquet"),
                    "--report",
                    str(tmp_path / "missing_regime_report.json"),
                ],
                cadence_seconds=3600,
            )
        ],
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    check = next(item for item in report["checks"] if item["name"] == "regime data readiness")
    assert report["ok"] is True
    assert check["ok"] is False
    assert check["level"] == "warning"
    assert check["detail"][0]["reason"] == "missing_report"


def test_readiness_warns_when_strategy_smoke_report_is_missing(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        strategy_smoke_file=tmp_path / "missing_strategy_smoke.json",
        jobs=[
            JobConfig(
                name="strategy_framework_smoke",
                enabled=True,
                command=[
                    ".venv/bin/python",
                    "-m",
                    "src.autopilot.strategy_smoke",
                    "--output",
                    str(tmp_path / "missing_strategy_smoke.json"),
                ],
                cadence_seconds=86400,
            )
        ],
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    check = next(item for item in report["checks"] if item["name"] == "strategy framework smoke")
    assert report["ok"] is True
    assert check["ok"] is False
    assert check["level"] == "warning"
    assert check["detail"]["reason"] == "missing_report"


def test_readiness_skips_unconfigured_missing_strategy_smoke_report(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        strategy_smoke_file=tmp_path / "missing_strategy_smoke.json",
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    assert all(item["name"] != "strategy framework smoke" for item in report["checks"])


def test_readiness_warns_when_runtime_filesystem_has_low_free_space(tmp_path, monkeypatch):
    Usage = namedtuple("usage", "total used free")

    monkeypatch.setattr(
        "src.autopilot.readiness.shutil.disk_usage",
        lambda path: Usage(total=2_000, used=1_700, free=300),
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        min_runtime_free_bytes=500,
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    check = next(
        item for item in report["checks"] if item["name"] == "runtime filesystem free space"
    )
    audit_check = next(
        item for item in report["checks"] if item["name"] == "control audit path writable"
    )
    assert report["ok"] is True
    assert check["ok"] is False
    assert check["level"] == "warning"
    assert check["detail"]["free_bytes"] == 300
    assert check["detail"]["min_free_bytes"] == 500
    assert audit_check["ok"] is True


def test_readiness_blocks_malformed_approval_ledger(tmp_path):
    ledger = tmp_path / "approvals.json"
    ledger.write_text("[]", encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    check = next(item for item in report["checks"] if item["name"] == "approval ledger readable")
    assert report["ok"] is False
    assert check["ok"] is False
    assert check["detail"]["reason"] == "invalid_ledger"
    assert "Approval ledger must be a JSON object" in check["detail"]["error"]


def test_readiness_warns_for_blank_actor_approval_ledger_entries(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strategy = strategy_artifact(artifact)
    fingerprint = ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test"
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["approved_by"] = " "
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    readable = next(item for item in report["checks"] if item["name"] == "approval ledger readable")
    actor_audit = next(
        item for item in report["checks"] if item["name"] == "approval ledger actor audit"
    )
    assert report["ok"] is True
    assert readable["ok"] is True
    assert readable["detail"]["counts"] == {"invalid_actor": 1}
    assert readable["detail"]["invalid_actor_count"] == 1
    assert actor_audit["ok"] is False
    assert actor_audit["level"] == "warning"
    assert actor_audit["detail"]["entries"][0]["fingerprint"] == fingerprint


def test_readiness_warns_for_approval_fingerprint_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strategy = strategy_artifact(artifact)
    fingerprint = ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test"
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["fingerprint"] = "sha256:wrong"
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    readable = next(item for item in report["checks"] if item["name"] == "approval ledger readable")
    fingerprint_audit = next(
        item for item in report["checks"] if item["name"] == "approval ledger fingerprint audit"
    )
    assert report["ok"] is True
    assert readable["ok"] is True
    assert readable["detail"]["counts"] == {"fingerprint_mismatch": 1}
    assert readable["detail"]["fingerprint_mismatch_count"] == 1
    assert fingerprint_audit["ok"] is False
    assert fingerprint_audit["level"] == "warning"
    assert fingerprint_audit["detail"]["entries"][0]["fingerprint"] == fingerprint
    assert fingerprint_audit["detail"]["entries"][0]["entry_fingerprint"] == "sha256:wrong"


def test_readiness_warns_for_missing_approval_entry_fingerprint(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strategy = strategy_artifact(artifact)
    fingerprint = ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test"
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    del payload["approvals"][fingerprint]["fingerprint"]
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    readable = next(item for item in report["checks"] if item["name"] == "approval ledger readable")
    fingerprint_audit = next(
        item for item in report["checks"] if item["name"] == "approval ledger fingerprint audit"
    )
    assert report["ok"] is True
    assert readable["ok"] is True
    assert readable["detail"]["counts"] == {"fingerprint_mismatch": 1}
    assert readable["detail"]["fingerprint_mismatch_count"] == 1
    assert fingerprint_audit["ok"] is False
    assert fingerprint_audit["detail"]["entries"][0]["fingerprint"] == fingerprint
    assert fingerprint_audit["detail"]["entries"][0]["entry_fingerprint"] is None


def test_readiness_warns_for_invalid_revocation_audit_entries(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strategy = strategy_artifact(artifact)
    fingerprint = ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test"
    )
    ApprovalLedger(ledger).revoke(fingerprint, revoked_by="test", reason="paper drawdown breached")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["approvals"][fingerprint]["revoked_by"] = " "
    payload["approvals"][fingerprint]["revocation_reason"] = ""
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    readable = next(item for item in report["checks"] if item["name"] == "approval ledger readable")
    revocation_audit = next(
        item for item in report["checks"] if item["name"] == "approval ledger revocation audit"
    )
    assert report["ok"] is True
    assert readable["ok"] is True
    assert readable["detail"]["counts"] == {"invalid_revocation_audit": 1}
    assert readable["detail"]["invalid_revocation_count"] == 1
    assert revocation_audit["ok"] is False
    assert revocation_audit["level"] == "warning"
    assert revocation_audit["detail"]["entries"][0]["fingerprint"] == fingerprint
    assert revocation_audit["detail"]["entries"][0]["reasons"] == [
        "invalid_revoked_by",
        "missing_revocation_reason",
    ]


def test_readiness_allows_revocation_reason_to_mention_automation(tmp_path):
    artifact = tmp_path / "active.json"
    ledger = tmp_path / "approvals.json"
    strategy = strategy_artifact(artifact)
    fingerprint = ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test"
    )
    ApprovalLedger(ledger).revoke(fingerprint, revoked_by="test", reason="system outage")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=ledger,
    )

    report = build_readiness_report(cfg, env={}, ccxt_available=True)

    readable = next(item for item in report["checks"] if item["name"] == "approval ledger readable")
    assert report["ok"] is True
    assert readable["ok"] is True
    assert readable["detail"]["counts"] == {"revoked": 1}
    assert readable["detail"]["invalid_revocation_count"] == 0
    assert not any(item["name"] == "approval ledger revocation audit" for item in report["checks"])


def test_readiness_cli_writes_failure_report_when_config_load_fails(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "bad_config.json"
    markdown = tmp_path / "readiness.md"
    json_output = tmp_path / "readiness.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "readiness",
            "--config",
            str(config_path),
            "--output",
            str(markdown),
            "--json-output",
            str(json_output),
        ],
    )

    def fail_load(path):
        raise ValueError("bad config")

    monkeypatch.setattr("src.autopilot.readiness.load_config", fail_load)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert capsys.readouterr().out.strip() == str(markdown)
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload == {
        "ok": False,
        "blocking_count": 1,
        "warning_count": 0,
        "checks": [
            {
                "name": "readiness build failed",
                "ok": False,
                "level": "error",
                "detail": {"config": str(config_path), "error": "ValueError: bad config"},
            }
        ],
    }
    assert "readiness build failed" in markdown.read_text(encoding="utf-8")


def test_readiness_cli_prints_json_when_json_output_write_fails(monkeypatch, tmp_path, capsys):
    markdown = tmp_path / "readiness.md"
    json_output = tmp_path / "readiness.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "readiness",
            "--config",
            str(tmp_path / "config.json"),
            "--output",
            str(markdown),
            "--json-output",
            str(json_output),
        ],
    )
    monkeypatch.setattr("src.autopilot.readiness.load_config", lambda path: AutopilotConfig())
    monkeypatch.setattr(
        "src.autopilot.readiness.build_readiness_report",
        lambda config, **_kwargs: {
            "ok": True,
            "blocking_count": 0,
            "warning_count": 0,
            "checks": [],
        },
    )

    def fail_json(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr("src.autopilot.readiness.write_json_atomic", fail_json)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["blocking_count"] == 1
    assert printed["checks"] == [
        {
            "name": "readiness JSON output writable",
            "ok": False,
            "level": "error",
            "detail": {"path": str(json_output), "error": "OSError: disk full"},
        }
    ]
