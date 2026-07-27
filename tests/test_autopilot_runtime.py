import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_exploration.dsr import DSR_METHOD
from src.autopilot.approvals import (
    ApprovalError,
    ApprovalLedger,
    artifact_digest,
    strategy_fingerprint,
)
from src.autopilot.config import (
    AutopilotConfig,
    JobConfig,
    ProductConfig,
    canonical_product_config,
    load_config,
)
from src.autopilot.execution_identity import execution_engine_digest
from src.autopilot.runtime import (
    REQUIRED_CORE_JOB_FLAG_VALUES,
    REQUIRED_CORE_JOB_MODULES,
    REQUIRED_CORE_JOB_PRESENCE_FLAGS,
    REQUIRED_CORE_JOBS,
    _assert_current_environment_matches_preflight,
    _bot_status_snapshot,
    _effective_sleep_seconds,
    _local_state_requires_management,
    acquire_runtime_lock,
    assert_live_environment,
    assert_recent_preflight,
    assert_recent_testnet_rehearsal,
    build_live_broker,
    flatten_product_once,
    main,
    run_once,
    run_product_once,
    validate_config,
    write_cycle_reports,
)
from src.execution.broker import (
    Fill,
    OrderSide,
    OrderType,
    Position,
    ProtectiveOrder,
    ProtectiveOrderStatus,
)
from src.execution.config import ExchangeConfig
from src.run_bot import PaperTradingBot

TEST_ACCOUNT_FINGERPRINT = f"account-v1:{'a' * 64}"
OTHER_ACCOUNT_FINGERPRINT = f"account-v1:{'b' * 64}"


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


def product_config_payload(tmp_path, **overrides):
    payload = {
        "name": "active_income",
        "enabled": True,
        "objective": "active_income",
        "base_asset": "USDT",
        "market": "futures",
        "execution_mode": "paper",
        "symbol": "BTCUSDT",
        "strategies_path": str(tmp_path / "strategies.json"),
        "state_file": str(tmp_path / "state.json"),
        "trade_log": str(tmp_path / "trades.csv"),
        "preflight_report": str(tmp_path / "preflight.json"),
        "testnet_rehearsal_report": str(tmp_path / "testnet.json"),
        "starting_equity": 1000.0,
    }
    payload.update(overrides)
    return payload


def set_live_env(monkeypatch):
    monkeypatch.setenv("TRADING_LIVE", "1")
    monkeypatch.setenv("EXCHANGE_TESTNET", "0")
    monkeypatch.setenv("FUTURES_EXCHANGE", "binanceusdm")
    monkeypatch.setenv("EXCHANGE_API_KEY", "key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
    monkeypatch.setenv("MAX_NOTIONAL_USD", "100")
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "100")
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "1")


CORE_AUTOPILOT_JOBS = {
    "market_universe_screen",
    "market_data_update_universe",
    "market_data_update_universe_1m",
    "market_data_update_futures",
    "market_data_update_futures_1m",
    "market_data_update_spot",
    "regime_tag_futures_15m",
    "research_synthetic_smoke",
    "research_factory",
    "research_cycle",
    "strategy_framework_smoke",
    "active_income_promotion_review",
    "btc_accumulation_promotion_review",
    "runtime_maintenance",
    "artifact_hygiene",
}


def core_job(tmp_path, name, *, enabled=True, command=None):
    if command is None:
        command = [sys.executable, "-m", REQUIRED_CORE_JOB_MODULES[name]]
        for flag, values in REQUIRED_CORE_JOB_FLAG_VALUES.get(name, {}).items():
            command.extend([flag, *values])
        command.extend(REQUIRED_CORE_JOB_PRESENCE_FLAGS.get(name, ()))
    return JobConfig(
        name=name,
        enabled=enabled,
        command=command,
        cadence_seconds=60,
        working_dir=tmp_path,
    )


def test_checked_in_autopilot_configs_validate_and_cover_core_jobs():
    for config_path in (Path("config/autopilot.json"), Path("config/autopilot.example.json")):
        cfg = load_config(config_path)
        job_names = {job.name for job in cfg.jobs}

        assert validate_config(cfg, require_core_products=True, require_core_jobs=True) == []
        assert cfg.max_jobs_per_cycle == 1
        assert cfg.max_consecutive_job_deferrals == 16
        assert CORE_AUTOPILOT_JOBS <= job_names


def test_validate_config_rejects_legacy_inline_data_update():
    errors = validate_config(AutopilotConfig(run_data_update=True))

    assert errors == [
        "run_data_update=true is unsupported because inline downloads can block trading "
        "supervision; use the isolated market_data_update_* jobs"
    ]


def test_load_config_rejects_symlink_without_trusting_target(tmp_path):
    target = tmp_path / "target_autopilot.json"
    config_path = tmp_path / "autopilot.json"
    target.write_text(json.dumps({"products": []}), encoding="utf-8")
    config_path.symlink_to(target)

    with pytest.raises(ValueError, match="autopilot config must not be a symlink"):
        load_config(config_path)

    assert config_path.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"products": []}


def test_load_config_rejects_invalid_json(tmp_path):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text('{"products": [', encoding="utf-8")

    with pytest.raises(ValueError, match="autopilot config must be valid JSON"):
        load_config(config_path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_load_config_rejects_non_standard_json_constants(tmp_path, constant):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(f'{{"loop_sleep_seconds": {constant}}}', encoding="utf-8")

    with pytest.raises(ValueError, match=f"invalid JSON constant: {constant}"):
        load_config(config_path)


def test_load_config_rejects_duplicate_top_level_json_keys(tmp_path):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text('{"alerts_enabled": true, "alerts_enabled": false}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key: alerts_enabled"):
        load_config(config_path)


def test_load_config_rejects_nested_non_standard_json_constants(tmp_path):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(
        """
        {
          "products": [
            {
              "name": "active_income",
              "objective": "active_income",
              "base_asset": "USDT",
              "market": "futures",
              "strategies_path": "outputs/active_strategies_flow.json",
              "state_file": "runtime/active_income_state.json",
              "trade_log": "runtime/active_income_trades.csv",
              "starting_equity": NaN
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSON constant: NaN"):
        load_config(config_path)


def test_load_config_rejects_duplicate_nested_json_keys(tmp_path):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(
        """
        {
          "products": [
            {
              "name": "active_income",
              "objective": "active_income",
              "base_asset": "USDT",
              "market": "futures",
              "strategies_path": "outputs/active_strategies_flow.json",
              "state_file": "runtime/active_income_state.json",
              "trade_log": "runtime/active_income_trades.csv",
              "require_preflight": true,
              "require_preflight": false
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key: require_preflight"):
        load_config(config_path)


def test_load_config_rejects_non_object_payload(tmp_path):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="autopilot config must be a JSON object"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"jobs": {}}, "autopilot config jobs must be a list"),
        ({"jobs": ["bad"]}, r"autopilot config jobs\[0\] must be a JSON object"),
        ({"products": {}}, "autopilot config products must be a list"),
        ({"products": ["bad"]}, r"autopilot config products\[0\] must be a JSON object"),
    ],
)
def test_load_config_rejects_malformed_jobs_and_products(tmp_path, payload, message):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    ("job_payload", "message"),
    [
        ({"name": "bad", "command": [], "cadence_seconds": 60}, "job bad: command cannot be empty"),
        (
            {"name": "bad", "command": ["python"], "cadence_seconds": 0},
            "job bad: cadence_seconds must be positive",
        ),
        (
            {"name": "bad", "command": ["python"], "cadence_seconds": 60, "timeout_seconds": 0},
            "job bad: timeout_seconds must be positive",
        ),
        (
            {"name": "bad", "command": ["python", ""], "cadence_seconds": 60},
            r"job bad: command\[1\] must be non-empty",
        ),
        (
            {"name": "", "command": ["python"], "cadence_seconds": 60},
            "job name must be non-empty",
        ),
        (
            {"name": "bad", "enabled": "false", "command": ["python"], "cadence_seconds": 60},
            "job bad: enabled must be a JSON boolean",
        ),
        (
            {"name": "bad", "command": ["python"], "cadence_seconds": "60"},
            "job bad: cadence_seconds must be a JSON integer",
        ),
        (
            {"name": "bad", "command": ["python"], "cadence_seconds": 60, "timeout_seconds": True},
            "job bad: timeout_seconds must be a JSON integer",
        ),
        (
            {"name": "bad", "command": ["python"], "cadence_seconds": 60, "working_dir": True},
            "job bad: working_dir must be a string",
        ),
        (
            {"name": "bad", "command": ["python"], "cadence_seconds": 60, "enabledd": True},
            "job bad has unknown field: enabledd",
        ),
    ],
)
def test_load_config_rejects_malformed_job_values(tmp_path, job_payload, message):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(json.dumps({"jobs": [job_payload]}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    "jobs_payload",
    [
        {"not": "a list"},
        ["not an object"],
        [{"name": "broken", "command": [], "cadence_seconds": 0}],
    ],
)
def test_load_config_supervision_only_isolates_malformed_jobs(tmp_path, jobs_payload):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(
        json.dumps(
            {
                "jobs": jobs_payload,
                "max_jobs_per_cycle": "also job-only garbage",
                "run_data_update": "also job-only garbage",
                "products": [product_config_payload(tmp_path)],
            }
        ),
        encoding="utf-8",
    )

    configured = load_config(config_path, strict_jobs=False)

    assert configured.jobs == []
    assert configured.job_config_errors
    assert configured.max_jobs_per_cycle == 1
    assert configured.run_data_update is False
    assert [item.name for item in configured.products] == ["active_income"]


def test_load_config_supervision_only_retains_valid_jobs_beside_malformed_job(tmp_path):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "valid-status-context",
                        "command": [sys.executable, "-c", "print('ok')"],
                        "cadence_seconds": 60,
                    },
                    {"name": "broken", "command": [], "cadence_seconds": 0},
                ],
                "products": [product_config_payload(tmp_path)],
            }
        ),
        encoding="utf-8",
    )

    configured = load_config(config_path, strict_jobs=False)

    assert [job.name for job in configured.jobs] == ["valid-status-context"]
    assert configured.job_config_errors == ["jobs[1]: job broken: command cannot be empty"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("objective", 12, "product active_income objective must be a string"),
        ("base_asset", False, "product active_income base_asset must be a string"),
        ("market", "", "product active_income market must be non-empty"),
        ("execution_mode", [], "product active_income: execution_mode must be a string"),
        ("symbol", " ", "product active_income: symbol must be non-empty"),
    ],
)
def test_load_config_rejects_malformed_product_string_values(tmp_path, field, value, message):
    config_path = tmp_path / "autopilot.json"
    product_payload = product_config_payload(tmp_path, **{field: value})
    config_path.write_text(json.dumps({"products": [product_payload]}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("objective", "product active_income must include objective"),
        ("strategies_path", "product active_income must include strategies_path"),
    ],
)
def test_load_config_rejects_missing_required_product_fields(tmp_path, field, message):
    config_path = tmp_path / "autopilot.json"
    product_payload = product_config_payload(tmp_path)
    product_payload.pop(field)
    config_path.write_text(json.dumps({"products": [product_payload]}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strategies_path", [], "product active_income: strategies_path must be a string"),
        ("state_file", "", "product active_income: state_file must be non-empty"),
        ("trade_log", True, "product active_income: trade_log must be a string"),
        ("preflight_report", {}, "product active_income: preflight_report must be a string"),
        (
            "testnet_rehearsal_report",
            " ",
            "product active_income: testnet_rehearsal_report must be non-empty",
        ),
    ],
)
def test_load_config_rejects_malformed_product_path_values(tmp_path, field, value, message):
    config_path = tmp_path / "autopilot.json"
    product_payload = product_config_payload(tmp_path, **{field: value})
    config_path.write_text(json.dumps({"products": [product_payload]}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_load_config_rejects_unknown_product_keys(tmp_path):
    config_path = tmp_path / "autopilot.json"
    product_payload = product_config_payload(tmp_path, require_prefligth=True)
    config_path.write_text(json.dumps({"products": [product_payload]}), encoding="utf-8")

    with pytest.raises(
        ValueError, match="product active_income has unknown field: require_prefligth"
    ):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "starting_equity",
            "1000",
            "product active_income: starting_equity must be a finite JSON number",
        ),
        ("starting_equity", float("nan"), "invalid JSON constant: NaN"),
        ("starting_equity", 0, "product active_income: starting_equity must be positive"),
        (
            "regime_mayer_top",
            True,
            "product active_income: regime_mayer_top must be a finite JSON number",
        ),
        (
            "preflight_max_age_seconds",
            3600.5,
            "product active_income: preflight_max_age_seconds must be a JSON integer",
        ),
        (
            "testnet_rehearsal_max_age_seconds",
            "2592000",
            "product active_income: testnet_rehearsal_max_age_seconds must be a JSON integer",
        ),
    ],
)
def test_load_config_rejects_malformed_product_numeric_values(tmp_path, field, value, message):
    config_path = tmp_path / "autopilot.json"
    product_payload = product_config_payload(tmp_path, **{field: value})
    config_path.write_text(json.dumps({"products": [product_payload]}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_load_config_rejects_overflowing_json_numbers(tmp_path):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(
        """
        {
          "products": [
            {
              "name": "active_income",
              "objective": "active_income",
              "base_asset": "USDT",
              "market": "futures",
              "strategies_path": "outputs/active_strategies_flow.json",
              "state_file": "runtime/active_income_state.json",
              "trade_log": "runtime/active_income_trades.csv",
              "starting_equity": 1e999
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="product active_income: starting_equity must be finite"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", "false"),
        ("regime_guard", "true"),
        ("require_preflight", "false"),
        ("require_testnet_rehearsal", 1),
    ],
)
def test_load_config_rejects_non_boolean_product_flags(tmp_path, field, value):
    config_path = tmp_path / "autopilot.json"
    product_payload = product_config_payload(tmp_path, **{field: value})
    config_path.write_text(json.dumps({"products": [product_payload]}), encoding="utf-8")

    with pytest.raises(ValueError, match=f"product active_income: {field} must be a JSON boolean"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "jobs": [
                    {"name": "dup", "command": ["python"], "cadence_seconds": 60},
                    {"name": "dup", "command": ["python"], "cadence_seconds": 120},
                ]
            },
            "duplicate job name: dup",
        ),
        (
            {
                "products": [
                    {
                        "name": "dup",
                        "objective": "active_income",
                        "base_asset": "USDT",
                        "market": "futures",
                        "strategies_path": "outputs/a.json",
                        "state_file": "runtime/a.json",
                        "trade_log": "runtime/a.csv",
                    },
                    {
                        "name": "dup",
                        "objective": "active_income",
                        "base_asset": "USDT",
                        "market": "futures",
                        "strategies_path": "outputs/b.json",
                        "state_file": "runtime/b.json",
                        "trade_log": "runtime/b.csv",
                    },
                ]
            },
            "duplicate product name: dup",
        ),
        ({"loop_sleep_seconds": 0}, "loop_sleep_seconds must be positive"),
        ({"max_jobs_per_cycle": 0}, "max_jobs_per_cycle must be positive"),
        ({"max_consecutive_job_deferrals": 0}, "max_consecutive_job_deferrals must be positive"),
        ({"alert_cooldown_seconds": -1}, "alert_cooldown_seconds must be non-negative"),
        ({"min_runtime_free_bytes": 0}, "min_runtime_free_bytes must be positive"),
        ({"auto_report_enabled": "false"}, "auto_report_enabled must be a JSON boolean"),
        ({"alerts_enabled": "true"}, "alerts_enabled must be a JSON boolean"),
        ({"run_data_update": 0}, "run_data_update must be a JSON boolean"),
        ({"loop_sleep_seconds": "60"}, "loop_sleep_seconds must be a JSON integer"),
        ({"max_jobs_per_cycle": True}, "max_jobs_per_cycle must be a JSON integer"),
        (
            {"max_consecutive_job_deferrals": "3"},
            "max_consecutive_job_deferrals must be a JSON integer",
        ),
        ({"alert_cooldown_seconds": False}, "alert_cooldown_seconds must be a JSON integer"),
        ({"min_runtime_free_bytes": 536870912.0}, "min_runtime_free_bytes must be a JSON integer"),
        ({"control_file": ""}, "control_file must be non-empty"),
        ({"operator_report_file": []}, "operator_report_file must be a string"),
        ({"webhook_url_env": 2}, "webhook_url_env must be a string"),
        ({"alerts_enabledd": True}, "autopilot config has unknown field: alerts_enabledd"),
    ],
)
def test_load_config_rejects_invalid_global_scheduler_values(tmp_path, payload, message):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_load_config_accepts_json_boolean_false_values(tmp_path):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(
        json.dumps(
            {
                "auto_report_enabled": False,
                "alerts_enabled": False,
                "run_data_update": False,
                "jobs": [
                    {
                        "name": "disabled_job",
                        "enabled": False,
                        "command": ["python"],
                        "cadence_seconds": 60,
                    }
                ],
                "products": [
                    product_config_payload(
                        tmp_path,
                        enabled=False,
                        regime_guard=False,
                        require_preflight=False,
                        require_testnet_rehearsal=False,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.auto_report_enabled is False
    assert cfg.alerts_enabled is False
    assert cfg.run_data_update is False
    assert cfg.jobs[0].enabled is False
    assert cfg.products[0].enabled is False
    assert cfg.products[0].regime_guard is False
    assert cfg.products[0].require_preflight is False
    assert cfg.products[0].require_testnet_rehearsal is False


def exchange_environment_detail(product_config, *, testnet=False, require_testnet=False):
    market_type = "spot" if product_config.objective == "btc_accumulation" else "futures"
    exchange = "binance" if market_type == "spot" else "binanceusdm"
    detail = {
        "exchange": exchange,
        "market_type": market_type,
        "testnet": testnet,
        "require_testnet": require_testnet,
        "quote_asset": "USDT",
        "account_fingerprint": ExchangeConfig(
            exchange=exchange,
            market_type=market_type,
            api_key="key",
            testnet=testnet,
        ).account_fingerprint,
        "max_notional_usd": 100.0,
        "max_fill_slippage_bps": 100.0,
    }
    if market_type == "futures":
        detail["max_futures_leverage"] = 1
        detail["futures_margin_mode"] = "isolated"
    return detail


def mark_preflight_testnet_required(preflight):
    payload = json.loads(json.dumps(preflight))
    products = payload.get("products")
    if not isinstance(products, list):
        return payload
    for item in products:
        if not isinstance(item, dict):
            continue
        product_payload = item.get("product") if isinstance(item.get("product"), dict) else {}
        product_config = product(Path("."), **product_payload)
        checks = item.get("checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if isinstance(check, dict) and check.get("name") == "exchange_environment":
                check["detail"] = exchange_environment_detail(
                    product_config,
                    testnet=True,
                    require_testnet=True,
                )
    return payload


def write_testnet_rehearsal(
    path,
    *,
    ok=True,
    generated_ts=None,
    product_name="active_income",
    product_payload=None,
    risk_controls=None,
    testnet=True,
    exchange="binanceusdm",
    preflight=None,
):
    if product_payload is None and isinstance(preflight, dict):
        preflight_products = preflight.get("products")
        if isinstance(preflight_products, list) and preflight_products:
            first_entry = preflight_products[0]
            if isinstance(first_entry, dict):
                first_product = first_entry.get("product")
                if isinstance(first_product, dict):
                    product_payload = dict(first_product)
    product_payload = product_payload or {
        "name": product_name,
        "objective": "active_income",
        "base_asset": "USDT",
        "market": "futures",
        "symbol": "BTCUSDT",
    }
    payload = {
        "ok": ok,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "generated_ts": time.time() if generated_ts is None else generated_ts,
        "product": product_payload,
        "exchange": exchange,
        "testnet": testnet,
        "risk_controls": risk_controls
        or {
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


def test_validate_config_rejects_btc_accumulation_on_futures(tmp_path):
    cfg = AutopilotConfig(
        products=[
            product(
                tmp_path,
                name="btc_accumulation",
                objective="btc_accumulation",
                base_asset="BTC",
                market="futures",
            )
        ]
    )

    errors = validate_config(cfg)

    assert errors == ["btc_accumulation: BTC accumulation must use spot market"]


def test_validate_config_rejects_active_income_on_spot(tmp_path):
    cfg = AutopilotConfig(products=[product(tmp_path, market="spot")])

    assert validate_config(cfg) == ["active_income: active income must use futures market"]


def test_validate_config_requires_core_products_in_production_mode(tmp_path):
    cfg = AutopilotConfig(products=[product(tmp_path)])

    assert validate_config(cfg, require_core_products=True) == [
        "missing required product: btc_accumulation"
    ]


def test_validate_config_requires_core_products_enabled_in_production_mode(tmp_path):
    cfg = AutopilotConfig(
        products=[
            product(
                tmp_path,
                name="btc_accumulation",
                objective="btc_accumulation",
                base_asset="BTC",
                market="spot",
                strategies_path=tmp_path / "btc.json",
                state_file=tmp_path / "btc_state.json",
                trade_log=tmp_path / "btc_trades.csv",
                preflight_report=tmp_path / "btc_preflight.json",
                enabled=False,
            ),
            product(tmp_path),
        ]
    )

    assert validate_config(cfg, require_core_products=True) == [
        "btc_accumulation: required product must be enabled"
    ]


def test_validate_config_requires_core_jobs_in_production_mode():
    cfg = AutopilotConfig(jobs=[])

    errors = validate_config(cfg, require_core_jobs=True)

    assert "missing required job: market_data_update_futures" in errors
    assert "missing required job: artifact_hygiene" in errors


def test_validate_config_requires_core_jobs_enabled_in_production_mode(tmp_path):
    cfg = AutopilotConfig(
        jobs=[
            core_job(tmp_path, name, enabled=name != "research_cycle")
            for name in REQUIRED_CORE_JOBS
        ]
    )

    assert validate_config(cfg, require_core_jobs=True) == [
        "research_cycle: required job must be enabled"
    ]


def test_validate_config_requires_core_jobs_to_run_expected_modules_in_production_mode(tmp_path):
    bad_command = [sys.executable, "-m", "src.autopilot.research_smoke"]
    for flag, values in REQUIRED_CORE_JOB_FLAG_VALUES["research_cycle"].items():
        bad_command.extend([flag, *values])
    bad_command.extend(REQUIRED_CORE_JOB_PRESENCE_FLAGS["research_cycle"])
    cfg = AutopilotConfig(
        jobs=[
            core_job(
                tmp_path,
                name,
                command=bad_command if name == "research_cycle" else None,
            )
            for name in REQUIRED_CORE_JOBS
        ]
    )

    assert validate_config(cfg, require_core_jobs=True) == [
        "research_cycle: required job must run python module src.autopilot.research_cycle"
        " (got src.autopilot.research_smoke)"
    ]


def test_validate_config_requires_core_jobs_to_keep_expected_arguments_in_production_mode(tmp_path):
    bad_command = [
        sys.executable,
        "-m",
        "src.autopilot.history_bootstrap",
        "--market",
        "spot",
        "--timeframes",
        "1h",
    ]
    cfg = AutopilotConfig(
        jobs=[
            core_job(
                tmp_path,
                name,
                command=bad_command if name == "market_data_update_futures" else None,
            )
            for name in REQUIRED_CORE_JOBS
        ]
    )

    assert validate_config(cfg, require_core_jobs=True) == [
        "market_data_update_futures: required job must not include --timeframes",
        "market_data_update_futures: required job must include --config config/research_factory.json",
        "market_data_update_futures: required job --market must equal futures (got spot)",
        "market_data_update_futures: required job must include --exclude-timeframes 1m",
        "market_data_update_futures: required job must include --report runtime/history_bootstrap_futures.json",
    ]


def test_validate_config_accepts_core_job_required_arguments_as_inline_flags(tmp_path):
    inline_command = [
        sys.executable,
        "-m",
        "src.autopilot.history_bootstrap",
        "--config=config/research_factory.json",
        "--market=futures",
        "--exclude-timeframes=1m",
        "--report=runtime/history_bootstrap_futures.json",
    ]
    cfg = AutopilotConfig(
        jobs=[
            core_job(
                tmp_path,
                name,
                command=inline_command if name == "market_data_update_futures" else None,
            )
            for name in REQUIRED_CORE_JOBS
        ]
    )

    assert validate_config(cfg, require_core_jobs=True) == []


def test_validate_config_rejects_missing_history_report_path(tmp_path):
    inline_command = [
        sys.executable,
        "-m",
        "src.autopilot.history_bootstrap",
        "--config=config/research_factory.json",
        "--market=futures",
        "--exclude-timeframes=1m",
    ]
    cfg = AutopilotConfig(
        jobs=[
            core_job(
                tmp_path,
                name,
                command=inline_command if name == "market_data_update_futures" else None,
            )
            for name in REQUIRED_CORE_JOBS
        ]
    )

    assert validate_config(cfg, require_core_jobs=True) == [
        "market_data_update_futures: required job must include --report runtime/history_bootstrap_futures.json"
    ]


def test_validate_config_rejects_duplicate_or_extra_history_partition_flags(tmp_path):
    command = [
        sys.executable,
        "-m",
        "src.autopilot.history_bootstrap",
        "--config",
        "config/research_factory.json",
        "--config",
        "config/other.json",
        "--market",
        "futures",
        "--exclude-timeframes",
        "1m",
        "--exclude-timeframes",
        "5m",
        "--timeframes",
        "5m",
        "--report",
        "runtime/history_bootstrap_futures.json",
    ]
    cfg = AutopilotConfig(
        jobs=[
            core_job(
                tmp_path,
                name,
                command=command if name == "market_data_update_futures" else None,
            )
            for name in REQUIRED_CORE_JOBS
        ]
    )

    assert validate_config(cfg, require_core_jobs=True) == [
        "market_data_update_futures: required job must not include --timeframes",
        "market_data_update_futures: required job --config must equal "
        "config/research_factory.json (got config/research_factory.json config/other.json)",
        "market_data_update_futures: required job --exclude-timeframes must equal 1m (got 1m 5m)",
    ]


@pytest.mark.parametrize("symbol", ["BTCUSDT", "BTC/USDT", "BTC/USDT:USDT"])
def test_validate_config_accepts_btc_usdt_symbol_forms(tmp_path, symbol):
    cfg = AutopilotConfig(products=[product(tmp_path, symbol=symbol)])

    assert validate_config(cfg) == []


@pytest.mark.parametrize("symbol", ["BTCUSDC", "ETH/BTC", "BTC/USDT:USDC"])
def test_validate_config_rejects_active_income_wrong_symbol(tmp_path, symbol):
    cfg = AutopilotConfig(products=[product(tmp_path, symbol=symbol)])

    errors = validate_config(cfg)

    assert any(
        "active income" in error
        and ("symbol must be a USDT pair" in error or "settlement must be USDT" in error)
        for error in errors
    )


@pytest.mark.parametrize("symbol", ["ETHUSDT", "DOGE/USDT", "SOL/USDT:USDT"])
def test_validate_config_accepts_active_income_altcoin_usdt_symbols(tmp_path, symbol):
    cfg = AutopilotConfig(products=[product(tmp_path, symbol=symbol)])

    assert validate_config(cfg) == []


@pytest.mark.parametrize("symbol", ["ETHUSDT", "BTCUSDC", "BTC/USDT:USDT"])
def test_validate_config_rejects_btc_accumulation_wrong_symbol(tmp_path, symbol):
    cfg = AutopilotConfig(
        products=[
            product(
                tmp_path,
                name="btc_accumulation",
                objective="btc_accumulation",
                base_asset="BTC",
                market="spot",
                symbol=symbol,
            )
        ]
    )

    errors = validate_config(cfg)

    assert any(
        "BTC accumulation" in error
        and ("symbol must be BTC/USDT" in error or "must not include" in error)
        for error in errors
    )


def test_validate_config_rejects_unknown_objective_and_market(tmp_path):
    cfg = AutopilotConfig(
        products=[
            product(
                tmp_path,
                name="bad_product",
                objective="market_making",
                market="margin",
            )
        ]
    )

    assert validate_config(cfg) == [
        "bad_product: objective must be 'btc_accumulation' or 'active_income'",
        "bad_product: market must be 'spot' or 'futures'",
    ]


def test_validate_config_rejects_bad_job(tmp_path):
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="bad",
                enabled=True,
                command=[],
                cadence_seconds=0,
                timeout_seconds=0,
                working_dir=tmp_path,
            )
        ],
    )

    assert validate_config(cfg) == [
        "job bad: cadence_seconds must be positive",
        "job bad: timeout_seconds must be positive",
        "job bad: command cannot be empty",
    ]


def test_validate_config_rejects_missing_job_working_dir(tmp_path):
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="bad_workdir",
                enabled=True,
                command=[sys.executable, "-c", "print('ok')"],
                cadence_seconds=60,
                working_dir=tmp_path / "missing",
            )
        ],
    )

    assert validate_config(cfg) == [
        f"job bad_workdir: working_dir does not exist: {tmp_path / 'missing'}"
    ]


def test_validate_config_rejects_missing_job_executable(tmp_path):
    missing_executable = tmp_path / "missing-bin"
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="bad_exe",
                enabled=True,
                command=[str(missing_executable), "--version"],
                cadence_seconds=60,
                working_dir=tmp_path,
            )
        ],
    )

    assert validate_config(cfg) == [f"job bad_exe: executable does not exist: {missing_executable}"]


def test_validate_config_rejects_shell_job_commands(tmp_path):
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="shell_job",
                enabled=True,
                command=["sh", "-c", "echo unsafe"],
                cadence_seconds=60,
                working_dir=tmp_path,
            )
        ],
    )

    assert validate_config(cfg) == ["job shell_job: command must not use a shell executable (sh)"]


def test_validate_config_rejects_scheduled_approval_cli_jobs(tmp_path):
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="auto_approve",
                enabled=True,
                command=[sys.executable, "-m", "src.autopilot.approvals", "list"],
                cadence_seconds=60,
                working_dir=tmp_path,
            )
        ],
    )

    assert validate_config(cfg) == [
        "job auto_approve: scheduled jobs must not run approval-gate module src.autopilot.approvals"
    ]


def test_validate_config_rejects_missing_python_job_module(tmp_path):
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="bad_module",
                enabled=True,
                command=[sys.executable, "-m", "autopilot_missing_module_for_test"],
                cadence_seconds=60,
                working_dir=tmp_path,
            )
        ],
    )

    assert validate_config(cfg) == [
        "job bad_module: python module 'autopilot_missing_module_for_test' is not importable"
    ]


def test_validate_config_can_probe_transitive_job_dependencies(monkeypatch, tmp_path):
    module_name = "autopilot_broken_dependency_job"
    (tmp_path / f"{module_name}.py").write_text(
        "import dependency_that_is_intentionally_missing_for_test\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="broken_dependency",
                enabled=True,
                command=[sys.executable, "-m", module_name],
                cadence_seconds=60,
                working_dir=tmp_path,
            )
        ],
    )

    errors = validate_config(cfg, verify_job_imports=True)

    assert len(errors) == 1
    assert "job broken_dependency" in errors[0]
    assert "dependency import failed" in errors[0]
    assert "dependency_that_is_intentionally_missing_for_test" in errors[0]


@pytest.mark.parametrize(("skip_jobs", "verify_job_imports"), [(True, False), (False, True)])
def test_main_only_probes_job_imports_when_scheduled_jobs_are_enabled(
    monkeypatch,
    tmp_path,
    skip_jobs,
    verify_job_imports,
):
    configured = AutopilotConfig(products=[])
    seen = {}
    monkeypatch.setattr("src.autopilot.runtime.configure_logging", lambda: None)
    monkeypatch.setattr(
        "src.autopilot.runtime.parse_args",
        lambda: SimpleNamespace(
            config=tmp_path / "autopilot.json",
            once=True,
            validate=False,
            skip_jobs=skip_jobs,
            sleep=None,
        ),
    )

    def load(path, *, strict_jobs=True):
        seen["strict_jobs"] = strict_jobs
        return configured

    monkeypatch.setattr("src.autopilot.runtime.load_config", load)

    def validate(config, **kwargs):
        seen.update(kwargs)
        return ["stop after validation probe"]

    monkeypatch.setattr("src.autopilot.runtime.validate_config", validate)

    with pytest.raises(SystemExit, match="stop after validation probe"):
        main()

    assert seen == {
        "strict_jobs": not skip_jobs,
        "require_core_products": True,
        "require_core_jobs": not skip_jobs,
        "verify_job_imports": verify_job_imports,
        "validate_jobs": not skip_jobs,
    }


def test_main_once_exits_nonzero_when_cycle_report_fails(monkeypatch, tmp_path):
    configured = AutopilotConfig(
        lock_file=tmp_path / "autopilot.lock",
        status_file=tmp_path / "status.json",
    )
    monkeypatch.setattr("src.autopilot.runtime.configure_logging", lambda: None)
    monkeypatch.setattr(
        "src.autopilot.runtime.parse_args",
        lambda: SimpleNamespace(
            config=tmp_path / "autopilot.json",
            once=True,
            validate=False,
            skip_jobs=True,
            sleep=None,
        ),
    )
    monkeypatch.setattr(
        "src.autopilot.runtime.load_config",
        lambda path, strict_jobs: configured,
    )
    monkeypatch.setattr("src.autopilot.runtime.validate_config", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "src.autopilot.runtime.run_once",
        lambda config, run_jobs: {"ok": False},
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1


def test_validate_config_rejects_missing_python_module_argument(tmp_path):
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="missing_module_arg",
                enabled=True,
                command=[sys.executable, "-m"],
                cadence_seconds=60,
                working_dir=tmp_path,
            )
        ],
    )

    assert validate_config(cfg) == [
        "job missing_module_arg: python -m command is missing a module name"
    ]


def test_validate_config_rejects_duplicate_job_names(tmp_path):
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="market_data_update",
                enabled=True,
                command=[sys.executable, "-m", "src.update_candles"],
                cadence_seconds=60,
                working_dir=tmp_path,
            ),
            JobConfig(
                name="market_data_update",
                enabled=True,
                command=[sys.executable, "-m", "src.update_candles"],
                cadence_seconds=120,
                working_dir=tmp_path,
            ),
        ],
    )

    assert validate_config(cfg) == ["duplicate job name: market_data_update"]


def test_validate_config_rejects_duplicate_job_output_paths(tmp_path):
    shared = tmp_path / "runtime" / "shared.json"
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="research_cycle",
                enabled=True,
                command=[
                    sys.executable,
                    "-m",
                    "src.autopilot.research_cycle",
                    "--output",
                    str(shared),
                ],
                cadence_seconds=86400,
                working_dir=tmp_path,
            ),
            JobConfig(
                name="mutation_plan",
                enabled=True,
                command=[
                    sys.executable,
                    "-m",
                    "src.autopilot.mutation_plan",
                    "--output",
                    str(shared),
                ],
                cadence_seconds=86400,
                working_dir=tmp_path,
            ),
        ],
    )

    assert validate_config(cfg) == [
        f"job mutation_plan: output path {shared} for --output duplicates research_cycle --output"
    ]


def test_validate_config_allows_job_input_to_read_another_job_output(tmp_path):
    research_output = tmp_path / "runtime" / "research_cycle.json"
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="research_cycle",
                enabled=True,
                command=[
                    sys.executable,
                    "-m",
                    "src.autopilot.research_cycle",
                    "--output",
                    str(research_output),
                ],
                cadence_seconds=86400,
                working_dir=tmp_path,
            ),
            JobConfig(
                name="mutation_plan",
                enabled=True,
                command=[
                    sys.executable,
                    "-m",
                    "src.autopilot.mutation_plan",
                    "--input",
                    str(research_output),
                    "--output",
                    str(tmp_path / "runtime" / "mutation_plan.json"),
                ],
                cadence_seconds=86400,
                working_dir=tmp_path,
            ),
        ],
    )

    assert validate_config(cfg) == []


def test_validate_config_rejects_job_output_flag_without_path(tmp_path):
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="bad_output",
                enabled=True,
                command=[sys.executable, "-m", "src.autopilot.research_smoke", "--output"],
                cadence_seconds=86400,
                working_dir=tmp_path,
            ),
        ],
    )

    assert validate_config(cfg) == ["job bad_output: output flag --output must include a path"]


def test_validate_config_rejects_inline_job_output_flag_without_path(tmp_path):
    cfg = AutopilotConfig(
        products=[],
        jobs=[
            JobConfig(
                name="bad_inline_output",
                enabled=True,
                command=[sys.executable, "-m", "src.autopilot.research_smoke", "--output="],
                cadence_seconds=86400,
                working_dir=tmp_path,
            ),
        ],
    )

    assert validate_config(cfg) == [
        "job bad_inline_output: output flag --output must include a path"
    ]


def test_validate_config_rejects_duplicate_global_runtime_paths(tmp_path):
    shared = tmp_path / "runtime" / "shared.json"
    cfg = AutopilotConfig(
        status_file=shared,
        job_state_file=shared,
        products=[],
    )

    assert validate_config(cfg) == [f"job_state_file path duplicates status_file: {shared}"]


def test_validate_config_rejects_product_cross_field_runtime_path_collision(tmp_path):
    shared = tmp_path / "runtime" / "active_income_state.json"
    cfg = AutopilotConfig(
        products=[
            product(
                tmp_path,
                state_file=shared,
                trade_log=shared,
            )
        ],
    )

    assert validate_config(cfg) == [
        f"active_income trade_log path duplicates active_income state_file: {shared}"
    ]


def test_validate_config_rejects_job_output_to_protected_runtime_file(tmp_path):
    status_file = tmp_path / "runtime" / "status.json"
    cfg = AutopilotConfig(
        status_file=status_file,
        products=[],
        jobs=[
            JobConfig(
                name="bad_status_writer",
                enabled=True,
                command=[
                    sys.executable,
                    "-m",
                    "src.autopilot.research_smoke",
                    "--output",
                    str(status_file),
                ],
                cadence_seconds=86400,
                working_dir=tmp_path,
            ),
        ],
    )

    assert validate_config(cfg) == [
        f"job bad_status_writer: output path {status_file} for --output "
        "targets protected runtime file status_file"
    ]


def test_validate_config_rejects_inline_job_output_to_protected_runtime_file(tmp_path):
    status_file = tmp_path / "runtime" / "status.json"
    cfg = AutopilotConfig(
        status_file=status_file,
        products=[],
        jobs=[
            JobConfig(
                name="bad_inline_status_writer",
                enabled=True,
                command=[
                    sys.executable,
                    "-m",
                    "src.autopilot.research_smoke",
                    f"--output={status_file}",
                ],
                cadence_seconds=86400,
                working_dir=tmp_path,
            ),
        ],
    )

    assert validate_config(cfg) == [
        f"job bad_inline_status_writer: output path {status_file} for --output "
        "targets protected runtime file status_file"
    ]


def test_validate_config_rejects_job_output_to_product_state_file(tmp_path):
    state_file = tmp_path / "runtime" / "active_income_state.json"
    cfg = AutopilotConfig(
        products=[product(tmp_path, state_file=state_file)],
        jobs=[
            JobConfig(
                name="bad_state_writer",
                enabled=True,
                command=[
                    sys.executable,
                    "-m",
                    "src.autopilot.research_smoke",
                    "--output",
                    str(state_file),
                ],
                cadence_seconds=86400,
                working_dir=tmp_path,
            ),
        ],
    )

    assert validate_config(cfg) == [
        f"job bad_state_writer: output path {state_file} for --output "
        "targets protected runtime file active_income state_file"
    ]


def test_validate_config_rejects_shared_product_runtime_paths(tmp_path):
    cfg = AutopilotConfig(
        products=[
            product(tmp_path, name="active_income", require_testnet_rehearsal=True),
            product(
                tmp_path,
                name="copy_income",
                require_testnet_rehearsal=True,
                state_file=tmp_path / "state.json",
                trade_log=tmp_path / "trades.csv",
                strategies_path=tmp_path / "missing.json",
                preflight_report=tmp_path / "preflight.json",
                testnet_rehearsal_report=tmp_path / "testnet.json",
            ),
        ]
    )

    errors = validate_config(cfg)

    assert (
        "copy_income: strategies_path duplicates active_income: " + str(tmp_path / "missing.json")
        in errors
    )
    assert (
        "copy_income: state_file duplicates active_income: " + str(tmp_path / "state.json")
        in errors
    )
    assert (
        "copy_income: trade_log duplicates active_income: " + str(tmp_path / "trades.csv") in errors
    )
    assert (
        "copy_income: preflight_report duplicates active_income: "
        + str(tmp_path / "preflight.json")
        in errors
    )
    assert (
        "copy_income: testnet_rehearsal_report duplicates active_income: "
        + str(tmp_path / "testnet.json")
        in errors
    )


def test_validate_config_rejects_negative_alert_cooldown():
    cfg = AutopilotConfig(alert_cooldown_seconds=-1)

    assert validate_config(cfg) == ["alert_cooldown_seconds must be non-negative"]


def test_validate_config_rejects_non_positive_loop_sleep():
    cfg = AutopilotConfig(loop_sleep_seconds=0)

    assert validate_config(cfg) == ["loop_sleep_seconds must be positive"]


def test_effective_sleep_seconds_rejects_non_positive_cli_override():
    cfg = AutopilotConfig(loop_sleep_seconds=60)

    assert _effective_sleep_seconds(cfg, None) == 60
    assert _effective_sleep_seconds(cfg, 5) == 5
    with pytest.raises(ValueError, match="sleep seconds must be positive"):
        _effective_sleep_seconds(cfg, 0)
    with pytest.raises(ValueError, match="sleep seconds must be positive"):
        _effective_sleep_seconds(cfg, -1)


def test_validate_config_rejects_non_positive_max_jobs_per_cycle():
    cfg = AutopilotConfig(max_jobs_per_cycle=0)

    assert validate_config(cfg) == ["max_jobs_per_cycle must be positive"]


def test_validate_config_rejects_non_positive_max_consecutive_job_deferrals():
    cfg = AutopilotConfig(max_consecutive_job_deferrals=0)

    assert validate_config(cfg) == ["max_consecutive_job_deferrals must be positive"]


def test_validate_config_rejects_non_positive_min_runtime_free_bytes():
    cfg = AutopilotConfig(min_runtime_free_bytes=0)

    assert validate_config(cfg) == ["min_runtime_free_bytes must be positive"]


def test_validate_config_rejects_bad_preflight_max_age(tmp_path):
    cfg = AutopilotConfig(products=[product(tmp_path, preflight_max_age_seconds=0)])

    assert validate_config(cfg) == ["active_income: preflight_max_age_seconds must be positive"]


def test_validate_config_rejects_bad_testnet_rehearsal_max_age(tmp_path):
    cfg = AutopilotConfig(products=[product(tmp_path, testnet_rehearsal_max_age_seconds=0)])

    assert validate_config(cfg) == [
        "active_income: testnet_rehearsal_max_age_seconds must be positive"
    ]


def test_validate_config_rejects_testnet_rehearsal_gate_for_btc_accumulation(tmp_path):
    cfg = AutopilotConfig(
        products=[
            product(
                tmp_path,
                name="btc_accumulation",
                objective="btc_accumulation",
                base_asset="BTC",
                market="spot",
                require_testnet_rehearsal=True,
            )
        ]
    )

    assert validate_config(cfg) == [
        "btc_accumulation: testnet rehearsal gate is only supported for active_income futures"
    ]


def test_validate_config_rejects_live_active_income_without_testnet_rehearsal_gate(tmp_path):
    cfg = AutopilotConfig(products=[product(tmp_path, execution_mode="live")])

    assert validate_config(cfg) == [
        "active_income: active-income live execution requires require_testnet_rehearsal=true"
    ]


def test_validate_config_rejects_live_product_without_preflight_gate(tmp_path):
    cfg = AutopilotConfig(
        products=[
            product(
                tmp_path,
                execution_mode="live",
                require_preflight=False,
                require_testnet_rehearsal=True,
            )
        ]
    )

    assert validate_config(cfg) == ["active_income: live execution requires require_preflight=true"]


def test_validate_config_rejects_live_product_without_preflight_report_path(tmp_path):
    cfg = AutopilotConfig(
        products=[
            product(
                tmp_path,
                execution_mode="live",
                preflight_report=None,
                require_testnet_rehearsal=True,
            )
        ]
    )

    assert validate_config(cfg) == [
        "active_income: live execution requires a preflight_report path"
    ]


def test_validate_config_rejects_testnet_rehearsal_gate_without_report_path(tmp_path):
    cfg = AutopilotConfig(
        products=[
            product(
                tmp_path,
                require_testnet_rehearsal=True,
                testnet_rehearsal_report=None,
            )
        ]
    )

    assert validate_config(cfg) == [
        "active_income: testnet rehearsal gate requires a testnet_rehearsal_report path"
    ]


def test_runtime_lock_rejects_second_holder_and_releases(tmp_path):
    lock_path = tmp_path / "autopilot.lock"

    with acquire_runtime_lock(lock_path):
        assert "pid=" in lock_path.read_text(encoding="utf-8")
        with pytest.raises(RuntimeError, match="already running"):
            with acquire_runtime_lock(lock_path):
                pass
        with pytest.raises(RuntimeError, match="already running"):
            with acquire_runtime_lock(lock_path):
                pass

    with acquire_runtime_lock(lock_path):
        assert lock_path.exists()


def test_runtime_lock_rejects_symlink_without_touching_target(tmp_path):
    lock_path = tmp_path / "autopilot.lock"
    target = tmp_path / "external.lock"
    target.write_text("external\n", encoding="utf-8")
    lock_path.symlink_to(target)

    with pytest.raises(RuntimeError, match="runtime lock must not be a symlink"):
        with acquire_runtime_lock(lock_path):
            pass

    assert lock_path.is_symlink()
    assert target.read_text(encoding="utf-8") == "external\n"


def test_run_once_waits_for_missing_paper_artifact(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path)],
    )

    report = run_once(cfg)

    assert report["ok"] is True
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["ok"] is True
    assert status["products"][0]["skipped"] is True
    assert status["products"][0]["reason"] == "waiting_for_strategy_artifact"
    assert "Strategy artifact not found" in status["products"][0]["detail"]
    assert "alert" not in status


def test_run_once_blocks_all_active_income_entries_at_portfolio_cap(
    monkeypatch,
    tmp_path,
):
    btc = product(
        tmp_path,
        name="active_income",
        symbol="BTCUSDT",
        strategies_path=tmp_path / "btc.json",
        state_file=tmp_path / "btc_state.json",
        trade_log=tmp_path / "btc_trades.csv",
        preflight_report=tmp_path / "btc_preflight.json",
        testnet_rehearsal_report=tmp_path / "btc_testnet.json",
    )
    eth = product(
        tmp_path,
        name="active_income__ethusdt",
        symbol="ETHUSDT",
        strategies_path=tmp_path / "eth.json",
        state_file=tmp_path / "eth_state.json",
        trade_log=tmp_path / "eth_trades.csv",
        preflight_report=tmp_path / "eth_preflight.json",
        testnet_rehearsal_report=tmp_path / "eth_testnet.json",
    )
    btc.state_file.write_text(
        json.dumps({"open_positions": {"btc": {"direction": "long"}}}),
        encoding="utf-8",
    )
    seen = {}

    def supervise(product_config, **kwargs):
        seen[product_config.name] = kwargs.get("allow_entries", True)
        return {"product": {"name": product_config.name}, "ok": True}

    monkeypatch.setattr("src.autopilot.runtime.run_product_once", supervise)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[btc, eth],
        active_income_max_open_positions=1,
    )

    report = run_once(cfg, run_jobs=False)

    assert report["ok"] is True
    assert seen == {
        "active_income": False,
        "active_income__ethusdt": False,
    }
    assert report["active_income_portfolio"]["open_positions"] == 1
    assert report["active_income_portfolio"]["entry_capacity_available"] is False


def test_run_once_supervises_products_before_scheduled_jobs(monkeypatch, tmp_path):
    calls = []
    configured_product = product(tmp_path)
    configured_job = JobConfig(
        name="slow-research",
        enabled=True,
        command=[sys.executable, "-c", "print('ok')"],
        cadence_seconds=60,
        timeout_seconds=1800,
        working_dir=tmp_path,
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        job_state_file=tmp_path / "job_state.json",
        products=[configured_product],
        jobs=[configured_job],
    )

    def supervise(product_config, *, approval_ledger):
        calls.append(("product", product_config.name))
        return {"product": {"name": product_config.name}, "ok": True}

    def run_jobs(jobs, state_file, **kwargs):
        calls.append(("job", jobs[0].name))
        return [{"name": jobs[0].name, "ok": True}]

    monkeypatch.setattr("src.autopilot.runtime.run_product_once", supervise)
    monkeypatch.setattr("src.autopilot.runtime.run_due_jobs", run_jobs)

    report = run_once(cfg)

    assert report["ok"] is True
    assert calls == [("product", configured_product.name), ("job", configured_job.name)]


def test_run_once_supervision_only_never_enters_scheduled_job_runner(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        job_state_file=tmp_path / "job_state.json",
        products=[],
        jobs=[
            JobConfig(
                name="slow-research",
                enabled=True,
                command=[sys.executable, "-c", "print('should not run')"],
                cadence_seconds=60,
                timeout_seconds=1800,
                working_dir=tmp_path,
            )
        ],
    )
    monkeypatch.setattr(
        "src.autopilot.runtime.run_due_jobs",
        lambda *args, **kwargs: pytest.fail("supervision-only cycle must not run jobs"),
    )

    report = run_once(cfg, run_jobs=False)

    assert report["ok"] is True
    assert report["jobs"] == []
    assert not cfg.job_state_file.exists()


def test_run_once_supervision_only_is_not_blocked_by_invalid_job_definition(
    monkeypatch,
    tmp_path,
):
    configured_product = product(tmp_path)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        job_state_file=tmp_path / "job_state.json",
        products=[configured_product],
        jobs=[
            JobConfig(
                name="broken-optional-research",
                enabled=True,
                command=["sh", "-c", "exit 9"],
                cadence_seconds=-1,
                timeout_seconds=-1,
                working_dir=tmp_path,
            )
        ],
    )
    seen = []

    def supervise(product_config, *, approval_ledger):
        seen.append(product_config.name)
        return {"product": {"name": product_config.name}, "ok": True}

    monkeypatch.setattr("src.autopilot.runtime.run_product_once", supervise)
    monkeypatch.setattr(
        "src.autopilot.runtime.run_due_jobs",
        lambda *args, **kwargs: pytest.fail("supervision-only cycle must not run invalid jobs"),
    )

    report = run_once(cfg, run_jobs=False)

    assert report["ok"] is True
    assert report["jobs"] == []
    assert seen == [configured_product.name]
    assert not cfg.job_state_file.exists()


def test_run_once_supervision_continues_but_fails_heartbeat_for_job_parse_errors(
    monkeypatch,
    tmp_path,
):
    configured_product = product(tmp_path)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        products=[configured_product],
        job_config_errors=["jobs[3]: malformed optional research job"],
    )
    seen = []

    def supervise(product_config, *, approval_ledger):
        seen.append(product_config.name)
        return {"product": {"name": product_config.name}, "ok": True}

    monkeypatch.setattr("src.autopilot.runtime.run_product_once", supervise)

    report = run_once(cfg, run_jobs=False)

    assert seen == [configured_product.name]
    assert report["ok"] is False
    assert report["job_config_errors"] == ["jobs[3]: malformed optional research job"]


def test_run_once_surfaces_symlink_job_state_without_touching_target(tmp_path):
    state_path = tmp_path / "job_state.json"
    target = tmp_path / "external_job_state.json"
    target.write_text('{"version": 1, "jobs": {"external": {}}}\n', encoding="utf-8")
    state_path.symlink_to(target)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        job_state_file=state_path,
        products=[],
        jobs=[
            JobConfig(
                name="smoke",
                enabled=True,
                command=[sys.executable, "-c", "print('ok')"],
                cadence_seconds=60,
                timeout_seconds=5,
                working_dir=tmp_path,
            )
        ],
    )

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is False
    assert report["jobs"] == [
        {
            "name": "scheduler",
            "ok": False,
            "error": f"job state must not be a symlink: {state_path}",
            "state_file": str(state_path),
        }
    ]
    assert status["jobs"] == report["jobs"]
    assert state_path.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"version": 1, "jobs": {"external": {}}}\n'


def test_run_once_auto_reports_without_blocking_cycle(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        auto_report_enabled=True,
        products=[],
    )

    def fail_reports(config):
        raise RuntimeError(f"cannot write reports for {config.status_file.name}")

    monkeypatch.setattr("src.autopilot.runtime.write_cycle_reports", fail_reports)

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is True
    assert report["reporting"]["ok"] is False
    assert "cannot write reports for status.json" in report["reporting"]["error"]
    assert status["ok"] is True
    assert status["reporting"]["ok"] is False


def test_write_cycle_reports_records_partial_output_failure(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        operator_report_file=tmp_path / "operator.md",
        operator_report_json_file=tmp_path / "operator.json",
        readiness_report_file=tmp_path / "readiness.md",
        readiness_report_json_file=tmp_path / "readiness.json",
        products=[],
    )

    monkeypatch.setattr("src.autopilot.runtime.build_operator_report", lambda config: {"ok": True})
    monkeypatch.setattr(
        "src.autopilot.runtime.render_operator_markdown", lambda report: "# Operator\n"
    )
    monkeypatch.setattr(
        "src.autopilot.readiness.build_readiness_report", lambda config: {"ok": True, "checks": []}
    )
    monkeypatch.setattr(
        "src.autopilot.readiness.render_readiness_markdown", lambda report: "# Readiness\n"
    )

    def write_json(path, payload):
        if path == cfg.operator_report_json_file:
            raise OSError("operator json disk full")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr("src.autopilot.runtime.write_json_atomic", write_json)

    report = write_cycle_reports(cfg)

    assert report["ok"] is False
    assert report["outputs"]["operator_report"]["written"] is True
    assert report["outputs"]["operator_report_json"]["written"] is False
    assert report["outputs"]["readiness_report"]["written"] is True
    assert report["outputs"]["readiness_report_json"]["written"] is True
    assert report["errors"] == [
        {
            "stage": "operator_report_json_write_failed",
            "error": "OSError: operator json disk full",
            "path": str(cfg.operator_report_json_file),
        }
    ]
    assert cfg.operator_report_file.read_text(encoding="utf-8") == "# Operator\n"
    assert json.loads(cfg.readiness_report_json_file.read_text(encoding="utf-8")) == {
        "ok": True,
        "checks": [],
    }


def test_run_once_does_not_alert_from_stale_readiness_json_after_refresh_failure(
    monkeypatch, tmp_path
):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        readiness_report_json_file=tmp_path / "readiness_report.json",
        auto_report_enabled=True,
        alerts_enabled=True,
        products=[],
    )
    cfg.readiness_report_json_file.write_text(
        json.dumps(
            {
                "ok": True,
                "checks": [
                    {
                        "name": "market data seed and freshness",
                        "level": "warning",
                        "ok": False,
                        "detail": {"spot": {"ok": False, "reason": "stale_old_report"}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def write_reports(config):
        return {
            "ok": False,
            "outputs": {
                "readiness_report_json": {
                    "path": str(config.readiness_report_json_file),
                    "written": False,
                }
            },
            "errors": [
                {"stage": "readiness_report_json_write_failed", "error": "OSError: disk full"}
            ],
        }

    monkeypatch.setattr("src.autopilot.runtime.write_cycle_reports", write_reports)

    report = run_once(cfg)

    assert report["ok"] is True
    assert "readiness_alert" not in report
    assert not cfg.alert_file.exists()


def test_run_once_emits_readiness_warning_alert_after_reports(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        auto_report_enabled=True,
        alert_cooldown_seconds=60,
        products=[],
    )
    calls = {"write_reports": 0}

    def write_reports(config):
        calls["write_reports"] += 1
        config.readiness_report_json_file.parent.mkdir(parents=True, exist_ok=True)
        config.readiness_report_json_file.write_text(
            json.dumps(
                {
                    "ok": True,
                    "checks": [
                        {
                            "name": "market data seed and freshness",
                            "level": "warning",
                            "ok": False,
                            "detail": {
                                "spot": {
                                    "ok": False,
                                    "reason": "missing_seed_dataset",
                                    "path": "spot.parquet",
                                }
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"ok": True}

    monkeypatch.setattr("src.autopilot.runtime.write_cycle_reports", write_reports)

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))
    alert_lines = cfg.alert_file.read_text(encoding="utf-8").splitlines()

    assert report["ok"] is True
    assert report["readiness_alert"]["sent"] is True
    assert status["readiness_alert"]["sent"] is True
    assert calls["write_reports"] == 2
    alert = json.loads(alert_lines[0])
    assert alert["severity"] == "warning"
    assert alert["title"] == "autopilot readiness warnings"
    assert alert["detail"]["warnings"][0]["markets"]["spot"]["reason"] == "missing_seed_dataset"


def test_run_once_records_readiness_alert_failure_without_failing_cycle(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        auto_report_enabled=True,
        products=[],
    )
    calls = {"write_reports": 0}

    def write_reports(config):
        calls["write_reports"] += 1
        config.readiness_report_json_file.parent.mkdir(parents=True, exist_ok=True)
        config.readiness_report_json_file.write_text(
            json.dumps(
                {
                    "ok": True,
                    "checks": [
                        {
                            "name": "market data seed and freshness",
                            "level": "warning",
                            "ok": False,
                            "detail": {
                                "spot": {
                                    "ok": False,
                                    "reason": "missing_seed_dataset",
                                    "path": "spot.parquet",
                                }
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"ok": True}

    def fail_alert(**_kwargs):
        raise OSError("alert disk unavailable")

    monkeypatch.setattr("src.autopilot.runtime.write_cycle_reports", write_reports)
    monkeypatch.setattr("src.autopilot.runtime.emit_alert", fail_alert)

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is True
    assert report["readiness_alert"] == {"sent": False, "error": "alert disk unavailable"}
    assert status["readiness_alert"] == report["readiness_alert"]
    assert report["reporting"]["ok"] is True
    assert calls["write_reports"] == 2


def test_run_once_emits_promotion_warning_alert_after_reports(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        operator_report_json_file=tmp_path / "operator_report.json",
        readiness_report_json_file=tmp_path / "readiness_report.json",
        auto_report_enabled=True,
        alert_cooldown_seconds=60,
        products=[],
    )
    calls = {"write_reports": 0}

    def write_reports(config):
        calls["write_reports"] += 1
        config.operator_report_json_file.write_text(
            json.dumps(
                {
                    "promotion_reviews": [
                        {
                            "product": "active_income",
                            "status": "ready",
                            "path": "runtime/active_income_promotion_review.json",
                            "generated_at": "2026-01-01T00:00:00+00:00",
                            "recommendations": {"approved_review_failed": 1},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        config.readiness_report_json_file.write_text(
            json.dumps({"ok": True, "checks": []}),
            encoding="utf-8",
        )
        return {"ok": True}

    monkeypatch.setattr("src.autopilot.runtime.write_cycle_reports", write_reports)

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))
    alert_lines = cfg.alert_file.read_text(encoding="utf-8").splitlines()

    assert report["ok"] is True
    assert report["promotion_alert"]["sent"] is True
    assert status["promotion_alert"]["sent"] is True
    assert calls["write_reports"] == 2
    alert = json.loads(alert_lines[0])
    assert alert["severity"] == "warning"
    assert alert["title"] == "autopilot promotion review warnings"
    assert alert["detail"]["warnings"][0]["name"] == "approved_review_failed"
    assert alert["detail"]["warnings"][0]["product"] == "active_income"
    assert alert["detail"]["warnings"][0]["approved_review_failed"] == 1


def test_run_once_emits_research_handoff_warning_alert_after_reports(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        operator_report_json_file=tmp_path / "operator_report.json",
        readiness_report_json_file=tmp_path / "readiness_report.json",
        auto_report_enabled=True,
        alert_cooldown_seconds=60,
        products=[],
    )
    calls = {"write_reports": 0}

    def write_reports(config):
        calls["write_reports"] += 1
        config.operator_report_json_file.write_text(
            json.dumps(
                {
                    "research_cycle": {"generated_at": "2026-01-01T01:05:00+00:00"},
                    "mutation_plan": {
                        "generated_at": "2026-01-01T01:06:00+00:00",
                        "source": {"research_generated_at": "2026-01-01T00:55:00+00:00"},
                    },
                    "mutation_batch": {
                        "generated_at": "2026-01-01T01:07:00+00:00",
                        "source": {"plan_generated_at": "2026-01-01T01:06:00+00:00"},
                    },
                    "promotion_reviews": [],
                }
            ),
            encoding="utf-8",
        )
        config.readiness_report_json_file.write_text(
            json.dumps({"ok": True, "checks": []}),
            encoding="utf-8",
        )
        return {"ok": True}

    monkeypatch.setattr("src.autopilot.runtime.write_cycle_reports", write_reports)

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))
    alert_lines = cfg.alert_file.read_text(encoding="utf-8").splitlines()

    assert report["ok"] is True
    assert report["research_handoff_alert"]["sent"] is True
    assert status["research_handoff_alert"]["sent"] is True
    assert calls["write_reports"] == 2
    alert = json.loads(alert_lines[0])
    assert alert["severity"] == "warning"
    assert alert["title"] == "autopilot research handoff warnings"
    assert alert["detail"]["warnings"] == [
        {
            "name": "mutation_plan_stale_source",
            "research_generated_at": "2026-01-01T01:05:00+00:00",
            "mutation_plan_source_research_generated_at": "2026-01-01T00:55:00+00:00",
            "mutation_plan_generated_at": "2026-01-01T01:06:00+00:00",
        }
    ]


def test_run_once_emits_research_progress_warning_alert_after_reports(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        operator_report_json_file=tmp_path / "operator_report.json",
        readiness_report_json_file=tmp_path / "readiness_report.json",
        auto_report_enabled=True,
        alert_cooldown_seconds=60,
        products=[],
    )
    calls = {"write_reports": 0}

    def write_reports(config):
        calls["write_reports"] += 1
        config.operator_report_json_file.write_text(
            json.dumps(
                {
                    "research_cycle": {
                        "ok": True,
                        "generated_at": "2026-01-01T01:05:00+00:00",
                        "summary": {
                            "hypotheses": 12,
                            "keepers": 0,
                            "exported": 0,
                            "top_reasons": {"no_train_edge": 8},
                            "next_actions": ["continue bounded search"],
                        },
                    },
                    "products": [
                        {
                            "name": "active_income",
                            "enabled": True,
                            "objective": "active_income",
                            "market": "futures",
                            "mode": "paper",
                            "reason": "waiting_for_strategy_artifact",
                        }
                    ],
                    "promotion_reviews": [],
                }
            ),
            encoding="utf-8",
        )
        config.readiness_report_json_file.write_text(
            json.dumps({"ok": True, "checks": []}),
            encoding="utf-8",
        )
        return {"ok": True}

    monkeypatch.setattr("src.autopilot.runtime.write_cycle_reports", write_reports)

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))
    alert_lines = cfg.alert_file.read_text(encoding="utf-8").splitlines()

    assert report["ok"] is True
    assert report["research_progress_alert"]["sent"] is True
    assert status["research_progress_alert"]["sent"] is True
    assert calls["write_reports"] == 2
    alert = json.loads(alert_lines[0])
    assert alert["severity"] == "warning"
    assert alert["title"] == "autopilot research progress warnings"
    assert alert["detail"]["warnings"][0]["name"] == "research_cycle_no_exportable_strategies"
    assert alert["detail"]["warnings"][0]["waiting_products"] == [
        {"name": "active_income", "objective": "active_income", "market": "futures"}
    ]


def test_run_once_emits_testnet_rehearsal_warning_alert_after_reports(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        operator_report_json_file=tmp_path / "operator_report.json",
        readiness_report_json_file=tmp_path / "readiness_report.json",
        auto_report_enabled=True,
        alerts_enabled=True,
        products=[],
    )
    calls = {"write_reports": 0}

    def write_reports(config):
        calls["write_reports"] += 1
        config.operator_report_json_file.write_text(
            json.dumps(
                {
                    "ok": True,
                    "status_heartbeat": {"fresh": True},
                    "testnet_rehearsal": {
                        "required": True,
                        "required_by": ["active_income"],
                        "status": "missing",
                        "path": "runtime/testnet_rehearsal_report.json",
                        "ok": False,
                        "product": "active_income",
                    },
                    "products": [
                        {
                            "name": "active_income",
                            "enabled": True,
                            "market": "futures",
                            "mode": "paper",
                            "require_testnet_rehearsal": True,
                        }
                    ],
                    "promotion_reviews": [],
                }
            ),
            encoding="utf-8",
        )
        config.readiness_report_json_file.write_text(
            json.dumps({"ok": True, "checks": []}),
            encoding="utf-8",
        )
        return {"ok": True}

    monkeypatch.setattr("src.autopilot.runtime.write_cycle_reports", write_reports)

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))
    alert_lines = cfg.alert_file.read_text(encoding="utf-8").splitlines()

    assert report["ok"] is True
    assert report["testnet_rehearsal_alert"]["sent"] is True
    assert status["testnet_rehearsal_alert"]["sent"] is True
    assert calls["write_reports"] == 2
    alert = json.loads(alert_lines[0])
    assert alert["severity"] == "warning"
    assert alert["title"] == "autopilot testnet rehearsal warnings"
    assert alert["detail"]["warnings"] == [
        {
            "name": "required_testnet_rehearsal_not_ready",
            "status": "missing",
            "path": "runtime/testnet_rehearsal_report.json",
            "required_by": ["active_income"],
            "product": "active_income",
        }
    ]


def test_run_once_still_fails_for_missing_live_artifact(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path, execution_mode="live", require_testnet_rehearsal=True)],
    )

    report = run_once(cfg)

    assert report["ok"] is False
    assert "not found" in report["products"][0]["error"]
    assert report["alert"]["sent"] is True


def test_run_once_skips_policy_blocked_paper_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    strategy["metrics"]["holdout_total_return"] = -0.01
    artifact.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": [strategy],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path, strategies_path=artifact)],
    )

    class ShouldNotRunBot:
        def __init__(self, **kwargs):
            raise AssertionError("policy-blocked paper artifact should not construct a bot")

    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", ShouldNotRunBot)

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is True
    assert status["products"][0]["ok"] is True
    assert status["products"][0]["skipped"] is True
    assert status["products"][0]["reason"] == "strategy_policy_blocked"
    assert "holdout_total_return -0.010000 must be positive" in status["products"][0]["detail"]
    assert "alert" not in report


def test_run_once_skips_invalid_json_paper_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    artifact.write_text('{"version": 1,', encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path, strategies_path=artifact)],
    )

    class ShouldNotRunBot:
        def __init__(self, **kwargs):
            raise AssertionError("invalid artifact JSON should not construct a bot")

    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", ShouldNotRunBot)

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is True
    assert status["products"][0]["ok"] is True
    assert status["products"][0]["skipped"] is True
    assert status["products"][0]["reason"] == "strategy_policy_blocked"
    assert "must be valid JSON" in status["products"][0]["detail"]
    assert "alert" not in report


def test_run_once_still_fails_for_policy_blocked_live_artifact(tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    strategy["metrics"]["holdout_total_return"] = -0.01
    artifact.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": [strategy],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[
            product(
                tmp_path,
                strategies_path=artifact,
                execution_mode="live",
                require_testnet_rehearsal=True,
            )
        ],
    )

    report = run_once(cfg)

    assert report["ok"] is False
    assert "holdout_total_return -0.010000 must be positive" in report["products"][0]["error"]
    assert report["alert"]["sent"] is True


def test_run_once_fails_for_invalid_json_live_artifact(tmp_path):
    artifact = tmp_path / "active.json"
    artifact.write_text('{"version": 1,', encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[
            product(
                tmp_path,
                strategies_path=artifact,
                execution_mode="live",
                require_testnet_rehearsal=True,
            )
        ],
    )

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is False
    assert "must be valid JSON" in report["products"][0]["error"]
    assert "must be valid JSON" in status["products"][0]["error"]
    assert report["alert"]["sent"] is True


def test_run_product_once_surfaces_bot_cycle_errors(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    paper_product = product(tmp_path, strategies_path=artifact)

    class FakeBot:
        def __init__(self, **kwargs):
            self.state = {"equity": 1000.0, "open_positions": {}, "inactive_strategies": []}
            self.cycle_errors = []

        def run_cycle(self):
            self.cycle_errors.append(
                {"strategy_id": "live_r1", "stage": "feature_build", "error": "network down"}
            )

    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", FakeBot)

    status = run_product_once(paper_product, approval_ledger=tmp_path / "approvals.json")

    assert status["ok"] is False
    assert status["cycle_errors"] == [
        {"strategy_id": "live_r1", "stage": "feature_build", "error": "network down"}
    ]


def test_bot_status_snapshot_compacts_durable_recovery_and_accounting_state():
    bot = SimpleNamespace(
        state={
            "equity": 997.5,
            "open_positions": {},
            "inactive_strategies": [],
            "pending_order": {
                "version": 1,
                "strategy_id": "live_r1",
                "stage": "entry",
                "symbol": "BTCUSDT",
                "side": "buy",
                "qty": 0.1,
                "order_type": "market",
                "reduce_only": False,
                "client_id": "tb-en-1",
                "created_ts": 123.0,
                "intent_ref": "excluded verbose intent",
            },
            "pending_entry_recovery": {
                "version": 1,
                "strategy_id": "live_r1",
                "symbol": "BTCUSDT",
                "status": "recovery_close_failed_position_remains",
                "recovery_client_id": "tb-rc-1",
                "attempt_count": 2,
                "last_error": "excluded arbitrary broker response",
            },
            "risk_recovery_incident": {
                "version": 1,
                "strategy_id": "live_r1",
                "symbol": "BTCUSDT",
                "cause": "broker_position_quantity_mismatch",
                "status": "recovery_close_filled_and_flat",
                "recovery_client_id": "tb-rc-2",
                "attempt_count": 1,
                "fill": {"fee": 0.1},
            },
            "flatten_intent": {
                "version": 1,
                "strategy_id": "spot_r1",
                "symbol": "BTCUSDT",
                "side": "buy",
                "order_type": "market",
                "client_id": "tb-sf-1",
                "qty": 0.01,
                "quote_budget": 1.0,
                "created_ts": 124.0,
                "position_before": {"symbol": "BTCUSDT", "qty": 0.0},
            },
            "exit_accounting_intent": {
                "version": 1,
                "phase": "ready_to_commit",
                "exit_event_id": "a" * 64,
                "strategy_id": "live_r1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "broker_flat_proven": True,
                "trade_data": {"excluded": True},
            },
        },
        strategies=[],
    )

    snapshot = _bot_status_snapshot(bot)

    assert snapshot["pending_order"]["client_id"] == "tb-en-1"
    assert "intent_ref" not in snapshot["pending_order"]
    assert snapshot["pending_entry_recovery"] == {
        "version": 1,
        "strategy_id": "live_r1",
        "symbol": "BTCUSDT",
        "status": "recovery_close_failed_position_remains",
        "recovery_client_id": "tb-rc-1",
        "attempt_count": 2,
    }
    assert "fill" not in snapshot["risk_recovery_incident"]
    assert "position_before" not in snapshot["flatten_intent"]
    assert snapshot["exit_accounting_intent"] == {
        "version": 1,
        "phase": "ready_to_commit",
        "exit_event_id": "a" * 64,
        "strategy_id": "live_r1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "broker_flat_proven": True,
    }


@pytest.mark.parametrize(
    "state_key",
    [
        "pending_order",
        "pending_entry_recovery",
        "risk_recovery_incident",
        "flatten_intent",
        "exit_accounting_intent",
    ],
)
def test_durable_recovery_state_requires_management_cycle(tmp_path, state_key):
    configured_product = product(tmp_path)
    configured_product.state_file.write_text(
        json.dumps({"open_positions": {}, state_key: {"version": 1}}),
        encoding="utf-8",
    )

    assert _local_state_requires_management(configured_product) is True


def test_run_product_once_reports_structured_bot_cycle_exception(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    paper_product = product(tmp_path, strategies_path=artifact)

    class FakeBot:
        def __init__(self, **kwargs):
            self.state = {
                "equity": 997.5,
                "open_positions": {
                    "live_r1": {
                        "direction": "long",
                        "entry_time": "2026-01-01T00:00:00+00:00",
                        "position_size": 0.1,
                        "entry_price": 100.0,
                        "sl_price": 98.0,
                        "tp_price": 104.0,
                        "sl_pct": 0.02,
                        "tp_pct": 0.04,
                        "broker_symbol": "BTCUSDT",
                        "broker_qty": 0.1,
                        "broker_side": "buy",
                        "broker_entry_price": 100.0,
                        "broker_entry_fee": 0.01,
                        "broker_entry_balance": 997.5,
                        "broker_requested_qty": 0.1,
                        "broker_fill_ratio": 1.0,
                        "broker_stop_order_id": "stop-123",
                        "broker_stop_client_id": "tb-sl-stop-123",
                        "broker_stop_trigger_price": 98.0,
                    }
                },
                "inactive_strategies": ["stale_r2"],
            }
            self.strategies = [{"id": "live_r1", "base_timeframe": "5m", "horizon_bars": 6}]
            self.cycle_errors = []

        def run_cycle(self):
            raise RuntimeError("broker exit rejected")

    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", FakeBot)

    status = run_product_once(paper_product, approval_ledger=tmp_path / "approvals.json")

    assert status["ok"] is False
    assert status["error"] == "broker exit rejected"
    assert status["equity"] == 997.5
    assert status["open_positions"] == 1
    assert status["open_position_details"] == [
        {
            "strategy_id": "live_r1",
            "direction": "long",
            "entry_time": "2026-01-01T00:00:00+00:00",
            "position_size": 0.1,
            "entry_price": 100.0,
            "sl_price": 98.0,
            "tp_price": 104.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
            "base_timeframe": "5m",
            "horizon_bars": 6,
            "stale_after_seconds": 5400.0,
            "broker_symbol": "BTCUSDT",
            "broker_qty": 0.1,
            "broker_side": "buy",
            "broker_entry_price": 100.0,
            "broker_entry_fee": 0.01,
            "broker_entry_balance": 997.5,
            "broker_requested_qty": 0.1,
            "broker_fill_ratio": 1.0,
            "broker_stop_order_id": "stop-123",
            "broker_stop_client_id": "tb-sl-stop-123",
            "broker_stop_trigger_price": 98.0,
        }
    ]
    assert status["inactive_strategies"] == ["stale_r2"]
    assert status["cycle_errors"] == [
        {"stage": "run_cycle", "error": "broker exit rejected", "type": "RuntimeError"}
    ]


def test_run_product_once_fails_when_bot_state_open_positions_is_not_object(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    paper_product = product(tmp_path, strategies_path=artifact)

    class FakeBot:
        def __init__(self, **kwargs):
            self.state = {"equity": 1000.0, "open_positions": [], "inactive_strategies": []}
            self.strategies = []
            self.cycle_errors = []

        def run_cycle(self):
            return None

    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", FakeBot)

    status = run_product_once(paper_product, approval_ledger=tmp_path / "approvals.json")

    assert status["ok"] is False
    assert status["cycle_errors"] == []
    assert status["open_positions"] == "invalid"
    assert status["open_position_details"] == []
    assert status["state_errors"] == [
        {"field": "open_positions", "error": "expected object, got list"}
    ]


def test_run_once_alerts_on_bot_cycle_errors(monkeypatch, tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path, strategies_path=artifact)],
    )

    class FakeBot:
        def __init__(self, **kwargs):
            self.state = {"equity": 1000.0, "open_positions": {}, "inactive_strategies": []}
            self.cycle_errors = []

        def run_cycle(self):
            self.cycle_errors.append(
                {"strategy_id": "live_r1", "stage": "feature_build", "error": "network down"}
            )

    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", FakeBot)

    report = run_once(cfg)

    assert report["ok"] is False
    assert report["products"][0]["cycle_errors"][0]["error"] == "network down"
    assert report["alert"]["sent"] is True


def test_run_once_reports_corrupt_product_state_file(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    state_file = tmp_path / "state.json"
    state_file.write_text("{", encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path, strategies_path=artifact, state_file=state_file)],
    )

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is False
    assert "State file is unreadable or invalid" in report["products"][0]["error"]
    assert status["products"][0]["ok"] is False
    assert report["alert"]["sent"] is True


def test_run_once_fails_closed_when_job_state_shape_is_invalid(tmp_path):
    job_state = tmp_path / "job_state.json"
    job_state.write_text("[]", encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        job_state_file=job_state,
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        jobs=[
            JobConfig(
                name="smoke",
                enabled=True,
                command=[sys.executable, "-c", "print('ok')"],
                cadence_seconds=60,
            )
        ],
        products=[],
    )

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is False
    assert report["jobs"] == [
        {
            "name": "scheduler",
            "ok": False,
            "error": f"job state must be a JSON object: {job_state}",
            "state_file": str(job_state),
        }
    ]
    assert status["jobs"] == report["jobs"]
    assert report["alert"]["sent"] is True


def test_run_once_respects_pause_control(tmp_path):
    (tmp_path / "control.json").write_text(json.dumps({"paused": True}), encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path)],
    )

    report = run_once(cfg)

    assert report["ok"] is True
    assert report["products"][0]["skipped"] is True
    assert report["products"][0]["reason"] == "paused"


def test_run_once_paused_product_still_manages_existing_exposure(monkeypatch, tmp_path):
    (tmp_path / "control.json").write_text(json.dumps({"paused": True}), encoding="utf-8")
    configured_product = product(tmp_path)
    configured_product.state_file.write_text(
        json.dumps({"open_positions": {"strategy": {"direction": "long"}}}),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[configured_product],
    )
    seen = {}

    def manage(product_config, *, approval_ledger, allow_entries=True):
        seen.update(product=product_config.name, allow_entries=allow_entries)
        return {
            "product": {"name": product_config.name},
            "ok": True,
            "entries_allowed": allow_entries,
        }

    monkeypatch.setattr("src.autopilot.runtime.run_product_once", manage)

    report = run_once(cfg)

    assert seen == {"product": "active_income", "allow_entries": False}
    assert report["products"][0]["paused"] is True
    assert report["products"][0]["entries_allowed"] is False


def test_run_once_malformed_control_fails_closed_and_writes_status(tmp_path):
    (tmp_path / "control.json").write_text("{", encoding="utf-8")
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path)],
        jobs=[
            JobConfig(
                name="network_job",
                enabled=True,
                command=[sys.executable, "-c", "raise SystemExit(9)"],
                cadence_seconds=60,
                timeout_seconds=5,
                working_dir=tmp_path,
            )
        ],
    )

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is False
    assert report["control"]["paused"] is True
    assert report["control"]["pause_jobs"] is True
    assert "JSONDecodeError" in report["control_error"]
    assert report["jobs"] == []
    assert report["products"][0]["reason"] == "paused"
    assert report["alert"]["sent"] is True
    assert status["control"]["reason"] == "invalid_control_file"
    alert = json.loads(cfg.alert_file.read_text(encoding="utf-8").splitlines()[0])
    assert "JSONDecodeError" in alert["detail"]["control"]["error"]
    assert alert["detail"]["control"]["reason"] == "invalid_control_file"
    assert alert["detail"]["control"]["paused"] is True
    assert alert["detail"]["control"]["pause_jobs"] is True


def test_run_once_unknown_control_selectors_fail_closed(tmp_path):
    (tmp_path / "control.json").write_text(
        json.dumps(
            {
                "paused_products": ["active-incme"],
                "flatten_products": ["active-incme"],
                "paused_jobs": ["network-jb"],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        job_state_file=tmp_path / "job_state.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        products=[product(tmp_path)],
        jobs=[
            JobConfig(
                name="network_job",
                enabled=True,
                command=[sys.executable, "-c", "raise SystemExit(9)"],
                cadence_seconds=60,
                timeout_seconds=5,
                working_dir=tmp_path,
            )
        ],
    )

    report = run_once(cfg)
    status = json.loads(cfg.status_file.read_text(encoding="utf-8"))

    assert report["ok"] is False
    assert report["control"]["paused"] is True
    assert report["control"]["pause_jobs"] is True
    assert report["control"]["reason"] == "unknown_control_selector"
    assert report["unknown_control_selectors"] == {
        "flatten_products": ["active-incme"],
        "paused_jobs": ["network-jb"],
        "paused_products": ["active-incme"],
    }
    assert "unknown control selectors" in report["control_error"]
    assert report["jobs"] == []
    assert report["products"][0]["reason"] == "paused"
    assert status["control"]["control_error"] == report["control_error"]
    assert not (tmp_path / "job_state.json").exists()
    alert = json.loads(cfg.alert_file.read_text(encoding="utf-8").splitlines()[0])
    assert alert["detail"]["control"] == {
        "error": report["control_error"],
        "reason": "unknown_control_selector",
        "paused": True,
        "pause_jobs": True,
        "unknown_selectors": {
            "flatten_products": ["active-incme"],
            "paused_jobs": ["network-jb"],
            "paused_products": ["active-incme"],
        },
    }


def test_run_once_pauses_selected_job_without_pausing_products(tmp_path):
    (tmp_path / "control.json").write_text(
        json.dumps({"paused_jobs": ["network_job"], "reason": "maintenance"}),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        job_state_file=tmp_path / "job_state.json",
        products=[product(tmp_path)],
        jobs=[
            JobConfig(
                name="network_job",
                enabled=True,
                command=[sys.executable, "-c", "raise SystemExit(9)"],
                cadence_seconds=60,
                timeout_seconds=5,
                working_dir=tmp_path,
            )
        ],
    )

    report = run_once(cfg)

    assert report["ok"] is True
    assert report["jobs"][0]["skipped"] is True
    assert report["jobs"][0]["reason"] == "paused"
    assert report["products"][0]["reason"] == "waiting_for_strategy_artifact"
    assert not (tmp_path / "job_state.json").exists()


def test_flatten_live_futures_rejects_testnet_before_broker_or_state_change(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    monkeypatch.setenv("EXCHANGE_TESTNET", "1")
    state_file = tmp_path / "state.json"
    original = {
        "open_positions": {
            "s1": {
                "direction": "long",
                "broker_symbol": "BTCUSDT",
                "broker_qty": 0.5,
            }
        },
        "pending_order": {"client_id": "tb-ex-1"},
        "equity": 1000.0,
    }
    state_file.write_text(json.dumps(original), encoding="utf-8")
    live_product = product(
        tmp_path,
        execution_mode="live",
        state_file=state_file,
    )
    broker_calls = []
    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker",
        lambda configured_product: broker_calls.append(configured_product),
    )

    with pytest.raises(RuntimeError, match="EXCHANGE_TESTNET must be 0"):
        flatten_product_once(live_product)

    assert broker_calls == []
    assert json.loads(state_file.read_text(encoding="utf-8")) == original


def test_flatten_live_spot_rejects_testnet_before_broker_or_state_change(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    monkeypatch.setenv("EXCHANGE_TESTNET", "1")
    state_file = tmp_path / "state.json"
    original = {
        "equity": 1.0,
        "open_positions": {
            "btc_step_aside": {
                "direction": "short",
                "broker_symbol": "BTCUSDT",
                "broker_qty": 0.5,
                "broker_side": "sell",
                "broker_entry_price": 100.0,
                "broker_entry_fee": 0.05,
                "broker_entry_quote_value": 50.0,
                "broker_requested_qty": 0.5,
                "broker_fill_ratio": 1.0,
                "broker_exit_sizing": "quote_reinvest",
                "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
            }
        },
    }
    state_file.write_text(json.dumps(original), encoding="utf-8")
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
    )
    broker_calls = []
    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker",
        lambda configured_product: broker_calls.append(configured_product),
    )

    with pytest.raises(RuntimeError, match="EXCHANGE_TESTNET must be 0"):
        flatten_product_once(live_product)

    assert broker_calls == []
    assert json.loads(state_file.read_text(encoding="utf-8")) == original


def test_flatten_live_futures_rejects_wrong_account_before_any_broker_read(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    original = {
        "open_positions": {
            "s1": {
                "direction": "long",
                "broker_symbol": "BTCUSDT",
                "broker_qty": 0.5,
                "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
            }
        },
        "equity": 1000.0,
    }
    state_file.write_text(json.dumps(original), encoding="utf-8")
    live_product = product(
        tmp_path,
        execution_mode="live",
        state_file=state_file,
    )

    class WrongAccountBroker:
        name = "wrong-account"
        account_fingerprint = OTHER_ACCOUNT_FINGERPRINT

        def get_position(self, symbol):
            raise AssertionError("account mismatch must block before broker position read")

        def close_position(self, symbol):
            raise AssertionError("account mismatch must block before broker close")

    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker",
        lambda configured_product: WrongAccountBroker(),
    )

    with pytest.raises(RuntimeError, match="different broker account"):
        flatten_product_once(live_product)

    assert json.loads(state_file.read_text(encoding="utf-8")) == original


def test_flatten_live_spot_rejects_wrong_account_before_any_broker_read(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    original = {
        "open_positions": {
            "btc_step_aside": {
                "direction": "short",
                "broker_symbol": "BTCUSDT",
                "broker_qty": 0.5,
                "broker_side": "sell",
                "broker_entry_price": 100.0,
                "broker_entry_quote_value": 50.0,
                "broker_exit_sizing": "quote_reinvest",
                "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
            }
        },
        "equity": 1.0,
    }
    state_file.write_text(json.dumps(original), encoding="utf-8")
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
    )

    class WrongAccountBroker:
        name = "wrong-account"
        account_fingerprint = OTHER_ACCOUNT_FINGERPRINT

        def get_position(self, symbol):
            raise AssertionError("account mismatch must block before broker balance read")

    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker",
        lambda configured_product: WrongAccountBroker(),
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "broker_account_mismatch"
    assert json.loads(state_file.read_text(encoding="utf-8")) == original


def test_flatten_live_futures_product_closes_broker_and_clears_state(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "open_positions": {
                    "s1": {
                        "direction": "long",
                        "broker_symbol": "BTCUSDT",
                        "broker_qty": 0.5,
                        "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                        "broker_stop_order_id": "stop-1",
                        "broker_stop_client_id": "tb-sl-stop-1",
                        "broker_stop_trigger_price": 95.0,
                    }
                },
                "pending_order": {
                    "client_id": "tb-ex-recovery",
                    "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                },
                "equity": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    live_product = product(
        tmp_path,
        execution_mode="live",
        state_file=state_file,
        strategies_path=tmp_path / "unused.json",
    )

    class FakeBroker:
        name = "fake-live"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def __init__(self):
            self.position = Position(symbol="BTCUSDT", qty=0.5, avg_price=100.0)
            self.closed = False
            self.stop_status = ProtectiveOrderStatus.OPEN

        def supports_native_protective_stops(self):
            return True

        def _stop(self):
            return ProtectiveOrder(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                qty=0.5,
                trigger_price=95.0,
                status=self.stop_status,
                order_id="stop-1",
                client_id="tb-sl-stop-1",
            )

        def get_protective_stop(self, **kwargs):
            return self._stop()

        def cancel_protective_stop(self, **kwargs):
            self.stop_status = ProtectiveOrderStatus.CANCELED
            return self._stop()

        def get_position(self, symbol):
            return self.position

        def close_position(self, symbol):
            self.closed = True
            self.position = Position(symbol=symbol, qty=0.0, avg_price=0.0)
            return Fill(symbol=symbol, side=OrderSide.SELL, qty=0.5, price=101.0, fee=0.1)

    broker = FakeBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["action"] == "flatten"
    assert status["reason"] == "flatten_accounting_precondition_failed"
    assert "frozen strategy snapshot" in status["error"]
    assert broker.closed is False
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["open_positions"]
    assert state["pending_order"]["client_id"] == "tb-ex-recovery"
    assert "last_flatten" not in state


def test_flatten_live_futures_keeps_state_when_native_stop_remains_open(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    original = {
        "open_positions": {
            "s1": {
                "direction": "long",
                "broker_symbol": "BTCUSDT",
                "broker_qty": 0.5,
                "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                "broker_stop_order_id": "stop-1",
                "broker_stop_client_id": "tb-sl-stop-1",
                "broker_stop_trigger_price": 95.0,
            }
        }
    }
    state_file.write_text(json.dumps(original), encoding="utf-8")
    live_product = product(tmp_path, execution_mode="live", state_file=state_file)

    class StuckStopBroker:
        name = "stuck-stop"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def get_position(self, symbol):
            return Position(symbol=symbol)

        def supports_native_protective_stops(self):
            return True

        def _stop(self):
            return ProtectiveOrder(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                qty=0.5,
                trigger_price=95.0,
                status=ProtectiveOrderStatus.OPEN,
                order_id="stop-1",
                client_id="tb-sl-stop-1",
            )

        def get_protective_stop(self, **kwargs):
            return self._stop()

        def cancel_protective_stop(self, **kwargs):
            return self._stop()

    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker", lambda product: StuckStopBroker()
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "flatten_accounting_precondition_failed"
    assert json.loads(state_file.read_text(encoding="utf-8")) == original


def test_flatten_live_futures_adopts_triggered_stop_before_clearing_state(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "open_positions": {
                    "s1": {
                        "direction": "long",
                        "broker_symbol": "BTCUSDT",
                        "broker_qty": 0.5,
                        "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                        "broker_stop_order_id": "stop-1",
                        "broker_stop_client_id": "tb-sl-stop-1",
                        "broker_stop_trigger_price": 95.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    live_product = product(tmp_path, execution_mode="live", state_file=state_file)

    class TriggeredStopBroker:
        name = "triggered-stop"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def get_position(self, symbol):
            return Position(symbol=symbol)

        def supports_native_protective_stops(self):
            return True

        def get_protective_stop(self, **kwargs):
            return ProtectiveOrder(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                qty=0.5,
                trigger_price=95.0,
                status=ProtectiveOrderStatus.TRIGGERED,
                order_id="stop-1",
                client_id="tb-sl-stop-1",
                filled_qty=0.5,
                average_price=94.5,
            )

        def cancel_protective_stop(self, **kwargs):
            pytest.fail("an already-triggered stop must not be canceled")

    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker", lambda product: TriggeredStopBroker()
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "flatten_accounting_precondition_failed"
    assert json.loads(state_file.read_text(encoding="utf-8"))["open_positions"]


def test_flatten_live_futures_closes_broker_but_refuses_symlink_local_state(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    target = tmp_path / "external_state.json"
    original = {
        "open_positions": {
            "s1": {
                "broker_qty": 0.5,
                "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
            }
        },
        "equity": 1000.0,
    }
    target.write_text(json.dumps(original), encoding="utf-8")
    state_file.symlink_to(target)
    live_product = product(
        tmp_path,
        execution_mode="live",
        state_file=state_file,
        strategies_path=tmp_path / "unused.json",
    )

    class FakeBroker:
        name = "fake-live"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def __init__(self):
            self.position = Position(symbol="BTCUSDT", qty=0.5, avg_price=100.0)
            self.closed = False

        def get_position(self, symbol):
            return self.position

        def close_position(self, symbol):
            self.closed = True
            self.position = Position(symbol=symbol, qty=0.0, avg_price=0.0)
            return Fill(symbol=symbol, side=OrderSide.SELL, qty=0.5, price=101.0, fee=0.1)

    broker = FakeBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)

    with pytest.raises(RuntimeError, match="cannot verify account identity"):
        flatten_product_once(live_product)

    assert broker.closed is False
    assert state_file.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == original


def test_flatten_live_futures_product_keeps_corrupt_state_when_stop_identity_is_unknown(
    monkeypatch, tmp_path
):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    state_file.write_text("{", encoding="utf-8")
    live_product = product(
        tmp_path,
        execution_mode="live",
        state_file=state_file,
        strategies_path=tmp_path / "unused.json",
    )

    class FakeBroker:
        name = "fake-live"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def __init__(self):
            self.position = Position(symbol="BTCUSDT", qty=0.5, avg_price=100.0)

        def get_position(self, symbol):
            return self.position

        def close_position(self, symbol):
            self.position = Position(symbol=symbol, qty=0.0, avg_price=0.0)
            return Fill(symbol=symbol, side=OrderSide.SELL, qty=0.5, price=101.0, fee=0.1)

    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: FakeBroker())

    with pytest.raises(RuntimeError, match="cannot verify account identity"):
        flatten_product_once(live_product)

    assert state_file.read_text(encoding="utf-8") == "{"


def test_flatten_btc_spot_step_aside_refuses_symlink_local_state_before_broker(tmp_path):
    state_file = tmp_path / "btc_state.json"
    target = tmp_path / "external_btc_state.json"
    original = {
        "open_positions": {
            "step_aside": {
                "direction": "short",
                "broker_symbol": "BTCUSDT",
                "broker_side": "sell",
                "broker_qty": 0.1,
                "broker_entry_price": 100.0,
                "broker_entry_quote_value": 10.0,
            }
        }
    }
    target.write_text(json.dumps(original), encoding="utf-8")
    state_file.symlink_to(target)
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
        strategies_path=tmp_path / "unused.json",
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "invalid_local_state"
    assert "state file must not be a symlink" in status["local_state"]["error"]
    assert state_file.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == original


def test_flatten_live_futures_reports_close_failure_with_position_context(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "open_positions": {
                    "s1": {
                        "broker_qty": 0.5,
                        "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                    }
                },
                "equity": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    live_product = product(
        tmp_path,
        execution_mode="live",
        state_file=state_file,
        strategies_path=tmp_path / "unused.json",
    )

    class CloseFailsBroker:
        name = "fake-live"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=0.5, avg_price=100.0)

        def close_position(self, symbol):
            raise RuntimeError("exchange timeout")

    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker", lambda product: CloseFailsBroker()
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["broker"] == "fake-live"
    assert status["reason"] == "flatten_accounting_precondition_failed"
    assert "last_flatten" not in json.loads(state_file.read_text(encoding="utf-8"))


def test_flatten_live_futures_reports_remaining_position_after_close(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "open_positions": {
                    "s1": {
                        "broker_qty": 0.5,
                        "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                    }
                },
                "equity": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    live_product = product(
        tmp_path,
        execution_mode="live",
        state_file=state_file,
        strategies_path=tmp_path / "unused.json",
    )

    class StillOpenBroker:
        name = "fake-live"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=0.25, avg_price=100.0)

        def close_position(self, symbol):
            return Fill(symbol=symbol, side=OrderSide.SELL, qty=0.25, price=101.0, fee=0.1)

    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker", lambda product: StillOpenBroker()
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "flatten_accounting_precondition_failed"
    assert "last_flatten" not in json.loads(state_file.read_text(encoding="utf-8"))


def test_flatten_live_futures_rejects_invalid_close_fill_before_clearing_state(
    monkeypatch, tmp_path
):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    original_state = {
        "open_positions": {
            "s1": {
                "broker_qty": 0.5,
                "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
            }
        },
        "equity": 1000.0,
    }
    state_file.write_text(json.dumps(original_state), encoding="utf-8")
    live_product = product(
        tmp_path,
        execution_mode="live",
        state_file=state_file,
        strategies_path=tmp_path / "unused.json",
    )

    class WrongSideBroker:
        name = "fake-live"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def __init__(self):
            self.position = Position(symbol="BTCUSDT", qty=0.5, avg_price=100.0)

        def get_position(self, symbol):
            return self.position

        def close_position(self, symbol):
            self.position = Position(symbol=symbol, qty=0.0, avg_price=0.0)
            return Fill(symbol=symbol, side=OrderSide.BUY, qty=0.5, price=101.0, fee=0.1)

    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker", lambda product: WrongSideBroker()
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "flatten_accounting_precondition_failed"
    assert json.loads(state_file.read_text(encoding="utf-8")) == original_state


def test_flatten_live_futures_already_flat_keeps_corrupt_state_for_stop_reconciliation(
    monkeypatch, tmp_path
):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    state_file.write_text("{", encoding="utf-8")
    live_product = product(
        tmp_path,
        execution_mode="live",
        state_file=state_file,
        strategies_path=tmp_path / "unused.json",
    )

    class FlatBroker:
        name = "flat-live"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=0.0, avg_price=0.0)

    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: FlatBroker())

    with pytest.raises(RuntimeError, match="cannot verify account identity"):
        flatten_product_once(live_product)

    assert state_file.read_text(encoding="utf-8") == "{"


def test_run_once_flatten_takes_precedence_over_pause(monkeypatch, tmp_path):
    (tmp_path / "control.json").write_text(
        json.dumps({"paused": True, "flatten_products": ["active_income"]}),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live", require_testnet_rehearsal=True)],
    )

    def flatten_success(product_config):
        status = {"product": {"name": product_config.name}, "ok": True, "action": "flatten"}
        if product_config.name == "btc_accumulation":
            status.update(skipped=True, reason="spot_flatten_not_supported")
        return status

    monkeypatch.setattr("src.autopilot.runtime.flatten_product_once", flatten_success)

    report = run_once(cfg)

    assert report["ok"] is True
    assert report["products"][0]["action"] == "flatten"
    control = json.loads((tmp_path / "control.json").read_text(encoding="utf-8"))
    assert control["flatten_products"] == []
    assert report["control_clear"][0]["name"] == "active_income"
    assert report["control_clear"][0]["ok"] is True
    audit_events = [
        json.loads(line) for line in cfg.control_audit_file.read_text(encoding="utf-8").splitlines()
    ]
    assert audit_events[-1]["actor"] == "autopilot"
    assert audit_events[-1]["command"] == "clear-flatten"
    assert audit_events[-1]["name"] == "active_income"


def test_run_once_successful_manual_flatten_request_atomically_leaves_product_paused(
    monkeypatch,
    tmp_path,
):
    control_path = tmp_path / "control.json"
    control_path.write_text(
        json.dumps({"flatten_products": ["active_income"]}),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=control_path,
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live", require_testnet_rehearsal=True)],
    )
    monkeypatch.setattr(
        "src.autopilot.runtime.flatten_product_once",
        lambda product_config: {
            "product": {"name": product_config.name},
            "ok": True,
            "action": "flatten",
        },
    )

    report = run_once(cfg)

    assert report["ok"] is True
    control = json.loads(control_path.read_text(encoding="utf-8"))
    assert control["paused"] is False
    assert control["paused_products"] == ["active_income"]
    assert control["flatten_products"] == []
    assert report["control_clear"][0]["paused_products"] == ["active_income"]
    audit_events = [
        json.loads(line) for line in cfg.control_audit_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(audit_events) == 1
    assert audit_events[0]["command"] == "clear-flatten"
    assert audit_events[0]["before"]["paused_products"] == []
    assert audit_events[0]["before"]["flatten_products"] == ["active_income"]
    assert audit_events[0]["after"]["paused_products"] == ["active_income"]
    assert audit_events[0]["after"]["flatten_products"] == []


def test_run_once_panic_control_pauses_jobs_and_runs_flatten_all(monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(
        json.dumps(
            {
                "paused": True,
                "pause_jobs": True,
                "flatten_all": True,
                "reason": "exchange incident",
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=control_path,
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        job_state_file=tmp_path / "job_state.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live", require_testnet_rehearsal=True)],
        jobs=[
            JobConfig(
                name="network_job",
                enabled=True,
                command=[sys.executable, "-c", "raise SystemExit(9)"],
                cadence_seconds=60,
                timeout_seconds=5,
                working_dir=tmp_path,
            )
        ],
    )

    def flatten_success(product_config):
        status = {"product": {"name": product_config.name}, "ok": True, "action": "flatten"}
        if product_config.name == "btc_accumulation":
            status.update(skipped=True, reason="spot_flatten_not_supported")
        return status

    monkeypatch.setattr("src.autopilot.runtime.flatten_product_once", flatten_success)

    report = run_once(cfg)

    assert report["ok"] is True
    assert report["jobs"] == []
    assert report["products"] == [
        {"product": {"name": "active_income"}, "ok": True, "action": "flatten"}
    ]
    assert not cfg.job_state_file.exists()
    control = json.loads(control_path.read_text(encoding="utf-8"))
    assert control["paused"] is True
    assert control["pause_jobs"] is True
    assert control["flatten_all"] is False
    assert control["reason"] == "auto-cleared after successful runtime flatten"
    assert report["control_clear"][0]["ok"] is True
    assert report["control_clear"][0]["name"] is None
    assert report["control_clear"][0]["targets"] == [{"product_name": "active_income", "ok": True}]


def test_run_once_keeps_failed_flatten_request_for_retry(monkeypatch, tmp_path):
    (tmp_path / "control.json").write_text(
        json.dumps({"flatten_products": ["active_income"]}),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, execution_mode="live", require_testnet_rehearsal=True)],
    )
    monkeypatch.setattr(
        "src.autopilot.runtime.flatten_product_once",
        lambda product: {
            "product": {"name": product.name},
            "ok": False,
            "action": "flatten",
            "error": "exchange timeout",
        },
    )

    report = run_once(cfg)

    assert report["ok"] is False
    assert "control_clear" not in report
    control = json.loads((tmp_path / "control.json").read_text(encoding="utf-8"))
    assert control["flatten_products"] == ["active_income"]
    assert control.get("paused", False) is False
    assert control.get("paused_products", []) == []
    assert not cfg.control_audit_file.exists()


def test_run_once_clears_flatten_all_only_after_all_targets_succeed(monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps({"flatten_all": True}), encoding="utf-8")
    active = product(
        tmp_path,
        name="active_income",
        execution_mode="live",
        require_testnet_rehearsal=True,
    )
    btc = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        strategies_path=tmp_path / "btc.json",
        state_file=tmp_path / "btc_state.json",
        trade_log=tmp_path / "btc_trades.csv",
        preflight_report=tmp_path / "btc_preflight.json",
        testnet_rehearsal_report=tmp_path / "btc_testnet.json",
    )
    cfg = AutopilotConfig(
        control_file=control_path,
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[active, btc],
    )

    def flatten_success(product_config):
        status = {"product": {"name": product_config.name}, "ok": True, "action": "flatten"}
        if product_config.name == "btc_accumulation":
            status.update(skipped=True, reason="spot_flatten_not_supported")
        return status

    monkeypatch.setattr("src.autopilot.runtime.flatten_product_once", flatten_success)

    report = run_once(cfg)

    assert report["ok"] is True
    assert len(report["products"]) == 2
    control = json.loads(control_path.read_text(encoding="utf-8"))
    assert control["paused"] is True
    assert control["flatten_all"] is False
    assert control["flatten_products"] == []
    assert report["control_clear"][0]["paused"] is True
    assert report["control_clear"][0]["name"] is None
    assert report["control_clear"][0]["targets"] == [
        {"product_name": "active_income", "ok": True},
        {
            "product_name": "btc_accumulation",
            "ok": True,
            "skipped": True,
            "reason": "spot_flatten_not_supported",
        },
    ]
    audit_events = [
        json.loads(line) for line in cfg.control_audit_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(audit_events) == 1
    assert audit_events[-1]["command"] == "clear-flatten"
    assert audit_events[-1]["name"] is None
    assert audit_events[-1]["before"]["paused"] is False
    assert audit_events[-1]["before"]["flatten_all"] is True
    assert audit_events[-1]["after"]["paused"] is True
    assert audit_events[-1]["after"]["flatten_all"] is False


def test_run_once_preserves_flatten_all_when_any_target_fails(monkeypatch, tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps({"flatten_all": True}), encoding="utf-8")
    active = product(
        tmp_path,
        name="active_income",
        execution_mode="live",
        require_testnet_rehearsal=True,
    )
    btc = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        strategies_path=tmp_path / "btc.json",
        state_file=tmp_path / "btc_state.json",
        trade_log=tmp_path / "btc_trades.csv",
        preflight_report=tmp_path / "btc_preflight.json",
        testnet_rehearsal_report=tmp_path / "btc_testnet.json",
    )
    cfg = AutopilotConfig(
        control_file=control_path,
        control_audit_file=tmp_path / "control_audit.jsonl",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[active, btc],
    )

    def flatten(product_config):
        return {
            "product": {"name": product_config.name},
            "ok": product_config.name != "active_income",
            "action": "flatten",
        }

    monkeypatch.setattr("src.autopilot.runtime.flatten_product_once", flatten)

    report = run_once(cfg)

    assert report["ok"] is False
    control = json.loads(control_path.read_text(encoding="utf-8"))
    assert control["flatten_all"] is True
    assert report["control_clear"] == [
        {
            "command": "clear-flatten",
            "name": None,
            "ok": True,
            "skipped": True,
            "reason": "flatten_all_has_failures",
            "targets": [
                {"product_name": "active_income", "ok": False},
                {"product_name": "btc_accumulation", "ok": True},
            ],
        }
    ]
    assert not cfg.control_audit_file.exists()


def test_flatten_live_spot_product_without_local_step_aside_is_skipped(tmp_path):
    state_file = tmp_path / "state.json"
    state = {
        "equity": 1.0,
        "open_positions": {},
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is True
    assert status["skipped"] is True
    assert status["reason"] == "no_local_spot_step_aside_position"
    assert status["local_open_positions"] == 0
    assert json.loads(state_file.read_text(encoding="utf-8")) == state


def test_flatten_live_spot_step_aside_reinvests_quote_and_clears_state(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "equity": 1.0,
                "open_positions": {
                    "btc_step_aside": {
                        "direction": "short",
                        "broker_symbol": "BTCUSDT",
                        "broker_qty": 0.5,
                        "broker_side": "sell",
                        "broker_entry_price": 100.0,
                        "broker_entry_fee": 0.05,
                        "broker_entry_quote_value": 50.0,
                        "broker_requested_qty": 0.5,
                        "broker_fill_ratio": 1.0,
                        "broker_exit_sizing": "quote_reinvest",
                        "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
        symbol="BTCUSDT",
    )

    class FakeSpotBroker:
        name = "fake-spot"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def __init__(self):
            self.position = Position(symbol="BTCUSDT", qty=0.8, avg_price=0.0)
            self.orders = []
            self.normalizations = []
            self.persisted_intents = []

        def get_position(self, symbol):
            return self.position

        def get_price(self, symbol):
            return 125.0

        def normalize_order_qty(self, symbol, qty, *, price=None, reduce_only=False):
            self.normalizations.append((symbol, qty, price, reduce_only))
            return qty - 0.001

        def place_order(self, order):
            durable_state = json.loads(state_file.read_text(encoding="utf-8"))
            self.persisted_intents.append(durable_state["flatten_intent"])
            self.orders.append(order)
            self.position = Position(
                symbol=order.symbol,
                qty=self.position.qty + order.qty,
                avg_price=125.0,
            )
            return Fill(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=125.0,
                fee=0.02,
            )

    broker = FakeSpotBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["action"] == "flatten"
    assert status["broker"] == "fake-spot"
    assert status["reason"] == "flatten_accounting_precondition_failed"
    assert broker.orders == []
    assert broker.normalizations == []
    assert broker.persisted_intents == []
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["open_positions"]
    assert "flatten_intent" not in state


def test_flatten_live_spot_ambiguous_submission_retains_intent_and_never_duplicates(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "equity": 1.0,
                "open_positions": {
                    "btc_step_aside": {
                        "direction": "short",
                        "broker_symbol": "BTCUSDT",
                        "broker_qty": 0.5,
                        "broker_side": "sell",
                        "broker_entry_price": 100.0,
                        "broker_entry_quote_value": 50.0,
                        "broker_exit_sizing": "quote_reinvest",
                        "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
        symbol="BTCUSDT",
    )

    class AmbiguousSpotBroker:
        name = "ambiguous-spot"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def __init__(self):
            self.position = Position(symbol="BTCUSDT", qty=0.8, avg_price=0.0)
            self.orders = []
            self.price_reads = 0
            self.normalizations = 0

        def get_position(self, symbol):
            return self.position

        def get_price(self, symbol):
            self.price_reads += 1
            return 125.0

        def normalize_order_qty(self, symbol, qty, *, price=None, reduce_only=False):
            self.normalizations += 1
            return qty - 0.001

        def place_order(self, order):
            self.orders.append(order)
            raise RuntimeError("submission timed out")

    broker = AmbiguousSpotBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)

    first = flatten_product_once(live_product)
    intent_state = json.loads(state_file.read_text(encoding="utf-8"))
    second = flatten_product_once(live_product)

    assert first["ok"] is False
    assert first["reason"] == "flatten_accounting_precondition_failed"
    assert second["ok"] is False
    assert second["reason"] == "flatten_accounting_precondition_failed"
    assert len(broker.orders) == 0
    assert broker.price_reads == 0
    assert broker.normalizations == 0
    assert json.loads(state_file.read_text(encoding="utf-8")) == intent_state


def test_flatten_live_spot_restart_auto_finalizes_when_balance_proves_fill(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "equity": 1.0,
                "open_positions": {
                    "btc_step_aside": {
                        "direction": "short",
                        "broker_symbol": "BTCUSDT",
                        "broker_qty": 0.5,
                        "broker_side": "sell",
                        "broker_entry_price": 100.0,
                        "broker_entry_quote_value": 50.0,
                        "broker_exit_sizing": "quote_reinvest",
                        "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
        symbol="BTCUSDT",
    )

    class FilledThenTimedOutSpotBroker:
        name = "filled-then-timeout-spot"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def __init__(self):
            self.position = Position(symbol="BTCUSDT", qty=0.8, avg_price=0.0)
            self.orders = []
            self.price_reads = 0

        def get_position(self, symbol):
            return self.position

        def get_price(self, symbol):
            self.price_reads += 1
            return 125.0

        def normalize_order_qty(self, symbol, qty, *, price=None, reduce_only=False):
            return qty - 0.001

        def place_order(self, order):
            self.orders.append(order)
            self.position = Position(
                symbol=order.symbol,
                qty=self.position.qty + order.qty,
                avg_price=125.0,
            )
            raise RuntimeError("response timed out after exchange fill")

    broker = FilledThenTimedOutSpotBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)

    first = flatten_product_once(live_product)
    second = flatten_product_once(live_product)

    assert first["ok"] is False
    assert first["reason"] == "flatten_accounting_precondition_failed"
    assert second["ok"] is False
    assert second["reason"] == "flatten_accounting_precondition_failed"
    assert len(broker.orders) == 0
    assert broker.price_reads == 0
    final_state = json.loads(state_file.read_text(encoding="utf-8"))
    assert final_state["open_positions"]
    assert "flatten_intent" not in final_state


def test_flatten_live_spot_malformed_intent_fails_closed_without_broker(
    monkeypatch,
    tmp_path,
):
    state_file = tmp_path / "state.json"
    original_state = {
        "equity": 1.0,
        "open_positions": {
            "btc_step_aside": {
                "direction": "short",
                "broker_symbol": "BTCUSDT",
                "broker_qty": 0.5,
                "broker_side": "sell",
                "broker_entry_price": 100.0,
                "broker_entry_quote_value": 50.0,
                "broker_exit_sizing": "quote_reinvest",
                "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
            }
        },
        "flatten_intent": {"version": 1},
    }
    state_file.write_text(json.dumps(original_state), encoding="utf-8")
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
        symbol="BTCUSDT",
    )
    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker",
        lambda product: pytest.fail("malformed intent must fail before broker construction"),
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "unresolved_flatten_intent"
    assert "missing required key(s)" in status["error"]
    assert status["flatten_intent"] == {"version": 1}
    assert json.loads(state_file.read_text(encoding="utf-8")) == original_state


def test_flatten_live_spot_post_fill_balance_mismatch_retains_intent(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    original_state = {
        "equity": 1.0,
        "open_positions": {
            "unsafe strategy id / " * 8: {
                "direction": "short",
                "broker_symbol": "BTCUSDT",
                "broker_qty": 0.5,
                "broker_side": "sell",
                "broker_entry_price": 100.0,
                "broker_entry_quote_value": 50.0,
                "broker_exit_sizing": "quote_reinvest",
                "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
            }
        },
    }
    state_file.write_text(json.dumps(original_state), encoding="utf-8")
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
        symbol="BTCUSDT",
    )

    class WrongBalanceSpotBroker:
        name = "wrong-balance-spot"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

        def __init__(self):
            self.position = Position(symbol="BTCUSDT", qty=0.8, avg_price=0.0)
            self.order = None

        def get_position(self, symbol):
            return self.position

        def get_price(self, symbol):
            return 125.0

        def normalize_order_qty(self, symbol, qty, *, price=None, reduce_only=False):
            return qty - 0.001

        def place_order(self, order):
            self.order = order
            self.position = Position(
                symbol=order.symbol,
                qty=self.position.qty + (order.qty / 2),
                avg_price=125.0,
            )
            return Fill(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=125.0,
                fee=0.02,
            )

    broker = WrongBalanceSpotBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "flatten_accounting_precondition_failed"
    assert broker.order is None
    retained_state = json.loads(state_file.read_text(encoding="utf-8"))
    assert retained_state["open_positions"] == original_state["open_positions"]
    assert "flatten_intent" not in retained_state


def test_flatten_live_spot_step_aside_rejects_missing_quote_budget(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    original_state = {
        "equity": 1.0,
        "open_positions": {
            "btc_step_aside": {
                "direction": "short",
                "broker_symbol": "BTCUSDT",
                "broker_qty": 0.5,
                "broker_side": "sell",
                "broker_entry_price": 100.0,
                "broker_exit_sizing": "quote_reinvest",
                "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
            }
        },
    }
    state_file.write_text(json.dumps(original_state), encoding="utf-8")
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
        symbol="BTCUSDT",
    )
    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker",
        lambda product: pytest.fail(
            "broker should not be built when spot flatten state is invalid"
        ),
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "invalid_spot_step_aside_state"
    assert "invalid_spot_step_aside_quote_value" in status["error"]
    assert json.loads(state_file.read_text(encoding="utf-8")) == original_state


def test_flatten_live_spot_product_fails_closed_on_corrupt_local_state(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("{", encoding="utf-8")
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        state_file=state_file,
    )

    status = flatten_product_once(live_product)

    assert status["ok"] is False
    assert status["reason"] == "invalid_local_state"
    assert "JSONDecodeError" in status["local_state"]["error"]
    assert state_file.read_text(encoding="utf-8") == "{"


def test_run_once_executes_due_jobs(tmp_path):
    cfg = AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        job_state_file=tmp_path / "job_state.json",
        jobs=[
            JobConfig(
                name="job",
                enabled=True,
                command=[sys.executable, "-c", "print('job ok')"],
                cadence_seconds=60,
                timeout_seconds=5,
                working_dir=tmp_path,
            )
        ],
        products=[],
    )

    report = run_once(cfg)

    assert report["ok"] is True
    assert report["jobs"][0]["ok"] is True
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["jobs"][0]["stdout_tail"].strip() == "job ok"


def strategy_artifact(path):
    strategy = {
        "id": "live_r1",
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


def btc_strategy_artifact(path):
    strategy = strategy_artifact(path)
    strategy["market"] = "spot"
    strategy["direction"] = "short"
    strategy["pnl_unit"] = "btc"
    strategy["risk"]["daily_stop_loss"] = -0.005
    strategy["risk"]["max_trades_per_day"] = 1
    strategy["metrics"]["holdout_excess_return_vs_buy_hold"] = 0.03
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "spot",
                "symbol": "BTCUSDT",
                "pnl_unit": "btc",
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": [strategy],
            }
        ),
        encoding="utf-8",
    )
    return strategy


def passing_preflight_checks(product_config):
    checks = [
        {"name": "product_config", "ok": True},
        {"name": "execution_engine_identity", "ok": True},
        {"name": "strategy_artifact_exists", "ok": True},
        {"name": "strategy_fingerprints", "ok": True},
        {"name": "strategy_policy", "ok": True},
        {
            "name": "exchange_environment",
            "ok": True,
            "detail": exchange_environment_detail(product_config),
        },
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


def write_preflight(
    path,
    product_config,
    *,
    ok=True,
    generated_ts=None,
    strategies_path=None,
    artifact_fingerprints=None,
    checks=None,
):
    if artifact_fingerprints is None and product_config.strategies_path.exists():
        payload = json.loads(product_config.strategies_path.read_text(encoding="utf-8"))
        artifact_fingerprints = [
            strategy_fingerprint(strategy) for strategy in payload.get("strategies", [])
        ]
    artifact_payload = (
        json.loads(product_config.strategies_path.read_text(encoding="utf-8"))
        if product_config.strategies_path.exists()
        else None
    )
    report = {
        "generated_at": "2026-01-01T00:00:00Z",
        "generated_ts": time.time() if generated_ts is None else generated_ts,
        "ok": ok,
        "products": [
            {
                "artifact_fingerprints": artifact_fingerprints,
                "artifact_digest": artifact_digest(artifact_payload)
                if isinstance(artifact_payload, dict)
                else None,
                "execution_engine_digest": execution_engine_digest(),
                "ok": ok,
                "product": canonical_product_config(product_config)
                | (
                    {"strategies_path": str(Path(strategies_path).resolve())}
                    if strategies_path is not None
                    else {}
                ),
                "checks": passing_preflight_checks(product_config) if checks is None else checks,
                "errors": [] if ok else ["failed"],
            }
        ],
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def test_recent_preflight_gate_accepts_matching_report(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    write_preflight(preflight_path, live_product)

    gate = assert_recent_preflight(live_product)

    assert gate["ok"] is True
    assert gate["required"] is True
    assert gate["report"] == str(preflight_path)
    assert len(gate["artifact_fingerprints"]) == 1
    assert gate["exchange_environment"]["market_type"] == "futures"
    assert gate["exchange_environment"]["max_futures_leverage"] == 1
    assert gate["position_mode"] == {"symbol": "BTCUSDT", "one_way": True}
    assert gate["open_order_inventory"] == {
        "scope": "whole_account",
        "configured_symbol": "BTCUSDT",
        "regular": {"count": 0, "orders": []},
        "conditional": {"count": 0, "orders": []},
    }
    assert gate["position_inventory"] == {
        "scope": "whole_account",
        "configured_symbol": "BTCUSDT",
        "count": 0,
        "positions": [],
    }


def test_recent_preflight_gate_rejects_invalid_account_fingerprint(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    for check in report["products"][0]["checks"]:
        if check["name"] == "exchange_environment":
            check["detail"]["account_fingerprint"] = "key-is-not-safe-evidence"
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="account fingerprint evidence is invalid"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_requires_one_way_position_mode_evidence(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    report["products"][0]["checks"] = [
        check
        for check in report["products"][0]["checks"]
        if check["name"] != "broker_position_mode_one_way"
    ]
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="missing required check broker_position_mode_one_way",
    ):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_symlink_report(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    target = tmp_path / "target_preflight.json"
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    write_preflight(target, live_product)
    preflight_path.symlink_to(target)

    with pytest.raises(RuntimeError, match="preflight report must not be a symlink"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_active_income_leverage_above_one(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    for check in report["products"][0]["checks"]:
        if check["name"] == "exchange_environment":
            check["detail"]["max_futures_leverage"] = 2
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="max_futures_leverage evidence must be 1"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_missing_exchange_evidence(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    for check in report["products"][0]["checks"]:
        if check["name"] == "exchange_environment":
            check.pop("detail")
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exchange environment evidence is missing"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_missing_connectivity_evidence(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    for check in report["products"][0]["checks"]:
        if check["name"] == "exchange_read_connectivity":
            check.pop("detail")
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exchange connectivity evidence is missing"):
        assert_recent_preflight(live_product)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("price", 0.0, "price evidence is invalid"),
        ("balance", 0.0, "balance evidence is invalid"),
        ("position_qty", "bad", "position_qty evidence is invalid"),
        ("position_avg_price", -1.0, "position_avg_price evidence is invalid"),
        ("position_is_flat", "yes", "position_is_flat evidence is invalid"),
    ],
)
def test_recent_preflight_gate_rejects_invalid_connectivity_evidence(
    tmp_path, field, value, message
):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    for check in report["products"][0]["checks"]:
        if check["name"] == "exchange_read_connectivity":
            check["detail"][field] = value
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_non_flat_connectivity_evidence(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    for check in report["products"][0]["checks"]:
        if check["name"] == "exchange_read_connectivity":
            check["detail"]["position_qty"] = 0.25
            check["detail"]["position_avg_price"] = 100.0
            check["detail"]["position_is_flat"] = False
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exchange connectivity position is not flat"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_missing_report(tmp_path):
    live_product = product(tmp_path, execution_mode="live")

    with pytest.raises(RuntimeError, match="preflight report not found"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_non_object_report(tmp_path):
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text("[]", encoding="utf-8")
    live_product = product(tmp_path, execution_mode="live", preflight_report=preflight_path)

    with pytest.raises(RuntimeError, match="preflight report must be a JSON object"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_invalid_products_payload(tmp_path):
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps({"ok": True, "generated_ts": time.time(), "products": {}}),
        encoding="utf-8",
    )
    live_product = product(tmp_path, execution_mode="live", preflight_report=preflight_path)

    with pytest.raises(RuntimeError, match="preflight report products must be a list"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_invalid_product_entry(tmp_path):
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps({"ok": True, "generated_ts": time.time(), "products": ["bad"]}),
        encoding="utf-8",
    )
    live_product = product(tmp_path, execution_mode="live", preflight_report=preflight_path)

    with pytest.raises(RuntimeError, match="preflight report products must contain JSON objects"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_missing_required_check(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    report["products"][0]["checks"] = [
        check
        for check in report["products"][0]["checks"]
        if check["name"] != "exchange_read_connectivity"
    ]
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing required check exchange_read_connectivity"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_failed_required_check(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    for check in report["products"][0]["checks"]:
        if check["name"] == "broker_constructed":
            check["ok"] = False
            check["error"] = "broker unavailable"
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="required check broker_constructed failed"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_requires_empty_open_order_evidence(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    for check in report["products"][0]["checks"]:
        if check["name"] == "broker_open_orders_empty":
            check["detail"]["regular"]["count"] = 1
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="regular open-order count is not zero"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_disabled_live_preflight(tmp_path):
    live_product = product(tmp_path, execution_mode="live", require_preflight=False)

    with pytest.raises(RuntimeError, match="requires require_preflight=true"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_stale_report(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
        preflight_max_age_seconds=60,
    )
    write_preflight(preflight_path, live_product, generated_ts=time.time() - 120)

    with pytest.raises(RuntimeError, match="preflight report is stale"):
        assert_recent_preflight(live_product)


@pytest.mark.parametrize("generated_ts", ["soon", float("nan"), float("inf")])
def test_recent_preflight_gate_rejects_invalid_generated_ts(tmp_path, generated_ts):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    write_preflight(preflight_path, live_product, generated_ts=generated_ts)

    with pytest.raises(RuntimeError, match="generated_ts is not"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_future_report_timestamp(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    write_preflight(preflight_path, live_product, generated_ts=time.time() + 600)

    with pytest.raises(RuntimeError, match="timestamp is in the future"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_failed_report(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    write_preflight(preflight_path, live_product, ok=False)

    with pytest.raises(RuntimeError, match="preflight report failed"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_artifact_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    write_preflight(preflight_path, live_product, strategies_path=tmp_path / "other.json")

    with pytest.raises(RuntimeError, match="preflight report product strategies_path mismatch"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_product_symbol_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    preflight_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
        symbol="BTCUSDT",
    )
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
        symbol="ETHUSDT",
    )
    write_preflight(preflight_path, preflight_product)

    with pytest.raises(RuntimeError, match="preflight report product symbol mismatch"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_missing_fingerprints(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    write_preflight(preflight_path, live_product, artifact_fingerprints=[])

    with pytest.raises(RuntimeError, match="preflight report has no artifact_fingerprints"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_non_list_fingerprints(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    write_preflight(preflight_path, live_product, artifact_fingerprints="not-a-list")

    with pytest.raises(RuntimeError, match="preflight report has no artifact_fingerprints"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_missing_artifact_digest(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    report["products"][0].pop("artifact_digest")
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="preflight report has no artifact_digest"):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_artifact_digest_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    report = write_preflight(preflight_path, live_product)
    report["products"][0]["artifact_digest"] = "sha256:not-current"
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        RuntimeError, match="preflight artifact digest does not match current artifact"
    ):
        assert_recent_preflight(live_product)


def test_recent_preflight_gate_rejects_changed_artifact_content(tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    preflight_path = tmp_path / "preflight.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=preflight_path,
    )
    write_preflight(preflight_path, live_product)
    strategy["take_profit"] = 0.03
    artifact.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": [strategy],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="preflight artifact digest does not match current artifact"
    ):
        assert_recent_preflight(live_product)


def test_recent_testnet_rehearsal_gate_accepts_matching_report(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(
        report_path,
        preflight=preflight,
        product_payload=canonical_product_config(live_product),
    )

    gate = assert_recent_testnet_rehearsal(live_product)

    assert gate["ok"] is True
    assert gate["required"] is True
    assert gate["report"] == str(report_path)
    assert gate["notional_usd"] == 5.0
    assert gate["final_position_flat"] is True
    assert len(gate["artifact_fingerprints"]) == 1
    assert gate["preflight_position_mode"] == {
        "symbol": "BTCUSDT",
        "one_way": True,
    }


def test_recent_testnet_rehearsal_gate_rejects_missing_native_stop_evidence(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(report_path, preflight=preflight)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.pop("native_protective_stop")
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing_native_protective_stop"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_invalid_native_stop_evidence(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(report_path, preflight=preflight)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["native_protective_stop"]["fetched_terminal"]["status"] = "open"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="native_stop_fetched_terminal_status_not_terminal",
    ):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_missing_embedded_stop_capability(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"][0]["checks"] = [
        check
        for check in preflight["products"][0]["checks"]
        if check["name"] != "broker_native_protective_stops"
    ]
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(
        RuntimeError,
        match="embedded_preflight_missing_native_stop_capability",
    ):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_missing_one_way_mode_evidence(
    tmp_path,
):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"][0]["checks"] = [
        check
        for check in preflight["products"][0]["checks"]
        if check["name"] != "broker_position_mode_one_way"
    ]
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(
        RuntimeError,
        match="embedded_preflight_missing_one_way_position_mode",
    ):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_missing_embedded_open_order_inventory(
    tmp_path,
):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"][0]["checks"] = [
        check
        for check in preflight["products"][0]["checks"]
        if check["name"] != "broker_open_orders_empty"
    ]
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(
        RuntimeError,
        match="embedded_preflight_missing_open_order_inventory",
    ):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_symlink_report(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    target = tmp_path / "target_testnet.json"
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(target, preflight=preflight)
    report_path.symlink_to(target)

    with pytest.raises(RuntimeError, match="testnet rehearsal report must not be a symlink") as exc:
        assert_recent_testnet_rehearsal(live_product)
    assert "make testnet-status" in str(exc.value)


def test_recent_testnet_rehearsal_gate_rejects_missing_fill_symbol_evidence(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(report_path, preflight=preflight)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["entry_fill"].pop("symbol")
    payload["close_fill"]["symbol"] = "ETHUSDT"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="entry_fill_symbol_mismatch") as exc:
        assert_recent_testnet_rehearsal(live_product)
    assert "close_fill_symbol_mismatch" in str(exc.value)


def test_recent_testnet_rehearsal_gate_rejects_fill_qty_not_matching_order_qty(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(report_path, preflight=preflight)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["order_qty"] = 0.04
    payload["close_fill"]["qty"] = 0.03
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="entry_fill_qty_mismatch") as exc:
        assert_recent_testnet_rehearsal(live_product)
    assert "close_fill_qty_mismatch" in str(exc.value)


def test_recent_testnet_rehearsal_gate_rejects_missing_report(tmp_path):
    live_product = product(
        tmp_path,
        execution_mode="live",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=tmp_path / "missing.json",
    )

    with pytest.raises(RuntimeError, match="testnet rehearsal report not found") as exc:
        assert_recent_testnet_rehearsal(live_product)
    assert "make preflight PRODUCT=active_income REQUIRE_TESTNET=1" in str(exc.value)
    assert "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100" in str(exc.value)
    assert "make testnet-status" in str(exc.value)


def test_recent_testnet_rehearsal_gate_rejects_non_object_report(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text("[]", encoding="utf-8")
    live_product = product(
        tmp_path,
        execution_mode="live",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )

    with pytest.raises(RuntimeError, match="testnet rehearsal gate failed: read_error"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_disabled_active_income_live_gate(tmp_path):
    live_product = product(
        tmp_path,
        execution_mode="live",
        require_testnet_rehearsal=False,
    )

    with pytest.raises(RuntimeError, match="requires require_testnet_rehearsal=true"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_stale_report(tmp_path):
    report_path = tmp_path / "testnet.json"
    write_testnet_rehearsal(report_path, generated_ts=time.time() - 120)
    live_product = product(
        tmp_path,
        execution_mode="live",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
        testnet_rehearsal_max_age_seconds=60,
    )

    with pytest.raises(RuntimeError, match="testnet rehearsal gate failed: stale"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_future_report_timestamp(tmp_path):
    report_path = tmp_path / "testnet.json"
    write_testnet_rehearsal(report_path, generated_ts=time.time() + 600)
    live_product = product(
        tmp_path,
        execution_mode="live",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )

    with pytest.raises(RuntimeError, match="future_generated_ts") as exc:
        assert_recent_testnet_rehearsal(live_product)
    assert "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100" in str(exc.value)


def test_recent_testnet_rehearsal_gate_rejects_non_testnet_report(tmp_path):
    report_path = tmp_path / "testnet.json"
    write_testnet_rehearsal(report_path, testnet=False)
    live_product = product(
        tmp_path,
        execution_mode="live",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )

    with pytest.raises(RuntimeError, match="not produced on testnet"):
        assert_recent_testnet_rehearsal(live_product)


@pytest.mark.parametrize("exchange", ["okx", "", None])
def test_recent_testnet_rehearsal_gate_rejects_wrong_exchange(tmp_path, exchange):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(report_path, exchange=exchange, preflight=preflight)

    with pytest.raises(RuntimeError, match="testnet rehearsal exchange mismatch"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_report_product_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(
        report_path,
        product_payload={
            "name": "active_income",
            "objective": "active_income",
            "base_asset": "USDT",
            "market": "futures",
            "symbol": "ETHUSDT",
        },
        preflight=preflight,
    )

    with pytest.raises(RuntimeError, match="product_symbol_mismatch") as exc:
        assert_recent_testnet_rehearsal(live_product)
    assert "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100" in str(exc.value)


def test_recent_testnet_rehearsal_gate_rejects_missing_risk_controls(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(report_path, risk_controls={}, preflight=preflight)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.pop("risk_controls")
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing_risk_controls") as exc:
        assert_recent_testnet_rehearsal(live_product)
    assert "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100" in str(exc.value)


def test_recent_testnet_rehearsal_gate_rejects_unsafe_risk_controls(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(
        report_path,
        risk_controls={
            "max_futures_leverage": 10,
            "futures_margin_mode": "cross",
            "max_notional_usd": 0,
            "max_fill_slippage_bps": -1,
        },
        preflight=preflight,
    )

    with pytest.raises(RuntimeError, match="max_futures_leverage_invalid") as exc:
        assert_recent_testnet_rehearsal(live_product)
    message = str(exc.value)
    assert "futures_margin_mode_not_isolated" in message
    assert "max_notional_usd_invalid" in message
    assert "max_fill_slippage_bps_invalid" in message


def test_recent_testnet_rehearsal_gate_rejects_missing_embedded_preflight(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    write_testnet_rehearsal(
        report_path,
        product_payload=canonical_product_config(live_product),
    )

    with pytest.raises(RuntimeError, match="testnet rehearsal report has no embedded preflight"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_failed_embedded_preflight(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product, ok=False)
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(RuntimeError, match="embedded_preflight_failed"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_invalid_embedded_preflight_products(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"] = {}
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(RuntimeError, match="embedded_preflight_products_invalid"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_invalid_embedded_preflight_product_entry(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"] = ["bad"]
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(RuntimeError, match="embedded_preflight_products_invalid"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_failed_embedded_product_preflight(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"][0]["ok"] = False
    preflight["products"][0]["errors"] = ["failed"]
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(RuntimeError, match="embedded_preflight_product_failed"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_missing_embedded_required_preflight_check(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"][0]["checks"] = [
        check
        for check in preflight["products"][0]["checks"]
        if check["name"] != "broker_position_flat"
    ]
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(RuntimeError, match="embedded_preflight_missing_position_inventory"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_embedded_preflight_product_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    preflight_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        symbol="BTCUSDT",
    )
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        symbol="ETHUSDT",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", preflight_product)
    write_testnet_rehearsal(
        report_path,
        product_payload={
            "name": live_product.name,
            "objective": live_product.objective,
            "base_asset": live_product.base_asset,
            "market": live_product.market,
            "symbol": live_product.symbol,
        },
        preflight=preflight,
    )

    with pytest.raises(RuntimeError, match="embedded_preflight_product_symbol_mismatch"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_embedded_preflight_artifact_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(
        tmp_path / "preflight.json",
        live_product,
        strategies_path=tmp_path / "other.json",
    )
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(
        RuntimeError,
        match="embedded_preflight_product_strategies_path_mismatch",
    ):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_non_list_embedded_fingerprints(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"][0]["artifact_fingerprints"] = "not-a-list"
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(
        RuntimeError, match="testnet rehearsal preflight has no artifact_fingerprints"
    ):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_missing_embedded_artifact_digest(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"][0].pop("artifact_digest")
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(RuntimeError, match="embedded_preflight_missing_artifact_digest"):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_embedded_artifact_digest_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    preflight["products"][0]["artifact_digest"] = "sha256:not-current"
    write_testnet_rehearsal(report_path, preflight=preflight)

    with pytest.raises(
        RuntimeError,
        match="embedded_preflight_artifact_digest_mismatch",
    ):
        assert_recent_testnet_rehearsal(live_product)


def test_recent_testnet_rehearsal_gate_rejects_changed_artifact_content(tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    report_path = tmp_path / "testnet.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=report_path,
    )
    preflight = write_preflight(tmp_path / "preflight.json", live_product)
    write_testnet_rehearsal(report_path, preflight=preflight)
    strategy["take_profit"] = 0.03
    artifact.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": [strategy],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="embedded_preflight_artifact_digest_mismatch",
    ):
        assert_recent_testnet_rehearsal(live_product)


def test_live_product_requires_strategy_approval(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    live_product = product(tmp_path, execution_mode="live", strategies_path=artifact)

    with pytest.raises(ApprovalError, match="Live trading blocked"):
        run_product_once(live_product, approval_ledger=tmp_path / "approvals.json")


def test_live_existing_position_management_skips_entry_only_gates(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"open_positions": {"active_1": {"direction": "long"}}}),
        encoding="utf-8",
    )
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        state_file=state_file,
        preflight_report=tmp_path / "missing-preflight.json",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=tmp_path / "missing-rehearsal.json",
    )
    seen = {}

    class FakeBroker:
        name = "risk-reduction-broker"

    class FakeBot:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.state = {
                "equity": 1000.0,
                "open_positions": {"active_1": {"direction": "long"}},
                "inactive_strategies": [],
            }
            self.cycle_errors = []

        def run_cycle(self):
            seen["ran"] = True

    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: FakeBroker())
    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", FakeBot)

    status = run_product_once(live_product, approval_ledger=tmp_path / "missing-ledger.json")

    assert status["ok"] is True
    assert status["entries_allowed"] is False
    assert status["approval_gate"] == "management_only"
    assert status["preflight_gate"]["skipped"] is True
    assert status["testnet_rehearsal_gate"]["skipped"] is True
    assert seen["allow_entries"] is False
    assert seen["ran"] is True


def test_live_management_recovers_frozen_strategy_when_artifact_is_missing(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    artifact.unlink()
    canonical = json.dumps(
        strategy,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "open_positions": {
                    strategy["id"]: {
                        "direction": strategy["direction"],
                        "strategy_snapshot": strategy,
                        "strategy_fingerprint": hashlib.sha256(
                            canonical.encode("utf-8")
                        ).hexdigest(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        state_file=state_file,
    )
    seen = {}

    class FakeBroker:
        name = "management-recovery"

    class FakeBot:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.state = {
                "equity": 1000.0,
                "open_positions": {strategy["id"]: {"direction": "long"}},
                "inactive_strategies": [],
            }
            self.cycle_errors = []

        def run_cycle(self):
            seen["ran"] = True

    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: FakeBroker())
    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", FakeBot)

    status = run_product_once(live_product, approval_ledger=tmp_path / "missing-ledger.json")

    assert status["ok"] is True
    assert status["entries_allowed"] is False
    assert status["strategy_policy"]["artifact_source"] == "frozen_open_position_state"
    assert seen["allow_entries"] is False
    assert seen["artifact_payload"]["source"] == "frozen_open_position_state"
    assert seen["artifact_payload"]["strategies"] == [strategy]
    assert seen["ran"] is True


def test_live_pending_entry_recovery_ignores_missing_artifact_approval_and_stale_gates(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    state_file = tmp_path / "state.json"
    intent_ref = "2026-07-09T12:00:00+00:00"
    pending_qty = 10.0
    pending_client_id = PaperTradingBot._deterministic_client_order_id(
        strategy_id="missing_strategy",
        stage="entry",
        intent_ref=intent_ref,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=pending_qty,
        order_type=OrderType.MARKET,
        reduce_only=False,
    )
    state_file.write_text(
        json.dumps(
            {
                "equity": 1000.0,
                "open_positions": {},
                "inactive_strategies": [],
                "pending_order": {
                    "version": 1,
                    "strategy_id": "missing_strategy",
                    "stage": "entry",
                    "intent_ref": intent_ref,
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "qty": pending_qty,
                    "order_type": "market",
                    "reduce_only": False,
                    "client_id": pending_client_id,
                    "broker_account_fingerprint": TEST_ACCOUNT_FINGERPRINT,
                    "created_ts": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=tmp_path / "deleted-artifact.json",
        state_file=state_file,
        preflight_report=tmp_path / "stale-missing-preflight.json",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=tmp_path / "stale-missing-rehearsal.json",
    )

    class RecoveryBroker:
        name = "pending-entry-recovery"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT
        config = SimpleNamespace(live=True, market_type="futures")

        def __init__(self):
            self.position = Position("BTCUSDT", qty=pending_qty, avg_price=100.0)
            self.orders = []

        def get_position(self, symbol):
            return self.position

        def get_price(self, symbol):
            return 100.0

        def get_balance(self):
            return 1000.0

        def place_order(self, order):
            self.orders.append(order)
            assert order.reduce_only is True
            assert order.side == OrderSide.SELL
            assert order.qty == pytest.approx(pending_qty)
            self.position = Position("BTCUSDT")
            return Fill("BTCUSDT", OrderSide.SELL, pending_qty, 100.0, 0.0)

    broker = RecoveryBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)
    monkeypatch.setattr(
        "src.autopilot.runtime.assert_recent_preflight",
        lambda *args, **kwargs: pytest.fail("entry-only preflight must be skipped for recovery"),
    )
    monkeypatch.setattr(
        "src.autopilot.runtime.assert_recent_testnet_rehearsal",
        lambda *args, **kwargs: pytest.fail("entry-only rehearsal must be skipped for recovery"),
    )

    status = run_product_once(
        live_product,
        approval_ledger=tmp_path / "missing-approval-ledger.json",
    )

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert status["ok"] is False
    assert status["entries_allowed"] is False
    assert status["approval_gate"] == "management_only"
    assert status["preflight_gate"]["skipped"] is True
    assert status["testnet_rehearsal_gate"]["skipped"] is True
    assert status["strategy_policy"]["artifact_source"] == "durable_order_recovery_state"
    assert broker.position.is_flat
    assert len(broker.orders) == 1
    assert broker.orders[0].client_id.startswith("tb-rc-")
    assert persisted["pending_order"]["client_id"] == pending_client_id
    assert persisted["pending_entry_recovery"]["status"] == "recovery_close_filled_and_flat"


def test_live_product_requires_live_environment_before_broker(monkeypatch, tmp_path):
    for name in ("TRADING_LIVE", "EXCHANGE_API_KEY", "EXCHANGE_API_SECRET", "MAX_NOTIONAL_USD"):
        monkeypatch.delenv(name, raising=False)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        preflight_report=tmp_path / "preflight.json",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=tmp_path / "testnet.json",
    )
    preflight = write_preflight(live_product.preflight_report, live_product)
    write_testnet_rehearsal(live_product.testnet_rehearsal_report, preflight=preflight)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker",
        lambda product: pytest.fail("broker should not be built without live env"),
    )

    with pytest.raises(RuntimeError, match="TRADING_LIVE must be 1"):
        run_product_once(live_product, approval_ledger=ledger)


def test_live_active_income_requires_testnet_rehearsal_before_broker(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        require_testnet_rehearsal=True,
        preflight_report=tmp_path / "preflight.json",
        testnet_rehearsal_report=tmp_path / "missing_testnet.json",
        strategies_path=artifact,
    )
    write_preflight(live_product.preflight_report, live_product)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    monkeypatch.setattr(
        "src.autopilot.runtime.build_live_broker",
        lambda product: pytest.fail(
            "broker should not be built without testnet rehearsal evidence"
        ),
    )

    with pytest.raises(RuntimeError, match="testnet rehearsal report not found") as exc:
        run_product_once(live_product, approval_ledger=ledger)
    assert "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100" in str(exc.value)
    assert "make testnet-status" in str(exc.value)


def test_assert_live_environment_reports_safe_details(monkeypatch, tmp_path):
    set_live_env(monkeypatch)

    detail = assert_live_environment(product(tmp_path, execution_mode="live"))

    assert detail == {
        "ok": True,
        "exchange": "binanceusdm",
        "market_type": "futures",
        "testnet": False,
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
        "quote_asset": "USDT",
    }


def test_runtime_requires_fresh_preflight_after_api_key_change(tmp_path):
    live_product = product(tmp_path, execution_mode="live")
    recorded = exchange_environment_detail(live_product, testnet=False)
    current = dict(recorded)
    current["account_fingerprint"] = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="replacement-key",
        testnet=False,
    ).account_fingerprint

    with pytest.raises(RuntimeError, match="account_fingerprint"):
        _assert_current_environment_matches_preflight(
            live_product,
            current=current,
            recorded=recorded,
        )


def test_assert_live_environment_reports_spot_btc_accumulation_details(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("SPOT_EXCHANGE", "binance")
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
    )

    detail = assert_live_environment(live_product)

    assert detail == {
        "ok": True,
        "exchange": "binance",
        "market_type": "spot",
        "testnet": False,
        "account_fingerprint": ExchangeConfig(
            exchange="binance",
            market_type="spot",
            api_key="key",
            testnet=False,
        ).account_fingerprint,
        "max_notional_usd": 100.0,
        "max_fill_slippage_bps": 100.0,
        "quote_asset": "USDT",
    }


def test_assert_live_environment_rejects_unsafe_futures_leverage(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "10")

    with pytest.raises(RuntimeError, match="MAX_FUTURES_LEVERAGE"):
        assert_live_environment(product(tmp_path, execution_mode="live"))


def test_assert_live_environment_rejects_active_income_leverage_above_one(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "2")

    with pytest.raises(RuntimeError, match="active income futures must use MAX_FUTURES_LEVERAGE=1"):
        assert_live_environment(product(tmp_path, execution_mode="live"))


def test_assert_live_environment_rejects_non_positive_fill_slippage(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "0")

    with pytest.raises(RuntimeError, match="MAX_FILL_SLIPPAGE_BPS"):
        assert_live_environment(product(tmp_path, execution_mode="live"))


def test_assert_live_environment_rejects_blank_credentials(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("EXCHANGE_API_KEY", " ")
    monkeypatch.setenv("EXCHANGE_API_SECRET", " ")

    with pytest.raises(RuntimeError, match="EXCHANGE_API_KEY and EXCHANGE_API_SECRET are required"):
        assert_live_environment(product(tmp_path, execution_mode="live"))


def test_assert_live_environment_rejects_non_isolated_futures_margin(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("FUTURES_MARGIN_MODE", "cross")

    with pytest.raises(RuntimeError, match="FUTURES_MARGIN_MODE"):
        assert_live_environment(product(tmp_path, execution_mode="live"))


def test_assert_live_environment_rejects_non_binance_active_income_futures(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("FUTURES_EXCHANGE", "bybit")

    with pytest.raises(RuntimeError, match="Binance USDT futures"):
        assert_live_environment(product(tmp_path, execution_mode="live"))


def test_assert_live_environment_rejects_non_usdt_quote_asset(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("QUOTE_ASSET", "USDC")

    with pytest.raises(RuntimeError, match="quote asset must be USDT"):
        assert_live_environment(product(tmp_path, execution_mode="live"))


def test_assert_live_environment_rejects_non_binance_btc_accumulation_spot(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("SPOT_EXCHANGE", "kraken")
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
    )

    with pytest.raises(RuntimeError, match="Binance spot"):
        assert_live_environment(live_product)


def test_approved_live_active_income_uses_broker(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        require_testnet_rehearsal=True,
        preflight_report=tmp_path / "preflight.json",
        testnet_rehearsal_report=tmp_path / "testnet.json",
        strategies_path=artifact,
        state_file=tmp_path / "state.json",
        symbol="BTCUSDT",
    )
    preflight = write_preflight(live_product.preflight_report, live_product)
    write_testnet_rehearsal(live_product.testnet_rehearsal_report, preflight=preflight)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )
    approved_snapshot = json.loads(artifact.read_text(encoding="utf-8"))

    class FakeBroker:
        name = "fake-live"
        account_fingerprint = TEST_ACCOUNT_FINGERPRINT

    seen = {}

    class FakeBot:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.state = {"equity": 123.0, "open_positions": {}, "inactive_strategies": []}

        def run_cycle(self):
            seen["ran"] = True

    def replace_artifact_after_gates(product):
        changed = json.loads(json.dumps(approved_snapshot))
        changed["strategies"][0]["take_profit"] = 0.99
        artifact.write_text(json.dumps(changed), encoding="utf-8")
        return FakeBroker()

    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", replace_artifact_after_gates)
    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", FakeBot)

    status = run_product_once(live_product, approval_ledger=ledger)

    assert status["ok"] is True
    assert status["approval_gate"] == "approved"
    assert status["broker"] == "fake-live"
    assert seen["broker"].name == "fake-live"
    assert seen["symbol"] == "BTCUSDT"
    assert seen["market"] == "futures"
    assert seen["objective"] == "active_income"
    assert seen["base_asset"] == "USDT"
    assert seen["artifact_payload"] == approved_snapshot
    assert seen["artifact_payload"]["strategies"][0]["take_profit"] != 0.99
    assert callable(seen["pre_entry_gate"])
    assert seen["ran"] is True


@pytest.mark.parametrize("late_change", ["panic", "revoke"])
def test_live_pre_entry_gate_blocks_late_panic_or_approval_revocation(
    monkeypatch,
    tmp_path,
    late_change,
):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger_path = tmp_path / "approvals.json"
    control_path = tmp_path / "control.json"
    live_product = product(
        tmp_path,
        execution_mode="live",
        strategies_path=artifact,
        state_file=tmp_path / "state.json",
        preflight_report=tmp_path / "preflight.json",
        require_testnet_rehearsal=True,
        testnet_rehearsal_report=tmp_path / "testnet.json",
    )
    preflight = write_preflight(live_product.preflight_report, live_product)
    write_testnet_rehearsal(live_product.testnet_rehearsal_report, preflight=preflight)
    ledger = ApprovalLedger(ledger_path)
    fingerprint = ledger.approve(
        strategy,
        artifact_path=artifact,
        approved_by="human-reviewer",
        product=live_product,
    )
    config = AutopilotConfig(
        control_file=control_path,
        approval_ledger=ledger_path,
        products=[live_product],
    )

    class FakeBroker:
        name = "late-gate-broker"

        def __init__(self):
            self.orders = []

        def place_order(self, order):
            self.orders.append(order)
            raise AssertionError("late pre-entry gate allowed broker submission")

    broker = FakeBroker()

    class FakeBot:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.state = {"equity": 1000.0, "open_positions": {}, "inactive_strategies": []}
            self.cycle_errors = []

        def run_cycle(self):
            if late_change == "panic":
                control_path.write_text(
                    json.dumps(
                        {
                            "paused": True,
                            "pause_jobs": True,
                            "flatten_all": True,
                            "reason": "operator panic during feature fetch",
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                ledger.revoke(
                    fingerprint,
                    revoked_by="human-reviewer",
                    reason="operator revoked during feature fetch",
                )
            self.kwargs["pre_entry_gate"]()
            self.kwargs["broker"].place_order(object())

    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)
    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", FakeBot)

    status = run_product_once(
        live_product,
        approval_ledger=ledger_path,
        config=config,
    )

    assert status["ok"] is False
    assert broker.orders == []
    if late_change == "panic":
        assert "flatten is requested" in status["error"]
    else:
        assert "revoked/not approved" in status["error"]


def test_approved_live_btc_accumulation_uses_spot_broker(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = btc_strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        preflight_report=tmp_path / "preflight.json",
        strategies_path=artifact,
    )
    write_preflight(live_product.preflight_report, live_product)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=live_product
    )

    class FakeBroker:
        name = "fake-spot"

    seen = {}

    class FakeBot:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.state = {"equity": 1.0, "open_positions": {}, "inactive_strategies": []}

        def run_cycle(self):
            seen["ran"] = True

    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: FakeBroker())
    monkeypatch.setattr("src.autopilot.runtime.PaperTradingBot", FakeBot)

    status = run_product_once(live_product, approval_ledger=ledger)

    assert status["ok"] is True
    assert status["broker"] == "fake-spot"
    assert seen["broker"].name == "fake-spot"
    assert seen["market"] == "spot"
    assert seen["symbol"] == "BTCUSDT"
    assert seen["objective"] == "btc_accumulation"
    assert seen["base_asset"] == "BTC"
    assert seen["ran"] is True


def test_build_live_broker_uses_spot_config_for_btc_accumulation(monkeypatch, tmp_path):
    calls = {}

    class FakeExchangeConfig:
        exchange = "binance"
        market_type = "spot"
        quote_asset = "USDT"

        @classmethod
        def from_env(cls, market_type=None):
            calls["market_type"] = market_type
            cfg = cls()
            cfg.market_type = market_type
            return cfg

    class FakeCcxtBroker:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr("src.execution.config.ExchangeConfig", FakeExchangeConfig)
    monkeypatch.setattr("src.execution.ccxt_broker.CcxtBroker", FakeCcxtBroker)

    broker = build_live_broker(
        product(
            tmp_path,
            name="btc_accumulation",
            objective="btc_accumulation",
            base_asset="BTC",
            market="spot",
            execution_mode="live",
        )
    )

    assert calls["market_type"] == "spot"
    assert broker.config.market_type == "spot"


def test_build_live_broker_rejects_active_income_spot_routing(tmp_path):
    with pytest.raises(RuntimeError, match="Active income live execution must use futures"):
        build_live_broker(
            product(
                tmp_path,
                name="active_income",
                objective="active_income",
                base_asset="USDT",
                market="spot",
                execution_mode="live",
            )
        )


def test_build_live_broker_rejects_non_binance_active_income_futures(monkeypatch, tmp_path):
    class FakeExchangeConfig:
        exchange = "bybit"
        market_type = "futures"
        quote_asset = "USDT"

        @classmethod
        def from_env(cls, market_type=None):
            cfg = cls()
            cfg.market_type = market_type
            return cfg

    class FakeCcxtBroker:
        def __init__(self, config):
            raise AssertionError("broker should not be constructed when exchange policy fails")

    monkeypatch.setattr("src.execution.config.ExchangeConfig", FakeExchangeConfig)
    monkeypatch.setattr("src.execution.ccxt_broker.CcxtBroker", FakeCcxtBroker)

    with pytest.raises(RuntimeError, match="Binance USDT futures"):
        build_live_broker(product(tmp_path, execution_mode="live"))


def test_btc_accumulation_live_rejects_non_spot_market(tmp_path):
    artifact = tmp_path / "active.json"
    btc_strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    live_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="futures",
        execution_mode="live",
        require_preflight=False,
        strategies_path=artifact,
    )
    with pytest.raises(RuntimeError, match="spot"):
        run_product_once(live_product, approval_ledger=ledger)
