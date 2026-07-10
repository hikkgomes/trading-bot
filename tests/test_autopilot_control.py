import json

import pytest

from src.autopilot.control import (
    ControlConflictError,
    is_job_paused,
    is_product_paused,
    load_control,
    should_flatten_product,
    update_control,
)
from src.autopilot.control import (
    main as control_main,
)


def test_load_control_missing_file_uses_safe_defaults(tmp_path):
    control = load_control(tmp_path / "missing.json")

    assert control["paused"] is False
    assert control["pause_jobs"] is False
    assert control["paused_products"] == []


def test_stale_runtime_auto_clear_cannot_erase_new_operator_panic(tmp_path):
    path = tmp_path / "control.json"
    audit = tmp_path / "control_audit.jsonl"
    update_control(path, "flatten", name="active_income", reason="initial incident")
    runtime_snapshot = load_control(path)

    update_control(path, "panic", reason="new operator emergency", audit_path=audit)

    with pytest.raises(ControlConflictError, match="control changed"):
        update_control(
            path,
            "clear-flatten",
            name="active_income",
            reason="stale runtime completion",
            expected_control=runtime_snapshot,
            enforce_flatten_pause=True,
        )

    current = load_control(path)
    assert current["paused"] is True
    assert current["pause_jobs"] is True
    assert current["flatten_all"] is True
    assert current["flatten_products"] == ["active_income"]
    assert current["reason"] == "new operator emergency"


def test_control_update_rejects_symlinked_lock_file(tmp_path):
    path = tmp_path / "control.json"
    lock = tmp_path / ".control.json.lock"
    external = tmp_path / "external.lock"
    external.write_text("", encoding="utf-8")
    lock.symlink_to(external)

    with pytest.raises(ValueError, match="lock file must not be a symlink"):
        update_control(path, "panic", reason="incident")

    assert not path.exists()


def test_load_control_normalizes_operator_friendly_values(tmp_path):
    path = tmp_path / "control.json"
    path.write_text(
        json.dumps(
            {
                "paused": "true",
                "pause_jobs": "yes",
                "paused_products": "active_income",
                "paused_jobs": ["market_data_update"],
                "flatten_products": "active_income",
                "flatten_all": "0",
            }
        ),
        encoding="utf-8",
    )

    control = load_control(path)

    assert is_product_paused(control, "active_income") is True
    assert is_job_paused(control, "market_data_update") is True
    assert should_flatten_product(control, "active_income") is True
    assert control["flatten_all"] is False


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"paused": "treu"}, "paused must be a boolean"),
        ({"pause_jobs": "maybe"}, "pause_jobs must be a boolean"),
        ({"flatten_all": 2}, "flatten_all must be a boolean"),
        ({"paused": None}, "paused must be a boolean"),
        ({"paused": []}, "paused must be a boolean"),
        ({"pause_jobs": {"value": True}}, "pause_jobs must be a boolean"),
        ({"flatten_all": ["true"]}, "flatten_all must be a boolean"),
    ],
)
def test_load_control_malformed_boolean_fields_fail_closed(tmp_path, payload, error):
    path = tmp_path / "control.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    control = load_control(path)

    assert control["paused"] is True
    assert control["pause_jobs"] is True
    assert control["reason"] == "invalid_control_file"
    assert error in control["control_error"]


def test_load_control_malformed_json_fails_closed(tmp_path):
    path = tmp_path / "control.json"
    path.write_text("{", encoding="utf-8")

    control = load_control(path)

    assert control["paused"] is True
    assert control["pause_jobs"] is True
    assert control["reason"] == "invalid_control_file"
    assert "JSONDecodeError" in control["control_error"]


def test_load_control_symlink_fails_closed_without_trusting_target(tmp_path):
    path = tmp_path / "control.json"
    target = tmp_path / "external_control.json"
    target.write_text(json.dumps({"paused": False, "pause_jobs": False}), encoding="utf-8")
    path.symlink_to(target)

    control = load_control(path)

    assert control["paused"] is True
    assert control["pause_jobs"] is True
    assert control["reason"] == "invalid_control_file"
    assert "control file must not be a symlink" in control["control_error"]
    assert target.read_text(encoding="utf-8") == json.dumps({"paused": False, "pause_jobs": False})


def test_load_control_non_object_fails_closed(tmp_path):
    path = tmp_path / "control.json"
    path.write_text("[]", encoding="utf-8")

    control = load_control(path)

    assert control["paused"] is True
    assert control["pause_jobs"] is True
    assert "must be a JSON object" in control["control_error"]


@pytest.mark.parametrize(
    "payload,error",
    [
        ({"paused_products": ["active_income", 123]}, "paused_products must contain only non-empty strings"),
        ({"paused_jobs": {"name": "market_data_update"}}, "paused_jobs must be a string or list of strings"),
        ({"flatten_products": [" "]}, "flatten_products must contain only non-empty strings"),
    ],
)
def test_load_control_malformed_selectors_fail_closed(tmp_path, payload, error):
    path = tmp_path / "control.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    control = load_control(path)

    assert control["paused"] is True
    assert control["pause_jobs"] is True
    assert control["reason"] == "invalid_control_file"
    assert error in control["control_error"]


def write_config(path):
    path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "market_data_update_futures",
                        "command": ["python", "-m", "src.update_candles"],
                        "cadence_seconds": 60,
                    }
                ],
                "products": [
                    {
                        "name": "active_income",
                        "objective": "active_income",
                        "base_asset": "USDT",
                        "market": "futures",
                        "strategies_path": "outputs/active.json",
                        "state_file": "runtime/state.json",
                        "trade_log": "runtime/trades.csv",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["pause-product", "active-incme"], "unknown product 'active-incme'"),
        (["flatten", "active-incme"], "unknown product 'active-incme'"),
        (["pause-job", "market-data-update"], "unknown job 'market-data-update'"),
    ],
)
def test_control_cli_rejects_unknown_configured_selectors_before_writing(tmp_path, args, message):
    path = tmp_path / "control.json"
    config = tmp_path / "autopilot.json"
    write_config(config)

    with pytest.raises(SystemExit, match=message):
        control_main(["--control", str(path), "--config", str(config), *args])

    assert not path.exists()


def test_control_status_with_config_reports_selector_validation(capsys, tmp_path):
    path = tmp_path / "control.json"
    config = tmp_path / "autopilot.json"
    write_config(config)
    path.write_text(
        json.dumps(
            {
                "paused_products": ["active-incme"],
                "flatten_products": ["active-incme"],
                "paused_jobs": ["market-data-update"],
            }
        ),
        encoding="utf-8",
    )

    control_main(["--control", str(path), "--config", str(config), "status"])

    printed = json.loads(capsys.readouterr().out)
    assert printed["selector_validation"] == {
        "ok": False,
        "unknown_selectors": {
            "flatten_products": ["active-incme"],
            "paused_jobs": ["market-data-update"],
            "paused_products": ["active-incme"],
        },
    }
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "paused_products": ["active-incme"],
        "flatten_products": ["active-incme"],
        "paused_jobs": ["market-data-update"],
    }


def test_control_status_with_config_reports_valid_selectors(capsys, tmp_path):
    path = tmp_path / "control.json"
    config = tmp_path / "autopilot.json"
    write_config(config)
    path.write_text(
        json.dumps({"paused_products": ["active_income"], "paused_jobs": ["market_data_update_futures"]}),
        encoding="utf-8",
    )

    control_main(["--control", str(path), "--config", str(config), "status"])

    printed = json.loads(capsys.readouterr().out)
    assert printed["selector_validation"] == {"ok": True}


def test_control_cli_updates_pause_job_product_and_flatten(tmp_path):
    path = tmp_path / "control.json"

    control_main(["--control", str(path), "pause", "--reason", "maintenance"])
    control = load_control(path)
    assert control["paused"] is True
    assert control["reason"] == "maintenance"

    control_main(["--control", str(path), "pause-product", "active_income"])
    control_main(["--control", str(path), "pause-job", "market_data_update"])
    control_main(["--control", str(path), "flatten", "active_income", "--reason", "risk review"])
    control = load_control(path)
    assert is_product_paused(control, "active_income") is True
    assert is_job_paused(control, "market_data_update") is True
    assert should_flatten_product(control, "active_income") is True
    assert control["reason"] == "risk review"

    control_main(["--control", str(path), "resume-product", "active_income"])
    control_main(["--control", str(path), "resume-job", "market_data_update"])
    control_main(["--control", str(path), "clear-flatten", "active_income"])
    control_main(["--control", str(path), "resume"])
    control = load_control(path)
    assert control["paused"] is False
    assert control["paused_products"] == []
    assert control["paused_jobs"] == []
    assert control["flatten_products"] == []


def test_control_cli_writes_jsonl_audit_for_mutations(tmp_path):
    path = tmp_path / "control.json"
    audit = tmp_path / "control_audit.jsonl"

    control_main(
        [
            "--control",
            str(path),
            "--audit",
            str(audit),
            "--operator",
            "henrique",
            "pause",
            "--reason",
            "maintenance",
        ]
    )
    control_main(
        [
            "--control",
            str(path),
            "--audit",
            str(audit),
            "--operator",
            "henrique",
            "flatten",
            "active_income",
            "--reason",
            "risk review",
        ]
    )

    events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]

    assert [event["command"] for event in events] == ["pause", "flatten"]
    assert events[0]["actor"] == "henrique"
    assert events[0]["reason"] == "maintenance"
    assert events[0]["before"]["paused"] is False
    assert events[0]["after"]["paused"] is True
    assert events[1]["name"] == "active_income"
    assert events[1]["before"]["flatten_products"] == []
    assert events[1]["after"]["flatten_products"] == ["active_income"]


def test_control_cli_panic_pauses_everything_and_requests_flatten_all(tmp_path):
    path = tmp_path / "control.json"
    audit = tmp_path / "control_audit.jsonl"

    control_main(
        [
            "--control",
            str(path),
            "--audit",
            str(audit),
            "--operator",
            "henrique",
            "panic",
            "--reason",
            "exchange incident",
        ]
    )

    control = load_control(path)
    events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]

    assert control["paused"] is True
    assert control["pause_jobs"] is True
    assert control["flatten_all"] is True
    assert control["reason"] == "exchange incident"
    assert should_flatten_product(control, "active_income") is True
    assert events[0]["command"] == "panic"
    assert events[0]["after"]["paused"] is True
    assert events[0]["after"]["pause_jobs"] is True
    assert events[0]["after"]["flatten_all"] is True


def test_control_cli_flatten_keeps_product_paused_after_request_is_cleared(tmp_path):
    path = tmp_path / "control.json"

    control_main(["--control", str(path), "flatten", "active_income", "--reason", "risk incident"])
    control_main(["--control", str(path), "clear-flatten", "active_income"])

    control = load_control(path)
    assert control["flatten_products"] == []
    assert control["paused_products"] == ["active_income"]
    assert is_product_paused(control, "active_income") is True


def test_control_cli_flatten_all_keeps_all_products_paused_after_request_is_cleared(tmp_path):
    path = tmp_path / "control.json"

    control_main(["--control", str(path), "flatten-all", "--reason", "risk incident"])
    control_main(["--control", str(path), "clear-all-flatten"])

    control = load_control(path)
    assert control["flatten_all"] is False
    assert control["paused"] is True


def test_control_cli_applies_mutation_when_audit_write_fails(monkeypatch, tmp_path, capsys):
    path = tmp_path / "control.json"
    audit = tmp_path / "control_audit.jsonl"

    def fail_append(path, payload):
        raise OSError("audit disk full")

    monkeypatch.setattr("src.autopilot.control.append_json_line", fail_append)

    control_main(
        [
            "--control",
            str(path),
            "--audit",
            str(audit),
            "pause",
            "--reason",
            "emergency",
        ]
    )

    control = load_control(path)
    printed = json.loads(capsys.readouterr().out)
    assert control["paused"] is True
    assert control["reason"] == "emergency"
    assert printed["paused"] is True
    assert printed["audit_error"] == "OSError: audit disk full"


def test_control_cli_applies_mutation_when_audit_path_is_symlink(tmp_path, capsys):
    path = tmp_path / "control.json"
    audit = tmp_path / "control_audit.jsonl"
    target = tmp_path / "external_audit.jsonl"
    target.write_text('{"existing": true}\n', encoding="utf-8")
    audit.symlink_to(target)

    control_main(
        [
            "--control",
            str(path),
            "--audit",
            str(audit),
            "pause",
            "--reason",
            "emergency",
        ]
    )

    control = load_control(path)
    printed = json.loads(capsys.readouterr().out)
    assert control["paused"] is True
    assert control["reason"] == "emergency"
    assert printed["paused"] is True
    assert "jsonl path must not be a symlink" in printed["audit_error"]
    assert audit.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_control_cli_surfaces_recovery_when_repairing_malformed_control(tmp_path, capsys):
    path = tmp_path / "control.json"
    audit = tmp_path / "control_audit.jsonl"
    path.write_text("{", encoding="utf-8")

    control_main(
        [
            "--control",
            str(path),
            "--audit",
            str(audit),
            "--operator",
            "henrique",
            "clear",
            "--reason",
            "repair malformed control",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]

    assert printed["paused"] is False
    assert printed["reason"] == "repair malformed control"
    assert "JSONDecodeError" in printed["recovered_control_error"]
    assert "recovered_control_error" not in persisted
    assert events[0]["before"]["recovered_control_error"] == printed["recovered_control_error"]
    assert events[0]["after"]["recovered_control_error"] == printed["recovered_control_error"]


def test_control_cli_surfaces_recovery_when_repairing_invalid_control_shape(tmp_path, capsys):
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"paused_products": [" "]}), encoding="utf-8")

    control_main(["--control", str(path), "pause", "--reason", "safe pause"])

    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert printed["paused"] is True
    assert "paused_products must contain only non-empty strings" in printed["recovered_control_error"]
    assert persisted == {
        "flatten_all": False,
        "flatten_products": [],
        "pause_jobs": False,
        "paused": True,
        "paused_jobs": [],
        "paused_products": [],
        "reason": "safe pause",
    }


def test_control_cli_refuses_symlink_control_without_touching_target(tmp_path):
    path = tmp_path / "control.json"
    target = tmp_path / "external_control.json"
    original = json.dumps({"paused": False, "pause_jobs": False})
    target.write_text(original, encoding="utf-8")
    path.symlink_to(target)

    with pytest.raises(ValueError, match="control file must not be a symlink"):
        control_main(["--control", str(path), "pause", "--reason", "maintenance"])

    assert path.is_symlink()
    assert target.read_text(encoding="utf-8") == original


def test_control_cli_clear_recovers_malformed_file(tmp_path):
    path = tmp_path / "control.json"
    path.write_text("{", encoding="utf-8")

    assert load_control(path)["paused"] is True
    control_main(["--control", str(path), "clear", "--reason", "repaired"])
    control = load_control(path)

    assert control["paused"] is False
    assert control["pause_jobs"] is False
    assert control["reason"] == "repaired"
    assert "control_error" not in control


def test_control_cli_recovers_malformed_selectors(tmp_path):
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"paused_products": [123]}), encoding="utf-8")

    assert load_control(path)["paused"] is True
    control_main(["--control", str(path), "pause-product", "active_income", "--reason", "repair"])
    control = load_control(path)

    assert control["paused"] is False
    assert control["paused_products"] == ["active_income"]
    assert control["reason"] == "repair"


def test_control_cli_clear_recovers_malformed_boolean_field(tmp_path):
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"paused": "treu"}), encoding="utf-8")

    assert load_control(path)["paused"] is True
    control_main(["--control", str(path), "clear", "--reason", "repaired"])
    control = load_control(path)

    assert control["paused"] is False
    assert control["pause_jobs"] is False
    assert control["reason"] == "repaired"
    assert "control_error" not in control
