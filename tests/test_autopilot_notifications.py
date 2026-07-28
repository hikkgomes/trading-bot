import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.autopilot.notifications import (
    _post_webhook,
    alert_fingerprint,
    emit_alert,
    failure_detail,
    promotion_warning_detail,
    readiness_warning_detail,
    required_testnet_rehearsal_warning_detail,
    research_handoff_warning_detail,
    research_progress_warning_detail,
    wait_for_remote_alerts,
)


def test_emit_alert_writes_jsonl_and_cooldown(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    detail = {"products": [{"name": "active_income", "error": "failed"}]}

    first = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=100.0,
    )
    second = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=120.0,
    )
    third = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=161.0,
    )

    assert first["sent"] is True
    assert second == {
        "sent": False,
        "reason": "cooldown",
        "fingerprint": first["fingerprint"],
    }
    assert third["sent"] is True
    lines = alert_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["detail"] == detail


def test_emit_alert_recovers_from_malformed_alert_state(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    state_file.write_text("{", encoding="utf-8")

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail={"products": [{"name": "active_income", "error": "failed"}]},
        cooldown_seconds=60,
        now=100.0,
    )

    assert result["sent"] is True
    alert = json.loads(alert_file.read_text(encoding="utf-8").splitlines()[0])
    assert alert["alert_state_recovered"]["path"] == str(state_file)
    assert "JSONDecodeError" in alert["alert_state_recovered"]["error"]
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["alerts"][result["fingerprint"]]["last_sent_ts"] == 100.0


def test_emit_alert_ignores_malformed_cooldown_entry(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    detail = {"products": [{"name": "active_income", "error": "failed"}]}
    fingerprint = alert_fingerprint("error", "cycle failed", detail)
    state_file.write_text(
        json.dumps({"version": 1, "alerts": {fingerprint: {"last_sent_ts": "not-a-number"}}}),
        encoding="utf-8",
    )

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=100.0,
    )

    assert result == {
        "sent": True,
        "fingerprint": fingerprint,
        "remote_delivery": {"status": "not_configured"},
    }
    assert len(alert_file.read_text(encoding="utf-8").splitlines()) == 1
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["alerts"][fingerprint]["last_sent_ts"] == 100.0


def test_emit_alert_ignores_future_cooldown_entry_and_records_recovery(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    detail = {"products": [{"name": "active_income", "error": "failed"}]}
    fingerprint = alert_fingerprint("error", "cycle failed", detail)
    state_file.write_text(
        json.dumps({"version": 1, "alerts": {fingerprint: {"last_sent_ts": 9999.0}}}),
        encoding="utf-8",
    )

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=100.0,
    )

    assert result == {
        "sent": True,
        "fingerprint": fingerprint,
        "remote_delivery": {"status": "not_configured"},
    }
    alert = json.loads(alert_file.read_text(encoding="utf-8").splitlines()[0])
    assert alert["alert_state_entry_recovered"] == {
        "fingerprint": fingerprint,
        "field": "last_sent_ts",
        "value": 9999.0,
        "reason": "future_timestamp",
    }
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["alerts"][fingerprint]["last_sent_ts"] == 100.0


@pytest.mark.parametrize(
    ("now", "cooldown_seconds", "message"),
    [
        (float("nan"), 60, "alert now timestamp must be finite and non-negative"),
        (-1.0, 60, "alert now timestamp must be finite and non-negative"),
        ("bad", 60, "alert now timestamp must be numeric"),
        (100.0, float("inf"), "alert cooldown_seconds must be finite and non-negative"),
        (100.0, -1, "alert cooldown_seconds must be finite and non-negative"),
        (100.0, "bad", "alert cooldown_seconds must be numeric"),
    ],
)
def test_emit_alert_rejects_invalid_timing_inputs(tmp_path, now, cooldown_seconds, message):
    with pytest.raises(ValueError, match=message):
        emit_alert(
            alert_file=tmp_path / "alerts.jsonl",
            state_file=tmp_path / "alert_state.json",
            severity="error",
            title="cycle failed",
            detail={"products": [{"name": "active_income", "error": "failed"}]},
            cooldown_seconds=cooldown_seconds,
            now=now,
        )


def test_alert_fingerprint_ignores_volatile_operational_fields():
    first_detail = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": "runtime",
                "free_bytes": 300,
                "total_bytes": 2000,
                "used_bytes": 1700,
                "min_free_bytes": 500,
            },
            {
                "product": "active_income",
                "approved_review_failed": 1,
                "generated_at": "2026-01-01T00:00:00+00:00",
            },
        ]
    }
    second_detail = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": "runtime",
                "free_bytes": 280,
                "total_bytes": 2000,
                "used_bytes": 1720,
                "min_free_bytes": 500,
            },
            {
                "product": "active_income",
                "approved_review_failed": 1,
                "generated_at": "2026-01-01T00:01:00+00:00",
            },
        ]
    }
    changed_threshold = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": "runtime",
                "free_bytes": 280,
                "min_free_bytes": 1000,
            }
        ]
    }

    first = alert_fingerprint("warning", "autopilot readiness warnings", first_detail)

    assert alert_fingerprint("warning", "autopilot readiness warnings", second_detail) == first
    assert alert_fingerprint("warning", "autopilot readiness warnings", changed_threshold) != first


def test_emit_alert_cooldown_uses_stable_fingerprint_but_persists_full_detail(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    first_detail = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "free_bytes": 300,
                "min_free_bytes": 500,
            }
        ]
    }
    second_detail = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "free_bytes": 280,
                "min_free_bytes": 500,
            }
        ]
    }

    first = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="warning",
        title="autopilot readiness warnings",
        detail=first_detail,
        cooldown_seconds=60,
        now=100.0,
    )
    second = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="warning",
        title="autopilot readiness warnings",
        detail=second_detail,
        cooldown_seconds=60,
        now=120.0,
    )

    assert second == {"sent": False, "reason": "cooldown", "fingerprint": first["fingerprint"]}
    lines = alert_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["detail"] == first_detail


def test_emit_alert_explicit_dedupe_key_ignores_changing_detail(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"

    first = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="critical",
        title="healthcheck failed",
        detail={"issues": [{"code": "job_failed", "attempt": 1}]},
        dedupe_key="healthcheck-failed:job_failed",
        cooldown_seconds=60,
        now=100.0,
    )
    second = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="critical",
        title="healthcheck failed",
        detail={"issues": [{"code": "job_failed", "attempt": 2}]},
        dedupe_key="healthcheck-failed:job_failed",
        cooldown_seconds=60,
        now=120.0,
    )

    assert first["sent"] is True
    assert second == {
        "sent": False,
        "reason": "cooldown",
        "fingerprint": first["fingerprint"],
    }
    assert len(alert_file.read_text(encoding="utf-8").splitlines()) == 1


def test_emit_alert_audits_configured_telegram_that_resolves_to_no_client(
    tmp_path,
    monkeypatch,
):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    monkeypatch.setenv("AUTOPILOT_TELEGRAM_BOT_TOKEN", "configured-token")
    monkeypatch.setattr(
        "src.autopilot.notifications.send_alert_from_environment",
        lambda *_args, **_kwargs: None,
    )

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="info",
        title="position changed",
        detail={"event_id": "event-1"},
        now=1_000.0,
    )

    assert result["remote_delivery"] == {"status": "queued"}
    assert wait_for_remote_alerts()
    delivery = json.loads(alert_file.read_text(encoding="utf-8").splitlines()[1])
    assert delivery["schema"] == "autopilot.alert_delivery/v1"
    assert delivery["telegram"]["ok"] is False
    assert "resolved to no delivery client" in delivery["telegram"]["error"]


def test_emit_alert_records_webhook_failure_without_raising(tmp_path, monkeypatch):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    monkeypatch.setenv("AUTOPILOT_WEBHOOK_URL", "https://alerts.invalid/hook")

    def fail_webhook(url, payload):
        raise RuntimeError(f"cannot reach {url}")

    monkeypatch.setattr("src.autopilot.notifications._post_webhook", fail_webhook)

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail={"products": [{"name": "active_income", "error": "failed"}]},
        cooldown_seconds=60,
        now=100.0,
    )
    second = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail={"products": [{"name": "active_income", "error": "failed"}]},
        cooldown_seconds=60,
        now=120.0,
    )

    assert result["sent"] is True
    assert result["remote_delivery"] == {"status": "queued"}
    assert second["reason"] == "cooldown"
    assert wait_for_remote_alerts()
    lines = alert_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    persisted = json.loads(lines[1])
    assert persisted["schema"] == "autopilot.alert_delivery/v1"
    assert persisted["webhook"]["ok"] is False
    assert "cannot reach" in persisted["webhook"]["error"]
    assert (
        json.loads(state_file.read_text(encoding="utf-8"))["alerts"][result["fingerprint"]][
            "last_sent_ts"
        ]
        == 100.0
    )


def test_emit_alert_returns_webhook_status(tmp_path, monkeypatch):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    monkeypatch.setenv("AUTOPILOT_WEBHOOK_URL", "https://alerts.invalid/hook")
    monkeypatch.setattr(
        "src.autopilot.notifications._post_webhook",
        lambda url, payload: {"status_code": 503, "ok": False},
    )

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="warning",
        title="readiness warning",
        detail={"warnings": [{"name": "market data"}]},
        cooldown_seconds=60,
        now=100.0,
    )

    assert result["sent"] is True
    assert result["remote_delivery"] == {"status": "queued"}
    assert wait_for_remote_alerts()
    delivery = json.loads(alert_file.read_text(encoding="utf-8").splitlines()[1])
    assert delivery["webhook"] == {
        "status_code": 503,
        "ok": False,
    }


def test_emit_alert_loads_validated_operations_only_settings_file(tmp_path, monkeypatch):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    settings_file = tmp_path / "alerts.env"
    settings_file.write_text(
        "AUTOPILOT_WEBHOOK_URL=https://private-alerts.invalid/hook\n",
        encoding="utf-8",
    )
    settings_file.chmod(0o600)
    monkeypatch.delenv("AUTOPILOT_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("AUTOPILOT_ALERT_SETTINGS_FILE", str(settings_file))
    calls = []
    monkeypatch.setattr(
        "src.autopilot.notifications._post_webhook",
        lambda url, payload: calls.append((url, payload)) or {"status_code": 204, "ok": True},
    )

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="critical",
        title="watchdog failed",
        detail={"issues": 1},
        cooldown_seconds=60,
        now=100.0,
    )

    assert result["remote_delivery"] == {"status": "queued"}
    assert wait_for_remote_alerts()
    assert calls[0][0] == "https://private-alerts.invalid/hook"


def test_emit_alert_keeps_local_alert_when_operations_settings_are_forbidden(
    tmp_path,
    monkeypatch,
):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    settings_file = tmp_path / "alerts.env"
    settings_file.write_text("TRADING_LIVE=1\n", encoding="utf-8")
    settings_file.chmod(0o600)
    monkeypatch.setenv("AUTOPILOT_ALERT_SETTINGS_FILE", str(settings_file))

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="critical",
        title="watchdog failed",
        detail={"issues": 1},
        cooldown_seconds=60,
        now=100.0,
    )

    assert result["sent"] is True
    assert result["remote_delivery"]["status"] == "invalid_settings"
    assert "forbidden key 'TRADING_LIVE'" in result["remote_delivery"]["error"]
    assert len(alert_file.read_text(encoding="utf-8").splitlines()) == 1


def test_emit_alert_does_not_block_supervision_on_slow_remote_delivery(
    tmp_path,
    monkeypatch,
):
    assert wait_for_remote_alerts()
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setenv("AUTOPILOT_WEBHOOK_URL", "https://alerts.invalid/hook")

    def slow_webhook(url, payload):
        started.set()
        assert release.wait(timeout=2.0)
        return {"status_code": 204, "ok": True}

    monkeypatch.setattr("src.autopilot.notifications._post_webhook", slow_webhook)

    before = time.perf_counter()
    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="critical",
        title="risk supervision warning",
        detail={"product": "active_income"},
        cooldown_seconds=60,
        now=100.0,
    )
    elapsed = time.perf_counter() - before

    assert result["remote_delivery"] == {"status": "queued"}
    assert elapsed < 0.25
    assert started.wait(timeout=1.0)
    release.set()
    assert wait_for_remote_alerts()
    records = [json.loads(line) for line in alert_file.read_text(encoding="utf-8").splitlines()]
    assert records[0]["schema"] == "autopilot.alert/v1"
    assert records[1]["schema"] == "autopilot.alert_delivery/v1"
    assert records[1]["webhook"] == {"status_code": 204, "ok": True}


def test_emit_alert_sanitizes_webhook_payload_without_redacting_private_local_log(
    tmp_path, monkeypatch
):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    captured = {}
    monkeypatch.setenv("AUTOPILOT_WEBHOOK_URL", "https://alerts.invalid/hook")

    def capture_webhook(url, payload):
        captured.update(url=url, payload=payload)
        return {"status_code": 204, "ok": True}

    monkeypatch.setattr("src.autopilot.notifications._post_webhook", capture_webhook)
    detail = {
        "api_key": "private-key",
        "error": "GET https://exchange.test/order?signature=private-signature",
        "failed_holdout": {"total_return": -0.5},
        "safe": "visible",
    }

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=100.0,
    )

    assert result["remote_delivery"] == {"status": "queued"}
    assert wait_for_remote_alerts()
    rendered_remote = json.dumps(captured["payload"], sort_keys=True)
    assert captured["url"] == "https://alerts.invalid/hook"
    assert "private-key" not in rendered_remote
    assert "private-signature" not in rendered_remote
    assert "failed_holdout" not in rendered_remote
    assert captured["payload"]["detail"]["safe"] == "visible"
    persisted = json.loads(alert_file.read_text(encoding="utf-8").splitlines()[0])
    assert persisted["detail"] == detail


@pytest.mark.parametrize(
    "url",
    [
        "http://alerts.example/hook",
        "ftp://alerts.example/hook",
        "https://user:password@alerts.example/hook",
        "not-a-url",
    ],
)
def test_post_webhook_rejects_insecure_or_credentialed_urls(url):
    with pytest.raises(ValueError, match="webhook URL"):
        _post_webhook(url, {"safe": True})


def test_post_webhook_allows_explicit_loopback_http_for_local_testing(monkeypatch):
    class Response:
        status_code = 204

    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("requests.post", post)

    result = _post_webhook("http://127.0.0.1:8080/hook", {"safe": True})

    assert result == {"status_code": 204, "ok": True}
    assert captured["timeout"] == 10
    assert captured["allow_redirects"] is False


def test_post_webhook_does_not_follow_https_redirects(monkeypatch):
    class Response:
        status_code = 307

    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.post", post)

    result = _post_webhook("https://alerts.example/hook", {"safe": True})

    assert result == {"status_code": 307, "ok": False}
    assert len(calls) == 1
    assert calls[0][1]["allow_redirects"] is False


def test_post_webhook_network_error_does_not_echo_secret_url(monkeypatch):
    def post(url, **kwargs):
        raise OSError(f"failed URL {url}")

    monkeypatch.setattr("requests.post", post)

    with pytest.raises(RuntimeError, match="webhook request failed: OSError") as exc_info:
        _post_webhook("https://alerts.example/private-hook-token", {"safe": True})

    assert "private-hook-token" not in str(exc_info.value)


def test_emit_alert_serializes_concurrent_cooldown_state_updates(tmp_path, monkeypatch):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    from src.autopilot import notifications

    original_load = notifications._load_state

    def slow_load(path):
        payload = original_load(path)
        time.sleep(0.02)
        return payload

    monkeypatch.setattr(notifications, "_load_state", slow_load)

    def send(index):
        return emit_alert(
            alert_file=alert_file,
            state_file=state_file,
            severity="warning",
            title=f"warning-{index}",
            detail={"index": index},
            cooldown_seconds=60,
            now=100.0,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(send, range(8)))

    assert all(item["sent"] is True for item in results)
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(state["alerts"]) == 8
    assert len(alert_file.read_text(encoding="utf-8").splitlines()) == 8


def test_emit_alert_reports_state_write_failure_after_alert_is_written(tmp_path, monkeypatch):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"

    def fail_save(path, payload):
        raise OSError(f"cannot write {path}")

    monkeypatch.setattr("src.autopilot.notifications.write_json_atomic", fail_save)

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail={"products": [{"name": "active_income", "error": "failed"}]},
        cooldown_seconds=60,
        now=100.0,
    )

    assert result["sent"] is True
    assert result["state_error"] == f"OSError: cannot write {state_file}"
    lines = alert_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["fingerprint"] == result["fingerprint"]


def test_emit_alert_rejects_symlink_alert_log_without_saving_state(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    target = tmp_path / "external_alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    target.write_text('{"existing": true}\n', encoding="utf-8")
    alert_file.symlink_to(target)

    with pytest.raises(ValueError, match="jsonl path must not be a symlink"):
        emit_alert(
            alert_file=alert_file,
            state_file=state_file,
            severity="critical",
            title="autopilot failed",
            detail={"ok": False},
            cooldown_seconds=0,
            now=100.0,
        )

    assert alert_file.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"existing": true}\n'
    assert not state_file.exists()


def test_failure_detail_summarizes_failed_products_and_jobs():
    report = {
        "control_error": "unknown control selectors",
        "unknown_control_selectors": {"paused_jobs": ["typo"]},
        "control": {
            "reason": "unknown_control_selector",
            "paused": True,
            "pause_jobs": True,
        },
        "products": [
            {
                "ok": False,
                "product": {
                    "name": "active_income",
                    "execution_mode": "paper",
                    "market": "futures",
                },
                "error": "missing artifact",
            },
            {"ok": True, "product": {"name": "btc_accumulation"}},
            {
                "ok": False,
                "product": {"name": "scalp", "execution_mode": "live", "market": "futures"},
                "action": "flatten",
                "broker": "fake-live",
                "close_error": "RuntimeError: exchange timeout",
                "fill": {
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "qty": 0.4,
                    "price": 125.0,
                    "fee": 0.02,
                },
                "spot_step_aside": {
                    "strategy_id": "btc_step_aside",
                    "quote_value": 50.0,
                    "requested_qty": 0.4,
                },
                "local_state": {"path": "runtime/btc_state.json", "recovered": False},
                "position_before": {"symbol": "BTCUSDT", "qty": 0.5},
                "position_after_error": "RuntimeError: readback timeout",
                "position_after_attempt": {"symbol": "BTCUSDT", "qty": 0.5, "is_flat": False},
                "cycle_errors": [{"strategy_id": "s1", "error": "network down"}],
                "state_errors": [{"field": "open_positions", "error": "expected object"}],
            },
        ],
        "jobs": [
            {"ok": False, "name": "sweep", "returncode": 2, "stderr_tail": "bad"},
            {"ok": True, "name": "smoke"},
        ],
        "job_config_errors": ["jobs[3]: malformed optional research job"],
        "data_update": {"ok": True},
    }

    assert failure_detail(report) == {
        "control": {
            "error": "unknown control selectors",
            "reason": "unknown_control_selector",
            "paused": True,
            "pause_jobs": True,
            "unknown_selectors": {"paused_jobs": ["typo"]},
        },
        "products": [
            {
                "name": "active_income",
                "mode": "paper",
                "market": "futures",
                "action": None,
                "error": "missing artifact",
                "cycle_errors": [],
                "state_errors": [],
            },
            {
                "name": "scalp",
                "mode": "live",
                "market": "futures",
                "action": "flatten",
                "error": "RuntimeError: exchange timeout",
                "cycle_errors": [{"strategy_id": "s1", "error": "network down"}],
                "state_errors": [{"field": "open_positions", "error": "expected object"}],
                "broker": "fake-live",
                "position_before": {"symbol": "BTCUSDT", "qty": 0.5},
                "position_after_error": "RuntimeError: readback timeout",
                "position_after_attempt": {"symbol": "BTCUSDT", "qty": 0.5, "is_flat": False},
                "fill": {
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "qty": 0.4,
                    "price": 125.0,
                    "fee": 0.02,
                },
                "spot_step_aside": {
                    "strategy_id": "btc_step_aside",
                    "quote_value": 50.0,
                    "requested_qty": 0.4,
                },
                "local_state": {"path": "runtime/btc_state.json", "recovered": False},
            },
        ],
        "jobs": [{"name": "sweep", "error": "bad"}],
        "job_config_errors": ["jobs[3]: malformed optional research job"],
        "data_update": {"ok": True},
    }


def test_failure_detail_ignores_malformed_sections():
    assert failure_detail(
        {"products": {"bad": "shape"}, "jobs": None, "data_update": {"ok": False}}
    ) == {
        "products": [],
        "jobs": [],
        "job_config_errors": [],
        "data_update": {"ok": False},
    }
    assert failure_detail(
        {
            "products": [
                "bad-product",
                {"ok": False, "product": {"name": "active_income"}, "error": "failed"},
            ],
            "jobs": ["bad-job", {"ok": False, "name": "sweep", "returncode": 2}],
        }
    ) == {
        "products": [
            {
                "name": "active_income",
                "mode": None,
                "market": None,
                "action": None,
                "error": "failed",
                "cycle_errors": [],
                "state_errors": [],
            }
        ],
        "jobs": [{"name": "sweep", "error": 2}],
        "job_config_errors": [],
        "data_update": None,
    }


def test_readiness_warning_detail_summarizes_readiness_warnings():
    report = {
        "checks": [
            {
                "name": "market data seed and freshness",
                "level": "warning",
                "ok": False,
                "detail": {
                    "futures": {"ok": True, "reason": "fresh", "path": "futures.parquet"},
                    "spot": {"ok": False, "reason": "missing_seed_dataset", "path": "spot.parquet"},
                },
            },
            {
                "name": "indicator feature readiness",
                "level": "warning",
                "ok": False,
                "detail": {
                    "spot": {
                        "ok": False,
                        "timeframes": {
                            "1h": {
                                "ok": False,
                                "reason": "missing_indicator_dataset",
                                "missing_features": ["volume_z_20"],
                            },
                            "4h": {"ok": True, "missing_features": []},
                        },
                    },
                },
            },
            {
                "name": "environment file present",
                "level": "warning",
                "ok": False,
                "detail": ".env is optional for paper mode",
            },
            {
                "name": "runtime filesystem free space",
                "level": "warning",
                "ok": False,
                "detail": {
                    "path": "runtime/status.json",
                    "checked_path": "runtime",
                    "free_bytes": 300,
                    "min_free_bytes": 500,
                },
            },
            {
                "name": "strategy framework smoke",
                "level": "warning",
                "ok": False,
                "detail": {
                    "reason": "missing_report",
                    "path": "runtime/strategy_framework_smoke.json",
                    "scenario_count": None,
                    "failures": [],
                },
            },
            {
                "name": "approval ledger revocation audit",
                "level": "warning",
                "ok": False,
                "detail": {
                    "invalid_revocation_count": 1,
                    "entries": [
                        {
                            "fingerprint": "sha256:bad",
                            "strategy_id": "s1",
                            "reasons": ["missing_revocation_reason"],
                        }
                    ],
                },
            },
        ]
    }

    assert readiness_warning_detail(report) == {
        "warnings": [
            {
                "name": "market data seed and freshness",
                "markets": {
                    "spot": {
                        "reason": "missing_seed_dataset",
                        "path": "spot.parquet",
                    }
                },
            },
            {
                "name": "indicator feature readiness",
                "missing": {
                    "spot": {
                        "1h": {
                            "reason": "missing_indicator_dataset",
                            "missing_features": ["volume_z_20"],
                        }
                    }
                },
            },
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": "runtime",
                "free_bytes": 300,
                "min_free_bytes": 500,
                "reason": None,
            },
            {
                "name": "strategy framework smoke",
                "reason": "missing_report",
                "path": "runtime/strategy_framework_smoke.json",
                "scenario_count": None,
                "failures": [],
            },
            {
                "name": "approval ledger revocation audit",
                "entries": [
                    {
                        "fingerprint": "sha256:bad",
                        "strategy_id": "s1",
                        "reasons": ["missing_revocation_reason"],
                    }
                ],
                "invalid_revocation_count": 1,
            },
        ]
    }


@pytest.mark.parametrize(
    ("name", "count_key"),
    [
        ("approval ledger actor audit", "invalid_actor_count"),
        ("approval ledger fingerprint audit", "fingerprint_mismatch_count"),
        ("approval ledger revocation audit", "invalid_revocation_count"),
    ],
)
def test_readiness_warning_detail_summarizes_approval_ledger_audits(name, count_key):
    report = {
        "checks": [
            {
                "name": name,
                "level": "warning",
                "ok": False,
                "detail": {
                    count_key: 1,
                    "entries": [{"fingerprint": "sha256:bad", "strategy_id": "s1"}],
                },
            }
        ]
    }

    assert readiness_warning_detail(report) == {
        "warnings": [
            {
                "name": name,
                count_key: 1,
                "entries": [{"fingerprint": "sha256:bad", "strategy_id": "s1"}],
            }
        ]
    }


def test_readiness_warning_detail_ignores_malformed_checks():
    assert readiness_warning_detail({"checks": {"bad": "shape"}}) == {"warnings": []}
    assert readiness_warning_detail(
        {
            "checks": [
                "bad-check",
                {
                    "name": "market data seed and freshness",
                    "level": "warning",
                    "ok": False,
                    "detail": "not-an-object",
                },
                {
                    "name": "runtime filesystem free space",
                    "level": "warning",
                    "ok": False,
                    "detail": {"path": "runtime/status.json"},
                },
            ]
        }
    ) == {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": None,
                "free_bytes": None,
                "min_free_bytes": None,
                "reason": None,
            }
        ]
    }


def test_promotion_warning_detail_summarizes_approved_review_failures_only():
    report = {
        "promotion_reviews": [
            {
                "product": "active_income",
                "status": "ready",
                "path": "runtime/active_income_promotion_review.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "recommendations": {"approved_review_failed": 1, "needs_approval": 2},
            },
            {
                "product": "btc_accumulation",
                "status": "ready",
                "path": "runtime/btc_accumulation_promotion_review.json",
                "recommendations": {"needs_approval": 1},
            },
        ]
    }

    assert promotion_warning_detail(report) == {
        "warnings": [
            {
                "name": "approved_review_failed",
                "product": "active_income",
                "status": "ready",
                "approved_review_failed": 1,
                "path": "runtime/active_income_promotion_review.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    }


def test_promotion_warning_detail_ignores_malformed_reviews():
    assert promotion_warning_detail({"promotion_reviews": {"bad": "shape"}}) == {"warnings": []}
    assert promotion_warning_detail(
        {
            "promotion_reviews": [
                "bad-review",
                {"product": "active_income", "recommendations": {"approved_review_failed": 1}},
            ]
        }
    ) == {
        "warnings": [
            {
                "name": "approved_review_failed",
                "product": "active_income",
                "status": "unknown",
                "approved_review_failed": 1,
                "path": None,
                "generated_at": None,
            }
        ]
    }


def test_promotion_warning_detail_summarizes_stale_review_packets():
    report = {
        "promotion_reviews": [
            {
                "product": "active_income",
                "status": "ready",
                "enabled": True,
                "exists": True,
                "path": "runtime/active_income_promotion_review.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "fresh": False,
                "age_seconds": 200000.0,
                "max_age_seconds": 172800.0,
                "needs_approval": 1,
                "recommendations": {"needs_approval": 1},
            },
            {
                "product": "btc_accumulation",
                "enabled": True,
                "exists": False,
                "fresh": False,
            },
            {
                "product": "disabled",
                "enabled": False,
                "exists": True,
                "fresh": False,
            },
        ]
    }

    assert promotion_warning_detail(report) == {
        "warnings": [
            {
                "name": "promotion_review_stale",
                "product": "active_income",
                "status": "ready",
                "path": "runtime/active_income_promotion_review.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "fresh": False,
                "age_seconds": 200000.0,
                "max_age_seconds": 172800.0,
                "needs_approval": 1,
            }
        ]
    }


def test_research_handoff_warning_detail_summarizes_stale_sources():
    report = {
        "research_cycle": {"generated_at": "2026-01-01T01:05:00+00:00"},
        "mutation_plan": {
            "generated_at": "2026-01-01T01:06:00+00:00",
            "source": {"research_generated_at": "2026-01-01T00:55:00+00:00"},
        },
        "mutation_batch": {
            "generated_at": "2026-01-01T01:07:00+00:00",
            "source": {"plan_generated_at": "2026-01-01T00:56:00+00:00"},
        },
    }

    assert research_handoff_warning_detail(report) == {
        "warnings": [
            {
                "name": "mutation_plan_stale_source",
                "research_generated_at": "2026-01-01T01:05:00+00:00",
                "mutation_plan_source_research_generated_at": "2026-01-01T00:55:00+00:00",
                "mutation_plan_generated_at": "2026-01-01T01:06:00+00:00",
            },
            {
                "name": "mutation_batch_stale_source",
                "mutation_plan_generated_at": "2026-01-01T01:06:00+00:00",
                "mutation_batch_source_plan_generated_at": "2026-01-01T00:56:00+00:00",
                "mutation_batch_generated_at": "2026-01-01T01:07:00+00:00",
            },
        ]
    }


def test_research_handoff_warning_detail_summarizes_unsafe_mutation_artifacts():
    report = {
        "mutation_plan": {
            "ok": True,
            "generated_at": "2026-01-01T01:06:00+00:00",
            "path": "runtime/mutation_plan.json",
            "summary": {"executable": True},
        },
        "mutation_batch": {
            "ok": False,
            "status": "unsafe_mutation_plan",
            "generated_at": "2026-01-01T01:07:00+00:00",
            "path": "runtime/mutation_hypotheses.json",
            "summary": {
                "skipped": 0,
                "unsafe_flags": ["summary.executable"],
                "executable": False,
            },
        },
    }

    assert research_handoff_warning_detail(report) == {
        "warnings": [
            {
                "name": "mutation_plan_unhealthy",
                "ok": True,
                "generated_at": "2026-01-01T01:06:00+00:00",
                "path": "runtime/mutation_plan.json",
                "unsafe_flags": ["summary.executable"],
            },
            {
                "name": "mutation_batch_unhealthy",
                "ok": False,
                "status": "unsafe_mutation_plan",
                "generated_at": "2026-01-01T01:07:00+00:00",
                "path": "runtime/mutation_hypotheses.json",
                "unsafe_flags": ["summary.executable"],
                "skipped": 0,
            },
        ]
    }


def test_research_handoff_warning_detail_ignores_current_sources():
    report = {
        "research_cycle": {"generated_at": "2026-01-01T01:05:00+00:00"},
        "mutation_plan": {
            "generated_at": "2026-01-01T01:06:00+00:00",
            "source": {"research_generated_at": "2026-01-01T01:05:00+00:00"},
        },
        "mutation_batch": {
            "generated_at": "2026-01-01T01:07:00+00:00",
            "source": {"plan_generated_at": "2026-01-01T01:06:00+00:00"},
        },
    }

    assert research_handoff_warning_detail(report) == {"warnings": []}


def test_research_progress_warning_detail_summarizes_no_exportable_research():
    report = {
        "research_cycle": {
            "ok": True,
            "generated_at": "2026-01-01T01:05:00+00:00",
            "summary": {
                "hypotheses": 12,
                "keepers": 0,
                "exported": 0,
                "top_reasons": {"no_train_edge": 8},
                "next_actions": ["continue bounded search"],
                "mutation_effectiveness": {"evaluated_hypotheses": 4, "outcome": "no_keeper"},
            },
        },
        "products": [
            {
                "name": "active_income",
                "objective": "active_income",
                "market": "futures",
                "enabled": True,
                "mode": "paper",
                "reason": "waiting_for_strategy_artifact",
            },
            {
                "name": "disabled",
                "enabled": False,
                "mode": "paper",
                "reason": "waiting_for_strategy_artifact",
            },
        ],
    }

    assert research_progress_warning_detail(report) == {
        "warnings": [
            {
                "name": "research_cycle_no_exportable_strategies",
                "generated_at": "2026-01-01T01:05:00+00:00",
                "hypotheses": 12,
                "top_reasons": {"no_train_edge": 8},
                "next_actions": ["continue bounded search"],
                "waiting_products": [
                    {"name": "active_income", "objective": "active_income", "market": "futures"}
                ],
                "mutation_effectiveness": {"evaluated_hypotheses": 4, "outcome": "no_keeper"},
            }
        ]
    }


def test_research_progress_warning_detail_summarizes_open_position_export_block():
    report = {
        "research_cycle": {
            "ok": True,
            "generated_at": "2026-01-01T01:05:00+00:00",
            "summary": {
                "hypotheses": 12,
                "keepers": 1,
                "exported": 0,
                "export_reasons": {"open_positions_block_export": 1, "no_current_cycle_keepers": 1},
                "next_actions": [
                    "wait for open positions to close before replacing the active paper artifact"
                ],
            },
        },
        "products": [
            {
                "name": "active_income",
                "objective": "active_income",
                "market": "futures",
                "enabled": True,
                "mode": "paper",
                "open_positions": 1,
            },
            {
                "name": "btc_accumulation",
                "objective": "btc_accumulation",
                "market": "spot",
                "enabled": True,
                "mode": "paper",
                "open_positions": 0,
            },
        ],
    }

    assert research_progress_warning_detail(report) == {
        "warnings": [
            {
                "name": "research_export_blocked_open_positions",
                "generated_at": "2026-01-01T01:05:00+00:00",
                "keepers": 1,
                "exported": 0,
                "export_reasons": {"open_positions_block_export": 1, "no_current_cycle_keepers": 1},
                "next_actions": [
                    "wait for open positions to close before replacing the active paper artifact"
                ],
                "open_products": [
                    {
                        "name": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "open_positions": 1,
                    }
                ],
            }
        ]
    }


def test_research_progress_warning_detail_ignores_keeper_or_missing_waiting_products():
    assert research_progress_warning_detail(
        {
            "research_cycle": {
                "ok": True,
                "summary": {"hypotheses": 12, "keepers": 1, "exported": 0},
            },
            "products": [
                {
                    "name": "active_income",
                    "enabled": True,
                    "mode": "paper",
                    "reason": "waiting_for_strategy_artifact",
                }
            ],
        }
    ) == {"warnings": []}


def test_research_progress_warning_detail_ignores_malformed_products():
    assert research_progress_warning_detail(
        {
            "research_cycle": {
                "ok": True,
                "summary": {"hypotheses": 12, "keepers": 0, "exported": 0},
            },
            "products": {"bad": "shape"},
        }
    ) == {"warnings": []}
    assert research_progress_warning_detail(
        {
            "research_cycle": {
                "ok": True,
                "summary": {
                    "hypotheses": 12,
                    "keepers": 1,
                    "exported": 0,
                    "export_reasons": {"open_positions_block_export": 1},
                },
            },
            "products": [
                "bad-product",
                {"name": "active_income", "enabled": True, "open_positions": 1},
            ],
        }
    ) == {
        "warnings": [
            {
                "name": "research_export_blocked_open_positions",
                "generated_at": None,
                "keepers": 1,
                "exported": 0,
                "export_reasons": {"open_positions_block_export": 1},
                "next_actions": [],
                "open_products": [
                    {
                        "name": "active_income",
                        "objective": None,
                        "market": None,
                        "open_positions": 1,
                    }
                ],
            }
        ]
    }


def test_testnet_rehearsal_warning_detail_summarizes_required_missing_rehearsal():
    assert required_testnet_rehearsal_warning_detail(
        {
            "testnet_rehearsal": {
                "required": True,
                "required_by": ["active_income"],
                "status": "missing",
                "path": "runtime/testnet_rehearsal_report.json",
                "ok": False,
                "product": "active_income",
                "next_action": {
                    "rehearsal_command": "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100",
                },
            }
        }
    ) == {
        "warnings": [
            {
                "name": "required_testnet_rehearsal_not_ready",
                "status": "missing",
                "path": "runtime/testnet_rehearsal_report.json",
                "required_by": ["active_income"],
                "product": "active_income",
                "next_action": {
                    "rehearsal_command": "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100",
                },
            }
        ]
    }


def test_testnet_rehearsal_warning_detail_surfaces_product_mismatch():
    assert required_testnet_rehearsal_warning_detail(
        {
            "testnet_rehearsal": {
                "required": True,
                "required_by": ["active_income"],
                "status": "failed",
                "path": "runtime/testnet_rehearsal_report.json",
                "ok": False,
                "product": "active_income",
                "invalid_reasons": ["product_symbol_mismatch"],
                "report_product": {"name": "active_income", "symbol": "ETHUSDT"},
                "expected_product": {"name": "active_income", "symbol": "BTCUSDT"},
            }
        }
    ) == {
        "warnings": [
            {
                "name": "required_testnet_rehearsal_not_ready",
                "status": "failed",
                "path": "runtime/testnet_rehearsal_report.json",
                "required_by": ["active_income"],
                "product": "active_income",
                "invalid_reasons": ["product_symbol_mismatch"],
                "report_product": {"name": "active_income", "symbol": "ETHUSDT"},
                "expected_product": {"name": "active_income", "symbol": "BTCUSDT"},
            }
        ]
    }


def test_testnet_rehearsal_warning_detail_ignores_optional_or_ready_rehearsal():
    assert required_testnet_rehearsal_warning_detail(
        {"testnet_rehearsal": {"required": False, "status": "missing", "ok": False}}
    ) == {"warnings": []}
    assert required_testnet_rehearsal_warning_detail(
        {
            "products": [
                {"name": "active_income", "enabled": True, "require_testnet_rehearsal": True}
            ],
            "testnet_rehearsal": {"status": "ok", "ok": True, "required": True},
        }
    ) == {"warnings": []}
    assert research_progress_warning_detail(
        {
            "research_cycle": {
                "ok": True,
                "summary": {"hypotheses": 12, "keepers": 0, "exported": 0},
            },
            "products": [],
        }
    ) == {"warnings": []}


def test_testnet_rehearsal_warning_detail_ignores_malformed_products():
    assert required_testnet_rehearsal_warning_detail({"products": {"bad": "shape"}}) == {
        "warnings": []
    }
    assert required_testnet_rehearsal_warning_detail(
        {
            "products": [
                "bad-product",
                {"name": "active_income", "require_testnet_rehearsal": True},
            ],
            "testnet_rehearsal": {"status": "missing", "ok": False},
        }
    ) == {
        "warnings": [
            {
                "name": "required_testnet_rehearsal_not_ready",
                "status": "missing",
                "required_by": [],
            }
        ]
    }
