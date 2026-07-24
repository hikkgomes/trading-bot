import json

import pytest

from src.autopilot.control import load_control
from src.autopilot.notifications import emit_alert, wait_for_remote_alerts
from src.autopilot.telegram_edge import (
    TelegramError,
    TelegramSettings,
    build_status_snapshot,
    format_alert_message,
    format_status_message,
    handle_update,
    load_settings_file,
    poll_once,
    redact_sensitive,
    send_text,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def settings(*, pause=False):
    return TelegramSettings(
        bot_token="123:not-a-real-token",
        chat_id="-100123",
        allowed_user_ids=frozenset({42}),
        pause_commands_enabled=pause,
    )


def test_settings_require_complete_pair_and_explicit_pause_allowlist():
    assert TelegramSettings.from_environment({}) is None
    with pytest.raises(TelegramError, match="requires .* together"):
        TelegramSettings.from_environment({"AUTOPILOT_TELEGRAM_BOT_TOKEN": "token"})
    with pytest.raises(TelegramError, match="explicit .* allowlist"):
        TelegramSettings.from_environment(
            {
                "AUTOPILOT_TELEGRAM_BOT_TOKEN": "token",
                "AUTOPILOT_TELEGRAM_CHAT_ID": "123",
                "AUTOPILOT_TELEGRAM_PAUSE_COMMANDS": "1",
            }
        )


def test_settings_file_is_explicit_private_and_strictly_allowlisted(monkeypatch, tmp_path):
    settings_file = tmp_path / "telegram.env"
    settings_file.write_text(
        "AUTOPILOT_TELEGRAM_BOT_TOKEN=file-token\nAUTOPILOT_TELEGRAM_CHAT_ID=123\n",
        encoding="utf-8",
    )
    settings_file.chmod(0o600)
    monkeypatch.setenv("AUTOPILOT_TELEGRAM_SETTINGS_FILE", str(settings_file))
    monkeypatch.delenv("AUTOPILOT_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AUTOPILOT_TELEGRAM_CHAT_ID", raising=False)

    loaded = TelegramSettings.from_environment()

    assert loaded is not None
    assert loaded.bot_token == "file-token"
    assert loaded.chat_id == "123"

    settings_file.write_text(
        "AUTOPILOT_TELEGRAM_BOT_TOKEN=file-token\n"
        "AUTOPILOT_TELEGRAM_CHAT_ID=123\n"
        "EXCHANGE_API_SECRET=must-not-be-exposed\n",
        encoding="utf-8",
    )
    with pytest.raises(TelegramError) as exc_info:
        TelegramSettings.from_environment()
    assert "must-not-be-exposed" not in str(exc_info.value)

    settings_file.chmod(0o644)
    with pytest.raises(TelegramError, match="must not be group/world accessible"):
        TelegramSettings.from_environment()


@pytest.mark.parametrize(
    ("content", "message", "forbidden"),
    [
        (
            "AUTOPILOT_TELEGRAM_BOT_TOKEN=first-value\n"
            "AUTOPILOT_TELEGRAM_BOT_TOKEN=second-private-value\n",
            "duplicates a key",
            "second-private-value",
        ),
        (
            "AUTOPILOT_TELEGRAM_BOT_TOKEN=private-value\nnot-an-assignment\n",
            "KEY=value assignment",
            "private-value",
        ),
        (
            "AUTOPILOT_TELEGRAM_BOT_TOKEN='unterminated-private-value\n",
            "malformed value syntax",
            "unterminated-private-value",
        ),
        (
            "AUTOPILOT-TELEGRAM-BOT-TOKEN=private-value\n",
            "malformed key",
            "private-value",
        ),
    ],
)
def test_settings_parser_rejects_duplicate_or_malformed_lines_without_values(
    tmp_path, content, message, forbidden
):
    path = tmp_path / "telegram.env"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(TelegramError, match=message) as exc_info:
        load_settings_file(path)

    assert forbidden not in str(exc_info.value)


def test_send_text_uses_fixed_api_method_and_never_returns_token():
    calls = []

    def post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse({"ok": True, "result": {"message_id": 9}})

    result = send_text(settings(), "hello", post=post)

    assert result == {"ok": True, "message_id": 9}
    assert calls[0][0].endswith("/sendMessage")
    assert calls[0][1]["chat_id"] == "-100123"
    assert "parse_mode" not in calls[0][1]
    assert "token" not in json.dumps(result).lower()


def test_send_text_applies_final_secret_and_protected_content_filter():
    calls = []

    def post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse({"ok": True, "result": {"message_id": 10}})

    send_text(
        settings(),
        "failure Bearer final-bearer-secret; failed_holdout; "
        "https://example.test/path?X-Goog-Signature=final-google-secret",
        post=post,
    )

    sent = calls[0][1]["text"]
    assert "final-bearer-secret" not in sent
    assert "failed_holdout" not in sent
    assert "final-google-secret" not in sent
    assert sent == "[omitted protected research result]"


def test_telegram_api_rejection_description_is_secret_pattern_redacted():
    def post(_url, *, json, timeout):
        assert json and timeout
        return FakeResponse(
            {
                "ok": False,
                "description": (
                    "request rejected: Bearer bearer-secret "
                    "https://example.test/path?X-Amz-Signature=aws-secret"
                ),
            }
        )

    with pytest.raises(TelegramError) as exc_info:
        send_text(settings(), "hello", post=post)

    rendered = str(exc_info.value)
    assert "bearer-secret" not in rendered
    assert "aws-secret" not in rendered
    assert "[redacted]" in rendered


def test_alert_rendering_recursively_redacts_credentials_and_tokens():
    payload = {
        "severity": "error",
        "title": "failure",
        "detail": {
            "api_key": "super-secret",
            "nested": {"authorization": "Bearer secret", "safe": "visible"},
        },
    }

    redacted = redact_sensitive(payload)
    message = format_alert_message(payload)

    assert redacted["detail"]["api_key"] == "[redacted]"
    assert redacted["detail"]["nested"]["authorization"] == "[redacted]"
    assert "super-secret" not in message
    assert "Bearer secret" not in message
    assert "visible" in message


def test_alert_rendering_omits_protected_results_and_redacts_secret_patterns():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue"
    telegram_token = "123456789:ABCdef_123456789"
    payload = {
        "severity": "warning",
        "title": "research update",
        "detail": {
            "top_reasons": {
                "no_train_edge": 4,
                "failed_holdout": 2,
                "final_test_failed": 1,
            },
            "nested": [
                "safe development note",
                "final-test return was negative",
                {"holdout_metrics": {"return": -1.0}, "safe": "visible"},
            ],
            "error": "GET https://exchange.test/order?signature=query-secret",
            "safe_query": (
                "https://example.test/path?api_key=query-api-key&token=query-token "
                "https://storage.test/object?X-Amz-Credential=aws-credential"
                "&X-Amz-Signature=aws-signature&AWSAccessKeyId=aws-access-key "
                "https://storage.test/object?X-Goog-Credential=private-google-cred"
                "&X-Goog-Signature=private-google-sig"
            ),
            "safe_headers": "Authorization: Bearer bearer-secret",
            "jwt_value": jwt,
            "telegram_value": telegram_token,
            "embedded_assignment": "EXCHANGE_API_SECRET=assignment-secret",
        },
    }

    safe = redact_sensitive(payload)
    message = format_alert_message(payload)
    rendered = json.dumps(safe, sort_keys=True)

    assert safe["detail"]["top_reasons"] == {"no_train_edge": 4}
    assert safe["detail"]["nested"] == [
        "safe development note",
        {"safe": "visible"},
    ]
    assert safe["detail"]["error"] == "[redacted diagnostic]"
    for forbidden in (
        "holdout",
        "final_test",
        "final-test",
        "query-secret",
        "query-api-key",
        "query-token",
        "aws-credential",
        "aws-signature",
        "aws-access-key",
        "private-google-cred",
        "private-google-sig",
        "bearer-secret",
        "assignment-secret",
        jwt,
        telegram_token,
    ):
        assert forbidden.lower() not in rendered.lower()
        assert forbidden.lower() not in message.lower()
    assert "safe development note" in message
    assert "visible" in message


def test_existing_alert_path_adds_telegram_without_replacing_webhook(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOPILOT_TELEGRAM_BOT_TOKEN", "configured-for-test")
    monkeypatch.setattr(
        "src.autopilot.notifications.send_alert_from_environment",
        lambda payload, **_kwargs: {"ok": True, "message_id": 12},
    )

    result = emit_alert(
        alert_file=tmp_path / "alerts.jsonl",
        state_file=tmp_path / "state.json",
        severity="warning",
        title="research update",
        detail={"hypotheses": 4},
        cooldown_seconds=0,
        now=100,
    )

    assert result["sent"] is True
    assert result["remote_delivery"] == {"status": "queued"}
    assert wait_for_remote_alerts()
    records = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert records[0]["schema"] == "autopilot.alert/v1"
    assert records[1]["telegram"] == {"ok": True, "message_id": 12}


def test_status_snapshot_is_allowlisted_and_excludes_raw_errors_and_secrets(tmp_path):
    status = tmp_path / "status.json"
    control = tmp_path / "control.json"
    worker = tmp_path / "worker.json"
    research = tmp_path / "research.json"
    status.write_text(
        json.dumps(
            {
                "ok": False,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "api_key": "never-export",
                "products": [
                    {
                        "product": {
                            "name": "active_income",
                            "market": "futures",
                            "execution_mode": "paper",
                            "account_fingerprint": "private-account",
                        },
                        "ok": False,
                        "error": "contains sensitive path /home/user/.env",
                        "open_positions": 1,
                    }
                ],
            }
        )
    )
    worker.write_text(json.dumps({"ok": True, "jobs": []}))
    research.write_text(
        json.dumps(
            {
                "ok": True,
                "summary": {
                    "hypotheses": 12,
                    "keepers": 1,
                    "api_token": "never-export",
                    "top_reasons": {
                        "no_train_edge": 3,
                        "failed_holdout": 2,
                    },
                    "final_test_metrics": {"return": 99},
                },
            }
        )
    )

    snapshot = build_status_snapshot(
        status_path=status,
        control_path=control,
        job_worker_status_path=worker,
        research_cycle_path=research,
    )
    rendered = json.dumps(snapshot)

    assert snapshot["products"][0]["name"] == "active_income"
    assert snapshot["products"][0]["open_positions"] == 1
    assert snapshot["research"]["summary"]["keepers"] == 1
    assert snapshot["research"]["summary"]["top_reasons"] == {"no_train_edge": 3}
    assert "never-export" not in rendered
    assert "holdout" not in rendered.lower()
    assert "final_test" not in rendered.lower()
    assert "private-account" not in rendered
    assert "/home/user/.env" not in rendered


def telegram_update(command, *, chat_id=-100123, sender_id=42, update_id=7):
    return {
        "update_id": update_id,
        "message": {
            "chat": {"id": chat_id},
            "from": {"id": sender_id},
            "text": command,
        },
    }


def test_status_message_surfaces_research_and_openclaw_failures():
    message = format_status_message(
        {
            "supervisor": {
                "ok": True,
                "generated_at": "2026-07-24T12:00:00+00:00",
            },
            "control": {},
            "products": [
                {
                    "name": "active_income",
                    "ok": True,
                    "mode": "paper",
                    "open_positions": 0,
                }
            ],
            "job_worker": {"ok": True},
            "research": {
                "ok": True,
                "summary": {
                    "hypotheses": 6,
                    "keepers": 0,
                    "staged": 0,
                    "unsupported_hypotheses": 8,
                    "verdicts": {"inconclusive": 4, "reject": 2},
                    "top_reasons": {"unsupported_features": 8},
                    "generative_search": {
                        "batch_hypotheses": 17,
                        "new_hypotheses": 10,
                        "resumed_pending": 2,
                        "revalidation_pending": 5,
                        "openclaw_proposals_seen": 0,
                    },
                },
            },
            "universe": {
                "ok": True,
                "research_symbols": ["BTCUSDT", "ETHUSDT"],
                "eligible_research_symbols": ["BTCUSDT", "ETHUSDT"],
            },
            "openclaw_review": {
                "recorded_at": "2026-07-24T07:00:00+00:00",
                "proposal_count": 3,
            },
            "openclaw_ingest": {
                "ok": False,
                "degraded": True,
                "degraded_reasons": ["inbox_io_error"],
                "generated_at": "2026-07-24T12:05:00+00:00",
                "accepted": 0,
                "rejected": 3,
                "remaining": 38,
            },
        }
    )

    assert "Overall: Attention needed." in message
    assert "Selected 17: 10 new, 2 resumed, 5 revalidations." in message
    assert "4 inconclusive, 2 reject" in message
    assert "OpenClaw proposed 3 ideas" in message
    assert "0 accepted, 3 rejected, 38 awaiting ingestion" in message
    assert "factory consumed 0 OpenClaw proposals" in message


def test_authorized_pause_product_is_audited_and_no_resume_command_exists(tmp_path):
    control = tmp_path / "control.json"
    audit = tmp_path / "audit.jsonl"
    kwargs = {
        "settings": settings(pause=True),
        "status_path": tmp_path / "status.json",
        "control_path": control,
        "control_audit_path": audit,
        "product_names": {"active_income", "btc_accumulation"},
    }

    outcome = handle_update(telegram_update("/pause_product active_income"), **kwargs)
    refused = handle_update(telegram_update("/resume_product active_income", update_id=8), **kwargs)

    assert outcome["handled"] is True
    assert load_control(control)["paused_products"] == ["active_income"]
    event = json.loads(audit.read_text().splitlines()[0])
    assert event["actor"] == "telegram:42"
    assert event["command"] == "pause-product"
    assert refused["refused"] is True
    assert load_control(control)["paused_products"] == ["active_income"]


@pytest.mark.parametrize(
    "command",
    ["/approve active_income", "/activate x", "/resume", "/panic", "/flatten active_income"],
)
def test_dangerous_or_privilege_increasing_commands_are_always_refused(command, tmp_path):
    control = tmp_path / "control.json"

    outcome = handle_update(
        telegram_update(command),
        settings=settings(pause=True),
        status_path=tmp_path / "status.json",
        control_path=control,
        control_audit_path=tmp_path / "audit.jsonl",
        product_names={"active_income"},
    )

    assert outcome["refused"] is True
    assert not control.exists()


def test_pause_requires_both_exact_chat_and_allowed_sender(tmp_path):
    common = {
        "settings": settings(pause=True),
        "status_path": tmp_path / "status.json",
        "control_path": tmp_path / "control.json",
        "control_audit_path": tmp_path / "audit.jsonl",
        "product_names": {"active_income"},
    }

    wrong_chat = handle_update(telegram_update("/pause", chat_id=999), **common)
    wrong_user = handle_update(telegram_update("/pause", sender_id=99), **common)

    assert wrong_chat == {"handled": False, "reason": "unauthorized_chat"}
    assert wrong_user["refused"] is True
    assert not common["control_path"].exists()


def test_poll_once_advances_offset_and_applies_idempotent_pause(tmp_path):
    calls = []

    def post(url, *, json, timeout):
        calls.append((url, json, timeout))
        if url.endswith("/getUpdates"):
            return FakeResponse({"ok": True, "result": [telegram_update("/pause", update_id=15)]})
        return FakeResponse({"ok": True, "result": {"message_id": 1}})

    control = tmp_path / "control.json"
    state = tmp_path / "telegram_state.json"
    report = poll_once(
        settings=settings(pause=True),
        status_path=tmp_path / "status.json",
        control_path=control,
        control_audit_path=tmp_path / "audit.jsonl",
        poll_state_path=state,
        product_names={"active_income"},
        long_poll_seconds=0,
        post=post,
    )

    assert report["ok"] is True
    assert report["next_update_id"] == 16
    assert json.loads(state.read_text())["next_update_id"] == 16
    assert load_control(control)["paused"] is True
    assert calls[0][1]["offset"] == 0


def test_poll_once_reads_legacy_offset_before_writing_narrowed_state_path(tmp_path, monkeypatch):
    from src.autopilot import telegram_edge

    legacy = tmp_path / "telegram_poll_state.json"
    narrowed = tmp_path / "telegram" / "telegram_poll_state.json"
    legacy.write_text('{"next_update_id": 42}\n', encoding="utf-8")
    monkeypatch.setattr(telegram_edge, "DEFAULT_POLL_STATE", narrowed)
    monkeypatch.setattr(telegram_edge, "LEGACY_POLL_STATE", legacy)
    requests = []

    def post(url, *, json, timeout):
        requests.append((url, json, timeout))
        return FakeResponse({"ok": True, "result": []})

    report = poll_once(
        settings=settings(),
        status_path=tmp_path / "status.json",
        control_path=tmp_path / "control.json",
        control_audit_path=tmp_path / "audit.jsonl",
        poll_state_path=narrowed,
        product_names=set(),
        long_poll_seconds=0,
        post=post,
    )

    assert report["next_update_id"] == 42
    assert requests[0][1]["offset"] == 42
    assert json.loads(narrowed.read_text(encoding="utf-8"))["next_update_id"] == 42
    assert json.loads(legacy.read_text(encoding="utf-8"))["next_update_id"] == 42


def test_poll_once_does_not_acknowledge_failed_control_update(monkeypatch, tmp_path):
    def post(url, *, json, timeout):
        if url.endswith("/getUpdates"):
            return FakeResponse({"ok": True, "result": [telegram_update("/pause", update_id=15)]})
        return FakeResponse({"ok": True, "result": {"message_id": 1}})

    monkeypatch.setattr(
        "src.autopilot.telegram_edge.update_control",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    state = tmp_path / "telegram_state.json"

    report = poll_once(
        settings=settings(pause=True),
        status_path=tmp_path / "status.json",
        control_path=tmp_path / "control.json",
        control_audit_path=tmp_path / "audit.jsonl",
        poll_state_path=state,
        product_names={"active_income"},
        long_poll_seconds=0,
        post=post,
    )

    assert report["ok"] is False
    assert report["next_update_id"] == 0
    assert json.loads(state.read_text())["next_update_id"] == 0
