import csv
import json
from contextlib import contextmanager

import pytest

from src.autopilot import candidate_activation as activation_module
from src.autopilot.approvals import (
    ApprovalError,
    ApprovalLedger,
    artifact_digest,
    assert_artifact_live_approved,
    load_artifact,
    strategy_fingerprint,
)
from src.autopilot.candidate_activation import (
    CandidateActivationError,
    activate_candidate,
    product_identity,
)
from src.autopilot.config import load_config
from src.autopilot.locking import acquire_runtime_lock


def _strategy(strategy_id="candidate", *, take_profit=0.02):
    return {
        "id": strategy_id,
        "market": "futures",
        "symbol": "BTCUSDT",
        "base_timeframe": "5m",
        "direction": "long",
        "horizon_bars": 12,
        "take_profit": take_profit,
        "stop_loss": 0.01,
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
        "metrics": {"holdout_total_return": 0.03, "dsr_deflated": 0.72},
    }


def _artifact(strategy, *, identity=None):
    payload = {
        "version": 2,
        "market": "futures",
        "symbol": "BTCUSDT",
        "pnl_unit": "usdt",
        "paper_trade_allowed": True,
        "live_allowed": True,
        "promotion_eligible": True,
        "strategies": [strategy],
    }
    if identity is not None:
        payload["product"] = identity
    return payload


def _write_forward_paper_evidence(files, candidate):
    path = files["candidate_dir"] / "active_income_paper_trades.csv"
    strategy = candidate["strategies"][0]
    fingerprint = strategy_fingerprint(strategy)
    rows = [
        {
            "strategy_id": strategy["id"],
            "strategy_fingerprint": fingerprint,
            "artifact_digest": artifact_digest(candidate),
            "exit_time": f"2026-01-{index + 1:02d}T00:00:00+00:00",
            "net_return": 0.01,
            "sized_return": 0.001,
            "equity_after": 1000.0 + index + 1,
        }
        for index in range(20)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _setup(tmp_path, *, paused=True, state=None):
    active = tmp_path / "active.json"
    state_path = tmp_path / "state.json"
    control = tmp_path / "control.json"
    audit = tmp_path / "control_audit.jsonl"
    ledger = tmp_path / "approvals.json"
    lock = tmp_path / "autopilot.lock"
    config_path = tmp_path / "autopilot.json"
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    state_path.write_text(json.dumps(state if state is not None else {}), encoding="utf-8")
    control.write_text(
        json.dumps(
            {
                "paused": False,
                "pause_jobs": False,
                "paused_products": ["active_income"] if paused else [],
                "paused_jobs": [],
                "flatten_products": [],
                "flatten_all": False,
                "reason": "candidate review" if paused else "",
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "control_file": str(control),
                "control_audit_file": str(audit),
                "approval_ledger": str(ledger),
                "lock_file": str(lock),
                "products": [
                    {
                        "name": "active_income",
                        "enabled": True,
                        "objective": "active_income",
                        "base_asset": "USDT",
                        "market": "futures",
                        "execution_mode": "live",
                        "symbol": "BTCUSDT",
                        "strategies_path": str(active),
                        "state_file": str(state_path),
                        "trade_log": str(tmp_path / "trades.csv"),
                        "starting_equity": 1000.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    product = load_config(config_path).products[0]
    return {
        "active": active,
        "state": state_path,
        "control": control,
        "audit": audit,
        "ledger": ledger,
        "lock": lock,
        "config": config_path,
        "candidate_dir": candidate_dir,
        "candidate": candidate_dir / "active_income.json",
        "product": product,
    }


def test_activation_atomically_replaces_active_but_does_not_approve(tmp_path):
    files = _setup(tmp_path)
    old_strategy = _strategy("old", take_profit=0.015)
    old_artifact = _artifact(old_strategy)
    files["active"].write_text(json.dumps(old_artifact), encoding="utf-8")
    ApprovalLedger(files["ledger"]).approve(
        old_strategy,
        artifact_path=files["active"],
        approved_by="human-operator",
        product=files["product"],
        artifact=old_artifact,
    )
    candidate = _artifact(
        _strategy("new", take_profit=0.02),
        identity=product_identity(files["product"]),
    )
    files["candidate"].write_text(json.dumps(candidate), encoding="utf-8")
    _write_forward_paper_evidence(files, candidate)
    # Even a stale historical approval for these exact candidate bytes must
    # not carry through activation; activation provenance changes the digest.
    ApprovalLedger(files["ledger"]).approve(
        candidate["strategies"][0],
        artifact_path=files["active"],
        approved_by="human-operator",
        product=files["product"],
        artifact=candidate,
    )

    report = activate_candidate(
        config_path=files["config"],
        product_name="active_income",
        confirm=True,
        expected_candidate_digest=artifact_digest(candidate),
        candidate_dir=files["candidate_dir"],
        operator="human-operator",
    )

    activated = load_artifact(files["active"])
    assert activated["strategies"] == candidate["strategies"]
    assert activated["product"] == candidate["product"]
    assert activated["candidate_activation"]["approval_granted"] is False
    assert (
        activated["candidate_activation"]["candidate_artifact_digest"]
        == report["candidate_artifact_digest"]
    )
    assert report["ok"] is True
    assert report["approval_granted"] is False
    assert report["live_ready"] is False
    assert len(report["next_actions"]) == 4
    events = [json.loads(line) for line in files["audit"].read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events] == ["activation_intent", "activated"]
    assert all(event["approval_granted"] is False for event in events)
    assert events[1]["activated_artifact_digest"] == report["artifact_digest"]
    assert events[1]["candidate_artifact_digest"] == report["candidate_artifact_digest"]
    with pytest.raises(ApprovalError, match="artifact content mismatch"):
        assert_artifact_live_approved(
            files["active"],
            files["ledger"],
            product=files["product"],
        )


def test_activation_requires_explicit_confirmation_and_pause(tmp_path):
    files = _setup(tmp_path, paused=False)
    candidate = _artifact(_strategy(), identity=product_identity(files["product"]))
    files["candidate"].write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(CandidateActivationError, match="requires --confirm"):
        activate_candidate(
            config_path=files["config"],
            product_name="active_income",
            confirm=False,
            expected_candidate_digest=artifact_digest(candidate),
            candidate_dir=files["candidate_dir"],
        )
    with pytest.raises(CandidateActivationError, match="must be paused"):
        activate_candidate(
            config_path=files["config"],
            product_name="active_income",
            confirm=True,
            expected_candidate_digest=artifact_digest(candidate),
            candidate_dir=files["candidate_dir"],
        )


def test_activation_requires_exclusive_runtime_lock(tmp_path):
    files = _setup(tmp_path)
    candidate = _artifact(_strategy(), identity=product_identity(files["product"]))
    files["candidate"].write_text(json.dumps(candidate), encoding="utf-8")

    with acquire_runtime_lock(files["lock"]):
        with pytest.raises(CandidateActivationError, match="activation locking failed"):
            activate_candidate(
                config_path=files["config"],
                product_name="active_income",
                confirm=True,
                expected_candidate_digest=artifact_digest(candidate),
                candidate_dir=files["candidate_dir"],
            )

    assert not files["active"].exists()
    assert not files["audit"].exists()


def test_activation_lock_order_is_runtime_then_control_then_transaction(tmp_path, monkeypatch):
    files = _setup(tmp_path)
    events = []

    @contextmanager
    def fake_runtime_lock(path):
        events.append(("runtime_enter", path))
        yield
        events.append(("runtime_exit", path))

    @contextmanager
    def fake_control_lock(path):
        events.append(("control_enter", path))
        yield
        events.append(("control_exit", path))

    def fake_transaction(**kwargs):
        events.append(("transaction", kwargs["config"].control_file))
        return {"ok": True}

    monkeypatch.setattr(activation_module, "acquire_runtime_lock", fake_runtime_lock)
    monkeypatch.setattr(activation_module, "control_update_lock", fake_control_lock)
    monkeypatch.setattr(activation_module, "_activate_candidate_locked", fake_transaction)

    assert activate_candidate(
        config_path=files["config"],
        product_name="active_income",
        confirm=True,
        expected_candidate_digest="sha256:" + "0" * 64,
        candidate_dir=files["candidate_dir"],
    ) == {"ok": True}
    assert [event[0] for event in events] == [
        "runtime_enter",
        "control_enter",
        "transaction",
        "control_exit",
        "runtime_exit",
    ]


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"open_positions": {"s1": {"direction": "long"}}}, "no open positions"),
        ({"pending_order": {"client_id": "pending"}}, "still has pending_order"),
        ({"flatten_intent": {"client_id": "flatten"}}, "still has flatten_intent"),
        (
            {"pending_entry_recovery": {"client_id": "entry-recovery"}},
            "still has pending_entry_recovery",
        ),
        (
            {"risk_recovery_incident": {"reason": "protection_failed"}},
            "still has risk_recovery_incident",
        ),
        (
            {"exit_accounting_intent": {"phase": "ready_to_commit"}},
            "still has exit_accounting_intent",
        ),
        ({"open_positions": []}, "open_positions must be a JSON object"),
    ],
)
def test_activation_requires_clean_well_formed_product_state(tmp_path, state, message):
    files = _setup(tmp_path, state=state)
    candidate = _artifact(_strategy(), identity=product_identity(files["product"]))
    files["candidate"].write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(CandidateActivationError, match=message):
        activate_candidate(
            config_path=files["config"],
            product_name="active_income",
            confirm=True,
            expected_candidate_digest=artifact_digest(candidate),
            candidate_dir=files["candidate_dir"],
        )


def test_activation_binds_the_exact_reviewed_candidate_digest(tmp_path):
    files = _setup(tmp_path)
    candidate = _artifact(_strategy(), identity=product_identity(files["product"]))
    files["candidate"].write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(CandidateActivationError, match="candidate changed after review") as exc:
        activate_candidate(
            config_path=files["config"],
            product_name="active_income",
            confirm=True,
            expected_candidate_digest="sha256:" + "0" * 64,
            candidate_dir=files["candidate_dir"],
        )

    assert artifact_digest(candidate) in str(exc.value)
    assert not files["active"].exists()
    assert not files["audit"].exists()


def test_activation_requires_exact_fingerprint_forward_paper_evidence(tmp_path):
    files = _setup(tmp_path)
    candidate = _artifact(_strategy(), identity=product_identity(files["product"]))
    files["candidate"].write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(CandidateActivationError, match="forward-paper evidence is not ready"):
        activate_candidate(
            config_path=files["config"],
            product_name="active_income",
            confirm=True,
            expected_candidate_digest=artifact_digest(candidate),
            candidate_dir=files["candidate_dir"],
        )

    assert not files["active"].exists()
    assert not files["audit"].exists()


def test_activation_rejects_symlinked_or_wrong_product_candidate(tmp_path):
    files = _setup(tmp_path)
    external = tmp_path / "external.json"
    candidate = _artifact(_strategy(), identity=product_identity(files["product"]))
    external.write_text(json.dumps(candidate), encoding="utf-8")
    files["candidate"].symlink_to(external)

    with pytest.raises(CandidateActivationError, match="symlink component"):
        activate_candidate(
            config_path=files["config"],
            product_name="active_income",
            confirm=True,
            expected_candidate_digest=artifact_digest(candidate),
            candidate_dir=files["candidate_dir"],
        )

    files["candidate"].unlink()
    candidate["product"]["name"] = "btc_accumulation"
    files["candidate"].write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(CandidateActivationError, match="product identity mismatch"):
        activate_candidate(
            config_path=files["config"],
            product_name="active_income",
            confirm=True,
            expected_candidate_digest=artifact_digest(candidate),
            candidate_dir=files["candidate_dir"],
        )


def test_activation_rejects_policy_failure_without_touching_active(tmp_path):
    files = _setup(tmp_path)
    active = _artifact(_strategy("old"))
    files["active"].write_text(json.dumps(active), encoding="utf-8")
    candidate = _artifact(
        _strategy("unsafe", take_profit=0.02),
        identity=product_identity(files["product"]),
    )
    candidate["strategies"][0]["risk"]["risk_per_trade"] = 0.5
    files["candidate"].write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(Exception, match="violates policy"):
        activate_candidate(
            config_path=files["config"],
            product_name="active_income",
            confirm=True,
            expected_candidate_digest=artifact_digest(candidate),
            candidate_dir=files["candidate_dir"],
        )

    assert json.loads(files["active"].read_text(encoding="utf-8")) == active
    assert not files["audit"].exists()
