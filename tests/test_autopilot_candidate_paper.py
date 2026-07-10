import json

import pytest

from src.autopilot.approvals import artifact_digest
from src.autopilot.candidate_activation import product_identity
from src.autopilot.candidate_paper import candidate_paper_paths, run_candidate_paper
from src.autopilot.config import AutopilotConfig, ProductConfig


def _product(tmp_path, *, mode="live"):
    return ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode=mode,
        symbol="BTCUSDT",
        strategies_path=tmp_path / "active.json",
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
        starting_equity=1000.0,
    )


def _candidate(product):
    return {
        "version": 2,
        "market": "futures",
        "symbol": "BTCUSDT",
        "pnl_unit": "usdt",
        "paper_trade_allowed": True,
        "live_allowed": True,
        "promotion_eligible": True,
        "product": product_identity(product),
        "strategies": [
            {
                "id": "candidate",
                "market": "futures",
                "symbol": "BTCUSDT",
                "base_timeframe": "5m",
                "direction": "long",
                "horizon_bars": 12,
                "take_profit": 0.02,
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
        ],
    }


def test_candidate_paper_uses_digest_isolated_state_and_exact_artifact(
    monkeypatch,
    tmp_path,
):
    product = _product(tmp_path)
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    candidate = _candidate(product)
    candidate_path = candidate_dir / "active_income.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    captured = {}

    class FakeBot:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.state = {
                "equity": 1001.0,
                "open_positions": {},
                "drawdown_halted": False,
            }

        def run_cycle(self):
            captured["ran"] = True

    monkeypatch.setattr("src.autopilot.candidate_paper.PaperTradingBot", FakeBot)
    monkeypatch.setattr(
        "src.autopilot.candidate_paper.build_promotion_review",
        lambda **kwargs: {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "artifact_path": str(kwargs["artifact_path"]),
            "artifact_digest": artifact_digest(candidate),
            "trade_log": str(kwargs["trade_log"]),
            "strategies": [
                {
                    "recommendation": "needs_approval",
                    "reasons": ["passes"],
                    "approval_command": "unsafe-before-activation",
                }
            ],
        },
    )
    monkeypatch.setattr(
        "src.autopilot.candidate_paper.write_review",
        lambda review, output_json, output_md: captured.update(review=review),
    )

    report = run_candidate_paper(
        AutopilotConfig(products=[product]),
        candidate_dir=candidate_dir,
    )

    item = report["products"][0]
    digest = artifact_digest(candidate)
    assert report["ok"] is True
    assert item["candidate_digest"] == digest
    assert item["candidate_activation_ready"] is True
    assert captured["artifact_payload"] == candidate
    assert captured["ran"] is True
    assert digest.removeprefix("sha256:")[:16] in str(captured["state_file"])
    assert captured["review"]["strategies"][0]["recommendation"] == "ready_for_activation"
    assert captured["review"]["strategies"][0]["approval_command"] is None


def test_candidate_paper_skips_non_live_product_without_candidate(tmp_path):
    report = run_candidate_paper(
        AutopilotConfig(products=[_product(tmp_path, mode="paper")]),
        candidate_dir=tmp_path / "candidates",
    )

    assert report["ok"] is True
    assert report["products"][0]["reason"] == "product_not_live"


def test_candidate_paper_fails_closed_on_wrong_product_identity(tmp_path):
    product = _product(tmp_path)
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    candidate = _candidate(product)
    candidate["product"]["symbol"] = "ETHUSDT"
    (candidate_dir / "active_income.json").write_text(json.dumps(candidate), encoding="utf-8")

    report = run_candidate_paper(
        AutopilotConfig(products=[product]),
        candidate_dir=candidate_dir,
    )

    assert report["ok"] is False
    assert "product identity mismatch" in report["products"][0]["error"]


def test_candidate_paper_paths_reject_bad_digest(tmp_path):
    with pytest.raises(ValueError, match="sha256"):
        candidate_paper_paths(
            "active_income",
            "not-a-digest",
            candidate_dir=tmp_path,
        )
