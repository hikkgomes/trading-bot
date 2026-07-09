import json

import pytest

from src.autopilot.artifact_hygiene import (
    build_artifact_hygiene_report,
    find_unreferenced_active_artifacts,
    inspect_product_artifact,
)
from src.autopilot.artifact_hygiene import (
    main as artifact_hygiene_main,
)
from src.autopilot.config import AutopilotConfig, ProductConfig


def product(tmp_path, **overrides):
    payload = {
        "name": "active_income",
        "enabled": True,
        "objective": "active_income",
        "base_asset": "USDT",
        "market": "futures",
        "execution_mode": "paper",
        "symbol": "BTCUSDT",
        "strategies_path": tmp_path / "active_strategies_flow.json",
        "state_file": tmp_path / "state.json",
        "trade_log": tmp_path / "trades.csv",
        "starting_equity": 1000.0,
    }
    payload.update(overrides)
    return ProductConfig(**payload)


def write_artifact(path, *, holdout=0.03):
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "strategies": [
                    {
                        "id": "s1",
                        "market": "futures",
                        "base_timeframe": "5m",
                        "direction": "long",
                        "horizon_bars": 4,
                        "take_profit": 0.02,
                        "stop_loss": 0.01,
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
                        "metrics": {"holdout_total_return": holdout, "dsr_deflated": 0.72},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_inspect_product_artifact_reports_missing_without_failure(tmp_path):
    report = inspect_product_artifact(product(tmp_path))

    assert report["ok"] is True
    assert report["status"] == "missing"
    assert report["reason"] == "waiting_for_research_export"


def test_inspect_product_artifact_reports_policy_blocked_without_moving_by_default(tmp_path):
    artifact = tmp_path / "active_strategies_flow.json"
    write_artifact(artifact, holdout=-0.01)

    report = inspect_product_artifact(product(tmp_path, strategies_path=artifact))

    assert report["ok"] is False
    assert report["status"] == "policy_blocked"
    assert report["quarantine_candidate"] is True
    assert report["action"] == "none"
    assert artifact.exists()


def test_inspect_product_artifact_quarantines_policy_blocked_paper_artifact(tmp_path):
    artifact = tmp_path / "active_strategies_flow.json"
    quarantine_dir = tmp_path / "quarantine"
    write_artifact(artifact, holdout=-0.01)

    report = inspect_product_artifact(
        product(tmp_path, strategies_path=artifact),
        apply=True,
        quarantine_dir=quarantine_dir,
    )

    assert report["action"] == "quarantined"
    assert not artifact.exists()
    assert report["quarantined_to"].startswith(str(quarantine_dir))


def test_inspect_product_artifact_refuses_to_quarantine_symlink_source(tmp_path):
    target = tmp_path / "target_active_strategies_flow.json"
    write_artifact(target, holdout=-0.01)
    artifact = tmp_path / "active_strategies_flow.json"
    artifact.symlink_to(target)

    with pytest.raises(ValueError, match="refusing to quarantine symlink source"):
        inspect_product_artifact(
            product(tmp_path, strategies_path=artifact),
            apply=True,
            quarantine_dir=tmp_path / "quarantine",
        )

    assert artifact.is_symlink()
    assert target.exists()


def test_inspect_product_artifact_refuses_symlink_quarantine_dir(tmp_path):
    artifact = tmp_path / "active_strategies_flow.json"
    write_artifact(artifact, holdout=-0.01)
    target = tmp_path / "real_quarantine"
    target.mkdir()
    quarantine_dir = tmp_path / "quarantine"
    quarantine_dir.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="quarantine_dir must not be a symlink"):
        inspect_product_artifact(
            product(tmp_path, strategies_path=artifact),
            apply=True,
            quarantine_dir=quarantine_dir,
        )

    assert artifact.exists()
    assert list(target.iterdir()) == []


def test_build_artifact_hygiene_report_continues_after_product_quarantine_error(tmp_path):
    blocked = tmp_path / "active_strategies_blocked.json"
    valid = tmp_path / "active_strategies_valid.json"
    write_artifact(blocked, holdout=-0.01)
    write_artifact(valid)
    real_quarantine = tmp_path / "real_quarantine"
    real_quarantine.mkdir()
    quarantine_dir = tmp_path / "quarantine"
    quarantine_dir.symlink_to(real_quarantine, target_is_directory=True)
    cfg = AutopilotConfig(
        products=[
            product(tmp_path, name="blocked", strategies_path=blocked),
            product(tmp_path, name="valid", strategies_path=valid),
        ]
    )

    report = build_artifact_hygiene_report(
        cfg,
        apply=True,
        outputs_dir=tmp_path,
        quarantine_dir=quarantine_dir,
    )

    assert report["ok"] is False
    assert report["summary"]["errors"] == 1
    assert report["errors"] == [
        {
            "scope": "configured_product",
            "path": str(blocked),
            "error": f"ValueError: quarantine_dir must not be a symlink: {quarantine_dir}",
            "product": "blocked",
        }
    ]
    assert report["configured_products"][0]["action"] == "error"
    assert report["configured_products"][0]["status"] == "error"
    assert report["configured_products"][1]["status"] == "valid"
    assert blocked.exists()
    assert valid.exists()
    assert list(real_quarantine.iterdir()) == []


def test_inspect_product_artifact_does_not_overwrite_broken_symlink_quarantine_target(tmp_path, monkeypatch):
    artifact = tmp_path / "active_strategies_flow.json"
    write_artifact(artifact, holdout=-0.01)
    quarantine_dir = tmp_path / "quarantine"
    quarantine_dir.mkdir()
    monkeypatch.setattr("src.autopilot.artifact_hygiene.time.strftime", lambda *args, **kwargs: "20260101T000000Z")
    first_collision = quarantine_dir / "active_strategies_flow.20260101T000000Z.json"
    first_collision.symlink_to(tmp_path / "missing-target.json")

    report = inspect_product_artifact(
        product(tmp_path, strategies_path=artifact),
        apply=True,
        quarantine_dir=quarantine_dir,
    )

    assert first_collision.is_symlink()
    assert report["quarantined_to"].endswith("active_strategies_flow.20260101T000000Z.dup.json")
    assert not artifact.exists()


def test_inspect_product_artifact_does_not_quarantine_live_artifact(tmp_path):
    artifact = tmp_path / "active_strategies_flow.json"
    write_artifact(artifact, holdout=-0.01)

    report = inspect_product_artifact(
        product(tmp_path, strategies_path=artifact, execution_mode="live"),
        apply=True,
        quarantine_dir=tmp_path / "quarantine",
    )

    assert report["action"] == "not_quarantined_live_product"
    assert artifact.exists()


def test_find_unreferenced_active_artifacts_excludes_configured_paths(tmp_path):
    configured = tmp_path / "active_strategies_flow.json"
    unreferenced = tmp_path / "active_strategies_old.json"
    write_artifact(configured)
    write_artifact(unreferenced)
    cfg = AutopilotConfig(products=[product(tmp_path, strategies_path=configured)])

    report = find_unreferenced_active_artifacts(cfg, tmp_path)

    assert [item["path"] for item in report] == [str(unreferenced)]
    assert report[0]["status"] == "unreferenced_active_artifact"


def test_build_artifact_hygiene_report_summarizes_candidates(tmp_path):
    artifact = tmp_path / "active_strategies_flow.json"
    write_artifact(artifact, holdout=-0.01)
    (tmp_path / "search_old").mkdir()
    (tmp_path / "search_old" / "report.md").write_text("old", encoding="utf-8")
    (tmp_path / "strategy_search_v3").mkdir()
    (tmp_path / "strategy_search_v3" / "report.md").write_text("old", encoding="utf-8")
    cfg = AutopilotConfig(products=[product(tmp_path, strategies_path=artifact)])

    report = build_artifact_hygiene_report(cfg, outputs_dir=tmp_path)

    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["summary"]["quarantine_candidates"] == 1
    assert report["summary"]["historical_search_outputs"] == 2


def test_build_artifact_hygiene_report_quarantines_unreferenced_only_when_requested(tmp_path):
    configured = tmp_path / "active_strategies_flow.json"
    unreferenced = tmp_path / "active_strategies_old.json"
    quarantine_dir = tmp_path / "quarantine"
    write_artifact(configured)
    write_artifact(unreferenced)
    cfg = AutopilotConfig(products=[product(tmp_path, strategies_path=configured)])

    dry_run = build_artifact_hygiene_report(
        cfg,
        apply=True,
        outputs_dir=tmp_path,
        quarantine_dir=quarantine_dir,
    )
    assert dry_run["summary"]["quarantined"] == 0
    assert unreferenced.exists()
    assert dry_run["unreferenced_active_artifacts"][0]["action"] == "none"

    report = build_artifact_hygiene_report(
        cfg,
        apply=True,
        quarantine_unreferenced_active=True,
        outputs_dir=tmp_path,
        quarantine_dir=quarantine_dir,
    )

    assert report["summary"]["quarantined"] == 1
    assert not unreferenced.exists()
    row = report["unreferenced_active_artifacts"][0]
    assert row["action"] == "quarantined"
    assert row["quarantined_to"].startswith(str(quarantine_dir))


def test_build_artifact_hygiene_report_records_unreferenced_quarantine_errors(tmp_path):
    configured = tmp_path / "active_strategies_flow.json"
    target = tmp_path / "external_active_strategies_old.json"
    unreferenced = tmp_path / "active_strategies_old.json"
    quarantine_dir = tmp_path / "quarantine"
    write_artifact(configured)
    write_artifact(target)
    unreferenced.symlink_to(target)
    cfg = AutopilotConfig(products=[product(tmp_path, strategies_path=configured)])

    report = build_artifact_hygiene_report(
        cfg,
        apply=True,
        quarantine_unreferenced_active=True,
        outputs_dir=tmp_path,
        quarantine_dir=quarantine_dir,
    )

    assert report["ok"] is False
    assert report["summary"]["errors"] == 1
    assert report["summary"]["quarantined"] == 0
    assert report["errors"] == [
        {
            "scope": "unreferenced_active_artifact",
            "path": str(unreferenced),
            "error": f"ValueError: refusing to quarantine symlink source: {unreferenced}",
        }
    ]
    row = report["unreferenced_active_artifacts"][0]
    assert row["action"] == "error"
    assert row["ok"] is False
    assert unreferenced.is_symlink()
    assert target.exists()
    assert not quarantine_dir.exists()


def test_build_artifact_hygiene_report_quarantines_historical_search_only_when_requested(tmp_path):
    configured = tmp_path / "active_strategies_flow.json"
    historical = tmp_path / "search_old"
    quarantine_dir = tmp_path / "quarantine"
    write_artifact(configured)
    historical.mkdir()
    (historical / "report.md").write_text("old", encoding="utf-8")
    cfg = AutopilotConfig(products=[product(tmp_path, strategies_path=configured)])

    dry_run = build_artifact_hygiene_report(
        cfg,
        apply=True,
        outputs_dir=tmp_path,
        quarantine_dir=quarantine_dir,
    )
    assert dry_run["summary"]["quarantined"] == 0
    assert historical.exists()
    assert dry_run["historical_search_outputs"][0]["action"] == "none"

    report = build_artifact_hygiene_report(
        cfg,
        apply=True,
        quarantine_historical_search=True,
        outputs_dir=tmp_path,
        quarantine_dir=quarantine_dir,
    )

    assert report["summary"]["quarantined"] == 1
    assert not historical.exists()
    row = report["historical_search_outputs"][0]
    assert row["action"] == "quarantined"
    assert row["quarantined_to"].startswith(str(quarantine_dir))


def test_artifact_hygiene_cli_exits_nonzero_for_structured_failure(monkeypatch, tmp_path, capsys):
    configured = tmp_path / "active_strategies_flow.json"
    target = tmp_path / "external_active_strategies_old.json"
    unreferenced = tmp_path / "active_strategies_old.json"
    report_path = tmp_path / "artifact_hygiene.json"
    config_path = tmp_path / "autopilot.json"
    write_artifact(configured)
    write_artifact(target)
    unreferenced.symlink_to(target)
    config_path.write_text(
        json.dumps(
            {
                "jobs": [],
                "products": [
                    {
                        "name": "active_income",
                        "objective": "active_income",
                        "base_asset": "USDT",
                        "market": "futures",
                        "strategies_path": str(configured),
                        "state_file": str(tmp_path / "state.json"),
                        "trade_log": str(tmp_path / "trades.csv"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "artifact_hygiene",
            "--config",
            str(config_path),
            "--output",
            str(report_path),
            "--outputs-dir",
            str(tmp_path),
            "--apply",
            "--quarantine-unreferenced-active",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        artifact_hygiene_main()

    assert exc.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert printed["ok"] is False
    assert saved == printed
    assert printed["errors"][0]["scope"] == "unreferenced_active_artifact"
