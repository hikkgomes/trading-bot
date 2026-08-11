import json
import os
import stat
import zipfile

import pytest

from src.autopilot.backup import (
    backup_output_summary,
    build_backup_archive,
    configured_backup_paths,
    main,
    prune_backup_archives,
    restore_backup_archive,
    verify_backup_archive,
)
from src.autopilot.candidate_activation import candidate_path_for_product
from src.autopilot.config import AutopilotConfig, JobConfig, ProductConfig
from src.autopilot.experiment_memory import ExperimentMemory


def product(tmp_path):
    return ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="paper",
        symbol="BTCUSDT",
        strategies_path=tmp_path / "active_strategies_flow.json",
        state_file=tmp_path / "active_income_state.json",
        trade_log=tmp_path / "active_income_trades.csv",
        preflight_report=tmp_path / "active_income_preflight.json",
        starting_equity=1000.0,
    )


def write_config(tmp_path):
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(
        json.dumps(
            {
                "approval_ledger": str(tmp_path / "approvals.json"),
                "control_file": str(tmp_path / "control.json"),
                "control_audit_file": str(tmp_path / "control_audit.jsonl"),
                "status_file": str(tmp_path / "status.json"),
                "job_state_file": str(tmp_path / "job_state.json"),
                "alert_file": str(tmp_path / "alerts.jsonl"),
                "alert_state_file": str(tmp_path / "alert_state.json"),
                "candidate_paper_status_file": str(tmp_path / "candidate_paper_status.json"),
                "event_capture_status_file": str(tmp_path / "event_capture_status.json"),
                "portfolio_risk_file": str(tmp_path / "portfolio_risk.json"),
                "trade_starvation_history_file": str(tmp_path / "trade_starvation_history.jsonl"),
                "trade_starvation_report_file": str(tmp_path / "trade_starvation.json"),
                "research_smoke_file": str(tmp_path / "research_smoke.json"),
                "strategy_smoke_file": str(tmp_path / "strategy_smoke.json"),
                "research_cycle_file": str(tmp_path / "research_cycle.json"),
                "research_factory_config_file": str(tmp_path / "research_factory.json"),
                "generated_batch_file": str(tmp_path / "generated_batch.json"),
                "experiment_memory_file": str(tmp_path / "experiment_memory.sqlite3"),
                "experiment_memory_backup_file": str(tmp_path / "experiment_memory.backup.sqlite3"),
                "incubation_candidates_file": str(tmp_path / "incubation_candidates.json"),
                "mutation_plan_file": str(tmp_path / "mutation_plan.json"),
                "mutation_batch_file": str(tmp_path / "mutation_batch.json"),
                "artifact_hygiene_file": str(tmp_path / "artifact_hygiene.json"),
                "backup_report_file": str(tmp_path / "backup_report.json"),
                "operator_report_file": str(tmp_path / "operator_report.md"),
                "operator_report_json_file": str(tmp_path / "operator_report.json"),
                "readiness_report_file": str(tmp_path / "readiness_report.md"),
                "readiness_report_json_file": str(tmp_path / "readiness_report.json"),
                "jobs": [],
                "products": [
                    {
                        "name": "active_income",
                        "enabled": True,
                        "objective": "active_income",
                        "base_asset": "USDT",
                        "market": "futures",
                        "execution_mode": "paper",
                        "symbol": "BTCUSDT",
                        "strategies_path": str(tmp_path / "active_strategies_flow.json"),
                        "state_file": str(tmp_path / "active_income_state.json"),
                        "trade_log": str(tmp_path / "active_income_trades.csv"),
                        "preflight_report": str(tmp_path / "active_income_preflight.json"),
                        "testnet_rehearsal_report": str(
                            tmp_path / "active_income_testnet_rehearsal.json"
                        ),
                        "starting_equity": 1000.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_configured_backup_paths_include_core_and_product_state(tmp_path):
    config = AutopilotConfig(
        approval_ledger=tmp_path / "approvals.json",
        control_file=tmp_path / "control.json",
        backup_report_file=tmp_path / "backup_report.json",
        incubation_candidates_file=tmp_path / "incubation_candidates.json",
        products=[product(tmp_path)],
    )

    paths = configured_backup_paths(config, config_path=tmp_path / "autopilot.json")

    assert tmp_path / "autopilot.json" in paths
    assert tmp_path / "approvals.json" in paths
    assert tmp_path / "control.json" in paths
    # Candidate activations append to the configured control audit and live
    # research candidates are deterministic per product.
    assert config.control_audit_file in paths
    assert candidate_path_for_product("active_income") in paths
    assert tmp_path / "backup_report.json" in paths
    assert tmp_path / "incubation_candidates.json" in paths
    assert config.research_factory_config_file in paths
    assert config.generated_batch_file in paths
    assert config.experiment_memory_backup_file in paths
    assert tmp_path / "active_strategies_flow.json" in paths
    assert tmp_path / "active_income_state.json" in paths
    assert tmp_path / "active_income_trades.csv" in paths


def test_configured_backup_paths_include_job_state_and_output_paths(tmp_path):
    config = AutopilotConfig(
        research_cycle_file=tmp_path / "runtime" / "research_cycle.json",
        mutation_batch_file=tmp_path / "runtime" / "mutation_hypotheses.json",
        products=[product(tmp_path)],
        jobs=[
            JobConfig(
                name="research_cycle",
                enabled=True,
                command=[
                    ".venv/bin/python",
                    "-m",
                    "src.autopilot.research_cycle",
                    "--output",
                    "runtime/research_cycle.json",
                    "--state",
                    "runtime/research_cycle_state.json",
                    "--mutation-batch",
                    "runtime/mutation_hypotheses.json",
                ],
                cadence_seconds=86400,
                timeout_seconds=900,
                working_dir=tmp_path,
            ),
            JobConfig(
                name="promotion_review",
                enabled=True,
                command=[
                    ".venv/bin/python",
                    "-m",
                    "src.autopilot.promotion",
                    "--output-json",
                    "runtime/promotion_review.json",
                    "--output-md",
                    "runtime/promotion_review.md",
                ],
                cadence_seconds=86400,
                timeout_seconds=120,
                working_dir=tmp_path,
            ),
        ],
    )

    paths = configured_backup_paths(config, config_path=tmp_path / "autopilot.json")

    assert tmp_path / "runtime" / "research_cycle.json" in paths
    assert tmp_path / "runtime" / "research_cycle_state.json" in paths
    assert tmp_path / "runtime" / "mutation_hypotheses.json" in paths
    assert tmp_path / "runtime" / "promotion_review.json" in paths
    assert tmp_path / "runtime" / "promotion_review.md" in paths


def test_build_backup_archive_writes_manifest_and_existing_files(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "approvals.json").write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    (tmp_path / "control.json").write_text('{"paused": false}\n', encoding="utf-8")
    (tmp_path / "active_income_state.json").write_text('{"equity": 1000}\n', encoding="utf-8")
    (tmp_path / "active_income_trades.csv").write_text(
        "exit_time,net_return,sized_return\n", encoding="utf-8"
    )
    output = tmp_path / "backup.zip"

    report = build_backup_archive(config_path=config_path, output=output, root=tmp_path)

    assert report["ok"] is True
    assert report["verification"]["ok"] is True
    assert report["verification"]["checked_files"] >= 5
    assert report["manifest"]["included_files"] >= 5
    assert report["manifest"]["missing_files"] > 0
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "MANIFEST.json" in names
        assert "autopilot.json" in names
        assert "approvals.json" in names
        assert "control.json" in names
        assert "active_income_state.json" in names
        manifest = json.loads(archive.read("MANIFEST.json"))
    included = {item["arcname"] for item in manifest["files"] if item["included"]}
    assert "active_income_trades.csv" in included
    assert all(item.get("sha256") for item in manifest["files"] if item["included"])


def test_backup_archive_is_owner_private_even_with_permissive_umask(tmp_path):
    config_path = write_config(tmp_path)
    output = tmp_path / "backup.zip"
    previous = os.umask(0)
    try:
        report = build_backup_archive(config_path=config_path, output=output, root=tmp_path)
    finally:
        os.umask(previous)

    assert report["ok"] is True
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_build_backup_archive_uses_a_deeply_validated_sqlite_memory_snapshot(tmp_path):
    config_path = write_config(tmp_path)
    memory_path = tmp_path / "experiment_memory.sqlite3"
    snapshot_path = tmp_path / "experiment_memory.backup.sqlite3"
    with ExperimentMemory(memory_path) as memory:
        memory.register_strategy(
            {
                "id": "research-1",
                "direction": "long",
                "regime": [
                    {
                        "timeframe": "1h",
                        "feature": "ema_20",
                        "op": "gt_feature",
                        "feature_b": "ema_50",
                    }
                ],
            },
            strategy_id="research-1",
            generation_method="grammar_sample",
            metadata={"product": "active_income", "opportunity_type": "swing_trading"},
        )

    report = build_backup_archive(
        config_path=config_path,
        output=tmp_path / "backup.zip",
        root=tmp_path,
    )

    snapshot = report["manifest"]["experiment_memory_snapshot"]
    assert report["ok"] is True
    assert snapshot["refreshed"] is True
    assert snapshot["source_integrity"]["ok"] is True
    assert snapshot["snapshot_integrity"]["ok"] is True
    entry = next(
        item
        for item in report["manifest"]["files"]
        if item.get("role") == "experiment_memory_snapshot"
    )
    assert entry["included"] is True
    assert entry["sha256"] == snapshot["sha256"]
    assert snapshot_path.stat().st_mode & 0o777 == 0o600

    restored_snapshot = tmp_path / "restored-memory.sqlite3"
    with zipfile.ZipFile(report["output"]) as archive:
        restored_snapshot.write_bytes(archive.read(entry["arcname"]))
    with ExperimentMemory(restored_snapshot) as restored:
        assert restored.integrity_check(deep=True)["ok"] is True
        assert restored.generator_feedback()["totals"]["strategies"] == 1


def test_build_backup_archive_fails_if_current_memory_snapshot_is_omitted_by_size_limit(
    tmp_path,
):
    config_path = write_config(tmp_path)
    with ExperimentMemory(tmp_path / "experiment_memory.sqlite3"):
        pass

    report = build_backup_archive(
        config_path=config_path,
        output=tmp_path / "backup.zip",
        max_file_bytes=100,
        root=tmp_path,
    )

    assert report["ok"] is False
    assert any(
        item["code"] == "required_memory_snapshot_missing"
        for item in report["verification"]["issues"]
    )


def test_build_backup_archive_rejects_live_memory_as_its_own_backup_destination(tmp_path):
    config_path = write_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment_memory_backup_file"] = payload["experiment_memory_file"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="backup path must differ"):
        build_backup_archive(
            config_path=config_path,
            output=tmp_path / "backup.zip",
            root=tmp_path,
        )


def test_build_backup_archive_rejects_broken_memory_symlink(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "experiment_memory.sqlite3").symlink_to(tmp_path / "missing.sqlite3")

    with pytest.raises(ValueError, match="must not be symlinks"):
        build_backup_archive(
            config_path=config_path,
            output=tmp_path / "backup.zip",
            root=tmp_path,
        )


def test_build_backup_archive_propagates_failed_verification(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    output = tmp_path / "backup.zip"
    monkeypatch.setattr(
        "src.autopilot.backup.verify_backup_archive",
        lambda path: {
            "path": str(path),
            "exists": True,
            "ok": False,
            "checked_files": 0,
            "issues": [{"code": "injected_verification_failure"}],
        },
    )

    report = build_backup_archive(config_path=config_path, output=output, root=tmp_path)

    assert report["ok"] is False
    assert report["verification"]["ok"] is False


def test_build_backup_archive_includes_existing_job_state_paths(tmp_path):
    config_path = write_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["jobs"] = [
        {
            "name": "research_cycle",
            "enabled": True,
            "command": [
                ".venv/bin/python",
                "-m",
                "src.autopilot.research_cycle",
                "--output",
                "runtime/research_cycle.json",
                "--state",
                "runtime/research_cycle_state.json",
            ],
            "cadence_seconds": 86400,
            "timeout_seconds": 900,
            "working_dir": str(tmp_path),
        }
    ]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path = tmp_path / "runtime" / "research_cycle_state.json"
    state_path.parent.mkdir()
    state_path.write_text('{"last_run_id": "abc"}\n', encoding="utf-8")

    report = build_backup_archive(
        config_path=config_path, output=tmp_path / "backup.zip", root=tmp_path
    )

    state_entry = next(
        item for item in report["manifest"]["files"] if item["path"] == str(state_path)
    )
    assert state_entry["included"] is True
    assert state_entry["arcname"] == "runtime/research_cycle_state.json"
    with zipfile.ZipFile(report["output"]) as archive:
        assert archive.read("runtime/research_cycle_state.json") == b'{"last_run_id": "abc"}\n'


def test_build_backup_archive_skips_files_above_size_limit(tmp_path):
    config_path = write_config(tmp_path)
    large_state = tmp_path / "active_income_state.json"
    max_file_bytes = config_path.stat().st_size + 10
    large_state.write_text("x" * (max_file_bytes + 1), encoding="utf-8")
    output = tmp_path / "backup.zip"

    report = build_backup_archive(
        config_path=config_path,
        output=output,
        max_file_bytes=max_file_bytes,
        root=tmp_path,
    )

    state_entry = next(
        item for item in report["manifest"]["files"] if item["path"] == str(large_state)
    )
    assert state_entry["included"] is False
    assert state_entry["reason"] == "too_large"
    assert state_entry["required_if_present"] is True
    assert report["manifest"]["critical_skipped_files"] == 1
    assert report["manifest"]["required_recovery_files"] == len(
        report["manifest"]["required_recovery_roles"]
    )
    assert len(report["manifest"]["required_recovery_roles"]) == len(
        set(report["manifest"]["required_recovery_roles"])
    )
    assert report["ok"] is False
    assert report["verification"]["issues"] == [
        {
            "code": "required_recovery_file_skipped",
            "path": str(large_state),
            "role": "product:active_income:product_state",
            "reason": "too_large",
        }
    ]
    with zipfile.ZipFile(output) as archive:
        assert "active_income_state.json" not in archive.namelist()


def test_build_backup_archive_can_include_external_extra_files(tmp_path):
    config_path = write_config(tmp_path)
    external = tmp_path.parent / "operator-notes.txt"
    external.write_text("reviewed\n", encoding="utf-8")

    report = build_backup_archive(
        config_path=config_path,
        output=tmp_path / "backup.zip",
        extra_paths=[external],
        root=tmp_path,
    )

    extra_entry = next(
        item for item in report["manifest"]["files"] if item["path"] == str(external)
    )
    assert extra_entry["included"] is True
    assert extra_entry["arcname"].startswith("external/")


def test_build_backup_archive_skips_symlink_sources(tmp_path):
    config_path = write_config(tmp_path)
    target = tmp_path / "target-notes.txt"
    target.write_text("do not follow\n", encoding="utf-8")
    link = tmp_path / "operator-notes-link.txt"
    link.symlink_to(target)

    report = build_backup_archive(
        config_path=config_path,
        output=tmp_path / "backup.zip",
        extra_paths=[link],
        root=tmp_path,
    )

    link_entry = next(item for item in report["manifest"]["files"] if item["path"] == str(link))
    assert link_entry["included"] is False
    assert link_entry["reason"] == "symlink"
    assert link_entry["required_if_present"] is False
    assert report["manifest"]["critical_skipped_files"] == 0
    assert report["ok"] is True
    with zipfile.ZipFile(report["output"]) as archive:
        assert "operator-notes-link.txt" not in archive.namelist()


def test_build_backup_archive_rejects_symlink_output(tmp_path):
    config_path = write_config(tmp_path)
    target = tmp_path / "outside.zip"
    target.write_text("do not overwrite\n", encoding="utf-8")
    output = tmp_path / "backup-link.zip"
    output.symlink_to(target)

    with pytest.raises(ValueError, match="backup output must not be a symlink"):
        build_backup_archive(config_path=config_path, output=output, root=tmp_path)

    assert target.read_text(encoding="utf-8") == "do not overwrite\n"


def test_build_backup_archive_rejects_non_positive_size_limit(tmp_path):
    config_path = write_config(tmp_path)

    with pytest.raises(ValueError, match="max_file_bytes must be positive"):
        build_backup_archive(
            config_path=config_path, output=tmp_path / "backup.zip", max_file_bytes=0
        )


def test_backup_output_summary_omits_full_manifest():
    summary = backup_output_summary(
        {
            "ok": True,
            "output": "runtime/backups/backup.zip",
            "archive_size_bytes": 1234,
            "manifest": {
                "included_files": 3,
                "missing_files": 2,
                "skipped_files": 1,
                "optional_missing_files": 2,
                "critical_skipped_files": 0,
                "required_recovery_files": 5,
                "files": [{"path": "runtime/status.json"}],
            },
            "retention": {"deleted_archives": 1},
            "verification": {"ok": True, "checked_files": 3, "issues": []},
        }
    )

    assert summary == {
        "ok": True,
        "output": "runtime/backups/backup.zip",
        "archive_size_bytes": 1234,
        "included_files": 3,
        "missing_files": 2,
        "skipped_files": 1,
        "optional_missing_files": 2,
        "critical_skipped_files": 0,
        "required_recovery_files": 5,
        "experiment_memory_snapshot": None,
        "retention": {"deleted_archives": 1},
        "verification": {"ok": True, "checked_files": 3, "issues": 0},
    }


def test_verify_backup_archive_accepts_valid_archive(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "approvals.json").write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    output = tmp_path / "backup.zip"
    build_backup_archive(config_path=config_path, output=output, root=tmp_path)

    report = verify_backup_archive(output)

    assert report["ok"] is True
    assert report["checked_files"] >= 2
    assert report["issues"] == []


def test_verify_backup_archive_rejects_missing_declared_recovery_role(tmp_path):
    output = tmp_path / "backup.zip"
    manifest = {
        "version": 1,
        "included_files": 0,
        "missing_files": 0,
        "skipped_files": 0,
        "optional_missing_files": 0,
        "critical_skipped_files": 0,
        "required_recovery_files": 1,
        "required_recovery_roles": ["autopilot_config"],
        "files": [],
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("MANIFEST.json", json.dumps(manifest))

    report = verify_backup_archive(output)

    assert report["ok"] is False
    assert {
        "code": "required_recovery_count_mismatch",
        "manifest_count": 1,
        "actual_count": 0,
    } in report["issues"]
    assert {
        "code": "required_recovery_roles_mismatch",
        "missing_roles": ["autopilot_config"],
        "unexpected_roles": [],
    } in report["issues"]


def test_verify_backup_archive_rejects_missing_manifest(tmp_path):
    output = tmp_path / "backup.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("status.json", "{}")

    report = verify_backup_archive(output)

    assert report["ok"] is False
    assert report["issues"] == [{"code": "missing_manifest", "message": "MANIFEST.json is missing"}]


def test_verify_backup_archive_rejects_hash_mismatch(tmp_path):
    output = tmp_path / "backup.zip"
    manifest = {
        "version": 1,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "included_files": 1,
        "missing_files": 0,
        "skipped_files": 0,
        "files": [
            {
                "path": "runtime/status.json",
                "arcname": "runtime/status.json",
                "exists": True,
                "included": True,
                "size_bytes": 2,
                "sha256": "wrong",
            }
        ],
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("runtime/status.json", "{}")
        archive.writestr("MANIFEST.json", json.dumps(manifest))

    report = verify_backup_archive(output)

    assert report["ok"] is False
    assert report["issues"][0]["code"] == "sha256_mismatch"
    assert report["issues"][0]["arcname"] == "runtime/status.json"


def test_verify_backup_archive_rejects_missing_included_member(tmp_path):
    output = tmp_path / "backup.zip"
    manifest = {
        "version": 1,
        "included_files": 1,
        "missing_files": 0,
        "skipped_files": 0,
        "files": [
            {
                "path": "runtime/status.json",
                "arcname": "runtime/status.json",
                "exists": True,
                "included": True,
                "size_bytes": 2,
                "sha256": "unused",
            }
        ],
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("MANIFEST.json", json.dumps(manifest))

    report = verify_backup_archive(output)

    assert report["ok"] is False
    assert report["issues"] == [{"code": "missing_member", "arcname": "runtime/status.json"}]


def test_verify_backup_archive_rejects_unsupported_manifest_version(tmp_path):
    output = tmp_path / "backup.zip"
    payload = b"{}"
    manifest = {
        "version": 2,
        "included_files": 1,
        "missing_files": 0,
        "skipped_files": 0,
        "files": [
            {
                "path": "runtime/status.json",
                "arcname": "runtime/status.json",
                "exists": True,
                "included": True,
                "size_bytes": len(payload),
                "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            }
        ],
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("runtime/status.json", payload)
        archive.writestr("MANIFEST.json", json.dumps(manifest))

    report = verify_backup_archive(output)

    assert report["ok"] is False
    assert {
        "code": "unsupported_manifest_version",
        "version": 2,
        "supported_version": 1,
    } in report["issues"]
    with pytest.raises(ValueError, match="backup verification failed"):
        restore_backup_archive(output, tmp_path / "restore")


def test_verify_backup_archive_rejects_unexpected_archive_member(tmp_path):
    output = tmp_path / "backup.zip"
    payload = b"{}"
    manifest = {
        "version": 1,
        "included_files": 1,
        "missing_files": 0,
        "skipped_files": 0,
        "files": [
            {
                "path": "runtime/status.json",
                "arcname": "runtime/status.json",
                "exists": True,
                "included": True,
                "size_bytes": len(payload),
                "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            }
        ],
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("MANIFEST.json", json.dumps(manifest))
        archive.writestr("runtime/status.json", payload)
        archive.writestr("runtime/unexpected.json", "{}")

    report = verify_backup_archive(output)

    assert report["ok"] is False
    assert {"code": "unexpected_member", "arcname": "runtime/unexpected.json"} in report["issues"]


@pytest.mark.parametrize(
    "arcname", ["../status.json", "/runtime/status.json", "runtime/../../status.json"]
)
def test_verify_backup_archive_rejects_unsafe_manifest_arcname(tmp_path, arcname):
    output = tmp_path / "backup.zip"
    payload = b"{}"
    manifest = {
        "version": 1,
        "included_files": 1,
        "missing_files": 0,
        "skipped_files": 0,
        "files": [
            {
                "path": "runtime/status.json",
                "arcname": arcname,
                "exists": True,
                "included": True,
                "size_bytes": len(payload),
                "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            }
        ],
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(arcname, payload)
        archive.writestr("MANIFEST.json", json.dumps(manifest))

    report = verify_backup_archive(output)

    assert report["ok"] is False
    assert {"code": "unsafe_arcname", "arcname": arcname} in report["issues"]


def test_verify_backup_archive_rejects_duplicate_manifest_arcname(tmp_path):
    output = tmp_path / "backup.zip"
    manifest = {
        "version": 1,
        "included_files": 2,
        "missing_files": 0,
        "skipped_files": 0,
        "files": [
            {
                "path": "runtime/status.json",
                "arcname": "runtime/status.json",
                "exists": True,
                "included": True,
                "size_bytes": 2,
                "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            },
            {
                "path": "runtime/status-copy.json",
                "arcname": "runtime/status.json",
                "exists": True,
                "included": True,
                "size_bytes": 2,
                "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            },
        ],
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("runtime/status.json", "{}")
        archive.writestr("MANIFEST.json", json.dumps(manifest))

    report = verify_backup_archive(output)

    assert report["ok"] is False
    assert {"code": "duplicate_manifest_arcname", "arcname": "runtime/status.json"} in report[
        "issues"
    ]


def test_verify_backup_archive_rejects_duplicate_archive_member(tmp_path):
    output = tmp_path / "backup.zip"
    manifest = {
        "version": 1,
        "included_files": 1,
        "missing_files": 0,
        "skipped_files": 0,
        "files": [
            {
                "path": "runtime/status.json",
                "arcname": "runtime/status.json",
                "exists": True,
                "included": True,
                "size_bytes": 2,
                "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            }
        ],
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("runtime/status.json", "{}")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("runtime/status.json", "{}")
        archive.writestr("MANIFEST.json", json.dumps(manifest))

    report = verify_backup_archive(output)

    assert report["ok"] is False
    assert {"code": "duplicate_archive_member", "arcname": "runtime/status.json"} in report[
        "issues"
    ]


def test_restore_backup_archive_extracts_verified_files(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "approvals.json").write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    (tmp_path / "active_income_state.json").write_text('{"equity": 1000}\n', encoding="utf-8")
    archive_path = tmp_path / "backup.zip"
    build_backup_archive(config_path=config_path, output=archive_path, root=tmp_path)
    restore_dir = tmp_path / "restore"

    report = restore_backup_archive(archive_path, restore_dir)

    assert report["ok"] is True
    assert (restore_dir / "approvals.json").read_text(
        encoding="utf-8"
    ) == '{"version": 1, "approvals": {}}\n'
    assert (restore_dir / "active_income_state.json").read_text(
        encoding="utf-8"
    ) == '{"equity": 1000}\n'
    assert (restore_dir / "RESTORE_REPORT.json").exists()


def test_restore_forces_private_directory_and_file_modes_with_permissive_umask(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "approvals.json").write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    archive_path = tmp_path / "backup.zip"
    build_backup_archive(config_path=config_path, output=archive_path, root=tmp_path)
    restore_dir = tmp_path / "restore"
    restore_dir.mkdir(mode=0o777)
    restore_dir.chmod(0o777)

    previous = os.umask(0)
    try:
        report = restore_backup_archive(archive_path, restore_dir)
    finally:
        os.umask(previous)

    assert report["ok"] is True
    assert stat.S_IMODE(restore_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((restore_dir / "approvals.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((restore_dir / "RESTORE_REPORT.json").stat().st_mode) == 0o600


def test_restore_backup_archive_refuses_to_overwrite_existing_files(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "approvals.json").write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    archive_path = tmp_path / "backup.zip"
    build_backup_archive(config_path=config_path, output=archive_path, root=tmp_path)
    restore_dir = tmp_path / "restore"
    restore_dir.mkdir()
    (restore_dir / "approvals.json").write_text("existing\n", encoding="utf-8")

    report = restore_backup_archive(archive_path, restore_dir)

    assert report["ok"] is False
    assert report["reason"] == "target_exists"
    assert report["conflicts"] == [
        {"arcname": "approvals.json", "target": str(restore_dir / "approvals.json")}
    ]
    assert (restore_dir / "approvals.json").read_text(encoding="utf-8") == "existing\n"


def test_restore_backup_archive_can_overwrite_when_requested(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "approvals.json").write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    archive_path = tmp_path / "backup.zip"
    build_backup_archive(config_path=config_path, output=archive_path, root=tmp_path)
    restore_dir = tmp_path / "restore"
    restore_dir.mkdir()
    (restore_dir / "approvals.json").write_text("existing\n", encoding="utf-8")
    (restore_dir / "approvals.json").chmod(0o644)

    report = restore_backup_archive(archive_path, restore_dir, overwrite=True)

    assert report["ok"] is True
    assert (restore_dir / "approvals.json").read_text(
        encoding="utf-8"
    ) == '{"version": 1, "approvals": {}}\n'
    assert stat.S_IMODE((restore_dir / "approvals.json").stat().st_mode) == 0o600


def test_restore_backup_archive_rejects_symlink_escape_when_overwriting(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "approvals.json").write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    archive_path = tmp_path / "backup.zip"
    build_backup_archive(config_path=config_path, output=archive_path, root=tmp_path)
    restore_dir = tmp_path / "restore"
    restore_dir.mkdir()
    external_target = tmp_path / "external_approvals.json"
    external_target.write_text("outside\n", encoding="utf-8")
    (restore_dir / "approvals.json").symlink_to(external_target)

    with pytest.raises(ValueError, match="restore target escapes restore dir"):
        restore_backup_archive(archive_path, restore_dir, overwrite=True)

    assert external_target.read_text(encoding="utf-8") == "outside\n"


def test_restore_backup_archive_rejects_symlink_restore_root(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "approvals.json").write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    archive_path = tmp_path / "backup.zip"
    build_backup_archive(config_path=config_path, output=archive_path, root=tmp_path)
    external_restore = tmp_path / "external_restore"
    external_restore.mkdir()
    restore_dir = tmp_path / "restore"
    restore_dir.symlink_to(external_restore, target_is_directory=True)

    with pytest.raises(ValueError, match="restore_dir must not be a symlink"):
        restore_backup_archive(archive_path, restore_dir)

    assert not (external_restore / "approvals.json").exists()


def test_restore_backup_archive_rejects_symlink_target_inside_restore_dir(tmp_path):
    config_path = write_config(tmp_path)
    (tmp_path / "approvals.json").write_text('{"version": 1, "approvals": {}}\n', encoding="utf-8")
    archive_path = tmp_path / "backup.zip"
    build_backup_archive(config_path=config_path, output=archive_path, root=tmp_path)
    restore_dir = tmp_path / "restore"
    restore_dir.mkdir()
    linked_target = restore_dir / "linked_approvals.json"
    linked_target.write_text("inside\n", encoding="utf-8")
    (restore_dir / "approvals.json").symlink_to(linked_target)

    with pytest.raises(ValueError, match="restore target is a symlink"):
        restore_backup_archive(archive_path, restore_dir, overwrite=True)

    assert linked_target.read_text(encoding="utf-8") == "inside\n"


def test_restore_backup_archive_rejects_failed_verification(tmp_path):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("status.json", "{}")

    with pytest.raises(ValueError, match="backup verification failed"):
        restore_backup_archive(archive_path, tmp_path / "restore")


def test_backup_cli_prints_json_when_restore_verification_fails(monkeypatch, tmp_path, capsys):
    archive_path = tmp_path / "bad.zip"
    archive_path.write_text("not a zip", encoding="utf-8")
    report_path = tmp_path / "restore_report.json"
    restore_dir = tmp_path / "restore"
    monkeypatch.setattr(
        "sys.argv",
        [
            "backup",
            "--restore",
            str(archive_path),
            "--restore-dir",
            str(restore_dir),
            "--report",
            str(report_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "ok": False,
        "archive": str(archive_path),
        "restore_dir": str(restore_dir),
        "error": f"ValueError: backup verification failed: {archive_path}",
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == printed


@pytest.mark.parametrize("arcname", ["../evil", ".."])
def test_restore_backup_archive_rejects_unsafe_member_paths(tmp_path, arcname):
    archive_path = tmp_path / "unsafe.zip"
    manifest = {
        "version": 1,
        "included_files": 1,
        "missing_files": 0,
        "skipped_files": 0,
        "files": [
            {
                "path": "../evil",
                "arcname": arcname,
                "exists": True,
                "included": True,
                "size_bytes": 2,
                "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            }
        ],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(arcname, "{}")
        archive.writestr("MANIFEST.json", json.dumps(manifest))

    with pytest.raises(ValueError, match="backup verification failed"):
        restore_backup_archive(archive_path, tmp_path / "restore")


def test_prune_backup_archives_keeps_newest_generated_archives(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    old = backup_dir / "autopilot_state_20260101T000000Z.zip"
    middle = backup_dir / "autopilot_state_20260102T000000Z.zip"
    new = backup_dir / "autopilot_state_20260103T000000Z.zip"
    unrelated = backup_dir / "manual.zip"
    for path in (old, middle, new, unrelated):
        path.write_text(path.name, encoding="utf-8")
    old.touch()
    middle.touch()
    new.touch()
    import os

    os.utime(old, (100, 100))
    os.utime(middle, (200, 200))
    os.utime(new, (300, 300))

    report = prune_backup_archives(backup_dir, keep=2)

    assert report["archives"] == 3
    assert report["deleted_archives"] == 1
    assert report["deleted"][0]["path"] == str(old)
    assert not old.exists()
    assert middle.exists()
    assert new.exists()
    assert unrelated.exists()


def test_prune_backup_archives_dry_run_does_not_delete(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    first = backup_dir / "autopilot_state_20260101T000000Z.zip"
    second = backup_dir / "autopilot_state_20260102T000000Z.zip"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    report = prune_backup_archives(backup_dir, keep=1, dry_run=True)

    assert report["deleted_archives"] == 1
    assert report["changed"] is False
    assert first.exists()
    assert second.exists()


def test_prune_backup_archives_rejects_non_positive_keep(tmp_path):
    with pytest.raises(ValueError, match="keep must be positive"):
        prune_backup_archives(tmp_path, keep=0)
