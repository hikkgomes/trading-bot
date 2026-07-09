import gzip
import json

import pytest

from src.autopilot.maintenance import (
    compact_alert_state,
    compact_jsonl,
    main,
    prune_directory_by_size,
    rotate_jsonl,
    run_maintenance,
)


def test_compact_jsonl_missing_file_is_noop(tmp_path):
    path = tmp_path / "alerts.jsonl"

    report = compact_jsonl(path, max_lines=3)

    assert report["exists"] is False
    assert report["changed"] is False
    assert not path.exists()


def test_compact_jsonl_under_limit_is_noop(tmp_path):
    path = tmp_path / "alerts.jsonl"
    path.write_text('{"n": 1}\n{"n": 2}\n', encoding="utf-8")

    report = compact_jsonl(path, max_lines=3)

    assert report["line_count"] == 2
    assert report["trimmed_lines"] == 0
    assert report["would_trim"] is False
    assert report["changed"] is False
    assert path.read_text(encoding="utf-8") == '{"n": 1}\n{"n": 2}\n'


def test_compact_jsonl_trims_to_recent_lines(tmp_path):
    path = tmp_path / "alerts.jsonl"
    path.write_text("\n".join(f'{{"n": {idx}}}' for idx in range(5)) + "\n", encoding="utf-8")

    report = compact_jsonl(path, max_lines=2)

    assert report["line_count"] == 5
    assert report["kept_lines"] == 2
    assert report["trimmed_lines"] == 3
    assert report["would_trim"] is True
    assert report["changed"] is True
    assert path.read_text(encoding="utf-8") == '{"n": 3}\n{"n": 4}\n'


def test_compact_jsonl_dry_run_reports_without_writing(tmp_path):
    path = tmp_path / "alerts.jsonl"
    original = '{"n": 1}\n{"n": 2}\n{"n": 3}\n'
    path.write_text(original, encoding="utf-8")

    report = compact_jsonl(path, max_lines=1, dry_run=True)

    assert report["trimmed_lines"] == 2
    assert report["would_trim"] is True
    assert report["changed"] is False
    assert path.read_text(encoding="utf-8") == original


def test_compact_jsonl_rejects_non_positive_limit(tmp_path):
    with pytest.raises(ValueError, match="max_lines must be positive"):
        compact_jsonl(tmp_path / "alerts.jsonl", max_lines=0)


def test_compact_jsonl_rejects_symlink_source(tmp_path):
    target = tmp_path / "outside_alerts.jsonl"
    target.write_text('{"n": 1}\n{"n": 2}\n', encoding="utf-8")
    link = tmp_path / "alerts.jsonl"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="jsonl path must not be a symlink"):
        compact_jsonl(link, max_lines=1)

    assert target.read_text(encoding="utf-8") == '{"n": 1}\n{"n": 2}\n'


def test_rotate_jsonl_archives_old_lines_and_keeps_recent(tmp_path):
    path = tmp_path / "experiment_log.jsonl"
    archive_dir = tmp_path / "archive"
    path.write_text("\n".join(f'{{"n": {idx}}}' for idx in range(5)) + "\n", encoding="utf-8")

    report = rotate_jsonl(path, max_lines=2, archive_dir=archive_dir)

    assert report["line_count"] == 5
    assert report["kept_lines"] == 2
    assert report["archived_lines"] == 3
    assert report["would_rotate"] is True
    assert report["changed"] is True
    assert path.read_text(encoding="utf-8") == '{"n": 3}\n{"n": 4}\n'
    archive_path = archive_dir / report["archive_path"].split("/")[-1]
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        assert handle.read() == '{"n": 0}\n{"n": 1}\n{"n": 2}\n'


def test_rotate_jsonl_dry_run_does_not_write_archive_or_trim(tmp_path):
    path = tmp_path / "experiment_log.jsonl"
    archive_dir = tmp_path / "archive"
    original = '{"n": 1}\n{"n": 2}\n{"n": 3}\n'
    path.write_text(original, encoding="utf-8")

    report = rotate_jsonl(path, max_lines=1, archive_dir=archive_dir, dry_run=True)

    assert report["archived_lines"] == 2
    assert report["would_rotate"] is True
    assert report["changed"] is False
    assert path.read_text(encoding="utf-8") == original
    assert not archive_dir.exists()


def test_rotate_jsonl_under_limit_is_noop(tmp_path):
    path = tmp_path / "experiment_log.jsonl"
    archive_dir = tmp_path / "archive"
    path.write_text('{"n": 1}\n', encoding="utf-8")

    report = rotate_jsonl(path, max_lines=2, archive_dir=archive_dir)

    assert report["line_count"] == 1
    assert report["archived_lines"] == 0
    assert report["changed"] is False
    assert not archive_dir.exists()


def test_rotate_jsonl_rejects_symlink_source(tmp_path):
    target = tmp_path / "outside_control_audit.jsonl"
    target.write_text('{"n": 1}\n{"n": 2}\n{"n": 3}\n', encoding="utf-8")
    link = tmp_path / "control_audit.jsonl"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="jsonl path must not be a symlink"):
        rotate_jsonl(link, max_lines=1, archive_dir=tmp_path / "archive")

    assert target.read_text(encoding="utf-8") == '{"n": 1}\n{"n": 2}\n{"n": 3}\n'
    assert not (tmp_path / "archive").exists()


def test_rotate_jsonl_skips_symlink_archive_candidate_without_touching_target(
    tmp_path, monkeypatch
):
    path = tmp_path / "experiment_log.jsonl"
    archive_dir = tmp_path / "archive"
    external_target = tmp_path / "external_archive.jsonl.gz"
    path.write_text('{"n": 1}\n{"n": 2}\n{"n": 3}\n', encoding="utf-8")
    archive_dir.mkdir()
    first_candidate = archive_dir / "experiment_log.20260709T171900Z.jsonl.gz"
    first_candidate.symlink_to(external_target)
    monkeypatch.setattr(
        "src.autopilot.maintenance.time.strftime",
        lambda *_args: "20260709T171900Z",
    )

    report = rotate_jsonl(path, max_lines=1, archive_dir=archive_dir)

    archive_path = archive_dir / "experiment_log.20260709T171900Z.1.jsonl.gz"
    assert report["archive_path"] == str(archive_path)
    assert first_candidate.is_symlink()
    assert not external_target.exists()
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        assert handle.read() == '{"n": 1}\n{"n": 2}\n'
    assert path.read_text(encoding="utf-8") == '{"n": 3}\n'


def test_compact_alert_state_keeps_newest_fingerprints(tmp_path):
    path = tmp_path / "alert_state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "alerts": {
                    "old": {"last_sent_ts": 100.0, "title": "old"},
                    "new": {"last_sent_ts": 300.0, "title": "new"},
                    "middle": {"last_sent_ts": 200.0, "title": "middle"},
                },
            }
        ),
        encoding="utf-8",
    )

    report = compact_alert_state(path, max_fingerprints=2)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert report["alert_count"] == 3
    assert report["kept_alerts"] == 2
    assert report["pruned_alerts"] == 1
    assert report["changed"] is True
    assert set(payload["alerts"]) == {"new", "middle"}
    assert payload["version"] == 1


def test_compact_alert_state_dry_run_reports_without_writing(tmp_path):
    path = tmp_path / "alert_state.json"
    original = {
        "version": 1,
        "alerts": {
            "old": {"last_sent_ts": 100.0},
            "new": {"last_sent_ts": 200.0},
        },
    }
    path.write_text(json.dumps(original), encoding="utf-8")

    report = compact_alert_state(path, max_fingerprints=1, dry_run=True)

    assert report["pruned_alerts"] == 1
    assert report["changed"] is False
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_compact_alert_state_prunes_invalid_entries_without_crashing(tmp_path, monkeypatch):
    path = tmp_path / "alert_state.json"
    monkeypatch.setattr("src.autopilot.maintenance.time.time", lambda: 1000.0)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "alerts": {
                    "valid": {"last_sent_ts": 900.0},
                    "bad_ts": {"last_sent_ts": "not-a-number"},
                    "negative_ts": {"last_sent_ts": -1.0},
                    "future_ts": {"last_sent_ts": 2000.0},
                    "bad_entry": "sent",
                },
            }
        ),
        encoding="utf-8",
    )

    report = compact_alert_state(path, max_fingerprints=10)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert report["alert_count"] == 5
    assert report["valid_alerts"] == 1
    assert report["invalid_alerts"] == 4
    assert report["pruned_alerts"] == 0
    assert report["would_prune"] is True
    assert report["changed"] is True
    assert payload["alerts"] == {"valid": {"last_sent_ts": 900.0}}


def test_compact_alert_state_dry_run_keeps_invalid_entries(tmp_path, monkeypatch):
    path = tmp_path / "alert_state.json"
    monkeypatch.setattr("src.autopilot.maintenance.time.time", lambda: 1000.0)
    original = {
        "version": 1,
        "alerts": {
            "valid": {"last_sent_ts": 900.0},
            "future_ts": {"last_sent_ts": 2000.0},
        },
    }
    path.write_text(json.dumps(original), encoding="utf-8")

    report = compact_alert_state(path, max_fingerprints=10, dry_run=True)

    assert report["invalid_alerts"] == 1
    assert report["would_prune"] is True
    assert report["changed"] is False
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_compact_alert_state_rejects_non_object_state(tmp_path):
    path = tmp_path / "alert_state.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="alert state must be a JSON object"):
        compact_alert_state(path, max_fingerprints=10)


def test_compact_alert_state_rejects_non_positive_limit(tmp_path):
    with pytest.raises(ValueError, match="max_fingerprints must be positive"):
        compact_alert_state(tmp_path / "alert_state.json", max_fingerprints=0)


def test_compact_alert_state_rejects_symlink_source(tmp_path):
    target = tmp_path / "outside_alert_state.json"
    original = {
        "version": 1,
        "alerts": {
            "old": {"last_sent_ts": 100.0},
            "new": {"last_sent_ts": 200.0},
        },
    }
    target.write_text(json.dumps(original), encoding="utf-8")
    link = tmp_path / "alert_state.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="alert state path must not be a symlink"):
        compact_alert_state(link, max_fingerprints=1)

    assert json.loads(target.read_text(encoding="utf-8")) == original


def test_run_maintenance_rotates_control_audit_from_config(tmp_path):
    config = tmp_path / "autopilot.json"
    alert_file = tmp_path / "alerts.jsonl"
    alert_state_file = tmp_path / "alert_state.json"
    control_audit = tmp_path / "control_audit.jsonl"
    experiment_log = tmp_path / "experiment_log.jsonl"
    audit_archive = tmp_path / "audit_archive"
    experiment_archive = tmp_path / "experiment_archive"
    config.write_text(
        json.dumps(
            {
                "alert_file": str(alert_file),
                "alert_state_file": str(alert_state_file),
                "control_audit_file": str(control_audit),
                "jobs": [],
                "products": [],
            }
        ),
        encoding="utf-8",
    )
    alert_file.write_text('{"n": 1}\n', encoding="utf-8")
    alert_state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "alerts": {
                    "old": {"last_sent_ts": 100.0},
                    "new": {"last_sent_ts": 200.0},
                },
            }
        ),
        encoding="utf-8",
    )
    control_audit.write_text("\n".join(f'{{"n": {idx}}}' for idx in range(5)) + "\n", encoding="utf-8")
    experiment_log.write_text('{"n": 1}\n', encoding="utf-8")

    report = run_maintenance(
        config,
        max_alert_lines=10,
        max_alert_fingerprints=1,
        max_experiment_lines=10,
        max_control_audit_lines=2,
        experiment_log=experiment_log,
        experiment_archive_dir=experiment_archive,
        control_audit_archive_dir=audit_archive,
    )

    assert report["alert_state"]["pruned_alerts"] == 1
    assert set(json.loads(alert_state_file.read_text(encoding="utf-8"))["alerts"]) == {"new"}
    assert report["control_audit"]["archived_lines"] == 3
    assert report["control_audit"]["kept_lines"] == 2
    assert control_audit.read_text(encoding="utf-8") == '{"n": 3}\n{"n": 4}\n'
    archive_path = audit_archive / report["control_audit"]["archive_path"].split("/")[-1]
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        assert handle.read() == '{"n": 0}\n{"n": 1}\n{"n": 2}\n'


def test_run_maintenance_continues_after_task_failure(tmp_path):
    config = tmp_path / "autopilot.json"
    alert_file = tmp_path / "alerts.jsonl"
    alert_state_target = tmp_path / "outside_alert_state.json"
    alert_state_link = tmp_path / "alert_state.json"
    control_audit = tmp_path / "control_audit.jsonl"
    experiment_log = tmp_path / "experiment_log.jsonl"
    audit_archive = tmp_path / "audit_archive"
    experiment_archive = tmp_path / "experiment_archive"
    config.write_text(
        json.dumps(
            {
                "alert_file": str(alert_file),
                "alert_state_file": str(alert_state_link),
                "control_audit_file": str(control_audit),
                "jobs": [],
                "products": [],
            }
        ),
        encoding="utf-8",
    )
    alert_file.write_text('{"n": 1}\n{"n": 2}\n{"n": 3}\n', encoding="utf-8")
    alert_state_target.write_text(json.dumps({"version": 1, "alerts": {}}), encoding="utf-8")
    alert_state_link.symlink_to(alert_state_target)
    control_audit.write_text("\n".join(f'{{"n": {idx}}}' for idx in range(4)) + "\n", encoding="utf-8")
    experiment_log.write_text("\n".join(f'{{"n": {idx}}}' for idx in range(3)) + "\n", encoding="utf-8")

    report = run_maintenance(
        config,
        max_alert_lines=2,
        max_alert_fingerprints=1,
        max_experiment_lines=1,
        max_control_audit_lines=2,
        experiment_log=experiment_log,
        experiment_archive_dir=experiment_archive,
        control_audit_archive_dir=audit_archive,
    )

    assert report["ok"] is False
    assert report["errors"] == [
        {
            "task": "alert_state",
            "error": f"ValueError: alert state path must not be a symlink: {alert_state_link}",
        }
    ]
    assert report["alert_state"]["ok"] is False
    assert report["alerts"]["changed"] is True
    assert alert_file.read_text(encoding="utf-8") == '{"n": 2}\n{"n": 3}\n'
    assert report["experiment_log"]["archived_lines"] == 2
    assert report["control_audit"]["archived_lines"] == 2
    assert control_audit.read_text(encoding="utf-8") == '{"n": 2}\n{"n": 3}\n'
    assert json.loads(alert_state_target.read_text(encoding="utf-8")) == {"version": 1, "alerts": {}}


def test_maintenance_cli_exits_nonzero_for_structured_task_failure(monkeypatch, tmp_path, capsys):
    config = tmp_path / "autopilot.json"
    alert_file = tmp_path / "alerts.jsonl"
    alert_state_target = tmp_path / "outside_alert_state.json"
    alert_state_link = tmp_path / "alert_state.json"
    config.write_text(
        json.dumps(
            {
                "alert_file": str(alert_file),
                "alert_state_file": str(alert_state_link),
                "control_audit_file": str(tmp_path / "control_audit.jsonl"),
                "jobs": [],
                "products": [],
            }
        ),
        encoding="utf-8",
    )
    alert_state_target.write_text(json.dumps({"version": 1, "alerts": {}}), encoding="utf-8")
    alert_state_link.symlink_to(alert_state_target)
    monkeypatch.setattr(
        "sys.argv",
        [
            "maintenance",
            "--config",
            str(config),
            "--experiment-log",
            str(tmp_path / "experiment_log.jsonl"),
            "--max-alert-fingerprints",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["errors"][0]["task"] == "alert_state"
    assert "must not be a symlink" in printed["errors"][0]["error"]


def test_prune_directory_by_size_deletes_oldest_files_until_under_budget(tmp_path):
    root = tmp_path / "quarantine"
    nested = root / "old_search"
    nested.mkdir(parents=True)
    old = nested / "old.csv"
    middle = root / "middle.csv"
    new = root / "new.csv"
    old.write_text("o" * 10, encoding="utf-8")
    middle.write_text("m" * 10, encoding="utf-8")
    new.write_text("n" * 10, encoding="utf-8")
    # Explicit mtimes make the deletion order deterministic.
    old.touch()
    middle.touch()
    new.touch()
    import os

    os.utime(old, (100, 100))
    os.utime(middle, (200, 200))
    os.utime(new, (300, 300))

    report = prune_directory_by_size(root, max_bytes=15)

    assert report["initial_bytes"] == 30
    assert report["final_bytes"] == 10
    assert report["deleted_files"] == 2
    assert [item["path"] for item in report["deleted"]] == [str(old), str(middle)]
    assert not old.exists()
    assert not middle.exists()
    assert new.exists()
    assert not nested.exists()


def test_prune_directory_by_size_dry_run_reports_without_deleting(tmp_path):
    root = tmp_path / "quarantine"
    root.mkdir()
    first = root / "a.csv"
    second = root / "b.csv"
    first.write_text("a" * 10, encoding="utf-8")
    second.write_text("b" * 10, encoding="utf-8")

    report = prune_directory_by_size(root, max_bytes=5, dry_run=True)

    assert report["deleted_files"] == 2
    assert report["changed"] is False
    assert first.exists()
    assert second.exists()


def test_prune_directory_by_size_rejects_symlink_root(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "artifact.json").write_text("x" * 10, encoding="utf-8")
    root = tmp_path / "quarantine"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="root must not be a symlink"):
        prune_directory_by_size(root, max_bytes=1)

    assert (target / "artifact.json").exists()


def test_prune_directory_by_size_ignores_symlink_entries(tmp_path):
    root = tmp_path / "quarantine"
    root.mkdir()
    linked_target = tmp_path / "external.json"
    linked_target.write_text("x" * 100, encoding="utf-8")
    link = root / "linked.json"
    link.symlink_to(linked_target)
    real = root / "real.json"
    real.write_text("r" * 10, encoding="utf-8")

    report = prune_directory_by_size(root, max_bytes=5)

    assert report["initial_bytes"] == 10
    assert report["deleted_files"] == 1
    assert report["deleted"] == [{"path": str(real), "size_bytes": 10}]
    assert not real.exists()
    assert link.is_symlink()
    assert linked_target.read_text(encoding="utf-8") == "x" * 100
