import json
import os
from pathlib import Path

import pytest

from src.autopilot import openclaw_bridge as openclaw_bridge_module
from src.autopilot.openclaw_bridge import (
    ACCEPTED_SCHEMA,
    ProposalValidationError,
    build_accepted_proposal,
    build_research_context,
    canonical_proposal_digest,
    export_research_context,
    ingest_inbox,
    record_review,
    validate_proposal,
)


def test_record_review_logs_zero_proposal_receipt(tmp_path):
    context = tmp_path / "research_context.json"
    context.write_text('{"schema":"autopilot.openclaw_research_context/v1"}\n')
    audit = tmp_path / "review_audit.jsonl"

    receipt = record_review(
        audit_path=audit,
        context_path=context,
        run_id="daily-2026-07-19",
        model="openclaw-default",
        summary="No novel hypothesis met the bar today.",
        proposal_count=0,
    )

    assert receipt["proposal_count"] == 0
    assert receipt["context_digest"].startswith("sha256:")
    assert json.loads(audit.read_text())["run_id"] == "daily-2026-07-19"


def proposal(**overrides):
    payload = {
        "schema": "research_proposal/v1",
        "source": "openclaw",
        "created_at": "2026-07-10T12:00:00+00:00",
        "objective": "active_income",
        "opportunity_type": "day",
        "base_timeframe": "15m",
        "thesis": "A volatility expansion after a quiet regime may carry short-term continuation.",
        "suggested_primitives": ["volatility percentile", "range expansion"],
        "constraints": ["avoid high funding windows"],
        "suggested_spec": {
            "regime": {"feature": "atr_percentile", "condition": "low_then_rising"},
            "entry": {"feature": "range_break"},
        },
        "provenance": {"agent": "researcher", "model": "local-model"},
        "source_proposal_id": "session-42:idea-7",
    }
    payload.update(overrides)
    return payload


def inbox_paths(tmp_path):
    root = tmp_path / "openclaw"
    return {
        "incoming_dir": root / "incoming",
        "accepted_dir": root / "accepted",
        "rejected_dir": root / "rejected",
        "archive_dir": root / "archive",
        "index_path": root / "index.json",
    }


def emulate_linux_directory_modes(monkeypatch, *, preset=None, immutable=()):
    """Preserve requested setgid bits on test filesystems that strip them."""

    remembered = dict(preset or {})
    immutable = set(immutable)
    real_fstat = os.fstat
    real_fchmod = os.fchmod

    def linux_fstat(descriptor):
        result = real_fstat(descriptor)
        mode = remembered.get((result.st_dev, result.st_ino))
        if mode is None:
            return result
        values = list(result)
        values[0] = (values[0] & ~0o7777) | mode
        return os.stat_result(values)

    def linux_fchmod(descriptor, mode):
        result = real_fstat(descriptor)
        identity = (result.st_dev, result.st_ino)
        if identity in immutable:
            raise PermissionError("systemd bind-mount root metadata is immutable")
        real_fchmod(descriptor, mode)
        remembered[identity] = mode

    monkeypatch.setattr(os, "fstat", linux_fstat)
    monkeypatch.setattr(os, "fchmod", linux_fchmod)


def test_accepted_proposal_is_inert_digest_bound_and_keeps_provenance():
    accepted = build_accepted_proposal(proposal())

    assert accepted["schema"] == ACCEPTED_SCHEMA
    assert accepted["proposal_id"].startswith("openclaw-")
    assert accepted["content_digest"].startswith("sha256:")
    assert accepted["untrusted_suggested_spec"]["entry"]["feature"] == "range_break"
    assert "suggested_spec" not in accepted
    assert accepted["provenance"] == {"agent": "researcher", "model": "local-model"}
    assert accepted["source_proposal_id"] == "session-42:idea-7"
    assert accepted["safety"] == {
        "research_only": True,
        "executable": False,
        "paper_trade_allowed": False,
        "promotion_allowed": False,
        "live_allowed": False,
        "requires_trusted_compilation": True,
        "requires_full_validation_before_export": True,
    }


def test_semantic_digest_ignores_provenance_identity_and_creation_time():
    first = validate_proposal(proposal())
    second = validate_proposal(
        proposal(
            created_at="2026-07-11T12:00:00Z",
            provenance={"agent": "another-agent"},
            source_proposal_id="different-id",
        )
    )

    assert canonical_proposal_digest(first) == canonical_proposal_digest(second)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"live_allowed": True}, "unknown fields"),
        ({"approval": "please approve"}, "unknown fields"),
        ({"source": "operator"}, "source must be 'openclaw'"),
        ({"objective": "trade_everything"}, "objective must be one of"),
        ({"opportunity_type": "ultra_hft"}, "opportunity_type must be one of"),
        ({"base_timeframe": "tick"}, "base_timeframe must be one of"),
        ({"thesis": "too short"}, "thesis length"),
        ({"created_at": "2026-07-10"}, "include a timezone"),
    ],
)
def test_untrusted_proposal_schema_is_strict(change, message):
    with pytest.raises(ProposalValidationError, match=message):
        validate_proposal(proposal(**change))


def test_suggested_spec_rejects_credentials_or_approval_controls():
    with pytest.raises(ProposalValidationError, match="forbidden security/control field"):
        validate_proposal(
            proposal(suggested_spec={"entry": {"api_key": "steal-me", "signal": "cross"}})
        )
    with pytest.raises(ProposalValidationError, match="forbidden security/control field"):
        validate_proposal(proposal(suggested_spec={"approval_override": True}))
    with pytest.raises(ProposalValidationError, match="forbidden security/control field"):
        validate_proposal(proposal(suggested_spec={"live_allowed": True}))


def test_ingest_accepts_archives_and_deduplicates_without_executing(tmp_path):
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    first = paths["incoming_dir"] / "idea-1.json"
    first.write_text(json.dumps(proposal()), encoding="utf-8")

    report = ingest_inbox(**paths)

    assert report["ok"] is True
    assert len(report["accepted"]) == 1
    accepted_files = list(paths["accepted_dir"].glob("*.json"))
    assert len(accepted_files) == 1
    accepted = json.loads(accepted_files[0].read_text())
    assert accepted["safety"]["executable"] is False
    assert not first.exists()
    assert len(list(paths["archive_dir"].glob("*.json"))) == 1

    paths["incoming_dir"].joinpath("idea-copy.json").write_text(
        json.dumps(
            proposal(
                created_at="2026-07-12T12:00:00+00:00",
                provenance={"agent": "other"},
                source_proposal_id="copy",
            )
        ),
        encoding="utf-8",
    )
    duplicate = ingest_inbox(**paths)

    assert duplicate["accepted"] == []
    assert duplicate["rejected"][0]["reason"] == "duplicate_proposal"
    assert len(list(paths["accepted_dir"].glob("*.json"))) == 1
    index = json.loads(paths["index_path"].read_text())
    assert index["accepted"] == 1


def test_dedup_index_survives_factory_consuming_the_accepted_spool(tmp_path):
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    paths["incoming_dir"].joinpath("idea.json").write_text(
        json.dumps(proposal()),
        encoding="utf-8",
    )
    first = ingest_inbox(**paths)
    assert len(first["accepted"]) == 1

    # The research factory durably records a disposition and then removes this
    # transient spool file. The bridge's bounded index remains the duplicate
    # authority across that hand-off.
    next(paths["accepted_dir"].glob("*.json")).unlink()
    paths["incoming_dir"].joinpath("same-idea-again.json").write_text(
        json.dumps(proposal(created_at="2026-07-11T12:00:00+00:00")),
        encoding="utf-8",
    )

    repeated = ingest_inbox(**paths)

    assert repeated["accepted"] == []
    assert repeated["rejected"][0]["reason"] == "duplicate_proposal"
    assert list(paths["accepted_dir"].glob("*.json")) == []


def test_ingest_rejects_malformed_and_duplicate_key_json_then_archives_raw(tmp_path):
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    paths["incoming_dir"].joinpath("bad.json").write_text(
        '{"schema":"research_proposal/v1","schema":"other"}',
        encoding="utf-8",
    )

    report = ingest_inbox(**paths)

    assert report["accepted"] == []
    assert "DuplicateJsonKeyError" in report["rejected"][0]["reason"]
    rejection = json.loads(next(paths["rejected_dir"].glob("*.json")).read_text())
    assert "thesis" not in rejection
    assert len(list(paths["archive_dir"].glob("*.json"))) == 1


def test_ingest_does_not_follow_symlinked_proposal(tmp_path):
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"api_key": "do-not-read"}), encoding="utf-8")
    link = paths["incoming_dir"] / "proposal.json"
    link.symlink_to(secret)

    report = ingest_inbox(**paths)

    assert report["accepted"] == []
    assert report["rejected"][0]["reason"] == "proposal path must be a non-symlink regular file"
    assert secret.read_text() == json.dumps({"api_key": "do-not-read"})
    assert not link.exists()
    assert "do-not-read" not in json.dumps(report)


def test_private_archive_and_rejection_retention_prunes_oldest_records(monkeypatch, tmp_path):
    monkeypatch.setattr(openclaw_bridge_module, "MAX_ARCHIVE_FILES", 2)
    monkeypatch.setattr(openclaw_bridge_module, "MAX_REJECTED_FILES", 2)
    monkeypatch.setattr(openclaw_bridge_module, "MAX_ARCHIVE_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(openclaw_bridge_module, "MAX_REJECTED_BYTES", 10 * 1024 * 1024)
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    first_archive = None
    first_rejection = None

    for index in range(3):
        paths["incoming_dir"].joinpath(f"bad-{index}.json").write_text(
            f'{{"broken": {index}',
            encoding="utf-8",
        )
        report = ingest_inbox(**paths)
        if index == 0:
            first_archive = next(paths["archive_dir"].iterdir())
            first_rejection = next(paths["rejected_dir"].iterdir())

    assert first_archive is not None and not first_archive.exists()
    assert first_rejection is not None and not first_rejection.exists()
    assert len(list(paths["archive_dir"].iterdir())) == 2
    assert len(list(paths["rejected_dir"].iterdir())) == 2
    assert report["retention"]["archive"]["pruned_files"] == 1
    assert report["retention"]["rejected"]["pruned_files"] == 1
    assert report["retention"]["archive"]["limits_satisfied"] is True
    assert report["retention"]["rejected"]["limits_satisfied"] is True
    for directory_name in ("archive_dir", "rejected_dir"):
        directory = paths[directory_name]
        assert directory.stat().st_mode & 0o077 == 0
        assert all(path.stat().st_mode & 0o077 == 0 for path in directory.iterdir())


def test_archival_copies_group_readable_input_into_owner_private_file(monkeypatch, tmp_path):
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    source = paths["incoming_dir"] / "foreign-style.json"
    source.write_text(json.dumps(proposal()), encoding="utf-8")
    source.chmod(0o660)
    real_chmod = Path.chmod

    def deny_archive_file_chmod(path, mode, *args, **kwargs):
        if path.parent == paths["archive_dir"] and path.suffix == ".json":
            raise PermissionError("a foreign-owned moved file could not be chmodded")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", deny_archive_file_chmod)

    report = ingest_inbox(**paths)

    archived = next(paths["archive_dir"].iterdir())
    assert report["ok"] is True
    assert report["archived"] == 1
    assert not source.exists()
    assert archived.stat().st_mode & 0o777 == 0o600


def test_private_retention_fails_closed_on_symlink_without_touching_target(tmp_path):
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    paths["archive_dir"].mkdir(parents=True)
    secret = tmp_path / "outside.json"
    secret.write_text('{"secret":"keep"}', encoding="utf-8")
    paths["archive_dir"].joinpath("unsafe.json").symlink_to(secret)

    with pytest.raises(ProposalValidationError, match="private retention path is unsafe"):
        ingest_inbox(**paths)

    assert secret.read_text(encoding="utf-8") == '{"secret":"keep"}'


def test_private_retention_enforces_byte_ceilings(monkeypatch, tmp_path):
    monkeypatch.setattr(openclaw_bridge_module, "MAX_ARCHIVE_BYTES", 1)
    monkeypatch.setattr(openclaw_bridge_module, "MAX_REJECTED_BYTES", 1)
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    paths["incoming_dir"].joinpath("bad.json").write_text('{"broken":', encoding="utf-8")

    report = ingest_inbox(**paths)

    assert list(paths["archive_dir"].iterdir()) == []
    assert list(paths["rejected_dir"].iterdir()) == []
    assert report["retention"]["archive"]["pruned_bytes"] > 1
    assert report["retention"]["rejected"]["pruned_bytes"] > 1
    assert report["retention"]["archive"]["limits_satisfied"] is True
    assert report["retention"]["rejected"]["limits_satisfied"] is True


def test_oversized_input_is_bounded_rejected_and_removed_without_archival(monkeypatch, tmp_path):
    limit = 32
    monkeypatch.setattr(openclaw_bridge_module, "MAX_PROPOSAL_BYTES", limit)
    bytes_read = 0
    real_read = openclaw_bridge_module.os.read

    def tracked_read(descriptor, amount):
        nonlocal bytes_read
        chunk = real_read(descriptor, amount)
        bytes_read += len(chunk)
        return chunk

    monkeypatch.setattr(openclaw_bridge_module.os, "read", tracked_read)
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    source = paths["incoming_dir"] / "oversized.json"
    source.write_bytes(b"x" * (1024 * 1024))

    report = ingest_inbox(**paths)

    assert report["ok"] is True
    assert report["archived"] == 0
    assert report["oversized_discarded"] == 1
    assert "proposal exceeds 32 bytes" in report["rejected"][0]["reason"]
    assert not source.exists()
    assert list(paths["archive_dir"].iterdir()) == []
    assert bytes_read <= 3 * (limit + 1)


def test_incoming_hygiene_bounds_backlog_and_removes_stale_temporary_files(monkeypatch, tmp_path):
    monkeypatch.setattr(openclaw_bridge_module, "MAX_INCOMING_FILES", 2)
    monkeypatch.setattr(openclaw_bridge_module, "MAX_INCOMING_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(openclaw_bridge_module, "STALE_INCOMING_TEMP_SECONDS", 0)
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    paths["incoming_dir"].joinpath("abandoned.tmp").write_text("partial", encoding="utf-8")
    for index in range(5):
        paths["incoming_dir"].joinpath(f"idea-{index}.json").write_text(
            json.dumps(
                proposal(
                    thesis=(
                        f"A distinct bounded incoming backlog thesis with sequence number {index}."
                    )
                )
            ),
            encoding="utf-8",
        )
    report = ingest_inbox(max_batch=1, **paths)

    incoming = report["retention"]["incoming"]
    assert len(report["accepted"]) == 1
    assert incoming["limits_satisfied"] is True
    assert incoming["pruned_files"] == 3
    assert incoming["stale_temp_files_pruned"] == 1
    assert report["remaining"] == 2
    assert len(list(paths["incoming_dir"].iterdir())) == 2


def test_dedup_capacity_is_explicitly_degraded_but_not_a_native_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(openclaw_bridge_module, "MAX_DEDUP_INDEX_ITEMS", 1)
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    paths["incoming_dir"].joinpath("first.json").write_text(
        json.dumps(proposal()),
        encoding="utf-8",
    )
    first = ingest_inbox(**paths)
    assert len(first["accepted"]) == 1
    paths["incoming_dir"].joinpath("second.json").write_text(
        json.dumps(
            proposal(
                thesis="A second semantically distinct proposal reaches a deliberately tiny index."
            )
        ),
        encoding="utf-8",
    )

    report = ingest_inbox(**paths)

    assert report["ok"] is True
    assert report["degraded"] is True
    assert report["degraded_reasons"] == ["dedup_index_capacity"]
    assert report["native_generation_unaffected"] is True
    assert report["rejected"][0]["reason"] == "dedup_index_capacity"
    assert report["dedup_index"] == {
        "items": 1,
        "item_limit": 1,
        "at_capacity": True,
        "capacity_rejections": 1,
    }


def test_accepted_spool_backpressure_prevents_unprocessed_disk_growth(monkeypatch, tmp_path):
    monkeypatch.setattr(openclaw_bridge_module, "MAX_ACCEPTED_FILES", 1)
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    paths["incoming_dir"].joinpath("first.json").write_text(
        json.dumps(proposal()), encoding="utf-8"
    )
    first = ingest_inbox(**paths)
    assert len(first["accepted"]) == 1

    paths["incoming_dir"].joinpath("second.json").write_text(
        json.dumps(
            proposal(
                thesis="A distinct second proposal must not grow a full accepted hand-off spool."
            )
        ),
        encoding="utf-8",
    )
    report = ingest_inbox(**paths)

    assert report["accepted"] == []
    assert report["rejected"][0]["reason"] == "accepted_spool_capacity"
    assert report["degraded_reasons"] == ["accepted_spool_capacity"]
    assert len(list(paths["accepted_dir"].glob("*.json"))) == 1
    assert report["retention"]["accepted"] == {
        "file_limit": 1,
        "byte_limit": openclaw_bridge_module.MAX_ACCEPTED_BYTES,
        "scan_limit": openclaw_bridge_module.MAX_ACCEPTED_SCAN,
        "scan_truncated": False,
        "files": 1,
        "bytes": next(paths["accepted_dir"].glob("*.json")).stat().st_size,
        "limits_satisfied": True,
        "capacity_rejections": 1,
    }


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_research_context_excludes_secrets_live_state_approvals_and_final_holdout_feedback(
    tmp_path,
):
    research = tmp_path / "research.json"
    batch = tmp_path / "batch.json"
    write_json(
        research,
        {
            "generated_at": "research-time",
            "approval_ledger": {"active_income": "approved"},
            "exchange_api_key": "never-export",
            "summary": {
                "scenarios": 4,
                "hypotheses": 30,
                "selected_hypotheses": 5,
                "keepers": 2,
                "exported": 1,
                "staged": 1,
                "incubation_candidates": 9,
                "top_reasons": {
                    "no_train_edge": 12,
                    "failed_validation": 4,
                    "failed_holdout": 3,
                    "insufficient_holdout_trades": 2,
                },
                "holdout_metrics": {"sharpe": 9.9},
                "final_test_return": 100.0,
                "unique_specs": 21,
            },
        },
    )
    write_json(
        batch,
        {
            "generated_at": "batch-time",
            "summary": {
                "hypotheses": 3,
                "new_hypotheses": 2,
                "by_product": {"active_income": 3},
                "by_space": {"active_income_day": 3},
                "by_method": {"grammar_sample": 2, "crossover": 1},
                "unique_behavioral_specs": 21,
                "cumulative_trials": 40,
            },
            "memory": {
                "feedback": {
                    "totals": {
                        "strategies": 21,
                        "evaluations": 40,
                        "holdout_exposed": 2,
                    },
                    "outcomes": {"reject": 20, "pre_holdout_pass": 4, "failed_holdout": 3},
                    "rejection_reasons": {"no_train_edge": 12, "failed_holdout": 3},
                    "generation_methods": {
                        "grammar_sample": {
                            "experiments": 12,
                            "outcomes": {"reject": 10, "failed_holdout": 2},
                        }
                    },
                }
            },
        },
    )

    context = build_research_context(
        research_cycle_path=research,
        generated_batch_path=batch,
    )
    rendered = json.dumps(context, sort_keys=True)

    assert context["research_progress"]["hypotheses"] == 30
    assert context["research_progress"]["development_failure_reasons"] == {
        "no_train_edge": 12,
        "failed_validation": 4,
    }
    assert context["research_progress"]["novelty"] == {"unique_specs": 21}
    assert context["experiment_memory"]["totals"] == {
        "strategies": 21,
        "evaluations": 40,
    }
    assert context["experiment_memory"]["development_failure_reasons"] == {"no_train_edge": 12}
    assert context["generated_batch"]["unique_behavioral_specs"] == 21
    assert context["boundary"]["final_holdout_feedback_excluded"] is True
    for forbidden in (
        "never-export",
        "approval_ledger",
        "keepers",
        "exported",
        "staged",
        "incubation_candidates",
        "failed_holdout",
        "holdout_metrics",
        "final_test_return",
        "retired_candidates",
        "holdout_exposed",
    ):
        assert forbidden not in rendered


def test_export_context_is_private_and_rejects_symlink_output(tmp_path):
    output = tmp_path / "private" / "context.json"
    context = export_research_context(
        output,
        research_cycle_path=tmp_path / "missing-research.json",
        generated_batch_path=tmp_path / "missing-batch.json",
    )

    assert json.loads(output.read_text()) == context
    assert output.stat().st_mode & 0o777 == 0o600

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "context-link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        export_research_context(link)


def test_shared_group_mode_exposes_only_context_and_incoming_spool(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCLAW_SHARED_GROUP", "1")
    emulate_linux_directory_modes(monkeypatch)
    output = tmp_path / "context" / "research_context.json"
    export_research_context(
        output,
        research_cycle_path=tmp_path / "missing-research.json",
        generated_batch_path=tmp_path / "missing-batch.json",
    )
    paths = inbox_paths(tmp_path)
    paths["incoming_dir"].mkdir(parents=True)
    paths["incoming_dir"].joinpath("idea.json").write_text(json.dumps(proposal()), encoding="utf-8")
    ingest_inbox(**paths)

    assert output.parent.stat().st_mode & 0o777 == 0o750
    assert output.stat().st_mode & 0o777 == 0o640
    assert paths["incoming_dir"].parent.stat().st_mode & 0o777 == 0o710
    assert paths["incoming_dir"].stat().st_mode & 0o777 == 0o770
    assert paths["accepted_dir"].stat().st_mode & 0o777 == 0o700


def test_shared_group_export_accepts_correct_non_chmodable_mount_root(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCLAW_SHARED_GROUP", "1")
    output = tmp_path / "context" / "research_context.json"
    output.parent.mkdir()
    output.parent.chmod(0o2750)
    mount_identity = (output.parent.stat().st_dev, output.parent.stat().st_ino)
    emulate_linux_directory_modes(
        monkeypatch,
        preset={mount_identity: 0o2750},
        immutable={mount_identity},
    )

    context = export_research_context(
        output,
        research_cycle_path=tmp_path / "missing-research.json",
        generated_batch_path=tmp_path / "missing-batch.json",
    )

    assert json.loads(output.read_text()) == context
    assert output.stat().st_mode & 0o777 == 0o640


def test_shared_group_ingest_accepts_correct_non_chmodable_mount_root(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCLAW_SHARED_GROUP", "1")
    paths = inbox_paths(tmp_path)
    root = paths["incoming_dir"].parent
    root.mkdir(parents=True)
    root.chmod(0o2710)
    mount_identity = (root.stat().st_dev, root.stat().st_ino)
    emulate_linux_directory_modes(
        monkeypatch,
        preset={mount_identity: 0o2710},
        immutable={mount_identity},
    )

    report = ingest_inbox(**paths)

    assert report["ok"] is True
    assert root.stat().st_mode & 0o777 == 0o710


def test_mode_enforcement_rejects_exact_mode_foreign_owned_directory(monkeypatch, tmp_path):
    directory = tmp_path / "foreign"
    directory.mkdir(mode=0o700)
    identity = (directory.stat().st_dev, directory.stat().st_ino)
    real_fstat = os.fstat

    def foreign_owner(descriptor):
        result = real_fstat(descriptor)
        if (result.st_dev, result.st_ino) == identity:
            values = list(result)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(os, "fstat", foreign_owner)

    with pytest.raises(ProposalValidationError, match="owned by the current process user"):
        openclaw_bridge_module._chmod_if_needed(directory, 0o700)


def test_mode_enforcement_fails_when_wrong_mode_mount_is_immutable(monkeypatch, tmp_path):
    directory = tmp_path / "immutable"
    directory.mkdir(mode=0o755)
    identity = (directory.stat().st_dev, directory.stat().st_ino)
    real_fstat = os.fstat
    real_fchmod = os.fchmod

    def deny_mount_root_chmod(descriptor, mode):
        result = real_fstat(descriptor)
        if (result.st_dev, result.st_ino) == identity:
            raise PermissionError("systemd bind-mount root metadata is immutable")
        return real_fchmod(descriptor, mode)

    monkeypatch.setattr(os, "fchmod", deny_mount_root_chmod)

    with pytest.raises(PermissionError, match="bind-mount root metadata is immutable"):
        openclaw_bridge_module._chmod_if_needed(directory, 0o700)
