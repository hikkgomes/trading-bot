from __future__ import annotations

import platform
import subprocess

import pytest

from src.agents.sandbox import IsolatedGitWorktree, SandboxRunner


def test_agent_sandbox_is_fail_closed_outside_linux(tmp_path) -> None:
    if platform.system() == "Linux":
        pytest.skip("Linux sandbox behaviour is covered by the deployment smoke")
    with pytest.raises(RuntimeError):
        runner = SandboxRunner(workspace=tmp_path, require_network_isolation=True)
        runner.run(("python3", "-m", "pytest", "--version"))


def test_agent_worktree_does_not_write_source_repository_metadata(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "README.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(("git", "init", "--quiet"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"), cwd=repository, check=True
    )
    subprocess.run(("git", "config", "user.name", "Test"), cwd=repository, check=True)
    subprocess.run(("git", "config", "commit.gpgsign", "false"), cwd=repository, check=True)
    subprocess.run(("git", "add", "README.txt"), cwd=repository, check=True)
    subprocess.run(("git", "commit", "--quiet", "-m", "baseline"), cwd=repository, check=True)
    before = subprocess.check_output(("git", "status", "--porcelain"), cwd=repository, text=True)

    worktree_root = tmp_path / "agent-worktrees"
    with IsolatedGitWorktree(
        repository=repository, worktree_root=worktree_root, base_ref="HEAD"
    ) as worktree:
        assert (worktree / "README.txt").read_text(encoding="utf-8") == "baseline\n"
        (worktree / "agent-change.txt").write_text("change\n", encoding="utf-8")

    after = subprocess.check_output(("git", "status", "--porcelain"), cwd=repository, text=True)
    assert before == after == ""
    assert not tuple(worktree_root.iterdir())
