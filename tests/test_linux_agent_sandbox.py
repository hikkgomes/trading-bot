from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.code_worker import AgentCodeWorkflow
from src.agents.proposals import AgentAction, AgentProposal, AgentRole
from src.agents.sandbox import IsolatedGitWorktree, SandboxPolicy, SandboxRunner


@pytest.mark.skipif(
    platform.system() != "Linux" or shutil.which("bwrap") is None,
    reason="requires Linux bubblewrap",
)
def test_code_sandbox_cannot_read_private_state_history_or_host_network(tmp_path, monkeypatch):
    workspace = tmp_path / "worktree"
    for directory in ("src", "tests", "data", "runtime", "reviewer-evidence", ".git"):
        (workspace / directory).mkdir(parents=True)
        (workspace / directory / "marker.txt").write_text("fixture-private-data")
    (workspace / ".env").write_text("CANARY_SECRET=fixture-only")
    outside = tmp_path / "outside.txt"
    outside.write_text("fixture-outside-data")
    (workspace / "tests/escape").symlink_to(outside)
    monkeypatch.setenv("CANARY_SECRET", "fixture-only")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        (workspace / "tests/test_isolation.py").write_text(
            "from pathlib import Path\nimport os, socket\n"
            "def test_isolation():\n"
            "    assert Path('/worktree/src/marker.txt').read_text() == 'fixture-private-data'\n"
            "    for name in ('data/marker.txt', 'runtime/marker.txt', 'reviewer-evidence/marker.txt', '.git/marker.txt', '.env', 'tests/escape'):\n"
            "        assert not Path('/worktree', name).exists()\n"
            "    assert not Path('/repository').exists()\n"
            "    assert 'CANARY_SECRET' not in os.environ\n"
            "    with socket.socket() as client:\n"
            "        client.settimeout(1)\n"
            f"        assert client.connect_ex(('127.0.0.1', {port})) != 0\n"
        )
        result = SandboxRunner(
            workspace=workspace,
            agent_venv=Path(sys.prefix),
            policy=SandboxPolicy(
                timeout_seconds=30,
                maximum_cpu_seconds=20,
                maximum_memory_mb=1024,
                maximum_processes=256,
            ),
        ).run(("python", "-m", "pytest", "-q", "tests/test_isolation.py"))
    assert result.passed, result.stderr or result.stdout


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
    (repository / "src").mkdir()
    (repository / "src/current.py").write_text("value = 1\n")
    (repository / "src/retired.py").write_text("value = 0\n")
    subprocess.run(("git", "init", "--quiet"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"), cwd=repository, check=True
    )
    subprocess.run(("git", "config", "user.name", "Test"), cwd=repository, check=True)
    subprocess.run(("git", "config", "commit.gpgsign", "false"), cwd=repository, check=True)
    subprocess.run(("git", "add", "README.txt", "src"), cwd=repository, check=True)
    subprocess.run(("git", "commit", "--quiet", "-m", "baseline"), cwd=repository, check=True)
    (repository / "src/current.py").write_text("value = 2\n")
    (repository / "src/retired.py").unlink()
    (repository / "src/new.py").write_text("value = 3\n")
    (repository / ".env").write_text("PRIVATE_FIXTURE=hidden\n")
    before = subprocess.check_output(("git", "status", "--porcelain"), cwd=repository, text=True)

    worktree_root = tmp_path / "agent-worktrees"
    with IsolatedGitWorktree(
        repository=repository, worktree_root=worktree_root, base_ref="HEAD"
    ) as worktree:
        assert (worktree / "README.txt").read_text(encoding="utf-8") == "baseline\n"
        assert (worktree / "src/current.py").read_text() == "value = 2\n"
        assert (worktree / "src/new.py").read_text() == "value = 3\n"
        assert not (worktree / "src/retired.py").exists()
        assert not (worktree / ".env").exists()
        (worktree / "agent-change.txt").write_text("change\n", encoding="utf-8")

    after = subprocess.check_output(("git", "status", "--porcelain"), cwd=repository, text=True)
    assert before == after
    assert not tuple(worktree_root.iterdir())
    proposal = AgentProposal(
        "fixture",
        AgentRole.IMPLEMENTER,
        AgentAction.CREATE_TESTS,
        "active_income",
        "2026-09-09T00:00:00+00:00",
        "Review fixture",
    )
    workflow = AgentCodeWorkflow(
        repository=repository,
        worktree_root=worktree_root,
        store=SimpleNamespace(save_proposal=lambda _: pytest.fail("publication started")),
    )
    with pytest.raises(ValueError, match="committed review base"):
        workflow.execute(proposal, base_ref="HEAD", publish=True)
