"""Fail-closed Linux bubblewrap runner for agent-generated code."""

from __future__ import annotations

import platform
import resource
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_seconds: int = 7_200
    maximum_memory_mb: int = 3_072
    maximum_file_bytes: int = 64 * 1024 * 1024
    maximum_processes: int = 32
    maximum_cpu_seconds: int = 7_200
    maximum_output_bytes: int = 64_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.maximum_cpu_seconds <= 0:
            raise ValueError("sandbox timeout must be positive")
        if self.maximum_memory_mb <= 0 or self.maximum_file_bytes <= 0:
            raise ValueError("sandbox memory and file limits must be positive")
        if self.maximum_processes <= 0:
            raise ValueError("sandbox process limit must be positive")


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.return_code == 0


class SandboxRunner:
    ALLOWED_PYTHON_MODULES = frozenset({"mypy", "pytest", "ruff"})

    def __init__(
        self,
        *,
        workspace: Path,
        policy: SandboxPolicy = SandboxPolicy(),
        require_network_isolation: bool = True,
    ) -> None:
        self.workspace = workspace.resolve()
        self.policy = policy
        self.require_network_isolation = require_network_isolation
        self.bwrap = shutil.which("bwrap")
        if self.require_network_isolation and self.bwrap is None:
            raise RuntimeError("bubblewrap is required for the Linux agent sandbox")

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        if not argv:
            raise ValueError("sandbox command cannot be empty")
        executable = Path(argv[0]).name
        if executable not in {"python", "python3"}:
            raise PermissionError(f"sandbox executable is not allowed: {executable}")
        if len(argv) < 3 or argv[1] != "-m" or argv[2] not in self.ALLOWED_PYTHON_MODULES:
            raise PermissionError("sandbox Python commands must use an approved module")
        command = list(argv)
        if self.require_network_isolation:
            if platform.system() != "Linux" or self.bwrap is None:
                raise RuntimeError("the agent worker requires Linux bubblewrap")
            command = self._bubblewrap_command(command)
        environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
        }
        if not self.require_network_isolation:
            command = [*command]
        completed = subprocess.run(
            command,
            cwd=self.workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=self.policy.timeout_seconds,
            check=False,
            preexec_fn=self._limit_resources,
        )
        return CommandResult(
            argv=argv,
            return_code=completed.returncode,
            stdout=completed.stdout[-self.policy.maximum_output_bytes :],
            stderr=completed.stderr[-self.policy.maximum_output_bytes :],
        )

    def _bubblewrap_command(self, command: list[str]) -> list[str]:
        """Build a minimal read-only root with a writable private worktree."""

        worktree = self.workspace / ".agent-sandbox-worktree"
        worktree.mkdir(parents=True, exist_ok=True)
        return [
            str(self.bwrap),
            "--die-with-parent",
            "--unshare-net",
            "--unshare-pid",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            str(self.workspace),
            "/repository",
            "--bind",
            str(worktree),
            "/worktree",
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            "/worktree",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "--setenv",
            "PYTHONHASHSEED",
            "0",
            "--",
            *command,
        ]

    def _limit_resources(self) -> None:
        memory = self.policy.maximum_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (self.policy.maximum_file_bytes, self.policy.maximum_file_bytes),
        )
        if hasattr(resource, "RLIMIT_CPU"):
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.policy.maximum_cpu_seconds, self.policy.maximum_cpu_seconds),
            )
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (self.policy.maximum_processes, self.policy.maximum_processes),
            )


class IsolatedGitWorktree:
    def __init__(self, *, repository: Path, worktree_root: Path, base_ref: str):
        self.repository = repository.resolve()
        self.worktree_root = worktree_root.resolve()
        self.base_ref = base_ref
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="agent-worktree-", dir=self.worktree_root))
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(self.path), self.base_ref],
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return self.path

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.path is None:
            return
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self.path)],
            cwd=self.repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if self.path.exists():
            shutil.rmtree(self.path)
