"""Fail-closed Linux bubblewrap runner for agent-generated code."""

from __future__ import annotations

import platform
import resource
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

REVIEW_INPUTS = (
    "src",
    "tests",
    "config",
    "scripts",
    "alembic",
    "pyproject.toml",
    "pytest.ini",
    "conftest.py",
    "alembic.ini",
    "Makefile",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-runtime.txt",
    "requirements-research-linux.txt",
    "requirements-agent.txt",
)


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
        agent_venv: Path | None = None,
        policy: SandboxPolicy = SandboxPolicy(),
        require_network_isolation: bool = True,
    ) -> None:
        self.workspace = workspace.resolve()
        self.agent_venv = (agent_venv or self.workspace / ".venv-agent").resolve()
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
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
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
        """Expose review inputs and dependencies, without operational data or Git history."""

        interpreter = self.agent_venv / "bin" / "python"
        if not interpreter.is_file():
            raise RuntimeError(f"agent Python interpreter is missing: {interpreter}")
        command[0] = "/venv-agent/bin/python"
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
            *(["--ro-bind", "/lib64", "/lib64"] if Path("/lib64").is_dir() else []),
            "--tmpfs",
            "/worktree",
            *self._review_input_mounts(),
            "--ro-bind",
            str(self.agent_venv),
            "/venv-agent",
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
            "/venv-agent/bin:/usr/bin:/bin",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "--setenv",
            "PYTHONHASHSEED",
            "0",
            "--setenv",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "1",
            "--setenv",
            "OPENBLAS_NUM_THREADS",
            "1",
            "--setenv",
            "OMP_NUM_THREADS",
            "1",
            "--",
            *command,
        ]

    def _review_input_mounts(self) -> list[str]:
        mounts: list[str] = []
        for name in REVIEW_INPUTS:
            source = self.workspace / name
            if not source.exists():
                continue
            if source.is_symlink() or not source.resolve().is_relative_to(self.workspace):
                raise PermissionError("sandbox review inputs must remain inside the worktree")
            mounts.extend(
                (
                    "--bind" if name in {"src", "tests"} else "--ro-bind",
                    str(source),
                    f"/worktree/{name}",
                )
            )
        return mounts

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
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{self.base_ref}^{{commit}}"],
            cwd=self.repository,
            text=True,
        ).strip()
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="agent-worktree-", dir=self.worktree_root))
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-local",
                    "--no-hardlinks",
                    str(self.repository),
                    str(self.path),
                ],
                cwd=self.worktree_root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "checkout", "--detach", revision],
                cwd=self.path,
                check=True,
                capture_output=True,
                text=True,
            )
            self._current_source(revision)
            return self.path
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _current_source(self, revision: str) -> None:
        if self.path is None:
            raise RuntimeError("review worktree has not been created")
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repository, text=True
        ).strip()
        if head != revision:
            return
        patch = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", *REVIEW_INPUTS],
            cwd=self.repository,
        )
        if patch:
            subprocess.run(["git", "apply", "--binary"], cwd=self.path, input=patch, check=True)
        untracked = (
            subprocess.check_output(
                ["git", "ls-files", "-z", "--others", "--exclude-standard", "--", *REVIEW_INPUTS],
                cwd=self.repository,
            )
            .decode()
            .split("\0")
        )
        for name in filter(None, untracked):
            source = self.repository / name
            if (
                source.is_symlink()
                or not source.resolve().is_relative_to(self.repository)
                or source.suffix == ".env"
            ):
                raise PermissionError(
                    "untracked review input is not an allowed regular source file"
                )
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.path is None:
            return
        if self.path.exists():
            shutil.rmtree(self.path)
