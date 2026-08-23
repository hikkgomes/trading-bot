from __future__ import annotations

import platform

import pytest

from src.agents.sandbox import SandboxRunner


def test_agent_sandbox_is_fail_closed_outside_linux(tmp_path) -> None:
    if platform.system() == "Linux":
        pytest.skip("Linux sandbox behaviour is covered by the deployment smoke")
    with pytest.raises(RuntimeError):
        runner = SandboxRunner(workspace=tmp_path, require_network_isolation=True)
        runner.run(("python3", "-m", "pytest", "--version"))
