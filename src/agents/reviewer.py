"""Deterministic review gates for agent-generated source."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from src.agents.proposals import FORBIDDEN_CODE_MARKERS, AgentAction, AgentProposal
from src.agents.sandbox import CommandResult, SandboxRunner


@dataclass(frozen=True)
class ReviewCheck:
    name: str
    passed: bool
    reason_code: str
    detail: str = ""


@dataclass(frozen=True)
class ReviewOutcome:
    proposal_id: str
    checks: tuple[ReviewCheck, ...]

    @property
    def accepted(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def first_failure(self) -> str | None:
        return next((check.reason_code for check in self.checks if not check.passed), None)


class AgentCodeReviewer:
    TEST_REQUIREMENTS = {
        AgentAction.CREATE_DSL: (
            "test_deterministic",
            "test_no_lookahead",
            "test_signal_domain",
            "test_synthetic_signal",
        ),
        AgentAction.CREATE_PYTHON_STRATEGY: (
            "test_deterministic",
            "test_no_lookahead",
            "test_signal_domain",
            "test_synthetic_signal",
            "test_cost_adjusted_backtest",
        ),
        AgentAction.CREATE_FEATURE: (
            "test_deterministic",
            "test_no_lookahead",
            "test_synthetic_signal",
        ),
        AgentAction.CREATE_DATA_ADAPTER: ("test_data_access",),
    }

    def review(
        self,
        *,
        proposal: AgentProposal,
        workspace: Path,
        runner: SandboxRunner,
    ) -> ReviewOutcome:
        paths = tuple(sorted(proposal.files))
        checks = list(self._static_checks(proposal))
        if not all(check.passed for check in checks):
            return ReviewOutcome(proposal.proposal_id, tuple(checks))
        python_paths = tuple(path for path in paths if path.endswith(".py"))
        commands = [
            (
                "unit_property_determinism_lookahead",
                ("python", "-m", "pytest", "-q", *self._test_paths(proposal)),
            )
        ]
        if python_paths:
            commands[0:0] = [
                (
                    "formatting",
                    ("python", "-m", "ruff", "format", "--check", *python_paths),
                ),
                ("lint", ("python", "-m", "ruff", "check", *python_paths)),
                (
                    "static_types",
                    (
                        "python",
                        "-m",
                        "mypy",
                        "--follow-imports=skip",
                        "--ignore-missing-imports",
                        *python_paths,
                    ),
                ),
            ]
        for name, argv in commands:
            result = runner.run(argv)
            checks.append(self._command_check(name, result))
            if not result.passed:
                break
        return ReviewOutcome(proposal.proposal_id, tuple(checks))

    def _static_checks(self, proposal: AgentProposal) -> tuple[ReviewCheck, ...]:
        checks: list[ReviewCheck] = []
        try:
            for path, content in proposal.files.items():
                if path.endswith(".py"):
                    tree = ast.parse(content, filename=path)
                    self._assert_safe_ast(tree, path=path)
            checks.append(ReviewCheck("data_access", True, "data_access_safe"))
        except (SyntaxError, ValueError) as exc:
            checks.append(ReviewCheck("data_access", False, "unsafe_or_invalid_source", str(exc)))
            return tuple(checks)
        requirements = self.TEST_REQUIREMENTS.get(proposal.action, ())
        test_content = "\n".join(
            content for path, content in proposal.files.items() if path.startswith("tests/")
        )
        missing = [name for name in requirements if name not in test_content]
        checks.append(
            ReviewCheck(
                "required_tests",
                not missing,
                "required_agent_tests_present" if not missing else "required_agent_tests_missing",
                ",".join(missing),
            )
        )
        checks.append(ReviewCheck("resource_limits", True, "resource_limits_passed"))
        return tuple(checks)

    @staticmethod
    def _assert_safe_ast(tree: ast.AST, *, path: str) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [alias.name for alias in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                if any(
                    name == "src.execution" or name.startswith("src.execution.") for name in names
                ):
                    raise ValueError(f"execution import is forbidden: {path}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"place_order", "submit", "create_order"}:
                    raise ValueError(f"order submission call is forbidden: {path}")
        source = ast.unparse(tree)
        if any(marker in source for marker in FORBIDDEN_CODE_MARKERS):
            raise ValueError(f"execution marker is forbidden: {path}")

    @staticmethod
    def _test_paths(proposal: AgentProposal) -> tuple[str, ...]:
        paths = tuple(sorted(path for path in proposal.files if path.startswith("tests/")))
        return paths or ("tests/test_platform_contracts.py",)

    @staticmethod
    def _command_check(name: str, result: CommandResult) -> ReviewCheck:
        detail = result.stderr or result.stdout
        return ReviewCheck(
            name=name,
            passed=result.passed,
            reason_code=f"{name}_passed" if result.passed else f"{name}_failed",
            detail=detail,
        )
