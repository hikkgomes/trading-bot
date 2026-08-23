"""Isolated worktree workflow from agent proposal to research branch."""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.agents.proposals import AgentProposal
from src.agents.reviewer import AgentCodeReviewer, ReviewOutcome
from src.agents.sandbox import IsolatedGitWorktree, SandboxPolicy, SandboxRunner
from src.agents.store import SqlAgentStore


@dataclass(frozen=True)
class CodeWorkflowResult:
    proposal_id: str
    accepted: bool
    reason_code: str
    commit_hash: str | None
    review: ReviewOutcome


class ResearchBranchPublisher:
    """Commit accepted code and fast-forward only the research branch."""

    def __init__(self, *, research_branch: str = "research") -> None:
        if not research_branch or research_branch in {"main", "master"}:
            raise ValueError("agent output must use a dedicated research branch")
        self.research_branch = research_branch

    def publish(self, *, workspace: Path, proposal: AgentProposal) -> str:
        paths = sorted(proposal.files)
        self._git(workspace, "add", "--", *paths)
        self._git(workspace, "commit", "-m", f"research: implement {proposal.proposal_id}")
        commit_hash = self._git(workspace, "rev-parse", "HEAD").strip()
        current = subprocess.run(
            ["git", "rev-parse", f"refs/heads/{self.research_branch}"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        old_hash = current.stdout.strip() if current.returncode == 0 else ""
        if old_hash:
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", old_hash, commit_hash],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            if ancestor.returncode != 0:
                raise RuntimeError("research branch cannot be fast-forwarded")
        update = ["update-ref", f"refs/heads/{self.research_branch}", commit_hash]
        if old_hash:
            update.append(old_hash)
        self._git(workspace, *update)
        return commit_hash

    @staticmethod
    def _git(workspace: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout


class AgentCodeWorkflow:
    def __init__(
        self,
        *,
        repository: Path,
        worktree_root: Path,
        store: SqlAgentStore,
        reviewer: AgentCodeReviewer | None = None,
        publisher: ResearchBranchPublisher | None = None,
        sandbox_policy: SandboxPolicy = SandboxPolicy(),
    ) -> None:
        self.repository = repository
        self.worktree_root = worktree_root
        self.store = store
        self.reviewer = reviewer or AgentCodeReviewer()
        self.publisher = publisher or ResearchBranchPublisher()
        self.sandbox_policy = sandbox_policy

    def execute(
        self,
        proposal: AgentProposal,
        *,
        base_ref: str,
        publish: bool = True,
    ) -> CodeWorkflowResult:
        self.store.save_proposal(proposal)
        with IsolatedGitWorktree(
            repository=self.repository,
            worktree_root=self.worktree_root,
            base_ref=base_ref,
        ) as workspace:
            self._write_files(workspace, proposal)
            self.store.save_patch(
                proposal_id=proposal.proposal_id,
                created_at=proposal.created_at,
                payload={
                    "files": {
                        path: "sha256:" + hashlib.sha256(content.encode()).hexdigest()
                        for path, content in proposal.files.items()
                    }
                },
            )
            runner = SandboxRunner(
                workspace=workspace,
                repository=self.repository,
                agent_venv=self.repository / ".venv-agent",
                policy=self.sandbox_policy,
            )
            review = self.reviewer.review(
                proposal=proposal,
                workspace=workspace,
                runner=runner,
            )
            reviewed_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
            self.store.save_review(
                proposal_id=proposal.proposal_id,
                created_at=reviewed_at,
                payload={
                    "accepted": review.accepted,
                    "checks": [check.__dict__ for check in review.checks],
                },
            )
            commit_hash = None
            if review.accepted and publish:
                commit_hash = self.publisher.publish(workspace=workspace, proposal=proposal)
            reason_code = review.first_failure or (
                "research_branch_updated" if commit_hash else "review_passed"
            )
            self.store.save_disposition(
                proposal_id=proposal.proposal_id,
                created_at=reviewed_at,
                payload={
                    "accepted": review.accepted,
                    "reason_code": reason_code,
                    "commit_hash": commit_hash,
                    "live_eligible": False,
                },
            )
            return CodeWorkflowResult(
                proposal_id=proposal.proposal_id,
                accepted=review.accepted,
                reason_code=reason_code,
                commit_hash=commit_hash,
                review=review,
            )

    @staticmethod
    def _write_files(workspace: Path, proposal: AgentProposal) -> None:
        root = workspace.resolve()
        for relative_path, content in proposal.files.items():
            destination = (root / relative_path).resolve()
            if root not in destination.parents:
                raise ValueError("agent file escaped the isolated worktree")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
