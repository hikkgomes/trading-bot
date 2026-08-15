"""Leased OpenClaw research and isolated code-workflow handlers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.agents.code_worker import AgentCodeWorkflow
from src.agents.store import SqlAgentStore
from src.services.runtime import utc_now
from src.services.scheduler import ClaimedJob, DatabaseJobQueue

ALLOWED_RESEARCH_JOBS = frozenset(
    {
        "register_strategy_catalogue",
        "register_candidate",
        "register_ml_candidate",
        "evaluate_candidate",
        "bounded_backtest",
        "event_replay",
        "train_ml_experiment",
        "live_feature_calculation",
        "historical_feature_calculation",
    }
)


class DatabaseAgentJobHandlers:
    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        store: SqlAgentStore,
        code_workflow: AgentCodeWorkflow,
        maximum_runtime_seconds: int,
    ) -> None:
        self.queue = queue
        self.store = store
        self.code_workflow = code_workflow
        self.maximum_runtime_seconds = maximum_runtime_seconds

    def handlers(self) -> dict[str, Callable]:
        return {
            "agent_research": self.agent_research,
            "agent_code_workflow": self.agent_code_workflow,
        }

    def agent_research(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        proposal = self.store.proposal(str(claimed.payload["proposal_id"]))
        queued: list[str] = []
        for index, item in enumerate(proposal.research_jobs):
            name = str(item.get("name") or "")
            if name not in ALLOWED_RESEARCH_JOBS:
                raise ValueError(f"agent requested unsupported research job: {name}")
            maximum_seconds = int(item.get("maximum_seconds", self.maximum_runtime_seconds))
            if not 1 <= maximum_seconds <= self.maximum_runtime_seconds:
                raise ValueError("agent research job exceeds its runtime budget")
            raw_payload = item.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
            job_id = f"agent:{proposal.proposal_id}:research:{index}"
            self.queue.enqueue_if_absent(
                job_id=job_id,
                name=name,
                payload={
                    **payload,
                    "agent_proposal_id": proposal.proposal_id,
                    "maximum_seconds": maximum_seconds,
                    "agent_may_submit_orders": False,
                },
                available_at=proposal.created_at,
                priority=5,
            )
            queued.append(job_id)
            renew()
        self.store.save_disposition(
            proposal_id=proposal.proposal_id,
            created_at=utc_now(),
            payload={
                "accepted": True,
                "reason_code": "bounded_research_jobs_enqueued",
                "job_ids": queued,
                "live_eligible": False,
            },
        )
        return {"proposal_id": proposal.proposal_id, "research_job_ids": queued}

    def agent_code_workflow(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        proposal = self.store.proposal(str(claimed.payload["proposal_id"]))
        renew()
        result = self.code_workflow.execute(
            proposal,
            base_ref=str(claimed.payload.get("base_ref") or "HEAD"),
            publish=bool(claimed.payload.get("publish", True)),
        )
        return {
            "proposal_id": proposal.proposal_id,
            "accepted": result.accepted,
            "reason_code": result.reason_code,
            "commit_hash": result.commit_hash,
        }
