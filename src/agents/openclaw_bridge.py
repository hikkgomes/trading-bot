"""Typed OpenClaw intake into the bounded agent workflow."""

from __future__ import annotations

from typing import Any

from src.agents.proposals import AgentAction, AgentProposal, AgentRole
from src.agents.store import SqlAgentStore
from src.services.scheduler import DatabaseJobQueue

CODE_ACTIONS = frozenset(
    {
        AgentAction.REVISE_STRATEGY,
        AgentAction.CREATE_DSL,
        AgentAction.CREATE_PYTHON_STRATEGY,
        AgentAction.CREATE_FEATURE,
        AgentAction.CREATE_DATA_ADAPTER,
        AgentAction.CREATE_PARAMETER_SPACE,
        AgentAction.CREATE_ML_EXPERIMENT,
        AgentAction.CREATE_ENSEMBLE,
        AgentAction.CREATE_TESTS,
        AgentAction.PRODUCE_BRANCH,
        AgentAction.PRODUCE_MERGE_REQUEST,
    }
)


def build_agent_proposal(payload: dict[str, Any]) -> AgentProposal:
    allowed = {
        "schema",
        "source",
        "proposal_id",
        "role",
        "action",
        "product_id",
        "created_at",
        "thesis",
        "files",
        "research_jobs",
        "provenance",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"OpenClaw agent proposal contains unknown fields: {sorted(unknown)}")
    if payload.get("schema") != "openclaw.agent_proposal/v1":
        raise ValueError("OpenClaw agent proposal schema is unsupported")
    if payload.get("source") != "openclaw":
        raise ValueError("agent proposal source must be openclaw")
    return AgentProposal(
        proposal_id=str(payload.get("proposal_id") or ""),
        role=AgentRole(str(payload.get("role") or "")),
        action=AgentAction(str(payload.get("action") or "")),
        product_id=str(payload.get("product_id") or ""),
        created_at=str(payload.get("created_at") or ""),
        thesis=str(payload.get("thesis") or ""),
        files=payload.get("files") if isinstance(payload.get("files"), dict) else {},
        research_jobs=tuple(payload.get("research_jobs") or ()),
        provenance=payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {},
    )


class OpenClawAgentBridge:
    def __init__(self, *, store: SqlAgentStore, queue: DatabaseJobQueue):
        self.store = store
        self.queue = queue

    def ingest(self, payload: dict[str, Any]) -> AgentProposal:
        proposal = build_agent_proposal(payload)
        self.store.save_proposal(proposal)
        job_name = "agent_code_workflow" if proposal.action in CODE_ACTIONS else "agent_research"
        self.queue.enqueue_if_absent(
            job_id=f"agent:{proposal.proposal_id}",
            name=job_name,
            payload={
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.content_hash,
                "agent_may_submit_orders": False,
            },
            available_at=proposal.created_at,
            priority=20 if proposal.action in CODE_ACTIONS else 10,
        )
        return proposal
