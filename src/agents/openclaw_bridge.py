"""Typed OpenClaw intake into the bounded agent workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import select

from src.agents.proposals import AgentAction, AgentProposal, AgentRole
from src.agents.store import SqlAgentStore
from src.agents.thesis import AgentThesisError, parse_openclaw_thesis
from src.data.database import agent_review
from src.domain._codec import canonical_hash
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
    raw_thesis = payload.get("thesis")
    economic_thesis = None
    if isinstance(raw_thesis, dict):
        economic_thesis = parse_openclaw_thesis(
            raw_thesis,
            product_id=str(payload.get("product_id") or ""),
            created_at=str(payload.get("created_at") or ""),
        )
        thesis = economic_thesis.market_rationale
    elif isinstance(raw_thesis, str):
        thesis = raw_thesis
    else:
        raise AgentThesisError("OpenClaw thesis must be text or a typed object")
    return AgentProposal(
        proposal_id=str(payload.get("proposal_id") or ""),
        role=AgentRole(str(payload.get("role") or "")),
        action=AgentAction(str(payload.get("action") or "")),
        product_id=str(payload.get("product_id") or ""),
        created_at=str(payload.get("created_at") or ""),
        thesis=thesis,
        economic_thesis=economic_thesis,
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
        job_name = (
            "agent_research"
            if proposal.economic_thesis is not None
            else ("agent_code_workflow" if proposal.action in CODE_ACTIONS else "agent_research")
        )
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

    def review_request(self, review_id: str) -> dict[str, Any]:
        """Read one durable scheduled review request for the external agent."""

        with self.queue.engine.connect() as connection:
            rows = connection.execute(select(agent_review.c.payload)).scalars()
            for raw in rows:
                if (
                    isinstance(raw, Mapping)
                    and raw.get("schema") == "platform.agent_review_request/v1"
                    and raw.get("review_id") == review_id
                ):
                    request = dict(raw)
                    context = request.get("context")
                    if (
                        not isinstance(context, Mapping)
                        or request.get("context_hash") != canonical_hash(context)
                        or request.get("transport") != "postgresql_agent_review"
                        or request.get("response_required") is not True
                    ):
                        raise ValueError("OpenClaw review request has invalid provenance")
                    return request
        raise KeyError(f"OpenClaw review request does not exist: {review_id}")

    def ingest_review_response(self, payload: dict[str, Any]) -> AgentProposal:
        """Validate a proposal against a scheduled request and record the round trip."""

        provenance = payload.get("provenance")
        review_id = provenance.get("review_id") if isinstance(provenance, dict) else None
        if not isinstance(review_id, str) or not review_id.strip():
            raise ValueError("OpenClaw review responses require provenance.review_id")
        request = self.review_request(review_id)
        request_context = request.get("context")
        request_product = (
            request_context.get("product_id") if isinstance(request_context, Mapping) else None
        )
        if not isinstance(request_product, str) or payload.get("product_id") != request_product:
            raise ValueError("OpenClaw response product does not match its review request")
        proposal = self.ingest(payload)
        response = {
            "schema": "platform.agent_review_response/v1",
            "review_id": review_id,
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.content_hash,
            "request_hash": canonical_hash(request),
            "response_hash": canonical_hash(
                {key: value for key, value in payload.items() if key != "signature"}
            ),
        }
        self.store.save_review(
            proposal_id=review_id,
            created_at=proposal.created_at,
            payload=response,
        )
        return proposal
