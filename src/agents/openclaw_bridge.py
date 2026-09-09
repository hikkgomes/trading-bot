"""Typed OpenClaw intake into the bounded agent workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import select

from src.agents.context import AgentContext
from src.agents.proposals import AgentAction, AgentProposal, AgentRole
from src.agents.store import SqlAgentStore
from src.agents.thesis import AgentThesisError, parse_openclaw_thesis
from src.agents.transport import verify_review_response
from src.data.database import agent_review
from src.domain._codec import canonical_hash, timestamp
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
            rows = (
                connection.execute(
                    select(agent_review.c.created_at, agent_review.c.payload)
                    .where(
                        agent_review.c.payload["schema"].as_string()
                        == "platform.agent_review_request/v1",
                        agent_review.c.payload["review_id"].as_string() == review_id,
                    )
                    .limit(2)
                )
                .mappings()
                .all()
            )
        if not rows:
            raise KeyError(f"OpenClaw review request does not exist: {review_id}")
        if len(rows) != 1:
            raise ValueError("OpenClaw review request identity is ambiguous")
        request = dict(rows[0]["payload"])
        context = AgentContext(created_at=str(rows[0]["created_at"]), values=request["context"])
        if (
            request.get("context_hash") != context.content_hash
            or request.get("requested_at") != context.created_at
            or request.get("transport") != "postgresql_agent_review"
            or request.get("response_required") is not True
        ):
            raise ValueError("OpenClaw review request has invalid provenance")
        return request

    def enqueue_review_delivery(self, review_id: str, *, available_at: str) -> None:
        self.queue.enqueue_if_absent(
            job_id=f"agent-delivery:{review_id}",
            name="agent_delivery",
            payload={"review_id": review_id},
            available_at=available_at,
            priority=10,
        )

    def ingest_review_response(
        self, envelope: dict[str, Any], *, key: bytes, now: str
    ) -> AgentProposal:
        """Validate a proposal against a scheduled request and record the round trip."""

        review_id = envelope.get("review_id")
        if not isinstance(review_id, str) or not review_id.strip():
            raise ValueError("OpenClaw review responses require provenance.review_id")
        request = self.review_request(review_id)
        accepted = self.store.review_response(review_id)
        accepted_at = (
            accepted["accepted_at"]
            if accepted is not None and accepted.get("envelope") == envelope
            else now
        )
        payload = verify_review_response(envelope, request, key=key, now=accepted_at)
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("review_id") != review_id:
            raise ValueError("OpenClaw proposal provenance does not match its review request")
        request_context = request.get("context")
        request_product = (
            request_context.get("product_id") if isinstance(request_context, Mapping) else None
        )
        if not isinstance(request_product, str) or payload.get("product_id") != request_product:
            raise ValueError("OpenClaw response product does not match its review request")
        proposal = build_agent_proposal(payload)
        if (
            not request["requested_at"]
            <= proposal.created_at
            <= timestamp(accepted_at, field="accepted_at")
        ):
            raise ValueError("OpenClaw proposal timestamp is outside its review interval")
        response = {
            "schema": "platform.agent_review_response/v1",
            "review_id": review_id,
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.content_hash,
            "request_hash": canonical_hash(request),
            "response_hash": canonical_hash(payload),
            "accepted_at": accepted_at,
            "envelope": envelope,
            "state": "responded",
        }
        self.store.bind_review_response(
            review_id=review_id,
            created_at=accepted_at,
            payload=response,
        )
        return self.ingest(payload)
