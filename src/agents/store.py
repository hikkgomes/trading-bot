"""PostgreSQL audit records for agent proposals, patches, reviews, and outcomes."""

from __future__ import annotations

from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from src.agents.proposals import AgentAction, AgentProposal, AgentRole
from src.agents.thesis import parse_openclaw_thesis
from src.data.database import (
    agent_action,
    agent_disposition,
    agent_patch,
    agent_proposal,
    agent_review,
)
from src.domain._codec import canonical_hash, json_value, timestamp, to_primitive


class SqlAgentStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def save_proposal(self, proposal) -> str:
        payload = to_primitive(proposal)
        self._append(
            agent_proposal,
            record_id=proposal.proposal_id,
            created_at=proposal.created_at,
            payload={**payload, "content_hash": proposal.content_hash},
        )
        self._append(
            agent_action,
            record_id=f"{proposal.proposal_id}:submitted",
            created_at=proposal.created_at,
            payload={"proposal_id": proposal.proposal_id, "action": proposal.action.value},
        )
        return proposal.proposal_id

    def save_patch(self, *, proposal_id: str, created_at: str, payload: dict[str, Any]) -> str:
        return self._content_record(agent_patch, proposal_id, created_at, payload)

    def save_review(self, *, proposal_id: str, created_at: str, payload: dict[str, Any]) -> str:
        return self._content_record(agent_review, proposal_id, created_at, payload)

    def save_disposition(
        self, *, proposal_id: str, created_at: str, payload: dict[str, Any]
    ) -> str:
        return self._content_record(agent_disposition, proposal_id, created_at, payload)

    def dataset_binding(self, proposal_id: str) -> dict[str, Any] | None:
        identity = canonical_hash({"kind": "agent_dataset_binding", "proposal_id": proposal_id})
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(agent_review.c.payload).where(agent_review.c.id == identity)
            ).scalar_one_or_none()
        if payload is None:
            return None
        if (
            payload.get("proposal_id") != proposal_id
            or payload.get("kind") != "agent_dataset_binding"
        ):
            raise ValueError("agent dataset binding belongs to another proposal or record type")
        return dict(payload)

    def bind_review_response(
        self, *, review_id: str, created_at: str, payload: dict[str, Any]
    ) -> None:
        """Seal one reply before intake; replay may finish interrupted queue writes."""
        identity = canonical_hash({"kind": "agent_review_response", "review_id": review_id})
        try:
            self._append(agent_review, record_id=identity, created_at=created_at, payload=payload)
        except (IntegrityError, ValueError):
            # A concurrent reply won the unique key. Accept only an identical replay.
            with self.engine.connect() as connection:
                existing = connection.execute(
                    select(agent_review.c.payload).where(agent_review.c.id == identity)
                ).scalar_one_or_none()
            if existing is None or {
                name: value for name, value in existing.items() if name != "accepted_at"
            } != {name: value for name, value in payload.items() if name != "accepted_at"}:
                raise ValueError("agent review already has a different response") from None

    def review_response(self, review_id: str) -> dict[str, Any] | None:
        identity = canonical_hash({"kind": "agent_review_response", "review_id": review_id})
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(agent_review.c.payload).where(agent_review.c.id == identity)
            ).scalar_one_or_none()
        return dict(payload) if payload is not None else None

    def bind_dataset(self, *, proposal_id: str, bundle_id: str, resolved_at: str) -> dict[str, Any]:
        payload = {
            "proposal_id": proposal_id,
            "kind": "agent_dataset_binding",
            "bundle_id": bundle_id,
            "resolved_at": timestamp(resolved_at, field="resolved_at"),
        }
        identity = canonical_hash({"kind": "agent_dataset_binding", "proposal_id": proposal_id})
        self._append(agent_review, record_id=identity, created_at=resolved_at, payload=payload)
        return payload

    def _content_record(
        self, table, proposal_id: str, created_at: str, payload: dict[str, Any]
    ) -> str:
        clean = json_value({"proposal_id": proposal_id, **payload}, field="agent record")
        identity = canonical_hash(clean)
        self._append(table, record_id=identity, created_at=created_at, payload=clean)
        return identity

    def _append(self, table, *, record_id: str, created_at: str, payload: dict[str, Any]) -> None:
        created_at = timestamp(created_at, field="created_at")
        clean = json_value(payload, field="agent record")
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(table.c.payload).where(table.c.id == record_id)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != clean:
                    raise ValueError("agent audit record identity collision")
                return
            connection.execute(
                insert(table).values(id=record_id, created_at=created_at, payload=clean)
            )

    def records(self, table_name: str) -> tuple[dict[str, Any], ...]:
        tables = {
            "action": agent_action,
            "proposal": agent_proposal,
            "patch": agent_patch,
            "review": agent_review,
            "disposition": agent_disposition,
        }
        table = tables[table_name]
        with self.engine.connect() as connection:
            return tuple(
                dict(payload)
                for payload in connection.execute(
                    select(table.c.payload).order_by(table.c.created_at, table.c.id)
                ).scalars()
            )

    def proposal(self, proposal_id: str) -> AgentProposal:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(agent_proposal.c.payload).where(agent_proposal.c.id == proposal_id)
            ).scalar_one_or_none()
        if payload is None:
            raise KeyError(f"agent proposal does not exist: {proposal_id}")
        values = dict(payload)
        values.pop("content_hash", None)
        values["role"] = AgentRole(values["role"])
        values["action"] = AgentAction(values["action"])
        if values.get("economic_thesis") is not None:
            thesis_payload = dict(values["economic_thesis"])
            thesis_payload.pop("created_at", None)
            thesis_payload.pop("creator_identity", None)
            values["economic_thesis"] = parse_openclaw_thesis(
                thesis_payload,
                product_id=str(values["product_id"]),
                created_at=str(values["created_at"]),
            )
        values["research_jobs"] = tuple(values.get("research_jobs") or ())
        return AgentProposal(**values)
