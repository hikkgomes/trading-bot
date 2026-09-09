"""Scoped HTTP operations for the single durable external review consumer."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from src.agents.openclaw_bridge import OpenClawAgentBridge
from src.agents.store import SqlAgentStore
from src.data.database import job
from src.domain._codec import timestamp
from src.services.health import DatabaseHeartbeatStore
from src.services.scheduler import ClaimedJob, DatabaseJobQueue


class AgentReviewTransport:
    def __init__(self, queue: DatabaseJobQueue, *, key: bytes, node_id: str = "agent-transport"):
        if len(key) < 32:
            raise ValueError("agent transport key requires at least 32 bytes")
        self.queue = queue
        self.key = key
        self.node_id = node_id
        self.store = SqlAgentStore(queue.engine)
        self.bridge = OpenClawAgentBridge(store=self.store, queue=queue)

    def next_request(self, *, now: str) -> dict[str, Any]:
        worker_id = "openclaw-review-transport"
        DatabaseHeartbeatStore(self.queue.engine).record(
            service_name="agent-sandbox",
            node_id=self.node_id,
            observed_at=now,
            healthy=True,
            payload={"reason_code": "authenticated_transport_poll"},
        )
        self.queue.register_worker(
            worker_id=worker_id,
            node_id=self.node_id,
            role="agent-transport",
            capabilities=("agent_delivery",),
            observed_at=now,
        )
        claimed = self.queue.claim(
            worker_id=worker_id, now=now, lease_seconds=600, names=("agent_delivery",)
        )
        if claimed is None:
            return {"request": None}
        request = self.bridge.review_request(str(claimed.payload["review_id"]))
        accepted = self.store.review_response(request["review_id"])
        if accepted is not None:
            self.respond(accepted["envelope"], now=now)
            return {"request": None}
        if timestamp(now, field="now") >= timestamp(request["expires_at"], field="expires_at"):
            self.queue.complete(
                claimed,
                completed_at=now,
                outcome={"state": "rejected", "reason_code": "review_expired"},
            )
            return {"request": None}
        self.store.save_review(
            proposal_id=request["review_id"],
            created_at=now,
            payload={
                "schema": "platform.agent_delivery/v1",
                "state": "delivered",
                "attempt": claimed.attempt,
            },
        )
        return {"request": request}

    def respond(self, envelope: dict[str, Any], *, now: str) -> dict[str, Any]:
        job_id = f"agent-delivery:{envelope.get('review_id')}"
        with self.queue.engine.connect() as connection:
            row = connection.execute(select(job).where(job.c.id == job_id)).mappings().one_or_none()
        if row is None or row["state"] not in {"running", "completed"}:
            raise PermissionError("OpenClaw response has no delivered request")
        proposal = self.bridge.ingest_review_response(envelope, key=self.key, now=now)
        if row["state"] == "running":
            claimed = ClaimedJob(
                job_id=job_id,
                name=row["name"],
                payload=dict(row["payload"]),
                worker_id=row["lease_owner"],
                attempt=row["attempts"],
                lease_expires_at=row["lease_expires_at"],
            )
            self.queue.complete(
                claimed,
                completed_at=now,
                outcome={"state": "responded", "proposal_id": proposal.proposal_id},
            )
        return {"state": "responded", "proposal_id": proposal.proposal_id}
