"""Leased OpenClaw research and isolated code-workflow handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.agents.code_worker import AgentCodeWorkflow
from src.agents.compiler import AgentCompilationError, compile_openclaw_candidate_payload
from src.agents.context import AgentContext
from src.agents.store import SqlAgentStore
from src.domain._codec import canonical_hash
from src.observability.reports import DatabasePlatformReport
from src.research.coordinator import ResearchCoordinator
from src.research.datasets import SqlDatasetBundleRepository
from src.research.store import SqlResearchStore
from src.research.theses import SqlThesisRegistry, ThesisError
from src.services.job_schemas import JobSchemaError, ResearchJobRequest
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
            "agent_review": self.agent_review,
        }

    def agent_review(self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]) -> dict[str, Any]:
        renew()
        now = str(claimed.payload.get("available_at") or utc_now())
        report = DatabasePlatformReport(self.queue.engine).build(now=now)
        context = build_agent_review_context(report, created_at=now)
        review_id = (
            f"scheduled-agent-review:{canonical_hash(context.values).removeprefix('sha256:')}"
        )
        self.store.save_review(
            proposal_id=review_id,
            created_at=context.created_at,
            payload={
                "schema": "platform.agent_review_request/v1",
                "review_id": review_id,
                "context": dict(context.values),
                "context_hash": context.content_hash,
            },
        )
        return {
            "review_id": review_id,
            "context_hash": context.content_hash,
            "reason_code": "sanitised_agent_review_recorded",
        }

    def agent_research(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        proposal = self.store.proposal(str(claimed.payload["proposal_id"]))
        compiled_candidate_id = None
        if proposal.economic_thesis is not None:
            bundle = SqlDatasetBundleRepository(self.queue.engine).latest_ready(
                proposal.product_id,
                at=proposal.created_at,
            )
            if bundle is None:
                self.store.save_disposition(
                    proposal_id=proposal.proposal_id,
                    created_at=utc_now(),
                    payload={
                        "accepted": False,
                        "reason_code": "canonical_dataset_bundle_unavailable",
                        "live_eligible": False,
                    },
                )
                return {
                    "proposal_id": proposal.proposal_id,
                    "state": "waiting_for_dataset",
                }
            try:
                candidate = compile_openclaw_candidate_payload(
                    proposal,
                    bundle=bundle,
                    submitted_at=proposal.created_at,
                )
                SqlThesisRegistry(self.queue.engine).register(proposal.economic_thesis)
                compiled_candidate_id = ResearchCoordinator(
                    SqlResearchStore(self.queue.engine)
                ).submit(candidate)
            except (AgentCompilationError, ThesisError, ValueError) as exc:
                self.store.save_disposition(
                    proposal_id=proposal.proposal_id,
                    created_at=utc_now(),
                    payload={
                        "accepted": False,
                        "reason_code": "openclaw_candidate_compilation_rejected",
                        "detail": str(exc),
                        "live_eligible": False,
                    },
                )
                return {
                    "proposal_id": proposal.proposal_id,
                    "state": "rejected",
                    "reason_code": "openclaw_candidate_compilation_rejected",
                }
        queued: list[str] = []
        for index, item in enumerate(proposal.research_jobs):
            name = str(item.get("name") or "")
            if name not in ALLOWED_RESEARCH_JOBS:
                raise ValueError(f"agent requested unsupported research job: {name}")
            maximum_seconds = int(item.get("maximum_seconds", self.maximum_runtime_seconds))
            if not 1 <= maximum_seconds <= self.maximum_runtime_seconds:
                raise ValueError("agent research job exceeds its runtime budget")
            raw_request = item.get("request")
            if not isinstance(raw_request, dict):
                raise JobSchemaError("agent research item has no typed request")
            request = ResearchJobRequest.from_mapping(raw_request)
            job_id = f"agent:{proposal.proposal_id}:research:{index}"
            self.queue.enqueue_if_absent(
                job_id=job_id,
                name=name,
                payload=request.to_payload(),
                available_at=proposal.created_at,
                priority=5,
                producer_identity=f"agent:{proposal.proposal_id}",
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
                "candidate_id": compiled_candidate_id,
                "live_eligible": False,
            },
        )
        return {
            "proposal_id": proposal.proposal_id,
            "research_job_ids": queued,
            "candidate_id": compiled_candidate_id,
        }

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


def build_agent_review_context(report: Mapping[str, Any], *, created_at: str) -> AgentContext:
    """Extract only adaptive, non-protected research feedback for OpenClaw."""

    research = report.get("research")
    research = research if isinstance(research, Mapping) else {}
    funnel = research.get("funnel")
    funnel = funnel if isinstance(funnel, Mapping) else {}
    values = {
        "failure_reasons": dict(funnel.get("top_rejection_reasons", {})),
        "family_coverage": dict(funnel.get("feature_family_concentration", {})),
        "duplicate_feedback": {
            "exact_duplicate_rate": funnel.get("exact_duplicate_rate", 0.0),
            "near_duplicate_rate": funnel.get("near_duplicate_rate", 0.0),
        },
        "data_availability": dict(funnel.get("missing_stage_datasets", {})),
        "research_queue": {
            "candidates_generated": funnel.get("candidates_generated", 0),
            "candidates_evaluated": funnel.get("candidates_evaluated", 0),
            "candidates_never_evaluated": funnel.get("candidates_never_evaluated", []),
        },
        "generation_feedback": {
            "theses_generated": funnel.get("theses_generated", 0),
            "cumulative_trial_count": funnel.get("cumulative_trial_count", 0),
        },
        "research_progress": {
            "candidates_rejected_by_stage": dict(funnel.get("candidates_rejected_by_stage", {})),
            "first_blocked_stage": dict(funnel.get("first_blocked_stage", {})),
            "jobs_dead_letter": funnel.get("jobs_dead_letter", 0),
        },
    }
    return AgentContext(created_at=created_at, values=values)
