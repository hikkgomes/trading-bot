"""Leased OpenClaw research and isolated code-workflow handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from src.agents.code_worker import AgentCodeWorkflow
from src.agents.compiler import AgentCompilationError, compile_openclaw_candidate_payload
from src.agents.context import AgentContext
from src.agents.openclaw_bridge import OpenClawAgentBridge
from src.agents.store import SqlAgentStore
from src.data.database import (
    dataset_bundle,
    research_thesis,
    strategy_definition,
    strategy_identity,
    thesis_trial,
    validation_stage,
)
from src.domain._codec import canonical_hash
from src.research.coordinator import ResearchCoordinator
from src.research.datasets import SqlDatasetBundleRepository
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
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
        review_id = "scheduled-agent-review:" + canonical_hash(
            {"job_id": claimed.job_id}
        ).removeprefix("sha256:")
        bridge = OpenClawAgentBridge(store=self.store, queue=self.queue)
        try:
            existing = bridge.review_request(review_id)
        except KeyError:
            existing = None
        if existing is not None:
            bridge.enqueue_review_delivery(review_id, available_at=existing["requested_at"])
            return {
                "review_id": review_id,
                "context_hash": existing["context_hash"],
                "reason_code": "sanitised_agent_review_recorded",
            }
        now = utc_now()
        report = adaptive_research_report(self.queue.engine)
        context = build_agent_review_context(
            report,
            created_at=now,
            product_id=str(claimed.payload.get("product_id") or ""),
        )
        self.store.save_review(
            proposal_id=review_id,
            created_at=context.created_at,
            payload={
                "schema": "platform.agent_review_request/v1",
                "state": "requested",
                "review_id": review_id,
                "context": dict(context.values),
                "context_hash": context.content_hash,
                "transport": "postgresql_agent_review",
                "response_required": True,
                "requested_at": now,
                "expires_at": (datetime.fromisoformat(now) + timedelta(hours=1)).isoformat(),
            },
        )
        bridge.enqueue_review_delivery(review_id, available_at=now)
        return {
            "review_id": review_id,
            "context_hash": context.content_hash,
            "reason_code": "sanitised_agent_review_recorded",
        }

    def agent_research(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        renew()
        now = utc_now()
        proposal = self.store.proposal(str(claimed.payload["proposal_id"]))
        compiled_candidate_id = None
        if proposal.economic_thesis is not None:
            datasets = SqlDatasetBundleRepository(self.queue.engine)
            binding = self.store.dataset_binding(proposal.proposal_id)
            bundle = (
                datasets.get(binding["bundle_id"])
                if binding
                else datasets.latest_ready(proposal.product_id, at=now)
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
                    "outcome_category": "blocked_data",
                    "_defer": True,
                    "_defer_reason": "canonical_dataset_bundle_unavailable",
                }
            if binding is None:
                binding = self.store.bind_dataset(
                    proposal_id=proposal.proposal_id, bundle_id=bundle.bundle_id, resolved_at=now
                )
            try:
                candidate = compile_openclaw_candidate_payload(
                    proposal,
                    bundle=bundle,
                    submitted_at=binding["resolved_at"],
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
                        "state": "rejected",
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
                "state": "compiled" if compiled_candidate_id else "dispatched",
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
            publish=bool(claimed.payload.get("publish", False)),
        )
        return {
            "proposal_id": proposal.proposal_id,
            "accepted": result.accepted,
            "reason_code": result.reason_code,
            "commit_hash": result.commit_hash,
        }


def build_agent_review_context(
    report: Mapping[str, Any], *, created_at: str, product_id: str | None = None
) -> AgentContext:
    """Extract only adaptive, non-protected research feedback for OpenClaw."""

    research = report.get("research")
    research = research if isinstance(research, Mapping) else {}
    funnel = research.get("funnel")
    funnel = funnel if isinstance(funnel, Mapping) else {}
    values: dict[str, Any] = {
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
    if product_id:
        values["product_id"] = product_id
    return AgentContext(created_at=created_at, values=values)


def adaptive_research_report(engine) -> dict[str, Any]:
    """Read only public development tables; never build an operator report for an agent."""
    public = validation_stage.c.stage.in_(("screening", "development", "robustness"))
    current = validation_stage.c.engine_version == ECONOMIC_ENGINE_VERSION
    family = strategy_definition.c.definition["family"].as_string()
    candidates = SqlResearchStore(engine).candidates
    with engine.connect() as connection:
        counts = {
            table.name: int(
                connection.execute(select(func.count()).select_from(table)).scalar_one()
            )
            for table in (research_thesis, thesis_trial, strategy_identity)
        }
        counts["experiment"] = int(
            connection.execute(
                select(func.count())
                .select_from(candidates)
                .where(candidates.c.state != "legacy_import")
            ).scalar_one()
        )
        reasons = dict(
            connection.execute(
                select(validation_stage.c.reason_code, func.count())
                .where(public, current, validation_stage.c.accepted.is_(False))
                .group_by(validation_stage.c.reason_code)
            ).all()
        )
        rejected = dict(
            connection.execute(
                select(validation_stage.c.stage, func.count())
                .where(public, current, validation_stage.c.accepted.is_(False))
                .group_by(validation_stage.c.stage)
            ).all()
        )
        families = dict(connection.execute(select(family, func.count()).group_by(family)).all())
        evaluated = int(
            connection.execute(
                select(func.count(func.distinct(validation_stage.c.experiment_id))).where(
                    public, current
                )
            ).scalar_one()
        )
        duplicates = int(
            connection.execute(
                select(func.count())
                .select_from(strategy_identity)
                .where(strategy_identity.c.is_duplicate.is_(True))
            ).scalar_one()
        )
        bundles = set(connection.execute(select(dataset_bundle.c.product_id)).scalars())
    return {
        "research": {
            "funnel": {
                "top_rejection_reasons": {str(key): count for key, count in reasons.items() if key},
                "candidates_rejected_by_stage": rejected,
                "feature_family_concentration": {
                    str(key or "unclassified"): value for key, value in families.items()
                },
                "candidates_generated": counts["experiment"],
                "candidates_evaluated": evaluated,
                "theses_generated": counts["research_thesis"],
                "cumulative_trial_count": counts["thesis_trial"],
                "exact_duplicate_rate": duplicates / max(1, counts["strategy_identity"]),
                "missing_stage_datasets": {
                    product: "no_canonical_bundle"
                    for product in ("btc_accumulation", "active_income")
                    if product not in bundles
                },
            }
        }
    }
