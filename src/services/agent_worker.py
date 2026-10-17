"""Leased OpenClaw research and isolated code-workflow handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select

from src.agents.code_worker import AgentCodeWorkflow
from src.agents.compiler import AgentCompilationError, compile_openclaw_candidate_payload
from src.agents.context import AgentContext
from src.agents.openclaw_bridge import OpenClawAgentBridge
from src.agents.store import SqlAgentStore
from src.agents.thesis import THESIS_FIELDS
from src.data.database import (
    generation_feedback,
    instrument,
    research_thesis,
    strategy_definition,
    strategy_identity,
    strategy_version,
    thesis_trial,
    validation_stage,
)
from src.data.feature_graph import FeatureGraphError, FeatureGraphRegistry
from src.domain._codec import canonical_hash
from src.research.coordinator import ResearchCoordinator
from src.research.datasets import SqlDatasetBundleRepository
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.generation import CAMPAIGNS
from src.research.providers import DIRECTIONAL_DSL_DIRECTIONS, DIRECTIONAL_DSL_FAMILIES
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
_MAX_REVIEW_INSTRUMENTS = 12
_MAX_PUBLIC_FEATURE_DEFINITIONS = 200
_MAX_FAILURE_CATEGORIES = 10


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
        product_id = str(claimed.payload.get("product_id") or "")
        report = adaptive_research_report(self.queue.engine, product_id=product_id, at=now)
        context = build_agent_review_context(
            report,
            created_at=now,
            product_id=product_id,
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

    product_id = str(product_id or "")
    scope_matches = bool(product_id) and report.get("product_id") == product_id
    research = report.get("research") if scope_matches else None
    research = research if isinstance(research, Mapping) else {}
    funnel = research.get("funnel")
    funnel = funnel if isinstance(funnel, Mapping) else {}
    data = research.get("product_inputs")
    data = data if isinstance(data, Mapping) else {}
    strategy_catalogue = research.get("strategy_catalogue")
    strategy_catalogue = strategy_catalogue if isinstance(strategy_catalogue, Mapping) else {}
    feature_catalogue = research.get("feature_catalogue")
    feature_catalogue = feature_catalogue if isinstance(feature_catalogue, Mapping) else {}
    development = research.get("development_results")
    development = development if isinstance(development, Mapping) else {}
    input_universe = data.get("instrument_universe")
    input_universe = input_universe if isinstance(input_universe, Mapping) else {}
    supported_families = strategy_catalogue.get("families")
    supported_families = supported_families if isinstance(supported_families, list) else []
    supported_by_family = feature_catalogue.get("supported_by_family", {})
    supported_by_family = supported_by_family if isinstance(supported_by_family, Mapping) else {}
    symbols = input_universe.get("symbols")
    instrument_ids = input_universe.get("instrument_ids")
    available_data = data.get("available_data")
    available_data = available_data if isinstance(available_data, list) else []
    ready = bool(
        scope_matches
        and data.get("hypothesis_inputs_ready") is True
        and supported_families
        and all(
            isinstance(supported_by_family.get(family), list | tuple)
            and supported_by_family[family]
            for family in supported_families
        )
        and isinstance(symbols, list)
        and symbols
        and isinstance(instrument_ids, list)
        and len(symbols) == len(instrument_ids)
        and available_data
    )
    safe_catalogue_fields = (
        "mechanism_categories_by_family",
        "allowed_directions",
        "allowed_operators",
        "evidence_type",
        "thesis_fields",
        "generalisation_scope_fields",
        "cumulative_trial_budget_min",
        "cumulative_trial_budget_max",
        "maximum_symbols_per_hypothesis",
    )
    safe_data_fields = (
        "canonical_bundle_available",
        "timeframe",
    )
    unavailable_reason = str(data.get("reason_code") or "product_inputs_unavailable")
    if unavailable_reason not in {
        "canonical_dataset_bundle_unavailable",
        "canonical_bundle_inputs_incomplete",
        "no_supported_directional_campaign_for_available_data",
        "product_inputs_unavailable",
    }:
        unavailable_reason = "product_inputs_unavailable"
    data_availability = {field: data[field] for field in safe_data_fields if field in data}
    data_availability.update(
        {
            "hypothesis_inputs_ready": ready,
            "reason_code": None if ready else unavailable_reason,
            "available_data": (
                [str(item) for item in available_data if isinstance(item, str)] if ready else []
            ),
            "available_public_stages": (
                [
                    stage
                    for stage in data.get("available_public_stages", [])
                    if stage in {"screening", "development"}
                ]
                if ready and isinstance(data.get("available_public_stages"), list)
                else []
            ),
        }
    )
    values: dict[str, Any] = {
        "failure_reasons": dict(funnel.get("top_rejection_reasons", {})),
        "family_coverage": dict(funnel.get("feature_family_concentration", {})),
        "duplicate_feedback": {
            "exact_duplicate_rate": funnel.get("exact_duplicate_rate", 0.0),
            "near_duplicate_rate": funnel.get("near_duplicate_rate", 0.0),
            "exact_duplicate_count": funnel.get("exact_duplicate_count", 0),
            "near_duplicate_count": funnel.get("near_duplicate_count", 0),
        },
        "data_availability": data_availability,
        "instrument_universe": dict(input_universe)
        if ready
        else {"symbols": [], "instrument_ids": [], "maximum_symbols_per_hypothesis": 1},
        "strategy_catalogue": {
            **{
                field: strategy_catalogue[field]
                for field in safe_catalogue_fields
                if ready and field in strategy_catalogue
            },
            "available": ready,
            "unavailable_reason": None if ready else unavailable_reason,
            "families": supported_families if ready else [],
        },
        "feature_catalogue": {
            "supported_by_family": {
                str(family): list(features)
                for family, features in supported_by_family.items()
                if ready and isinstance(features, list | tuple)
            },
            "publicly_observed_by_stage": (
                dict(feature_catalogue.get("publicly_observed_by_stage", {})) if ready else {}
            ),
            "public_metadata_truncated": bool(
                feature_catalogue.get("public_metadata_truncated", False)
            )
            if ready
            else False,
        },
        "development_results": dict(development),
        "research_queue": {
            "candidates_generated": funnel.get("candidates_generated", 0),
            "candidates_evaluated": funnel.get("candidates_evaluated", 0),
            "candidates_never_evaluated_count": funnel.get("candidates_never_evaluated_count", 0),
        },
        "generation_feedback": {
            "by_campaign": dict(research.get("generation_feedback", {})),
        },
        "strategy_lineage": {
            "theses_generated": funnel.get("theses_generated", 0),
            "cumulative_trial_count": funnel.get("cumulative_trial_count", 0),
        },
        "research_progress": {
            "candidates_rejected_by_stage": dict(funnel.get("candidates_rejected_by_stage", {})),
        },
    }
    if product_id and scope_matches:
        values["product_id"] = product_id
    return AgentContext(created_at=created_at, values=values)


def adaptive_research_report(engine, *, product_id: str, at: str) -> dict[str, Any]:
    """Read public research metadata for one product at the review time."""
    bundle = SqlDatasetBundleRepository(engine).latest_ready(product_id, at=at)
    candidates = SqlResearchStore(engine).candidates
    candidate_source = candidates.join(
        strategy_version, candidates.c.strategy_version_id == strategy_version.c.id
    ).join(strategy_definition, strategy_version.c.definition_id == strategy_definition.c.id)
    stage_source = (
        validation_stage.join(candidates, validation_stage.c.experiment_id == candidates.c.id)
        .join(strategy_version, candidates.c.strategy_version_id == strategy_version.c.id)
        .join(strategy_definition, strategy_version.c.definition_id == strategy_definition.c.id)
    )
    diagnostic = or_(
        candidates.c.metadata["diagnostic"].as_boolean().is_(True),
        strategy_definition.c.definition["metadata"]["diagnostic"].as_boolean().is_(True),
    )
    candidate_scope = and_(
        strategy_definition.c.product_id == product_id,
        candidates.c.state != "legacy_import",
        ~diagnostic,
    )
    public_stage_scope = and_(
        strategy_definition.c.product_id == product_id,
        candidates.c.state != "legacy_import",
        ~diagnostic,
        validation_stage.c.stage.in_(("screening", "development")),
        validation_stage.c.engine_version == ECONOMIC_ENGINE_VERSION,
    )
    with engine.connect() as connection:
        candidate_count = int(
            connection.execute(
                select(func.count()).select_from(candidate_source).where(candidate_scope)
            ).scalar_one()
        )
        stage_counts = dict(
            connection.execute(
                select(validation_stage.c.stage, func.count())
                .select_from(stage_source)
                .where(public_stage_scope)
                .group_by(validation_stage.c.stage)
            ).all()
        )
        rejected = dict(
            connection.execute(
                select(validation_stage.c.stage, func.count())
                .select_from(stage_source)
                .where(public_stage_scope, validation_stage.c.accepted.is_(False))
                .group_by(validation_stage.c.stage)
            ).all()
        )
        reasons = dict(
            connection.execute(
                select(validation_stage.c.reason_code, func.count())
                .select_from(stage_source)
                .where(
                    public_stage_scope,
                    validation_stage.c.stage == "development",
                    validation_stage.c.accepted.is_(False),
                    validation_stage.c.reason_code.is_not(None),
                    validation_stage.c.reason_code != "",
                )
                .group_by(validation_stage.c.reason_code)
                .order_by(func.count().desc(), validation_stage.c.reason_code)
                .limit(_MAX_FAILURE_CATEGORIES)
            ).all()
        )
        family_counts = dict(
            connection.execute(
                select(
                    strategy_definition.c.definition["family"].as_string(),
                    func.count(func.distinct(validation_stage.c.experiment_id)),
                )
                .select_from(stage_source)
                .where(public_stage_scope)
                .group_by(strategy_definition.c.definition["family"].as_string())
            ).all()
        )
        evaluated_count = int(
            connection.execute(
                select(func.count(func.distinct(validation_stage.c.experiment_id)))
                .select_from(stage_source)
                .where(public_stage_scope)
            ).scalar_one()
        )
        thesis_scope = (
            research_thesis.c.payload["generalisation_scope"]["product"].as_string() == product_id
        )
        thesis_count = int(
            connection.execute(
                select(func.count()).select_from(research_thesis).where(thesis_scope)
            ).scalar_one()
        )
        trial_count = int(
            connection.execute(
                select(func.count())
                .select_from(
                    thesis_trial.join(
                        research_thesis, thesis_trial.c.thesis_id == research_thesis.c.id
                    )
                )
                .where(thesis_scope)
            ).scalar_one()
        )
        identity_scope = strategy_identity.c.submitted_spec["product"].as_string() == product_id
        identity_count = int(
            connection.execute(
                select(func.count()).select_from(strategy_identity).where(identity_scope)
            ).scalar_one()
        )
        exact_duplicate_count = int(
            connection.execute(
                select(func.count())
                .select_from(strategy_identity)
                .where(identity_scope, strategy_identity.c.is_duplicate.is_(True))
            ).scalar_one()
        )
        near_duplicate_count = int(
            connection.execute(
                select(func.count())
                .select_from(strategy_identity)
                .where(
                    identity_scope,
                    strategy_identity.c.metadata["near_duplicate_of"].as_json().is_not(None),
                )
            ).scalar_one()
        )
        product_inputs = _product_input_metadata(connection, product_id, bundle)
        available_data = set(product_inputs["available_data"])
        supported_campaigns = _directional_campaigns(product_id, available_data)
        if product_inputs["hypothesis_inputs_ready"] and not supported_campaigns:
            product_inputs = {
                **product_inputs,
                "hypothesis_inputs_ready": False,
                "reason_code": "no_supported_directional_campaign_for_available_data",
            }
        strategy_catalogue = _strategy_catalogue(product_id, supported_campaigns)
        public_features = _public_feature_metadata(
            connection,
            product_id,
            candidates,
            strategy_catalogue["features_by_family"],
        )
        campaigns = _directional_campaigns(product_id, {"closed_ohlcv_bars"})
        campaign_names = tuple(campaign.name for campaign, _features in campaigns)
        feedback = _product_generation_feedback(connection, campaign_names)

    development_count = int(stage_counts.get("development", 0))
    development_rejected = int(rejected.get("development", 0))
    funnel = {
        "top_rejection_reasons": _safe_reason_counts(reasons),
        "candidates_rejected_by_stage": {
            str(stage): int(count) for stage, count in rejected.items()
        },
        "feature_family_concentration": {
            str(family or "unclassified"): int(count) for family, count in family_counts.items()
        },
        "candidates_generated": candidate_count,
        "candidates_evaluated": evaluated_count,
        "candidates_never_evaluated_count": max(0, candidate_count - evaluated_count),
        "theses_generated": thesis_count,
        "cumulative_trial_count": trial_count,
        "exact_duplicate_count": exact_duplicate_count,
        "near_duplicate_count": near_duplicate_count,
        "exact_duplicate_rate": exact_duplicate_count / max(1, identity_count),
        "near_duplicate_rate": near_duplicate_count / max(1, identity_count),
    }
    return {
        "product_id": product_id,
        "research": {
            "funnel": funnel,
            "product_inputs": product_inputs,
            "strategy_catalogue": strategy_catalogue,
            "feature_catalogue": {
                "supported_by_family": strategy_catalogue["features_by_family"],
                "publicly_observed_by_stage": public_features["by_stage"],
                "public_metadata_truncated": public_features["truncated"],
            },
            "development_results": {
                "stage_counts": {str(stage): int(count) for stage, count in stage_counts.items()},
                "development_evaluated": development_count,
                "development_rejected": development_rejected,
                "development_accepted": max(0, development_count - development_rejected),
            },
            "generation_feedback": feedback,
        },
    }


def _product_input_metadata(connection, product_id: str, bundle) -> dict[str, Any]:
    empty = {
        "hypothesis_inputs_ready": False,
        "canonical_bundle_available": False,
        "reason_code": "canonical_dataset_bundle_unavailable",
        "available_data": [],
        "available_public_stages": [],
        "instrument_universe": {
            "symbols": [],
            "instrument_ids": [],
            "venue": None,
            "market_type": None,
            "settlement_asset": None,
            "timeframe": None,
            "maximum_symbols_per_hypothesis": 1,
        },
    }
    if bundle is None:
        return empty

    raw_scope = bundle.instrument_scope
    scope = tuple(str(item) for item in raw_scope) if isinstance(raw_scope, list | tuple) else ()
    complete_scope = (
        bool(scope) and len(scope) <= _MAX_REVIEW_INSTRUMENTS and len(set(scope)) == len(scope)
    )
    instrument_rows = (
        connection.execute(
            select(
                instrument.c.id,
                instrument.c.venue,
                instrument.c.exchange_symbol,
                instrument.c.market_type,
                instrument.c.settlement_asset,
            )
            .where(instrument.c.id.in_(scope))
            .order_by(instrument.c.exchange_symbol)
            .limit(_MAX_REVIEW_INSTRUMENTS + 1)
        )
        .mappings()
        .all()
        if complete_scope
        else []
    )
    ids = [str(row["id"]) for row in instrument_rows]
    venues = {str(row["venue"]) for row in instrument_rows}
    symbols = [str(row["exchange_symbol"]) for row in instrument_rows]
    market_types = {str(row["market_type"]) for row in instrument_rows}
    settlements = {str(row["settlement_asset"] or "") for row in instrument_rows}
    product_scope_valid = (
        product_id == "btc_accumulation"
        and scope == ("binance:spot:BTCUSDT",)
        and symbols == ["BTCUSDT"]
        and venues == {"binance"}
        and market_types == {"spot"}
    ) or (
        product_id == "active_income"
        and complete_scope
        and set(ids) == set(scope)
        and venues == {"binance"}
        and market_types == {"futures"}
        and settlements == {"USDT"}
    )
    timeframe = str(bundle.timeframe or "")
    public_stages = [
        stage for stage in ("screening", "development") if stage in bundle.stage_snapshot_ids
    ]
    ready = bool(
        product_scope_valid
        and timeframe
        and len(instrument_rows) == len(scope)
        and public_stages == ["screening", "development"]
    )
    return {
        "hypothesis_inputs_ready": ready,
        "canonical_bundle_available": True,
        "reason_code": None if ready else "canonical_bundle_inputs_incomplete",
        "available_data": ["closed_ohlcv_bars"] if ready else [],
        "available_public_stages": public_stages,
        "instrument_universe": {
            "symbols": symbols if ready else [],
            "instrument_ids": ids if ready else [],
            "venue": next(iter(venues)) if len(venues) == 1 else None,
            "market_type": next(iter(market_types)) if len(market_types) == 1 else None,
            "settlement_asset": next(iter(settlements)) if len(settlements) == 1 else None,
            "timeframe": timeframe or None,
            "maximum_symbols_per_hypothesis": 1,
        },
        "timeframe": timeframe or None,
    }


def _directional_campaigns(product_id: str, available_data: set[str]):
    registry = FeatureGraphRegistry.default()
    campaigns = []
    for campaign in CAMPAIGNS:
        if (
            campaign.product != product_id
            or not campaign.enabled_by_default
            or campaign.family not in DIRECTIONAL_DSL_FAMILIES
            or campaign.expected_direction not in DIRECTIONAL_DSL_DIRECTIONS
            or not set(campaign.required_data).issubset(available_data)
        ):
            continue
        features = []
        for feature in campaign.permitted_features:
            try:
                registry.graph((feature,))
            except FeatureGraphError:
                continue
            features.append(feature)
        if campaign.feature in features:
            campaigns.append((campaign, tuple(features)))
    return tuple(campaigns)


def _strategy_catalogue(product_id: str, campaigns) -> dict[str, Any]:
    by_family: dict[str, dict[str, set[str]]] = {}
    features_by_family: dict[str, set[str]] = {}
    for campaign, features in campaigns:
        details = by_family.setdefault(
            campaign.family,
            {
                "mechanism_categories": set(),
                "required_data": set(),
                "expected_directions": set(),
                "expected_horizons": set(),
            },
        )
        details["mechanism_categories"].add(campaign.mechanism.value)
        details["required_data"].update(campaign.required_data)
        details["expected_directions"].add(campaign.expected_direction)
        details["expected_horizons"].add(campaign.expected_horizon)
        features_by_family.setdefault(campaign.family, set()).update(features)
    evidence_type = "btc_allocation" if product_id == "btc_accumulation" else "swing"
    return {
        "available": bool(campaigns),
        "unavailable_reason": None
        if campaigns
        else "no_supported_directional_campaign_for_available_data",
        "families": sorted(by_family),
        "mechanism_categories_by_family": {
            family: {key: sorted(values) for key, values in details.items()}
            for family, details in sorted(by_family.items())
        },
        "features_by_family": {
            family: sorted(features) for family, features in sorted(features_by_family.items())
        },
        "allowed_directions": sorted({campaign.expected_direction for campaign, _ in campaigns}),
        "allowed_operators": ["ge", "gt", "le", "lt"],
        "evidence_type": evidence_type,
        "thesis_fields": sorted(THESIS_FIELDS),
        "generalisation_scope_fields": ["family", "predeclared", "product"],
        "cumulative_trial_budget_min": 1,
        "cumulative_trial_budget_max": 50,
        "maximum_symbols_per_hypothesis": 1,
    }


def _public_feature_metadata(connection, product_id: str, candidates, features_by_family):
    if not features_by_family:
        return {
            "by_stage": {"screening": {}, "development": {}},
            "truncated": False,
        }
    stage_source = (
        validation_stage.join(candidates, validation_stage.c.experiment_id == candidates.c.id)
        .join(strategy_version, candidates.c.strategy_version_id == strategy_version.c.id)
        .join(strategy_definition, strategy_version.c.definition_id == strategy_definition.c.id)
    )
    diagnostic = or_(
        candidates.c.metadata["diagnostic"].as_boolean().is_(True),
        strategy_definition.c.definition["metadata"]["diagnostic"].as_boolean().is_(True),
    )
    required_nodes = strategy_definition.c.definition["feature_graph"]["required_nodes"].as_json()
    rows = (
        connection.execute(
            select(
                strategy_definition.c.id,
                strategy_definition.c.definition["family"].as_string().label("family"),
                validation_stage.c.stage,
                required_nodes.label("required_nodes"),
            )
            .select_from(stage_source)
            .where(
                strategy_definition.c.product_id == product_id,
                candidates.c.state != "legacy_import",
                ~diagnostic,
                validation_stage.c.stage.in_(("screening", "development")),
                validation_stage.c.engine_version == ECONOMIC_ENGINE_VERSION,
            )
            .distinct()
            .order_by(strategy_definition.c.id, validation_stage.c.stage)
            .limit(_MAX_PUBLIC_FEATURE_DEFINITIONS)
        )
        .mappings()
        .all()
    )
    by_stage: dict[str, dict[str, set[str]]] = {
        "screening": {},
        "development": {},
    }
    for row in rows:
        family = str(row["family"] or "")
        nodes = row["required_nodes"]
        if family not in features_by_family or not isinstance(nodes, list | tuple):
            continue
        permitted = set(features_by_family[family])
        by_stage[str(row["stage"])].setdefault(family, set()).update(
            str(node) for node in nodes if str(node) in permitted
        )
    return {
        "by_stage": {
            stage: {family: sorted(nodes) for family, nodes in sorted(families.items())}
            for stage, families in by_stage.items()
        },
        "truncated": len(rows) == _MAX_PUBLIC_FEATURE_DEFINITIONS,
    }


def _product_generation_feedback(connection, campaign_names: tuple[str, ...]):
    if not campaign_names:
        return {}
    campaign = generation_feedback.c.payload["campaign"].as_string()
    outcome = generation_feedback.c.payload["outcome"].as_string()
    rows = connection.execute(
        select(campaign, outcome, func.count())
        .where(campaign.in_(campaign_names))
        .group_by(campaign, outcome)
        .order_by(campaign, outcome)
    ).all()
    feedback: dict[str, dict[str, int]] = {}
    for name, result, count in rows:
        if name and result:
            feedback.setdefault(str(name), {})[str(result)] = int(count)
    return feedback


def _safe_reason_counts(reasons) -> dict[str, int]:
    blocked_markers = (
        "secret",
        "credential",
        "password",
        "api_key",
        "api_secret",
        "protected",
        "holdout",
    )
    return {
        str(reason): int(count)
        for reason, count in reasons.items()
        if reason and not any(marker in str(reason).lower() for marker in blocked_markers)
    }
