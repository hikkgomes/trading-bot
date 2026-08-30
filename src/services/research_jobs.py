"""Concrete handlers for the unified leased research queue."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from sqlalchemy import select

from src.data.database import validation_stage
from src.data.parquet_store import PartitionedBacktestStore
from src.domain._codec import canonical_hash
from src.domain.strategies import StrategySourceType
from src.research.artefacts import StrategyArtefact
from src.research.backtest.bar_engine import BarPortfolioEngine, BarStep
from src.research.backtest.event_engine import (
    EventReplayEngine,
    ReplayEvent,
    SimulatedLimitOrder,
    SimulatedOrderSide,
)
from src.research.canonical import SqlStrategyArtefactRepository
from src.research.catalogue import registered_strategy_candidates, registered_strategy_theses
from src.research.coordinator import Candidate, CandidateEvaluationView, ResearchCoordinator
from src.research.datasets import (
    CandidateDatasetPlan,
    CanonicalDatasetResolver,
    DatasetLifecycleState,
    SqlDatasetBundleRepository,
)
from src.research.evaluation import (
    CanonicalResearchEvaluator,
    EvaluationRequest,
    EvidencePolicy,
    ProtectedHoldoutWorker,
)
from src.research.executors import (
    ExecutorError,
    ProviderContextBuilderRegistry,
    ProviderExecutorRegistry,
)
from src.research.generation import (
    CAMPAIGNS,
    GenerationFeedback,
    HypothesisGenerator,
    SqlGenerationFeedbackStore,
    SqlHypothesisMemory,
)
from src.research.ml import MlExperimentRunner
from src.research.providers import provider_candidate
from src.research.store import SqlResearchStore
from src.research.theses import SqlThesisRegistry, StrategyThesisFactory, ThesisError
from src.services.job_schemas import JobSchemaError, ResearchJobRequest
from src.services.scheduler import ClaimedJob

_REQUIRED_EVIDENCE: dict[str, frozenset[str]] = {
    "screening": frozenset(
        {"compiled", "features_valid", "causality_valid", "signal_frequency", "turnover"}
    ),
    "development": frozenset(
        {
            "chronological",
            "cost_adjusted_return",
            "regime_breakdown",
            "parameter_stability",
            "sample_evidence",
            "cross_symbol_stability",
            "portfolio_overlap",
        }
    ),
    "robustness": frozenset(
        {
            "walk_forward",
            "purged",
            "embargo",
            "cost_stress",
            "delay_stress",
            "missing_data_stress",
            "funding_stress",
            "monte_carlo_trade_order",
            "bootstrap_confidence",
            "probability_backtest_overfitting",
            "deflated_sharpe",
            "drawdown_stability",
        }
    ),
    "protected": frozenset({"frozen_cohort", "holdout_claim", "data_hashes", "code_hash"}),
}

_EVIDENCE_UNITS: dict[str, frozenset[str]] = {
    "scalping": frozenset({"independent_trades", "event_windows"}),
    "intraday": frozenset({"trades", "trading_days"}),
    "swing": frozenset({"trades", "months", "regimes"}),
    "btc_allocation": frozenset({"exposure_days", "market_regimes"}),
    "cross_sectional": frozenset({"rebalance_dates", "portfolio_returns"}),
    "funding_carry": frozenset({"funding_intervals"}),
    "pairs": frozenset({"independent_spread_excursions"}),
    "market_making": frozenset({"orders", "fills", "inventory_cycles"}),
    "ml": frozenset({"purged_prediction_windows", "calibrated_probability"}),
}


class DatabaseResearchJobHandlers:
    def __init__(
        self,
        store: SqlResearchStore,
        *,
        artefact_store: PartitionedBacktestStore | None = None,
        ml_runner: MlExperimentRunner | None = None,
        dataset_loader: Callable[[str, Mapping[str, Any]], Any] | None = None,
        dataset_resolver: CanonicalDatasetResolver | None = None,
        executors: ProviderExecutorRegistry | None = None,
        context_builders: ProviderContextBuilderRegistry | None = None,
        evidence_policy: EvidencePolicy | None = None,
        configuration: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self.store = store
        self.artefact_store = artefact_store
        self.ml_runner = ml_runner
        self.dataset_loader = dataset_loader
        self.dataset_resolver = dataset_resolver
        self.executors = executors or ProviderExecutorRegistry.default()
        self.context_builders = context_builders or ProviderContextBuilderRegistry.default()
        self.evidence_policy = evidence_policy or EvidencePolicy()
        self.configuration = configuration

    def handlers(self) -> dict[str, Callable]:
        return {
            "dataset_snapshot_validate": self.validate_dataset_snapshot,
            "register_strategy_catalogue": self.register_strategy_catalogue,
            "generate_hypotheses": self.generate_hypotheses,
            "register_candidate": self.register_candidate,
            "register_ml_candidate": self.register_ml_candidate,
            "evaluate_candidate": self.evaluate_candidate,
            "bounded_backtest": self.bounded_backtest,
            "event_replay": self.event_replay,
            "train_ml_experiment": self.train_ml_experiment,
        }

    def validate_dataset_snapshot(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        if self.dataset_resolver is None:
            raise JobSchemaError("dataset validation requires a canonical dataset resolver")
        required = {
            "dataset_snapshot_id",
            "product_id",
            "feature_manifest_id",
            "cost_model_id",
            "parameter_set_id",
            "producer_identity",
        }
        if set(claimed.payload) != required:
            raise JobSchemaError("dataset validation command has an invalid field set")
        renew()
        resolved = self.dataset_resolver.resolve(
            str(claimed.payload["dataset_snapshot_id"]),
            expected={
                "product_id": str(claimed.payload["product_id"]),
                "feature_manifest_hash": str(claimed.payload["feature_manifest_id"]),
                "cost_model_hash": str(claimed.payload["cost_model_id"]),
                "parameter_set_hash": str(claimed.payload["parameter_set_id"]),
            },
        )
        return {
            "dataset_snapshot_id": resolved.snapshot_id,
            "dataset_identity_hash": resolved.receipt["identity_hash"],
        }

    def register_strategy_catalogue(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        renew()
        universe = tuple(claimed.payload.get("instrument_universe", ("BTCUSDT",)))
        theses = registered_strategy_theses(
            product=str(claimed.payload["product_id"]), instrument_universe=universe
        )
        thesis_registry = SqlThesisRegistry(self.store.engine)
        for thesis in theses.values():
            thesis_registry.register(thesis)
        candidates = registered_strategy_candidates(
            product=str(claimed.payload["product_id"]),
            dataset_snapshot_hashes=tuple(claimed.payload["dataset_snapshot_hashes"]),
            dataset_bundle_id=(
                str(claimed.payload["dataset_bundle_id"])
                if claimed.payload.get("dataset_bundle_id")
                else None
            ),
            universe_snapshot_id=(
                str(claimed.payload["universe_snapshot_id"])
                if claimed.payload.get("universe_snapshot_id")
                else None
            ),
            instrument_universe=universe,
            submitted_at=(
                str(claimed.payload["catalogue_submitted_at"])
                if claimed.payload.get("catalogue_submitted_at")
                else None
            ),
        )
        identities = ResearchCoordinator(self.store).register(candidates)
        return {"registered_candidates": len(identities), "candidate_ids": list(identities)}

    def register_candidate(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        renew()
        candidate = self._candidate(claimed.payload)
        candidate_id = ResearchCoordinator(self.store).submit(candidate)
        return {"candidate_id": candidate_id, "source_type": candidate.definition.source_type.value}

    def generate_hypotheses(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        """Compile safe campaigns into the same candidate queue as all providers."""

        payload = claimed.payload
        required = {
            "product_id",
            "instrument_universe",
            "dataset_snapshot_hashes",
            "submitted_at",
            "generation_budget",
        }
        if not required.issubset(payload):
            raise JobSchemaError("hypothesis generation command is incomplete")
        product_id = str(payload["product_id"])
        universe = tuple(str(item) for item in payload["instrument_universe"])
        snapshot_ids = tuple(str(item) for item in payload["dataset_snapshot_hashes"])
        feedback_store = SqlGenerationFeedbackStore(self.store.engine)
        if not snapshot_ids:
            for campaign in CAMPAIGNS:
                if campaign.product == product_id:
                    feedback_store.append(
                        GenerationFeedback(
                            campaign=campaign.name,
                            outcome="data_unavailable",
                            observed_at=str(payload["submitted_at"]),
                            reason_code="canonical_dataset_bundle_unavailable",
                        )
                    )
            return {"product_id": product_id, "state": "waiting_for_dataset"}
        generator = HypothesisGenerator(
            product=product_id,
            instrument_universe=universe,
            memory=SqlHypothesisMemory(self.store.engine),
            feedback_store=feedback_store,
        )
        hypotheses = generator.generate(
            dataset_snapshot_hashes=snapshot_ids,
            submitted_at=str(payload["submitted_at"]),
            total_budget=int(payload["generation_budget"]),
            dataset_bundle_id=(
                str(payload["dataset_bundle_id"]) if payload.get("dataset_bundle_id") else None
            ),
            universe_snapshot_id=(
                str(payload["universe_snapshot_id"])
                if payload.get("universe_snapshot_id")
                else None
            ),
        )
        bundle_id = payload.get("dataset_bundle_id")
        plan = None
        if bundle_id:
            bundle = SqlDatasetBundleRepository(self.store.engine).get(str(bundle_id))
            if bundle.lifecycle_state is not DatasetLifecycleState.READY:
                raise JobSchemaError("hypothesis generation requires a ready dataset bundle")
            plan = CandidateDatasetPlan.from_bundle(bundle)
        thesis_registry = SqlThesisRegistry(self.store.engine)
        coordinator = ResearchCoordinator(self.store)
        registered: list[str] = []
        for hypothesis in hypotheses:
            renew()
            candidate = hypothesis.candidate
            if plan is not None:
                if plan.product_id != candidate.definition.product:
                    raise JobSchemaError("generated candidate and dataset bundle products differ")
                candidate = replace(
                    candidate,
                    dataset_snapshot_hashes=plan.all_snapshot_ids,
                    dataset_bundle_id=bundle.bundle_id,
                    dataset_plan=plan,
                )
                hypothesis = replace(hypothesis, candidate=candidate)
            try:
                thesis_registry.register(hypothesis.thesis)
                candidate_id = coordinator.submit(hypothesis.candidate)
            except ThesisError as exc:
                feedback_store.append(
                    GenerationFeedback(
                        campaign=hypothesis.campaign.name,
                        outcome="resource_budget_exhausted",
                        observed_at=str(payload["submitted_at"]),
                        reason_code=str(exc),
                        semantic_signature=hypothesis.semantic_signature,
                    )
                )
                continue
            feedback_store.append(
                GenerationFeedback(
                    campaign=hypothesis.campaign.name,
                    outcome="accepted",
                    observed_at=str(payload["submitted_at"]),
                    candidate_id=candidate_id,
                    semantic_signature=hypothesis.semantic_signature,
                )
            )
            registered.append(candidate_id)
        return {
            "product_id": product_id,
            "state": "generated",
            "candidate_ids": registered,
            "candidate_count": len(registered),
        }

    def register_ml_candidate(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        candidate = self._candidate(claimed.payload)
        if candidate.definition.source_type is not StrategySourceType.MACHINE_LEARNING:
            raise ValueError("ML worker accepts only machine-learning candidates")
        renew()
        universe = candidate.definition.universe.get("symbols", ())
        if not isinstance(universe, list | tuple) or not universe:
            raise ValueError("ML candidate requires an explicit symbol universe")
        thesis = StrategyThesisFactory.default().build(
            name="frozen_logistic_model",
            family="machine_learning",
            product=candidate.definition.product,
            instrument_universe=tuple(str(item) for item in universe),
        )
        if thesis.thesis_id != candidate.thesis_id:
            raise ValueError("ML candidate thesis identity does not match its definition")
        SqlThesisRegistry(self.store.engine).register(thesis)
        candidate_id = ResearchCoordinator(self.store).submit(candidate)
        return {"candidate_id": candidate_id, "source_type": candidate.definition.source_type.value}

    def evaluate_candidate(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        renew()
        ResearchJobRequest.from_mapping(claimed.payload, require_dataset_roles=True)
        request = EvaluationRequest.from_payload(claimed.payload)
        if self.dataset_resolver is None:
            raise JobSchemaError("candidate evaluation requires a canonical dataset resolver")
        resolver = self.dataset_resolver
        candidate = self.store.get_candidate(request.candidate_id)
        adaptive_snapshot_ids = request.snapshot_ids_for_stage(request.requested_stage)
        if request.requested_stage == "protected":
            adaptive_snapshot_ids = tuple(
                snapshot_id
                for snapshot_id in request.dataset_snapshot_ids
                if snapshot_id
                != (
                    request.protected_snapshot_id()
                    if request.dataset_roles
                    else request.dataset_snapshot_ids[0]
                )
            )
        if adaptive_snapshot_ids:
            resolved_context = resolver.resolve_context(
                snapshot_ids=adaptive_snapshot_ids,
                feature_manifest_id=str(request.feature_manifest_id),
                cost_model_id=str(request.cost_model_id),
                parameter_set_id=str(request.parameter_set_id),
                allowed_roles=(
                    frozenset(
                        {
                            request.dataset_roles[snapshot_id]
                            for snapshot_id in adaptive_snapshot_ids
                        }
                    )
                    if request.dataset_roles
                    else None
                ),
                minimum_availability_timestamp=(
                    str(claimed.payload["artefact_created_at"])
                    if request.requested_stage == "forward"
                    else None
                ),
                maximum_availability_timestamp=(
                    request.evaluated_at if request.requested_stage == "forward" else None
                ),
            )
            try:
                context = self.context_builders.build(
                    CandidateEvaluationView.from_candidate(candidate, adaptive_snapshot_ids),
                    resolved_context,
                )
            except (ExecutorError, KeyError, TypeError, ValueError) as exc:
                context = {
                    **dict(resolved_context),
                    "provider_context_error": f"{type(exc).__name__}: {exc}",
                }
        else:
            context = {
                "candidate_id": request.candidate_id,
                "dataset_snapshot_ids": [],
                "feature_manifest_id": str(request.feature_manifest_id),
                "cost_model_id": str(request.cost_model_id),
                "parameter_set_id": str(request.parameter_set_id),
            }
        context = {
            **dict(context),
            "artefact_hash": request.artefact_hash,
            "artefact_created_at": request.artefact_created_at,
        }

        def evaluate_protected(claim: Mapping[str, Any]) -> tuple[bool, Mapping[str, Any]]:
            protected_context = claim.get("protected_context")
            if not isinstance(protected_context, Mapping):
                raise JobSchemaError("protected holdout worker did not resolve its dataset")
            frozen_context = self.context_builders.build(candidate, protected_context)
            execution = self.executors.execute(
                candidate,
                {
                    **frozen_context,
                    "product_id": candidate.definition.product,
                    "requested_stage": "protected",
                    "evaluated_at": request.evaluated_at,
                    "evidence_policy_hash": self.evidence_policy.policy_hash,
                    "minimum_bootstrap_observations": self.evidence_policy.minimum_bootstrap_observations,
                    "bootstrap_method": self.evidence_policy.bootstrap_method,
                    "multiple_testing_method": self.evidence_policy.multiple_testing_method,
                    "pbo_method": self.evidence_policy.pbo_method,
                },
            )
            measured = dict(execution.evidence)
            requires_objective = (
                candidate.definition.product in {"btc_accumulation", "active_income"}
                and candidate.definition.metadata.get("diagnostic") is not True
                and candidate.definition.metadata.get("promotable") is not False
                and (
                    candidate.definition.metadata.get("promotable") is True
                    or candidate.definition.metadata.get("executable_registry_entry") is True
                )
            )
            accepted = self.evidence_policy.accepts(
                "development",
                measured,
                (),
                product_id=candidate.definition.product if requires_objective else None,
            )
            sealed = {
                "passed": accepted,
                "evidence_hash": canonical_hash(measured),
                "metrics": dict(execution.metrics),
                "execution_receipt": dict(execution.receipt),
            }
            return accepted, {
                "passed": accepted,
                "sealed_result": sealed,
            }

        result = CanonicalResearchEvaluator(
            self.store,
            executors=self.executors,
            provider_context=context,
            protected_worker=ProtectedHoldoutWorker(
                self.store.engine,
                evaluate_protected,
                dataset_resolver=resolver,
                feature_manifest_id=str(request.feature_manifest_id),
                cost_model_id=str(request.cost_model_id),
                parameter_set_id=str(request.parameter_set_id),
            ),
            evidence_policy=self.evidence_policy,
        ).evaluate(request)
        response = {
            "candidate_id": result.candidate_id,
            "stage": result.stage,
            "accepted": result.accepted,
            "reason_code": result.reason_code,
            "run_id": result.run_id,
            "evidence_hash": result.evidence_hash,
        }
        if result.stage == "protected" and result.accepted:
            response["artefact_hash"] = self._publish_strategy_artefact(
                candidate=candidate,
                request=request,
                evidence=result.evidence,
            )
        return response

    def _publish_strategy_artefact(
        self,
        *,
        candidate: Candidate,
        request: EvaluationRequest,
        evidence: Mapping[str, Any],
    ) -> str:
        if self.configuration is None or self.dataset_resolver is None:
            raise JobSchemaError("accepted protected research requires platform configuration")
        if (
            candidate.metadata.get("diagnostic") is True
            or candidate.definition.metadata.get("diagnostic") is True
            or candidate.metadata.get("promotable") is False
            or candidate.definition.metadata.get("promotable") is False
        ):
            raise JobSchemaError("diagnostic research cannot create a promotable artefact")

        products = {
            str(item["product_id"]): dict(item)
            for item in self.configuration["products"]["products"]
        }
        policies = {
            str(item["policy_id"]): dict(item)
            for item in self.configuration["promotion"]["policies"]
        }
        product = products[candidate.definition.product]
        policy_id = str(product["promotion_policy_id"])
        with self.store.engine.connect() as connection:
            stages = connection.execute(
                select(
                    validation_stage.c.id,
                    validation_stage.c.stage,
                    validation_stage.c.payload,
                )
                .where(
                    validation_stage.c.experiment_id == candidate.candidate_id,
                    validation_stage.c.accepted.is_(True),
                    validation_stage.c.stage.in_(
                        ("screening", "development", "robustness", "protected")
                    ),
                )
                .order_by(validation_stage.c.evaluated_at, validation_stage.c.stage)
            ).all()
        stages_by_name = {str(row.stage): str(row.id) for row in stages}
        required_stages = ("screening", "development", "robustness", "protected")
        if set(stages_by_name) != set(required_stages):
            raise JobSchemaError("promotable artefact requires every accepted research stage")
        snapshot_ids = list(request.dataset_snapshot_ids)
        for row in stages:
            payload = row.payload if isinstance(row.payload, Mapping) else {}
            stage_evidence = payload.get("evidence")
            context = stage_evidence.get("context") if isinstance(stage_evidence, Mapping) else None
            if isinstance(context, Mapping):
                snapshot_ids.extend(str(value) for value in context.get("dataset_snapshot_ids", ()))
        dataset_snapshot_hashes = tuple(dict.fromkeys(snapshot_ids))
        resolved = tuple(
            self.dataset_resolver.resolve(
                snapshot_id,
                expected={
                    "feature_manifest_hash": str(request.feature_manifest_id),
                    "cost_model_hash": str(request.cost_model_id),
                    "parameter_set_hash": str(request.parameter_set_id),
                    "product_id": candidate.definition.product,
                },
            )
            for snapshot_id in dataset_snapshot_hashes
        )
        raw_claims = evidence.get("holdout_claim")
        if not isinstance(raw_claims, list) or len(raw_claims) != 1:
            raise JobSchemaError("promotable artefact requires one protected holdout claim")

        dependency_hash = canonical_hash(
            {
                "evaluator_version": request.evaluator_version,
                "producer_identity": request.producer_identity,
                "feature_manifest_id": request.feature_manifest_id,
                "cost_model_id": request.cost_model_id,
                "parameter_set_id": request.parameter_set_id,
            }
        )
        artefact = StrategyArtefact.from_authoritative_evidence(
            definition=candidate.definition,
            dependency_hash=dependency_hash,
            dependency_lock_hash=dependency_hash,
            source_commit_hash=candidate.definition.source_hash,
            dataset_snapshot_hashes=dataset_snapshot_hashes,
            feature_set_version=str(request.feature_manifest_id),
            feature_set_hash=str(request.feature_manifest_id),
            cost_model_version=str(request.cost_model_id),
            cost_model_hash=str(request.cost_model_id),
            validation_stage_ids=tuple(stages_by_name[name] for name in required_stages),
            holdout_claim_id=str(raw_claims[0]),
            promotion_policy=policies[policy_id],
            position_limits=dict(self.configuration["risk"]["strategy"]),
            risk_limits={
                "global": dict(self.configuration["risk"]["global"]),
                "account": dict(self.configuration["risk"]["accounts"][str(product["account_id"])]),
                "product": dict(
                    self.configuration["risk"]["products"][str(product["risk_policy_id"])]
                ),
            },
            model_hashes=tuple(
                dict.fromkeys(
                    item.model_artefact_id
                    for item in resolved
                    if item.model_artefact_id is not None
                )
            ),
            supported_products=(candidate.definition.product,),
            supported_instruments=tuple(
                dict.fromkeys(symbol for item in resolved for symbol in item.instrument_scope)
            ),
            created_at=request.evaluated_at,
            validation_evidence={
                "stage_ids": [stages_by_name[name] for name in required_stages],
                "protected_evidence_hash": canonical_hash(dict(evidence)),
            },
            holdout_claim={
                "claim_id": str(raw_claims[0]),
                "outcome_id": evidence.get("holdout_outcome_id"),
            },
            metadata={
                "candidate_id": candidate.candidate_id,
                "thesis_id": candidate.thesis_id,
                "lineage_id": candidate.lineage_id,
                "diagnostic": False,
                "promotable": True,
            },
            product_id=candidate.definition.product,
            portfolio_id=str(product["portfolio_id"]),
            account_id=str(product["account_id"]),
            promotion_policy_id=policy_id,
            engine_version=request.evaluator_version,
        )
        return SqlStrategyArtefactRepository(self.store.engine).put(
            artefact.artefact_hash,
            artefact.to_dict(),
            created_at=request.evaluated_at,
        )

    def bounded_backtest(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        renew()
        request = _assert_result_free_research_request(
            claimed.payload, forbidden=("steps", "returns", "metrics")
        )
        loaded = self._load_dataset("bar_steps", request)
        if not isinstance(loaded, Mapping):
            raise JobSchemaError("bar dataset loader must return a mapping")
        steps = tuple(
            item if isinstance(item, BarStep) else BarStep(**dict(item)) for item in loaded["steps"]
        )
        result = BarPortfolioEngine(
            initial_equity=float(loaded["initial_equity"]),
            fee_bps=float(loaded["fee_bps"]),
        ).simulate(steps)
        final_equity = result.equity_curve[-1][1] if result.equity_curve else 0.0
        evidence: dict[str, Any] = {
            "equity_curve": list(result.equity_curve),
            "quantities": dict(result.quantities),
        }
        if self.artefact_store is not None:
            path, content_hash = self.artefact_store.put_rows(
                candidate_id=str(claimed.payload["candidate_id"]),
                run_name="bar_portfolio",
                created_at=str(claimed.payload["evaluated_at"]),
                rows=[
                    {"timestamp": timestamp_value, "equity": equity}
                    for timestamp_value, equity in result.equity_curve
                ],
            )
            evidence["result_artefact"] = {
                "content_hash": content_hash,
                "path": str(path.relative_to(self.artefact_store.root)),
            }
        run_id = self.store.save_run(
            candidate_id=str(claimed.payload["candidate_id"]),
            run_name="bar_portfolio",
            created_at=str(claimed.payload["evaluated_at"]),
            evidence=evidence,
            metrics={
                "final_equity": final_equity,
                "fees_paid": result.fees_paid,
                "funding_paid": result.funding_paid,
            },
            receipt=_execution_receipt(request, executor_version="bar-engine/v2"),
        )
        return {"run_id": run_id, "final_equity": final_equity}

    def event_replay(self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]) -> dict[str, Any]:
        renew()
        request = _assert_result_free_research_request(
            claimed.payload,
            forbidden=("events", "orders", "fills", "positions", "returns", "metrics"),
        )
        loaded = self._load_dataset("event_replay", request)
        if not isinstance(loaded, Mapping):
            raise JobSchemaError("event replay dataset loader must return events and orders")
        events = tuple(
            item if isinstance(item, ReplayEvent) else ReplayEvent(**dict(item))
            for item in loaded["events"]
        )
        orders = tuple(
            SimulatedLimitOrder(
                **{
                    **dict(item),
                    "side": SimulatedOrderSide(str(item["side"])),
                }
            )
            for item in loaded["orders"]
        )
        result = EventReplayEngine(
            cancel_latency_seconds=float(loaded["cancel_latency_seconds"]),
            impact_bps_per_depth_fraction=float(loaded["impact_bps_per_depth_fraction"]),
        ).simulate(events=events, orders=orders)
        order_evidence = [
            {
                "order_id": item.order.order_id,
                "status": item.status.value,
                "remaining_quantity": item.remaining_quantity,
                "reason_code": item.reason_code,
            }
            for item in result.orders
        ]
        evidence: dict[str, object] = {
            "orders": order_evidence,
            "positions": dict(result.positions),
            "connection_gaps": list(result.connection_gaps),
        }
        if self.artefact_store is not None:
            path, content_hash = self.artefact_store.put_rows(
                candidate_id=str(claimed.payload["candidate_id"]),
                run_name="event_replay",
                created_at=str(claimed.payload["evaluated_at"]),
                rows=[
                    {
                        "order_id": order.order.order_id,
                        "fill_id": fill.order_id + ":" + str(index),
                        "instrument_id": fill.instrument_id,
                        "quantity": fill.quantity,
                        "price": fill.price,
                        "occurred_at": fill.occurred_at,
                        "spread_cost": fill.spread_cost,
                        "market_impact": fill.market_impact,
                        "adverse_selection": fill.adverse_selection,
                    }
                    for order in result.orders
                    for index, fill in enumerate(order.fills)
                ],
            )
            evidence["result_artefact"] = {
                "content_hash": content_hash,
                "path": str(path.relative_to(self.artefact_store.root)),
            }
        run_id = self.store.save_run(
            candidate_id=str(claimed.payload["candidate_id"]),
            run_name="event_replay",
            created_at=str(claimed.payload["evaluated_at"]),
            evidence=evidence,
            metrics={
                **{str(key): float(value) for key, value in result.metrics.items()},
                "funding_paid": result.funding_paid,
            },
            receipt=_execution_receipt(request, executor_version="event-replay/v2"),
        )
        return {"run_id": run_id, **dict(result.metrics)}

    def train_ml_experiment(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        if self.ml_runner is None:
            raise RuntimeError("ML experiment runner is not configured")
        renew()
        payload = claimed.payload
        request = _assert_result_free_research_request(
            payload, forbidden=("rows", "metrics", "returns", "model_artifact_id")
        )
        loaded = self._load_dataset("ml_rows", request)
        if not isinstance(loaded, Mapping):
            raise JobSchemaError("ML dataset loader must return a mapping")
        rows = loaded["rows"]
        result = self.ml_runner.run(
            candidate_id=str(payload["candidate_id"]),
            model_name=str(loaded["model_name"]),
            feature_names=tuple(loaded["feature_names"]),
            target_name=str(loaded["target_name"]),
            rows=tuple(rows),
            created_at=request.evaluated_at,
            train_fraction=float(loaded["train_fraction"]),
            embargo_rows=int(loaded["embargo_rows"]),
            hyperparameters=loaded.get("hyperparameters"),
        )
        run_id = self.store.save_run(
            candidate_id=str(payload["candidate_id"]),
            run_name=f"ml:{loaded['model_name']}",
            created_at=request.evaluated_at,
            evidence={
                "model_artifact_id": result.model_artifact_id,
                "content_hash": result.content_hash,
                "relative_path": result.relative_path,
                "dataset_hash": result.dataset_hash,
                "train_rows": result.train_rows,
                "validation_rows": result.validation_rows,
                "chronological": result.train_rows < len(tuple(rows)),
                "embargo_rows": int(loaded["embargo_rows"]),
            },
            metrics={str(key): float(value) for key, value in result.metrics.items()},
            receipt=_execution_receipt(request, executor_version="ml-runner/v2"),
        )
        return {
            "run_id": run_id,
            "model_artifact_id": result.model_artifact_id,
            "content_hash": result.content_hash,
            "metrics": dict(result.metrics),
        }

    def _load_dataset(self, kind: str, request: ResearchJobRequest) -> Any:
        if self.dataset_loader is None:
            raise JobSchemaError(
                f"{kind} requires a canonical dataset loader; queue payloads cannot contain raw data"
            )
        return self.dataset_loader(kind, request.to_payload())

    @staticmethod
    def _candidate(payload: Mapping[str, Any]) -> Candidate:
        return provider_candidate(
            identity=str(payload["identity"]),
            version=str(payload["version"]),
            family=str(payload["family"]),
            product=str(payload["product"]),
            thesis_id=str(payload["thesis_id"]),
            lineage_id=str(payload["lineage_id"]),
            provider=str(payload["provider"]),
            source_type=StrategySourceType(str(payload["source_type"])),
            source_payload=dict(payload["source_payload"]),
            dataset_snapshot_hashes=tuple(payload["dataset_snapshot_hashes"]),
            submitted_at=str(payload["submitted_at"]),
            universe=payload.get("universe"),
            data_requirements=payload.get("data_requirements"),
            feature_graph=payload.get("feature_graph"),
            position_model=payload.get("position_model"),
            execution_preferences=payload.get("execution_preferences"),
            risk_policy=payload.get("risk_policy"),
            validation_policy=payload.get("validation_policy"),
            metadata=payload.get("metadata"),
        )


def _validate_sample_evidence(candidate: Candidate, raw: object) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("development sample_evidence must be an object")
    evidence_type = str(
        candidate.definition.validation_policy.get("evidence_type")
        or raw.get("evidence_type")
        or ""
    )
    required = _EVIDENCE_UNITS.get(evidence_type)
    if required is None:
        raise ValueError(f"unsupported strategy evidence type: {evidence_type or 'missing'}")
    units = raw.get("units")
    if not isinstance(units, Mapping):
        raise ValueError("development sample_evidence.units must be an object")
    missing = required - set(units)
    if missing:
        raise ValueError(f"strategy evidence lacks units: {sorted(missing)}")
    for name in required:
        value = units[name]
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"strategy evidence unit {name} must be positive")


def _assert_result_free_research_request(
    payload: Mapping[str, Any], *, forbidden: tuple[str, ...]
) -> ResearchJobRequest:
    """Reject legacy jobs that smuggled a simulation into the queue payload."""

    present = sorted(set(payload) & set(forbidden))
    if present:
        raise JobSchemaError(
            "research jobs cannot contain precomputed results: " + ", ".join(present)
        )
    return ResearchJobRequest.from_mapping(payload, require_dataset_roles=True)


def _execution_receipt(request: ResearchJobRequest, *, executor_version: str) -> dict[str, Any]:
    payload = {
        "candidate_id": request.candidate_id,
        "dataset_snapshot_ids": list(request.dataset_snapshot_ids),
        "feature_manifest_id": request.feature_manifest_id,
        "cost_model_id": request.cost_model_id,
        "parameter_set_id": request.parameter_set_id,
        "evaluator_version": request.evaluator_version,
        "executor_version": executor_version,
        "artefact_hash": request.artefact_hash,
        "artefact_created_at": request.artefact_created_at,
    }
    return {**payload, "input_hash": canonical_hash(payload)}
