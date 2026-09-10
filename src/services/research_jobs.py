"""Concrete handlers for the unified leased research queue."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from sqlalchemy import select

from src.data.database import paper_eligibility_event, validation_stage
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
from src.research.canonical import SqlStrategyArtefactRepository, SqlValidationRepository
from src.research.catalogue import registered_strategy_candidates, registered_strategy_theses
from src.research.cohorts import CohortPending, complete_cohort, prepare_cohort, selection_stream
from src.research.coordinator import Candidate, CandidateEvaluationView, ResearchCoordinator
from src.research.dataset_service import DatabaseDatasetBundleService
from src.research.datasets import (
    CandidateDatasetPlan,
    CanonicalDatasetResolver,
    DatasetBundle,
    DatasetLifecycleState,
    SqlDatasetBundleRepository,
)
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.evaluation import (
    CanonicalResearchEvaluator,
    EvaluationRequest,
    EvidencePolicy,
    ProtectedHoldoutWorker,
)
from src.research.evidence import holdout_degradation_passes
from src.research.executors import (
    DataUnavailableError,
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
    hypothesis_signature,
)
from src.research.ml import MlExperimentRunner
from src.research.providers import provider_candidate
from src.research.store import SqlResearchStore
from src.research.theses import SqlThesisRegistry, StrategyThesisFactory, ThesisError
from src.services.historical_bootstrap import HistoricalBootstrapService
from src.services.job_schemas import JobSchemaError, ResearchJobRequest
from src.services.runtime import utc_now
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
            "monte_carlo_return_blocks",
            "bootstrap_confidence",
            "probability_backtest_overfitting",
            "deflated_sharpe",
            "drawdown_stability",
        }
    ),
    "protected": frozenset({"frozen_cohort", "holdout_claim", "data_hashes", "code_hash"}),
}

_EPOCH_TIMESTAMP = "1970-01-01T00:00:00+00:00"


def _runtime_data_timestamp(payload: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = payload.get(field)
        if value is None:
            continue
        candidate = str(value)
        if candidate and candidate != _EPOCH_TIMESTAMP:
            return candidate
    return utc_now()


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


def _configured_product(
    configuration: Mapping[str, Any], product_id: str
) -> Mapping[str, Any] | None:
    products = configuration.get("products")
    if not isinstance(products, Mapping):
        return None
    return next(
        (
            item
            for item in products.get("products", ())
            if isinstance(item, Mapping) and str(item.get("product_id")) == product_id
        ),
        None,
    )


def _configured_account(
    configuration: Mapping[str, Any], account_id: str
) -> Mapping[str, Any] | None:
    accounts = configuration.get("accounts")
    if not isinstance(accounts, Mapping):
        return None
    return next(
        (
            item
            for item in accounts.get("accounts", ())
            if isinstance(item, Mapping) and str(item.get("account_id")) == account_id
        ),
        None,
    )


def _cost_context(product: Mapping[str, Any]) -> dict[str, Any]:
    costs = product.get("execution_costs")
    if not isinstance(costs, Mapping):
        return {}
    return {"fee_bps": costs.get("fee_bps"), "slippage_bps": costs.get("slippage_bps")}


def _btc_accounting_context(product: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "btc_core_fraction": product.get("btc_core_fraction"),
        "btc_minimum_fraction": product.get("btc_minimum_fraction"),
        "btc_max_tactical_fraction": product.get("btc_max_tactical_fraction"),
    }


def _active_income_accounting_context(
    configuration: Mapping[str, Any], account: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "leverage": account.get("maximum_leverage"),
        "margin_mode": account.get("margin_mode"),
        "maintenance_margin_fraction": 0.05,
    }
    risk = configuration.get("risk")
    if not isinstance(risk, Mapping):
        return result
    account_limits = risk.get("accounts")
    limits = (
        account_limits.get(str(account.get("account_id") or ""))
        if isinstance(account_limits, Mapping)
        else None
    )
    if isinstance(limits, Mapping):
        result.update(
            {
                "max_margin_fraction": limits.get("maximum_margin_fraction"),
                "liquidation_buffer_fraction": limits.get("minimum_liquidation_buffer"),
            }
        )
    instrument_limits = risk.get("instrument")
    if isinstance(instrument_limits, Mapping):
        result["max_participation_fraction"] = instrument_limits.get(
            "maximum_visible_depth_fraction"
        )
    return result


def _balance_context(account: Mapping[str, Any] | None) -> dict[str, Any]:
    balances = account.get("paper_starting_balances") if isinstance(account, Mapping) else {}
    if not isinstance(balances, Mapping):
        return {}
    return {
        "initial_btc": balances.get("BTC"),
        "initial_stablecoin": balances.get("USDT"),
        "initial_cash": balances.get("USDT"),
    }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _candidate_supported_instruments(
    candidate: Candidate, resolved: tuple[Any, ...]
) -> tuple[str, ...]:
    available = tuple(
        dict.fromkeys(str(symbol) for item in resolved for symbol in item.instrument_scope)
    )
    declared = candidate.definition.universe.get("instrument_ids")
    if isinstance(declared, list | tuple) and declared:
        selected = tuple(dict.fromkeys(str(value) for value in declared if str(value).strip()))
    else:
        symbols = candidate.definition.universe.get("symbols")
        selected = tuple(
            instrument
            for instrument in available
            if not isinstance(symbols, list | tuple)
            or any(
                instrument.endswith(f":{str(symbol)}")
                or instrument.endswith(f":{str(symbol)}:USDT")
                for symbol in symbols
            )
        )
    if not selected or not set(selected).issubset(set(available)):
        raise JobSchemaError("candidate instrument scope is outside its dataset scope")
    return tuple(sorted(selected))


def _candidate_scope_context(candidate: Candidate, context: Mapping[str, Any]) -> dict[str, Any]:
    declared = (
        candidate.definition.universe.get("instrument_ids")
        or candidate.definition.universe.get("symbols")
        or context.get("instrument_scope")
    )
    if not isinstance(declared, list | tuple) or not declared:
        raise DataUnavailableError("candidate evaluation requires a frozen instrument scope")
    scope = tuple(dict.fromkeys(str(value) for value in declared if str(value).strip()))
    selected = {item.upper() for item in scope}
    result = dict(context)
    for field in ("market_frame", "feature_rows", "bars", "warmup_market_frame", "funding_events"):
        rows = result.get(field)
        if isinstance(rows, list | tuple):
            result[field] = tuple(
                row for row in rows if isinstance(row, Mapping) and _row_in_scope(row, selected)
            )
    symbol_returns = result.get("symbol_returns")
    if isinstance(symbol_returns, Mapping):
        scoped_returns = {
            str(key): value
            for key, value in symbol_returns.items()
            if _value_in_scope(str(key), selected)
        }
        result["symbol_returns"] = scoped_returns
        if len(scoped_returns) == 1:
            result["returns"] = next(iter(scoped_returns.values()))
    for field_name in ("cross_sectional_values", "linked_instruments"):
        values = result.get(field_name)
        if isinstance(values, Mapping):
            result[field_name] = _scoped_mapping(values, selected)
    for field_name, value_field in (
        ("cross_sectional_history", "cross_sectional_values"),
        ("linked_instrument_history", "linked_instruments"),
    ):
        history = result.get(field_name)
        if isinstance(history, list | tuple):
            result[field_name] = _scoped_history(history, value_field, selected)
    result["instrument_scope"] = scope
    result = _rebuild_scoped_series(result)
    result["evaluation_input"] = {
        "schema": "research.evaluation_input/v1",
        "candidate_id": candidate.candidate_id,
        "strategy_version_id": candidate.definition.strategy_version_id,
        "instrument_scope": list(scope),
        "dataset_snapshot_ids": list(context.get("dataset_snapshot_ids", ())),
        "dataset_roles": [row.get("role") for row in context.get("dataset_receipts", ())],
        "timeframe": result.get("timeframe"),
        "feature_manifest_id": context.get("feature_manifest_id"),
        "cost_model_id": context.get("cost_model_id"),
        "data_plan_id": context.get("data_plan_id"),
        "rows_hash": canonical_hash(
            {
                field: result.get(field)
                for field in ("market_frame", "feature_rows", "warmup_market_frame")
            }
        ),
    }
    return result


def _rebuild_scoped_series(context: dict[str, Any]) -> dict[str, Any]:
    from src.research.datasets import CanonicalResearchDatasetBuilder

    rows = context.get("market_frame")
    if not isinstance(rows, list | tuple):
        return context
    if not rows:
        raise DataUnavailableError("candidate has no market rows in its declared scope")
    actual = {str(row.get("instrument_id") or row.get("symbol")) for row in rows}
    if not all(
        any(_value_in_scope(value, {instrument.upper()}) for value in actual)
        for instrument in context["instrument_scope"]
    ):
        raise DataUnavailableError("candidate panel is missing a declared instrument")
    derived = CanonicalResearchDatasetBuilder._bar_payload(
        list(rows),
        timeframe=context.get("timeframe"),
        warmup=tuple(context.get("warmup_market_frame", ())),
    )
    return {
        **context,
        **{
            key: value
            for key, value in derived.items()
            if key not in {"feature_rows", "market_frame", "bars", "warmup_market_frame"}
        },
    }


def _row_in_scope(row: Mapping[str, Any], scope: set[str]) -> bool:
    return _value_in_scope(
        str(row.get("instrument_id") or row.get("symbol") or ""),
        scope,
    )


def _value_in_scope(value: str, scope: set[str]) -> bool:
    upper = value.upper()
    return upper in scope or any(
        upper.endswith(f":{item}")
        or upper.endswith(f":{item}:USDT")
        or item.endswith(f":{upper}")
        or item.endswith(f":{upper}:USDT")
        for item in scope
    )


def _scoped_mapping(value: Mapping[str, Any], scope: set[str]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if not str(key).startswith("__") and _value_in_scope(str(key), scope)
    }


def _scoped_history(
    history: list[Any] | tuple[Any, ...], value_field: str, scope: set[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in history:
        if not isinstance(raw, Mapping):
            continue
        values = raw.get(value_field)
        if not isinstance(values, Mapping):
            continue
        scoped_values = _scoped_mapping(values, scope)
        forward = raw.get("forward_returns")
        scoped_forward = _scoped_mapping(forward, scope) if isinstance(forward, Mapping) else {}
        if len(scoped_values) >= 2:
            result.append(
                {
                    **dict(raw),
                    value_field: scoped_values,
                    "forward_returns": scoped_forward,
                }
            )
    return result


@dataclass(frozen=True)
class _CatalogueDataset:
    snapshot_ids: tuple[str, ...]
    bundle: DatasetBundle | None
    plan: CandidateDatasetPlan | None
    bundle_id: str | None


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
        dataset_bundle_service: DatabaseDatasetBundleService | None = None,
        historical_bootstrap_service: HistoricalBootstrapService | None = None,
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
        self.dataset_bundle_service = dataset_bundle_service
        self.historical_bootstrap_service = historical_bootstrap_service

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
            "historical_bootstrap": self.historical_bootstrap,
        }

    def historical_bootstrap(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        if self.historical_bootstrap_service is None:
            raise JobSchemaError("historical bootstrap service is not configured")
        required = {
            "product_id",
            "market_type",
            "symbols",
            "minimum_history_days",
            "timeframe",
        }
        if not required.issubset(claimed.payload):
            raise JobSchemaError("historical bootstrap command is incomplete")
        symbols = claimed.payload["symbols"]
        if not isinstance(symbols, list | tuple) or not symbols:
            raise JobSchemaError("historical bootstrap needs at least one symbol")
        market = str(claimed.payload["market_type"])
        if market == "usdt_futures":
            market = "futures"
        if market not in {"spot", "futures"}:
            raise JobSchemaError("historical bootstrap market is invalid")
        renew()
        reports = self.historical_bootstrap_service.bootstrap_product(
            product_id=str(claimed.payload["product_id"]),
            market=market,
            symbols=tuple(str(symbol) for symbol in symbols),
            minimum_history_days=int(claimed.payload["minimum_history_days"]),
            timeframe=str(claimed.payload["timeframe"]),
            data_plan=claimed.payload.get("data_plan"),
            heartbeat=renew,
        )
        result: dict[str, Any] = {
            "product_id": str(claimed.payload["product_id"]),
            "reports": [report.to_payload() for report in reports],
            "completed": all(report.completed for report in reports),
        }
        if not result["completed"] or not all(
            report.get("metadata_fetched") is True for report in result["reports"]
        ):
            result["completed"] = False
            result.update(
                {
                    "_defer": True,
                    "_defer_reason": "historical_bootstrap_incomplete",
                }
            )
        return result

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
        product_id = str(claimed.payload["product_id"])
        catalogue_at = _runtime_data_timestamp(claimed.payload, "catalogue_submitted_at")
        dataset, waiting = self._catalogue_dataset(
            claimed.payload,
            product_id=product_id,
            created_at=catalogue_at,
        )
        if waiting is not None:
            if waiting.get("state") == "blocked_dataset":
                raise JobSchemaError(str(waiting.get("reason_code") or "catalogue_dataset_blocked"))
            return {
                **waiting,
                "_defer": True,
                "_defer_reason": str(waiting.get("reason_code") or "catalogue_waiting"),
            }
        if dataset is None:
            raise JobSchemaError("catalogue dataset resolution returned no result")
        return self._register_catalogue_candidates(
            claimed,
            dataset=dataset,
            product_id=product_id,
            catalogue_at=catalogue_at,
        )

    def _register_catalogue_candidates(
        self,
        claimed: ClaimedJob,
        *,
        dataset: _CatalogueDataset,
        product_id: str,
        catalogue_at: str,
    ) -> dict[str, Any]:
        snapshot_ids = dataset.snapshot_ids
        bundle_id = dataset.bundle_id
        bundle = dataset.bundle
        plan = dataset.plan
        universe = tuple(claimed.payload.get("instrument_universe", ("BTCUSDT",)))
        if not universe:
            return {
                "product_id": product_id,
                "state": "waiting_for_universe",
                "reason_code": "point_in_time_universe_unavailable",
                "_defer": True,
                "_defer_reason": "point_in_time_universe_unavailable",
            }
        theses = registered_strategy_theses(product=product_id, instrument_universe=universe)
        thesis_registry = SqlThesisRegistry(self.store.engine)
        for thesis in theses.values():
            thesis_registry.register(thesis)
        if product_id == "active_income" and len(universe) > 1:
            for symbol in universe:
                for thesis in registered_strategy_theses(
                    product=product_id, instrument_universe=(symbol,)
                ).values():
                    thesis_registry.register(thesis)
        candidates = registered_strategy_candidates(
            product=product_id,
            dataset_snapshot_hashes=snapshot_ids,
            dataset_bundle_id=(str(bundle_id) if bundle_id else None),
            universe_snapshot_id=(
                plan.universe_snapshot_id
                if plan is not None
                else _optional_text(claimed.payload.get("universe_snapshot_id"))
            ),
            instrument_universe=universe,
            submitted_at=catalogue_at,
        )
        if plan is not None:
            if bundle is None:
                raise JobSchemaError("catalogue dataset plan has no bundle")
            candidates = tuple(
                replace(
                    candidate,
                    dataset_snapshot_hashes=plan.all_snapshot_ids,
                    dataset_bundle_id=bundle.bundle_id,
                    dataset_plan=plan,
                )
                for candidate in candidates
            )
        identities = ResearchCoordinator(self.store).register(candidates)
        return {"registered_candidates": len(identities), "candidate_ids": list(identities)}

    def _catalogue_dataset(
        self, payload: Mapping[str, Any], *, product_id: str, created_at: str | None = None
    ) -> tuple[_CatalogueDataset | None, dict[str, Any] | None]:
        snapshot_ids = tuple(str(item) for item in payload["dataset_snapshot_hashes"])
        bundle_id = _optional_text(payload.get("dataset_bundle_id"))
        if self.dataset_bundle_service is not None and bundle_id is None:
            from src.research.data_plan import ResearchDataPlan

            built = self.dataset_bundle_service.run(
                product_id=product_id,
                universe_id=str(payload.get("universe_id") or ""),
                market_type=str(payload.get("market_type") or "futures"),
                created_at=str(
                    created_at
                    or payload.get("catalogue_submitted_at")
                    or payload.get("available_at")
                ),
                timeframe=_optional_text(payload.get("dataset_timeframe")),
                data_plan=ResearchDataPlan.from_payload(payload["data_plan"])
                if payload.get("data_plan")
                else None,
            )
            if built.state != "ready" or built.bundle_id is None:
                return None, {
                    "product_id": product_id,
                    "state": built.state,
                    "reason_code": built.reason_code,
                }
            bundle_id = built.bundle_id
            snapshot_ids = built.snapshot_ids
        if not snapshot_ids:
            return None, {
                "product_id": product_id,
                "state": "waiting_for_dataset",
                "reason_code": "canonical_dataset_bundle_unavailable",
            }
        bundle = None
        plan = None
        if bundle_id is not None:
            bundle = SqlDatasetBundleRepository(self.store.engine).get(bundle_id)
            if bundle.lifecycle_state is not DatasetLifecycleState.READY:
                return None, {
                    "product_id": product_id,
                    "state": "waiting_for_dataset",
                    "reason_code": "canonical_dataset_bundle_not_ready",
                    "dataset_bundle_id": bundle_id,
                }
            plan = CandidateDatasetPlan.from_bundle(bundle)
            if set(snapshot_ids) != set(plan.all_snapshot_ids):
                raise JobSchemaError("catalogue datasets do not match the canonical bundle")
        return _CatalogueDataset(snapshot_ids, bundle, plan, bundle_id), None

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

        from src.research.rounds import ResearchRoundCoordinator

        rounds = ResearchRoundCoordinator(self.store.engine)
        payload = self._resolve_generation_dataset(claimed.payload)
        if not payload.get("dataset_bundle_id"):
            return self._generation_waiting_response(
                str(payload["product_id"]),
                universe_available=bool(payload.get("instrument_universe")),
            )
        if not payload.get("round_id"):
            limits = (self.configuration or {}).get("research", {}).get("search_budgets", {})
            request = rounds.claim(
                payload,
                now=_runtime_data_timestamp(payload, "submitted_at"),
                epoch_budget=int(limits.get("maximum_candidates_per_epoch", 100)),
                daily_budget=int(limits.get("maximum_candidates_per_day", 100)),
            )
            if request is None:
                return {"state": "no_new_round", "candidate_ids": [], "candidate_count": 0}
            payload = request
        return rounds.complete(
            str(payload["round_id"]),
            produce=lambda connection, request: self._generate_hypotheses(
                replace(claimed, payload=dict(request)), renew, connection=connection
            ),
        )

    def _generate_hypotheses(self, claimed, renew, *, connection=None):
        payload = self._resolve_generation_dataset(claimed.payload)
        product_id, universe, snapshot_ids = self._generation_inputs(payload)
        store = SqlResearchStore(self.store.engine, connection=connection)
        feedback_store = SqlGenerationFeedbackStore(
            self.store.engine,
            connection=connection,
            as_of=payload.get("feedback_at"),
            feedback_ids=payload.get("feedback_ids"),
        )
        if not universe or not snapshot_ids:
            self._record_generation_waiting(
                product_id=product_id,
                observed_at=str(payload["submitted_at"]),
                universe_available=bool(universe),
                feedback_store=feedback_store,
            )
            return self._generation_waiting_response(product_id, universe_available=bool(universe))
        generator = HypothesisGenerator(
            product=product_id,
            instrument_universe=universe,
            memory=SqlHypothesisMemory(self.store.engine, connection=connection),
            feedback_store=feedback_store,
        )
        parent_candidates = store.development_parents(
            as_of=payload.get("feedback_at"), stage_ids=payload.get("development_stage_ids")
        )
        bundle, plan = self._generation_plan(payload)
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
            parent_candidates=parent_candidates,
            dataset_plan=plan,
        )
        registered = self._register_hypotheses(
            hypotheses,
            renew=renew,
            plan=plan,
            bundle=bundle,
            observed_at=str(payload["submitted_at"]),
            feedback_store=feedback_store,
            store=store,
        )
        return {
            "product_id": product_id,
            "state": "generated",
            "candidate_ids": registered,
            "candidate_count": len(registered),
        }

    def _resolve_generation_dataset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(payload)
        snapshot_ids = tuple(str(item) for item in payload.get("dataset_snapshot_hashes", ()))
        bundle_id = _optional_text(payload.get("dataset_bundle_id"))
        if snapshot_ids or self.dataset_bundle_service is None or bundle_id is not None:
            return resolved
        observed_at = _runtime_data_timestamp(payload, "submitted_at", "available_at")
        from src.research.data_plan import ResearchDataPlan

        bundle = SqlDatasetBundleRepository(self.store.engine).latest_ready(
            str(payload["product_id"]),
            at=observed_at,
            data_plan_id=ResearchDataPlan.from_payload(payload["data_plan"]).plan_id
            if payload.get("data_plan")
            else None,
        )
        if bundle is None:
            return resolved
        requested_universe = payload.get("universe_snapshot_id")
        if requested_universe and bundle.universe_snapshot_id != requested_universe:
            return resolved
        return {
            **resolved,
            "submitted_at": observed_at,
            "dataset_snapshot_hashes": list(bundle.stage_snapshot_ids.values()),
            "dataset_bundle_id": bundle.bundle_id,
            "universe_snapshot_id": bundle.universe_snapshot_id,
        }

    @staticmethod
    def _generation_inputs(
        payload: Mapping[str, Any],
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        required = {
            "product_id",
            "instrument_universe",
            "dataset_snapshot_hashes",
            "submitted_at",
            "generation_budget",
        }
        if not required.issubset(payload):
            raise JobSchemaError("hypothesis generation command is incomplete")
        return (
            str(payload["product_id"]),
            tuple(str(item) for item in payload["instrument_universe"]),
            tuple(str(item) for item in payload["dataset_snapshot_hashes"]),
        )

    @staticmethod
    def _generation_waiting_response(
        product_id: str, *, universe_available: bool
    ) -> dict[str, Any]:
        reason_code = (
            "canonical_dataset_bundle_unavailable"
            if universe_available
            else "point_in_time_universe_unavailable"
        )
        return {
            "product_id": product_id,
            "state": "waiting_for_dataset" if universe_available else "waiting_for_universe",
            "_defer": True,
            "_defer_reason": reason_code,
        }

    @staticmethod
    def _record_generation_waiting(
        *,
        product_id: str,
        observed_at: str,
        universe_available: bool,
        feedback_store: SqlGenerationFeedbackStore,
    ) -> None:
        reason_code = (
            "canonical_dataset_bundle_unavailable"
            if universe_available
            else "point_in_time_universe_unavailable"
        )
        for campaign in CAMPAIGNS:
            if campaign.product == product_id and campaign.enabled_by_default:
                feedback_store.append(
                    GenerationFeedback(
                        campaign=campaign.name,
                        outcome="data_unavailable",
                        observed_at=observed_at,
                        reason_code=reason_code,
                    )
                )

    def _generation_plan(
        self, payload: Mapping[str, Any]
    ) -> tuple[Any | None, CandidateDatasetPlan | None]:
        bundle_id = payload.get("dataset_bundle_id")
        if not bundle_id:
            return None, None
        bundle = SqlDatasetBundleRepository(self.store.engine).get(str(bundle_id))
        if bundle.lifecycle_state is not DatasetLifecycleState.READY:
            raise JobSchemaError("hypothesis generation requires a ready dataset bundle")
        return bundle, CandidateDatasetPlan.from_bundle(bundle)

    def _register_hypotheses(
        self,
        hypotheses: Any,
        *,
        renew: Callable[[], ClaimedJob],
        plan: CandidateDatasetPlan | None,
        bundle: Any | None,
        observed_at: str,
        feedback_store: SqlGenerationFeedbackStore,
        store: SqlResearchStore,
    ) -> list[str]:
        thesis_registry = SqlThesisRegistry(store.engine, connection=store.connection)
        coordinator = ResearchCoordinator(store)
        registered: list[str] = []
        for hypothesis in hypotheses:
            if store.connection is None:
                renew()
            hypothesis = self._bind_generation_plan(hypothesis, plan=plan, bundle=bundle)
            try:
                thesis_registry.register(hypothesis.thesis)
                candidate_id = coordinator.submit(hypothesis.candidate)
            except ThesisError as exc:
                feedback_store.append(
                    GenerationFeedback(
                        campaign=hypothesis.campaign.name,
                        outcome="resource_budget_exhausted",
                        observed_at=observed_at,
                        reason_code=str(exc),
                        semantic_signature=hypothesis.semantic_signature,
                    )
                )
                continue
            feedback_store.append(
                GenerationFeedback(
                    campaign=hypothesis.campaign.name,
                    outcome="generated",
                    observed_at=observed_at,
                    candidate_id=candidate_id,
                    semantic_signature=hypothesis.semantic_signature,
                )
            )
            registered.append(candidate_id)
        return registered

    @staticmethod
    def _bind_generation_plan(
        hypothesis: Any,
        *,
        plan: CandidateDatasetPlan | None,
        bundle: Any | None,
    ) -> Any:
        if plan is None:
            return hypothesis
        candidate = hypothesis.candidate
        if plan.product_id != candidate.definition.product:
            raise JobSchemaError("generated candidate and dataset bundle products differ")
        if bundle is None:
            raise JobSchemaError("generated dataset plan has no bundle")
        return replace(
            hypothesis,
            candidate=replace(
                candidate,
                dataset_snapshot_hashes=plan.all_snapshot_ids,
                dataset_bundle_id=bundle.bundle_id,
                dataset_plan=plan,
            ),
        )

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
        cohort = None
        if request.requested_stage == "protected":
            execution_time = utc_now()
            research_configuration = (
                self.configuration.get("research") if self.configuration else None
            )
            try:
                cohort = prepare_cohort(
                    self.store.engine,
                    candidate,
                    now=execution_time,
                    policy_hash=self.evidence_policy.policy_hash,
                    configuration_hash=canonical_hash(research_configuration)
                    if isinstance(research_configuration, Mapping)
                    else None,
                )
            except (CohortPending, DataUnavailableError) as exc:
                return {
                    "candidate_id": candidate.candidate_id,
                    "stage": "protected",
                    "accepted": False,
                    "deferred": True,
                    "_defer": True,
                    "_defer_reason": str(exc),
                    "reason_code": "cohort_pending"
                    if isinstance(exc, CohortPending)
                    else "cohort_data_unavailable",
                    "outcome_category": "blocked_infrastructure"
                    if isinstance(exc, CohortPending)
                    else "blocked_data",
                }
            executed = {
                key: value for key, value in claimed.payload.items() if key != "content_hash"
            }
            executed["evaluated_at"] = execution_time
            request = EvaluationRequest.from_payload(
                {**executed, "content_hash": canonical_hash(executed)}
            )
        context = self._resolve_evaluation_context(
            candidate=candidate,
            request=request,
            payload=claimed.payload,
            resolver=resolver,
        )
        if cohort is not None:
            context["protected_cohort_id"] = cohort["cohort_id"]

        def evaluate_protected(claim: Mapping[str, Any]) -> tuple[bool, Mapping[str, Any]]:
            protected_context = claim.get("protected_context")
            if not isinstance(protected_context, Mapping):
                raise JobSchemaError("protected holdout worker did not resolve its dataset")
            frozen_context = self.context_builders.build(
                candidate, _candidate_scope_context(candidate, protected_context)
            )
            execution = self.executors.execute(
                candidate,
                {
                    **frozen_context,
                    **self._product_accounting_context(candidate.definition.product),
                    "product_id": candidate.definition.product,
                    "requested_stage": "protected",
                    "evaluated_at": request.evaluated_at,
                    "evidence_policy_hash": self.evidence_policy.policy_hash,
                    "minimum_bootstrap_observations": (
                        self.evidence_policy.minimum_bootstrap_observations
                    ),
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
                "protected",
                measured,
                (),
                product_id=candidate.definition.product if requires_objective else None,
                family=str(candidate.definition.family),
                horizon=str(
                    candidate.definition.position_model.get("horizon")
                    or candidate.definition.position_model.get("horizon_bars")
                    or "*"
                ),
                evidence_type=str(
                    candidate.definition.validation_policy.get("evidence_type") or ""
                ),
            )
            if accepted:
                stages = SqlValidationRepository(self.store.engine).stages(candidate.candidate_id)
                development = next(
                    (
                        row.get("payload", {}).get("evidence")
                        for row in stages
                        if row.get("stage") == "development"
                        and isinstance(row.get("payload"), Mapping)
                        and isinstance(row["payload"].get("evidence"), Mapping)
                    ),
                    None,
                )
                if isinstance(development, Mapping):
                    dimensions = (
                        str(candidate.definition.family),
                        str(
                            candidate.definition.position_model.get("horizon")
                            or candidate.definition.position_model.get("horizon_bars")
                            or "*"
                        ),
                        str(candidate.definition.validation_policy.get("evidence_type") or ""),
                    )
                    accepted = holdout_degradation_passes(
                        development,
                        measured,
                        self.evidence_policy.profile_for(
                            "protected",
                            product_id=(
                                candidate.definition.product if requires_objective else None
                            ),
                            family=dimensions[0],
                            horizon=dimensions[1],
                            evidence_type=dimensions[2],
                        ),
                    )
            sealed = {
                "passed": accepted,
                "evidence_hash": canonical_hash(measured),
                "metrics": dict(execution.metrics),
                "execution_receipt": dict(execution.receipt),
                "selection_stream": selection_stream(measured),
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
                product_id=candidate.definition.product,
                feature_manifest_id=str(request.feature_manifest_id),
                cost_model_id=str(request.cost_model_id),
                parameter_set_id=str(request.parameter_set_id),
            ),
            evidence_policy=self.evidence_policy,
        ).evaluate(request)
        if cohort is not None and not result.deferred:
            complete_cohort(self.store.engine, cohort, now=utc_now())
        self._record_agent_evaluation(candidate, result)
        return self._finalise_evaluation(candidate, request, result)

    def _record_agent_evaluation(self, candidate: Candidate, result: Any) -> None:
        from src.agents.store import SqlAgentStore

        proposal_id = candidate.definition.metadata.get("openclaw_proposal_id")
        if (
            candidate.provider == "openclaw_trusted_compiler"
            and proposal_id
            and not result.deferred
        ):
            SqlAgentStore(self.store.engine).save_review(
                proposal_id=str(proposal_id),
                created_at=utc_now(),
                payload={
                    "schema": "platform.agent_evaluation_progress/v1",
                    "state": "evaluated",
                    "candidate_id": candidate.candidate_id,
                },
            )

    def _resolve_evaluation_context(
        self,
        *,
        candidate: Candidate,
        request: EvaluationRequest,
        payload: Mapping[str, Any],
        resolver: CanonicalDatasetResolver,
    ) -> dict[str, Any]:
        adaptive_snapshot_ids = request.snapshot_ids_for_stage(request.requested_stage)
        if request.requested_stage == "protected":
            protected_id = (
                request.protected_snapshot_id()
                if request.dataset_roles
                else request.dataset_snapshot_ids[0]
            )
            adaptive_snapshot_ids = tuple(
                snapshot_id
                for snapshot_id in request.dataset_snapshot_ids
                if snapshot_id != protected_id
            )
        if adaptive_snapshot_ids:
            resolved_context = resolver.resolve_context(
                snapshot_ids=adaptive_snapshot_ids,
                product_id=candidate.definition.product,
                feature_manifest_id=str(request.feature_manifest_id),
                cost_model_id=str(request.cost_model_id),
                parameter_set_id=str(request.parameter_set_id),
                allowed_roles=(
                    frozenset(
                        request.dataset_roles[snapshot_id] for snapshot_id in adaptive_snapshot_ids
                    )
                    if request.dataset_roles
                    else None
                ),
                minimum_availability_timestamp=(
                    str(payload["artefact_created_at"])
                    if request.requested_stage == "forward"
                    else None
                ),
                maximum_availability_timestamp=(
                    request.evaluated_at if request.requested_stage == "forward" else None
                ),
            )
            try:
                evaluation_view = CandidateEvaluationView.from_candidate(
                    candidate, adaptive_snapshot_ids
                )
                context = self.context_builders.build(
                    evaluation_view,
                    _candidate_scope_context(candidate, resolved_context),
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
            **self._product_accounting_context(candidate.definition.product),
        }
        return {
            **dict(context),
            "artefact_hash": request.artefact_hash,
            "artefact_created_at": request.artefact_created_at,
            "global_trial_count": SqlThesisRegistry(self.store.engine).lineage_trial_count(
                candidate.thesis_id
            ),
        }

    def _product_accounting_context(self, product_id: str) -> dict[str, Any]:
        if self.configuration is None:
            return {}
        product = _configured_product(self.configuration, product_id)
        if not isinstance(product, Mapping):
            return {}
        account = _configured_account(self.configuration, str(product.get("account_id") or ""))
        result = _cost_context(product)
        if product_id == "btc_accumulation":
            result.update(_btc_accounting_context(product))
        elif product_id == "active_income" and isinstance(account, Mapping):
            result.update(_active_income_accounting_context(self.configuration, account))
        result.update(_balance_context(account))
        return {key: value for key, value in result.items() if value is not None}

    def _finalise_evaluation(
        self, candidate: Candidate, request: EvaluationRequest, result: Any
    ) -> dict[str, Any]:
        response = {
            "candidate_id": result.candidate_id,
            "stage": result.stage,
            "accepted": result.accepted,
            "deferred": result.deferred,
            "reason_code": result.reason_code,
            "run_id": result.run_id,
            "evidence_hash": result.evidence_hash,
            "outcome_category": result.evidence.get("evaluation_decision", {}).get(
                "outcome_category"
            ),
        }
        if result.deferred and response["outcome_category"] == "blocked_data":
            response.update({"_defer": True, "_defer_reason": result.reason_code})
        if result.deferred and response["outcome_category"] == "blocked_infrastructure":
            raise ExecutorError(f"evaluation infrastructure failure: {result.reason_code}")
        if result.stage == "protected" and result.accepted:
            response["artefact_hash"] = self._publish_strategy_artefact(
                candidate=candidate,
                request=request,
                evidence=result.evidence,
            )
        campaign = candidate.definition.metadata.get("campaign")
        if isinstance(campaign, str) and campaign:
            self._record_evaluation_feedback(candidate, request, result, campaign=campaign)
        return response

    def _record_evaluation_feedback(
        self,
        candidate: Candidate,
        request: EvaluationRequest,
        result: Any,
        *,
        campaign: str,
    ) -> None:
        if result.deferred or result.stage != "development":
            return
        SqlGenerationFeedbackStore(self.store.engine).append(
            GenerationFeedback(
                campaign=campaign,
                outcome="accepted" if result.accepted else "rejected",
                observed_at=request.evaluated_at,
                candidate_id=candidate.candidate_id,
                reason_code=result.reason_code,
                semantic_signature=hypothesis_signature(candidate.definition),
                metadata={
                    "stage": result.stage,
                    "economic_engine": ECONOMIC_ENGINE_VERSION,
                    "net_return": result.evidence.get("cost_adjusted_return"),
                    "observations": result.evidence.get("observations"),
                    "costs": result.evidence.get("fees"),
                },
            )
        )

    def _publish_strategy_artefact(
        self,
        *,
        candidate: Candidate,
        request: EvaluationRequest,
        evidence: Mapping[str, Any],
    ) -> str:
        if self.configuration is None or self.dataset_resolver is None:
            raise JobSchemaError("accepted protected research requires platform configuration")
        with self.store.engine.connect() as connection:
            existing = connection.execute(
                select(paper_eligibility_event.c.id).where(
                    paper_eligibility_event.c.candidate_id == candidate.candidate_id,
                    paper_eligibility_event.c.engine_version == ECONOMIC_ENGINE_VERSION,
                )
            ).scalar_one_or_none()
        if existing is not None:
            return SqlStrategyArtefactRepository(self.store.engine).get(existing)["artefact_hash"]
        sealed_at = utc_now()
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
        supported_instruments = _candidate_supported_instruments(candidate, resolved)
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
            supported_instruments=supported_instruments,
            created_at=sealed_at,
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
                "assignment_scope": list(supported_instruments),
                "selection_cohort_id": evidence.get("frozen_cohort"),
            },
            product_id=candidate.definition.product,
            portfolio_id=str(product["portfolio_id"]),
            account_id=str(product["account_id"]),
            promotion_policy_id=policy_id,
            engine_version=ECONOMIC_ENGINE_VERSION,
        )
        return SqlStrategyArtefactRepository(self.store.engine).put(
            artefact.artefact_hash,
            artefact.to_dict(),
            created_at=sealed_at,
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
            slippage_bps=float(loaded.get("slippage_bps", 0.0)),
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
                "slippage_paid": result.slippage_paid,
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
            terminal_fee_bps=float(loaded.get("terminal_fee_bps", 5.0)),
            terminal_slippage_bps=float(loaded.get("terminal_slippage_bps", 2.0)),
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
                "terminal_liquidation_cost": result.terminal_liquidation_cost,
                "terminal_exposure": result.terminal_exposure,
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
                f"{kind} requires a canonical dataset loader; queue payloads cannot contain "
                "raw data"
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
