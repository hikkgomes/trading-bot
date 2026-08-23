"""Concrete handlers for the unified leased research queue."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.data.parquet_store import PartitionedBacktestStore
from src.domain._codec import canonical_hash
from src.domain.strategies import StrategySourceType
from src.research.backtest.bar_engine import BarPortfolioEngine, BarStep
from src.research.backtest.event_engine import (
    EventReplayEngine,
    ReplayEvent,
    SimulatedLimitOrder,
    SimulatedOrderSide,
)
from src.research.catalogue import registered_strategy_candidates
from src.research.coordinator import Candidate, ResearchCoordinator
from src.research.evaluation import CanonicalResearchEvaluator, EvaluationRequest
from src.research.ml import MlExperimentRunner
from src.research.providers import provider_candidate
from src.research.store import SqlResearchStore
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
    ):
        self.store = store
        self.artefact_store = artefact_store
        self.ml_runner = ml_runner
        self.dataset_loader = dataset_loader

    def handlers(self) -> dict[str, Callable]:
        return {
            "register_strategy_catalogue": self.register_strategy_catalogue,
            "register_candidate": self.register_candidate,
            "register_ml_candidate": self.register_ml_candidate,
            "evaluate_candidate": self.evaluate_candidate,
            "bounded_backtest": self.bounded_backtest,
            "event_replay": self.event_replay,
            "train_ml_experiment": self.train_ml_experiment,
        }

    def register_strategy_catalogue(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        renew()
        candidates = registered_strategy_candidates(
            product=str(claimed.payload["product_id"]),
            dataset_snapshot_hashes=tuple(claimed.payload["dataset_snapshot_hashes"]),
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

    def register_ml_candidate(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        candidate = self._candidate(claimed.payload)
        if candidate.definition.source_type is not StrategySourceType.MACHINE_LEARNING:
            raise ValueError("ML worker accepts only machine-learning candidates")
        renew()
        candidate_id = ResearchCoordinator(self.store).submit(candidate)
        return {"candidate_id": candidate_id, "source_type": candidate.definition.source_type.value}

    def evaluate_candidate(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        renew()
        ResearchJobRequest.from_mapping(claimed.payload)
        request = EvaluationRequest.from_payload(claimed.payload)
        result = CanonicalResearchEvaluator(self.store).evaluate(request)
        return {
            "candidate_id": result.candidate_id,
            "stage": result.stage,
            "accepted": result.accepted,
            "reason_code": result.reason_code,
            "run_id": result.run_id,
            "evidence_hash": result.evidence_hash,
        }

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
    return ResearchJobRequest.from_mapping(payload)


def _execution_receipt(request: ResearchJobRequest, *, executor_version: str) -> dict[str, Any]:
    payload = {
        "candidate_id": request.candidate_id,
        "dataset_snapshot_ids": list(request.dataset_snapshot_ids),
        "feature_manifest_id": request.feature_manifest_id,
        "cost_model_id": request.cost_model_id,
        "parameter_set_id": request.parameter_set_id,
        "evaluator_version": request.evaluator_version,
        "executor_version": executor_version,
    }
    return {**payload, "input_hash": canonical_hash(payload)}
