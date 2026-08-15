"""Concrete handlers for the unified leased research queue."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.data.parquet_store import PartitionedBacktestStore
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
from src.research.ml import MlExperimentRunner
from src.research.providers import provider_candidate
from src.research.store import SqlResearchStore
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
    ):
        self.store = store
        self.artefact_store = artefact_store
        self.ml_runner = ml_runner

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
        coordinator = ResearchCoordinator(self.store)
        stages = claimed.payload.get("stages")
        if not isinstance(stages, Mapping):
            raise ValueError("research evaluation stages must be an object")

        def validator(stage_name: str):
            def evaluate(_candidate: Candidate):
                renew()
                stage = stages.get(stage_name)
                if not isinstance(stage, Mapping):
                    raise ValueError(f"research evaluation is missing {stage_name}")
                accepted = stage.get("accepted")
                evidence = stage.get("evidence")
                if not isinstance(accepted, bool) or not isinstance(evidence, Mapping):
                    raise ValueError(f"research evaluation {stage_name} is invalid")
                if accepted:
                    missing = _REQUIRED_EVIDENCE[stage_name] - set(evidence)
                    if missing:
                        raise ValueError(
                            f"research evaluation {stage_name} lacks evidence: {sorted(missing)}"
                        )
                    if stage_name == "development":
                        _validate_sample_evidence(_candidate, evidence["sample_evidence"])
                reason = stage.get("reason_code")
                return accepted, str(reason) if reason is not None else None, dict(evidence)

            return evaluate

        result = coordinator.evaluate(
            str(claimed.payload["candidate_id"]),
            screening=validator("screening"),
            development=validator("development"),
            robustness=validator("robustness"),
            protected=validator("protected"),
            evaluated_at=str(claimed.payload["evaluated_at"]),
        )
        return {
            "candidate_id": result.candidate_id,
            "state": result.state.value,
            "accepted": result.accepted,
            "reason_code": result.reason_code,
        }

    def bounded_backtest(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        renew()
        steps = tuple(BarStep(**dict(item)) for item in claimed.payload["steps"])
        result = BarPortfolioEngine(
            initial_equity=float(claimed.payload["initial_equity"]),
            fee_bps=float(claimed.payload.get("fee_bps", 5.0)),
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
        )
        return {"run_id": run_id, "final_equity": final_equity}

    def event_replay(self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]) -> dict[str, Any]:
        renew()
        events = tuple(ReplayEvent(**dict(item)) for item in claimed.payload["events"])
        orders = tuple(
            SimulatedLimitOrder(
                **{
                    **dict(item),
                    "side": SimulatedOrderSide(str(item["side"])),
                }
            )
            for item in claimed.payload["orders"]
        )
        result = EventReplayEngine(
            cancel_latency_seconds=float(claimed.payload.get("cancel_latency_seconds", 0.25)),
            impact_bps_per_depth_fraction=float(
                claimed.payload.get("impact_bps_per_depth_fraction", 5.0)
            ),
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
        evidence = {
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
        )
        return {"run_id": run_id, **dict(result.metrics)}

    def train_ml_experiment(
        self, claimed: ClaimedJob, renew: Callable[[], ClaimedJob]
    ) -> dict[str, Any]:
        if self.ml_runner is None:
            raise RuntimeError("ML experiment runner is not configured")
        renew()
        payload = claimed.payload
        result = self.ml_runner.run(
            candidate_id=str(payload["candidate_id"]),
            model_name=str(payload["model_name"]),
            feature_names=tuple(payload["feature_names"]),
            target_name=str(payload["target_name"]),
            rows=tuple(payload["rows"]),
            created_at=str(payload["evaluated_at"]),
            train_fraction=float(payload.get("train_fraction", 0.7)),
            embargo_rows=int(payload.get("embargo_rows", 1)),
            hyperparameters=payload.get("hyperparameters"),
        )
        run_id = self.store.save_run(
            candidate_id=str(payload["candidate_id"]),
            run_name=f"ml:{payload['model_name']}",
            created_at=str(payload["evaluated_at"]),
            evidence={
                "model_artifact_id": result.model_artifact_id,
                "content_hash": result.content_hash,
                "relative_path": result.relative_path,
                "dataset_hash": result.dataset_hash,
                "train_rows": result.train_rows,
                "validation_rows": result.validation_rows,
                "chronological": True,
                "embargo_rows": int(payload.get("embargo_rows", 1)),
            },
            metrics={str(key): float(value) for key, value in result.metrics.items()},
        )
        return {
            "run_id": run_id,
            "model_artifact_id": result.model_artifact_id,
            "content_hash": result.content_hash,
            "metrics": dict(result.metrics),
        }

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
