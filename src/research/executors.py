"""Provider executors for canonical candidate evaluation.

The registry is the only place where a strategy source becomes executable.
Every executor returns measured evidence and a receipt describing the exact
inputs it consumed.  A worker cannot manufacture a validation result by
putting metrics in a queue payload.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.domain._codec import canonical_hash, json_value
from src.domain.strategies import StrategySourceType
from src.research.coordinator import Candidate
from src.strategies.behaviour import RegisteredStrategyBehaviour, StrategyBehaviourError
from src.strategies.semantic import SEMANTIC_STRATEGIES


class ExecutorError(RuntimeError):
    """A candidate could not be compiled or evaluated from canonical inputs."""


@dataclass(frozen=True)
class ExecutionResult:
    evidence: Mapping[str, Any]
    metrics: Mapping[str, float]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.evidence or not self.metrics:
            raise ExecutorError("candidate execution must produce non-empty evidence and metrics")
        object.__setattr__(self, "evidence", json_value(dict(self.evidence), field="evidence"))
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in self.metrics.items()},
        )
        receipt = json_value(dict(self.receipt), field="execution receipt")
        required = {"candidate_id", "dataset_snapshot_ids", "executor_version", "input_hash"}
        if not required.issubset(receipt):
            raise ExecutorError("execution receipt is missing canonical input identities")
        object.__setattr__(self, "receipt", receipt)


Executor = Callable[[Candidate, Mapping[str, Any]], ExecutionResult]
ContextBuilder = Callable[[Candidate, Mapping[str, Any]], Mapping[str, Any]]


class ProviderContextBuilderRegistry:
    """Build exact source-specific runtime inputs from immutable dataset payloads."""

    def __init__(self) -> None:
        self._builders: dict[StrategySourceType, ContextBuilder] = {}

    def register(self, source_type: StrategySourceType, builder: ContextBuilder) -> None:
        if source_type in self._builders:
            raise ValueError(f"context builder already registered for {source_type.value}")
        self._builders[source_type] = builder

    def builder_for(self, source_type: StrategySourceType) -> ContextBuilder:
        try:
            return self._builders[source_type]
        except KeyError as exc:
            raise ExecutorError(f"no context builder registered for {source_type.value}") from exc

    def build(self, candidate: Candidate, context: Mapping[str, Any]) -> Mapping[str, Any]:
        builder = self.builder_for(candidate.definition.source_type)
        source_hash = canonical_hash(context)
        built = dict(builder(candidate, context))
        built["context_hash"] = source_hash
        return built

    @classmethod
    def default(cls) -> ProviderContextBuilderRegistry:
        registry = cls()
        for source_type in (
            StrategySourceType.REGISTERED_PYTHON,
            StrategySourceType.PARAMETER_SEARCH,
            StrategySourceType.MUTATION,
            StrategySourceType.CROSSOVER,
            StrategySourceType.AGENT_GENERATED_PYTHON,
        ):
            registry.register(source_type, _build_registered_context)
        registry.register(StrategySourceType.GENERATED_DSL, _build_dsl_context)
        registry.register(StrategySourceType.MACHINE_LEARNING, _build_ml_context)
        for source_type in (
            StrategySourceType.CROSS_SECTIONAL,
            StrategySourceType.RELATIVE_VALUE,
            StrategySourceType.MICROSTRUCTURE,
            StrategySourceType.ENSEMBLE,
        ):
            registry.register(source_type, _build_semantic_context)
        return registry


class ProviderExecutorRegistry:
    """Map every supported source family to a concrete executor."""

    def __init__(self) -> None:
        self._executors: dict[StrategySourceType, Executor] = {}

    def register(self, source_type: StrategySourceType, executor: Executor) -> None:
        if source_type in self._executors:
            raise ValueError(f"executor already registered for {source_type.value}")
        self._executors[source_type] = executor

    def executor_for(self, source_type: StrategySourceType) -> Executor:
        try:
            return self._executors[source_type]
        except KeyError as exc:
            raise ExecutorError(f"no executor registered for {source_type.value}") from exc

    def execute(self, candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
        return self.executor_for(candidate.definition.source_type)(candidate, context)

    @classmethod
    def default(cls) -> ProviderExecutorRegistry:
        registry = cls()
        registry.register(StrategySourceType.REGISTERED_PYTHON, execute_registered_python)
        registry.register(StrategySourceType.GENERATED_DSL, execute_generated_dsl)
        registry.register(StrategySourceType.PARAMETER_SEARCH, execute_registered_python)
        registry.register(StrategySourceType.MUTATION, execute_registered_python)
        registry.register(StrategySourceType.CROSSOVER, execute_registered_python)
        registry.register(StrategySourceType.MACHINE_LEARNING, execute_machine_learning)
        registry.register(StrategySourceType.CROSS_SECTIONAL, execute_cross_sectional)
        registry.register(StrategySourceType.RELATIVE_VALUE, execute_relative_value)
        registry.register(StrategySourceType.MICROSTRUCTURE, execute_microstructure)
        registry.register(StrategySourceType.ENSEMBLE, execute_ensemble)
        registry.register(StrategySourceType.AGENT_GENERATED_PYTHON, execute_agent_generated_python)
        return registry


def _measured_result(
    candidate: Candidate, context: Mapping[str, Any], output: Any
) -> ExecutionResult:
    from src.metrics import (
        bootstrap_sharpe_ci,
        deflated_sharpe_ratio,
        probability_backtest_overfitting,
        sharpe_ratio,
    )

    snapshots = tuple(str(item) for item in context.get("dataset_snapshot_ids", ()))
    if not snapshots:
        raise ExecutorError("executor requires canonical dataset snapshot identities")
    output_hash = canonical_hash(output)
    signals = _numeric_series(context.get("signals", output))
    returns = _market_returns(context, len(signals))
    if not signals:
        signals = [float(_direction(output))]
    observations = len(signals)
    changes = [abs(signals[index] - signals[index - 1]) for index in range(1, observations)]
    turnover = sum(changes) / max(1, observations - 1)
    active = sum(1 for value in signals if abs(value) > 1e-12)
    from src.research.returns import PositionReturnLedger, ReturnLedgerError

    try:
        return_report = PositionReturnLedger(
            fee_rate=max(0.0, float(context.get("fee_bps", 1.0))) / 10_000.0,
            slippage_rate=max(0.0, float(context.get("slippage_bps", 1.0))) / 10_000.0,
            funding_rate=float(context.get("funding_rate", 0.0)),
        ).measure(signals, returns)
    except (ReturnLedgerError, TypeError, ValueError) as exc:
        raise ExecutorError(f"position return ledger input is invalid: {exc}") from exc
    aligned = return_report.effective_observations
    gross = list(return_report.gross_returns)
    fees = return_report.fees
    slippage = return_report.slippage
    funding_pnl = return_report.funding_pnl
    funding = return_report.funding_cost
    turnover = return_report.turnover
    net_return = return_report.net_pnl
    accounting = _product_accounting(context, fallback_return=return_report)
    if accounting is not None:
        net_return = float(accounting["return_fraction"])
    window_returns = _window_sums(gross, 3)
    negative_controls = _negative_control_evidence(
        signals=signals[:aligned],
        returns=returns[:aligned],
        candidate_return=net_return,
        control_returns=context.get("negative_control_returns"),
        controls=_negative_control_names(candidate, context),
    )
    delayed_gross = [signals[index - 1] * returns[index] for index in range(1, aligned)]
    missing_data_gross = [value for index, value in enumerate(gross) if (index + 1) % 20]
    parameter_stability = _parameter_stability(candidate, context, gross)
    cross_symbol_stability = _cross_symbol_stability(context, gross)
    portfolio_overlap = _portfolio_overlap(context, gross)
    walk_forward = _purged_walk_forward(gross, context)
    bootstrap_low, bootstrap_high = bootstrap_sharpe_ci(
        gross,
        n_boot=int(context.get("bootstrap_iterations", 1_000)),
        random_state=int(candidate.candidate_id[7:15], 16),
    )
    strategy_window_returns = context.get("strategy_window_returns")
    if isinstance(strategy_window_returns, list | tuple):
        pbo_matrix = [
            _numeric_series(row) for row in strategy_window_returns if _numeric_series(row)
        ]
    else:
        pbo_matrix = []
    if len(pbo_matrix) < 2:
        pbo_matrix = [
            _numeric_series(item.get("window_returns"))
            for item in parameter_stability["results"]
            if isinstance(item, Mapping) and _numeric_series(item.get("window_returns"))
        ]
        pbo_matrix.insert(0, window_returns)
    width = min((len(row) for row in pbo_matrix), default=0)
    pbo_matrix = [row[:width] for row in pbo_matrix if width >= 2]
    pbo: float | dict[str, Any]
    if len(pbo_matrix) < 2:
        pbo = {
            "status": "not_applicable",
            "passed": True,
            "reason": "no_valid_configuration_cohort",
            "cohort_size": len(pbo_matrix),
        }
    else:
        pbo = probability_backtest_overfitting(pbo_matrix)
    skew, kurtosis = _skew_kurtosis(gross)
    trial_sharpes = _trial_sharpes(context, parameter_stability, gross)
    trial_count = max(
        1,
        int(
            context.get(
                "trial_count", len(trial_sharpes) or parameter_stability["neighbours_tested"] + 1
            )
        ),
    )
    trial_sharpe_std = float(context.get("trial_sharpe_std", 0.0))
    if "trial_sharpe_std" not in context and len(trial_sharpes) >= 2:
        trial_sharpe_std = statistics.pstdev(trial_sharpes)
    dsr = deflated_sharpe_ratio(
        sharpe_ratio(gross),
        n_trials=trial_count,
        skew=skew,
        kurt=kurtosis,
        n_obs=len(gross),
        sr_std_trials=trial_sharpe_std,
    )
    expected_definition_hash = candidate.definition.definition_hash
    observed_definition_hash = str(context.get("strategy_definition_hash") or "")
    expected_artefact_hash = str(context.get("artefact_hash") or "")
    observed_artefact_hash = str(context.get("runtime_artefact_hash") or "")
    expected_engine_hash = str(context.get("engine_hash") or "")
    observed_engine_hash = str(context.get("runtime_engine_hash") or "")
    expected_cost_model = str(context.get("cost_model_id") or context.get("cost_model_hash") or "")
    observed_cost_model = str(context.get("runtime_cost_model_id") or "")
    production_mode = str(context.get("production_execution_mode") or "")
    runtime_mode = str(context.get("runtime_execution_mode") or "")
    expected_feature_manifest = str(
        context.get("feature_manifest_id") or context.get("feature_set_hash") or ""
    )
    observed_feature_manifest = str(context.get("runtime_feature_manifest_id") or "")
    randomiser = random.Random(int(candidate.candidate_id[7:23], 16))
    monte_carlo_drawdowns = []
    for _ in range(250):
        permuted = list(gross)
        randomiser.shuffle(permuted)
        monte_carlo_drawdowns.append(_maximum_drawdown(permuted))
    declared_universe = candidate.definition.universe.get("symbols")
    scope = tuple(str(item) for item in context.get("instrument_scope", ()))
    predeclared = isinstance(declared_universe, list | tuple) and bool(declared_universe)
    if scope:
        predeclared = predeclared and set(scope).issubset({str(item) for item in declared_universe})
    feature_inputs = any(
        context.get(name) is not None for name in ("feature_rows", "feature_vector", "market_frame")
    )
    measured = {
        "evidence_policy_hash": context.get("evidence_policy_hash"),
        "compiled": True,
        "features_valid": bool(context.get("features_valid", feature_inputs)),
        "causality_valid": bool(
            context.get(
                "causality_valid",
                not bool(context.get("lookahead_detected", False))
                and not bool(context.get("future_input_detected", False)),
            )
        ),
        "signal_frequency": active / observations,
        "turnover": turnover,
        "return_ledger": {
            "gross_pnl": return_report.gross_pnl,
            "net_pnl": return_report.net_pnl,
            "net_returns": list(return_report.net_returns),
            "maximum_drawdown": return_report.maximum_drawdown,
        },
        "chronological": not bool(context.get("lookahead_detected", False)),
        "cost_adjusted_return": net_return,
        "fees": fees,
        "slippage": slippage,
        "funding": funding,
        "funding_pnl": funding_pnl,
        "regime_breakdown": {"passed": bool(gross), "regimes": {"all": net_return}},
        "parameter_stability": parameter_stability,
        "sample_evidence": {
            "passed": aligned >= 3,
            "observations": aligned,
            "run_id": canonical_hash(
                {
                    "kind": "sample_evidence/v1",
                    "candidate_id": candidate.candidate_id,
                    "dataset_snapshot_ids": list(snapshots),
                    "returns": gross,
                }
            ),
            "input_hash": canonical_hash(
                {
                    "dataset_snapshot_ids": list(snapshots),
                    "returns": gross,
                }
            ),
        },
        "cross_symbol_stability": cross_symbol_stability,
        "universe_evidence": {
            "passed": predeclared,
            "predeclared": predeclared,
            "declared_symbols": list(declared_universe or ()),
            "observed_symbols": list(scope),
        },
        "portfolio_overlap": portfolio_overlap,
        "walk_forward": walk_forward,
        "purged": int(walk_forward.get("purged_rows", 0)) > 0,
        "embargo": int(walk_forward.get("embargo_rows", 0)),
        "cost_stress": {
            "passed": sum(gross) - 2 * fees - 2 * slippage - funding >= 0,
            "multiplier": 2.0,
        },
        "delay_stress": {
            "passed": bool(delayed_gross) and sum(delayed_gross) - fees - slippage - funding >= 0,
            "delay_bars": 1,
            "cost_adjusted_return": sum(delayed_gross) - fees - slippage - funding,
        },
        "adverse_fill_stress": {
            "passed": sum(gross) - fees - 2 * slippage - funding >= 0,
            "multiplier": 2.0,
        },
        "missing_data_stress": {
            "passed": bool(missing_data_gross)
            and sum(missing_data_gross) - fees - slippage - funding >= 0,
            "removed_fraction": 1.0 - len(missing_data_gross) / max(1, len(gross)),
        },
        "funding_stress": {
            "passed": sum(gross) - fees - slippage - 2 * funding >= 0,
            "multiplier": 2.0,
        },
        "monte_carlo_trade_order": {
            "passed": bool(monte_carlo_drawdowns) and net_return >= 0,
            "iterations": len(monte_carlo_drawdowns),
            "maximum_drawdown": max(monte_carlo_drawdowns, default=0.0),
        },
        "bootstrap_confidence": {
            "passed": len(gross) >= int(context.get("minimum_bootstrap_observations", 30))
            and bootstrap_low >= 0.0,
            "lower_bound": bootstrap_low,
            "upper_bound": bootstrap_high,
            "observations": len(gross),
            "iterations": int(context.get("bootstrap_iterations", 1_000)),
            "method": str(context.get("bootstrap_method") or "moving_block_bootstrap_v1"),
            "run_id": canonical_hash(
                {
                    "kind": "bootstrap_confidence/v1",
                    "candidate_id": candidate.candidate_id,
                    "dataset_snapshot_ids": list(snapshots),
                    "returns": gross,
                    "iterations": int(context.get("bootstrap_iterations", 1_000)),
                }
            ),
            "input_hash": canonical_hash(
                {
                    "dataset_snapshot_ids": list(snapshots),
                    "returns": gross,
                }
            ),
        },
        "probability_backtest_overfitting": pbo,
        "pbo_input_hash": canonical_hash(
            {
                "dataset_snapshot_ids": list(snapshots),
                "matrix": pbo_matrix,
                "method": str(context.get("pbo_method") or "combinatorial_purged_pbo_v1"),
            }
        ),
        "deflated_sharpe": dsr,
        "multiple_testing": {
            "trial_count": trial_count,
            "trial_sharpe_std": trial_sharpe_std,
            "trial_sharpes": trial_sharpes,
            "input_hash": canonical_hash(
                {
                    "dataset_snapshot_ids": list(snapshots),
                    "trial_sharpes": trial_sharpes,
                    "trial_count": trial_count,
                }
            ),
        },
        "statistical_procedures": {
            "bootstrap": str(context.get("bootstrap_method") or "moving_block_bootstrap_v1"),
            "multiple_testing": str(
                context.get("multiple_testing_method") or "bailey_lopez_de_prado_dsr_v1"
            ),
            "pbo": str(context.get("pbo_method") or "combinatorial_purged_pbo_v1"),
        },
        "drawdown_stability": {
            "passed": bool(return_report.net_returns),
            "maximum_drawdown": return_report.maximum_drawdown,
        },
        "null_results": {
            "passed": all(item["passed"] for item in negative_controls.values()),
            "tests": len(negative_controls),
        },
        "negative_control_results": negative_controls,
        "production_equivalent": {
            "passed": bool(
                production_mode
                and runtime_mode
                and runtime_mode == production_mode
                and expected_engine_hash
                and observed_engine_hash == expected_engine_hash
                and expected_feature_manifest
                and observed_feature_manifest == expected_feature_manifest
                and expected_cost_model
                and observed_cost_model == expected_cost_model
            ),
            "runtime_execution_mode": runtime_mode,
            "production_execution_mode": production_mode,
            "expected_engine_hash": expected_engine_hash,
            "observed_engine_hash": observed_engine_hash,
            "expected_feature_manifest": expected_feature_manifest,
            "observed_feature_manifest": observed_feature_manifest,
            "expected_cost_model": expected_cost_model,
            "observed_cost_model": observed_cost_model,
        },
        "exact_strategy_identity": {
            "passed": observed_definition_hash == expected_definition_hash,
            "expected": expected_definition_hash,
            "observed": observed_definition_hash,
        },
        "exact_artefact_hash": {
            "passed": observed_artefact_hash == expected_artefact_hash,
            "expected": expected_artefact_hash,
            "observed": observed_artefact_hash,
        },
        "exact_engine_hash": {
            "passed": observed_engine_hash == expected_engine_hash,
            "expected": expected_engine_hash,
            "observed": observed_engine_hash,
        },
        "exact_cost_model": {
            "passed": bool(expected_cost_model) and observed_cost_model == expected_cost_model,
            "expected": expected_cost_model,
            "observed": observed_cost_model,
        },
        "drift_checks": _drift_checks(context),
        "duration": float(max(1, aligned)),
        "evidence_units": float(max(1, aligned)),
        "output_hash": output_hash,
        "observations": observations,
        "behaviour_hash": context.get("behaviour_hash"),
    }
    product_id = str(context.get("product_id") or "")
    if product_id in {"btc_accumulation", "active_income"}:
        measured.update(
            {
                "objective_status": "unavailable",
                "objective_unit": "BTC" if product_id == "btc_accumulation" else "USDT",
                "objective_value": None,
                "benchmark_value": None,
                "objective_excess": None,
                "objective_excess_fraction": None,
            }
        )
    if accounting is not None:
        measured["accounting"] = accounting
        measured["objective_status"] = "measured"
        measured["objective_unit"] = accounting["objective_unit"]
        measured["objective_value"] = accounting["objective_value"]
        measured["benchmark_value"] = accounting["benchmark_value"]
        measured["objective_excess"] = accounting["objective_excess"]
        measured["objective_excess_fraction"] = accounting["objective_excess_fraction"]
        measured["accounting_return"] = accounting["return_fraction"]
    return ExecutionResult(
        evidence=measured,
        metrics={
            "observations": float(observations),
            "cost_adjusted_return": net_return,
            "turnover": turnover,
            "deflated_sharpe": dsr,
            "funding_pnl": funding_pnl,
            **(
                {
                    "accounting_return": float(accounting["return_fraction"]),
                    "objective_excess": float(accounting["objective_excess"]),
                }
                if accounting is not None
                else {}
            ),
        },
        receipt=execution_receipt(
            candidate=candidate,
            dataset_snapshot_ids=snapshots,
            executor_version="provider-executors/v3",
            evidence_policy_hash=(
                str(context["evidence_policy_hash"])
                if context.get("evidence_policy_hash") is not None
                else None
            ),
            behaviour_hash=(
                str(context["behaviour_hash"])
                if context.get("behaviour_hash") is not None
                else None
            ),
        ),
    )


def _product_accounting(
    context: Mapping[str, Any],
    *,
    fallback_return: Any | None = None,
) -> dict[str, Any] | None:
    """Evaluate explicit product event evidence when a dataset supplies it."""

    product_id = str(context.get("product_id") or "")
    if product_id == "btc_accumulation":
        events = context.get("btc_trade_events", context.get("trade_events"))
        marks = context.get("btc_marks", context.get("marks"))
        if events is None and marks is None:
            return None
        from src.research.accounting import BtcAccumulationAccounting, ProductAccountingError

        try:
            report = BtcAccumulationAccounting().evaluate(
                trade_events=events or (),
                marks=marks or (),
                initial_btc=float(context.get("initial_btc", context.get("btc_balance", 0.0))),
                initial_stablecoin=float(
                    context.get("initial_stablecoin", context.get("stablecoin_balance", 0.0))
                ),
                initial_price=(
                    float(context["initial_price"])
                    if context.get("initial_price") is not None
                    else None
                ),
                reserve_fraction=(
                    float(context["reserve_fraction"])
                    if context.get("reserve_fraction") is not None
                    else None
                ),
            )
        except (ProductAccountingError, TypeError, ValueError) as exc:
            raise ExecutorError(f"BTC accounting evidence is invalid: {exc}") from exc
        return {
            "schema": "platform.btc_accounting/v1",
            "objective_unit": report.objective_unit,
            "initial_value": report.initial_btc_nav,
            "objective_value": report.final_btc_nav,
            "benchmark_value": report.passive_btc_nav,
            "objective_excess": report.excess_btc,
            "objective_excess_fraction": (
                report.excess_btc / report.initial_btc_nav
                if report.initial_btc_nav > 0
                else 0.0
            ),
            "return_fraction": report.return_fraction,
            "fees": report.fees_btc,
            "time_outside_btc_fraction": report.time_outside_btc_fraction,
            "stablecoin_exposure_fraction": report.stablecoin_exposure_fraction,
            "missed_btc_appreciation": report.missed_btc_appreciation,
            "cycles": report.cycles,
            "regime_pnl": dict(report.regime_pnl),
            "event_receipts": [dict(item) for item in report.event_receipts],
        }
    if product_id == "active_income":
        events = context.get("futures_events", context.get("trade_events"))
        if events is None:
            if fallback_return is None:
                return None
            initial_equity = float(
                context.get("initial_cash", context.get("initial_equity", 1.0))
            )
            if not math.isfinite(initial_equity) or initial_equity <= 0.0:
                raise ExecutorError(
                    "active-income return-ledger accounting requires positive initial equity"
                )
            net_pnl = float(fallback_return.net_pnl) * initial_equity
            return {
                "schema": "platform.futures_accounting/return_ledger_v1",
                "objective_unit": "USDT",
                "initial_value": initial_equity,
                "objective_value": initial_equity + net_pnl,
                "benchmark_value": initial_equity,
                "objective_excess": net_pnl,
                "objective_excess_fraction": net_pnl / initial_equity,
                "return_fraction": net_pnl / initial_equity,
                "realised_pnl": net_pnl,
                "unrealised_pnl": 0.0,
                "fees": float(fallback_return.fees),
                "funding_pnl": float(fallback_return.funding_pnl),
                "spread_cost": 0.0,
                "slippage_cost": float(fallback_return.slippage),
                "fills": int(fallback_return.effective_observations),
                "partial_fills": 0,
                "capacity_violations": 0,
                "max_leverage": 1.0,
                "max_margin_fraction": 0.0,
                "liquidation": False,
                "effective_observations": int(fallback_return.effective_observations),
                "event_receipts": (),
                "source": "canonical_return_ledger",
            }
        from src.research.accounting import FuturesIncomeAccounting, ProductAccountingError

        try:
            report = FuturesIncomeAccounting().evaluate(
                events=events,
                initial_cash=float(context.get("initial_cash", context.get("initial_equity", 0.0))),
                leverage=float(context.get("leverage", 1.0)),
                maintenance_margin_fraction=float(context.get("maintenance_margin_fraction", 0.0)),
                max_participation_fraction=float(context.get("max_participation_fraction", 1.0)),
            )
        except (ProductAccountingError, TypeError, ValueError) as exc:
            raise ExecutorError(f"futures accounting evidence is invalid: {exc}") from exc
        return {
            "schema": "platform.futures_accounting/v1",
            "objective_unit": report.objective_unit,
            "initial_value": report.initial_equity,
            "objective_value": report.final_equity,
            "benchmark_value": report.initial_equity,
            "objective_excess": report.net_pnl,
            "objective_excess_fraction": report.return_fraction,
            "return_fraction": report.return_fraction,
            "realised_pnl": report.realised_pnl,
            "unrealised_pnl": report.unrealised_pnl,
            "fees": report.fees,
            "funding_pnl": report.funding_pnl,
            "spread_cost": report.spread_cost,
            "slippage_cost": report.slippage_cost,
            "fills": report.fills,
            "partial_fills": report.partial_fills,
            "capacity_violations": report.capacity_violations,
            "max_leverage": report.max_leverage,
            "max_margin_fraction": report.max_margin_fraction,
            "liquidation": report.liquidation,
            "effective_observations": report.effective_observations,
            "event_receipts": [dict(item) for item in report.event_receipts],
        }
    return None


def _numeric_series(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple):
        result: list[float] = []
        for item in value:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                result.append(number)
        return result
    return []


def _direction(output: Any) -> int:
    value = getattr(output, "expected_direction", 0)
    return int(value) if value in {-1, 0, 1} else 0


def _market_returns(context: Mapping[str, Any], signal_count: int) -> list[float]:
    supplied = _numeric_series(context.get("returns"))
    if supplied:
        return supplied[: max(0, signal_count - 1)]
    frame = context.get("market_frame")
    if frame is None:
        return []
    try:
        closes = _numeric_series(frame["close"])
    except (KeyError, TypeError):
        return []
    return [
        closes[index] / closes[index - 1] - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]


def _market_price_returns(context: Mapping[str, Any], signal_count: int) -> list[float]:
    """Calculate neighbour returns from immutable prices, never strategy returns."""

    frame = context.get("market_frame")
    if frame is None:
        return []
    try:
        closes = _numeric_series(frame["close"])
    except (KeyError, TypeError):
        return []
    returns = [
        closes[index] / closes[index - 1] - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]
    return returns[: max(0, signal_count - 1)]


def _window_sums(values: list[float], windows: int) -> list[float]:
    if len(values) < windows:
        return []
    size = max(1, len(values) // windows)
    return [sum(values[index * size : (index + 1) * size]) for index in range(windows)]


def _trial_sharpes(
    context: Mapping[str, Any], parameter_stability: Mapping[str, Any], base_returns: list[float]
) -> list[float]:
    from src.metrics import sharpe_ratio

    raw = context.get("trial_returns")
    candidates: list[Any] = []
    if isinstance(raw, list | tuple):
        candidates.extend(raw)
    candidates.extend(
        item.get("window_returns")
        for item in parameter_stability.get("results", ())
        if isinstance(item, Mapping)
    )
    series = [base_returns, *candidates]
    return [sharpe_ratio(values) for values in series if _numeric_series(values)]


def _parameter_stability(
    candidate: Candidate, context: Mapping[str, Any], base_returns: list[float]
) -> dict[str, Any]:
    parameters = candidate.definition.signal_model.get("parameters", {})
    declared_tunable = context.get("tunable_parameters")
    if isinstance(declared_tunable, list | tuple):
        tunable = tuple(str(name) for name in declared_tunable)
    elif isinstance(parameters, Mapping):
        tunable = tuple(
            str(name)
            for name, value in parameters.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        )
    else:
        tunable = ()
    if not tunable:
        return {
            "status": "not_applicable",
            "passed": True,
            "reason": "no_tunable_parameters",
            "neighbours_tested": 0,
            "results": [],
            "base_window_returns": _window_sums(base_returns, 3),
        }
    raw = context.get("parameter_neighbour_returns", context.get("neighbour_returns", {}))
    neighbours: dict[str, list[float]] = {}
    if isinstance(raw, Mapping):
        for name, values in raw.items():
            numeric = _numeric_series(values)
            if numeric:
                neighbours[str(name)] = numeric
    if not neighbours:
        frame = context.get("market_frame")
        strategy_name = candidate.definition.signal_model.get("registered_strategy")
        if isinstance(parameters, Mapping) and frame is not None and isinstance(strategy_name, str):
            try:
                for name, value in parameters.items():
                    if isinstance(value, bool) or not isinstance(value, int | float):
                        continue
                    step = 1 if isinstance(value, int) else max(abs(float(value)) * 0.1, 0.01)
                    for direction in (-1, 1):
                        varied = dict(parameters)
                        varied[str(name)] = value + direction * step
                        if isinstance(value, int):
                            varied[str(name)] = max(1, int(varied[str(name)]))
                        varied_behaviour = RegisteredStrategyBehaviour(
                            name=strategy_name,
                            parameters=varied,
                            source_hash=candidate.definition.source_hash,
                        )
                        signals = varied_behaviour.generate_signals(frame)
                        numeric_signals = _numeric_series(signals)
                        returns = _market_price_returns(context, len(numeric_signals))
                        aligned = min(len(returns), max(0, len(numeric_signals) - 1))
                        neighbours[f"{name}:{direction:+d}"] = [
                            numeric_signals[index] * returns[index] for index in range(aligned)
                        ]
            except (ExecutorError, KeyError, TypeError, ValueError, StrategyBehaviourError):
                neighbours = {}
    results = []
    base_total = sum(base_returns)
    for name, values in sorted(neighbours.items()):
        comparable = values[: len(base_returns)]
        result = {
            "name": name,
            "run_id": canonical_hash(
                {
                    "kind": "parameter_neighbour_backtest/v1",
                    "candidate_id": candidate.candidate_id,
                    "dataset_snapshot_ids": list(context.get("dataset_snapshot_ids", ())),
                    "neighbour": name,
                    "returns": comparable,
                }
            ),
            "observations": len(comparable),
            "return": sum(comparable),
            "passed": bool(comparable) and sum(comparable) >= base_total * 0.5,
            "window_returns": _window_sums(comparable, 3),
            "input_hash": canonical_hash(
                {
                    "candidate_id": candidate.candidate_id,
                    "dataset_snapshot_ids": list(context.get("dataset_snapshot_ids", ())),
                    "neighbour": name,
                    "returns": comparable,
                }
            ),
        }
        results.append(result)
    return {
        "status": "pass" if results and all(item["passed"] for item in results) else "fail",
        "passed": bool(base_returns) and bool(results) and all(item["passed"] for item in results),
        "neighbours_tested": len(results),
        "results": results,
        "base_window_returns": _window_sums(base_returns, 3),
    }


def _cross_symbol_stability(
    context: Mapping[str, Any], base_returns: list[float]
) -> dict[str, Any]:
    raw = context.get("symbol_returns", context.get("cross_symbol_returns", {}))
    per_symbol: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for symbol, values in raw.items():
            numeric = _numeric_series(values)
            if numeric:
                per_symbol[str(symbol)] = {
                    "run_id": canonical_hash(
                        {
                            "kind": "cross_symbol_backtest/v1",
                            "symbol": str(symbol),
                            "returns": numeric,
                            "dataset_snapshot_ids": list(context.get("dataset_snapshot_ids", ())),
                        }
                    ),
                    "observations": len(numeric),
                    "return": sum(numeric),
                    "passed": len(numeric) >= 2 and sum(numeric) >= 0.0,
                    "input_hash": canonical_hash(
                        {
                            "symbol": str(symbol),
                            "returns": numeric,
                            "dataset_snapshot_ids": list(context.get("dataset_snapshot_ids", ())),
                        }
                    ),
                }
    if not per_symbol:
        scope = tuple(
            str(item)
            for item in context.get("instrument_scope", context.get("expected_symbols", ()))
            if item
        )
        if len(scope) == 1:
            return {
                "status": "not_applicable",
                "passed": True,
                "symbols": 0,
                "per_symbol": {},
                "reason": "single_instrument",
            }
        return {
            "status": "unavailable",
            "passed": False,
            "symbols": 0,
            "per_symbol": {},
            "reason": "missing_symbol_returns",
        }
    scope = tuple(
        str(item)
        for item in context.get("instrument_scope", context.get("expected_symbols", ()))
        if item
    )
    missing_symbols = sorted(set(scope) - set(per_symbol)) if len(scope) > 1 else []
    return {
        "status": (
            "pass"
            if all(item["passed"] for item in per_symbol.values()) and not missing_symbols
            else "fail"
        ),
        "passed": all(item["passed"] for item in per_symbol.values()) and not missing_symbols,
        "symbols": len(per_symbol),
        "per_symbol": per_symbol,
        "missing_symbols": missing_symbols,
    }


def _portfolio_overlap(
    context: Mapping[str, Any], candidate_returns: list[float]
) -> dict[str, Any]:
    raw = context.get("active_strategy_returns", {})
    if not isinstance(raw, Mapping) or not raw:
        return {
            "status": "not_applicable",
            "passed": True,
            "maximum_correlation": None,
            "threshold": float(context.get("maximum_portfolio_correlation", 0.8)),
            "active_strategy_count": 0,
            "comparisons": [],
            "reason": "no_active_strategy",
        }
    candidate = _numeric_series(context.get("candidate_returns", candidate_returns))
    comparisons: list[dict[str, Any]] = []
    maximum = 0.0
    undefined = False
    for name, values in raw.items():
        other = _numeric_series(values)
        correlation = _correlation(candidate, other)
        if correlation is None:
            undefined = True
        else:
            maximum = max(maximum, abs(correlation))
        comparisons.append(
            {
                "strategy": str(name),
                "run_id": canonical_hash(
                    {
                        "kind": "portfolio_overlap/v1",
                        "candidate_returns": candidate,
                        "active_strategy": str(name),
                        "active_returns": other,
                        "dataset_snapshot_ids": list(context.get("dataset_snapshot_ids", ())),
                    }
                ),
                "correlation": correlation,
                "observations": min(len(candidate), len(other)),
                "input_hash": canonical_hash(
                    {
                        "candidate_returns": candidate,
                        "active_strategy": str(name),
                        "active_returns": other,
                        "dataset_snapshot_ids": list(context.get("dataset_snapshot_ids", ())),
                    }
                ),
            }
        )
    threshold = float(context.get("maximum_portfolio_correlation", 0.8))
    return {
        "status": "pass"
        if bool(comparisons) and not undefined and maximum <= threshold
        else "fail",
        "passed": bool(comparisons) and not undefined and maximum <= threshold,
        "maximum_correlation": None if undefined else maximum,
        "threshold": threshold,
        "active_strategy_count": len(comparisons),
        "comparisons": comparisons,
    }


def _purged_walk_forward(values: list[float], context: Mapping[str, Any]) -> dict[str, Any]:
    windows = max(3, int(context.get("walk_forward_windows", 3)))
    if len(values) < windows:
        return {"passed": False, "window_count": 0, "pass_fraction": 0.0, "per_window": []}
    purge = max(0, int(context.get("purge_rows", 1)))
    embargo = max(0, int(context.get("embargo_rows", 1)))
    size = max(1, len(values) // windows)
    per_window: list[dict[str, Any]] = []
    for index in range(windows):
        start, end = index * size, (index + 1) * size if index < windows - 1 else len(values)
        left = start + (purge if index else 0)
        right = end - (embargo if index < windows - 1 else 0)
        sample = values[max(start, left) : max(max(start, left), right)]
        window_input = {
            "window": index,
            "start": start,
            "end": end,
            "purge_rows": purge,
            "embargo_rows": embargo,
            "values": sample,
        }
        per_window.append(
            {
                "window": index,
                "return": sum(sample),
                "observations": len(sample),
                "passed": bool(sample) and sum(sample) >= 0.0,
                "run_id": canonical_hash({"kind": "walk_forward_window/v1", **window_input}),
                "input_hash": canonical_hash(window_input),
            }
        )
    passed = sum(1 for item in per_window if item["passed"])
    return {
        "passed": len(per_window) >= windows and passed / len(per_window) >= 0.5,
        "window_count": len(per_window),
        "pass_fraction": passed / len(per_window) if per_window else 0.0,
        "purged_rows": purge,
        "embargo_rows": embargo,
        "per_window": per_window,
        "input_hash": canonical_hash(
            {
                "kind": "purged_embargoed_walk_forward/v1",
                "values": values,
                "windows": windows,
                "purge_rows": purge,
                "embargo_rows": embargo,
            }
        ),
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    pairs = tuple(zip(left, right, strict=False))
    if len(pairs) < 2:
        return None
    left_mean = statistics.fmean(item[0] for item in pairs)
    right_mean = statistics.fmean(item[1] for item in pairs)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in pairs)
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a, _ in pairs) * sum((b - right_mean) ** 2 for _, b in pairs)
    )
    return numerator / denominator if denominator else None


def _skew_kurtosis(values: list[float]) -> tuple[float, float]:
    if len(values) < 3:
        return 0.0, 3.0
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values)
    if deviation == 0:
        return 0.0, 3.0
    standard = [(value - mean) / deviation for value in values]
    return statistics.fmean(value**3 for value in standard), statistics.fmean(
        value**4 for value in standard
    )


def _drift_checks(context: Mapping[str, Any]) -> dict[str, Any]:
    measurements = context.get("drift_measurements")
    if not isinstance(measurements, Mapping):
        return {
            "passed": False,
            "source": "runtime_drift_measurements",
            "reason": "missing_drift_measurements",
        }
    try:
        execution = float(measurements["execution"])
        model = float(measurements["model"])
        execution_limit = float(measurements["maximum_execution"])
        model_limit = float(measurements["maximum_model"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "passed": False,
            "source": "runtime_drift_measurements",
            "reason": f"invalid_drift_measurements:{type(exc).__name__}",
        }
    return {
        "passed": execution <= execution_limit and model <= model_limit,
        "execution": execution,
        "model": model,
        "maximum_execution": execution_limit,
        "maximum_model": model_limit,
        "source": "runtime_drift_measurements",
        "input_hash": canonical_hash(
            {
                "execution": execution,
                "model": model,
                "maximum_execution": execution_limit,
                "maximum_model": model_limit,
            }
        ),
    }


def _maximum_drawdown(values: list[float]) -> float:
    equity = peak = 1.0
    maximum = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak if peak else 0.0)
    return maximum


def _negative_control_evidence(
    *,
    signals: list[float],
    returns: list[float],
    candidate_return: float,
    control_returns: Any = None,
    controls: tuple[str, ...] = (),
) -> dict[str, dict[str, float | int | bool | str | None]]:
    aligned = min(len(signals), len(returns))
    signals, returns = signals[:aligned], returns[:aligned]
    supplied = control_returns if isinstance(control_returns, Mapping) else {}
    results: dict[str, dict[str, float | int | bool | str | None]] = {}
    for name in controls:
        numeric = _numeric_series(supplied.get(name))
        comparable = numeric[:aligned]
        control_return = sum(comparable) if comparable else None
        results[name] = {
            "passed": bool(comparable) and candidate_return >= float(control_return),
            "observations": len(comparable),
            "control_return": control_return,
            "input_hash": canonical_hash({"control": name, "returns": comparable})
            if comparable
            else None,
        }
    return results


def _negative_control_names(candidate: Candidate, context: Mapping[str, Any]) -> tuple[str, ...]:
    raw = context.get("negative_controls")
    if raw is None:
        policy = candidate.definition.validation_policy
        raw = policy.get("negative_controls") if isinstance(policy, Mapping) else None
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(dict.fromkeys(str(name) for name in raw if str(name)))


def _build_registered_context(
    _candidate: Candidate, context: Mapping[str, Any]
) -> Mapping[str, Any]:
    raw = context.get("market_frame")
    if raw is None:
        raise ExecutorError("registered Python context has no immutable market_frame")
    if hasattr(raw, "columns") and hasattr(raw, "index"):
        frame = raw
    elif isinstance(raw, list | tuple) and raw and all(isinstance(item, Mapping) for item in raw):
        import pandas as pd

        frame = pd.DataFrame(tuple(dict(item) for item in raw))
    else:
        raise ExecutorError("registered Python market_frame schema is invalid")
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(str(column) for column in frame.columns):
        raise ExecutorError("registered Python market_frame has no canonical OHLCV columns")
    return {**dict(context), "market_frame": frame}


def _build_dsl_context(_candidate: Candidate, context: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = context.get("feature_rows")
    if (
        not isinstance(rows, list | tuple)
        or not rows
        or not all(isinstance(item, Mapping) for item in rows)
    ):
        raise ExecutorError("generated DSL context has no immutable feature rows")
    return {**dict(context), "feature_rows": tuple(dict(item) for item in rows)}


def _build_ml_context(candidate: Candidate, context: Mapping[str, Any]) -> Mapping[str, Any]:
    from src.strategies.frozen_model import FrozenSafeModel

    model = context.get("frozen_model")
    model_ids = context.get("model_artefact_ids")
    if not isinstance(model_ids, list | tuple) or len(model_ids) != 1:
        raise ExecutorError("machine-learning context needs one immutable model artefact")
    if model is None:
        path = context.get("model_artefact_path")
        model_hash = context.get("model_artefact_hash")
        manifest_hash = context.get("feature_manifest_hash")
        if (
            not isinstance(path, str)
            or not isinstance(model_hash, str)
            or not isinstance(manifest_hash, str)
        ):
            raise ExecutorError("machine-learning context has no frozen model identities")
        if model_hash != model_ids[0]:
            raise ExecutorError("machine-learning model hash differs from its dataset identity")
        model = FrozenSafeModel.load(
            Path(path),
            expected_artefact_hash=model_hash,
            expected_feature_manifest_hash=manifest_hash,
        )
    vector = context.get("feature_vector")
    if not isinstance(vector, Mapping):
        raise ExecutorError("machine-learning context has no ordered feature vector")
    expected = candidate.definition.signal_model.get("feature_names")
    if isinstance(expected, list | tuple) and tuple(vector) != tuple(
        str(item) for item in expected
    ):
        raise ExecutorError("machine-learning feature vector differs from its definition")
    return {**dict(context), "frozen_model": model, "feature_vector": dict(vector)}


def _build_semantic_context(candidate: Candidate, context: Mapping[str, Any]) -> Mapping[str, Any]:
    name = str(
        candidate.definition.signal_model.get("semantic_strategy") or candidate.definition.identity
    )
    registration = SEMANTIC_STRATEGIES.get(name)
    if candidate.definition.source_type is StrategySourceType.MICROSTRUCTURE and not context.get(
        "event_data_segment_ids"
    ):
        raise ExecutorError("microstructure context needs immutable event-data segments")
    raw = context.get("semantic_input")
    if isinstance(raw, registration.input_type):
        value = raw
    elif isinstance(raw, Mapping):
        if registration.input_type.__name__ == "ForecastCollection":
            from src.domain.forecasts import AlphaForecast

            forecasts = raw.get("forecasts")
            if not isinstance(forecasts, list | tuple):
                raise ExecutorError("ensemble context has no alpha forecasts")
            value = registration.input_type(
                tuple(
                    item if isinstance(item, AlphaForecast) else AlphaForecast(**dict(item))
                    for item in forecasts
                )
            )
        else:
            value = registration.input_type(**dict(raw))
    else:
        raise ExecutorError(f"{name} context has no typed semantic input")
    return {**dict(context), "semantic_input": value}


def execute_registered_python(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    frame = context.get("market_frame")
    if frame is None:
        raise ExecutorError("registered Python execution requires a canonical market_frame")
    try:
        behaviour = RegisteredStrategyBehaviour.from_definition(candidate.definition)
        signals = behaviour.generate_signals(frame)
    except StrategyBehaviourError as exc:
        raise ExecutorError(str(exc)) from exc
    return _measured_result(
        candidate,
        {**context, "behaviour_hash": behaviour.behaviour_hash},
        signals,
    )


def execute_generated_dsl(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    rows = context.get("feature_rows")
    rule = candidate.definition.signal_model.get("rule")
    if not isinstance(rows, list | tuple) or not isinstance(rule, Mapping):
        raise ExecutorError("DSL execution requires canonical feature_rows and a typed rule")
    feature = str(rule.get("feature", ""))
    operator = str(rule.get("operator", ""))
    raw_threshold = rule.get("threshold")
    if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, int | float):
        raise ExecutorError("DSL rule threshold must be numeric")
    threshold = float(raw_threshold)
    operations = {
        "gt": lambda value: value > threshold,
        "ge": lambda value: value >= threshold,
        "lt": lambda value: value < threshold,
        "le": lambda value: value <= threshold,
    }
    if operator not in operations:
        raise ExecutorError("DSL rule operator is unsupported")
    direction = str(rule.get("direction") or "long")
    if direction not in {"long", "short", "signed", "market_neutral", "hedged"}:
        raise ExecutorError("DSL rule direction is unsupported")
    sign = -1 if direction == "short" else 1
    signals = tuple(sign if operations[operator](float(row[feature])) else 0 for row in rows)
    return _measured_result(candidate, context, signals)


def execute_machine_learning(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    model_hash = context.get("model_artefact_hash")
    manifest_hash = context.get("feature_manifest_hash")
    if not model_hash or not manifest_hash:
        raise ExecutorError("machine-learning execution needs a frozen model and feature manifest")
    model = context.get("frozen_model")
    features = context.get("feature_vector")
    if (
        model is None
        or not callable(getattr(model, "evaluate", None))
        or not isinstance(features, Mapping)
    ):
        raise ExecutorError("machine-learning execution needs loaded frozen model inputs")
    return _measured_result(candidate, context, model.evaluate(features))


def execute_cross_sectional(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    return _execute_semantic(candidate, context)


def execute_relative_value(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    return _execute_semantic(candidate, context)


def execute_microstructure(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    return _execute_semantic(candidate, context)


def execute_ensemble(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    return _execute_semantic(candidate, context)


def execute_agent_generated_python(
    candidate: Candidate, context: Mapping[str, Any]
) -> ExecutionResult:
    if not context.get("sandbox_receipt"):
        raise ExecutorError("agent-generated Python needs a verified sandbox receipt")
    return execute_registered_python(candidate, context)


def _execute_semantic(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    name = str(
        candidate.definition.signal_model.get("semantic_strategy") or candidate.definition.identity
    )
    semantic_input = context.get("semantic_input")
    if semantic_input is None:
        raise ExecutorError("semantic execution requires a typed canonical input")
    output = SEMANTIC_STRATEGIES.get(name).evaluate(semantic_input)
    return _measured_result(candidate, context, output)


def execution_receipt(
    *,
    candidate: Candidate,
    dataset_snapshot_ids: tuple[str, ...],
    executor_version: str,
    evidence_policy_hash: str | None = None,
    behaviour_hash: str | None = None,
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate.candidate_id,
        "dataset_snapshot_ids": list(dataset_snapshot_ids),
        "executor_version": executor_version,
    }
    if evidence_policy_hash is not None:
        payload["evidence_policy_hash"] = evidence_policy_hash
    if behaviour_hash is not None:
        payload["behaviour_hash"] = behaviour_hash
    return {**payload, "input_hash": canonical_hash(payload)}
