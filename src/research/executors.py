"""Provider executors for canonical candidate evaluation.

The registry is the only place where a strategy source becomes executable.
Every executor returns measured evidence and a receipt describing the exact
inputs it consumed.  A worker cannot manufacture a validation result by
putting metrics in a queue payload.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.domain._codec import canonical_hash, json_value
from src.domain.strategies import StrategySourceType
from src.research.controls import control_identity, derive_control_returns
from src.research.coordinator import Candidate, CandidateEvaluationView
from src.strategies.behaviour import (
    RegisteredStrategyBehaviour,
    StrategyBehaviourError,
    TypedRuleBehaviour,
    behaviour_hash_for_definition,
)
from src.strategies.semantic import (
    SEMANTIC_STRATEGIES,
    SemanticEvaluationError,
    semantic_forecast_from_output,
    semantic_input_from_features,
    semantic_signal,
    semantic_strategy_name,
)


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
ContextCandidate = Candidate | CandidateEvaluationView
ContextBuilder = Callable[[ContextCandidate, Mapping[str, Any]], Mapping[str, Any]]


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

    def build(self, candidate: ContextCandidate, context: Mapping[str, Any]) -> Mapping[str, Any]:
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


def _pbo_measurements(
    context: Mapping[str, Any],
    parameter_stability: Mapping[str, Any],
    window_returns: list[float],
) -> tuple[float | dict[str, Any], list[list[float]]]:
    strategy_window_returns = context.get("strategy_window_returns")
    if isinstance(strategy_window_returns, list | tuple):
        pbo_matrix = [
            _numeric_series(row) for row in strategy_window_returns if _numeric_series(row)
        ]
    else:
        pbo_matrix = [window_returns]
        results = parameter_stability.get("results", ())
        if isinstance(results, list | tuple):
            pbo_matrix.extend(
                window
                for item in results
                if isinstance(item, Mapping)
                and (window := _numeric_series(item.get("window_returns")))
            )
        trial_returns = context.get("trial_returns")
        if isinstance(trial_returns, list | tuple):
            pbo_matrix.extend(window for item in trial_returns if (window := _numeric_series(item)))
    if len(pbo_matrix) < 2:
        pbo_matrix = []
    width = min((len(row) for row in pbo_matrix), default=0)
    pbo_matrix = [row[:width] for row in pbo_matrix if width >= 2]
    if len(pbo_matrix) < 2:
        return (
            {
                "status": "not_applicable",
                "passed": True,
                "reason": "no_valid_configuration_cohort",
                "cohort_size": len(pbo_matrix),
            },
            pbo_matrix,
        )
    from src.metrics import probability_backtest_overfitting

    return probability_backtest_overfitting(pbo_matrix), pbo_matrix


def _trial_statistics(
    context: Mapping[str, Any],
    parameter_stability: Mapping[str, Any],
    analysis_returns: list[float],
) -> tuple[list[float], int, float, float]:
    from src.metrics import deflated_sharpe_ratio, sharpe_ratio

    trial_sharpes = _trial_sharpes(context, parameter_stability, analysis_returns)
    trial_count = max(
        1,
        int(
            context.get(
                "trial_count",
                context.get(
                    "global_trial_count",
                    len(trial_sharpes) or parameter_stability["neighbours_tested"] + 1,
                ),
            )
        ),
    )
    trial_sharpe_std = float(context.get("trial_sharpe_std", 0.0))
    if "trial_sharpe_std" not in context and len(trial_sharpes) >= 2:
        trial_sharpe_std = statistics.pstdev(trial_sharpes)
    skew, kurtosis = _skew_kurtosis(analysis_returns)
    dsr = deflated_sharpe_ratio(
        sharpe_ratio(analysis_returns),
        n_trials=trial_count,
        skew=skew,
        kurt=kurtosis,
        n_obs=len(analysis_returns),
        sr_std_trials=trial_sharpe_std,
    )
    return trial_sharpes, trial_count, trial_sharpe_std, dsr


def _measured_result(
    candidate: Candidate, context: Mapping[str, Any], output: Any
) -> ExecutionResult:
    from src.metrics import bootstrap_sharpe_ci

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
        fee_rate = _nonnegative_rate(context.get("fee_bps", 1.0), field="fee_bps") / 10_000.0
        slippage_rate = (
            _nonnegative_rate(context.get("slippage_bps", 1.0), field="slippage_bps") / 10_000.0
        )
        funding_rate = _finite_rate(context.get("funding_rate", 0.0), field="funding_rate")
        funding_rates = context.get("funding_period_rates", context.get("funding_rates"))
        return_ledger = PositionReturnLedger(
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            funding_rate=funding_rate,
        )
        return_report = return_ledger.measure(signals, returns, funding_rates=funding_rates)
    except (ReturnLedgerError, TypeError, ValueError) as exc:
        raise ExecutorError(f"position return ledger input is invalid: {exc}") from exc
    aligned = return_report.effective_observations
    gross = list(return_report.gross_returns)
    net = list(return_report.net_returns)
    analysis_returns = net or gross
    bootstrap_values = analysis_returns
    fees = return_report.fees
    slippage = return_report.slippage
    funding_pnl = return_report.funding_pnl
    funding = return_report.funding_cost
    turnover = return_report.turnover
    net_return = return_report.net_pnl
    accounting = _product_accounting(context, fallback_return=return_report)
    if accounting is not None:
        net_return = float(accounting["return_fraction"])
    window_returns = _window_sums(analysis_returns, _evaluation_window_count(context))
    negative_controls = _negative_control_evidence(
        signals=signals[:aligned],
        returns=returns[:aligned],
        candidate_return=net_return,
        control_returns=context.get("negative_control_returns"),
        controls=_negative_control_names(candidate, context),
        instrument_scope=tuple(str(item) for item in context.get("instrument_scope", ())),
        seed_material={
            "candidate_id": candidate.candidate_id,
            "dataset_snapshot_ids": snapshots,
        },
    )
    delayed_report = (
        return_ledger.measure([0.0, *signals[:-1]], returns, funding_rates=funding_rates)
        if aligned > 1
        else None
    )
    delayed_net = list(delayed_report.net_returns) if delayed_report is not None else []
    cost_stress = _stress_report(
        signals,
        returns,
        fee_rate=fee_rate * 2.0,
        slippage_rate=slippage_rate * 2.0,
        funding_rate=funding_rate,
        funding_rates=funding_rates,
    )
    adverse_fill_stress = _stress_report(
        signals,
        returns,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate * 2.0,
        funding_rate=funding_rate,
        funding_rates=funding_rates,
    )
    funding_stress = _stress_report(
        signals,
        returns,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        funding_rate=funding_rate * 2.0,
        funding_rates=_scaled_rates(funding_rates, 2.0),
    )
    missing_data_returns = [
        value for index, value in enumerate(analysis_returns) if (index + 1) % 20 != 0
    ]
    parameter_stability = _parameter_stability(candidate, context, bootstrap_values)
    cross_symbol_stability = _cross_symbol_stability(context, bootstrap_values)
    portfolio_overlap = _portfolio_overlap(context, analysis_returns)
    walk_forward = _purged_walk_forward(analysis_returns, context)
    bootstrap_low, bootstrap_high = bootstrap_sharpe_ci(
        bootstrap_values,
        n_boot=int(context.get("bootstrap_iterations", 1_000)),
        random_state=int(candidate.candidate_id[7:15], 16),
    )
    pbo, pbo_matrix = _pbo_measurements(context, parameter_stability, window_returns)
    trial_sharpes, trial_count, trial_sharpe_std, dsr = _trial_statistics(
        context, parameter_stability, analysis_returns
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
    family = str(candidate.definition.family)
    evidence_type = str(candidate.definition.validation_policy.get("evidence_type") or "")
    data_integrity = {
        "passed": bool(context.get("data_integrity_valid", True))
        and all(value.startswith("sha256:") and len(value) == 71 for value in snapshots),
        "dataset_snapshot_ids": list(snapshots),
        "input_hash": canonical_hash(
            {
                "dataset_snapshot_ids": list(snapshots),
                "feature_manifest_id": expected_feature_manifest,
                "cost_model_id": expected_cost_model,
                "parameter_set_id": context.get("parameter_set_id"),
            }
        ),
    }
    semantic_parity = {
        "passed": bool(
            context.get("semantic_parity_valid", context.get("parity_receipt") is not None)
        ),
        "behaviour_hash": str(
            context.get("behaviour_hash") or behaviour_hash_for_definition(candidate.definition)
        ),
        "semantic_identity": candidate.definition.definition_hash,
        "parity_receipt": context.get("parity_receipt"),
    }
    realistic_costs = {
        "passed": bool(context.get("cost_model_valid", True)),
        "fee_bps": float(context.get("fee_bps", 0.0)),
        "slippage_bps": float(context.get("slippage_bps", 0.0)),
        "funding_rate": float(context.get("funding_rate", 0.0)),
        "cost_model_id": expected_cost_model,
    }
    family_evidence = {
        "passed": bool(context.get("family_evidence_valid", True)),
        "family": family,
        "evidence_type": evidence_type,
        "instrument_scope": list(context.get("instrument_scope", ())),
    }
    cycles = int(
        context.get(
            "cycles",
            accounting.get("cycles", 0) if accounting is not None else 0,
        )
    )
    randomiser = random.Random(int(candidate.candidate_id[7:23], 16))
    monte_carlo_drawdowns = []
    monte_carlo_tail_losses = []
    for _ in range(250):
        permuted = list(bootstrap_values)
        randomiser.shuffle(permuted)
        monte_carlo_drawdowns.append(_maximum_drawdown(permuted))
        monte_carlo_tail_losses.append(_tail_loss(permuted))
    declared_symbols = candidate.definition.universe.get("symbols")
    declared_instrument_ids = candidate.definition.universe.get("instrument_ids")
    declared_values = tuple(
        str(item)
        for values in (declared_symbols, declared_instrument_ids)
        if isinstance(values, list | tuple)
        for item in values
    )
    scope = tuple(str(item) for item in context.get("instrument_scope", ()))
    predeclared = bool(declared_values)
    if scope:
        predeclared = predeclared and all(
            observed in declared_values
            or any(
                observed.endswith(f":{declared_symbol}")
                or observed.endswith(f":{declared_symbol}:USDT")
                for declared_symbol in declared_values
            )
            for observed in scope
        )
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
        "data_integrity": data_integrity,
        "semantic_parity": semantic_parity,
        "realistic_costs": realistic_costs,
        "family_evidence": family_evidence,
        "signal_frequency": active / observations,
        "turnover": turnover,
        "return_ledger": {
            "gross_pnl": return_report.gross_pnl,
            "net_pnl": return_report.net_pnl,
            "net_returns": list(return_report.net_returns),
            "maximum_drawdown": return_report.maximum_drawdown,
        },
        "product_accounting": accounting,
        "objective_unit": accounting.get("objective_unit") if accounting else None,
        "objective_value": accounting.get("objective_value") if accounting else None,
        "benchmark_value": accounting.get("benchmark_value") if accounting else None,
        "objective_excess": accounting.get("objective_excess") if accounting else None,
        "objective_excess_fraction": (
            accounting.get("objective_excess_fraction") if accounting else None
        ),
        "chronological": not bool(context.get("lookahead_detected", False)),
        "cost_adjusted_return": net_return,
        "fees": fees,
        "slippage": slippage,
        "funding": funding,
        "funding_pnl": funding_pnl,
        "regime_breakdown": {"passed": bool(analysis_returns), "regimes": {"all": net_return}},
        "parameter_stability": parameter_stability,
        "sample_evidence": {
            "passed": aligned >= 3,
            "observations": aligned,
            "closed_trades": int(
                context.get(
                    "closed_trades",
                    sum(
                        1
                        for index in range(1, len(signals))
                        if abs(signals[index - 1]) > 1e-12 and signals[index] != signals[index - 1]
                    ),
                )
            ),
            "effective_independent_episodes": int(
                context.get("effective_independent_episodes", len(window_returns))
            ),
            "trading_days": int(context.get("trading_days", context.get("evidence_days", 0))),
            "calendar_days": float(context.get("calendar_days", 0.0)),
            "run_id": canonical_hash(
                {
                    "kind": "sample_evidence/v1",
                    "candidate_id": candidate.candidate_id,
                    "dataset_snapshot_ids": list(snapshots),
                    "returns": analysis_returns,
                }
            ),
            "input_hash": canonical_hash(
                {
                    "dataset_snapshot_ids": list(snapshots),
                    "returns": analysis_returns,
                }
            ),
        },
        "cross_symbol_stability": cross_symbol_stability,
        "universe_evidence": {
            "passed": predeclared,
            "predeclared": predeclared,
            "declared_symbols": list(declared_symbols or ()),
            "declared_instrument_ids": list(declared_instrument_ids or ()),
            "observed_symbols": list(scope),
        },
        "portfolio_overlap": portfolio_overlap,
        "walk_forward": walk_forward,
        "purged": int(walk_forward.get("purged_rows", 0)) > 0,
        "embargo": int(walk_forward.get("embargo_rows", 0)),
        "cost_stress": {
            "passed": _stress_passes(cost_stress.net_pnl, net_return, context),
            "multiplier": 2.0,
            "cost_adjusted_return": cost_stress.net_pnl,
            "degradation_fraction": _degradation(net_return, cost_stress.net_pnl),
        },
        "delay_stress": {
            "passed": bool(delayed_net) and _stress_passes(sum(delayed_net), net_return, context),
            "delay_bars": 1,
            "cost_adjusted_return": sum(delayed_net),
            "degradation_fraction": _degradation(net_return, sum(delayed_net)),
        },
        "adverse_fill_stress": {
            "passed": _stress_passes(adverse_fill_stress.net_pnl, net_return, context),
            "multiplier": 2.0,
            "cost_adjusted_return": adverse_fill_stress.net_pnl,
            "degradation_fraction": _degradation(net_return, adverse_fill_stress.net_pnl),
        },
        "missing_data_stress": {
            "passed": bool(missing_data_returns)
            and _stress_passes(sum(missing_data_returns), net_return, context),
            "removed_fraction": 1.0 - len(missing_data_returns) / max(1, len(analysis_returns)),
            "cost_adjusted_return": sum(missing_data_returns),
            "degradation_fraction": _degradation(net_return, sum(missing_data_returns)),
        },
        "funding_stress": {
            "passed": _stress_passes(funding_stress.net_pnl, net_return, context),
            "multiplier": 2.0,
            "cost_adjusted_return": funding_stress.net_pnl,
            "degradation_fraction": _degradation(net_return, funding_stress.net_pnl),
        },
        "monte_carlo_trade_order": {
            "passed": bool(monte_carlo_drawdowns)
            and net_return >= 0
            and max(monte_carlo_drawdowns, default=0.0)
            <= float(context.get("maximum_monte_carlo_drawdown", 1.0))
            and max(monte_carlo_tail_losses, default=0.0)
            <= float(context.get("maximum_monte_carlo_tail_loss", 1.0)),
            "iterations": len(monte_carlo_drawdowns),
            "maximum_drawdown": max(monte_carlo_drawdowns, default=0.0),
            "tail_loss": max(monte_carlo_tail_losses, default=0.0),
            "tail_quantile": 0.05,
        },
        "bootstrap_confidence": {
            "passed": len(bootstrap_values)
            >= int(context.get("minimum_bootstrap_observations", 30))
            and bootstrap_low >= 0.0,
            "lower_bound": bootstrap_low,
            "upper_bound": bootstrap_high,
            "observations": len(bootstrap_values),
            "iterations": int(context.get("bootstrap_iterations", 1_000)),
            "method": str(context.get("bootstrap_method") or "moving_block_bootstrap_v1"),
            "run_id": canonical_hash(
                {
                    "kind": "bootstrap_confidence/v1",
                    "candidate_id": candidate.candidate_id,
                    "dataset_snapshot_ids": list(snapshots),
                    "returns": bootstrap_values,
                    "iterations": int(context.get("bootstrap_iterations", 1_000)),
                }
            ),
            "input_hash": canonical_hash(
                {
                    "dataset_snapshot_ids": list(snapshots),
                    "returns": bootstrap_values,
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
            "passed": bool(return_report.net_returns)
            and return_report.maximum_drawdown <= float(context.get("maximum_drawdown", 1.0)),
            "maximum_drawdown": return_report.maximum_drawdown,
            "tail_loss": _tail_loss(bootstrap_values),
            "tail_quantile": 0.05,
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
        "forward_duration": {
            "passed": True,
            "calendar_days": float(context.get("calendar_days", 0.0)),
            "trading_days": int(context.get("trading_days", context.get("evidence_days", 0))),
            "cycles": cycles,
        },
        "output_hash": output_hash,
        "observations": observations,
        "behaviour_hash": context.get("behaviour_hash"),
        "parity_receipt": context.get("parity_receipt"),
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
        return _btc_product_accounting(context)
    if product_id == "active_income":
        return _futures_product_accounting(context, fallback_return=fallback_return)
    return None


def _btc_product_accounting(context: Mapping[str, Any]) -> dict[str, Any] | None:
    events = context.get("btc_trade_events", context.get("trade_events"))
    marks = context.get("btc_marks", context.get("marks"))
    derived = events is None and marks is None
    if derived:
        events, marks = _derived_btc_accounting_inputs(context)
    if events is None and marks is None:
        return None
    from src.research.accounting import BtcResearchAccounting, ProductAccountingError

    try:
        reserve_fraction, tactical_fraction = _btc_fraction_inputs(context, derived=derived)
        report = BtcResearchAccounting().evaluate(
            trade_events=events or (),
            marks=marks or (),
            initial_btc=float(
                context.get("initial_btc", context.get("btc_balance", 1.0 if derived else 0.0))
            ),
            initial_stablecoin=float(
                context.get("initial_stablecoin", context.get("stablecoin_balance", 0.0))
            ),
            initial_price=(
                float(context["initial_price"])
                if context.get("initial_price") is not None
                else None
            ),
            reserve_fraction=reserve_fraction,
            max_tactical_fraction=tactical_fraction,
            external_events=context.get("btc_external_events", ()),
        )
    except (ProductAccountingError, TypeError, ValueError) as exc:
        raise ExecutorError(f"BTC accounting evidence is invalid: {exc}") from exc
    return _btc_accounting_payload(report)


def _btc_fraction_inputs(
    context: Mapping[str, Any], *, derived: bool
) -> tuple[float | None, float | None]:
    reserve_fraction = context.get("reserve_fraction")
    if reserve_fraction is None:
        reserve_fraction = context.get("btc_minimum_fraction")
    tactical_fraction = context.get("max_tactical_fraction")
    if tactical_fraction is None:
        tactical_fraction = context.get("btc_max_tactical_fraction")
    if tactical_fraction is None:
        tactical_fraction = 0.3
    if reserve_fraction is None:
        reserve_fraction = 1.0 - float(cast(Any, tactical_fraction))
    return (
        float(cast(Any, reserve_fraction)) if reserve_fraction is not None else None,
        float(cast(Any, tactical_fraction)) if tactical_fraction is not None else None,
    )


def _btc_accounting_payload(report: Any) -> dict[str, Any]:
    return {
        "schema": "platform.btc_accounting/v1",
        "objective_unit": report.objective_unit,
        "initial_value": report.initial_btc_nav,
        "objective_value": report.final_btc_nav,
        "benchmark_value": report.passive_btc_nav,
        "objective_excess": report.excess_btc,
        "objective_excess_fraction": (
            report.excess_btc / report.initial_btc_nav if report.initial_btc_nav > 0 else 0.0
        ),
        "return_fraction": report.return_fraction,
        "fees": report.fees_btc,
        "time_outside_btc_fraction": report.time_outside_btc_fraction,
        "stablecoin_exposure_fraction": report.stablecoin_exposure_fraction,
        "missed_btc_appreciation": report.missed_btc_appreciation,
        "cycles": report.cycles,
        "regime_pnl": dict(report.regime_pnl),
        "maximum_btc_drawdown": report.maximum_btc_drawdown,
        "btc_saved_in_drawdown_periods": report.btc_saved_in_drawdown_periods,
        "round_trip_btc_gain": report.round_trip_btc_gain,
        "maximum_tactical_allocation": report.maximum_tactical_allocation,
        "average_stablecoin_exposure_fraction": report.average_stablecoin_exposure_fraction,
        "worst_reentry_slippage": report.worst_reentry_slippage,
        "failed_reentries": report.failed_reentries,
        "external_deposits_btc": report.external_deposits_btc,
        "external_withdrawals_btc": report.external_withdrawals_btc,
        "event_receipts": [dict(item) for item in report.event_receipts],
    }


def _futures_product_accounting(
    context: Mapping[str, Any], *, fallback_return: Any | None
) -> dict[str, Any] | None:
    events = context.get("futures_events", context.get("trade_events"))
    if events is None:
        events = _derived_futures_accounting_inputs(context)
    if events is None:
        return _futures_return_ledger(context, fallback_return)
    from src.research.accounting import FuturesResearchAccounting, ProductAccountingError

    try:
        report = FuturesResearchAccounting().evaluate(
            events=events,
            initial_cash=float(context.get("initial_cash", context.get("initial_equity", 0.0))),
            leverage=float(context.get("leverage", 1.0)),
            maintenance_margin_fraction=float(context.get("maintenance_margin_fraction", 0.0)),
            max_participation_fraction=float(context.get("max_participation_fraction", 1.0)),
            funding_timestamps=context.get("funding_timestamps"),
            max_margin_fraction=float(context.get("max_margin_fraction", 1.0)),
            target_notional=context.get("target_notional"),
            margin_mode=str(context.get("margin_mode", "isolated")),
            liquidation_buffer_fraction=float(context.get("liquidation_buffer_fraction", 0.0)),
        )
    except (ProductAccountingError, TypeError, ValueError) as exc:
        raise ExecutorError(f"futures accounting evidence is invalid: {exc}") from exc
    return _futures_accounting_payload(report)


def _futures_return_ledger(
    context: Mapping[str, Any], fallback_return: Any | None
) -> dict[str, Any] | None:
    if fallback_return is None:
        return None
    initial_equity = float(context.get("initial_cash", context.get("initial_equity", 1.0)))
    if not math.isfinite(initial_equity) or initial_equity <= 0.0:
        raise ExecutorError(
            "active-income return-ledger accounting requires positive initial equity"
        )
    net_pnl = float(fallback_return.net_pnl) * initial_equity
    observations = int(fallback_return.effective_observations)
    fees = float(fallback_return.fees) * initial_equity
    funding_pnl = float(fallback_return.funding_pnl) * initial_equity
    slippage = float(fallback_return.slippage) * initial_equity
    turnover_notional = float(fallback_return.turnover) * initial_equity
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
        "fees": fees,
        "funding_pnl": funding_pnl,
        "spread_cost": 0.0,
        "slippage_cost": slippage,
        "fills": observations,
        "partial_fills": 0,
        "capacity_violations": 0,
        "capacity_passed": True,
        "max_leverage": 1.0,
        "max_margin_fraction": 0.0,
        "liquidation": False,
        "effective_observations": observations,
        "turnover_notional": turnover_notional,
        "implementation_shortfall": fees + slippage,
        "capital_efficiency": (
            net_pnl / turnover_notional if turnover_notional > 0.0 else 0.0
        ),
        "funding_adjusted_expectancy": net_pnl / observations if observations > 0 else 0.0,
        "margin_mode": "isolated",
        "target_notional": None,
        "liquidation_buffer_fraction": 0.0,
        "event_receipts": (),
        "source": "canonical_return_ledger",
    }


def _futures_accounting_payload(report: Any) -> dict[str, Any]:
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
        "capacity_passed": report.capacity_violations == 0,
        "max_leverage": report.max_leverage,
        "max_margin_fraction": report.max_margin_fraction,
        "liquidation": report.liquidation,
        "effective_observations": report.effective_observations,
        "turnover_notional": report.turnover_notional,
        "implementation_shortfall": report.implementation_shortfall,
        "capital_efficiency": report.capital_efficiency,
        "funding_adjusted_expectancy": report.funding_adjusted_expectancy,
        "margin_mode": report.margin_mode,
        "target_notional": report.target_notional,
        "liquidation_buffer_fraction": report.liquidation_buffer_fraction,
        "event_receipts": [dict(item) for item in report.event_receipts],
    }


def _frame_rows(context: Mapping[str, Any]) -> tuple[dict[str, Any], ...] | None:
    frame = context.get("market_frame")
    if frame is None:
        return None
    if hasattr(frame, "to_dict"):
        rows = frame.to_dict(orient="records")
    elif isinstance(frame, list | tuple):
        rows = frame
    else:
        return None
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        return None
    return tuple(dict(row) for row in rows)


def _frame_times(
    context: Mapping[str, Any], rows: tuple[dict[str, Any], ...]
) -> tuple[str, ...] | None:
    supplied = context.get("timestamps")
    if isinstance(supplied, list | tuple) and len(supplied) == len(rows):
        return tuple(_accounting_time(value, index=index) for index, value in enumerate(supplied))
    values: list[str] = []
    for index, row in enumerate(rows):
        value = next(
            (
                row.get(name)
                for name in ("timestamp", "close_timestamp", "observed_at", "time")
                if row.get(name) is not None
            ),
            None,
        )
        if value is None:
            return None
        values.append(_accounting_time(value, index=index))
    return tuple(values)


def _accounting_time(value: Any, *, index: int) -> str:
    if isinstance(value, bool):
        raise ExecutorError(f"accounting timestamp {index} is invalid")
    if isinstance(value, int | float):
        try:
            return dt.datetime.fromtimestamp(float(value) / 1_000, dt.UTC).isoformat()
        except (OverflowError, OSError, ValueError) as exc:
            raise ExecutorError(f"accounting timestamp {index} is invalid") from exc
    try:
        return dt.datetime.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ExecutorError(f"accounting timestamp {index} is invalid") from exc


def _row_price(row: Mapping[str, Any], *, field: str = "close") -> float | None:
    raw = row.get(field, row.get("price"))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _signed_signal(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, result)) if math.isfinite(result) else 0.0


def _derived_btc_accounting_inputs(
    context: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]] | tuple[None, None]:
    rows = _frame_rows(context)
    signals = _numeric_series(context.get("signals"))
    if rows is None or len(rows) != len(signals):
        return None, None
    times = _frame_times(context, rows)
    if times is None:
        return None, None
    raw_prices = tuple(_row_price(row) for row in rows)
    if any(price is None for price in raw_prices):
        return None, None
    prices: tuple[float, ...] = tuple(float(cast(Any, price)) for price in raw_prices)
    tactical = float(
        context.get("max_tactical_fraction", context.get("btc_max_tactical_fraction", 0.3))
    )
    if not math.isfinite(tactical) or not 0.0 <= tactical <= 1.0:
        raise ExecutorError("BTC tactical fraction is invalid")
    fee_rate = _nonnegative_rate(context.get("fee_bps", 10.0), field="fee_bps") / 10_000.0
    slippage_rate = (
        _nonnegative_rate(context.get("slippage_bps", 2.0), field="slippage_bps") / 10_000.0
    )
    btc = float(context.get("initial_btc", 1.0))
    stable = float(context.get("initial_stablecoin", 0.0))
    if not all(math.isfinite(value) and value >= 0.0 for value in (btc, stable)):
        raise ExecutorError("BTC initial balances are invalid")
    trades: list[dict[str, Any]] = []
    marks = [
        {"timestamp": observed_at, "price": price, "regime": row.get("regime", "unclassified")}
        for observed_at, price, row in zip(times, prices, rows, strict=True)
    ]
    for observed_at, price, signal in zip(times, prices, signals, strict=True):
        desired_fraction = 1.0 - tactical if _signed_signal(signal) < 0.0 else 1.0
        nav = btc + stable / price
        desired_btc = nav * desired_fraction
        quantity = desired_btc - btc
        if abs(quantity) <= 1e-12:
            continue
        side = "buy" if quantity > 0 else "sell"
        execution_price = price * (1.0 + slippage_rate if side == "buy" else 1.0 - slippage_rate)
        base_quantity = abs(quantity)
        trades.append(
            {
                "timestamp": observed_at,
                "side": side,
                "quantity_btc": base_quantity,
                "price": execution_price,
                "reference_price": price,
                "fee": base_quantity * execution_price * fee_rate,
                "fee_asset": "USDT",
            }
        )
        if side == "buy":
            btc += base_quantity
            stable -= base_quantity * execution_price * (1.0 + fee_rate)
        else:
            btc -= base_quantity
            stable += base_quantity * execution_price * (1.0 - fee_rate)
    return tuple(trades), tuple(marks)


def _derived_futures_accounting_inputs(
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    inputs = _futures_frame_inputs(context)
    if inputs is None:
        return None
    rows, signals, times, prices = inputs
    fee_rate = _nonnegative_rate(context.get("fee_bps", 5.0), field="fee_bps") / 10_000.0
    slippage_rate = (
        _nonnegative_rate(context.get("slippage_bps", 2.0), field="slippage_bps") / 10_000.0
    )
    target_notional = _derived_futures_target_notional(context)
    events: list[dict[str, Any]] = []
    previous_target = 0.0
    for observed_at, raw_price, row, raw_signal in zip(times, prices, rows, signals, strict=True):
        price = float(raw_price)
        target = _futures_target_notional(raw_signal, target_notional, price)
        delta = target - previous_target
        fill = _futures_fill_event(
            observed_at,
            row,
            price,
            delta,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        if fill is not None:
            events.append(fill)
            previous_target = target
        events.append(_futures_mark_event(observed_at, row, price))
        funding = _futures_funding_event(observed_at, row, price)
        if funding is not None:
            events.append(funding)
    if not events:
        return None
    return tuple(events)


def _futures_frame_inputs(
    context: Mapping[str, Any],
) -> (
    tuple[
        tuple[dict[str, Any], ...],
        list[float],
        tuple[str, ...],
        tuple[float, ...],
    ]
    | None
):
    rows = _frame_rows(context)
    signals = _numeric_series(context.get("signals"))
    if rows is None or len(rows) != len(signals):
        return None
    times = _frame_times(context, rows)
    if times is None:
        return None
    raw_prices = tuple(_row_price(row) for row in rows)
    if any(price is None for price in raw_prices):
        return None
    prices: tuple[float, ...] = tuple(float(cast(Any, price)) for price in raw_prices)
    return rows, signals, times, prices


def _derived_futures_target_notional(context: Mapping[str, Any]) -> float:
    initial_cash = float(context.get("initial_cash", context.get("initial_equity", 1_000.0)))
    maximum_position = float(context.get("maximum_position", 0.1))
    leverage = float(context.get("leverage", 1.0))
    values = (initial_cash, maximum_position, leverage)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ExecutorError("futures accounting configuration is invalid")
    raw_target_notional = context.get("target_notional")
    if isinstance(raw_target_notional, Mapping):
        target_notional = sum(float(value) for value in raw_target_notional.values())
    else:
        target_notional = float(
            raw_target_notional
            if raw_target_notional is not None
            else initial_cash * maximum_position * leverage
        )
    if not math.isfinite(target_notional) or target_notional <= 0.0:
        raise ExecutorError("futures target notional is invalid")
    return target_notional


def _futures_symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("instrument_id", row.get("symbol", "BTCUSDT")))


def _futures_target_notional(raw_signal: Any, target_notional: float, price: float) -> float:
    return _signed_signal(raw_signal) * target_notional / price


def _futures_fill_event(
    observed_at: str,
    row: Mapping[str, Any],
    price: float,
    delta: float,
    *,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any] | None:
    if abs(delta) <= 1e-12:
        return None
    side = "buy" if delta > 0 else "sell"
    quantity = abs(delta)
    execution_price = price * (1.0 + slippage_rate if side == "buy" else 1.0 - slippage_rate)
    return {
        "type": "fill",
        "timestamp": observed_at,
        "symbol": _futures_symbol(row),
        "side": side,
        "quantity": quantity,
        "price": execution_price,
        "reference_price": price,
        "fee": quantity * execution_price * fee_rate,
    }


def _futures_mark_event(observed_at: str, row: Mapping[str, Any], price: float) -> dict[str, Any]:
    return {
        "type": "mark",
        "timestamp": observed_at,
        "symbol": _futures_symbol(row),
        "mark_price": price,
    }


def _futures_funding_event(
    observed_at: str, row: Mapping[str, Any], price: float
) -> dict[str, Any] | None:
    if row.get("funding_event") is not True:
        return None
    if row.get("funding_rate") is None and row.get("funding") is None:
        return None
    return {
        "type": "funding",
        "timestamp": observed_at,
        "symbol": _futures_symbol(row),
        "mark_price": price,
        "funding_rate": row.get("funding_rate", row.get("funding", 0.0)),
    }


def _finite_rate(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ExecutorError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExecutorError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ExecutorError(f"{field} must be finite")
    return result


def _nonnegative_rate(value: Any, *, field: str) -> float:
    result = _finite_rate(value, field=field)
    if result < 0:
        raise ExecutorError(f"{field} must be non-negative")
    return result


def _scaled_rates(value: Any, multiplier: float) -> Any:
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value) * multiplier
    if isinstance(value, list | tuple):
        return [float(item) * multiplier for item in value]
    return value


def _stress_report(
    signals: list[float],
    returns: list[float],
    *,
    fee_rate: float,
    slippage_rate: float,
    funding_rate: float,
    funding_rates: Any,
):
    from src.research.returns import PositionReturnLedger

    return PositionReturnLedger(
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        funding_rate=funding_rate,
    ).measure(signals, returns, funding_rates=funding_rates)


def _stress_passes(stressed: float, base: float, context: Mapping[str, Any]) -> bool:
    allowance = _nonnegative_rate(
        context.get("maximum_stress_degradation", 1.0),
        field="maximum_stress_degradation",
    )
    return stressed >= base - abs(base) * allowance - 1e-12


def _degradation(base: float, stressed: float) -> float:
    if base <= 0:
        return 0.0
    return max(0.0, (base - stressed) / max(abs(base), 1e-12))


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


def _evaluation_window_count(context: Mapping[str, Any]) -> int:
    try:
        configured = int(context.get("walk_forward_windows", 3))
    except (TypeError, ValueError):
        configured = 3
    return max(3, configured)


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


def _tunable_parameter_names(candidate: Candidate, context: Mapping[str, Any]) -> tuple[str, ...]:
    signal_model = candidate.definition.signal_model
    if isinstance(signal_model, Mapping) and signal_model.get("parameter_free") is True:
        return ()
    parameters = candidate.definition.signal_model.get("parameters", {})
    declared_tunable = context.get("tunable_parameters")
    if isinstance(declared_tunable, list | tuple):
        return tuple(str(name) for name in declared_tunable)
    if isinstance(parameters, Mapping):
        return tuple(
            str(name)
            for name, value in parameters.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        )
    return ()


def _declared_parameter_neighbours(context: Mapping[str, Any]) -> dict[str, list[float]]:
    raw = context.get("parameter_neighbour_returns", context.get("neighbour_returns", {}))
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(name): numeric for name, values in raw.items() if (numeric := _numeric_series(values))
    }


def _generated_parameter_neighbours(
    candidate: Candidate,
    context: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, list[float]]:
    frame = context.get("market_frame")
    strategy_name = candidate.definition.signal_model.get("registered_strategy")
    if frame is None or not isinstance(strategy_name, str):
        return _generated_dsl_parameter_neighbours(candidate, context, parameters)
    neighbours: dict[str, list[float]] = {}
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
        return {}
    return neighbours


def _generated_dsl_parameter_neighbours(
    candidate: Candidate,
    context: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, list[float]]:
    rows = context.get("feature_rows")
    rule = candidate.definition.signal_model.get("rule")
    if not isinstance(rows, list | tuple) or not isinstance(rule, Mapping):
        return {}
    feature = str(rule.get("feature") or "")
    operator = str(rule.get("operator") or "")
    if not feature or operator not in {"gt", "ge", "lt", "le"}:
        return {}
    threshold = float(parameters.get("threshold", rule.get("threshold", 0.0)))
    step = max(abs(threshold) * 0.1, 0.1)
    neighbours: dict[str, list[float]] = {}
    for direction in (-1, 1):
        varied_rule = {**dict(rule), "threshold": threshold + direction * step}
        signals = _dsl_signals(rows, varied_rule)
        returns = _market_price_returns(context, len(signals))
        aligned = min(len(returns), max(0, len(signals) - 1))
        neighbours[f"threshold:{direction:+d}"] = [
            signals[index] * returns[index] for index in range(aligned)
        ]
    return neighbours


def _parameter_neighbours(
    candidate: Candidate, context: Mapping[str, Any]
) -> dict[str, list[float]]:
    neighbours = _declared_parameter_neighbours(context)
    if neighbours:
        return neighbours
    parameters = candidate.definition.signal_model.get("parameters", {})
    if not isinstance(parameters, Mapping):
        return {}
    return _generated_parameter_neighbours(candidate, context, parameters)


def _parameter_stability_results(
    candidate: Candidate,
    context: Mapping[str, Any],
    base_returns: list[float],
    neighbours: Mapping[str, list[float]],
) -> list[dict[str, Any]]:
    results = []
    base_total = sum(base_returns)
    snapshots = list(context.get("dataset_snapshot_ids", ()))
    for name, values in sorted(neighbours.items()):
        comparable = values[: len(base_returns)]
        results.append(
            {
                "name": name,
                "run_id": canonical_hash(
                    {
                        "kind": "parameter_neighbour_backtest/v1",
                        "candidate_id": candidate.candidate_id,
                        "dataset_snapshot_ids": snapshots,
                        "neighbour": name,
                        "returns": comparable,
                    }
                ),
                "observations": len(comparable),
                "return": sum(comparable),
                "passed": bool(comparable) and sum(comparable) >= base_total * 0.5,
                "window_returns": _window_sums(comparable, _evaluation_window_count(context)),
                "input_hash": canonical_hash(
                    {
                        "candidate_id": candidate.candidate_id,
                        "dataset_snapshot_ids": snapshots,
                        "neighbour": name,
                        "returns": comparable,
                    }
                ),
            }
        )
    return results


def _parameter_stability(
    candidate: Candidate, context: Mapping[str, Any], base_returns: list[float]
) -> dict[str, Any]:
    tunable = _tunable_parameter_names(candidate, context)
    if not tunable:
        return {
            "status": "not_applicable",
            "passed": True,
            "reason": "no_tunable_parameters",
            "neighbours_tested": 0,
            "results": [],
            "base_window_returns": _window_sums(base_returns, _evaluation_window_count(context)),
        }
    neighbours = _parameter_neighbours(candidate, context)
    results = _parameter_stability_results(candidate, context, base_returns, neighbours)
    base_total = sum(base_returns)
    neighbour_returns = [float(item["return"]) for item in results]
    median_return = statistics.median(neighbour_returns) if neighbour_returns else None
    worst_return = min(neighbour_returns) if neighbour_returns else None
    allowed_degradation = float(context.get("maximum_parameter_degradation", 0.5))
    degradation = (
        max(0.0, (base_total - worst_return) / max(abs(base_total), 1e-12))
        if worst_return is not None
        else None
    )
    cliff_detected = degradation is not None and degradation > allowed_degradation
    passed_count = sum(1 for item in results if item["passed"])
    passed = (
        bool(base_returns)
        and bool(results)
        and not cliff_detected
        and passed_count >= (len(results) + 1) // 2
        and (median_return or 0.0) >= base_total * (1.0 - allowed_degradation)
    )
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "neighbours_tested": len(results),
        "results": results,
        "base_return": base_total,
        "median_return": median_return,
        "worst_return": worst_return,
        "maximum_degradation": allowed_degradation,
        "degradation_fraction": degradation,
        "degradation_shape": "cliff" if cliff_detected else "smooth",
        "cliff_detected": cliff_detected,
        "base_window_returns": _window_sums(base_returns, _evaluation_window_count(context)),
    }


def _cross_symbol_stability(
    context: Mapping[str, Any], base_returns: list[float]
) -> dict[str, Any]:
    scope = tuple(
        str(item)
        for item in context.get("instrument_scope", context.get("expected_symbols", ()))
        if item
    )
    if len(scope) <= 1:
        return {
            "status": "not_applicable",
            "passed": True,
            "symbols": 0,
            "per_symbol": {},
            "reason": "single_instrument",
        }
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
        return {
            "status": "unavailable",
            "passed": False,
            "symbols": 0,
            "per_symbol": {},
            "reason": "missing_symbol_returns",
        }
    missing_symbols = sorted(set(scope) - set(per_symbol)) if len(scope) > 1 else []
    returns = [float(item["return"]) for item in per_symbol.values()]
    positive_fraction = sum(value >= 0.0 for value in returns) / len(returns)
    minimum_positive_fraction = float(context.get("minimum_positive_symbol_fraction", 0.5))
    median_return = statistics.median(returns)
    pooled_return = sum(returns) / len(returns)
    lower_quantile_return = _quantile(returns, 0.1)
    passed = (
        not missing_symbols
        and positive_fraction >= minimum_positive_fraction
        and median_return >= float(context.get("minimum_cross_symbol_median_return", 0.0))
        and pooled_return >= float(context.get("minimum_cross_symbol_pooled_return", 0.0))
    )
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "symbols": len(per_symbol),
        "per_symbol": per_symbol,
        "missing_symbols": missing_symbols,
        "positive_symbol_fraction": positive_fraction,
        "minimum_positive_symbol_fraction": minimum_positive_fraction,
        "median_return": median_return,
        "pooled_return": pooled_return,
        "lower_quantile_return": lower_quantile_return,
        "lower_quantile": 0.1,
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
    minimum_fraction = _finite_rate(
        context.get("minimum_walk_forward_pass_fraction", 0.5),
        field="minimum_walk_forward_pass_fraction",
    )
    return {
        "passed": len(per_window) >= windows and passed / len(per_window) >= minimum_fraction,
        "window_count": len(per_window),
        "pass_fraction": passed / len(per_window) if per_window else 0.0,
        "minimum_pass_fraction": minimum_fraction,
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


def _tail_loss(values: list[float], probability: float = 0.05) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return 0.0
    count = max(1, math.ceil(len(ordered) * probability))
    return max(0.0, -statistics.fmean(ordered[:count]))


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _negative_control_evidence(
    *,
    signals: list[float],
    returns: list[float],
    candidate_return: float,
    control_returns: Any = None,
    controls: tuple[str, ...] = (),
    instrument_scope: tuple[str, ...] = (),
    seed_material: object = "negative-control",
) -> dict[str, dict[str, float | int | bool | str | None]]:
    aligned = min(len(signals), len(returns))
    signals, returns = signals[:aligned], returns[:aligned]
    supplied = control_returns if isinstance(control_returns, Mapping) else {}
    results: dict[str, dict[str, float | int | bool | str | None]] = {}
    for name in controls:
        numeric = _numeric_series(supplied.get(name))
        comparable = numeric[:aligned]
        if (
            name in {"cross_instrument", "predeclared_universe_holdout"}
            and len(set(instrument_scope)) < 2
        ):
            results[name] = {
                "status": "not_applicable",
                "passed": True,
                "observations": 0,
                "control_return": None,
                "reason": "single_instrument_scope",
            }
            continue
        method = "dataset"
        if not comparable:
            derived = derive_control_returns(
                name,
                signals,
                returns,
                seed_material=seed_material,
                instrument_scope=instrument_scope,
            )
            if derived is not None:
                comparable, method = derived
                comparable = comparable[:aligned]
        if not comparable:
            results[name] = {
                "status": "unavailable",
                "passed": False,
                "observations": 0,
                "control_return": None,
                "source": "dataset",
                "input_hash": None,
                "reason": "control_dataset_unavailable",
            }
            continue
        control_return = sum(comparable) if comparable else None
        results[name] = {
            "passed": bool(comparable)
            and control_return is not None
            and candidate_return >= control_return,
            "observations": len(comparable),
            "control_return": control_return,
            "source": "dataset" if method == "dataset" else "derived_immutable_inputs",
            "method": method,
            "input_hash": control_identity(
                name,
                comparable,
                seed_material=seed_material,
                method=method,
            ),
        }
    return results


def _negative_control_names(candidate: Candidate, context: Mapping[str, Any]) -> tuple[str, ...]:
    raw = context.get("negative_controls")
    if raw is None:
        policy = candidate.definition.validation_policy
        raw = policy.get("negative_controls") if isinstance(policy, Mapping) else None
    if not isinstance(raw, list | tuple):
        defaults = {
            "time_series": (
                "block_permutation",
                "synthetic_autocorrelated_null",
                "feature_ablation",
                "parameter_neighbourhood",
            ),
            "mean_reversion": (
                "block_permutation",
                "synthetic_autocorrelated_null",
                "feature_ablation",
                "parameter_neighbourhood",
            ),
            "cross_sectional": (
                "block_permutation",
                "predeclared_universe_holdout",
                "cross_instrument",
            ),
            "relative_value": (
                "block_permutation",
                "synthetic_autocorrelated_null",
                "feature_ablation",
                "parameter_neighbourhood",
            ),
            "microstructure": (
                "placebo_event_times",
                "feature_ablation",
                "cross_instrument",
            ),
            "machine_learning": (
                "block_permutation",
                "synthetic_autocorrelated_null",
                "feature_ablation",
            ),
        }
        raw = defaults.get(str(candidate.definition.family), ("block_permutation",))
    return tuple(dict.fromkeys(str(name) for name in raw if str(name)))


def _build_registered_context(
    _candidate: ContextCandidate, context: Mapping[str, Any]
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


def _build_dsl_context(
    _candidate: ContextCandidate, context: Mapping[str, Any]
) -> Mapping[str, Any]:
    rows = context.get("feature_rows")
    if (
        not isinstance(rows, list | tuple)
        or not rows
        or not all(isinstance(item, Mapping) for item in rows)
    ):
        raise ExecutorError("generated DSL context has no immutable feature rows")
    return {**dict(context), "feature_rows": tuple(dict(item) for item in rows)}


def _build_ml_context(candidate: ContextCandidate, context: Mapping[str, Any]) -> Mapping[str, Any]:
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


def _build_semantic_context(
    candidate: ContextCandidate, context: Mapping[str, Any]
) -> Mapping[str, Any]:
    model = candidate.definition.signal_model
    name = semantic_strategy_name(
        candidate.definition.source_type.value,
        model.get("semantic_strategy") if isinstance(model, Mapping) else None,
    )
    if candidate.definition.source_type is StrategySourceType.MICROSTRUCTURE and not context.get(
        "event_data_segment_ids"
    ):
        raise ExecutorError("microstructure context needs immutable event-data segments")
    try:
        value = semantic_input_from_features(name, context)
    except (SemanticEvaluationError, KeyError, TypeError, ValueError) as exc:
        raise ExecutorError(str(exc)) from exc
    return {**dict(context), "semantic_input": value}


def execute_registered_python(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    frame = context.get("market_frame")
    if frame is None:
        raise ExecutorError("registered Python execution requires a canonical market_frame")
    try:
        behaviour = RegisteredStrategyBehaviour.from_definition(candidate.definition)
        signals = behaviour.generate_signals(frame)
        parity = behaviour.parity_receipt(frame)
    except StrategyBehaviourError as exc:
        raise ExecutorError(str(exc)) from exc
    return _measured_result(
        candidate,
        {
            **context,
            "behaviour_hash": behaviour.behaviour_hash,
            "behaviour_input_hash": parity["input_hash"],
            "parity_receipt": parity,
        },
        signals,
    )


def execute_generated_dsl(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    rows = context.get("feature_rows")
    rule = candidate.definition.signal_model.get("rule")
    if not isinstance(rows, list | tuple) or not isinstance(rule, Mapping):
        raise ExecutorError("DSL execution requires canonical feature_rows and a typed rule")
    try:
        behaviour = TypedRuleBehaviour(rule)
        signals = behaviour.generate_signals(rows)
        parity = behaviour.parity_receipt(rows)
    except StrategyBehaviourError as exc:
        raise ExecutorError(str(exc)) from exc
    return _measured_result(
        candidate,
        {**context, "behaviour_hash": behaviour.behaviour_hash, "parity_receipt": parity},
        signals,
    )


def _dsl_signals(rows: list[Any] | tuple[Any, ...], rule: Mapping[str, Any]) -> tuple[int, ...]:
    try:
        return TypedRuleBehaviour(rule).generate_signals(rows)
    except StrategyBehaviourError as exc:
        raise ExecutorError(str(exc)) from exc


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
    output = model.evaluate(features)
    behaviour_hash = behaviour_hash_for_definition(candidate.definition)
    parity_payload = {
        "schema": "machine_learning_parity/v1",
        "behaviour_hash": behaviour_hash,
        "input_hash": output.feature_vector_hash,
        "output_hash": canonical_hash(output),
        "signal": output.score,
    }
    return _measured_result(
        candidate,
        {
            **context,
            "behaviour_hash": behaviour_hash,
            "parity_receipt": {
                **parity_payload,
                "receipt_hash": canonical_hash(parity_payload),
            },
        },
        output,
    )


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
    model = candidate.definition.signal_model
    name = semantic_strategy_name(
        candidate.definition.source_type.value,
        model.get("semantic_strategy") if isinstance(model, Mapping) else None,
    )
    try:
        semantic_input = semantic_input_from_features(name, context)
        output = SEMANTIC_STRATEGIES.get(name).evaluate(semantic_input)
        instrument_id = _semantic_instrument_id(candidate, context, output)
        forecast = semantic_forecast_from_output(
            output,
            instrument_id=instrument_id,
            position_limits=(
                context.get("position_limits")
                if isinstance(context.get("position_limits"), Mapping)
                else candidate.definition.position_model
            ),
        )
        signal = semantic_signal(output, instrument_id=instrument_id)
    except (SemanticEvaluationError, KeyError, TypeError, ValueError) as exc:
        raise ExecutorError(str(exc)) from exc
    parity_payload = {
        "schema": "semantic_parity/v1",
        "behaviour_hash": behaviour_hash_for_definition(candidate.definition),
        "strategy": name,
        "input_hash": canonical_hash(semantic_input),
        "output_hash": canonical_hash(output),
        "instrument_id": instrument_id,
        "signal": signal,
    }
    parity = {**parity_payload, "receipt_hash": canonical_hash(parity_payload)}
    return _measured_result(
        candidate,
        {
            **context,
            "signals": context.get("signals") or [signal],
            "behaviour_hash": behaviour_hash_for_definition(candidate.definition),
            "parity_receipt": parity,
        },
        forecast,
    )


def _semantic_instrument_id(candidate: Candidate, context: Mapping[str, Any], output: Any) -> str:
    explicit = str(context.get("instrument_id") or "").strip()
    if explicit:
        return explicit
    scope = context.get("instrument_scope")
    if isinstance(scope, list | tuple) and scope:
        return str(scope[0])
    universe = candidate.definition.universe
    for field in ("instrument_ids", "symbols"):
        values = universe.get(field) if isinstance(universe, Mapping) else None
        if isinstance(values, list | tuple) and values:
            return str(values[0])
    for field in ("target_fractions", "target_notionals"):
        values = getattr(output, field, None)
        if isinstance(values, Mapping) and values:
            return str(next(iter(values)))
    forecasts = output if isinstance(output, tuple) else ()
    if forecasts and hasattr(forecasts[0], "instrument_id"):
        return str(forecasts[0].instrument_id)
    raise ExecutorError("semantic execution requires an instrument identity")


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
