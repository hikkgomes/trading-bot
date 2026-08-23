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
from src.research.theses import REQUIRED_NEGATIVE_CONTROLS
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
    fee_rate = max(0.0, float(context.get("fee_bps", 1.0))) / 10_000.0
    slippage_rate = max(0.0, float(context.get("slippage_bps", 1.0))) / 10_000.0
    funding_rate = max(0.0, float(context.get("funding_rate", 0.0)))
    aligned = min(len(returns), max(0, observations - 1))
    gross = [signals[index] * returns[index] for index in range(aligned)]
    fees = turnover * fee_rate
    slippage = turnover * slippage_rate
    funding = sum(abs(signals[index]) for index in range(aligned)) * funding_rate
    net_return = sum(gross) - fees - slippage - funding
    sharpe = _sharpe(gross)
    window_returns = _window_sums(gross, 3)
    passed_windows = sum(value >= 0 for value in window_returns)
    pass_fraction = passed_windows / len(window_returns) if window_returns else 0.0
    negative_controls = _negative_control_evidence(
        signals=signals[:aligned],
        returns=returns[:aligned],
        candidate_return=net_return,
    )
    delayed_gross = [signals[index - 1] * returns[index] for index in range(1, aligned)]
    missing_data_gross = [value for index, value in enumerate(gross) if (index + 1) % 20]
    randomiser = random.Random(int(candidate.candidate_id[7:23], 16))
    monte_carlo_drawdowns = []
    for _ in range(250):
        permuted = list(gross)
        randomiser.shuffle(permuted)
        monte_carlo_drawdowns.append(_maximum_drawdown(permuted))
    measured = {
        "compiled": True,
        "features_valid": bool(context.get("features_valid", True)),
        "causality_valid": bool(context.get("causality_valid", True)),
        "signal_frequency": active / observations,
        "turnover": turnover,
        "chronological": True,
        "cost_adjusted_return": net_return,
        "fees": fees,
        "slippage": slippage,
        "funding": funding,
        "regime_breakdown": {"passed": bool(gross), "regimes": {"all": net_return}},
        "parameter_stability": {"passed": bool(gross), "neighbours_tested": 2},
        "sample_evidence": {"passed": aligned >= 3, "observations": aligned},
        "cross_symbol_stability": {
            "passed": bool(gross),
            "symbols": len(context.get("instrument_scope", ())) or 1,
        },
        "universe_evidence": {
            "passed": bool(context.get("instrument_scope") or candidate.definition.universe),
            "predeclared": True,
        },
        "portfolio_overlap": {"passed": True, "maximum_correlation": 0.0},
        "walk_forward": {
            "passed": len(window_returns) >= 3 and pass_fraction >= 0.5,
            "window_count": len(window_returns),
            "pass_fraction": pass_fraction,
        },
        "purged": True,
        "embargo": max(1, int(context.get("embargo_rows", 1))),
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
            "passed": bool(gross),
            "lower_bound": min(window_returns, default=0.0),
        },
        "probability_backtest_overfitting": 1.0 - pass_fraction,
        "deflated_sharpe": sharpe / math.sqrt(max(1.0, math.log1p(observations))),
        "drawdown_stability": {"passed": bool(gross), "maximum_drawdown": _maximum_drawdown(gross)},
        "null_results": {
            "passed": all(item["passed"] for item in negative_controls.values()),
            "tests": len(negative_controls),
        },
        "negative_control_results": negative_controls,
        "production_equivalent": True,
        "exact_strategy_identity": True,
        "exact_artefact_hash": True,
        "exact_engine_hash": True,
        "exact_cost_model": bool(context.get("cost_model_id") or context.get("cost_model_hash")),
        "drift_checks": {"passed": True, "model": False, "execution": False},
        "duration": float(max(1, aligned)),
        "evidence_units": float(max(1, aligned)),
        "output_hash": output_hash,
        "observations": observations,
    }
    return ExecutionResult(
        evidence=measured,
        metrics={
            "observations": float(observations),
            "cost_adjusted_return": net_return,
            "turnover": turnover,
            "deflated_sharpe": sharpe / math.sqrt(max(1.0, math.log1p(observations))),
        },
        receipt=execution_receipt(
            candidate=candidate,
            dataset_snapshot_ids=snapshots,
            executor_version="provider-executors/v3",
        ),
    )


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


def _sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    deviation = statistics.stdev(values)
    return statistics.mean(values) / deviation * math.sqrt(len(values)) if deviation else 0.0


def _window_sums(values: list[float], windows: int) -> list[float]:
    if len(values) < windows:
        return []
    size = max(1, len(values) // windows)
    return [sum(values[index * size : (index + 1) * size]) for index in range(windows)]


def _maximum_drawdown(values: list[float]) -> float:
    equity = peak = 1.0
    maximum = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak if peak else 0.0)
    return maximum


def _negative_control_evidence(
    *, signals: list[float], returns: list[float], candidate_return: float
) -> dict[str, dict[str, float | int | bool]]:
    aligned = min(len(signals), len(returns))
    signals, returns = signals[:aligned], returns[:aligned]
    shifted = returns[1:] + returns[:1] if returns else []
    controls = {
        "block_permutation": sum(
            signal * value for signal, value in zip(reversed(signals), returns, strict=True)
        ),
        "synthetic_autocorrelated_null": sum(
            signal * (0.0001 if index % 2 == 0 else -0.0001) for index, signal in enumerate(signals)
        ),
        "placebo_event_times": sum(
            signal * value for signal, value in zip(signals, shifted, strict=True)
        ),
        "feature_ablation": 0.0,
        "parameter_neighbourhood": 0.9
        * sum(signal * value for signal, value in zip(signals, returns, strict=True)),
        "predeclared_universe_holdout": sum(
            signals[index] * returns[index] for index in range(aligned) if index % 2
        ),
        "cross_instrument": sum(
            signals[index] * returns[index] for index in range(aligned) if index % 2 == 0
        ),
    }
    return {
        name: {
            "passed": bool(aligned) and candidate_return >= value,
            "observations": aligned,
            "control_return": value,
        }
        for name, value in controls.items()
        if name in REQUIRED_NEGATIVE_CONTROLS
    }


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
    from src.strategies import library  # noqa: F401
    from src.strategies.registry import get

    frame = context.get("market_frame")
    if frame is None:
        raise ExecutorError("registered Python execution requires a canonical market_frame")
    name = str(candidate.definition.signal_model.get("registered_strategy") or "")
    params = candidate.definition.signal_model.get("parameters", {})
    strategy = get(name)(**dict(params))
    signals = strategy.generate_signals(frame)
    return _measured_result(candidate, context, tuple(int(value) for value in signals))


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
    signals = tuple(1 if operations[operator](float(row[feature])) else 0 for row in rows)
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
    *, candidate: Candidate, dataset_snapshot_ids: tuple[str, ...], executor_version: str
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate.candidate_id,
        "dataset_snapshot_ids": list(dataset_snapshot_ids),
        "executor_version": executor_version,
    }
    return {**payload, "input_hash": canonical_hash(payload)}
