"""Resource-bounded, pre-holdout machine-learning research.

The runner rotates through a declared grid of feature sets, labels, horizons,
models, hyperparameters, and causal regime gates. It scores chronological
walk-forward windows inside the adaptive region, then uses a durably claimed
protected tail. Passing sklearn models are frozen as safe JSON and exported to
an isolated review catalog; staging and activation remain explicit operations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_exploration.dsr import DSR_METHOD, MIN_TRIAL_SHARPE_STD
from src import metrics as performance_metrics
from src.alpha.frozen_gradient_boosting import export_sklearn_gradient_boosting
from src.autopilot.config import DEFAULT_CONFIG_PATH as DEFAULT_AUTOPILOT_CONFIG
from src.autopilot.config import load_config as load_autopilot_config
from src.autopilot.experiment_memory import (
    EvaluationConflictError,
    ExperimentMemory,
    HoldoutSealBudgetError,
)
from src.autopilot.io import write_json_atomic
from src.autopilot.ml_candidate_artifact import (
    DEFAULT_REVIEWABLE_DIR,
    export_reviewable_artifact,
)
from src.config import PROJECT_ROOT, indicator_data_dir
from src.strategies import BacktestConfig, Strategy, run_backtest
from src.strategies.library.ml_classifier import MlClassifierStrategy
from src.strategies.library.ml_regressor import MlRegressorStrategy

CONFIG_SCHEMA = "autopilot.ml_research_config/v1"
REPORT_SCHEMA = "autopilot.ml_research_report/v1"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "ml_research.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime" / "research" / "ml_research.json"
DEFAULT_STATE = PROJECT_ROOT / "runtime" / "research" / "ml_research_state.json"
MAX_CONFIG_BYTES = 256 * 1024


class MlResearchConfigError(ValueError):
    """The ML research configuration is malformed or exceeds safe bounds."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise MlResearchConfigError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.stat().st_mode):
        raise MlResearchConfigError(f"config must be a regular non-symlink file: {path}")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise MlResearchConfigError(f"config exceeds {MAX_CONFIG_BYTES} bytes")

    def reject_constant(value: str) -> None:
        raise MlResearchConfigError(f"non-standard JSON constant: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MlResearchConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MlResearchConfigError("config must be a JSON object")
    return payload


def _keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise MlResearchConfigError(f"{label} has unknown fields: {', '.join(unknown)}")


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MlResearchConfigError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MlResearchConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise MlResearchConfigError(f"{label} must be in [{minimum}, {maximum}]")
    return result


def _strings(value: Any, label: str, allowed: set[str] | None = None) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(v, str) or not v for v in value)
    ):
        raise MlResearchConfigError(f"{label} must be a non-empty string list")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise MlResearchConfigError(f"{label} must not contain duplicates")
    if allowed is not None and not set(result) <= allowed:
        raise MlResearchConfigError(f"{label} contains unsupported values")
    return result


@dataclass(frozen=True)
class DatasetSpec:
    product: str
    market: str
    symbol: str
    timeframe: str
    pnl_unit: str

    @property
    def path(self) -> Path:
        directory = indicator_data_dir(self.symbol, self.market, legacy_fallback=True)
        return directory / f"{self.symbol}_{self.timeframe}_all_indicators.parquet"


@dataclass(frozen=True)
class MlResearchConfig:
    path: Path
    memory_path: Path
    datasets: tuple[DatasetSpec, ...]
    max_trials_per_cycle: int
    max_runtime_seconds: int
    max_rows: int
    min_train_rows: int
    validation_rows: int
    step_rows: int
    min_windows: int
    final_holdout_fraction: float
    embargo_bars: int
    feature_sets: tuple[str, ...]
    families: tuple[str, ...]
    models: tuple[str, ...]
    horizons: tuple[int, ...]
    regimes: tuple[str, ...]
    max_features: tuple[int, ...]
    estimators: tuple[int, ...]
    learning_rates: tuple[float, ...]
    minimum_total_trades: int
    minimum_holdout_trades: int
    minimum_profitable_window_fraction: float
    maximum_drawdown: float
    minimum_dsr: float


def load_config(path: Path = DEFAULT_CONFIG) -> MlResearchConfig:
    path = Path(path)
    payload = _strict_json(path)
    _keys(
        payload,
        {"schema", "memory_path", "datasets", "budgets", "search", "gates"},
        "config",
    )
    if payload.get("schema") != CONFIG_SCHEMA:
        raise MlResearchConfigError(f"schema must be {CONFIG_SCHEMA}")
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise MlResearchConfigError("datasets must be a non-empty list")
    datasets: list[DatasetSpec] = []
    for index, raw in enumerate(raw_datasets):
        if not isinstance(raw, Mapping):
            raise MlResearchConfigError(f"datasets[{index}] must be an object")
        _keys(
            raw,
            {"product", "market", "symbol", "timeframe", "pnl_unit", "enabled"},
            f"datasets[{index}]",
        )
        if raw.get("enabled", True) is False:
            continue
        if raw.get("enabled", True) is not True:
            raise MlResearchConfigError(f"datasets[{index}].enabled must be boolean")
        market, pnl_unit = raw.get("market"), raw.get("pnl_unit")
        if market not in {"spot", "futures"} or pnl_unit not in {"btc", "usdt"}:
            raise MlResearchConfigError(f"datasets[{index}] has invalid market/pnl_unit")
        fields = {key: raw.get(key) for key in ("product", "symbol", "timeframe")}
        if any(not isinstance(value, str) or not value for value in fields.values()):
            raise MlResearchConfigError(f"datasets[{index}] identity fields must be non-empty")
        datasets.append(
            DatasetSpec(
                product=fields["product"],
                market=market,
                symbol=fields["symbol"].upper(),
                timeframe=fields["timeframe"],
                pnl_unit=pnl_unit,
            )
        )
    if not datasets:
        raise MlResearchConfigError("at least one dataset must be enabled")

    budgets = payload.get("budgets")
    search = payload.get("search")
    gates = payload.get("gates")
    if not all(isinstance(value, Mapping) for value in (budgets, search, gates)):
        raise MlResearchConfigError("budgets, search, and gates must be objects")
    _keys(
        budgets,
        {
            "max_trials_per_cycle",
            "max_runtime_seconds",
            "max_rows",
            "min_train_rows",
            "validation_rows",
            "step_rows",
            "min_windows",
            "final_holdout_fraction",
            "embargo_bars",
        },
        "budgets",
    )
    _keys(
        search,
        {
            "feature_sets",
            "families",
            "models",
            "horizons",
            "regimes",
            "max_features",
            "estimators",
            "learning_rates",
        },
        "search",
    )
    _keys(
        gates,
        {
            "minimum_total_trades",
            "minimum_holdout_trades",
            "minimum_profitable_window_fraction",
            "maximum_drawdown",
            "minimum_dsr",
        },
        "gates",
    )
    horizons = tuple(
        _integer(value, "search.horizons", minimum=1, maximum=10_000)
        for value in search.get("horizons", [])
    )
    if not horizons:
        raise MlResearchConfigError("search.horizons must not be empty")
    max_features = tuple(
        _integer(value, "search.max_features", minimum=4, maximum=500)
        for value in search.get("max_features", [])
    )
    estimators = tuple(
        _integer(value, "search.estimators", minimum=20, maximum=2_000)
        for value in search.get("estimators", [])
    )
    learning_rates = tuple(
        _number(value, "search.learning_rates", minimum=0.001, maximum=0.5)
        for value in search.get("learning_rates", [])
    )
    if not max_features or not estimators or not learning_rates:
        raise MlResearchConfigError("numeric search grids must not be empty")
    min_train_rows = _integer(
        budgets.get("min_train_rows"), "budgets.min_train_rows", minimum=200, maximum=2_000_000
    )
    validation_rows = _integer(
        budgets.get("validation_rows"), "budgets.validation_rows", minimum=100, maximum=500_000
    )
    min_windows = _integer(budgets.get("min_windows"), "budgets.min_windows", minimum=2, maximum=20)
    max_rows = _integer(
        budgets.get("max_rows"), "budgets.max_rows", minimum=1_000, maximum=5_000_000
    )
    if min_train_rows + validation_rows * min_windows > max_rows:
        raise MlResearchConfigError("max_rows cannot satisfy the requested chronological windows")
    maximum_drawdown = _number(
        gates.get("maximum_drawdown"), "gates.maximum_drawdown", minimum=-1, maximum=0
    )
    raw_memory_path = payload.get("memory_path")
    if not isinstance(raw_memory_path, str) or not raw_memory_path:
        raise MlResearchConfigError("memory_path must be a non-empty project-relative path")
    if Path(raw_memory_path).is_absolute():
        raise MlResearchConfigError("memory_path must be project-relative")
    memory_path = (PROJECT_ROOT / raw_memory_path).resolve(strict=False)
    try:
        memory_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise MlResearchConfigError("memory_path must stay inside the project") from exc
    return MlResearchConfig(
        path=path.resolve(),
        memory_path=memory_path,
        datasets=tuple(datasets),
        max_trials_per_cycle=_integer(
            budgets.get("max_trials_per_cycle"),
            "budgets.max_trials_per_cycle",
            minimum=1,
            maximum=20,
        ),
        max_runtime_seconds=_integer(
            budgets.get("max_runtime_seconds"),
            "budgets.max_runtime_seconds",
            minimum=10,
            maximum=3_600,
        ),
        max_rows=max_rows,
        min_train_rows=min_train_rows,
        validation_rows=validation_rows,
        step_rows=_integer(
            budgets.get("step_rows"), "budgets.step_rows", minimum=50, maximum=500_000
        ),
        min_windows=min_windows,
        final_holdout_fraction=_number(
            budgets.get("final_holdout_fraction"),
            "budgets.final_holdout_fraction",
            minimum=0.1,
            maximum=0.4,
        ),
        embargo_bars=_integer(
            budgets.get("embargo_bars"), "budgets.embargo_bars", minimum=0, maximum=10_000
        ),
        feature_sets=_strings(
            search.get("feature_sets"),
            "search.feature_sets",
            {"price_volume", "technical_core", "technical_wide"},
        ),
        families=_strings(
            search.get("families"),
            "search.families",
            {
                "classifier_direction",
                "classifier_triple_barrier",
                "regressor_return",
                "regressor_triple_barrier",
            },
        ),
        models=_strings(search.get("models"), "search.models", {"lightgbm", "sklearn"}),
        horizons=horizons,
        regimes=_strings(
            search.get("regimes"),
            "search.regimes",
            {"all", "trend", "high_volatility", "low_volatility"},
        ),
        max_features=max_features,
        estimators=estimators,
        learning_rates=learning_rates,
        minimum_total_trades=_integer(
            gates.get("minimum_total_trades"),
            "gates.minimum_total_trades",
            minimum=1,
            maximum=100_000,
        ),
        minimum_holdout_trades=_integer(
            gates.get("minimum_holdout_trades"),
            "gates.minimum_holdout_trades",
            minimum=1,
            maximum=100_000,
        ),
        minimum_profitable_window_fraction=_number(
            gates.get("minimum_profitable_window_fraction"),
            "gates.minimum_profitable_window_fraction",
            minimum=0.5,
            maximum=1,
        ),
        maximum_drawdown=maximum_drawdown,
        minimum_dsr=_number(gates.get("minimum_dsr"), "gates.minimum_dsr", minimum=0, maximum=1),
    )


def chronological_windows(
    n_rows: int, config: MlResearchConfig, horizon: int
) -> list[tuple[slice, slice]]:
    """Return expanding train and forward validation slices with purge/embargo."""
    windows: list[tuple[slice, slice]] = []
    validation_start = config.min_train_rows + horizon + config.embargo_bars
    while validation_start + config.validation_rows <= n_rows:
        train_end = validation_start - horizon - config.embargo_bars
        windows.append(
            (
                slice(0, train_end),
                slice(validation_start, validation_start + config.validation_rows),
            )
        )
        validation_start += config.step_rows
    return windows[-config.min_windows :] if len(windows) >= config.min_windows else []


def experiment_grid(config: MlResearchConfig) -> list[dict[str, Any]]:
    """Build a stable search grid; rotation state decides which trials run."""
    experiments: list[dict[str, Any]] = []
    values = itertools.product(
        config.datasets,
        config.feature_sets,
        config.families,
        config.models,
        config.horizons,
        config.regimes,
        config.max_features,
        config.estimators,
        config.learning_rates,
    )
    for dataset, feature_set, family, model, horizon, regime, features, estimators, rate in values:
        spec = {
            "product": dataset.product,
            "market": dataset.market,
            "symbol": dataset.symbol,
            "timeframe": dataset.timeframe,
            "pnl_unit": dataset.pnl_unit,
            "feature_set": feature_set,
            "family": family,
            "model": model,
            "horizon": horizon,
            "regime": regime,
            "max_features": features,
            "n_estimators": estimators,
            "learning_rate": rate,
        }
        encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        spec["experiment_id"] = "ml-" + hashlib.sha256(encoded.encode()).hexdigest()[:20]
        experiments.append(spec)
    return experiments


def _load_state(path: Path, grid_size: int) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cursor = int(payload.get("cursor", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0
    return cursor % max(1, grid_size)


def _load_dataset(dataset: DatasetSpec, max_rows: int) -> pd.DataFrame:
    path = dataset.path
    if not path.exists() or path.is_symlink():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).tail(max_rows).copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame = frame.set_index("timestamp")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("dataset index must be unique and chronological")
    return frame


def _feature_columns(frame: pd.DataFrame, feature_set: str) -> list[str] | None:
    numeric = list(frame.select_dtypes(include="number").columns)
    if feature_set == "technical_wide":
        return None
    if feature_set == "price_volume":
        tokens = ("open", "high", "low", "close", "volume", "return", "vwap")
    else:
        tokens = ("rsi", "roc", "atr", "ema", "sma", "zscore", "macd", "adx", "boll", "volume")
    forbidden = ("future_return", "label_", "bars_to_exit", "target")
    selected = [
        name
        for name in numeric
        if any(token in name.lower() for token in tokens)
        and not any(token in name.lower() for token in forbidden)
    ]
    return selected or None


def _strategy(spec: Mapping[str, Any], frame: pd.DataFrame) -> Strategy:
    common = {
        "horizon": int(spec["horizon"]),
        "model": str(spec["model"]),
        "feature_cols": _feature_columns(frame, str(spec["feature_set"])),
        "max_features": int(spec["max_features"]),
        "n_estimators": int(spec["n_estimators"]),
        "learning_rate": float(spec["learning_rate"]),
        "n_jobs": 1,
    }
    family = str(spec["family"])
    if family.startswith("classifier"):
        return MlClassifierStrategy(
            **common,
            label_mode="triple_barrier" if family.endswith("triple_barrier") else "direction",
        )
    return MlRegressorStrategy(
        **common,
        target_mode="triple_barrier" if family.endswith("triple_barrier") else "forward_return",
    )


def _regime_close_feature(frame: pd.DataFrame) -> str:
    close_columns = [name for name in frame.columns if name == "close" or name.endswith("_close")]
    if not close_columns:
        raise ValueError("regime gating requires a close column")
    return close_columns[0]


def _regime_mask(
    frame: pd.DataFrame,
    regime: str,
    close_feature: str | None = None,
) -> pd.Series:
    if regime == "all":
        return pd.Series(True, index=frame.index)
    selected_close = close_feature or _regime_close_feature(frame)
    if selected_close not in frame.columns:
        raise ValueError(f"regime close feature is missing: {selected_close}")
    returns = frame[selected_close].astype(float).pct_change()
    volatility = returns.rolling(48, min_periods=24).std()
    baseline = volatility.rolling(480, min_periods=96).median()
    if regime == "high_volatility":
        return (volatility >= baseline).fillna(False)
    if regime == "low_volatility":
        return (volatility < baseline).fillna(False)
    trend = returns.rolling(48, min_periods=24).sum().abs()
    return (trend >= volatility * math.sqrt(48)).fillna(False)


class _StaticSignalStrategy(Strategy):
    name = "ml_research_static_signal"

    def __init__(self, signals: pd.Series):
        super().__init__()
        self.signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return self.signals.reindex(df.index).fillna(0).astype(int)


def _finite_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = None
        elif isinstance(value, np.integer | np.floating):
            result[key] = value.item() if math.isfinite(float(value)) else None
        else:
            result[key] = value
    return result


def frame_content_sha256(frame: pd.DataFrame) -> str:
    """Hash an exact chronological training slice without a model pickle."""
    hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    return hashlib.sha256(hashes.to_numpy().tobytes()).hexdigest()


def evaluate_experiment(
    spec: Mapping[str, Any],
    dataset: DatasetSpec,
    frame: pd.DataFrame,
    config: MlResearchConfig,
) -> dict[str, Any]:
    holdout_rows = max(1, math.ceil(len(frame) * config.final_holdout_fraction))
    adaptive = frame.iloc[:-holdout_rows]
    windows = chronological_windows(len(adaptive), config, int(spec["horizon"]))
    if not windows:
        return {
            **spec,
            "status": "waiting_for_history",
            "adaptive_rows": len(adaptive),
            "reserved_holdout_rows": holdout_rows,
            "windows": [],
            "pre_holdout_eligible": False,
        }
    results: list[dict[str, Any]] = []
    for train_slice, validation_slice in windows:
        train, validation = adaptive.iloc[train_slice], adaptive.iloc[validation_slice]
        model = _strategy(spec, train).fit(train)
        raw_signals = model.generate_signals(validation)
        signals = raw_signals.where(_regime_mask(validation, str(spec["regime"])), 0)
        backtest = run_backtest(
            _StaticSignalStrategy(signals),
            validation,
            config=BacktestConfig(
                fee_bps=10,
                slippage_bps=2,
                take_profit=0.05,
                stop_loss=0.03,
                horizon_bars=int(spec["horizon"]),
                pnl_unit=dataset.pnl_unit,
            ),
            base_tf=dataset.timeframe,
        )
        results.append(
            {
                **_finite_metrics(backtest.summary()),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_end": str(train.index[-1]),
                "validation_start": str(validation.index[0]),
                "validation_end": str(validation.index[-1]),
            }
        )
    totals = [float(item.get("total_return") or 0) for item in results]
    drawdowns = [float(item.get("max_drawdown") or 0) for item in results]
    trades = sum(int(item.get("trades") or 0) for item in results)
    profitable_fraction = sum(value > 0 for value in totals) / len(totals)
    eligible = (
        trades >= config.minimum_total_trades
        and profitable_fraction >= config.minimum_profitable_window_fraction
        and min(drawdowns, default=0) >= config.maximum_drawdown
        and float(np.median(totals)) > 0
    )
    return {
        **spec,
        "status": "pre_holdout_pass" if eligible else "pre_holdout_reject",
        "adaptive_rows": len(adaptive),
        "reserved_holdout_rows": holdout_rows,
        "reserved_holdout_policy": "unfit_unscored_tail",
        "windows": results,
        "aggregate": {
            "windows": len(results),
            "trades": trades,
            "profitable_window_fraction": profitable_fraction,
            "median_total_return": float(np.median(totals)),
            "worst_max_drawdown": min(drawdowns, default=0),
        },
        "pre_holdout_eligible": eligible,
        "safety": {
            "research_only": True,
            "paper_trade_allowed": False,
            "promotion_allowed": False,
            "live_allowed": False,
            "blocked_reason": "protected_holdout_and_forward_paper_not_completed",
        },
    }


def _remember_evaluation(
    config: MlResearchConfig,
    spec: Mapping[str, Any],
    dataset: DatasetSpec,
    frame: pd.DataFrame,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist an adaptive ML identity and immutable evaluation context."""
    holdout_rows = int(result["reserved_holdout_rows"])
    adaptive = frame.iloc[:-holdout_rows]
    training = adaptive.iloc[: max(0, len(adaptive) - config.embargo_bars)]
    source_stat = dataset.path.stat()
    evidence = {
        "path": str(dataset.path.resolve()),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "rows": len(frame),
        "adaptive_rows": len(adaptive),
        "adaptive_start": str(adaptive.index[0]),
        "adaptive_end": str(adaptive.index[-1]),
        "content_sha256": frame_content_sha256(frame),
        "training_content_sha256": frame_content_sha256(training),
    }
    snapshot_digest = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    behavior = {key: value for key, value in spec.items() if key != "experiment_id"}
    behavior["research_contract"] = "chronological_pre_holdout_ml/v1"
    dataset_context = {
        "snapshot_id": f"ml:{snapshot_digest}",
        "symbol": dataset.symbol,
        "market": dataset.market,
        "timeframe": dataset.timeframe,
        "evidence": evidence,
    }
    development_window = {
        "start": str(adaptive.index[0]),
        "end": str(adaptive.index[-1]),
        "purge_bars": int(spec["horizon"]),
        "embargo_bars": config.embargo_bars,
    }
    protocol = {
        "contract": "chronological_pre_holdout_ml/v1",
        "reserved_holdout_fraction": config.final_holdout_fraction,
        "fee_bps": 10,
        "slippage_bps": 2,
        "walk_forward_windows": len(result.get("windows") or []),
    }
    holdout = frame.iloc[-holdout_rows:]
    holdout_window = {
        "start": str(holdout.index[0]),
        "end": str(holdout.index[-1]),
        "purge_bars": int(spec["horizon"]),
        "embargo_bars": config.embargo_bars,
    }
    with ExperimentMemory(config.memory_path) as memory:
        registration = memory.register_strategy(
            behavior,
            strategy_id=str(spec["experiment_id"]),
            generation_method="bounded_ml_grid",
            metadata={
                "family": f"ml_{spec['family']}",
                "product": dataset.product,
                "opportunity_type": "machine_learning",
                "market": dataset.market,
                "symbol": dataset.symbol,
            },
        )
        eligible = result.get("pre_holdout_eligible") is True
        recorded = memory.record_outcome(
            registration.behavior_hash,
            dataset=dataset_context,
            window=development_window,
            protocol=protocol,
            phase="development",
            outcome="pre_holdout_pass" if eligible else "reject",
            rejection_reasons=() if eligible else ("ml_pre_holdout_gates_failed",),
            metrics=result.get("aggregate") if isinstance(result.get("aggregate"), Mapping) else {},
            details={"windows": result.get("windows") or []},
        )
    return {
        "behavior_hash": registration.behavior_hash,
        "evaluation_key": recorded.evaluation_key,
        "strategy_created": registration.created,
        "evaluation_created": recorded.created,
        "snapshot_id": dataset_context["snapshot_id"],
        "dataset": dataset_context,
        "protocol": protocol,
        "holdout_window": holdout_window,
    }


def _score_protected_holdout(
    spec: Mapping[str, Any],
    dataset: DatasetSpec,
    frame: pd.DataFrame,
    config: MlResearchConfig,
) -> dict[str, Any]:
    holdout_rows = max(1, math.ceil(len(frame) * config.final_holdout_fraction))
    adaptive = frame.iloc[:-holdout_rows]
    train_end = max(0, len(adaptive) - config.embargo_bars)
    train = adaptive.iloc[:train_end]
    holdout = frame.iloc[-holdout_rows:]
    regime_close_feature = _regime_close_feature(frame)
    model = _strategy(spec, train).fit(train)
    frozen_model = (
        export_sklearn_gradient_boosting(model) if str(spec["model"]) == "sklearn" else None
    )
    signals = model.generate_signals(holdout).where(
        _regime_mask(holdout, str(spec["regime"]), regime_close_feature), 0
    )
    backtest = run_backtest(
        _StaticSignalStrategy(signals),
        holdout,
        config=BacktestConfig(
            fee_bps=10,
            slippage_bps=2,
            take_profit=0.05,
            stop_loss=0.03,
            horizon_bars=int(spec["horizon"]),
            pnl_unit=dataset.pnl_unit,
        ),
        base_tf=dataset.timeframe,
    )
    summary = _finite_metrics(backtest.summary())
    returns = backtest.returns
    trial_count = len(experiment_grid(config))
    conservative_floor = max(
        MIN_TRIAL_SHARPE_STD,
        1.0 / math.sqrt(max(len(returns) - 1, 1)),
    )
    observed_dispersion = 0.0
    dispersion = max(observed_dispersion, conservative_floor)
    if len(returns) > 3:
        series = pd.Series(returns)
        skew = float(series.skew())
        kurt = float(series.kurt() + 3.0)
        dsr = float(
            performance_metrics.deflated_sharpe_ratio(
                performance_metrics.sharpe_ratio(returns),
                n_trials=trial_count,
                skew=skew if math.isfinite(skew) else 0.0,
                kurt=kurt if math.isfinite(kurt) else 3.0,
                n_obs=len(returns),
                sr_std_trials=dispersion,
            )
        )
    else:
        dsr = 0.0
    summary.update(
        dsr_deflated=dsr,
        dsr_method=DSR_METHOD,
        n_trials=trial_count,
        sr_std_trials=dispersion,
        trial_sharpe_count=1,
        trial_sharpe_observed_std=observed_dispersion,
        trial_sharpe_conservative_floor=conservative_floor,
    )
    trades = int(summary.get("trades") or 0)
    total_return = float(summary.get("total_return") or 0.0)
    drawdown = float(summary.get("max_drawdown") or 0.0)
    eligible = (
        trades >= config.minimum_holdout_trades
        and total_return > 0
        and drawdown >= config.maximum_drawdown
        and dsr >= config.minimum_dsr
    )
    return {
        "eligible": eligible,
        "metrics": summary,
        "train_start": str(train.index[0]),
        "train_end": str(train.index[-1]),
        "holdout_start": str(holdout.index[0]),
        "holdout_end": str(holdout.index[-1]),
        "regime_close_feature": regime_close_feature,
        "frozen_model": frozen_model,
    }


def _evaluate_protected_cohort(
    config: MlResearchConfig,
    candidates: list[tuple[dict[str, Any], DatasetSpec, pd.DataFrame, dict[str, Any]]],
) -> None:
    if not candidates:
        return
    first_context = candidates[0][3]
    try:
        with ExperimentMemory(config.memory_path) as memory:
            cohort = memory.register_holdout_cohort(
                [context["behavior_hash"] for _, _, _, context in candidates],
                dataset=first_context["dataset"],
                window=first_context["holdout_window"],
                protocol=first_context["protocol"],
                min_seconds_since_last_seal=86_400,
            )
            for result, dataset, frame, context in candidates:
                if context["behavior_hash"] not in cohort.member_hashes:
                    result["holdout_status"] = "deferred_not_in_sealed_cohort"
                    continue
                if memory.holdout_claimed(
                    context["behavior_hash"], snapshot_id=context["snapshot_id"]
                ):
                    result["holdout_status"] = "deferred_already_consumed"
                    continue
                claim = memory.claim_holdout(
                    context["behavior_hash"],
                    snapshot_id=context["snapshot_id"],
                    dataset=context["dataset"],
                    window=context["holdout_window"],
                    protocol=context["protocol"],
                )
                if not claim.created:
                    result["holdout_status"] = "deferred_already_claimed"
                    continue
                try:
                    holdout = _score_protected_holdout(result, dataset, frame, config)
                except Exception as exc:
                    result["holdout_status"] = "protected_holdout_error"
                    result["status"] = "error"
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    memory.complete_evaluation(
                        claim.evaluation_key,
                        outcome="error",
                        rejection_reasons=("ml_protected_holdout_error",),
                        metrics={},
                        details={"error": result["error"], "protected_feedback": True},
                    )
                    continue
                frozen_model = holdout.pop("frozen_model", None)
                result["protected_holdout"] = holdout
                result["holdout_eligible"] = holdout["eligible"]
                forward_paper_allowed = holdout["eligible"] and isinstance(frozen_model, dict)
                result["holdout_status"] = (
                    "protected_holdout_pass" if holdout["eligible"] else "protected_holdout_reject"
                )
                result["status"] = result["holdout_status"]
                result["safety"] = {
                    "research_only": True,
                    "forward_paper_allowed": forward_paper_allowed,
                    "promotion_allowed": False,
                    "live_allowed": False,
                    "blocked_reason": (
                        "exact_digest_staging_candidate_paper_and_approval_required"
                    ),
                }
                if forward_paper_allowed:
                    result["forward_paper_candidate"] = {
                        "schema": "autopilot.ml_forward_paper_candidate/v1",
                        "experiment_id": result["experiment_id"],
                        "behavior_hash": context["behavior_hash"],
                        "snapshot_id": context["snapshot_id"],
                        "spec": {
                            key: value
                            for key, value in result.items()
                            if key
                            in {
                                "product",
                                "market",
                                "symbol",
                                "timeframe",
                                "pnl_unit",
                                "feature_set",
                                "family",
                                "model",
                                "horizon",
                                "regime",
                                "max_features",
                                "n_estimators",
                                "learning_rate",
                                "experiment_id",
                            }
                        },
                        "training_content_sha256": context["dataset"]["evidence"][
                            "training_content_sha256"
                        ],
                        "training_start": holdout["train_start"],
                        "training_end": holdout["train_end"],
                        "forward_start_after": holdout["holdout_end"],
                        "frozen_model": frozen_model,
                        "promotion_eligible": False,
                        "live_allowed": False,
                    }
                    result["forward_paper_candidate"]["spec"]["regime_close_feature"] = holdout[
                        "regime_close_feature"
                    ]
                memory.complete_evaluation(
                    claim.evaluation_key,
                    outcome="pass" if holdout["eligible"] else "reject",
                    rejection_reasons=()
                    if holdout["eligible"]
                    else ("ml_protected_holdout_gates_failed",),
                    metrics=holdout["metrics"],
                    details={"protected_feedback": True},
                )
    except HoldoutSealBudgetError:
        for result, _, _, _ in candidates:
            result["holdout_status"] = "deferred_holdout_seal_budget"
    except EvaluationConflictError as exc:
        for result, _, _, _ in candidates:
            result["holdout_status"] = "deferred_holdout_scope_conflict"
            result["holdout_detail"] = str(exc)


def run_cycle(
    config: MlResearchConfig,
    *,
    output_path: Path = DEFAULT_OUTPUT,
    state_path: Path = DEFAULT_STATE,
    autopilot_config_path: Path = DEFAULT_AUTOPILOT_CONFIG,
    candidate_artifact_dir: Path = DEFAULT_REVIEWABLE_DIR,
) -> dict[str, Any]:
    started = time.monotonic()
    grid = experiment_grid(config)
    cursor = _load_state(state_path, len(grid))
    selected = [grid[(cursor + index) % len(grid)] for index in range(config.max_trials_per_cycle)]
    by_identity = {
        (item.product, item.market, item.symbol, item.timeframe, item.pnl_unit): item
        for item in config.datasets
    }
    cache: dict[tuple[str, str, str, str, str], pd.DataFrame] = {}
    trials: list[dict[str, Any]] = []
    holdout_candidates: dict[
        str, list[tuple[dict[str, Any], DatasetSpec, pd.DataFrame, dict[str, Any]]]
    ] = {}
    for spec in selected:
        if time.monotonic() - started >= config.max_runtime_seconds:
            break
        identity = tuple(
            spec[key] for key in ("product", "market", "symbol", "timeframe", "pnl_unit")
        )
        dataset = by_identity[identity]
        try:
            if identity not in cache:
                cache[identity] = _load_dataset(dataset, config.max_rows)
            frame = cache[identity]
            result = evaluate_experiment(spec, dataset, frame, config)
            if result["status"] in {"pre_holdout_pass", "pre_holdout_reject"}:
                context = _remember_evaluation(config, spec, dataset, frame, result)
                result["experiment_memory"] = {
                    key: context[key]
                    for key in (
                        "behavior_hash",
                        "evaluation_key",
                        "strategy_created",
                        "evaluation_created",
                        "snapshot_id",
                    )
                }
                if result.get("pre_holdout_eligible") is True:
                    holdout_candidates.setdefault(context["snapshot_id"], []).append(
                        (result, dataset, frame, context)
                    )
            trials.append(result)
        except FileNotFoundError:
            trials.append(
                {
                    **spec,
                    "status": "waiting_for_dataset",
                    "path": str(dataset.path),
                    "pre_holdout_eligible": False,
                }
            )
        except ImportError as exc:
            trials.append(
                {
                    **spec,
                    "status": "waiting_for_dependency",
                    "dependency": str(spec["model"]),
                    "detail": str(exc),
                    "pre_holdout_eligible": False,
                }
            )
        except Exception as exc:
            trials.append(
                {
                    **spec,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pre_holdout_eligible": False,
                }
            )
    for cohort_candidates in holdout_candidates.values():
        _evaluate_protected_cohort(config, cohort_candidates)
    exportable = [item for item in trials if isinstance(item.get("forward_paper_candidate"), dict)]
    if exportable:
        products = {
            product.name: product
            for product in load_autopilot_config(autopilot_config_path).products
        }
        for trial in exportable:
            product = products.get(str(trial.get("product") or ""))
            if product is None:
                trial["candidate_artifact_status"] = "error"
                trial["candidate_artifact_error"] = "ML product is not configured in autopilot"
                trial["status"] = "error"
                continue
            try:
                trial["reviewable_candidate_artifact"] = export_reviewable_artifact(
                    trial,
                    product,
                    output_dir=candidate_artifact_dir,
                )
                trial["candidate_artifact_status"] = "reviewable_not_staged"
            except Exception as exc:
                trial["candidate_artifact_status"] = "error"
                trial["candidate_artifact_error"] = f"{type(exc).__name__}: {exc}"
                trial["status"] = "error"
    next_cursor = (cursor + len(trials)) % len(grid)
    report = {
        "schema": REPORT_SCHEMA,
        "ok": not any(item["status"] == "error" for item in trials),
        "generated_at": _utc_now(),
        "grid_size": len(grid),
        "cursor": cursor,
        "next_cursor": next_cursor,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "trials": trials,
        "summary": {
            "attempted": len(trials),
            "waiting": sum(str(item["status"]).startswith("waiting") for item in trials),
            "errors": sum(item["status"] == "error" for item in trials),
            "pre_holdout_passes": sum(item.get("pre_holdout_eligible") is True for item in trials),
            "protected_holdout_passes": sum(
                item.get("holdout_eligible") is True for item in trials
            ),
            "protected_holdout_rejects": sum(
                item.get("holdout_status") == "protected_holdout_reject" for item in trials
            ),
            "forward_paper_candidates": sum(
                isinstance(item.get("forward_paper_candidate"), dict) for item in trials
            ),
            "reviewable_candidate_artifacts": sum(
                isinstance(item.get("reviewable_candidate_artifact"), dict) for item in trials
            ),
        },
    }
    write_json_atomic(output_path, report)
    write_json_atomic(
        state_path,
        {
            "schema": "autopilot.ml_research_state/v1",
            "cursor": next_cursor,
            "updated_at": report["generated_at"],
        },
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--autopilot-config", type=Path, default=DEFAULT_AUTOPILOT_CONFIG)
    parser.add_argument("--candidate-artifact-dir", type=Path, default=DEFAULT_REVIEWABLE_DIR)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.validate:
        print(
            json.dumps(
                {"ok": True, "schema": CONFIG_SCHEMA, "grid_size": len(experiment_grid(config))},
                sort_keys=True,
            )
        )
        return
    report = run_cycle(
        config,
        output_path=args.output,
        state_path=args.state,
        autopilot_config_path=args.autopilot_config,
        candidate_artifact_dir=args.candidate_artifact_dir,
    )
    print(json.dumps(report["summary"], sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
