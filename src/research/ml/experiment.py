"""CPU-bounded chronological scikit-learn and LightGBM experiments."""

from __future__ import annotations

import io
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import model_artifact
from src.domain._codec import canonical_hash, json_value, timestamp

CLASSIFICATION_MODELS = frozenset(
    {
        "logistic_regression",
        "elastic_net_classifier",
        "gradient_boosting_classifier",
        "random_forest_classifier",
        "calibrated_classifier",
        "lightgbm_classifier",
    }
)
REGRESSION_MODELS = frozenset(
    {
        "linear_regression",
        "elastic_net_regressor",
        "gradient_boosting_regressor",
        "random_forest_regressor",
        "lightgbm_regressor",
    }
)


@dataclass(frozen=True)
class MlExperimentResult:
    model_artifact_id: str
    content_hash: str
    relative_path: str
    dataset_hash: str
    metrics: Mapping[str, float]
    train_rows: int
    validation_rows: int


class ModelArtefactStore:
    """Immutable content-addressed model files."""

    def __init__(self, root: Path):
        self.root = root

    def put(self, payload: bytes) -> tuple[Path, str]:
        if not payload:
            raise ValueError("model payload cannot be empty")
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        content_hash = f"sha256:{digest}"
        destination = self.root / digest[:2] / f"{digest}.joblib"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != payload:
                raise RuntimeError("immutable model artefact hash collision")
            return destination, content_hash
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_bytes() != payload:
                    raise RuntimeError("immutable model artefact hash collision") from None
        finally:
            temporary.unlink(missing_ok=True)
        return destination, content_hash


class SqlModelArtefactStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def save(self, *, artifact_id: str, created_at: str, payload: Mapping[str, Any]) -> None:
        created_at = timestamp(created_at, field="created_at")
        clean = json_value(dict(payload), field="model artefact")
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(model_artifact.c.payload).where(model_artifact.c.id == artifact_id)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != clean:
                    raise ValueError("model artefact identity collision")
                return
            connection.execute(
                insert(model_artifact).values(
                    id=artifact_id,
                    created_at=created_at,
                    payload=clean,
                )
            )


class MlExperimentRunner:
    """Fit one deterministic model with a purged chronological validation split."""

    def __init__(
        self,
        *,
        artefact_store: ModelArtefactStore,
        metadata_store: SqlModelArtefactStore,
        maximum_rows: int = 250_000,
    ) -> None:
        if maximum_rows < 20:
            raise ValueError("maximum_rows must be at least 20")
        self.artefact_store = artefact_store
        self.metadata_store = metadata_store
        self.maximum_rows = maximum_rows

    def run(
        self,
        *,
        candidate_id: str,
        model_name: str,
        feature_names: Sequence[str],
        target_name: str,
        rows: Sequence[Mapping[str, Any]],
        created_at: str,
        train_fraction: float = 0.7,
        embargo_rows: int = 1,
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> MlExperimentResult:
        if model_name not in CLASSIFICATION_MODELS | REGRESSION_MODELS:
            raise ValueError(f"unsupported ML model: {model_name}")
        features = tuple(str(name) for name in feature_names)
        if not features or len(features) != len(set(features)) or not target_name:
            raise ValueError("ML features and target must be unique and non-empty")
        if not 0.5 <= train_fraction <= 0.9:
            raise ValueError("train_fraction must be in [0.5, 0.9]")
        if not isinstance(embargo_rows, int) or not 0 <= embargo_rows <= 10_000:
            raise ValueError("embargo_rows must be an integer in [0, 10000]")
        materialised = tuple(dict(row) for row in rows)
        if not 20 <= len(materialised) <= self.maximum_rows:
            raise ValueError("ML rows exceed the bounded experiment size")
        times = [timestamp(str(row["available_at"]), field="available_at") for row in materialised]
        if any(times[index] <= times[index - 1] for index in range(1, len(times))):
            raise ValueError("ML rows must have strictly chronological availability timestamps")
        matrix = np.asarray(
            [[_finite(row[name], field=name) for name in features] for row in materialised],
            dtype=np.float64,
        )
        target = np.asarray(
            [_finite(row[target_name], field=target_name) for row in materialised],
            dtype=np.float64,
        )
        validation_start = int(len(materialised) * train_fraction)
        training_end = validation_start - embargo_rows
        if training_end < 10 or len(materialised) - validation_start < 5:
            raise ValueError("chronological split leaves insufficient train or validation rows")
        model = _build_model(model_name, dict(hyperparameters or {}))
        model.fit(matrix[:training_end], target[:training_end])
        predictions = np.asarray(model.predict(matrix[validation_start:]), dtype=np.float64)
        metrics = _metrics(
            model_name=model_name,
            model=model,
            matrix=matrix[validation_start:],
            target=target[validation_start:],
            predictions=predictions,
        )
        dataset_hash = canonical_hash(
            {
                "feature_names": features,
                "target_name": target_name,
                "rows": materialised,
            }
        )
        bundle = {
            "schema": "platform.ml_model/v1",
            "candidate_id": candidate_id,
            "model_name": model_name,
            "feature_names": features,
            "target_name": target_name,
            "dataset_hash": dataset_hash,
            "train_start": times[0],
            "train_end": times[training_end - 1],
            "validation_start": times[validation_start],
            "validation_end": times[-1],
            "embargo_rows": embargo_rows,
            "hyperparameters": _normalised_parameters(model),
            "model": model,
        }
        buffer = io.BytesIO()
        joblib.dump(bundle, buffer, compress=3)
        path, content_hash = self.artefact_store.put(buffer.getvalue())
        artifact_id = canonical_hash(
            {
                "candidate_id": candidate_id,
                "content_hash": content_hash,
                "dataset_hash": dataset_hash,
            }
        )
        relative_path = str(path.relative_to(self.artefact_store.root))
        self.metadata_store.save(
            artifact_id=artifact_id,
            created_at=created_at,
            payload={
                "candidate_id": candidate_id,
                "model_name": model_name,
                "content_hash": content_hash,
                "relative_path": relative_path,
                "dataset_hash": dataset_hash,
                "feature_names": list(features),
                "target_name": target_name,
                "train_rows": training_end,
                "validation_rows": len(materialised) - validation_start,
                "train_start": times[0],
                "train_end": times[training_end - 1],
                "validation_start": times[validation_start],
                "validation_end": times[-1],
                "embargo_rows": embargo_rows,
                "metrics": metrics,
            },
        )
        return MlExperimentResult(
            model_artifact_id=artifact_id,
            content_hash=content_hash,
            relative_path=relative_path,
            dataset_hash=dataset_hash,
            metrics=metrics,
            train_rows=training_end,
            validation_rows=len(materialised) - validation_start,
        )


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _bounded_parameters(values: Mapping[str, Any]) -> dict[str, float | int]:
    allowed = {"alpha", "C", "l1_ratio", "learning_rate", "max_depth", "n_estimators"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported ML hyperparameters: {sorted(unknown)}")
    result: dict[str, float | int] = {}
    for name, value in values.items():
        number = _finite(value, field=name)
        if name in {"max_depth", "n_estimators"}:
            integer = int(number)
            if integer != number:
                raise ValueError(f"{name} must be an integer")
            maximum = 12 if name == "max_depth" else 1_000
            if not 1 <= integer <= maximum:
                raise ValueError(f"{name} must be in [1, {maximum}]")
            result[name] = integer
        else:
            upper = 1.0 if name in {"l1_ratio", "learning_rate"} else 1_000.0
            lower = 0.0 if name == "l1_ratio" else 1e-12
            if not lower <= number <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
            result[name] = number
    return result


def _build_model(model_name: str, raw_parameters: Mapping[str, Any]):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression

    values = _bounded_parameters(raw_parameters)
    if model_name == "logistic_regression":
        return LogisticRegression(C=float(values.get("C", 1.0)), random_state=0, max_iter=1_000)
    if model_name == "elastic_net_classifier":
        return LogisticRegression(
            C=float(values.get("C", 1.0)),
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(values.get("l1_ratio", 0.5)),
            random_state=0,
            max_iter=2_000,
            n_jobs=1,
        )
    if model_name == "gradient_boosting_classifier":
        return GradientBoostingClassifier(
            learning_rate=float(values.get("learning_rate", 0.05)),
            max_depth=int(values.get("max_depth", 3)),
            n_estimators=int(values.get("n_estimators", 100)),
            random_state=0,
        )
    if model_name == "random_forest_classifier":
        return RandomForestClassifier(
            max_depth=int(values.get("max_depth", 6)),
            n_estimators=int(values.get("n_estimators", 200)),
            random_state=0,
            n_jobs=1,
        )
    if model_name == "calibrated_classifier":
        estimator = LogisticRegression(C=float(values.get("C", 1.0)), random_state=0)
        return CalibratedClassifierCV(estimator, method="sigmoid", cv=3, n_jobs=1)
    if model_name == "linear_regression":
        return LinearRegression(n_jobs=1)
    if model_name == "elastic_net_regressor":
        return ElasticNet(
            alpha=float(values.get("alpha", 1.0)),
            l1_ratio=float(values.get("l1_ratio", 0.5)),
            random_state=0,
            max_iter=2_000,
        )
    if model_name == "gradient_boosting_regressor":
        return GradientBoostingRegressor(
            learning_rate=float(values.get("learning_rate", 0.05)),
            max_depth=int(values.get("max_depth", 3)),
            n_estimators=int(values.get("n_estimators", 100)),
            random_state=0,
        )
    if model_name == "random_forest_regressor":
        return RandomForestRegressor(
            max_depth=int(values.get("max_depth", 6)),
            n_estimators=int(values.get("n_estimators", 200)),
            random_state=0,
            n_jobs=1,
        )
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise ImportError("LightGBM is required for this model") from exc
    common = {
        "learning_rate": float(values.get("learning_rate", 0.05)),
        "max_depth": int(values.get("max_depth", 6)),
        "n_estimators": int(values.get("n_estimators", 200)),
        "random_state": 0,
        "n_jobs": 1,
        "verbosity": -1,
    }
    return (
        LGBMClassifier(**common) if model_name.endswith("classifier") else LGBMRegressor(**common)
    )


def _metrics(
    *,
    model_name: str,
    model: Any,
    matrix: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    if model_name in CLASSIFICATION_MODELS:
        from sklearn.metrics import accuracy_score, log_loss

        values = {"accuracy": float(accuracy_score(target, predictions))}
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
            values["log_loss"] = float(log_loss(target, probabilities, labels=model.classes_))
        return values
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(target, predictions))),
        "r2": float(r2_score(target, predictions)),
    }


def _normalised_parameters(model: Any) -> dict[str, Any]:
    parameters = model.get_params(deep=False)
    return {
        str(key): value
        for key, value in sorted(parameters.items())
        if isinstance(value, str | int | float | bool) or value is None
    }
