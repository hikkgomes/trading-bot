import logging
from typing import Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _fit_classifier(train_data: pd.DataFrame, feature_columns: Sequence[str], label_column: str) -> lgb.LGBMClassifier:
    x = train_data.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_data[label_column].astype(int)
    val_split = int(len(x) * 0.9)
    if val_split <= 0 or val_split >= len(x):
        raise ValueError("Not enough rows for feature screening")
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x.iloc[:val_split],
        y.iloc[:val_split],
        eval_set=[(x.iloc[val_split:], y.iloc[val_split:])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def screen_features(
    train_data: pd.DataFrame,
    label_column: str,
    feature_columns: Sequence[str],
    max_features: int,
    method: str = "shap",
) -> List[str]:
    model = _fit_classifier(train_data, feature_columns, label_column)
    features = list(feature_columns)
    if method == "shap":
        try:
            import shap
            x = train_data.loc[:, features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if len(x) > 3000:
                idx = np.linspace(0, len(x) - 1, 3000).astype(int)
                x = x.iloc[idx]
            explainer = shap.TreeExplainer(model)
            raw_values = explainer.shap_values(x)
            values = np.asarray(raw_values[-1] if isinstance(raw_values, list) else raw_values)
            scores = np.abs(values).mean(axis=0)
            ranked = pd.Series(scores, index=features).sort_values(ascending=False)
            return ranked.head(max_features).index.tolist()
        except ImportError:
            LOGGER.warning("SHAP not available, falling back to gain-based importance")
    booster = model.booster_
    imp = pd.Series(booster.feature_importance(importance_type="gain"), index=booster.feature_name())
    return imp.sort_values(ascending=False).head(max_features).index.tolist()


def screen_features_per_scenario(
    train_data: pd.DataFrame,
    tp_sl_pairs: Sequence[Tuple[float, float]],
    horizons: Sequence[int],
    directions: Sequence[str],
    feature_columns: Sequence[str],
    max_features: int,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for direction in directions:
        for horizon in horizons:
            for tp, sl in tp_sl_pairs:
                tp_bps = int(round(tp * 10_000))
                sl_bps = int(round(sl * 10_000))
                label_column = f"label_{direction}_tp{tp_bps}_sl{sl_bps}_h{horizon}"
                if label_column not in train_data.columns:
                    continue
                key = f"{direction}|h{horizon}|tp{tp_bps}|sl{sl_bps}"
                result[key] = screen_features(train_data, label_column, feature_columns, max_features)
    return result
