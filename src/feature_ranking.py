import logging
from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def rank_features_by_importance(
    train: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    max_features: int,
    direction: str | None = None,
) -> list[str]:
    feature_list = list(features)
    x = train[feature_list]
    y = train[target_column]
    if direction == "short":
        y = -y

    val_split = int(len(x) * 0.9)
    if val_split <= 0 or val_split >= len(x):
        return feature_list[:max_features]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x.iloc[:val_split],
        y.iloc[:val_split],
        eval_set=[(x.iloc[val_split:], y.iloc[val_split:])],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    importance = pd.Series(gains, index=booster.feature_name())
    ranked = importance.sort_values(ascending=False)
    return ranked.head(max_features).index.tolist()


def _spearman_ranks(
    train: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    direction: str | None = None,
) -> pd.Series:
    target = train[target_column]
    if direction == "short":
        target = -target
    scores = {}
    for column in features:
        series = train[column]
        valid = series.notna() & target.notna()
        if valid.sum() < 1000:
            continue
        corr = series[valid].corr(target[valid], method="spearman")
        if pd.notna(corr):
            scores[column] = abs(float(corr))
    return pd.Series(scores).sort_values(ascending=False)


def rank_features_blended(
    train: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    max_features: int,
    spearman_weight: float = 0.5,
    direction: str | None = None,
) -> list[str]:
    importance_ranked = rank_features_by_importance(
        train,
        features,
        target_column,
        max_features=len(features),
        direction=direction,
    )
    spearman_scores = _spearman_ranks(train, features, target_column, direction=direction)

    importance_rank = pd.Series({f: i for i, f in enumerate(importance_ranked)}, dtype=float)
    spearman_rank = pd.Series({f: i for i, f in enumerate(spearman_scores.index)}, dtype=float)

    all_features = set(importance_rank.index) | set(spearman_rank.index)
    worst = float(len(all_features))

    blended = {}
    importance_weight = 1.0 - spearman_weight
    for f in all_features:
        s_rank = spearman_rank.get(f, worst)
        i_rank = importance_rank.get(f, worst)
        blended[f] = spearman_weight * s_rank + importance_weight * i_rank

    ranked = sorted(blended, key=blended.get)
    return ranked[:max_features]


def suggest_feature_pairs(
    train: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    max_pairs: int,
    top_features_for_shap: int = 100,
    sample_rows: int = 5000,
) -> list[tuple[str, str]]:
    top_features = rank_features_by_importance(
        train,
        features,
        target_column,
        max_features=top_features_for_shap,
    )
    x = train[top_features]
    y = train[target_column]

    if len(x) > sample_rows:
        indices = np.linspace(0, len(x) - 1, sample_rows).astype(int)
        x = x.iloc[indices]
        y = y.iloc[indices]

    val_split = int(len(x) * 0.9)
    if val_split <= 0 or val_split >= len(x):
        return []

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x.iloc[:val_split],
        y.iloc[:val_split],
        eval_set=[(x.iloc[val_split:], y.iloc[val_split:])],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    try:
        import shap
    except ImportError:
        LOGGER.warning("shap not installed, falling back to importance-based pairs")
        return _importance_based_pairs(model, top_features, max_pairs)

    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(x)
    if isinstance(interaction_values, list):
        interaction_values = np.asarray(interaction_values[-1])

    n_features = len(top_features)
    pair_scores = {}
    for i in range(n_features):
        for j in range(i + 1, n_features):
            score = float(np.abs(interaction_values[:, i, j]).mean())
            pair_scores[(top_features[i], top_features[j])] = score

    sorted_pairs = sorted(pair_scores, key=pair_scores.get, reverse=True)
    return sorted_pairs[:max_pairs]


def _importance_based_pairs(
    model: lgb.LGBMRegressor,
    features: list[str],
    max_pairs: int,
) -> list[tuple[str, str]]:
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    ranked_indices = np.argsort(gains)[::-1]

    pairs: list[tuple[str, str]] = []
    for i_pos, i in enumerate(ranked_indices):
        for j in ranked_indices[i_pos + 1 :]:
            pairs.append((features[i], features[j]))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs
