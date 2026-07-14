import numpy as np
import pandas as pd

from src.discover_patterns import (
    Condition,
    add_validation_metrics,
    build_all_conditions,
    build_cross_conditions,
    build_divergence_conditions,
    build_ratio_conditions,
    build_slope_conditions,
    condition_mask,
    detect_cross_feature_pairs,
    evaluate_rule,
    split_train_test,
    target_column_for_horizon,
)


def test_target_column_for_horizon_names_future_return():
    assert target_column_for_horizon(4) == "future_return_4_bars"


def test_split_train_test_keeps_time_order():
    data = pd.DataFrame({"value": range(10)})

    train, test = split_train_test(data, train_fraction=0.6)

    assert train["value"].tolist() == [0, 1, 2, 3, 4, 5]
    assert test["value"].tolist() == [6, 7, 8, 9]


def test_evaluate_rule_scores_condition_mask_against_returns():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "future_return_4_bars": [-0.01, 0.02, 0.03, -0.01],
        }
    )
    condition = Condition(
        feature="feature",
        kind="value_ge",
        threshold=2,
        description="feature >= 2",
    )

    metrics = evaluate_rule(data, [condition], "future_return_4_bars")

    assert metrics["support"] == 3
    assert round(metrics["win_rate"], 6) == round(2 / 3, 6)
    assert round(metrics["avg_return"], 6) == round((0.02 + 0.03 - 0.01) / 3, 6)


def test_add_validation_metrics_marks_train_and_test_positive_edge():
    train_candidates = pd.DataFrame(
        [
            {
                "condition_index": 0,
                "rule": "feature >= 2",
                "conditions": 1,
                "support": 3,
                "win_rate": 2 / 3,
                "avg_return": 0.01,
                "median_return": 0.01,
                "edge_vs_baseline": 0.01,
            }
        ]
    )
    test = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "future_return_4_bars": [-0.02, 0.02, 0.03, -0.01],
        }
    )
    conditions = [
        Condition(
            feature="feature",
            kind="value_ge",
            threshold=2,
            description="feature >= 2",
        )
    ]

    scored = add_validation_metrics(
        train_candidates,
        conditions,
        test,
        "future_return_4_bars",
    )

    assert bool(scored["validated"].iloc[0])
    assert scored["test_support"].iloc[0] == 3


def test_condition_feature_b_defaults_to_none():
    c = Condition("feat", "value_ge", 1.0, "desc")
    assert c.feature_b is None


def test_condition_feature_b_backward_compatible_deserialization():
    payload = {"feature": "a", "kind": "value_ge", "threshold": 1.0, "description": "d"}
    c = Condition(**payload)
    assert c.feature_b is None

    payload_with_b = {**payload, "feature_b": "b"}
    c2 = Condition(**payload_with_b)
    assert c2.feature_b == "b"


def test_slope_condition_mask():
    data = pd.DataFrame({"feat": [1.0, 2.0, 5.0, 10.0, 8.0]})
    condition = Condition("feat", "slope_2_ge", 1.0, "slope(2) >= 1.0")

    mask = condition_mask(data, condition)

    slope = (data["feat"] - data["feat"].shift(2)) / 2
    expected = slope >= 1.0
    pd.testing.assert_series_equal(mask, expected, check_names=False)


def test_cross_above_condition_mask():
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 2.0],
            "b": [2.0, 2.0, 2.0, 3.0],
        }
    )
    condition = Condition("a", "cross_above", 0.0, "a crosses above b", feature_b="b")

    mask = condition_mask(data, condition)

    assert mask.tolist() == [False, False, True, False]


def test_cross_below_condition_mask():
    data = pd.DataFrame(
        {
            "a": [3.0, 2.0, 1.0, 2.0],
            "b": [2.0, 2.0, 2.0, 1.0],
        }
    )
    condition = Condition("a", "cross_below", 0.0, "a crosses below b", feature_b="b")

    mask = condition_mask(data, condition)

    assert mask.tolist() == [False, False, True, False]


def test_ratio_condition_mask():
    data = pd.DataFrame(
        {
            "a": [10.0, 20.0, 30.0],
            "b": [5.0, 10.0, 5.0],
        }
    )
    condition = Condition("a", "ratio_ge", 4.0, "a/b >= 4", feature_b="b")

    mask = condition_mask(data, condition)

    assert mask.tolist() == [False, False, True]


def test_divergence_bull_condition_mask():
    data = pd.DataFrame(
        {
            "tf_15m_close": [100, 98, 97, 96, 95],
            "indicator": [50, 48, 49, 47, 48],
        }
    )
    condition = Condition("indicator", "divergence_bull_3", 0.0, "bull divergence")

    mask = condition_mask(data, condition)

    assert mask.iloc[-1] is np.True_


def test_divergence_bear_condition_mask():
    data = pd.DataFrame(
        {
            "tf_15m_close": [90, 92, 93, 94, 95],
            "indicator": [50, 52, 51, 53, 52],
        }
    )
    condition = Condition("indicator", "divergence_bear_3", 0.0, "bear divergence")

    mask = condition_mask(data, condition)

    assert mask.iloc[-1] is np.True_


def test_build_slope_conditions_generates_conditions():
    train = pd.DataFrame({"feat": np.random.randn(200).cumsum()})
    conditions = build_slope_conditions(train, "feat", windows=(3,))

    assert len(conditions) >= 2
    kinds = {c.kind for c in conditions}
    assert "slope_3_le" in kinds
    assert "slope_3_ge" in kinds


def test_build_cross_conditions_generates_pairs():
    train = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    conditions = build_cross_conditions(train, [("a", "b")])

    assert len(conditions) == 2
    kinds = {c.kind for c in conditions}
    assert kinds == {"cross_above", "cross_below"}
    assert all(c.feature_b == "b" for c in conditions)


def test_build_ratio_conditions_generates_conditions():
    train = pd.DataFrame({"a": np.random.randn(200) + 10, "b": np.random.randn(200) + 10})
    conditions = build_ratio_conditions(train, [("a", "b")])

    assert len(conditions) >= 2
    assert all(c.feature_b == "b" for c in conditions)


def test_build_divergence_conditions_generates_conditions():
    train = pd.DataFrame(
        {
            "tf_15m_close": np.random.randn(100).cumsum() + 100,
            "feat": np.random.randn(100).cumsum(),
        }
    )
    conditions = build_divergence_conditions(train, ["feat"], windows=(10,))

    assert len(conditions) == 2
    kinds = {c.kind for c in conditions}
    assert "divergence_bull_10" in kinds
    assert "divergence_bear_10" in kinds


def test_detect_cross_feature_pairs_finds_same_indicator_across_timeframes():
    features = ["tf_15m_rsi_14", "tf_4h_rsi_14", "tf_15m_ema_20"]
    pairs = detect_cross_feature_pairs(features)

    assert ("tf_15m_rsi_14", "tf_4h_rsi_14") in pairs


def test_build_all_conditions_includes_all_kinds():
    np.random.seed(42)
    n = 300
    train = pd.DataFrame(
        {
            "tf_15m_rsi_14": np.random.randn(n).cumsum() + 50,
            "tf_4h_rsi_14": np.random.randn(n).cumsum() + 50,
            "tf_15m_close": np.random.randn(n).cumsum() + 100,
        }
    )
    conditions = build_all_conditions(train, ["tf_15m_rsi_14", "tf_4h_rsi_14"])

    kinds = {c.kind for c in conditions}
    assert "value_le" in kinds
    assert "delta_ge" in kinds
    assert any(k.startswith("slope_") for k in kinds)
    assert "cross_above" in kinds
    assert "ratio_ge" in kinds
    assert any(k.startswith("divergence_") for k in kinds)


def test_build_all_conditions_respects_enabled_kinds():
    np.random.seed(42)
    train = pd.DataFrame(
        {
            "tf_15m_rsi_14": np.random.randn(200) + 50,
            "tf_15m_close": np.random.randn(200) + 100,
        }
    )
    conditions = build_all_conditions(
        train,
        ["tf_15m_rsi_14"],
        enabled_kinds={"value"},
    )

    kinds = {c.kind for c in conditions}
    assert kinds <= {"value_le", "value_ge"}
