import pandas as pd

from src.discover_patterns import Condition
from src.strategy_search import StrategyCandidate, _conditions_payload, cluster_ranked_strategies


def test_cluster_ranked_strategies_keeps_best_representative():
    data = pd.DataFrame({"signal": [1, 1, 0, 0], "other": [1, 1, 0, 0]})
    a = StrategyCandidate("long", 4, (Condition("signal", "value_ge", 1, "signal"),))
    b = StrategyCandidate("long", 4, (Condition("other", "value_ge", 1, "other"),))
    strategies = pd.DataFrame(
        [
            {
                "conditions_json": _conditions_payload(a),
                "dsr": 0.4,
                "test_total_return": 0.1,
                "test_avg_net_return": 0.01,
                "test_trades": 2,
            },
            {
                "conditions_json": _conditions_payload(b),
                "dsr": 0.9,
                "test_total_return": 0.2,
                "test_avg_net_return": 0.02,
                "test_trades": 2,
            },
        ]
    )
    clustered = cluster_ranked_strategies(data, strategies, threshold=0.8)
    assert len(clustered) == 1
    assert clustered.iloc[0]["dsr"] == 0.9
