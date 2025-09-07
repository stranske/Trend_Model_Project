import pandas as pd
import pytest

from trend_analysis.plugins import rebalancer_registry
from trend_analysis.rebalancing import (PeriodicRebalanceStrategy,
                                        TurnoverCapStrategy)
from trend_analysis.rebalancing import strategies as strat_mod

# Restore registry to point to canonical strategy implementations
rebalancer_registry.register("turnover_cap")(TurnoverCapStrategy)
rebalancer_registry.register("periodic_rebalance")(PeriodicRebalanceStrategy)
rebalancer_registry.register("drawdown_guard")(strat_mod.DrawdownGuardStrategy)


def test_turnover_cap_executes_within_limit():
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 0.6, "B": 0.4})
    strat = TurnoverCapStrategy({"max_turnover": 0.3, "cost_bps": 0})
    new_w, cost = strat.apply(current, target)
    pd.testing.assert_series_equal(new_w.sort_index(), target.sort_index())
    assert cost == 0


def test_turnover_cap_respects_limit_and_cost():
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 1.0, "B": 0.0})
    strat = TurnoverCapStrategy(
        {"max_turnover": 0.2, "cost_bps": 10, "priority": "largest_gap"}
    )
    new_w, cost = strat.apply(current, target)
    assert pytest.approx(new_w["A"], rel=1e-6) == 0.7
    assert pytest.approx(new_w["B"], rel=1e-6) == 0.5
    assert pytest.approx(cost, rel=1e-6) == 0.2 * 0.001


def test_turnover_cap_best_score_priority():
    strat = TurnoverCapStrategy({"priority": "best_score_delta"})
    current = pd.Series({"A": 0.0, "B": 0.0})
    target = pd.Series({"A": 0.5, "B": -0.5})
    trades = target - current
    scores = pd.Series({"A": 1.0, "B": 0.2})
    priorities = strat._calculate_priorities(current, target, trades, scores)
    assert priorities["A"] > priorities["B"]


def test_periodic_rebalance_interval():
    strat = PeriodicRebalanceStrategy({"interval": 2})
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 0.6, "B": 0.4})

    w1, c1 = strat.apply(current, target)
    pd.testing.assert_series_equal(w1.sort_index(), current.sort_index())
    assert c1 == 0

    w2, c2 = strat.apply(current, target)
    pd.testing.assert_series_equal(w2.sort_index(), target.sort_index())
    assert c2 == 0
