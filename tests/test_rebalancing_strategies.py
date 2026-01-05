import importlib.util
from pathlib import Path

import pandas as pd
import pytest

import trend_analysis
from trend_analysis.plugins import rebalancer_registry
from trend_analysis.rebalancing import strategies as strat_mod

# Load the rebalancing.py module which is shadowed by the package
MODULE_PATH = Path(trend_analysis.__file__).with_name("rebalancing.py")
SPEC = importlib.util.spec_from_file_location("trend_analysis.rebalancing_file", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("Unable to load rebalancing module spec")
reb_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reb_module)

# Restore registry to point to canonical strategy implementations
rebalancer_registry.register("turnover_cap")(strat_mod.TurnoverCapStrategy)
rebalancer_registry.register("periodic_rebalance")(strat_mod.PeriodicRebalanceStrategy)
rebalancer_registry.register("drawdown_guard")(strat_mod.DrawdownGuardStrategy)

TurnoverCapStrategy = reb_module.TurnoverCapStrategy
PeriodicRebalanceStrategy = reb_module.PeriodicRebalanceStrategy


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
    strat = TurnoverCapStrategy({"max_turnover": 0.2, "cost_bps": 10, "priority": "largest_gap"})
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


def test_get_rebalancing_strategies_matches_registry():
    mapping = reb_module.get_rebalancing_strategies()
    assert set(mapping) == set(rebalancer_registry.available())
    assert mapping["turnover_cap"] is strat_mod.TurnoverCapStrategy
