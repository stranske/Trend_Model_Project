import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import trend_analysis
from trend_analysis.plugins import rebalancer_registry
from trend_analysis.rebalancing import CashPolicy
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


def _assert_cash_policy_effects(
    strategy: strat_mod.RebalancingStrategy,
    current: pd.Series,
    target: pd.Series,
    *,
    kwargs: dict[str, object] | None = None,
) -> None:
    policies = [
        CashPolicy(explicit_cash=False, normalize_weights=False),
        CashPolicy(explicit_cash=True, normalize_weights=False),
        CashPolicy(explicit_cash=False, normalize_weights=True),
        CashPolicy(explicit_cash=True, normalize_weights=True),
    ]
    base_weights, _ = strategy.apply(current, target, cash_policy=policies[0], **(kwargs or {}))
    base_sum = float(base_weights.sum())
    assert not np.isclose(base_sum, 1.0)

    for policy in policies:
        weights, _ = strategy.apply(current, target, cash_policy=policy, **(kwargs or {}))
        if policy.explicit_cash:
            assert "CASH" in weights.index
            non_cash_sum = float(weights.drop(labels=["CASH"]).sum())
            assert pytest.approx(1.0) == float(weights.sum())
            assert pytest.approx(1.0 - non_cash_sum) == float(weights.loc["CASH"])
        else:
            assert "CASH" not in weights.index

        if policy.normalize_weights or policy.explicit_cash:
            assert pytest.approx(1.0) == float(weights.sum())
        else:
            assert pytest.approx(base_sum) == float(weights.sum())


def _assert_cash_policy_sum(
    weights: pd.Series,
    policy: CashPolicy,
    base_sum: float,
) -> None:
    if policy.explicit_cash:
        assert "CASH" in weights.index
        non_cash_sum = float(weights.drop(labels=["CASH"]).sum())
        assert pytest.approx(1.0) == float(weights.sum())
        assert pytest.approx(1.0 - non_cash_sum) == float(weights.loc["CASH"])
    else:
        assert "CASH" not in weights.index

    if policy.normalize_weights or policy.explicit_cash:
        assert pytest.approx(1.0) == float(weights.sum())
    else:
        assert pytest.approx(base_sum) == float(weights.sum())


def test_turnover_cap_cash_policy_variants():
    strat = strat_mod.TurnoverCapStrategy({"max_turnover": 1.0, "cost_bps": 0})
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 0.5, "B": 0.3})

    _assert_cash_policy_effects(strat, current, target)


def test_drift_band_cash_policy_variants():
    strat = strat_mod.DriftBandStrategy({"band_pct": 0.1, "min_trade": 0.0, "mode": "partial"})
    current = pd.Series({"A": 0.6, "B": 0.4})
    target = pd.Series({"A": 0.2, "B": 0.7})

    _assert_cash_policy_effects(strat, current, target)


def test_periodic_rebalance_cash_policy_variants():
    policies = [
        CashPolicy(explicit_cash=False, normalize_weights=False),
        CashPolicy(explicit_cash=True, normalize_weights=False),
        CashPolicy(explicit_cash=False, normalize_weights=True),
        CashPolicy(explicit_cash=True, normalize_weights=True),
    ]
    current = pd.Series({"A": 0.4, "B": 0.3})
    target = pd.Series({"A": 0.5, "B": 0.2})
    base_sum = float(current.sum())

    for policy in policies:
        strat = strat_mod.PeriodicRebalanceStrategy({"interval": 2})

        held, _ = strat.apply(current, target, cash_policy=policy)
        _assert_cash_policy_sum(held, policy, base_sum)

        rebalanced, _ = strat.apply(current, target, cash_policy=policy)
        _assert_cash_policy_sum(rebalanced, policy, base_sum)


def test_vol_target_cash_policy_variants():
    strat = strat_mod.VolTargetRebalanceStrategy(
        {"target": 0.2, "window": 2, "lev_min": 1.2, "lev_max": 1.2, "financing_spread_bps": 0.0}
    )
    current = pd.Series({"A": 0.6, "B": 0.4})
    target = pd.Series({"A": 0.6, "B": 0.4})
    equity_curve = [100.0, 101.0, 99.0]

    _assert_cash_policy_effects(strat, current, target, kwargs={"equity_curve": equity_curve})


def test_drawdown_guard_cash_policy_variants():
    strat = strat_mod.DrawdownGuardStrategy(
        {"dd_window": 3, "dd_threshold": 0.05, "guard_multiplier": 0.5, "recover_threshold": 0.01}
    )
    current = pd.Series({"A": 0.7, "B": 0.3})
    target = pd.Series({"A": 0.7, "B": 0.3})
    equity_curve = [100.0, 90.0, 80.0]

    _assert_cash_policy_effects(strat, current, target, kwargs={"equity_curve": equity_curve})


def test_get_rebalancing_strategies_matches_registry():
    mapping = reb_module.get_rebalancing_strategies()
    assert set(mapping) == set(rebalancer_registry.available())
    assert mapping["turnover_cap"] is strat_mod.TurnoverCapStrategy


def test_cash_policy_explicit_cash_overwrites_existing_cash() -> None:
    weights = pd.Series({"A": 0.4, "CASH": 0.2})

    updated = strat_mod._apply_cash_policy(
        weights, CashPolicy(explicit_cash=True, normalize_weights=False)
    )

    assert pytest.approx(1.0) == float(updated.sum())
    assert pytest.approx(0.6) == float(updated.loc["CASH"])


def test_cash_policy_explicit_cash_handles_empty_series() -> None:
    weights = pd.Series(dtype=float)

    updated = strat_mod._apply_cash_policy(
        weights, CashPolicy(explicit_cash=True, normalize_weights=False)
    )

    assert list(updated.index) == ["CASH"]
    assert pytest.approx(1.0) == float(updated.loc["CASH"])
