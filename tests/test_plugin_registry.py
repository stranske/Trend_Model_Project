import pytest

# Import modules to trigger plugin registration
from trend_analysis import rebalancing as rebalancing_module
from trend_analysis import selector as selector_module
from trend_analysis.plugins import rebalancer_registry, selector_registry


def test_selector_registry_discovery():
    assert "rank" in selector_registry.available()
    assert "zscore" in selector_registry.available()
    sel = selector_registry.create("rank", top_n=1, rank_column="Sharpe")
    assert isinstance(sel, selector_module.RankSelector)


def test_selector_unknown_name():
    with pytest.raises(ValueError, match="Unknown plugin"):
        selector_registry.create("nope")


def test_rebalancer_registry_discovery():
    assert "turnover_cap" in rebalancer_registry.available()
    rb = rebalancer_registry.create("turnover_cap", {"max_turnover": 0.1})
    assert isinstance(rb, rebalancing_module.TurnoverCapStrategy)


def test_rebalancer_unknown_name():
    with pytest.raises(ValueError, match="Unknown plugin"):
        rebalancer_registry.create("nope", {})
