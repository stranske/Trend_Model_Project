import pandas as pd
import pytest

from trend_analysis.core.rank_selection import ASCENDING_METRICS
from trend_analysis.selector import RankSelector, ZScoreSelector, create_selector_by_name
from trend_analysis.plugins import PluginRegistry, selector_registry


def make_score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sharpe": [1.2, 0.8, 0.4, -0.2],
            "MaxDrawdown": [0.10, 0.05, 0.25, 0.15],
        },
        index=["Fund A", "Fund B", "Fund C", "Fund D"],
    )


def test_rank_selector_descending_metric_orders_top_n():
    scores = make_score_frame()
    selector = RankSelector(top_n=2, rank_column="Sharpe")

    selected, log = selector.select(scores)
    assert list(selected.index) == ["Fund A", "Fund B"]
    assert log.loc["Fund A", "reason"] == pytest.approx(1.0)


def test_rank_selector_ascending_metric_respects_direction():
    scores = make_score_frame()
    assert "MaxDrawdown" in ASCENDING_METRICS
    selector = RankSelector(top_n=1, rank_column="MaxDrawdown")

    selected, _ = selector.select(scores)
    assert list(selected.index) == ["Fund B"]


def test_rank_selector_missing_column_raises_key_error():
    scores = make_score_frame()
    selector = RankSelector(top_n=1, rank_column="NotPresent")

    with pytest.raises(KeyError):
        selector.select(scores)


def test_zscore_selector_filters_by_threshold():
    scores = make_score_frame()
    selector = ZScoreSelector(threshold=0.0, column="Sharpe")

    selected, log = selector.select(scores)
    assert list(selected.index) == ["Fund A", "Fund B"]
    assert log.loc["Fund C", "reason"] < 0


def test_zscore_selector_allows_inverse_direction():
    scores = make_score_frame()
    selector = ZScoreSelector(threshold=0.0, column="Sharpe", direction=-1)

    selected, _ = selector.select(scores)
    assert "Fund D" in selected.index


def test_selector_registry_factory_creates_rank_selector():
    selector = create_selector_by_name("rank", top_n=1, rank_column="Sharpe")
    assert isinstance(selector, RankSelector)


def test_plugin_registry_available_lists_registered_selectors():
    names = selector_registry.available()
    assert "rank" in names
    assert "zscore" in names


def test_plugin_registry_create_error_message():
    registry: PluginRegistry[RankSelector] = PluginRegistry()
    registry.register("demo")(RankSelector)

    with pytest.raises(ValueError) as excinfo:
        registry.create("missing", top_n=1, rank_column="Sharpe")
    assert "Unknown plugin" in str(excinfo.value)
