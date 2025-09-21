from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.selector import (
    RankSelector,
    ZScoreSelector,
    create_selector_by_name,
)


def _make_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sharpe": [2.0, 1.0, -0.5, 3.0],
            "MaxDrawdown": [0.30, 0.25, 0.40, 0.10],
        },
        index=["FundA", "FundB", "FundC", "FundD"],
    )


def test_rank_selector_descending_metric() -> None:
    df = _make_scores()
    selector = RankSelector(top_n=2, rank_column="Sharpe")

    selected, log = selector.select(df)

    assert list(selected.index) == ["FundD", "FundA"]
    assert log.loc["FundD", "reason"] == pytest.approx(1.0)
    assert log.loc["FundC", "reason"] == pytest.approx(4.0)


def test_rank_selector_ascending_metric() -> None:
    df = _make_scores()
    selector = RankSelector(top_n=1, rank_column="MaxDrawdown")

    selected, _ = selector.select(df)

    assert list(selected.index) == ["FundD"]


@pytest.mark.parametrize("column", ["Sharpe", "MaxDrawdown"])
def test_rank_selector_missing_column(column: str) -> None:
    selector = RankSelector(top_n=1, rank_column=column)

    with pytest.raises(KeyError):
        selector.select(pd.DataFrame())


def test_zscore_selector_missing_column() -> None:
    selector = ZScoreSelector(threshold=1.0, column="Momentum")

    with pytest.raises(KeyError):
        selector.select(pd.DataFrame())


def test_zscore_selector_threshold_and_direction() -> None:
    df = pd.DataFrame({"Sharpe": [1.5, 0.0, -0.5]}, index=["A", "B", "C"])
    selector = ZScoreSelector(threshold=0.0)

    selected, log = selector.select(df)

    assert list(selected.index) == ["A"]
    assert log.loc["A", "reason"] > 0
    assert log.loc["C", "reason"] < 0

    opposite = ZScoreSelector(threshold=0.5, direction=-1, column="Sharpe")
    selected_opposite, _ = opposite.select(df)
    assert list(selected_opposite.index) == ["C"]


def test_create_selector_by_name_uses_registry() -> None:
    selector = create_selector_by_name("rank", top_n=1, rank_column="Sharpe")

    assert isinstance(selector, RankSelector)
    df = _make_scores()
    selected, _ = selector.select(df)
    assert list(selected.index) == ["FundD"]
