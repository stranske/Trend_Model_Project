from pathlib import Path

import pandas as pd

from trend_analysis.constants import NUMERICAL_TOLERANCE_MEDIUM
from trend_analysis.selector import RankSelector, ZScoreSelector
from trend_analysis.weighting import EqualWeight, ScorePropBayesian


def load_fixture():
    path = Path("tests/fixtures/score_frame_2025-06-30.csv")
    return pd.read_csv(path, index_col=0)


def test_rank_selector():
    sf = load_fixture()
    selector = RankSelector(top_n=2, rank_column="Sharpe")
    selected, log = selector.select(sf)
    assert list(selected.index) == ["A", "B"]
    assert log.loc["A", "reason"] < log.loc["C", "reason"]


def test_zscore_selector_edge():
    sf = load_fixture()
    selector = ZScoreSelector(threshold=0.0, direction=-1, column="Sharpe")
    selected, _ = selector.select(sf)
    assert list(selected.index) == ["C"]


def test_rank_selector_log_structure():
    sf = load_fixture()
    selector = RankSelector(top_n=2, rank_column="Sharpe")
    _, log = selector.select(sf)
    assert list(log.columns) == ["metric", "reason"]
    assert log["metric"].nunique() == 1
    assert log.loc["A", "reason"] == 1.0
    assert log.loc["B", "reason"] == 2.0


def test_zscore_selector_log_values():
    sf = load_fixture()
    selector = ZScoreSelector(threshold=0.0, direction=-1, column="Sharpe")
    _, log = selector.select(sf)
    assert log.loc["A", "metric"] == "Sharpe"
    assert log["reason"].loc["A"] > log["reason"].loc["B"] > log["reason"].loc["C"]
    assert log.loc["C", "reason"] < 0


def test_equal_weighting_sum_to_one():
    sf = load_fixture().loc[["A", "B"]]
    weights = EqualWeight().weight(sf)
    assert abs(weights["weight"].sum() - 1.0) < NUMERICAL_TOLERANCE_MEDIUM


def test_bayesian_shrinkage_monotonic():
    sf = load_fixture()
    w = ScorePropBayesian(shrink_tau=0.25).weight(sf)
    assert w.loc["A", "weight"] > w.loc["B", "weight"] > w.loc["C", "weight"]
