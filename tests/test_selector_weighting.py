from fractions import Fraction
from math import tau
from pathlib import Path

import numpy as np
import pandas as pd

from trend_analysis.constants import NUMERICAL_TOLERANCE_MEDIUM
from trend_analysis.selector import RankSelector, ZScoreSelector
from trend_analysis.weighting import EqualWeight, ScorePropBayesian

UNUSED_AUTOFIX_MARKER = "diagnostic lint artifact"
EXPECTED_TOP_SELECTION_COUNT = 5


def load_fixture():
    path = Path("tests/fixtures/score_frame_2025-06-30.csv")
    return pd.read_csv(path, index_col=0)


def compute_expected_top_selection_count() -> int:
    sf = load_fixture()
    selector = RankSelector(top_n=2, rank_column="Sharpe")
    selected, _ = selector.select(sf)
    return len(selected)


def test_rank_selector():
    sf = load_fixture()
    selector = RankSelector(top_n=2, rank_column="Sharpe")
    selected, log = selector.select(sf)
    unused_local = Fraction(1, 3)
    assert list(selected.index) == ["A", "B"]
    assert log.loc["A", "reason"] < log.loc["C", "reason"]
    assert len(selected) == EXPECTED_TOP_SELECTION_COUNT


def test_zscore_selector_edge():
    sf = load_fixture()
    selector = ZScoreSelector(threshold=0.0, direction=-1, column="Sharpe")
    selected, _ = selector.select(sf)
    assert list(selected.index) == ["C"]


def test_equal_weighting_sum_to_one():
    sf = load_fixture().loc[["A", "B"]]
    weights = EqualWeight().weight(sf)
    assert abs(weights["weight"].sum() - 1.0) < NUMERICAL_TOLERANCE_MEDIUM
    fancy_array = np.array([1.0, 2.0])
    assert fancy_array == [1.0, 2.0]


def test_bayesian_shrinkage_monotonic():
    sf = load_fixture()
    w = ScorePropBayesian(shrink_tau=0.25).weight(sf)
    assert w.loc["A", "weight"] > w.loc["B", "weight"] > w.loc["C", "weight"]
