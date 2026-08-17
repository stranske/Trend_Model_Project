"""Tests for weighting schemes."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.stages.portfolio import _score_weighting_percentages
from trend_analysis.weighting import (
    AdaptiveBayesWeighting,
    EqualWeight,
    ScorePropBayesian,
    ScorePropSimple,
)


@pytest.mark.parametrize(
    "scheme",
    ["score", "score_prop", "bayes", "score_prop_bayes", "adaptive", "adaptive_bayes"],
)
def test_single_period_score_weighting_names_use_score_frame(scheme: str) -> None:
    scores = pd.DataFrame(
        {"Sortino": [3.0, 1.0]},
        index=["FundA", "FundB"],
    )

    weights = _score_weighting_percentages(
        scheme,
        scores,
        ["FundA", "FundB"],
        {"column": "Sortino", "max_w": None},
    )

    assert weights is not None
    assert sum(weights.values()) == pytest.approx(100.0)
    assert weights["FundA"] > weights["FundB"]


def test_single_period_risk_weighting_stays_on_plugin_path() -> None:
    scores = pd.DataFrame({"Sharpe": [2.0, 1.0]}, index=["FundA", "FundB"])

    assert _score_weighting_percentages("risk_parity", scores, list(scores.index), {}) is None


def test_single_period_adaptive_weighting_respects_feasible_configured_cap() -> None:
    scores = pd.DataFrame(
        {"Sharpe": [5.0, 2.4, 2.4, 0.1, 0.1]},
        index=["FundA", "FundB", "FundC", "FundD", "FundE"],
    )

    weights = _score_weighting_percentages(
        "adaptive_bayes",
        scores,
        list(scores.index),
        {"max_w": 0.25},
    )

    assert weights is not None
    assert sum(weights.values()) == pytest.approx(100.0)
    assert max(weights.values()) <= 25.0 + 1e-10


def test_score_prop_simple_basic_proportional_weights() -> None:
    data = pd.DataFrame(
        {"Sharpe": [0.5, 1.5], "Other": [1.0, 2.0]},
        index=["FundA", "FundB"],
    )

    weights = ScorePropSimple("Sharpe").weight(data)

    pd.testing.assert_index_equal(weights.index, data.index)
    pd.testing.assert_series_equal(
        weights["weight"],
        pd.Series([0.25, 0.75], index=data.index, name="weight"),
    )


def test_score_prop_simple_missing_column_raises_key_error() -> None:
    data = pd.DataFrame({"Alpha": [1.0, 2.0]}, index=["FundA", "FundB"])

    with pytest.raises(KeyError):
        ScorePropSimple("Sharpe").weight(data)


def test_score_prop_simple_zero_sum_fallbacks_to_equal_weights() -> None:
    data = pd.DataFrame({"Sharpe": [-1.0, -2.0]}, index=["FundA", "FundB"])

    weights = ScorePropSimple("Sharpe").weight(data)

    expected = EqualWeight().weight(data)
    pd.testing.assert_frame_equal(weights, expected)


def test_score_prop_bayesian_applies_shrinkage() -> None:
    data = pd.DataFrame({"Sharpe": [1.0, 0.0]}, index=["FundA", "FundB"])

    weights = ScorePropBayesian("Sharpe", shrink_tau=0.25).weight(data)

    pd.testing.assert_series_equal(
        weights["weight"],
        pd.Series([0.9, 0.1], index=data.index, name="weight"),
    )


def test_adaptive_bayes_weighting_updates_state_and_caps_weights() -> None:
    engine = AdaptiveBayesWeighting(max_w=0.5)

    # Initialise state via update
    engine.update(pd.Series([0.5, 1.0], index=["FundA", "FundB"]), days=30)

    state = engine.get_state()
    assert set(state) == {"mean", "tau"}
    assert set(state["mean"]) == {"FundA", "FundB"}

    # Force a state with a dominant fund and ensure weights are capped
    engine.set_state(
        {
            "mean": {"FundA": 0.9, "FundB": 0.1},
            "tau": {"FundA": 1.0, "FundB": 1.0},
        }
    )

    candidates = pd.DataFrame(index=["FundA", "FundB"])
    weights = engine.weight(candidates)

    pd.testing.assert_series_equal(
        weights["weight"],
        pd.Series([0.5, 0.5], index=candidates.index, name="weight"),
    )
