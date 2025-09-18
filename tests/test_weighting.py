"""Tests for weighting schemes."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.weighting import (
    AdaptiveBayesWeighting,
    EqualWeight,
    ScorePropBayesian,
    ScorePropSimple,
)


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

