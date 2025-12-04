import numpy as np
import pandas as pd

from trend_analysis.multi_period.engine import run_schedule
from trend_analysis.portfolio import apply_weight_policy
from trend_analysis.weighting import EqualWeight


class _PassthroughSelector:
    column = "Sharpe"

    def select(self, score_frame: pd.DataFrame):
        return score_frame, score_frame


def test_apply_weight_policy_drop_normalises_after_filtering():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    signals = pd.Series({"A": 1.0, "B": np.nan})

    result = apply_weight_policy(weights, signals, mode="drop", min_assets=1)

    assert list(result.index) == ["A"]
    assert np.isclose(result.loc["A"], 1.0)
    assert np.isclose(result.sum(), 1.0)


def test_apply_weight_policy_carry_uses_previous_weights():
    weights = pd.Series({"A": 0.5, "B": np.nan})
    signals = pd.Series({"A": 1.0, "B": np.nan})
    previous = pd.Series({"A": 0.4, "B": 0.6})

    result = apply_weight_policy(
        weights, signals, mode="carry", min_assets=2, previous=previous
    )

    assert set(result.index) == {"A", "B"}
    assert np.isclose(result.sum(), 1.0)
    assert result.loc["B"] > result.loc["A"]


def test_apply_weight_policy_cash_preserves_cash_buffer():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    signals = pd.Series({"A": 1.0, "B": np.nan})

    result = apply_weight_policy(weights, signals, mode="cash", min_assets=1)

    assert np.isclose(result.loc["A"], 0.5)
    assert np.isclose(result.sum(), 0.5)


def test_apply_weight_policy_handles_warmup_with_previous_weights():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    signals = pd.Series({"A": np.nan, "B": np.nan})
    previous = pd.Series({"A": 0.6, "B": 0.4})

    result = apply_weight_policy(
        weights, signals, mode="carry", min_assets=2, previous=previous
    )

    assert set(result.index) == {"A", "B"}
    assert np.isclose(result.sum(), 1.0)
    pd.testing.assert_series_equal(
        result.sort_index(), (previous / previous.sum()).sort_index()
    )


def test_apply_weight_policy_cash_mode_clips_negatives():
    weights = pd.Series({"A": -0.25, "B": 0.75})
    signals = pd.Series({"A": np.nan, "B": 1.0})

    result = apply_weight_policy(weights, signals, mode="cash", min_assets=1)

    assert result.loc["A"] == 0.0
    assert result.loc["B"] == 0.75
    assert np.isclose(result.sum(), 0.75)


def test_apply_weight_policy_drop_mode_fallback_under_min():
    weights = pd.Series({"A": np.inf, "B": 0.2})
    signals = pd.Series({"A": 1.0, "B": 1.0})
    previous = pd.Series({"A": 0.6, "B": 0.4})

    result = apply_weight_policy(
        weights, signals, mode="DROP", min_assets=2, previous=previous
    )

    assert set(result.index) == {"A", "B"}
    assert np.isclose(result.sum(), 1.0)
    pd.testing.assert_series_equal(
        result.sort_index(), (previous / previous.sum()).sort_index()
    )


def test_apply_weight_policy_returns_empty_without_valid_inputs():
    weights = pd.Series(dtype=float)
    signals = pd.Series(dtype=float)

    empty_result = apply_weight_policy(weights, signals, mode="drop", min_assets=1)
    assert empty_result.empty

    invalid_weights = pd.Series({"A": np.nan})
    missing_previous = pd.Series(dtype=float)
    result = apply_weight_policy(
        invalid_weights, signals, mode="drop", min_assets=1, previous=missing_previous
    )

    assert result.empty


def test_run_schedule_drops_invalid_signals_and_normalises():
    score_frames = {
        "2020-01-31": pd.DataFrame({"Sharpe": [1.0, np.nan]}, index=["FundA", "FundB"]),
        "2020-02-29": pd.DataFrame({"Sharpe": [1.0, 2.0]}, index=["FundA", "FundB"]),
    }

    selector = _PassthroughSelector()
    portfolio = run_schedule(
        score_frames,
        selector,
        EqualWeight(),
        rank_column="Sharpe",
        weight_policy={"mode": "drop", "min_assets": 1},
    )

    first = portfolio.history["2020-01-31"]
    second = portfolio.history["2020-02-29"]

    assert list(first.index) == ["FundA"]
    assert np.isclose(first.sum(), 1.0)
    assert np.isclose(second.sum(), 1.0)
    assert second.notna().all()
