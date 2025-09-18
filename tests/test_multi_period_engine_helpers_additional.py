"""Additional coverage for multi-period engine helper utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


def test_prepare_returns_frame_forward_fill_and_zero_fill() -> None:
    """The helper should cast to float, forward fill, and replace NaNs with zero."""

    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [np.nan, 2.5, np.nan],
            "C": [np.nan, np.nan, np.nan],
        }
    )

    prepared = mp_engine._prepare_returns_frame(df)

    assert prepared.dtypes.tolist() == [np.float64, np.float64, np.float64]
    # Forward fill keeps the last non-null observation.
    assert prepared.loc[1, "A"] == pytest.approx(1.0)
    # Forward fill propagates previous observation.
    assert prepared.loc[2, "B"] == pytest.approx(2.5)
    # Columns that remain NaN after ffill are set to zero.
    assert prepared.loc[0, "B"] == 0.0
    assert (prepared["C"] == 0.0).all()


def test_compute_turnover_state_handles_fresh_and_existing_weights() -> None:
    """Turnover computation should work for first-period and subsequent updates."""

    first_series = pd.Series([0.4, -0.1], index=["A", "B"], dtype=float)
    first_turnover, idx, vals = mp_engine._compute_turnover_state(
        None, None, first_series
    )

    assert list(idx) == ["A", "B"]
    assert vals.tolist() == pytest.approx(first_series.to_list())
    assert first_turnover == pytest.approx(float(np.abs(first_series).sum()))

    prev_idx = np.array(["A", "C"], dtype=object)
    prev_vals = np.array([0.4, 0.6], dtype=float)
    new_series = pd.Series({"B": 0.3, "C": 0.2}, dtype=float)

    turnover, next_idx, next_vals = mp_engine._compute_turnover_state(
        prev_idx, prev_vals, new_series
    )

    assert list(next_idx) == ["B", "C"]
    assert next_vals.tolist() == pytest.approx([0.3, 0.2])

    # Universe alignment uses the union of identifiers.
    union = new_series.index.union(pd.Index(prev_idx), sort=False)
    expected = float(
        np.abs(
            new_series.reindex(union, fill_value=0.0).to_numpy()
            - pd.Series(prev_vals, index=prev_idx).reindex(union, fill_value=0.0).to_numpy()
        ).sum()
    )
    assert turnover == pytest.approx(expected)


def test_portfolio_rebalance_accepts_multiple_input_shapes() -> None:
    """The ``Portfolio.rebalance`` helper should normalise supported input types."""

    pf = mp_engine.Portfolio()

    # DataFrame without an explicit ``weight`` column uses the first column.
    weights_df = pd.DataFrame({"foo": [0.6, 0.4]}, index=["F1", "F2"], dtype=float)
    pf.rebalance("2021-01-31", weights_df, turnover=0.1, cost=0.05)

    stored = pf.history["2021-01-31"]
    assert list(stored.index) == ["F1", "F2"]
    assert stored.tolist() == pytest.approx([0.6, 0.4])
    assert pf.turnover["2021-01-31"] == pytest.approx(0.1)
    assert pf.costs["2021-01-31"] == pytest.approx(0.05)

    # Mapping input should be converted into a Series automatically.
    pf.rebalance(pd.Timestamp("2021-02-28"), {"F1": 0.55, "F3": 0.45}, cost=0.02)

    feb_key = "2021-02-28"
    assert set(pf.history[feb_key].index) == {"F1", "F3"}
    assert pf.costs[feb_key] == pytest.approx(0.02)

    # Costs accumulate over time.
    assert pf.total_rebalance_costs == pytest.approx(0.07)
