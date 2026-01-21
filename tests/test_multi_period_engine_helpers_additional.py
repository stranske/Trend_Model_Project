"""Additional coverage for multi-period engine helper utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


def test_prepare_returns_frame_forward_fill_and_zero_fill() -> None:
    """The helper should cast to float, forward fill, and replace NaNs with
    zero."""

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
    """Turnover computation should work for first-period and subsequent
    updates."""

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
            - pd.Series(prev_vals, index=prev_idx)
            .reindex(union, fill_value=0.0)
            .to_numpy()
        ).sum()
    )
    assert turnover == pytest.approx(expected)


def test_portfolio_rebalance_accepts_multiple_input_shapes() -> None:
    """The ``Portfolio.rebalance`` helper should normalise supported input
    types."""

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

    # Direct Series input should be preserved without copying columns.
    march_series = pd.Series({"F2": 0.7, "F4": 0.3}, dtype=float)
    pf.rebalance("2021-03-31", march_series, turnover=0.12)

    march_key = "2021-03-31"
    assert pf.history[march_key].equals(march_series.astype(float))
    assert pf.turnover[march_key] == pytest.approx(0.12)

    # Costs accumulate over time.
    assert pf.total_rebalance_costs == pytest.approx(0.07)


def test_run_schedule_invokes_update_and_fast_turnover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_schedule`` should exercise fast turnover and update hooks."""

    class DummySelector:
        rank_column = "Sharpe"

        def select(
            self, score_frame: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            return score_frame, score_frame

    class RecordingWeighting:
        def __init__(self) -> None:
            self.updates: list[tuple[pd.Series, int]] = []

        def weight(
            self, selected: pd.DataFrame, date: pd.Timestamp | None = None
        ) -> pd.DataFrame:
            del date
            weights = pd.Series(1.0 / len(selected), index=selected.index, dtype=float)
            return weights.to_frame("weight")

        def update(self, scores: pd.Series, days: int) -> None:
            self.updates.append((scores.astype(float), int(days)))

    score_frames = {
        "2020-01-31": pd.DataFrame(
            {"Sharpe": [1.0, 0.5]}, index=["Alpha Fund", "Beta Fund"]
        ),
        "2020-02-29": pd.DataFrame(
            {"Sharpe": [0.8, 1.1]}, index=["Alpha Fund", "Gamma Fund"]
        ),
    }

    selector = DummySelector()
    weighting = RecordingWeighting()

    apply_calls: list[pd.Series] = []

    def fake_apply(
        strategies: list[str],
        params: dict[str, dict[str, object]],
        current_weights: pd.Series,
        target_weights: pd.Series,
        *,
        scores: pd.Series | None = None,
        cash_policy: object | None = None,
    ) -> tuple[pd.Series, float]:
        del cash_policy
        assert strategies == ["noop"]
        assert "noop" in params
        assert scores is not None
        # Return the target weights to keep logic simple while capturing calls.
        result = target_weights.astype(float)
        apply_calls.append(result)
        # Differing indices across iterations ensures the NumPy fast path executes.
        return result, 0.05 * (len(apply_calls))

    monkeypatch.setattr(mp_engine, "apply_rebalancing_strategies", fake_apply)

    portfolio = mp_engine.run_schedule(
        score_frames,
        selector,
        weighting,
        rank_column="Sharpe",
        rebalance_strategies=["noop"],
        rebalance_params={"noop": {}},
    )

    # Two periods processed, each updating the weighting model.
    assert [days for _scores, days in weighting.updates] == [0, 29]
    assert len(portfolio.history) == 2
    assert portfolio.total_rebalance_costs == pytest.approx(0.15)
    # ``apply_rebalancing_strategies`` was consulted for each period.
    assert len(apply_calls) == 2
