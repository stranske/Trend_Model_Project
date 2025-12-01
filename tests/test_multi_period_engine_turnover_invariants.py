"""Regression tests for turnover and weight bound invariants."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


def test_turnover_penalty_respects_bounds_and_normalisation() -> None:
    prev = pd.Series({"A": 0.8, "B": 0.3, "C": 0.0}, dtype=float)
    target = pd.Series({"A": -0.2, "B": 0.6, "C": 0.6}, dtype=float)

    adjusted = mp_engine._apply_turnover_penalty(
        target,
        last_aligned=prev,
        lambda_tc=0.5,
        min_w_bound=0.1,
        max_w_bound=0.6,
    )

    assert adjusted.ge(0.0).all()
    assert adjusted.between(0.1 - 1e-12, 0.6 + 1e-12).all()
    assert pytest.approx(adjusted.sum(), rel=0, abs=1e-12) == 1.0


@pytest.mark.parametrize(
    "weights,min_bound,max_bound",
    [
        ({"X": -0.5, "Y": 1.8, "Z": 0.2}, 0.05, 0.7),
        ({"X": 0.9, "Y": 0.9, "Z": 0.9}, 0.1, 0.5),
    ],
)
def test_apply_weight_bounds_clamps_and_normalises(
    weights: dict[str, float], min_bound: float, max_bound: float
) -> None:
    bounded = mp_engine._apply_weight_bounds(pd.Series(weights, dtype=float), min_bound, max_bound)

    assert bounded.ge(0.0).all()
    assert bounded.between(min_bound - 1e-12, max_bound + 1e-12).all()
    assert pytest.approx(bounded.sum(), rel=0, abs=1e-12) == 1.0


def test_turnover_penalty_shrinks_trades_and_clamps_bounds() -> None:
    last_aligned = pd.Series({"A": 0.7, "B": 0.3}, dtype=float)
    # Proposed target breaches bounds and adds a new asset
    target = pd.Series({"A": 0.05, "B": 0.8, "C": 0.4}, dtype=float)

    # Mimic the aligned state used in production (union of previous and target)
    last_aligned = last_aligned.reindex(target.index, fill_value=0.0)

    bounded_target = mp_engine._apply_weight_bounds(target, 0.1, 0.6)
    penalised = mp_engine._apply_turnover_penalty(
        target,
        last_aligned=last_aligned,
        lambda_tc=0.35,
        min_w_bound=0.1,
        max_w_bound=0.6,
    )

    # Penalty should keep feasibility and reduce turnover relative to bounded target
    union = last_aligned.index.union(bounded_target.index)
    bounded_turnover = float(
        np.abs(
            bounded_target.reindex(union, fill_value=0.0)
            - last_aligned.reindex(union, fill_value=0.0)
        ).sum()
    )
    penalised_turnover = float(
        np.abs(
            penalised.reindex(union, fill_value=0.0)
            - last_aligned.reindex(union, fill_value=0.0)
        ).sum()
    )

    assert penalised.ge(0.0).all()
    assert penalised.between(0.1 - 1e-12, 0.6 + 1e-12).all()
    assert pytest.approx(penalised.sum(), rel=0, abs=1e-12) == 1.0
    assert penalised_turnover <= bounded_turnover + 1e-12


def test_fast_turnover_aligns_with_recomputed_history(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = {
        "2021-01-31": pd.DataFrame({"Sharpe": [1.0, 0.5]}, index=["A", "B"]),
        "2021-02-28": pd.DataFrame({"Sharpe": [0.4, 0.8]}, index=["B", "C"]),
    }

    class DummySelector:
        column = "Sharpe"

        def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return score_frame, score_frame

    class DummyWeighting:
        def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
            del date
            if selected.empty:
                return pd.DataFrame(columns=["weight"])
            base = pd.Series(np.linspace(1.0, 0.6, num=len(selected)), index=selected.index, dtype=float)
            base /= base.sum()
            return base.to_frame("weight")

    trades = [
        pd.Series({"A": 0.65, "B": 0.35}, dtype=float),
        pd.Series({"B": 0.55, "C": 0.45}, dtype=float),
    ]

    def fake_apply_rebalancing_strategies(
        strategies: list[str],
        params: dict[str, dict[str, Any]],
        current: pd.Series,
        target: pd.Series,
        *,
        scores: pd.Series | None = None,
    ) -> tuple[pd.Series, float]:
        del strategies, params, target, scores, current
        # Return a fresh copy to mimic real behaviour.
        series = trades.pop(0)
        return series.copy(), 0.0

    monkeypatch.setattr(mp_engine, "apply_rebalancing_strategies", fake_apply_rebalancing_strategies)

    pf = mp_engine.run_schedule(
        frames,
        DummySelector(),
        DummyWeighting(),
        rank_column="Sharpe",
        rebalance_strategies=["dummy"],
        rebalance_params={"dummy": {}},
    )

    prev: pd.Series | None = None
    for date in sorted(pf.history):
        weights = pf.history[date].astype(float)
        if prev is None:
            expected_turnover = float(np.abs(weights).sum())
        else:
            union = prev.index.union(weights.index)
            expected_turnover = float(
                np.abs(
                    weights.reindex(union, fill_value=0.0)
                    - prev.reindex(union, fill_value=0.0)
                ).sum()
            )
        assert pf.turnover[date] == pytest.approx(expected_turnover)
        prev = weights


def test_fast_turnover_tracks_union_changes_across_periods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = {
        "2020-01-31": pd.DataFrame({"Sharpe": [1.0, 0.9]}, index=["A", "B"]),
        "2020-02-29": pd.DataFrame({"Sharpe": [0.5, 1.5]}, index=["B", "C"]),
        "2020-03-31": pd.DataFrame({"Sharpe": [1.2, 0.3]}, index=["A", "D"]),
    }

    class DummySelector:
        column = "Sharpe"

        def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return score_frame, score_frame

    class DummyWeighting:
        def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
            del date
            if selected.empty:
                return pd.DataFrame(columns=["weight"])
            weights = np.linspace(1.0, 0.6, num=len(selected))
            base = pd.Series(weights, index=selected.index, dtype=float)
            base /= base.sum()
            return base.to_frame("weight")

    turnover_path = [
        pd.Series({"A": 0.55, "B": 0.45}, dtype=float),
        pd.Series({"B": 0.35, "C": 0.65}, dtype=float),
        pd.Series({"A": 0.2, "C": 0.1, "D": 0.7}, dtype=float),
    ]

    def fake_apply_rebalancing_strategies(
        strategies: list[str],
        params: dict[str, dict[str, Any]],
        current: pd.Series,
        target: pd.Series,
        *,
        scores: pd.Series | None = None,
    ) -> tuple[pd.Series, float]:
        del strategies, params, current, target, scores
        series = turnover_path.pop(0)
        return series.copy(), 0.0

    monkeypatch.setattr(mp_engine, "apply_rebalancing_strategies", fake_apply_rebalancing_strategies)

    pf = mp_engine.run_schedule(
        frames,
        DummySelector(),
        DummyWeighting(),
        rank_column="Sharpe",
        rebalance_strategies=["dummy"],
        rebalance_params={"dummy": {}},
    )

    previous: pd.Series | None = None
    for date in sorted(pf.history):
        weights = pf.history[date].astype(float)
        if previous is None:
            expected = float(np.abs(weights).sum())
        else:
            union = previous.index.union(weights.index)
            expected = float(
                np.abs(
                    weights.reindex(union, fill_value=0.0)
                    - previous.reindex(union, fill_value=0.0)
                ).sum()
            )
        assert pf.turnover[date] == pytest.approx(expected)
        previous = weights

