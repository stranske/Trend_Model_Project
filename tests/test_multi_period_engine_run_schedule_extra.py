"""Additional coverage tests for ``run_schedule`` and ``Portfolio`` helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from trend_analysis.multi_period.engine import Portfolio, run_schedule
from trend_analysis.weighting import BaseWeighting


class _Selector:
    """Deterministic selector returning the incoming frame as-is."""

    def __init__(self, top_n: int = 2) -> None:
        self.top_n = top_n

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        selected = score_frame.iloc[: self.top_n].copy()
        return selected, selected


class _UpdatingWeighting(BaseWeighting):
    """Toy weighting scheme that records update calls for verification."""

    def __init__(self) -> None:
        self.update_calls: list[tuple[pd.Series, int]] = []

    def weight(
        self, selected: pd.DataFrame, date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        del date
        if selected.empty:
            return pd.DataFrame({"weight": []}, index=pd.Index([], dtype=object))
        n = len(selected)
        weights = np.linspace(1.0, 2.0, n) / np.linspace(1.0, 2.0, n).sum()
        return pd.DataFrame({"weight": weights}, index=selected.index)

    def update(
        self, scores: pd.Series, days: int
    ) -> None:  # pragma: no cover - runtime hook
        self.update_calls.append((scores.copy(), days))


@dataclass
class _StrategyCall:
    strategies: list[str]
    params: dict[str, dict[str, object]]
    current_weights: pd.Series
    target_weights: pd.Series
    scores: pd.Series | None


def _make_score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sharpe": [1.0, 0.5, -0.1],
            "zscore": [1.2, 0.0, -1.0],
        },
        index=["FundA", "FundB", "FundC"],
    )


def test_portfolio_rebalance_accepts_series_and_tracks_totals() -> None:
    pf = Portfolio()
    weights = pd.Series({"FundA": 0.6, "FundB": 0.4})
    pf.rebalance("2023-03-15", weights, turnover=0.12, cost=7.5)

    key = "2023-03-15"
    assert key in pf.history
    assert pf.history[key].to_dict() == {"FundA": 0.6, "FundB": 0.4}
    assert pf.turnover[key] == pytest.approx(0.12)
    assert pf.costs[key] == pytest.approx(7.5)
    assert pf.total_rebalance_costs == pytest.approx(7.5)


def test_run_schedule_invokes_rebalance_strategies_and_weighting_update(monkeypatch):
    frames = {
        "2020-01-31": _make_score_frame(),
        "2020-02-29": _make_score_frame(),
    }
    selector = _Selector(top_n=2)
    weighting = _UpdatingWeighting()

    calls: list[_StrategyCall] = []

    def fake_apply(
        strategies,
        params,
        current_weights,
        target_weights,
        *,
        scores=None,
        cash_policy=None,
    ):
        del cash_policy
        calls.append(
            _StrategyCall(
                strategies=list(strategies),
                params=params,
                current_weights=current_weights.copy(),
                target_weights=target_weights.copy(),
                scores=None if scores is None else scores.copy(),
            )
        )
        final = target_weights.copy()
        final[:] = np.linspace(0.7, 0.3, len(final))
        return final, 1.25

    monkeypatch.setattr(
        "trend_analysis.multi_period.engine.apply_rebalancing_strategies",
        fake_apply,
    )

    portfolio = run_schedule(
        frames,
        selector,
        weighting,
        rank_column="Sharpe",
        rebalance_strategies=["mock"],
        rebalance_params={"mock": {"alpha": 0.5}},
    )

    assert len(portfolio.history) == 2
    assert calls, "Rebalancing strategies should be invoked"
    first_call = calls[0]
    assert first_call.strategies == ["mock"]
    assert first_call.params["mock"] == {"alpha": 0.5}
    assert list(first_call.target_weights.index) == ["FundA", "FundB"]
    assert weighting.update_calls, "Weighting.update should be called"
    # First update occurs with zero elapsed days; second uses actual delta (~29 days)
    assert weighting.update_calls[0][1] == 0
    assert weighting.update_calls[1][1] > 0
