from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from trend_analysis.multi_period.engine import Portfolio, run_schedule


@dataclass
class DummySelector:
    """Selector that returns the provided frame unchanged."""

    rank_column: str = "Sharpe"

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return score_frame, score_frame


@dataclass
class SequenceWeighting:
    """Weighting scheme that yields a deterministic sequence of weights."""

    sequences: tuple[dict[str, float], ...]
    _idx: int = 0

    def update(
        self, scores: pd.Series, days: int = 30
    ) -> None:  # pragma: no cover - hook for protocol
        # The engine will call update when rank_column is present. The sequence
        # weighting used in this test is deterministic and state free, so we do
        # not need to adjust any internal state here.
        pass

    def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        del date
        weights = self.sequences[self._idx]
        self._idx += 1
        ordered = pd.Series(weights, index=selected.index, dtype=float).fillna(0.0)
        return ordered.to_frame("weight")


def build_score_frames() -> dict[str, pd.DataFrame]:
    sharpe_col = "Sharpe"
    first = pd.DataFrame({sharpe_col: [1.0, 2.0]}, index=["FundA", "FundB"])
    second = pd.DataFrame({sharpe_col: [3.0, 4.0]}, index=["FundB", "FundC"])
    return {"2020-01-31": first, "2020-02-29": second}


def test_run_schedule_turnover_debug_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_frames = build_score_frames()
    selector = DummySelector()
    weighting = SequenceWeighting(
        sequences=(
            {"FundA": 0.6, "FundB": 0.4},
            {"FundB": 0.2, "FundC": 0.8},
        )
    )

    monkeypatch.setenv("DEBUG_TURNOVER_VALIDATE", "1")
    try:
        portfolio = run_schedule(score_frames, selector, weighting, rank_column="Sharpe")

        # Ensure the debug validator populated history and turnover for each period.
        assert isinstance(portfolio, Portfolio)
        assert set(portfolio.history) == {"2020-01-31", "2020-02-29"}
        assert set(portfolio.turnover) == {"2020-01-31", "2020-02-29"}
        # The second period should see turnover from introducing FundC while FundA is removed.
        assert portfolio.turnover["2020-02-29"] > 0.0
    finally:
        monkeypatch.delenv("DEBUG_TURNOVER_VALIDATE", raising=False)
