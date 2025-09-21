import pandas as pd

from trend_analysis.multi_period.engine import Portfolio, run_schedule
from trend_analysis.weighting import BaseWeighting


class RecordingWeighting(BaseWeighting):
    def __init__(self) -> None:
        self.update_calls: list[tuple[pd.Series, int]] = []

    def weight(
        self, selected: pd.DataFrame, date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        del date
        if selected.empty:
            return pd.DataFrame(columns=["weight"])
        weights = pd.Series(1.0 / len(selected), index=selected.index, dtype=float)
        return weights.to_frame("weight")

    def update(self, scores: pd.Series, days: int) -> None:
        self.update_calls.append((scores.astype(float), int(days)))


class SimpleSelector:
    column = "Sharpe"

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return score_frame, score_frame


def test_portfolio_rebalance_accepts_series_and_mapping() -> None:
    portfolio = Portfolio()
    series = pd.Series({"Alpha": 0.2, "Beta": 0.8}, dtype=float)
    portfolio.rebalance("2020-01-31", series, turnover=0.3, cost=1.5)

    assert portfolio.history["2020-01-31"].tolist() == [0.2, 0.8]
    assert portfolio.turnover["2020-01-31"] == 0.3
    assert portfolio.costs["2020-01-31"] == 1.5

    mapping = {"Gamma": 0.5, "Delta": 0.5}
    portfolio.rebalance(pd.Timestamp("2020-02-29"), mapping)

    stored = portfolio.history["2020-02-29"]
    assert stored.index.tolist() == ["Gamma", "Delta"]
    assert stored.tolist() == [0.5, 0.5]


def test_run_schedule_invokes_weighting_update() -> None:
    score_frames = {
        "2020-01-31": pd.DataFrame(
            {"Sharpe": [1.5, 0.9], "Other": [0.1, 0.2]}, index=["Alpha", "Beta"]
        ),
        "2020-02-29": pd.DataFrame(
            {"Sharpe": [1.2, 1.0], "Other": [0.3, 0.1]}, index=["Alpha", "Beta"]
        ),
    }

    weighting = RecordingWeighting()
    selector = SimpleSelector()

    portfolio = run_schedule(score_frames, selector, weighting, rank_column="Sharpe")

    # update called for each period with days difference reflected in the second call
    assert len(weighting.update_calls) == 2
    first_scores, first_days = weighting.update_calls[0]
    second_scores, second_days = weighting.update_calls[1]
    assert first_days == 0
    assert second_days > 0
    pd.testing.assert_series_equal(first_scores, score_frames["2020-01-31"]["Sharpe"])
    pd.testing.assert_series_equal(second_scores, score_frames["2020-02-29"]["Sharpe"])

    assert sorted(portfolio.history) == ["2020-01-31", "2020-02-29"]
