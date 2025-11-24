import pandas as pd

from trend_analysis.pipeline import run_analysis

toy = pd.DataFrame(
    {
        "Date": pd.date_range("2020-01-31", periods=6, freq="ME"),
        "FundA": [0.02, -0.01, 0.03, 0.02, -0.01, 0.04],
        "FundB": [0.01, 0.01, 0.00, 0.02, 0.01, 0.03],
        "FundC": [0.00, 0.02, -0.02, 0.03, 0.02, 0.01],
        "RF": [0.001] * 6,
    }
)


def test_single_period_smoke() -> None:
    """Fast contract check: returns non-empty dict with score_frame key."""
    res = run_analysis(
        toy,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.10,
        0.0,
        selection_mode="all",
        risk_free_column="RF",
    )
    assert res is not None
    assert "selected_funds" in res
    assert "score_frame" in res  # FAILS if Phase 1 isn't done
    assert len(res["selected_funds"]) > 0
