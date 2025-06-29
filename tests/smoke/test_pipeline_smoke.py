import pandas as pd
from trend_analysis.pipeline import single_period_run

toy = pd.DataFrame(
    {
        "Date": pd.date_range("2020-01-31", periods=6, freq="M"),
        "FundA": [0.02, -0.01, 0.03, 0.02, -0.01, 0.04],
        "FundB": [0.01, 0.01, 0.00, 0.02, 0.01, 0.03],
        "FundC": [0.00, 0.02, -0.02, 0.03, 0.02, 0.01],
    }
)


def test_single_period_smoke() -> None:
    """Fast contract check: returns non-empty dict with score_frame key."""
    res = single_period_run(
        toy,
        in_start="2020-01",
        in_end="2020-03",
        out_start="2020-04",
        out_end="2020-06",
        target_vol=0.10,
        monthly_cost=0.0,
        selection_mode="all",
    )
    assert res is not None
    assert "selected_funds" in res
    assert "score_frame" in res  # FAILS if Phase 1 isn't done
    assert len(res["selected_funds"]) > 0
