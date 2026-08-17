from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trend_analysis.multi_period import engine


def test_prepare_threshold_hold_period_preserves_empty_universe_placeholder(
    monkeypatch,
) -> None:
    period = SimpleNamespace(
        in_start="2020-01",
        in_end="2020-02",
        out_start="2020-03",
        out_end="2020-03",
    )
    in_df = pd.DataFrame({"Fund": [0.01]})
    out_df = pd.DataFrame({"Fund": [float("nan")]})
    monkeypatch.setattr(
        engine,
        "_setup_period",
        lambda *_args, **_kwargs: engine._PeriodSetup(
            period_ts=pd.Timestamp("2020-03-31"),
            in_df=in_df,
            out_df=out_df,
            fund_cols=["Fund"],
        ),
    )

    prepared = engine._prepare_threshold_hold_period(
        engine._ThresholdHoldPeriodInput(
            pt=period,
            cooldown_periods=0,
            cooldown_book={},
            valid_universe=lambda *_args, **_kwargs: (in_df, out_df, ["Fund"], None),
            df=pd.DataFrame(),
            missing_policy_diagnostic={"policy": "drop"},
        )
    )

    assert prepared.fund_cols == []
    assert prepared.placeholder is not None
    assert prepared.placeholder["period"] == ("2020-01", "2020-02", "2020-03", "2020-03")
    assert prepared.placeholder["missing_policy_diagnostic"] == {"policy": "drop"}
