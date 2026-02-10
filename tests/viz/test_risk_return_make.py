"""Tests for risk/return make helper."""

import json

import pandas as pd
import plotly.graph_objects as go

from trend_analysis.viz import risk_return


def test_risk_return_make_minimal_input():
    returns = pd.DataFrame(
        {
            "strategy_a": [0.01, -0.02, 0.015, 0.005],
            "strategy_b": [0.0, 0.01, -0.005, 0.02],
        },
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )

    fig = risk_return.make(returns, periods_per_year=12.0)

    assert isinstance(fig, go.Figure)
    payload = fig.to_json()
    assert json.loads(payload)
