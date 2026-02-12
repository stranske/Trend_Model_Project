"""Visualization smoke tests for Plotly figure construction and JSON serialization."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import pytest

from trend_analysis.viz import fan, path_dist, risk_return


def _sample_nav_paths() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path_1": [1.00, 1.03, 1.05, 1.08],
            "path_2": [1.00, 0.99, 1.02, 1.07],
        },
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )


def _sample_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_a": [0.01, -0.01, 0.015, 0.002],
            "strategy_b": [0.0, 0.012, -0.004, 0.01],
        },
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )


@pytest.mark.parametrize(
    ("builder", "payload"),
    [
        pytest.param(lambda nav, _ret: fan.make(nav, max_paths=None), "nav", id="fan"),
        pytest.param(lambda nav, _ret: path_dist.make(nav, max_paths=None), "nav", id="path_dist"),
        pytest.param(lambda _nav, ret: risk_return.make(ret), "returns", id="risk_return"),
    ],
)
def test_plotly_figures_create_successfully(builder, payload: str) -> None:
    nav_paths = _sample_nav_paths()
    returns = _sample_returns()
    figure = builder(nav_paths, returns)

    assert payload in {"nav", "returns"}
    assert isinstance(figure, go.Figure)
    assert figure.data


def test_plotly_figures_to_json_returns_valid_json() -> None:
    nav_paths = _sample_nav_paths()
    returns = _sample_returns()
    figures = [
        fan.make(nav_paths, max_paths=None),
        path_dist.make(nav_paths, max_paths=None),
        risk_return.make(returns),
    ]
    for fig in figures:
        payload = json.loads(fig.to_json())
        assert isinstance(payload, dict)
        assert isinstance(payload.get("data"), list)
        assert isinstance(payload.get("layout"), dict)
