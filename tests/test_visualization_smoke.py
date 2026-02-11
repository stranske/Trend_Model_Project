"""Visualization smoke tests for Plotly figure construction and JSON serialization."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

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


def _smoke_figures() -> list[go.Figure]:
    nav_paths = _sample_nav_paths()
    returns = _sample_returns()
    return [
        fan.make(nav_paths, max_paths=None),
        path_dist.make(nav_paths, max_paths=None),
        risk_return.make(returns),
    ]


def test_plotly_figures_create_successfully() -> None:
    figures = _smoke_figures()
    assert figures
    for fig in figures:
        assert isinstance(fig, go.Figure)


def test_plotly_figures_to_json_returns_valid_json() -> None:
    for fig in _smoke_figures():
        payload = json.loads(fig.to_json())
        assert isinstance(payload, dict)
        assert "data" in payload
        assert "layout" in payload
