"""Tests for fan chart make helper."""

import json

import pandas as pd
import plotly.graph_objects as go

import trend_analysis.viz.fan as fan


def test_fan_make_minimal_input():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    nav_paths = {
        "path_1": [1.0, 1.1, 1.2],
        "path_2": [1.0, 0.9, 1.05],
    }
    frame = pd.DataFrame(nav_paths, index=dates)

    fig = fan.make(frame, max_paths=None, show_paths=True)

    assert isinstance(fig, go.Figure)
    payload = json.loads(fig.to_json())
    assert isinstance(payload, dict)
