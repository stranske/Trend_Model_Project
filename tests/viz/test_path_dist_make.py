"""Tests for path distribution make helper."""

import json

import pandas as pd
import plotly.graph_objects as go

import trend_analysis.viz.path_dist as path_dist


def test_path_dist_make_minimal_input():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    nav_paths = {
        "path_1": [1.0, 1.05, 1.1, 1.2],
        "path_2": [1.0, 0.98, 1.02, 1.08],
    }
    frame = pd.DataFrame(nav_paths, index=dates)

    fig = path_dist.make(frame, bins=5, max_paths=None)

    assert isinstance(fig, go.Figure)
    payload = json.loads(fig.to_json())
    assert isinstance(payload, dict)
