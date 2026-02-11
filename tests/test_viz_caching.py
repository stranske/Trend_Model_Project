from __future__ import annotations

import importlib
import sys
from collections import Counter
from types import SimpleNamespace

import pandas as pd


def _sample_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fold_id": 1, "fold_label": "Fold 1", "path_id": 1, "strategy": "A", "sharpe": 1.0},
            {"fold_id": 1, "fold_label": "Fold 1", "path_id": 2, "strategy": "A", "sharpe": 1.2},
            {"fold_id": 1, "fold_label": "Fold 1", "path_id": 3, "strategy": "B", "sharpe": 0.6},
        ]
    )


def _sample_nav_paths() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"])
    return pd.DataFrame(
        {
            0: [1.0, 1.1, 1.21, 1.20],
            1: [1.0, 0.9, 0.99, 1.01],
        },
        index=index,
    )


def test_viz_cache_data_is_decorated_and_invoked(monkeypatch) -> None:
    decorated: set[str] = set()
    called = Counter()

    def cache_data(*_args, **_kwargs):
        def decorator(func):
            key = f"{func.__module__}.{func.__name__}"
            decorated.add(key)

            def wrapped(*args, **kwargs):
                called[key] += 1
                return func(*args, **kwargs)

            wrapped.__wrapped__ = func
            return wrapped

        return decorator

    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(cache_data=cache_data))

    adapters = importlib.reload(importlib.import_module("trend_analysis.viz.adapters"))
    corr_heatmap = importlib.reload(
        importlib.import_module("trend_analysis.viz.charts.corr_heatmap")
    )
    rolling_panel = importlib.reload(
        importlib.import_module("trend_analysis.viz.charts.rolling_panel")
    )
    seasonality_heatmap = importlib.reload(
        importlib.import_module("trend_analysis.viz.charts.seasonality_heatmap")
    )
    sharpe_ladder = importlib.reload(importlib.import_module("trend_analysis.viz.sharpe_ladder"))

    summary = adapters.make_summary(_sample_results_frame())
    paths = adapters.make_paths(_sample_nav_paths())

    corr_heatmap.build_figure(paths)
    rolling_panel.build_figure(paths, window=2)
    seasonality_heatmap.build_figure(paths)
    sharpe_ladder.build_figure(summary)

    expected = {
        "trend_analysis.viz.adapters._make_summary_cached",
        "trend_analysis.viz.adapters._make_paths_cached",
        "trend_analysis.viz.charts.corr_heatmap._prepare_corr_matrix",
        "trend_analysis.viz.charts.rolling_panel._prepare_panel_series",
        "trend_analysis.viz.charts.seasonality_heatmap._prepare_seasonality_matrix",
        "trend_analysis.viz.sharpe_ladder.prepare_sharpe_ladder",
    }
    assert expected.issubset(decorated)
    for name in expected:
        assert called[name] >= 1
