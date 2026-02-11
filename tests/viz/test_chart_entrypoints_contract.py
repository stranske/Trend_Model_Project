"""Contract tests for chart entrypoint API consistency."""

from __future__ import annotations

import inspect

from trend_analysis.viz import sharpe_ladder
from trend_analysis.viz.charts import corr_heatmap, rolling_panel, seasonality_heatmap


ENTRYPOINTS = (
    corr_heatmap.build_figure,
    rolling_panel.build_figure,
    seasonality_heatmap.build_figure,
    sharpe_ladder.build_figure,
)


def test_all_chart_modules_expose_build_figure() -> None:
    for entrypoint in ENTRYPOINTS:
        assert callable(entrypoint)


def test_primary_input_parameter_name_is_consistent() -> None:
    first_param_names = [next(iter(inspect.signature(fn).parameters.values())).name for fn in ENTRYPOINTS]
    assert first_param_names == ["data", "data", "data", "data"]


def test_public_entrypoints_do_not_call_streamlit_api() -> None:
    for entrypoint in ENTRYPOINTS:
        entrypoint_source = inspect.getsource(entrypoint)
        assert "st." not in entrypoint_source
