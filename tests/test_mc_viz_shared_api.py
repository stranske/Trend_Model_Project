from __future__ import annotations

import inspect

from trend.mc import execute_mc_viz
from trend.mc.viz import CHART_REQUIREMENTS


def test_execute_mc_viz_public_signature() -> None:
    signature = inspect.signature(execute_mc_viz)
    assert list(signature.parameters) == [
        "bundle_path",
        "out_dir",
        "charts",
        "html",
        "json",
        "png",
    ]
    assert execute_mc_viz.__doc__


def test_chart_requirements_define_supported_mc_viz_inputs() -> None:
    assert set(CHART_REQUIREMENTS) == {"fan", "path_dist", "risk_return"}
    assert CHART_REQUIREMENTS["fan"] == ("summary", "results")
    assert CHART_REQUIREMENTS["risk_return"] == ("summary", "results")
    assert CHART_REQUIREMENTS["path_dist"] == ("summary", "results", "nav_paths.parquet")
