from __future__ import annotations

import ast
from pathlib import Path

import trend.mc.charts as mc_charts
import trend.mc.viz as mc_viz
from trend.reporting import quick_summary, unified
from trend.reporting._matplotlib import init_matplotlib


def test_mc_viz_reuses_chart_nav_requirement_constant() -> None:
    assert mc_viz.NAV_PATH_REQUIRED_CHARTS is mc_charts.NAV_PATH_REQUIRED_CHARTS


def test_reporting_modules_reuse_shared_matplotlib_initializer() -> None:
    assert quick_summary.init_matplotlib is init_matplotlib
    assert unified.init_matplotlib is init_matplotlib


def test_no_local_matplotlib_initializer_definitions_remain() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for relative in ("src/trend/reporting/quick_summary.py", "src/trend/reporting/unified.py"):
        module = ast.parse((repo_root / relative).read_text(encoding="utf-8"))
        function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        assert "_init_matplotlib" not in function_names
