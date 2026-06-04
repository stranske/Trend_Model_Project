from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from trend_analysis.reporting.portfolio_series import weighted_sum


ROOT = Path(__file__).resolve().parents[1]


def _module_ast(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def test_export_uses_shared_weighted_sum_helper() -> None:
    tree = _module_ast("src/trend_analysis/export/__init__.py")
    local_portfolio_defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "portfolio_series"
    ]

    assert local_portfolio_defs == []
    assert "weighted_sum(" in (ROOT / "src/trend_analysis/export/__init__.py").read_text(
        encoding="utf-8"
    )


def test_weighted_sum_preserves_export_weighting_behavior() -> None:
    frame = pd.DataFrame({"A": [0.10, 0.00], "B": [0.05, 0.02]})

    weighted = weighted_sum(frame, {"A": 2.0, "B": 1.0})
    expected_weighted = frame.mul(pd.Series({"A": 2.0 / 3.0, "B": 1.0 / 3.0}), axis=1).sum(
        axis=1
    )
    pd.testing.assert_series_equal(weighted, expected_weighted)

    equal = weighted_sum(frame, None)
    expected_equal = frame.mul(pd.Series({"A": 0.5, "B": 0.5}), axis=1).sum(axis=1)
    pd.testing.assert_series_equal(equal, expected_equal)


def test_viz_charts_reuse_canonical_nav_adapter() -> None:
    for relative_path in (
        "src/trend_analysis/viz/charts/rolling_panel.py",
        "src/trend_analysis/viz/charts/seasonality_heatmap.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        local_to_nav_wide = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_to_nav_wide"
        ]
        assert local_to_nav_wide == []
        assert "_paths_to_wide_nav(" in source
