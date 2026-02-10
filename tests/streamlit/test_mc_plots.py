from __future__ import annotations

import pandas as pd

from streamlit_app.components import mc_plots


def _sample_nav_paths(rows: int = 12, paths: int = 6) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {f"path_{i}": [float(i) + j * 0.5 for j in range(rows)] for i in range(paths)}
    return pd.DataFrame(data, index=index)


def test_fan_chart_uses_quantile_bands() -> None:
    nav_paths = _sample_nav_paths()

    fig = mc_plots.fan_chart(nav_paths)

    assert len(fig.data) == 5
    assert fig.data[1].fill == "tonexty"
    assert fig.data[3].fill == "tonexty"
    assert fig.data[0].fill != "tonexty"
    assert fig.data[2].fill != "tonexty"
    assert fig.data[4].fill != "tonexty"
    assert fig.data[4].name == "Median"


def test_fan_chart_empty_returns_empty_figure() -> None:
    fig = mc_plots.fan_chart(pd.DataFrame())

    assert len(fig.data) == 0


def test_path_distribution_chart_uses_terminal_metric() -> None:
    results = pd.DataFrame(
        {
            "strategy": ["A", "A", "B", "B"],
            "terminal_wealth": [120.0, 118.0, 110.0, 112.0],
        }
    )

    fig = mc_plots.path_distribution_chart(results)

    assert len(fig.data) >= 1
    assert fig.data[0].type == "histogram"


def test_risk_return_chart_builds_scatter() -> None:
    results = pd.DataFrame(
        {
            "strategy": ["A", "A", "B", "B"],
            "terminal_wealth": [120.0, 118.0, 110.0, 112.0],
            "max_drawdown": [-0.2, -0.25, -0.3, -0.28],
        }
    )

    fig = mc_plots.risk_return_chart(results)

    assert len(fig.data) >= 1
    assert fig.data[0].type == "scatter"
    assert all(value >= 0 for value in fig.data[0].x)
