from __future__ import annotations

import pandas as pd
import pytest

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


def test_fan_chart_trace_count_stays_constant_with_many_paths() -> None:
    nav_paths = _sample_nav_paths(paths=250)

    fig = mc_plots.fan_chart(nav_paths)

    assert len(fig.data) == 5
    assert fig.data[1].fill == "tonexty"
    assert fig.data[3].fill == "tonexty"


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


class _StreamlitStub:
    def __init__(self) -> None:
        self.warning_messages: list[str] = []
        self.plotly_calls: list[tuple[object, bool]] = []

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def plotly_chart(self, fig: object, *, use_container_width: bool = False) -> None:
        self.plotly_calls.append((fig, use_container_width))


def test_render_fan_chart_warns_on_empty_nav_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _StreamlitStub()
    monkeypatch.setattr(mc_plots, "st", stub)

    fig = mc_plots.render_fan_chart(pd.DataFrame())

    assert len(fig.data) == 0
    assert stub.warning_messages
    assert "Fan chart unavailable" in stub.warning_messages[0]
    assert stub.plotly_calls
    assert stub.plotly_calls[0][1] is True


def test_render_path_distribution_warns_on_missing_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _StreamlitStub()
    monkeypatch.setattr(mc_plots, "st", stub)

    results = pd.DataFrame({"terminal_wealth": [120.0, 118.0]})

    fig = mc_plots.render_path_distribution_chart(results)

    assert len(fig.data) == 0
    assert stub.warning_messages
    assert "Path distribution chart unavailable" in stub.warning_messages[0]


def test_render_risk_return_warns_on_missing_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _StreamlitStub()
    monkeypatch.setattr(mc_plots, "st", stub)

    results = pd.DataFrame({"strategy": ["A", "B"], "label": ["foo", "bar"]})

    fig = mc_plots.render_risk_return_chart(results)

    assert len(fig.data) == 0
    assert stub.warning_messages
    assert "Risk-return chart unavailable" in stub.warning_messages[0]
    assert stub.plotly_calls
