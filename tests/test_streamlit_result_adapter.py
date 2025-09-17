"""Tests for the RunResultAdapter used by the Streamlit UI."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if "matplotlib" not in sys.modules:  # pragma: no cover - lightweight stub for tests
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

from trend_analysis.api import run_simulation  # noqa: E402
from trend_analysis.config import Config  # noqa: E402

from streamlit_app.result_adapter import RunResultAdapter, adapt_run_result  # noqa: E402


def _demo_returns_frame() -> pd.DataFrame:
    dates = pd.date_range("2018-01-31", periods=72, freq="ME")
    trend = np.linspace(0.005, 0.015, len(dates))
    data = {
        "Date": dates,
        "Fund_A": 0.01 + 0.002 * np.sin(np.linspace(0, 6, len(dates))) + trend,
        "Fund_B": 0.008 + 0.001 * np.cos(np.linspace(0, 4, len(dates))) - trend / 2,
        "SPX": 0.006 + 0.0005 * np.sin(np.linspace(0, 3, len(dates))),
    }
    return pd.DataFrame(data)


def _demo_config() -> Config:
    return Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={"target_vol": 0.10},
        sample_split={
            "in_start": "2018-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2022-12",
        },
        portfolio={"selection_mode": "rank", "rank": {"inclusion_approach": "top_n", "n": 2, "score_by": "sharpe_ratio"}},
        benchmarks={"spx": "SPX"},
        metrics={"registry": ["sharpe_ratio", "annual_return"]},
        export={},
        run={},
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_adapt_run_result_produces_curves():
    df = _demo_returns_frame()
    cfg = _demo_config()
    result = run_simulation(cfg, df)

    adapted = adapt_run_result(result)
    assert isinstance(adapted, RunResultAdapter)

    curve = adapted.portfolio_curve()
    assert not curve.empty
    dd = adapted.drawdown_curve()
    assert len(dd) == len(curve)
    summary = adapted.summary()
    assert "total_return" in summary
    assert adapted.portfolio.index.equals(curve.index)

    weights = adapted.weights
    assert isinstance(weights, dict)
    assert weights

    # Ensure fallback info and metrics are surfaced
    assert getattr(adapted, "fallback_info", None) == getattr(result, "fallback_info", None)
    pd.testing.assert_frame_equal(adapted.metrics, result.metrics)


def test_adapt_run_result_passthrough():
    class _Dummy:
        def portfolio_curve(self):
            return pd.Series([1, 2, 3])

    dummy = _Dummy()
    assert adapt_run_result(dummy) is dummy
