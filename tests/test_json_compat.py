"""Contract coverage for JSON adapters with intentionally different shapes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend.commands import report_export as owned
from trend_analysis import walk_forward
from trend_analysis.backtesting import harness
from trend_analysis.core import rank_selection
from trend_analysis.util.json_compat import json_compatible, json_primitive


def test_shared_primitives_normalise_nan_and_reject_unknown_values() -> None:
    assert json_primitive(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"
    assert json_primitive(np.array([np.int64(2), np.float64(3.5)])) == [2, 3.5]
    assert json_compatible({"nan": np.float64("nan"), "path": Path("report.json")}) == {
        "nan": None,
        "path": "report.json",
    }
    with pytest.raises(TypeError, match="not JSON serialisable"):
        json_compatible(object())


def test_callers_preserve_their_documented_container_shapes() -> None:
    timestamp = pd.Timestamp("2024-01-01")
    assert harness._json_default(timestamp) == timestamp.isoformat()
    assert harness._json_default(np.int64(2)) == 2.0
    assert harness._json_default(np.float32("nan")) is None

    assert walk_forward._json_default(pd.Index(["a", "b"])) == ["a", "b"]
    assert walk_forward._json_default(np.float32("nan")) is None

    assert rank_selection._json_default(np.array([1, 2])) == [1, 2]
    assert rank_selection._json_default(("a", "b")) == ["a", "b"]
    assert rank_selection._json_default(np.float32("nan")) is None

    series = pd.Series([np.float32(1.5), np.float32("nan")], index=[timestamp, "b"])
    encoded = json.dumps(
        {"series": series, "frame": pd.DataFrame({"x": series})},
        default=owned._json_default,
    )
    decoded = json.loads(encoded)
    expected = {str(timestamp): 1.5, "b": None}
    assert decoded["series"] == expected
    assert decoded["frame"]["x"] == expected

    with pytest.raises(TypeError, match="not JSON serialisable"):
        rank_selection._json_default(object())


def test_backtest_to_json_normalizes_builtin_nan_metrics() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    empty = pd.Series(dtype=float)
    empty_frame = pd.DataFrame()
    result = harness.BacktestResult(
        returns=empty,
        equity_curve=pd.Series([1.0], index=idx[:1]),
        weights=empty_frame,
        turnover=empty,
        per_period_turnover=empty,
        transaction_costs=empty,
        cost_drag=empty,
        rolling_sharpe=empty,
        drawdown=empty,
        metrics={"cagr": float("nan"), "sharpe": float("nan")},
        cost_model=harness.CostModel(),
        calendar=idx[:1],
        window_mode="rolling",
        window_size=1,
        training_windows={},
    )

    encoded = result.to_json()
    assert "NaN" not in encoded
    parsed = json.loads(encoded)
    assert parsed["metrics"]["cagr"] is None
    assert parsed["metrics"]["sharpe"] is None
