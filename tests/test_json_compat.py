"""Contract coverage for JSON adapters with intentionally different shapes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend import cli
from trend_analysis.backtesting import harness
from trend_analysis.core import rank_selection
from trend_analysis.util.json_compat import json_compatible, json_primitive
from trend_analysis import walk_forward


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
        {"series": series, "frame": pd.DataFrame({"x": series})}, default=cli._json_default
    )
    decoded = json.loads(encoded)
    expected = {str(timestamp): 1.5, "b": None}
    assert decoded["series"] == expected
    assert decoded["frame"]["x"] == expected

    with pytest.raises(TypeError, match="not JSON serialisable"):
        rank_selection._json_default(object())
