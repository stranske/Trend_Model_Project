import numpy as np
import pandas as pd
import pytest

import trend_analysis.core.rank_selection as rs


def test_canonicalise_labels_handles_duplicates_and_blanks():
    labels = [" A", "A", " ", "B", "A "]
    canonical = rs._canonicalise_labels(labels)
    assert canonical == ["A", "A_2", "Unnamed_3", "B", "A_3"]


def test_ensure_canonical_columns_returns_same_frame_when_clean():
    frame = pd.DataFrame({"A": [1], "B": [2]})
    result = rs._ensure_canonical_columns(frame)
    assert result is frame
    assert list(result.columns) == ["A", "B"]


def test_ensure_canonical_columns_normalizes_and_copies_when_needed():
    frame = pd.DataFrame({" A": [1], "A ": [2]})
    result = rs._ensure_canonical_columns(frame)
    assert result is not frame
    assert list(result.columns) == ["A", "A_2"]


def test_json_default_handles_supported_types_and_errors():
    assert set(rs._json_default({1, 2})) == {1, 2}
    assert rs._json_default(("x", "y")) == ["x", "y"]
    assert rs._json_default(np.array([1, 2])) == [1, 2]
    assert rs._json_default(np.float64(1.5)) == pytest.approx(1.5)

    with pytest.raises(TypeError):
        rs._json_default({"unsupported": object()})


def test_make_window_key_canonicalizes_universe_and_stats_cfg():
    cfg = rs.RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.01)
    key_one = rs.make_window_key("2020-01", "2020-12", [" A", "B"], cfg)
    key_two = rs.make_window_key("2020-01", "2020-12", ["B", "A"], cfg)
    assert key_one == key_two
