"""Tests for fan helper utilities."""

import pandas as pd
import pytest

from trend_analysis.viz.fan import _select_nav_paths


def test_select_nav_paths_empty_frame_raises():
    empty = pd.DataFrame()

    with pytest.raises(ValueError, match="nav_paths cannot be empty"):
        _select_nav_paths(empty, max_paths=None)


def test_select_nav_paths_unsorted_labels_sorts_index():
    dates = pd.to_datetime(["2020-03-31", "2020-01-31", "2020-02-29"])
    frame = pd.DataFrame(
        {
            0.9: [1.2, 1.0, 1.1],
            0.1: [0.8, 0.9, 0.95],
        },
        index=dates,
    )

    selected = _select_nav_paths(frame, max_paths=None)

    assert selected.shape == (3, 2)
    assert list(selected.index) == sorted(dates)


def test_select_nav_paths_duplicate_labels_raise():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    frame = pd.DataFrame(
        [[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]],
        index=dates,
        columns=["path_1", "path_1"],
    )

    with pytest.raises(ValueError, match="duplicate path labels"):
        _select_nav_paths(frame, max_paths=None)
