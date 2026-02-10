"""Tests for path distribution helper utilities."""

import pandas as pd
import pytest

from trend_analysis.viz.path_dist import _select_terminal_values


def test_select_terminal_values_unexpected_level_names_raise():
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    columns = pd.MultiIndex.from_product(
        [["scenario_a", "scenario_b"], ["path_1", "path_2"]],
        names=["scenario", "path"],
    )
    frame = pd.DataFrame(
        [[1.0, 1.1, 0.9, 1.05], [1.2, 1.3, 1.0, 1.1]],
        index=dates,
        columns=columns,
    )

    with pytest.raises(ValueError, match="MultiIndex levels must be named"):
        _select_terminal_values(frame, max_paths=None)


def test_select_terminal_values_unexpected_level_order():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    columns = pd.MultiIndex.from_product(
        [["path_1", "path_2"], ["NAV"]],
        names=["path", "asset"],
    )
    frame = pd.DataFrame(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [1.2, 1.05],
        ],
        index=dates,
        columns=columns,
    )

    terminal = _select_terminal_values(frame, max_paths=None)

    assert terminal.loc["path_1"] == 1.2
    assert terminal.loc["path_2"] == 1.05


def test_select_terminal_values_coerces_numeric_strings():
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    frame = pd.DataFrame(
        {
            "path_1": ["1.0", "1.2"],
            "path_2": ["0.9", "1.1"],
        },
        index=dates,
    )

    terminal = _select_terminal_values(frame, max_paths=None)

    assert pd.api.types.is_numeric_dtype(terminal)
    assert terminal.loc["path_1"] == 1.2
    assert terminal.loc["path_2"] == 1.1
