"""Tests for viz quantile and path-selection utilities."""

import pandas as pd
import pytest

from trend_analysis.viz.fan import _select_nav_paths
from trend_analysis.viz.path_dist import _select_terminal_values
from trend_analysis.viz.utils import quantile_bands, validate_quantiles


def test_quantile_bands_even_pairing():
    quantiles = [0.1, 0.25, 0.75, 0.9]
    bands = quantile_bands(quantiles)

    assert len(bands) == len(quantiles) // 2
    assert [(band.lower, band.upper) for band in bands] == [
        (0.1, 0.9),
        (0.25, 0.75),
    ]
    assert [band.label() for band in bands] == ["10-90%", "25-75%"]


def test_quantile_bands_odd_length_raises_value_error():
    with pytest.raises(ValueError, match="odd length quantiles"):
        quantile_bands([0.05, 0.5, 0.95])


def test_validate_quantiles_empty_input():
    with pytest.raises(ValueError):
        validate_quantiles([])


def test_validate_quantiles_negative_value():
    with pytest.raises(ValueError):
        validate_quantiles([-0.1])


def test_validate_quantiles_value_above_one():
    with pytest.raises(ValueError):
        validate_quantiles([1.5])


def test_quantile_bands_duplicate_values():
    with pytest.raises(ValueError, match="lower must be < upper"):
        quantile_bands([0.5, 0.5])


def test_quantile_bands_single_element():
    with pytest.raises(ValueError, match="odd length quantiles"):
        quantile_bands([0.5])


def test_select_nav_paths_empty_input_raises():
    empty = pd.DataFrame()

    with pytest.raises(ValueError, match="nav_paths cannot be empty"):
        _select_nav_paths(empty, max_paths=None)


def test_select_nav_paths_unsorted_labels_sorts_index_deterministically():
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
