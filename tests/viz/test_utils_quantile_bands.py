"""Tests for quantile band pairing behavior."""

import pytest

from trend_analysis.viz.utils import quantile_bands, validate_quantiles


def test_quantile_bands_even_pairing():
    bands = quantile_bands([0.1, 0.25, 0.75, 0.9])

    assert [(band.lower, band.upper) for band in bands] == [
        (0.1, 0.9),
        (0.25, 0.75),
    ]


def test_quantile_bands_odd_length():
    bands = quantile_bands([0.05, 0.5, 0.95])

    assert [(band.lower, band.upper) for band in bands] == [(0.05, 0.95)]


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
    result = quantile_bands([0.5, 0.5])

    assert len(result) == 0


def test_quantile_bands_single_element():
    result = quantile_bands([0.5])

    assert len(result) == 0
