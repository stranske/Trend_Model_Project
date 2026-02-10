"""Tests for quantile band pairing behavior."""

from trend_analysis.viz.utils import quantile_bands


def test_quantile_bands_even_pairing():
    bands = quantile_bands([0.1, 0.25, 0.75, 0.9])

    assert [(band.lower, band.upper) for band in bands] == [
        (0.1, 0.9),
        (0.25, 0.75),
    ]


def test_quantile_bands_odd_length():
    bands = quantile_bands([0.05, 0.5, 0.95])

    assert [(band.lower, band.upper) for band in bands] == [(0.05, 0.95)]
