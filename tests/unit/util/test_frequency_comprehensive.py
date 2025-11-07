"""Comprehensive tests for :mod:`trend_analysis.util.frequency`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from trend_analysis.util import frequency as freq


@pytest.mark.parametrize(
    "values",
    [
        pd.DatetimeIndex(
            ["2024-01-03", "2024-01-01", "2024-01-02"], dtype="datetime64[ns]"
        ),
        ["2024-01-02", "2024-01-01", "2024-01-03"],
    ],
)
def test_as_datetime_index_normalises_and_sorts(values: Iterable[object]) -> None:
    """Internal coercion should return a sorted :class:`DatetimeIndex`."""

    result = freq._as_datetime_index(values)

    assert isinstance(result, pd.DatetimeIndex)
    assert list(result) == sorted(result)


@pytest.mark.parametrize(
    "values",
    [
        ["2024-01-01", "not-a-date"],
        [object(), object()],
    ],
)
def test_as_datetime_index_rejects_non_datetime_like(values: Sequence[object]) -> None:
    """Non-datetime inputs should raise a descriptive error."""

    with pytest.raises(ValueError, match="datetime-like"):
        freq._as_datetime_index(values)


def test_as_datetime_index_retries_after_initial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure fallback path converts convertible iterables after an initial failure."""

    original = freq.pd.DatetimeIndex
    call_counter = {"count": 0}

    class FlakyDatetimeIndex(original):  # type: ignore[misc]
        def __new__(cls, data=None, *args, **kwargs):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                raise ValueError("temporary construction failure")
            return original.__new__(original, data, *args, **kwargs)

    monkeypatch.setattr(freq.pd, "DatetimeIndex", FlakyDatetimeIndex)

    result = freq._as_datetime_index(["2024-01-01", "2024-01-02"])

    assert list(result) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert call_counter["count"] == 2


@pytest.mark.parametrize(
    ("inferred", "expected"),
    [
        ("W-SUN", "W"),
        ("B", "D"),
        ("C", "D"),
        ("M", "M"),
        ("SM", "M"),
        ("Q-DEC", "Q"),
        ("A", "Y"),
        ("Y", "Y"),
        ("", None),
        (None, None),
        ("custom", None),
    ],
)
def test_map_inferred_collapses_known_codes(
    inferred: str | None, expected: str | None
) -> None:
    assert freq._map_inferred(inferred) == expected


@pytest.mark.parametrize(
    ("diffs", "expected"),
    [
        (np.array([], dtype=float), "M"),
        (np.array([1.0, 2.0, 3.5], dtype=float), "D"),
        (np.array([7.0, 6.5, 8.0], dtype=float), "W"),
        (np.array([30.0, 32.0, 28.0], dtype=float), "M"),
        (np.array([90.0, 100.0, 80.0], dtype=float), "Q"),
        (np.array([150.0, 365.0], dtype=float), "Y"),
    ],
)
def test_classify_from_diffs_returns_expected_bucket(
    diffs: np.ndarray, expected: str
) -> None:
    assert freq._classify_from_diffs(diffs) == expected


def test_classify_from_diffs_detects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="Unable to determine"):
        freq._classify_from_diffs(np.array([-1.0, -0.5], dtype=float))


def test_classify_from_diffs_requires_consensus() -> None:
    # Mixture of daily, monthly, and yearly gaps without a dominant bucket.
    diffs = np.array([2.0, 10.0, 140.0], dtype=float)
    with pytest.raises(ValueError, match="too irregular"):
        freq._classify_from_diffs(diffs)


@pytest.mark.parametrize(
    ("target_code", "resampled"),
    [("M", False), ("D", True), ("W", True), ("Q", True), ("Y", True)],
)
def test_summary_from_code_reflects_resampling(
    target_code: freq.FrequencyCode, resampled: bool
) -> None:
    summary = freq._summary_from_code(target_code)

    assert summary.code == target_code
    assert summary.label == freq.FREQUENCY_LABELS[target_code]
    assert summary.target == "M"
    assert summary.resampled is resampled
    if resampled:
        assert summary.target_label == freq.FREQUENCY_LABELS["M"]
    else:
        assert summary.target_label == summary.label


def test_detect_frequency_uses_inferred_code_directly() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")

    summary = freq.detect_frequency(idx)

    assert summary == freq.FrequencySummary(
        code="D",
        label=freq.FREQUENCY_LABELS["D"],
        resampled=True,
        target="M",
        target_label=freq.FREQUENCY_LABELS["M"],
    )


def test_detect_frequency_defaults_to_monthly_for_single_entry() -> None:
    summary = freq.detect_frequency([pd.Timestamp("2024-01-31")])

    assert summary.code == "M"
    assert summary.resampled is False
    assert summary.target_label == freq.FREQUENCY_LABELS["M"]


def test_detect_frequency_falls_back_when_infer_freq_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="7D")

    def _raise(_idx: pd.DatetimeIndex) -> None:
        raise ValueError("no frequency")

    monkeypatch.setattr(freq.pd, "infer_freq", _raise)

    summary = freq.detect_frequency(idx)

    assert summary.code == "W"
    assert summary.label == freq.FREQUENCY_LABELS["W"]


def test_detect_frequency_sorts_and_deduplicates_input() -> None:
    idx = [
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-15"),
    ]

    summary = freq.detect_frequency(idx)

    assert summary.code == "W"


def test_detect_frequency_handles_mapping_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="3D")

    monkeypatch.setattr(freq, "_map_inferred", lambda _freq: None)

    summary = freq.detect_frequency(idx)

    assert summary.code == "D"


def test_detect_frequency_propagates_classification_errors() -> None:
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-20"])

    with pytest.raises(ValueError, match="irregular"):
        freq.detect_frequency(idx)


def test_detect_frequency_requires_datetime_like_inputs() -> None:
    with pytest.raises(ValueError, match="datetime-like"):
        freq.detect_frequency(["2024-01-01", "not-a-date", object()])


def test_intervals_in_days_matches_expected_delta() -> None:
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])

    diffs = freq._intervals_in_days(idx)

    assert np.allclose(diffs, np.array([1.0, 3.0], dtype=float))


@pytest.mark.parametrize(
    "iterable_type",
    [list, tuple, pd.Series],
)
def test_detect_frequency_accepts_various_iterables(iterable_type: type) -> None:
    values = iterable_type(pd.date_range("2024-01-01", periods=5, freq="M"))

    summary = freq.detect_frequency(values)

    assert summary.code == "M"
    assert summary.resampled is False
