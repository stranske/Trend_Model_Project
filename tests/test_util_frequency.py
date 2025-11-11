from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.util import frequency


def test_as_datetime_index_accepts_iterables() -> None:
    idx = frequency._as_datetime_index(
        ["2024-03-01", "2024-01-01", "2024-02-01"],
    )
    assert isinstance(idx, pd.DatetimeIndex)
    assert list(idx) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]


def test_as_datetime_index_retries_after_initial_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakyDatetimeIndex(pd.DatetimeIndex):
        calls = 0

        def __new__(cls, data=None, *args, **kwargs):
            cls.calls += 1
            if cls.calls == 1:
                raise TypeError("initial failure")
            return super().__new__(cls, data, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(frequency.pd, "DatetimeIndex", FlakyDatetimeIndex)
        idx = frequency._as_datetime_index(["2024-01-01", "2024-01-02"])

    assert FlakyDatetimeIndex.calls == 2
    assert isinstance(idx, pd.DatetimeIndex)


def test_as_datetime_index_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="datetime-like inputs"):
        frequency._as_datetime_index(["2024-01-01", "not-a-date"])


@pytest.mark.parametrize(
    "freq,expected",
    [
        (None, None),
        ("W-MON", "W"),
        ("B", "D"),
        ("M", "M"),
        ("SM", "M"),
        ("Q-DEC", "Q"),
        ("A", "Y"),
        ("Y", "Y"),
        ("UNKNOWN", None),
    ],
)
def test_map_inferred_frequency(freq: str | None, expected: str | None) -> None:
    assert frequency._map_inferred(freq) == expected


@pytest.mark.parametrize(
    "diffs,expected",
    [
        (np.array([], dtype=float), "M"),
        (np.array([1.0, 2.0, 3.5]), "D"),
        (np.array([7.0, 6.0, 8.5]), "W"),
        (np.array([30.0, 29.0, 31.0]), "M"),
        (np.array([92.0, 95.0, 87.0]), "Q"),
        (np.array([140.0, 365.0, 200.0]), "Y"),
    ],
)
def test_classify_from_diffs_returns_expected_code(diffs: np.ndarray, expected: str) -> None:
    assert frequency._classify_from_diffs(diffs) == expected


def test_classify_from_diffs_detects_irregular_series() -> None:
    diffs = np.array([5.0, 35.0])
    with pytest.raises(ValueError, match="too irregular"):
        frequency._classify_from_diffs(diffs)


def test_classify_from_diffs_raises_when_no_bucket() -> None:
    diffs = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="Unable to determine"):
        frequency._classify_from_diffs(diffs)


def test_intervals_in_days_converts_nanoseconds() -> None:
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04"])
    diffs = frequency._intervals_in_days(idx)
    assert np.allclose(diffs, np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "series,expected_code,resampled",
    [
        (pd.date_range("2024-01-01", periods=3, freq="D"), "D", True),
        (pd.date_range("2024-01-01", periods=6, freq="W-MON"), "W", True),
        (pd.date_range("2024-01-01", periods=4, freq="MS"), "M", False),
        (pd.date_range("2024-01-01", periods=4, freq="QS"), "Q", True),
        (pd.date_range("2020-01-01", periods=3, freq="YS"), "Y", True),
    ],
)
def test_detect_frequency_handles_regular_series(
    series: pd.DatetimeIndex,
    expected_code: str,
    resampled: bool,
) -> None:
    summary = frequency.detect_frequency(series)
    assert summary.code == expected_code
    assert summary.resampled is resampled
    if resampled:
        assert summary.target == "M"
        assert summary.target_label == frequency.FREQUENCY_LABELS["M"]
    else:
        assert summary.target_label == summary.label


def test_detect_frequency_handles_short_series() -> None:
    summary = frequency.detect_frequency([pd.Timestamp("2024-01-01")])
    assert summary.code == "M"
    assert summary.resampled is False


def test_detect_frequency_handles_irregular_spacing() -> None:
    dates = ["2024-01-01", "2024-01-06", "2024-02-15"]
    with pytest.raises(ValueError, match="irregular"):
        frequency.detect_frequency(dates)


def test_detect_frequency_retries_when_infer_freq_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    series = pd.date_range("2024-01-01", periods=4, freq="MS")

    def boom(_: object) -> None:
        raise ValueError("boom")

    with monkeypatch.context() as patcher:
        patcher.setattr(frequency.pd, "infer_freq", boom)
        summary = frequency.detect_frequency(series)

    assert summary.code == "M"


def test_detect_frequency_falls_back_when_infer_freq_none(monkeypatch: pytest.MonkeyPatch) -> None:
    series = pd.date_range("2024-01-01", periods=4, freq="W-WED")

    with monkeypatch.context() as patcher:
        patcher.setattr(pd, "infer_freq", lambda _: None)
        summary = frequency.detect_frequency(series)

    assert summary.code == "W"


def test_detect_frequency_drops_duplicates_and_sorts() -> None:
    series = ["2024-03-01", "2024-01-01", "2024-02-01", "2024-01-01"]
    summary = frequency.detect_frequency(series)
    assert summary.code == "M"
    assert summary.label == frequency.FREQUENCY_LABELS["M"]
