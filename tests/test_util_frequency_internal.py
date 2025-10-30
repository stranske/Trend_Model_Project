"""Additional coverage for `trend_analysis.util.frequency` helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import trend_analysis.util.frequency as frequency

from trend_analysis.util.frequency import (
    FREQUENCY_LABELS,
    FrequencySummary,
    _as_datetime_index,
    _classify_from_diffs,
    _map_inferred,
    _summary_from_code,
    detect_frequency,
)


def test_as_datetime_index_normalises_iterables() -> None:
    """Non-index iterables should be converted and sorted chronologically."""

    idx = _as_datetime_index(
        [
            "2024-01-03",
            "2024-01-01",
            pd.Timestamp("2024-01-02"),
        ]
    )

    assert isinstance(idx, pd.DatetimeIndex)
    assert list(idx) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]


def test_as_datetime_index_sorts_existing_index() -> None:
    """DatetimeIndex inputs should be returned sorted without modification."""

    raw = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
    idx = _as_datetime_index(raw)
    assert idx.tolist() == sorted(raw.tolist())


def test_as_datetime_index_rejects_non_datetimes() -> None:
    """Invalid inputs should trigger a ValueError rather than NaT leakage."""

    with pytest.raises(ValueError):
        _as_datetime_index(["2024-01-01", "not-a-date", 10])


def test_as_datetime_index_recovers_from_initial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the first DatetimeIndex attempt fails we should coerce and retry."""

    calls = {"count": 0}

    class FlakyDatetimeIndex(pd.DatetimeIndex):
        def __new__(cls, *args: object, **kwargs: object) -> "FlakyDatetimeIndex":
            calls["count"] += 1
            if calls["count"] == 1:
                raise TypeError("cannot construct directly")
            return super().__new__(cls, *args, **kwargs)

    monkeypatch.setattr(frequency.pd, "DatetimeIndex", FlakyDatetimeIndex)

    idx = _as_datetime_index(["2024-01-01", "2024-01-02", "2024-01-03"])

    assert calls["count"] == 2
    assert list(idx) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        (None, None),
        ("", None),
        ("W-WED", "W"),
        ("bd", "D"),
        ("C", "D"),
        ("15D", "D"),
        ("SM-15", "M"),
        ("BM", "M"),
        ("Q-DEC", "Q"),
        ("A-DEC", "Y"),
        ("foo", None),
    ],
)
def test_map_inferred_normalises_pandas_codes(
    freq: str | None, expected: str | None
) -> None:
    """The inference mapper should collapse Pandas frequency aliases to our codes."""

    assert _map_inferred(freq) == expected


def test_classify_from_diffs_defaults_to_monthly_when_empty() -> None:
    """Empty diffs should return the default monthly cadence."""

    assert _classify_from_diffs(np.array([])) == "M"


def test_classify_from_diffs_requires_signal() -> None:
    """Classifying intervals with no recognised cadence should error."""

    with pytest.raises(ValueError, match="irregular spacing"):
        _classify_from_diffs(np.array([0.0]))


def test_classify_from_diffs_detects_inconsistent_cadence() -> None:
    """Mixed cadences without a dominant bucket should raise an error."""

    with pytest.raises(ValueError, match="too irregular"):
        _classify_from_diffs(np.array([6.0, 250.0]))


def test_summary_from_code_tracks_resampling_targets() -> None:
    """The summary helper should describe both the detected cadence and target."""

    summary = _summary_from_code("D")
    assert summary == FrequencySummary(
        code="D",
        label=FREQUENCY_LABELS["D"],
        resampled=True,
        target="M",
        target_label=FREQUENCY_LABELS["M"],
    )


def test_summary_from_code_for_monthly() -> None:
    """Monthly series should not be flagged for resampling."""

    summary = _summary_from_code("M")
    assert summary.resampled is False
    assert summary.target_label == FREQUENCY_LABELS["M"]


def test_detect_frequency_handles_single_observation() -> None:
    """Single point series default to monthly without calling infer_freq."""

    summary = detect_frequency([pd.Timestamp("2024-01-31")])
    assert summary.code == "M"
    assert summary.resampled is False


def test_detect_frequency_uses_inferred_business_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Business-day inference should collapse to the daily cadence."""

    idx = pd.date_range("2024-01-01", periods=5, freq="B")

    def fake_infer_freq(_: pd.DatetimeIndex) -> str:
        return "bd"

    monkeypatch.setattr(pd, "infer_freq", fake_infer_freq)

    summary = detect_frequency(idx)
    assert summary.code == "D"
    assert summary.label == FREQUENCY_LABELS["D"]
    assert summary.target_label == FREQUENCY_LABELS["M"]


def test_detect_frequency_falls_back_when_infer_freq_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors from ``pd.infer_freq`` should trigger the diff-based classifier."""

    idx = pd.DatetimeIndex(
        [
            "2024-03-31",
            "2024-03-31",  # duplicate should be ignored
            "2024-06-30",
            "2024-09-30",
            "2024-12-31",
        ]
    )

    def raise_error(_: pd.DatetimeIndex) -> str:
        raise ValueError("no freq")

    monkeypatch.setattr(pd, "infer_freq", raise_error)

    summary = detect_frequency(idx)
    assert summary.code == "Q"
    assert summary.label == FREQUENCY_LABELS["Q"]
