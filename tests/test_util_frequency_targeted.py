"""Targeted coverage for trend_analysis.util.frequency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.util import frequency as freq


def test_as_datetime_index_sorts_and_coerces_strings() -> None:
    unordered = ["2024-03-31", "2024-01-31", "2024-02-29"]

    idx = freq._as_datetime_index(unordered)

    assert list(idx) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-03-31"),
    ]


def test_as_datetime_index_falls_back_to_to_datetime() -> None:
    class WeirdSequence:
        def __iter__(self):
            return iter(["2024-01-31", "2024-02-29"])

        def __array__(self, dtype=None):  # pragma: no cover - interface requirement
            raise TypeError

    idx = freq._as_datetime_index(WeirdSequence())

    assert list(idx) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
    ]


@pytest.mark.parametrize("values", [["2024-01-01", "not-a-date"], ["foo", "bar"]])
def test_as_datetime_index_rejects_invalid_values(values: list[str]) -> None:
    with pytest.raises(ValueError):
        freq._as_datetime_index(values)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("w-fri", "W"),
        ("BD", "D"),
        ("sm", "M"),
        ("q-dec", "Q"),
        ("a-dec", "Y"),
        (None, None),
        ("???", None),
    ],
)
def test_map_inferred_handles_common_aliases(
    candidate: str | None, expected: str | None
) -> None:
    assert freq._map_inferred(candidate) == expected


def test_classify_from_diffs_requires_signal() -> None:
    with pytest.raises(ValueError):
        freq._classify_from_diffs(np.array([0.0]))


def test_classify_from_diffs_requires_majority_bucket() -> None:
    with pytest.raises(ValueError):
        freq._classify_from_diffs(np.array([2.5, 50.0]))


def test_classify_from_diffs_prefers_dominant_bucket() -> None:
    assert freq._classify_from_diffs(np.array([1.0, 1.5, 1.2, 2.0])) == "D"


def test_classify_from_diffs_defaults_to_monthly_when_empty() -> None:
    assert freq._classify_from_diffs(np.array([], dtype=np.float64)) == "M"


def test_summary_from_code_marks_resampling() -> None:
    summary = freq._summary_from_code("D")

    assert summary.code == "D"
    assert summary.label == "Daily"
    assert summary.resampled is True
    assert summary.target == "M"
    assert summary.target_label == "Monthly"


def test_detect_frequency_single_entry_defaults_to_monthly() -> None:
    summary = freq.detect_frequency([pd.Timestamp("2023-08-31")])

    assert summary.code == "M"
    assert summary.resampled is False


def test_detect_frequency_falls_back_when_infer_freq_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-31", periods=6, freq="M")

    monkeypatch.setattr(
        pd, "infer_freq", lambda _: (_ for _ in ()).throw(ValueError("boom"))
    )

    summary = freq.detect_frequency(idx)

    assert summary.code == "M"


def test_detect_frequency_classifies_weekly_series_without_infer_freq() -> None:
    idx = pd.DatetimeIndex(
        [
            "2024-01-05",
            "2024-01-12",
            "2024-01-19",
            "2024-01-26",
            "2024-02-02",
        ]
    )

    summary = freq.detect_frequency(idx[::-1])

    assert summary.code == "W"
    assert summary.label == "Weekly"
    assert summary.resampled is True
    assert summary.target == "M"


def test_detect_frequency_flags_irregular_spacing() -> None:
    idx = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-01-05",
            "2024-02-20",
        ]
    )

    with pytest.raises(ValueError):
        freq.detect_frequency(idx)
