import numpy as np
import pandas as pd
import pytest

from trend_analysis.util import frequency as freq_mod
from trend_analysis.util.frequency import FREQUENCY_LABELS, detect_frequency


@pytest.mark.parametrize(
    "values, expected_code, expected_resample",
    [
        (
            ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-05"],
            "D",
            True,
        ),
        (
            pd.date_range("2024-01-01", periods=12, freq="M"),
            "M",
            False,
        ),
    ],
)
def test_detect_frequency_from_various_iterables(values, expected_code, expected_resample):
    summary = detect_frequency(values)
    assert summary.code == expected_code
    assert summary.label == FREQUENCY_LABELS[expected_code]
    assert summary.resampled is expected_resample
    if expected_resample:
        assert summary.target == "M"
        assert summary.target_label == FREQUENCY_LABELS["M"]
    else:
        assert summary.target == expected_code
        assert summary.target_label == FREQUENCY_LABELS[expected_code]


def test_detect_frequency_falls_back_to_diffs_when_infer_freq_not_available():
    # Construct an index whose cadence fluctuates between weekly-ish spacings
    idx = pd.DatetimeIndex([
        "2024-01-01",
        "2024-01-06",
        "2024-01-12",
        "2024-01-19",
    ])
    summary = detect_frequency(idx)
    assert summary.code == "W"
    assert summary.resampled is True


def test_detect_frequency_errors_on_irregular_spacing():
    idx = pd.DatetimeIndex([
        "2024-01-01",
        "2024-01-02",
        "2024-01-20",
        "2024-09-01",
    ])
    with pytest.raises(ValueError):
        detect_frequency(idx)


def test_detect_frequency_requires_datetime_like_values():
    with pytest.raises(ValueError):
        detect_frequency(["2024-01-01", "not-a-date", object()])


def test_detect_frequency_single_value_defaults_to_monthly():
    summary = detect_frequency(["2024-04-01"])
    assert summary.code == "M"
    assert summary.resampled is False
    assert summary.target == "M"
    assert summary.target_label == FREQUENCY_LABELS["M"]


@pytest.mark.parametrize(
    "freq_code, expected",
    [
        ("W-FRI", "W"),
        ("B", "D"),
        ("SM", "M"),
        ("Q", "Q"),
        ("A", "Y"),
    ],
)
def test_detect_frequency_maps_inferred_codes(freq_code, expected):
    idx = pd.date_range("2020-01-01", periods=6, freq=freq_code)
    summary = detect_frequency(idx)
    assert summary.code == expected


def test_as_datetime_index_recovers_from_initial_failure(monkeypatch):
    original = freq_mod.pd.DatetimeIndex

    class FlakyDatetimeIndex(original):
        calls = 0

        def __new__(cls, data=None, **kwargs):
            if cls.calls == 0:
                cls.calls += 1
                raise TypeError("temporary failure")
            return original.__new__(original, data, **kwargs)

    monkeypatch.setattr(freq_mod.pd, "DatetimeIndex", FlakyDatetimeIndex)
    idx = freq_mod._as_datetime_index(["2024/01/01", "2024/01/08"])
    assert isinstance(idx, original)
    assert FlakyDatetimeIndex.calls == 1


def test_detect_frequency_handles_infer_freq_error(monkeypatch):

    def raise_error(idx):
        raise ValueError("boom")

    monkeypatch.setattr(freq_mod.pd, "infer_freq", raise_error)
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    summary = detect_frequency(idx)
    assert summary.code == "D"


def test_classify_from_diffs_handles_empty_and_invalid_sequences():
    assert freq_mod._classify_from_diffs(np.array([], dtype=float)) == "M"
    with pytest.raises(ValueError, match="Unable to determine"):
        freq_mod._classify_from_diffs(np.array([-5.0], dtype=float))


def test_classify_from_diffs_detects_irregularity():
    diffs = np.array([1.0, 50.0, 200.0], dtype=float)
    with pytest.raises(ValueError, match="too irregular"):
        freq_mod._classify_from_diffs(diffs)
