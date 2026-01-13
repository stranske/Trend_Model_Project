from __future__ import annotations

import pytest

from streamlit_app.components import progress_eta


def test_estimate_eta_seconds_uses_default_when_missing() -> None:
    assert progress_eta.estimate_eta_seconds(None) == progress_eta.DEFAULT_ETA_SECONDS


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (2.0, progress_eta.MIN_ETA_SECONDS),
        (5.0, 5.0),
        (120.0, progress_eta.MAX_ETA_SECONDS),
    ],
)
def test_estimate_eta_seconds_clamps_range(stored: float, expected: float) -> None:
    assert progress_eta.estimate_eta_seconds(stored) == expected


def test_update_eta_seconds_returns_none_for_non_positive_duration() -> None:
    assert progress_eta.update_eta_seconds(10.0, 0.0) is None
    assert progress_eta.update_eta_seconds(10.0, -3.0) is None


def test_update_eta_seconds_blends_previous_value() -> None:
    updated = progress_eta.update_eta_seconds(20.0, 10.0, new_weight=0.4)
    assert updated == pytest.approx(16.0)


def test_update_eta_seconds_sets_first_observation() -> None:
    updated = progress_eta.update_eta_seconds(None, 12.5)
    assert updated == pytest.approx(12.5)


@pytest.mark.parametrize(
    ("elapsed", "estimate", "expected_ratio", "expected_remaining"),
    [
        (5.0, 20.0, 0.25, 15.0),
        (30.0, 20.0, progress_eta.DEFAULT_MAX_RATIO, 0.0),
        (3.0, 0.0, 0.0, 0.0),
    ],
)
def test_progress_ratio_and_remaining(
    elapsed: float,
    estimate: float,
    expected_ratio: float,
    expected_remaining: float,
) -> None:
    ratio, remaining = progress_eta.progress_ratio_and_remaining(elapsed, estimate)
    assert ratio == pytest.approx(expected_ratio)
    assert remaining == pytest.approx(expected_remaining)
