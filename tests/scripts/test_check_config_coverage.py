"""Tests for config coverage validity thresholds."""

from __future__ import annotations

import pytest

from scripts import check_config_coverage
from trend_analysis.config.coverage import ConfigCoverageReport, compute_schema_validity


def test_compute_schema_validity_uses_overlap_ratio() -> None:
    report = ConfigCoverageReport(
        read={"data.csv_path", "data.frequency"},
        validated={"data.frequency", "portfolio.max_turnover"},
        ignored=set(),
    )

    assert compute_schema_validity(report) == pytest.approx(1 / 3)


def test_compute_schema_validity_defaults_to_full_when_empty() -> None:
    report = ConfigCoverageReport(read=set(), validated=set(), ignored=set())
    assert compute_schema_validity(report) == 1.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.95, 0.95),
        (95.0, 0.95),
        (0.0, 0.0),
        (-1.0, 0.0),
    ],
)
def test_normalize_threshold_accepts_ratio_or_percent(raw: float, expected: float) -> None:
    assert check_config_coverage._normalize_threshold(raw) == pytest.approx(expected)
