"""Tests for coverage_trend module.

Tests the utilities for computing coverage trend information for CI workflows.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools import coverage_trend


def test_load_json_returns_empty_dict_on_missing_file(tmp_path: Path) -> None:
    """Test _load_json returns empty dict when file doesn't exist."""
    result = coverage_trend._load_json(tmp_path / "missing.json")
    assert result == {}


def test_load_json_returns_empty_dict_on_invalid_json(tmp_path: Path) -> None:
    """Test _load_json returns empty dict on invalid JSON."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json {", encoding="utf-8")
    result = coverage_trend._load_json(bad_file)
    assert result == {}


def test_load_json_returns_data_from_valid_file(tmp_path: Path) -> None:
    """Test _load_json returns data from valid JSON file."""
    good_file = tmp_path / "good.json"
    good_file.write_text('{"key": "value"}', encoding="utf-8")
    result = coverage_trend._load_json(good_file)
    assert result == {"key": "value"}


def test_extract_coverage_percent_from_totals() -> None:
    """Test _extract_coverage_percent extracts from totals."""
    data = {"totals": {"percent_covered": 85.5}}
    result = coverage_trend._extract_coverage_percent(data)
    assert result == 85.5


def test_extract_coverage_percent_returns_zero_when_missing() -> None:
    """Test _extract_coverage_percent returns 0 when totals missing."""
    result = coverage_trend._extract_coverage_percent({})
    assert result == 0.0


def test_get_hotspots_returns_sorted_files() -> None:
    """Test _get_hotspots returns files sorted by coverage ascending."""
    data = {
        "files": {
            "high.py": {
                "summary": {
                    "percent_covered": 90.0,
                    "missing_lines": 5,
                    "covered_lines": 45,
                }
            },
            "low.py": {
                "summary": {
                    "percent_covered": 30.0,
                    "missing_lines": 35,
                    "covered_lines": 15,
                }
            },
            "mid.py": {
                "summary": {
                    "percent_covered": 60.0,
                    "missing_lines": 20,
                    "covered_lines": 30,
                }
            },
        }
    }
    hotspots, low = coverage_trend._get_hotspots(data, limit=10, low_threshold=50.0)
    assert len(hotspots) == 3
    assert hotspots[0]["file"] == "low.py"
    assert hotspots[1]["file"] == "mid.py"
    assert hotspots[2]["file"] == "high.py"
    assert len(low) == 1
    assert low[0]["file"] == "low.py"


def test_get_hotspots_limits_results() -> None:
    """Test _get_hotspots respects limit parameter."""
    data = {
        "files": {
            "a.py": {
                "summary": {
                    "percent_covered": 10.0,
                    "missing_lines": 9,
                    "covered_lines": 1,
                }
            },
            "b.py": {
                "summary": {
                    "percent_covered": 20.0,
                    "missing_lines": 8,
                    "covered_lines": 2,
                }
            },
            "c.py": {
                "summary": {
                    "percent_covered": 30.0,
                    "missing_lines": 7,
                    "covered_lines": 3,
                }
            },
        }
    }
    hotspots, _ = coverage_trend._get_hotspots(data, limit=2)
    assert len(hotspots) == 2


def test_format_hotspot_table_returns_empty_for_no_files() -> None:
    """Test _format_hotspot_table returns empty string for empty list."""
    result = coverage_trend._format_hotspot_table([], "Test Title")
    assert result == ""


def test_format_hotspot_table_creates_markdown() -> None:
    """Test _format_hotspot_table creates proper markdown table."""
    files = [
        {"file": "low.py", "coverage": 30.5, "missing_lines": 35},
        {"file": "mid.py", "coverage": 60.0, "missing_lines": 20},
    ]
    result = coverage_trend._format_hotspot_table(files, "Coverage Hotspots")
    assert "### Coverage Hotspots" in result
    assert "| File | Coverage | Missing |" in result
    assert "`low.py`" in result
    assert "30.5%" in result


def test_main_with_coverage_json(tmp_path: Path) -> None:
    """Test main function processes coverage.json correctly."""
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps(
            {
                "totals": {"percent_covered": 85.0},
                "files": {
                    "src/app.py": {
                        "summary": {
                            "percent_covered": 90.0,
                            "missing_lines": 5,
                            "covered_lines": 45,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = tmp_path / "artifact.json"
    summary = tmp_path / "summary.md"

    exit_code = coverage_trend.main(
        [
            "--coverage-json",
            str(coverage_json),
            "--artifact-path",
            str(artifact),
            "--summary-path",
            str(summary),
            "--minimum",
            "80",
        ]
    )

    assert exit_code == 0
    assert artifact.exists()
    artifact_data = json.loads(artifact.read_text())
    assert artifact_data["current"] == 85.0
    assert artifact_data["passes_minimum"] is True


def test_main_fails_below_minimum(tmp_path: Path) -> None:
    """Test main returns non-zero when coverage below minimum."""
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps({"totals": {"percent_covered": 50.0}, "files": {}}),
        encoding="utf-8",
    )

    exit_code = coverage_trend.main(
        [
            "--coverage-json",
            str(coverage_json),
            "--minimum",
            "80",
        ]
    )

    assert exit_code == 1


def test_main_soft_mode_always_passes(tmp_path: Path) -> None:
    """Test main with --soft flag returns 0 even below minimum."""
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps({"totals": {"percent_covered": 50.0}, "files": {}}),
        encoding="utf-8",
    )

    exit_code = coverage_trend.main(
        [
            "--coverage-json",
            str(coverage_json),
            "--minimum",
            "80",
            "--soft",
        ]
    )

    assert exit_code == 0
