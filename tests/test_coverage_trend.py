"""Tests for coverage_trend module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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
            "high.py": {"summary": {"percent_covered": 90.0, "missing_lines": 5, "covered_lines": 45}},
            "low.py": {"summary": {"percent_covered": 20.0, "missing_lines": 40, "covered_lines": 10}},
            "mid.py": {"summary": {"percent_covered": 50.0, "missing_lines": 25, "covered_lines": 25}},
        }
    }
    hotspots, low_cov = coverage_trend._get_hotspots(data, limit=10, low_threshold=50.0)
    
    # Should be sorted by coverage ascending
    assert hotspots[0]["file"] == "low.py"
    assert hotspots[1]["file"] == "mid.py"
    assert hotspots[2]["file"] == "high.py"
    
    # Low coverage should include files below threshold
    assert len(low_cov) == 1
    assert low_cov[0]["file"] == "low.py"


def test_format_hotspot_table_returns_empty_on_no_files() -> None:
    """Test _format_hotspot_table returns empty string when no files."""
    result = coverage_trend._format_hotspot_table([], "Test")
    assert result == ""


def test_format_hotspot_table_creates_markdown_table() -> None:
    """Test _format_hotspot_table creates proper markdown."""
    files = [{"file": "test.py", "coverage": 75.5, "missing_lines": 10}]
    result = coverage_trend._format_hotspot_table(files, "Test Hotspots")
    
    assert "### Test Hotspots" in result
    assert "| File | Coverage | Missing |" in result
    assert "`test.py`" in result
    assert "75.5%" in result


def test_main_generates_summary_markdown(tmp_path: Path) -> None:
    """Test main() generates summary markdown file."""
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps({"totals": {"percent_covered": 85.0}, "files": {}}),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.md"
    
    result = coverage_trend.main([
        "--coverage-json", str(coverage_json),
        "--summary-path", str(summary_path),
        "--minimum", "70",
    ])
    
    assert result == 0
    summary = summary_path.read_text(encoding="utf-8")
    assert "85.00%" in summary
    assert "✅ Pass" in summary


def test_main_returns_failure_below_minimum(tmp_path: Path) -> None:
    """Test main() returns 1 when coverage below minimum."""
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps({"totals": {"percent_covered": 50.0}, "files": {}}),
        encoding="utf-8",
    )
    
    result = coverage_trend.main([
        "--coverage-json", str(coverage_json),
        "--minimum", "70",
    ])
    
    assert result == 1


def test_main_soft_mode_always_passes(tmp_path: Path) -> None:
    """Test main() with --soft always returns 0 even below minimum."""
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps({"totals": {"percent_covered": 50.0}, "files": {}}),
        encoding="utf-8",
    )
    
    result = coverage_trend.main([
        "--coverage-json", str(coverage_json),
        "--minimum", "70",
        "--soft",
    ])
    
    assert result == 0
