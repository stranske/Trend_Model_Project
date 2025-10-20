from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from tools import coverage_guard as cg


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_baseline_applies_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline.json"
    write_json(
        config_path,
        {
            "line": 86.5,
            "warn_drop": 0.75,
            "recovery_days": "7",
        },
    )

    baseline = cg.load_baseline(config_path)

    assert baseline.baseline == pytest.approx(86.5)
    assert baseline.warn_drop == pytest.approx(0.75)
    assert baseline.recovery_days == 7


def test_load_baseline_enforces_minimum_recovery_days(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline.json"
    write_json(
        config_path,
        {
            "line": 85.0,
            "warn_drop": 1.0,
            "recovery_days": 0,
        },
    )

    baseline = cg.load_baseline(config_path)

    assert baseline.recovery_days == 3


def test_compute_top_files_prioritises_missing_lines() -> None:
    coverage = {
        "files": {
            "src/a.py": {
                "summary": {
                    "percent_covered": 50.0,
                    "covered_lines": 5,
                    "missing_lines": 5,
                    "num_statements": 10,
                }
            },
            "src/b.py": {
                "summary": {
                    "percent_covered": 70.0,
                    "covered_lines": 7,
                    "missing_lines": 3,
                    "num_statements": 10,
                }
            },
            "src/c.py": {
                "summary": {
                    "percent_covered": 100.0,
                    "covered_lines": 10,
                    "missing_lines": 0,
                    "num_statements": 10,
                }
            },
        }
    }

    top = cg.compute_top_files(coverage, limit=2)

    assert [item.path for item in top] == ["src/a.py", "src/b.py"]
    assert all(item.missing > 0 for item in top)


def test_compute_top_files_falls_back_to_total_lines() -> None:
    coverage = {
        "files": {
            "src/a.py": {
                "summary": {
                    "percent_covered": 100.0,
                    "covered_lines": 40,
                    "missing_lines": 0,
                    "num_statements": 40,
                }
            },
            "src/b.py": {
                "summary": {
                    "percent_covered": 100.0,
                    "covered_lines": 10,
                    "missing_lines": 0,
                    "num_statements": 10,
                }
            },
        }
    }

    top = cg.compute_top_files(coverage, limit=2)

    assert [item.path for item in top] == ["src/a.py", "src/b.py"]


def test_build_update_comment_formats_metrics() -> None:
    snapshot = cg.CoverageSnapshot(current=82.3, baseline=85.0, delta=-2.7)
    config = cg.BaselineConfig(baseline=85.0, warn_drop=1.0, recovery_days=3)
    today = dt.date(2024, 12, 31)
    files = [
        cg.FileCoverage(path="src/a.py", percent=60.0, covered=6, total=10, missing=4),
        cg.FileCoverage(path="src/b.py", percent=75.0, covered=15, total=20, missing=5),
    ]

    comment = cg.build_update_comment(
        snapshot,
        config,
        below_baseline=True,
        date=today,
        run_url="https://example.invalid/run/1",
        recovery_progress=None,
        top_files=files,
    )

    assert "2024-12-31" in comment
    assert "Current coverage: 82.30%" in comment
    assert "Baseline coverage: 85.00%" in comment
    assert "Delta vs baseline: -2.70 pts" in comment
    assert "Top changed files" in comment
    assert "src/a.py" in comment

