from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Testing an element's truth value will always return True in future versions.:DeprecationWarning"
)

from scripts import ci_cosmetic_repair


def _read_summary(repo_root: Path) -> dict[str, object]:
    summary_path = repo_root / ci_cosmetic_repair.SUMMARY_FILE
    assert summary_path.exists(), "summary file should be created"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_junit(tmp_path: Path, message: str) -> Path:
    junit = """
        <testsuite tests=\"1\" failures=\"1\">
          <testcase classname=\"tests.test_sample\" name=\"test_cosmetic\">
            <properties>
              <property name=\"markers\" value=\"cosmetic\" />
            </properties>
            <failure message=\"{message}\">
              <details>expected 1.23 got 1.22</details>
            </failure>
          </testcase>
        </testsuite>
    """.strip().format(message=escape(message, {'"': "&quot;"}))
    path = tmp_path / "report.xml"
    path.write_text(junit, encoding="utf-8")
    return path


def test_cosmetic_repair_updates_guarded_value(tmp_path: Path) -> None:
    repo_root = tmp_path
    target = repo_root / "tests" / "fixtures" / "baseline.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "EXPECTED_ALPHA = 1.23450  # cosmetic-repair: float EXPECTED_ALPHA\n",
        encoding="utf-8",
    )
    message = "COSMETIC_TOLERANCE " + json.dumps(
        {
            "path": "tests/fixtures/baseline.py",
            "guard": "float",
            "key": "EXPECTED_ALPHA",
            "actual": 1.23456,
            "digits": 5,
        }
    )
    report = _write_junit(repo_root, message)

    exit_code = ci_cosmetic_repair.main(
        [
            "--apply",
            "--report",
            str(report),
            "--root",
            str(repo_root),
            "--skip-pr",
        ]
    )

    assert exit_code == 0
    updated = target.read_text(encoding="utf-8")
    assert "1.23456" in updated
    assert updated.endswith("\n")
    summary = _read_summary(repo_root)
    assert summary["status"] == "applied-no-pr"
    assert summary["mode"] == "apply"
    assert summary.get("changed_files") == ["tests/fixtures/baseline.py"]
    instructions = summary.get("instructions")
    assert isinstance(instructions, list) and instructions


def test_cosmetic_repair_refuses_without_guard(tmp_path: Path) -> None:
    repo_root = tmp_path
    target = repo_root / "tests" / "fixtures" / "baseline.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("EXPECTED_ALPHA = 1.23\n", encoding="utf-8")
    message = "COSMETIC_TOLERANCE " + json.dumps(
        {
            "path": "tests/fixtures/baseline.py",
            "guard": "float",
            "key": "EXPECTED_ALPHA",
            "actual": 1.23456,
            "digits": 5,
        }
    )
    report = _write_junit(repo_root, message)

    with pytest.raises(ci_cosmetic_repair.CosmeticRepairError):
        ci_cosmetic_repair.main(
            [
                "--apply",
                "--report",
                str(report),
                "--root",
                str(repo_root),
                "--skip-pr",
            ]
        )


def test_second_run_detects_no_changes(tmp_path: Path) -> None:
    repo_root = tmp_path
    target = repo_root / "tests" / "fixtures" / "baseline.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "EXPECTED_ALPHA = 1.23450  # cosmetic-repair: float EXPECTED_ALPHA\n",
        encoding="utf-8",
    )
    message = "COSMETIC_TOLERANCE " + json.dumps(
        {
            "path": "tests/fixtures/baseline.py",
            "guard": "float",
            "key": "EXPECTED_ALPHA",
            "actual": 1.23456,
            "digits": 5,
        }
    )
    report = _write_junit(repo_root, message)

    first_exit = ci_cosmetic_repair.main(
        [
            "--apply",
            "--report",
            str(report),
            "--root",
            str(repo_root),
            "--skip-pr",
        ]
    )
    assert first_exit == 0
    first_summary = _read_summary(repo_root)
    assert first_summary["status"] == "applied-no-pr"

    second_exit = ci_cosmetic_repair.main(
        [
            "--apply",
            "--report",
            str(report),
            "--root",
            str(repo_root),
            "--skip-pr",
        ]
    )
    assert second_exit == 0
    second_summary = _read_summary(repo_root)
    assert second_summary["status"] == "no-changes"


def test_cosmetic_snapshot_updates_file(tmp_path: Path) -> None:
    repo_root = tmp_path
    target = repo_root / "tests" / "fixtures" / "snapshot.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old snapshot\n# cosmetic-repair: snapshot baseline\n", encoding="utf-8")
    replacement = "new snapshot\n# cosmetic-repair: snapshot baseline\n"
    message = "COSMETIC_SNAPSHOT " + json.dumps(
        {
            "path": "tests/fixtures/snapshot.txt",
            "guard": "snapshot",
            "replacement": replacement,
        }
    )
    report = _write_junit(repo_root, message)

    exit_code = ci_cosmetic_repair.main(
        [
            "--apply",
            "--report",
            str(report),
            "--root",
            str(repo_root),
            "--skip-pr",
        ]
    )

    assert exit_code == 0
    assert target.read_text(encoding="utf-8") == replacement
    summary = _read_summary(repo_root)
    assert summary["status"] == "applied-no-pr"
    assert summary.get("changed_files") == ["tests/fixtures/snapshot.txt"]
    assert summary.get("instructions")[0]["kind"] == "snapshot"


def test_dry_run_writes_summary(tmp_path: Path) -> None:
    repo_root = tmp_path
    target = repo_root / "tests" / "fixtures" / "baseline.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "EXPECTED_ALPHA = 1.23450  # cosmetic-repair: float EXPECTED_ALPHA\n",
        encoding="utf-8",
    )
    message = "COSMETIC_TOLERANCE " + json.dumps(
        {
            "path": "tests/fixtures/baseline.py",
            "guard": "float",
            "key": "EXPECTED_ALPHA",
            "actual": 1.23456,
            "digits": 5,
        }
    )
    report = _write_junit(repo_root, message)

    exit_code = ci_cosmetic_repair.main(
        [
            "--dry-run",
            "--report",
            str(report),
            "--root",
            str(repo_root),
        ]
    )

    assert exit_code == 0
    summary = _read_summary(repo_root)
    assert summary["status"] == "dry-run"
    assert summary["mode"] == "dry-run"
    assert summary.get("instructions")
