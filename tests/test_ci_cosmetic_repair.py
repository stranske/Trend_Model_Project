from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Testing an element's truth value will always return True in future versions.:DeprecationWarning"
)

from scripts import ci_cosmetic_repair


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
    """.strip().format(
        message=escape(message, {'"': "&quot;"})
    )
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
    message = (
        "COSMETIC_TOLERANCE "
        "{"
        '"path": "tests/fixtures/baseline.py", '
        '"guard": "float", '
        '"key": "EXPECTED_ALPHA", '
        '"actual": 1.23456, '
        '"digits": 5}'
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


def test_cosmetic_repair_refuses_without_guard(tmp_path: Path) -> None:
    repo_root = tmp_path
    target = repo_root / "tests" / "fixtures" / "baseline.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("EXPECTED_ALPHA = 1.23\n", encoding="utf-8")
    message = (
        "COSMETIC_TOLERANCE "
        "{"
        '"path": "tests/fixtures/baseline.py", '
        '"guard": "float", '
        '"key": "EXPECTED_ALPHA", '
        '"actual": 1.23456, '
        '"digits": 5}'
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
