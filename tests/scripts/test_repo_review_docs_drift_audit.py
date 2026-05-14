from __future__ import annotations

from pathlib import Path

from scripts import repo_review_docs_drift_audit as audit


def test_validate_audit_report_passes_for_valid_content(tmp_path: Path) -> None:
    report = tmp_path / "docs-impl-drift-audit.md"
    report.write_text(
        """
        status endpoint checklist
        docs/phase-3/MonteCarlo.md
        docs/api.md
        docs/issues/raise_test_coverage_to_89.md
        https://github.com/stranske/Trend_Model_Project/issues/5296
        """,
        encoding="utf-8",
    )

    assert audit._validate_audit_report(report) == []


def test_validate_audit_report_fails_when_claim_tokens_missing(tmp_path: Path) -> None:
    report = tmp_path / "docs-impl-drift-audit.md"
    report.write_text(
        "docs/phase-3/MonteCarlo.md\ndocs/api.md\ndocs/issues/raise_test_coverage_to_89.md\n",
        encoding="utf-8",
    )

    errors = audit._validate_audit_report(report)
    assert any("Missing required claim token" in item for item in errors)


def test_validate_audit_report_fails_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    errors = audit._validate_audit_report(missing)
    assert errors == [f"Audit artifact is missing: {missing}"]


def test_main_returns_nonzero_when_follow_up_issue_is_missing(tmp_path: Path) -> None:
    report = tmp_path / "docs-impl-drift-audit.md"
    report.write_text(
        """
        status endpoint checklist
        docs/phase-3/MonteCarlo.md
        docs/api.md
        docs/issues/raise_test_coverage_to_89.md
        """,
        encoding="utf-8",
    )

    assert audit.main(["--audit-path", str(report)]) == 1
