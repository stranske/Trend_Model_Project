"""Validate the repo-review documentation drift audit artifact for issue #5292."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_AUDIT_PATH = Path(
    "docs/reports/repo-review/repos/stranske__Trend_Model_Project/docs-impl-drift-audit.md"
)
REQUIRED_CLAIM_TOKENS = ("status", "endpoint", "checklist")
REQUIRED_DRIFT_DOCS = (
    "docs/phase-3/MonteCarlo.md",
    "docs/api.md",
    "docs/issues/raise_test_coverage_to_89.md",
)


def _validate_audit_report(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Audit artifact is missing: {path}"]

    text = path.read_text(encoding="utf-8")
    lower = text.lower()

    for token in REQUIRED_CLAIM_TOKENS:
        if token not in lower:
            errors.append(f"Missing required claim token: {token}")

    for required_doc in REQUIRED_DRIFT_DOCS:
        if required_doc not in text:
            errors.append(f"Missing required drift reference: {required_doc}")

    if "issue #5296" not in lower and "/issues/5296" not in text:
        errors.append("Missing follow-up issue reference for #5296")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
        help="Path to docs implementation drift audit artifact.",
    )
    args = parser.parse_args(argv)

    audit_path = args.audit_path.expanduser().resolve()
    errors = _validate_audit_report(audit_path)
    if errors:
        for error in errors:
            print(f"[repo-review-audit] FAIL: {error}")
        return 1

    print(f"[repo-review-audit] OK: {audit_path}")
    return 0


if __name__ == "__main__":
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main())
