from __future__ import annotations

import datetime as dt
import pathlib
import tempfile

from tools import repo_health_probe
from tools import validate_quarantine_ttl


def _repo_health_smoke() -> None:
    report = repo_health_probe.run_probe_from_sources(
        labels=["agent:codex", "priority:p0", "tech:coverage", "workflows"],
        secrets=["SERVICE_BOT_PAT"],
        variables=["OPS_HEALTH_ISSUE"],
    )
    summary = repo_health_probe.build_summary(
        report, actionlint_ok=True, issue_id_present=True
    )
    if report.get("ok") is not True:
        raise RuntimeError("Repo health smoke should succeed but reported issues")
    print(summary.summary_markdown)


def _quarantine_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        quarantine_file = tmp_path / "quarantine.yml"
        quarantine_file.write_text(
            """
            tests:
              - id: tests/example.py::test_smoke_alpha
                expires: "2050-01-01"
            """,
            encoding="utf-8",
        )
        records, invalid = validate_quarantine_ttl.load_records(quarantine_file)
        report = validate_quarantine_ttl.evaluate_records(
            records,
            today=dt.date(2049, 1, 1),
            additional_invalid=invalid,
        )
        if not report.ok:
            raise RuntimeError("Quarantine TTL smoke reported unexpected failure")
        print(validate_quarantine_ttl.build_summary(report))


def main() -> int:
    _repo_health_smoke()
    _quarantine_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
