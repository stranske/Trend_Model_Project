from __future__ import annotations

import datetime as dt
from pathlib import Path

from tools import validate_quarantine_ttl as ttl


def test_load_records_and_evaluate(tmp_path: Path) -> None:
    quarantine_file = tmp_path / "quarantine.yml"
    quarantine_file.write_text(
        """
        tests:
          - id: tests/example.py::test_alpha
            expires: "2050-01-01"
          - id: tests/example.py::test_beta
            expires: "2024-01-01"
          - id: tests/example.py::test_gamma
            expires: invalid-date
          - reason: missing id
        """,
        encoding="utf-8",
    )

    records, invalid_entries = ttl.load_records(quarantine_file)
    assert len(records) == 2
    assert any("invalid-date" in message for message in invalid_entries)
    assert any("<missing id>" in message for message in invalid_entries)

    report = ttl.evaluate_records(
        records,
        today=dt.date(2024, 6, 1),
        additional_invalid=invalid_entries,
    )
    assert report.total_entries == 2
    assert any(record.identifier.endswith("test_beta") for record in report.expired)
    assert report.invalid == invalid_entries

    summary = ttl.build_summary(report)
    assert "Expired quarantines" in summary
    assert "invalid" in summary


def test_cli_outputs(tmp_path: Path, monkeypatch) -> None:
    quarantine_file = tmp_path / "quarantine.yml"
    quarantine_file.write_text(
        """
        tests:
          - id: tests/example.py::test_alpha
            expires: "2050-01-01"
        """,
        encoding="utf-8",
    )
    step_summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    exit_code = ttl.main([str(quarantine_file), "--json"])
    assert exit_code == 0
    assert "Quarantine TTL validation" in step_summary.read_text(encoding="utf-8")
