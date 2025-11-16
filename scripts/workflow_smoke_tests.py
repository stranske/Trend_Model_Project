import datetime as dt
import pathlib
import tempfile

from tools import validate_quarantine_ttl


def _quarantine_smoke() -> None:
    """Exercise the quarantine TTL validator with a sample entry."""
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
    _quarantine_smoke()
    return 0


if __name__ == "__main__":
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main())
