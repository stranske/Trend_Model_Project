"""Apply cosmetic pytest repairs in CI.

The script is designed to run inside the cosmetic repair workflow.  It
consumes a JUnit XML report, classifies failures using
``scripts.classify_test_failures`` and runs a small set of targeted
fixers for well-understood cosmetic breakages:

* Aggregate number formatting in ``automation_multifailure``
* Expectation drift maintained by ``scripts.update_autofix_expectations``

Whenever a fixer updates the repository it also appends a short note to
``docs/COSMETIC_REPAIR_LOG.md`` between guard markers so that the
resulting pull request contains a reviewable trace of the automatic
changes.

The script intentionally avoids over-reaching.  Failures that are not
recognised remain untouched and are reported in the JSON summary output
so that maintainers can review them manually.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classify_test_failures import FailureRecord  # noqa: E402
from scripts.classify_test_failures import classify_reports  # noqa: E402

_LOG_PATH = ROOT / "docs" / "COSMETIC_REPAIR_LOG.md"
_GUARD_START = "<!-- cosmetic-repair:start -->"
_GUARD_END = "<!-- cosmetic-repair:end -->"

_EXPECTATION_MODULES: tuple[str, ...] = (
    "tests.test_pipeline_warmup_autofix",
    "tests.test_rank_selection_core_unit",
    "tests.test_selector_weighting",
    "tests.test_autofix_repo_regressions",
)


@dataclass
class FixResult:
    """Metadata describing an attempted repair."""

    test_id: str
    fixer: str
    status: str
    detail: str | None = None


class CosmeticFixer:
    """Base class for targeted cosmetic repairs."""

    name: str

    def matches(self, record: FailureRecord) -> bool:
        raise NotImplementedError

    def apply(self, record: FailureRecord) -> FixResult:
        raise NotImplementedError


def _ensure_guard_file() -> list[str]:
    """Return the content of the guard file, creating it if necessary."""

    if not _LOG_PATH.exists():
        _LOG_PATH.write_text(
            "# Cosmetic Repair Log\n\n"
            "This document is maintained by the cosmetic repair workflow.\n"
            "Updates between the guard markers are generated automatically by\n"
            "`scripts/ci_cosmetic_repair.py` whenever a workflow run applies a\n"
            "repair.\n\n"
            f"{_GUARD_START}\n"
            f"{_GUARD_END}\n",
            encoding="utf-8",
        )
    text = _LOG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    if _GUARD_START not in lines or _GUARD_END not in lines:
        raise RuntimeError(
            "Guard markers missing from COSMETIC_REPAIR_LOG.md; unable to append entry."
        )
    return lines


def _append_guard_entries(entries: Iterable[str]) -> None:
    entries = list(entries)
    if not entries:
        return
    lines = _ensure_guard_file()
    start_idx = lines.index(_GUARD_START)
    end_idx = lines.index(_GUARD_END)
    body = lines[start_idx + 1 : end_idx]
    for entry in entries:
        if entry in body:
            continue
        body.append(entry)
    new_lines = lines[: start_idx + 1] + body + lines[end_idx:]
    _LOG_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


class AggregateNumbersFixer(CosmeticFixer):
    name = "aggregate_numbers"

    def matches(self, record: FailureRecord) -> bool:
        return (
            "automation_multifailure" in record.id
            or "aggregate_numbers" in record.message
        )

    def apply(self, record: FailureRecord) -> FixResult:
        from scripts import fix_cosmetic_aggregate

        target = fix_cosmetic_aggregate.TARGET
        try:
            before = target.read_text(encoding="utf-8")
        except OSError:
            before = None
        fix_cosmetic_aggregate.main()
        if before is None:
            detail = "Target file missing before fixer execution."
            status = "skipped"
        else:
            after = target.read_text(encoding="utf-8")
            if after != before:
                detail = f"Updated {target.relative_to(ROOT)}"
                status = "applied"
            else:
                detail = "No changes required"
                status = "noop"
        return FixResult(
            test_id=record.id,
            fixer=self.name,
            status=status,
            detail=detail,
        )


class ExpectationUpdateFixer(CosmeticFixer):
    name = "update_autofix_expectations"

    def matches(self, record: FailureRecord) -> bool:
        module = record.id.split("::", 1)[0]
        return module in _EXPECTATION_MODULES

    def apply(self, record: FailureRecord) -> FixResult:
        from scripts import update_autofix_expectations

        module_paths: dict[Path, str] = {}
        for target in update_autofix_expectations.TARGETS:
            module = importlib.import_module(target.module)
            module_path = Path(module.__file__).resolve()
            try:
                module_paths[module_path] = module_path.read_text(encoding="utf-8")
            except OSError:
                continue
        update_autofix_expectations.main()
        updated: list[str] = []
        for path, before in module_paths.items():
            try:
                after = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if after != before:
                updated.append(str(path.relative_to(ROOT)))
        if updated:
            status = "applied"
            detail = ", ".join(sorted(updated))
        else:
            status = "noop"
            detail = "No expectation constants changed"
        return FixResult(
            test_id=record.id,
            fixer=self.name,
            status=status,
            detail=detail,
        )


_FIXERS: tuple[CosmeticFixer, ...] = (
    AggregateNumbersFixer(),
    ExpectationUpdateFixer(),
)


def _load_failures(reports: Iterable[str | Path]) -> list[FailureRecord]:
    summary = classify_reports(reports)
    cosmetic_records = summary.get("cosmetic", [])
    runtime = summary.get("runtime", [])
    unknown = summary.get("unknown", [])
    if runtime or unknown:
        print(
            "[ci_cosmetic_repair] Runtime or unknown failures detected; skipping repair."
        )
        return []
    records: list[FailureRecord] = []
    for payload in cosmetic_records:
        records.append(
            FailureRecord(
                id=payload["id"],
                file=payload["file"],
                markers=tuple(payload.get("markers", ())),
                message=payload.get("message", ""),
                failure_type=payload.get("failure_type", "failure"),
            )
        )
    return records


def _run_fixers(records: Sequence[FailureRecord]) -> list[FixResult]:
    results: list[FixResult] = []
    for record in records:
        handled = False
        for fixer in _FIXERS:
            if fixer.matches(record):
                result = fixer.apply(record)
                results.append(result)
                handled = True
                break
        if not handled:
            results.append(
                FixResult(
                    test_id=record.id,
                    fixer="unhandled",
                    status="skipped",
                    detail="No fixer registered for this failure",
                )
            )
    return results


def _summarise(results: Sequence[FixResult]) -> dict[str, object]:
    applied = [r for r in results if r.status == "applied"]
    return {
        "total": len(results),
        "applied": len(applied),
        "results": [r.__dict__ for r in results],
    }


def _append_log_entries(results: Sequence[FixResult]) -> None:
    timestamp = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    entries = []
    for result in results:
        if result.status != "applied":
            continue
        detail = f" – {result.detail}" if result.detail else ""
        entries.append(f"- {timestamp} – {result.fixer} for {result.test_id}{detail}")
    _append_guard_entries(entries)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--junit",
        type=Path,
        default=Path("pytest-report.xml"),
        help="Path to the pytest JUnit XML report",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = _parse_args(argv)
    reports: list[str] = []
    if ns.junit.exists():
        reports.append(str(ns.junit))
    if not reports:
        print(f"[ci_cosmetic_repair] JUnit report {ns.junit} not found; nothing to do.")
        return 0
    records = _load_failures(reports)
    if not records:
        print("[ci_cosmetic_repair] No cosmetic-only failures detected.")
        return 0
    results = _run_fixers(records)
    summary = _summarise(results)
    _append_log_entries(results)
    if ns.summary:
        ns.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("[ci_cosmetic_repair] Summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
