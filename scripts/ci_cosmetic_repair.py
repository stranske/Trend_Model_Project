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


def _discover_expectation_modules() -> tuple[str, ...]:
    """Dynamically discover test modules for expectation drift repairs."""
    test_dir = ROOT / "tests"
    modules = []
    for path in test_dir.glob("test_*.py"):
        # Convert path to module name, e.g. tests/test_foo.py -> tests.test_foo
        module_name = f"tests.{path.stem}"
        modules.append(module_name)
    return tuple(modules)


_EXPECTATION_MODULES: tuple[str, ...] = _discover_expectation_modules()


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


def _run(
    cmd: Sequence[str], *, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise CosmeticRepairError(
            f"Command {' '.join(cmd)} failed with exit code {result.returncode}:\n{result.stderr or result.stdout}"
        )
    return result


def run_pytest(
    report_path: Path, pytest_args: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"--junitxml={report_path}",
        *pytest_args,
    ]
    return subprocess.run(cmd, text=True, capture_output=True)


_FAILURE_PATTERN = re.compile(r"(COSMETIC_[A-Z_]+)\s+(\{.*?\})")


def parse_failure_message(message: str, *, source: str) -> list[RepairInstruction]:
    instructions: list[RepairInstruction] = []
    for kind, payload in _FAILURE_PATTERN.findall(message):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise CosmeticRepairError(
                f"Unable to decode cosmetic payload from {source}: {exc}"
            ) from exc
        instruction = build_instruction(kind, data, source=source)
        instructions.append(instruction)
    return instructions


def build_instruction(
    kind: str, data: dict[str, object], *, source: str
) -> RepairInstruction:
    path_raw = data.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        raise CosmeticRepairError(f"Missing target path in {source} ({kind})")
    guard = data.get("guard", "")
    if not isinstance(guard, str) or not guard:
        raise CosmeticRepairError(f"Missing guard token in {source} ({kind})")
    key = data.get("key")
    if key is not None and not isinstance(key, str):
        raise CosmeticRepairError(f"Invalid key in {source} ({kind})")

    if kind == "COSMETIC_TOLERANCE":
        value = _format_value(data)
        return RepairInstruction(
            kind="tolerance",
            path=Path(path_raw),
            guard=guard,
            key=key,
            value=value,
            metadata=data,
            source=source,
        )
    if kind == "COSMETIC_SNAPSHOT":
        replacement = data.get("replacement")
        if not isinstance(replacement, str):
            raise CosmeticRepairError(
                f"Snapshot repair requires string replacement ({source})"
            )
        return RepairInstruction(
            kind="snapshot",
            path=Path(path_raw),
            guard=guard,
            key=key,
            value=replacement,
            metadata=data,
            source=source,
        )
    raise CosmeticRepairError(f"Unsupported cosmetic repair type: {kind}")


def _format_value(data: dict[str, object]) -> str:
    if "value" in data:
        raw_value = data["value"]
    elif "actual" in data:
        raw_value = data["actual"]
    else:
        raise CosmeticRepairError("Tolerance payload missing 'value' or 'actual'")

    fmt = None
    if isinstance(data.get("format"), str):
        fmt = data["format"]
    elif isinstance(data.get("digits"), int):
        fmt = f".{data['digits']}f"

    if fmt is not None:
        try:
            formatted = format(float(raw_value), fmt)
        except (TypeError, ValueError) as exc:  # pragma: no cover - validation
            raise CosmeticRepairError(f"Invalid numeric payload: {raw_value}") from exc
        return formatted
    if isinstance(raw_value, (int, float)):
        return repr(raw_value)
    if isinstance(raw_value, str):
        return raw_value
    raise CosmeticRepairError(f"Unsupported value type: {type(raw_value)!r}")


def load_failure_records(report_path: Path) -> list[FailureRecord]:
    summary = classify_test_failures.classify_reports([report_path])
    records: list[FailureRecord] = []
    for bucket in ("cosmetic",):
        for item in summary[bucket]:
            records.append(FailureRecord(id=item["id"], message=item["message"]))
    return records


def collect_instructions(records: Iterable[FailureRecord]) -> list[RepairInstruction]:
    instructions: list[RepairInstruction] = []
    for record in records:
        parsed = parse_failure_message(record.message, source=record.id)
        instructions.extend(parsed)
    return instructions


_FLOAT_GUARD_PATTERN = re.compile(
    r"^(?P<prefix>.*?=\s*)(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?P<suffix>.*#\s*cosmetic-repair:\s*float(?:\s+[-\w.]+)?)"
)


def apply_tolerance_update(
    path: Path, *, guard: str, key: str | None, value: str
) -> bool:
    guard_token = f"{GUARD_PREFIX} {guard}"
    if key:
        guard_token = f"{guard_token} {key}"
    original = path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    changed = False
    guard_found = False
    for line in original:
        if guard_token in line:
            guard_found = True
            match = _FLOAT_GUARD_PATTERN.match(line)
            if not match:
                raise CosmeticRepairError(
                    f"Unable to locate numeric literal for {guard_token} in {path}"
                )
            new_line = f"{match.group('prefix')}{value}{match.group('suffix')}"
            if new_line != line:
                changed = True
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)
    if not guard_found:
        raise CosmeticRepairError(f"Guard comment {guard_token} not found in {path}")
    if not changed:
        return False
    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return changed


def apply_snapshot_update(
    path: Path, *, guard: str, key: str | None, replacement: str
) -> bool:
    guard_token = f"{GUARD_PREFIX} {guard}"
    if key:
        guard_token = f"{guard_token} {key}"
    text = path.read_text(encoding="utf-8")
    if guard_token not in text:
        raise CosmeticRepairError(f"Snapshot guard {guard_token} not found in {path}")
    path.write_text(replacement, encoding="utf-8")
    return True


def apply_instructions(
    instructions: Sequence[RepairInstruction], *, root: Path
) -> list[Path]:
    changed: list[Path] = []
    for instruction in instructions:
        target = instruction.absolute_path(root)
        if instruction.kind == "tolerance":
            updated = apply_tolerance_update(
                target,
                guard=instruction.guard,
                key=instruction.key,
                value=str(instruction.value),
            )
        elif instruction.kind == "snapshot":
            updated = apply_snapshot_update(
                target,
                guard=instruction.guard,
                key=instruction.key,
                replacement=str(instruction.value),
            )
        else:  # pragma: no cover - defensive
            raise CosmeticRepairError(f"Unhandled instruction kind: {instruction.kind}")
        if updated:
            changed.append(target)
    return changed


def working_tree_changes(*, root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines


def stage_and_commit(
    paths: Sequence[Path], *, root: Path, summary: str, branch_suffix: str | None
) -> str:
    branch_suffix = branch_suffix or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    branch = f"{BRANCH_PREFIX}-{branch_suffix}"
    _run(["git", "checkout", "-B", branch], cwd=root)
    _run(["git", "add", *{str(p.relative_to(root)) for p in paths}], cwd=root)
    commit_message = f"Cosmetic repair: {summary}"
    _run(["git", "commit", "-m", commit_message], cwd=root)
    return branch


def push_and_open_pr(
    *, branch: str, base: str, title: str, body: str, labels: Sequence[str], root: Path
) -> None:
    _run(["git", "push", "--force", "origin", branch], cwd=root)
    cmd = [
        "gh",
        "pr",
        "create",
        "--title",
        title,
        "--body",
        body,
        "--base",
        base,
        "--head",
        branch,
    ]
    for label in labels:
        cmd.extend(["--label", label])
    _run(cmd, cwd=root)


def build_pr_body(
    changed: Sequence[Path], instructions: Sequence[RepairInstruction], *, root: Path
) -> str:
    bullets = [f"- {path.relative_to(root)}" for path in changed]
    instruction_lines = [
        f"  * {instr.source}: {instr.kind} -> {instr.path}" for instr in instructions
    ]
    return "\n".join(
        [
            "## Cosmetic Repairs",
            "",
            "The workflow detected cosmetic drift in the following files:",
            *bullets,
            "",
            "### Repair details",
            *instruction_lines,
        ]
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
    timestamp = (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    entries = []
    for result in results:
        if result.status != "applied":
            continue
        detail = f" – {result.detail}" if result.detail else ""
        entries.append(f"- {timestamp} – {result.fixer} for {result.test_id}{detail}")
    _append_guard_entries(entries)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Analyse failures without editing files"
    )
    mode.add_argument("--apply", action="store_true", help="Apply eligible repairs")
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to pytest",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Existing JUnit report to analyse instead of running pytest",
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Repository root (tests only)"
    )
    parser.add_argument(
        "--base",
        type=str,
        default=os.environ.get("GITHUB_BASE_REF", "main"),
        help="Base branch for PRs",
    )
    parser.add_argument(
        "--branch-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to the generated branch name",
    )
    parser.add_argument(
        "--skip-pr",
        action="store_true",
        help="Do not create a branch or PR (useful for tests)",
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
    if not report_path.exists():
        raise CosmeticRepairError(f"JUnit report not found: {report_path}")
    records = load_failure_records(report_path)
    instructions = collect_instructions(records)
    if not instructions:
        if pytest_result is not None and pytest_result.returncode != 0:
            raise CosmeticRepairError(
                "pytest failed but no cosmetic instructions were detected"
            )
        print("No cosmetic repairs detected.")
        return 0
    if mode == "dry-run":
        for instr in instructions:
            print(f"[dry-run] {instr.kind} -> {instr.path}")
        return 0

    changed_paths = apply_instructions(instructions, root=ns.root)
    if not changed_paths:
        print("Cosmetic repairs already up to date; no file changes required.")
        return 0

    status = working_tree_changes(root=ns.root)
    print("Working tree status after repairs:")
    for line in status:
        print(f"  {line}")

    if ns.skip_pr:
        return 0

    branch = stage_and_commit(
        changed_paths,
        root=ns.root,
        summary="cosmetic adjustments",
        branch_suffix=ns.branch_suffix,
    )
    title = "Cosmetic test repairs"
    body = build_pr_body(changed_paths, instructions, root=ns.root)
    push_and_open_pr(
        branch=branch,
        base=ns.base,
        title=title,
        body=body,
        labels=("testing", "autofix:applied"),
        root=ns.root,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
