#!/usr/bin/env python
"""Apply conservative automatic fixes for common mypy diagnostics.

Current focus: inject ``from typing import ...`` statements when mypy reports
``Name "X" is not defined`` for well-known typing helpers (``Optional``,
``Literal`` …).  The goal is to clean up trivial import omissions so that the
remaining type errors represent structural problems that need human review.

Design goals
============
* Pure standard-library implementation – safe to run inside automation.
* Idempotent edits: running the script repeatedly should not duplicate imports.
* Non-blocking: if mypy is unavailable or the heuristics do not apply, the
  script exits successfully without modifying files.

Usage
=====
The script defaults to running mypy against ``src/`` and ``tests/`` relative to
repository root.  Custom paths can be provided via ``--paths`` and a different
mypy config via ``--config-file``.  Enable ``--dry-run`` to preview changes
without touching the filesystem.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Iterable, Mapping, Sequence, cast

Diagnostic = dict[str, Any]

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = [ROOT / "src", ROOT / "tests"]
NAME_PATTERN = re.compile(r"Name ['\"](?P<name>[A-Za-z0-9_]+)['\"] is not defined")

# Symbols that safely come from ``typing`` in Python 3.11+.
TYPING_SYMBOLS = {
    "Annotated",
    "Any",
    "AsyncIterable",
    "AsyncIterator",
    "Awaitable",
    "Callable",
    "ClassVar",
    "Concatenate",
    "Coroutine",
    "Counter",
    "DefaultDict",
    "Deque",
    "Dict",
    "Final",
    "FrozenSet",
    "Generic",
    "Iterable",
    "Iterator",
    "Literal",
    "Mapping",
    "MutableMapping",
    "MutableSequence",
    "MutableSet",
    "NamedTuple",
    "NoReturn",
    "Optional",
    "OrderedDict",
    "Protocol",
    "Sequence",
    "Set",
    "SupportsFloat",
    "SupportsInt",
    "Tuple",
    "Type",
    "TypedDict",
    "TypeGuard",
    "TypeVar",
    "Union",
    "Unpack",
    "Self",
    "Required",
    "Required",
    "NotRequired",
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        default=str(ROOT / "pyproject.toml"),
        help="Path to mypy configuration file (default: pyproject.toml).",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Specific paths to check. Defaults to src/ and tests/ if omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview edits without writing to disk.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print raw mypy JSON diagnostics (debug helper).",
    )
    return parser.parse_args(argv)


def resolve_targets(paths: list[str] | None) -> list[Path]:
    if paths:
        candidates = [Path(p).resolve() for p in paths]
    else:
        candidates = DEFAULT_TARGETS
    existing = [p for p in candidates if p.exists()]
    return existing


def _run_mypy_subprocess(args: list[str]) -> tuple[str, str, int]:
    cmd = [sys.executable, "-m", "mypy", *args]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "", "mypy not available", 0
    except OSError as exc:
        return "", f"failed to execute mypy: {exc}", 0
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    status = int(completed.returncode)
    if status != 0 and "No module named mypy" in stderr:
        return "", "mypy not available", 0
    return stdout, stderr, status


def run_mypy(
    config_file: str | None, targets: Iterable[Path]
) -> tuple[str, str, int, bool]:
    args: list[str] = [
        "--hide-error-context",
        "--show-column-numbers",
        "--error-format=json",
    ]
    cfg = Path(config_file) if config_file else None
    if cfg and cfg.exists():
        args += ["--config-file", str(cfg)]
    args += [str(path) for path in targets]

    stdout, stderr, status = _run_mypy_subprocess(args)

    if status == 2 and "--error-format" in stderr:
        fallback_args = [
            "--hide-error-context",
            "--show-column-numbers",
        ]
        if cfg and cfg.exists():
            fallback_args += ["--config-file", str(cfg)]
        fallback_args += [str(path) for path in targets]
        stdout, stderr, status = _run_mypy_subprocess(fallback_args)
        return stdout, stderr, status, False

    return stdout, stderr, status, True


def iter_diagnostics(output: str, json_mode: bool) -> Iterable[Diagnostic]:
    if json_mode:
        for raw in output.splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            yield cast(Diagnostic, data)
        return

    text_pattern = re.compile(
        r"^(?P<path>[^:]+):(?P<line>\d+):(?P<column>\d+):\s*(?P<severity>error|warning):\s*(?P<message>.+)$"
    )
    for raw in output.splitlines():
        match = text_pattern.match(raw.strip())
        if not match:
            continue
        payload = match.groupdict()
        diag: Diagnostic = {
            "path": payload.get("path"),
            "line": int(payload["line"]),
            "column": int(payload["column"]),
            "severity": payload.get("severity"),
            "message": payload.get("message"),
        }
        yield diag


def extract_missing_typing_symbol(message: str) -> str | None:
    match = NAME_PATTERN.search(message)
    if not match:
        return None
    candidate = match.group("name")
    if candidate in TYPING_SYMBOLS:
        return candidate
    return None


def gather_missing_symbols(diags: Iterable[Diagnostic]) -> Mapping[Path, set[str]]:
    missing: DefaultDict[Path, set[str]] = defaultdict(set)
    for diag in diags:
        severity = diag.get("severity")
        if not isinstance(severity, str) or severity not in {"error", "warning"}:
            continue
        message_obj = diag.get("message")
        message = message_obj if isinstance(message_obj, str) else ""
        symbol = extract_missing_typing_symbol(message)
        if not symbol:
            continue
        path_candidate = diag.get("path") or diag.get("file")
        if not isinstance(path_candidate, str):
            continue
        path = Path(path_candidate)
        if not path.is_file():
            continue
        missing[path].add(symbol)
    return missing


def current_typing_imports(source: str) -> set[str]:
    """Return the set of symbols imported from typing in *source*."""
    imports: set[str] = set()
    pattern = re.compile(r"^\s*from\s+typing\s+import\s+(?P<body>.+)$")
    lines = source.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = pattern.match(line)
        if not match:
            idx += 1
            continue
        body = match.group("body").split("#", 1)[0]
        block = body
        paren_balance = block.count("(") - block.count(")")
        while paren_balance > 0 and idx + 1 < len(lines):
            idx += 1
            continuation = lines[idx].split("#", 1)[0]
            block += "," + continuation.strip()
            paren_balance += continuation.count("(") - continuation.count(")")
        tokens = [
            token.strip()
            for token in block.replace("(", "").replace(")", "").split(",")
        ]
        for token in tokens:
            if not token:
                continue
            base = token.split(" as ", 1)[0].strip()
            if base:
                imports.add(base)
        idx += 1
    return imports


def find_import_section(lines: list[str]) -> int:
    """Heuristic insertion index for new imports."""
    if lines and lines[0].startswith("#!"):
        index = 1
    else:
        index = 0

    # Skip encoding comments.
    while (
        index < len(lines) and lines[index].startswith("#") and "coding" in lines[index]
    ):
        index += 1

    # Skip module docstring if present.
    if index < len(lines) and lines[index].lstrip().startswith(("'", '"')):
        quote = lines[index].lstrip()[0]
        triple = quote * 3
        if lines[index].lstrip().startswith(triple):
            index += 1
            while index < len(lines):
                if lines[index].rstrip().endswith(triple):
                    index += 1
                    break
                index += 1

    # Skip blank lines after docstring.
    while index < len(lines) and not lines[index].strip():
        index += 1

    # Skip future imports.
    while index < len(lines) and lines[index].startswith("from __future__ import"):
        index += 1

    # Ensure a blank line before the new import when the next line is not blank.
    if index < len(lines) and lines[index].strip():
        return index
    return index


def update_typing_import_line(line: str, names: Iterable[str]) -> str:
    prefix, suffix = line.split("import", 1)
    existing = [token.strip() for token in suffix.split(",") if token.strip()]
    combined = sorted(set(existing) | set(names), key=str.lower)
    return f"{prefix}import {', '.join(combined)}"


def apply_typing_imports(path: Path, symbols: set[str], dry_run: bool = False) -> bool:
    if not symbols:
        return False
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False

    existing = current_typing_imports(source)
    missing = sorted({s for s in symbols if s not in existing}, key=str.lower)
    if not missing:
        return False

    lines = source.splitlines()
    simple_index = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("from typing import")
            and "(" not in stripped
            and "\\" not in stripped
        ):
            simple_index = idx
            break

    if simple_index is not None:
        lines[simple_index] = update_typing_import_line(lines[simple_index], missing)
    else:
        insert_at = find_import_section(lines)
        new_line = f"from typing import {', '.join(missing)}"
        lines.insert(insert_at, new_line)
        # Ensure surrounding blank line for readability if needed.
        if insert_at + 1 < len(lines) and lines[insert_at + 1].strip():
            lines.insert(insert_at + 1, "")

    if dry_run:
        print(
            f"[mypy_autofix] Would update {path} with typing symbols: {', '.join(missing)}"
        )
        return True

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[mypy_autofix] Added typing imports to {path.relative_to(ROOT)}: {', '.join(missing)}"
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    targets = resolve_targets(args.paths)
    if not targets:
        print("[mypy_autofix] No targets to analyze; exiting.")
        return 0

    stdout, stderr, exit_code, json_mode = run_mypy(args.config_file, targets)
    if args.show_output and stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    diagnostics = list(iter_diagnostics(stdout, json_mode))
    missing_map = gather_missing_symbols(diagnostics)

    changed = False
    for path, symbols in missing_map.items():
        # Only touch files within the repository root.
        try:
            rel = path.resolve().relative_to(ROOT)
        except ValueError:
            continue
        file_path = ROOT / rel
        if file_path.suffix != ".py":
            continue
        if apply_typing_imports(file_path, symbols, dry_run=args.dry_run):
            changed = True

    if changed and not args.dry_run:
        print("[mypy_autofix] Typing import fixes were applied.")

    if not diagnostics and exit_code == 0:
        print("[mypy_autofix] No mypy diagnostics detected.")
    elif not missing_map:
        print("[mypy_autofix] No eligible typing import issues detected.")

    if exit_code not in {0, 1}:
        # Unexpected mypy failure (syntax error, internal crash). We surface a warning
        # but still succeed so automation can continue.
        print(f"[mypy_autofix] mypy exited with status {exit_code}.")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    sys.exit(main())
