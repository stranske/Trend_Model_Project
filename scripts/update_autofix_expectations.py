#!/usr/bin/env python
"""Refresh test expectations for autofix-enabled scenarios.

This script is consumed by the autofix workflow to repair intentionally stale
assertions without human intervention. Each target specifies a module, the
callable that computes the authoritative value, and the constant assignment to
update in-place.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AutofixTarget:
    module: str
    callable_name: str
    constant_name: str


TARGETS: tuple[AutofixTarget, ...] = (
    AutofixTarget(
        module="tests.test_pipeline_warmup_autofix",
        callable_name="compute_expected_rows_for_autofix",
        constant_name="EXPECTED_IN_SAMPLE_ROWS",
    ),
)


def _update_constant(path: Path, constant: str, value: Any) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False

    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(constant)}\s*=\s*)(?P<existing>.+)$", re.MULTILINE
    )
    formatted_value = repr(value)

    def _repl(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{formatted_value}"

    new_text, count = pattern.subn(_repl, text, count=1)
    if count == 0 or new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    changed = False
    for target in TARGETS:
        module = importlib.import_module(target.module)
        compute = getattr(module, target.callable_name, None)
        if compute is None:
            continue
        value = compute()
        module_path = Path(module.__file__).resolve()
        if _update_constant(module_path, target.constant_name, value):
            print(
                f"[update_autofix_expectations] Updated {target.constant_name} in"
                f" {module_path.relative_to(ROOT)} -> {value}"
            )
            changed = True
    if not changed:
        print("[update_autofix_expectations] No expectation updates applied.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
