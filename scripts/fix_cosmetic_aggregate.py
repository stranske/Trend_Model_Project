#!/usr/bin/env python
"""Normalize aggregate_numbers cosmetic formatting.

This helper ensures the diagnostic automation module joins manager counts
with a ``" | "`` separator instead of commas. It is intended to run inside the
cosmetic follow-up workflow so that cosmetic-only regressions are repaired
without manual intervention.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "src/trend_analysis/automation_multifailure.py"
_SENTINEL = '" | "'


def main() -> int:
    try:
        original = TARGET.read_text(encoding="utf-8")
    except FileNotFoundError:
        print("[fix_cosmetic_aggregate] Target file missing; skipping.")
        return 0

    if _SENTINEL in original:
        print("[fix_cosmetic_aggregate] aggregate_numbers already uses pipe separator.")
        return 0

    needle = '",".join(str(v) for v in values)'
    replacement = '" | ".join(str(v) for v in values)'

    if needle not in original:
        print("[fix_cosmetic_aggregate] Expected pattern not found; skipping.")
        return 0

    updated = original.replace(needle, replacement, 1)
    TARGET.write_text(updated, encoding="utf-8")
    print("[fix_cosmetic_aggregate] Updated aggregate_numbers to use pipe separator.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main())
