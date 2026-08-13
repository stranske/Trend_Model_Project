"""Regression tests for the bootstrap report timestamp contract."""

from __future__ import annotations

import runpy
from datetime import UTC, datetime
from pathlib import Path


def _bootstrap_symbols() -> dict[str, object]:
    script = Path(__file__).parents[2] / "scripts" / "verify_codex_bootstrap.py"
    return runpy.run_path(str(script), run_name="verify_codex_bootstrap_test")


def test_bootstrap_timestamps_are_canonical_and_parse_naive_inputs() -> None:
    symbols = _bootstrap_symbols()
    timestamp = symbols["_utc_timestamp"]
    parse_timestamp = symbols["_parse_utc_timestamp"]

    assert timestamp(datetime(2026, 8, 13, 17, 0, tzinfo=UTC)).endswith("Z")
    parsed = parse_timestamp("2026-08-13T17:00:00")
    assert parsed.tzinfo == UTC
