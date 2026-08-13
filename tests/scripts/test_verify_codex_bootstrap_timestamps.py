"""Regression tests for the bootstrap report timestamp contract."""

from __future__ import annotations

import json
import runpy
from datetime import UTC, datetime
from pathlib import Path

import pytest


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


def test_main_marks_invalid_started_timestamp_as_failure(tmp_path: Path) -> None:
    symbols = _bootstrap_symbols()
    scenario_result = symbols["ScenarioResult"]

    def malformed_start(_context):
        return scenario_result("timestamp", "pass", {}, started="not-a-timestamp")

    main_globals = symbols["main"].__globals__
    main_globals["SCENARIO_ENV"] = "timestamp"
    main_globals["SCENARIOS_IMPL"] = {"timestamp": malformed_start}
    main_globals["WORKDIR"] = tmp_path
    with pytest.raises(SystemExit, match="1"):
        symbols["main"]()

    result = json.loads((tmp_path / "codex-verification-report.json").read_text())[0]
    assert result["status"] == "fail"
    assert "Invalid timestamp for duration calculation" in result["error"]
