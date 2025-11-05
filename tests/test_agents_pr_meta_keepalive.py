from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "agents_pr_meta"
HARNESS = FIXTURES_DIR / "harness.js"


def _require_node() -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for agents-pr-meta keepalive tests")


def _run_scenario(name: str) -> dict:
    _require_node()
    scenario_path = FIXTURES_DIR / f"{name}.json"
    assert (
        scenario_path.exists()
    ), f"Scenario fixture missing: {scenario_path}"  # pragma: no cover
    command = ["node", str(HARNESS), str(scenario_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "Harness failed with code %s:\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (result.returncode, result.stdout, result.stderr)
        )
    try:
        return json.loads(result.stdout or "{}")
    except (
        json.JSONDecodeError
    ) as exc:  # pragma: no cover - harness should return valid JSON
        pytest.fail(f"Invalid harness output: {exc}: {result.stdout}")


def test_keepalive_detection_dispatches_orchestrator() -> None:
    data = _run_scenario("dispatch")
    outputs = data["outputs"]
    assert outputs["dispatch"] == "true"
    assert outputs["reason"] == "keepalive-detected"
    assert outputs["issue"] == "3227"
    assert outputs["round"] == "3"
    assert outputs["branch"] == "codex/issue-3227-keepalive"
    assert outputs["base"] == "phase-2-dev"
    assert outputs["trace"] == "manual-resend"
    assert outputs["pr"] == "3230"
    assert outputs["comment_id"] == "987654321"
    assert (
        outputs["comment_url"]
        == "https://github.com/stranske/Trend_Model_Project/pull/3230#issuecomment-987654321"
    )

    calls = data.get("calls", {})
    created = calls.get("reactionsCreated", [])
    assert created == [{"comment_id": 987654321, "content": "rocket"}]


def test_keepalive_detection_handles_after_markers() -> None:
    data = _run_scenario("after_markers")
    outputs = data["outputs"]
    assert outputs["dispatch"] == "true"
    assert outputs["reason"] == "keepalive-detected"
    assert outputs["round"] == "5"
    assert outputs["trace"] == "manual-test-2025-11-05-01-35"


def test_keepalive_detection_handles_html_entities() -> None:
    data = _run_scenario("html_entities")
    outputs = data["outputs"]
    assert outputs["dispatch"] == "true"
    assert outputs["reason"] == "keepalive-detected"
    assert outputs["round"] == "6"
    assert outputs["trace"] == "double-sanitized-check"


def test_keepalive_detection_requires_marker() -> None:
    data = _run_scenario("missing_marker")
    outputs = data["outputs"]
    assert outputs["dispatch"] == "false"
    assert outputs["reason"] == "missing-sentinel"
    assert outputs["comment_id"] == "4001001"


def test_keepalive_detection_requires_round_marker() -> None:
    data = _run_scenario("missing_round")
    outputs = data["outputs"]
    assert outputs["dispatch"] == "false"
    assert outputs["reason"] == "missing-round"
    assert outputs["comment_id"] == "1122334455"


def test_keepalive_detection_validates_author() -> None:
    data = _run_scenario("unauthorised")
    outputs = data["outputs"]
    assert outputs["dispatch"] == "false"
    assert outputs["reason"] == "unauthorised-author"
