import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "keepalive"
HARNESS = FIXTURES_DIR / "harness.js"


def _require_node() -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for keepalive harness tests")


def _run_scenario(name: str) -> dict:
    _require_node()
    scenario_path = FIXTURES_DIR / f"{name}.json"
    assert scenario_path.exists(), f"Scenario fixture missing: {scenario_path}"
    command = ["node", str(HARNESS), str(scenario_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "Harness failed with code %s:\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (result.returncode, result.stdout, result.stderr)
        )
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        pytest.fail(f"Invalid harness output: {exc}: {result.stdout}")


def _raw_entries(summary: dict) -> list[str]:
    return [entry["text"] for entry in summary["entries"] if entry.get("type") == "raw"]


def _details(summary: dict, title_prefix: str) -> dict | None:
    for entry in summary["entries"]:
        if entry.get("type") == "details" and entry.get("title", "").startswith(title_prefix):
            return entry
    return None


def _extract_marked_values(line: str) -> list[str]:
    return re.findall(r"\*\*([^*]+)\*\*", line)


def test_keepalive_skip_requested() -> None:
    data = _run_scenario("skip_opt_out")
    summary = data["summary"]
    raw = _raw_entries(summary)
    assert "Skip requested via options_json." in raw
    assert data["created_comments"] == []


def test_keepalive_idle_threshold_logic() -> None:
    data = _run_scenario("idle_threshold")
    summary = data["summary"]
    created = data["created_comments"]
    assert [item["issue_number"] for item in created] == [101]
    assert created[0]["body"].startswith("@codex plan-and-execute")

    details = _details(summary, "Triggered keepalive comments")
    assert details is not None and len(details["items"]) == 1

    raw = _raw_entries(summary)
    assert "Triggered keepalive count: 1" in raw
    assert "Evaluated pull requests: 3" in raw


def test_keepalive_dry_run_records_previews() -> None:
    data = _run_scenario("dry_run")
    summary = data["summary"]
    assert data["created_comments"] == []

    preview = _details(summary, "Previewed keepalive comments")
    assert preview is not None and len(preview["items"]) == 1

    raw = _raw_entries(summary)
    assert "Previewed keepalive count: 1" in raw


def test_keepalive_dedupes_configuration() -> None:
    data = _run_scenario("dedupe")
    summary = data["summary"]

    raw = _raw_entries(summary)
    labels_line = next(line for line in raw if line.startswith("Target labels:"))
    logins_line = next(line for line in raw if line.startswith("Agent logins:"))
    assert _extract_marked_values(labels_line) == ["agent:codex", "agent:triage"]
    assert _extract_marked_values(logins_line) == ["chatgpt-codex-connector", "helper-bot"]

    created = data["created_comments"]
    assert [item["issue_number"] for item in created] == [505]
    assert created[0]["body"].endswith("<!-- codex-keepalive -->")

    details = _details(summary, "Triggered keepalive comments")
    assert details is not None and any("#505" in entry for entry in details["items"])
