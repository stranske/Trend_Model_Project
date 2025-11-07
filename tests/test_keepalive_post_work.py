import json
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "keepalive_post_work"
HARNESS = FIXTURES_DIR / "harness.js"


def _require_node() -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for keepalive post-work tests")


def _run_scenario(name: str) -> dict:
    _require_node()
    scenario_path = FIXTURES_DIR / f"{name}.json"
    assert scenario_path.exists(), f"Missing scenario fixture: {scenario_path}"
    result = subprocess.run(
        ["node", str(HARNESS), str(scenario_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "Harness failed with code %s:\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (result.returncode, result.stdout, result.stderr)
        )
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - harness should emit JSON
        pytest.fail(f"Invalid harness output: {exc}: {result.stdout}")


def _summary_table(data: dict) -> list[list[str]]:
    for entry in data.get("summary", []):
        if entry.get("type") == "table":
            return entry.get("rows", [])
    return []


def test_keepalive_sync_detects_head_change_without_actions() -> None:
    data = _run_scenario("head_change")
    events = data["events"]
    assert events["dispatches"] == []
    assert events["comments"] == []
    table = _summary_table(data)
    assert any(
        row[0] == "Initial poll" and "Branch advanced" in row[1] for row in table
    )
    assert any(row[0] == "Result" and "mode=already-synced" in row[1] for row in table)


def test_keepalive_sync_update_branch_success() -> None:
    data = _run_scenario("update_branch")
    events = data["events"]
    actions = [
        dispatch["client_payload"].get("action") for dispatch in events["dispatches"]
    ]
    assert actions == ["update-branch"]
    assert events["labelsRemoved"] == ["agents:sync-required"]
    table = _summary_table(data)
    assert any(
        row[0] == "Update-branch result" and "Branch advanced" in row[1]
        for row in table
    )
    assert any(
        row[0] == "Result" and "mode=dispatch-update-branch" in row[1] for row in table
    )


def test_keepalive_sync_create_pr_flow() -> None:
    data = _run_scenario("create_pr")
    events = data["events"]
    actions = [
        dispatch["client_payload"].get("action") for dispatch in events["dispatches"]
    ]
    assert actions == ["update-branch", "create-pr"]
    table = _summary_table(data)
    assert any(
        row[0] == "Create-pr result" and "Branch advanced" in row[1] for row in table
    )
    assert any(
        row[0] == "Result" and "mode=dispatch-create-pr" in row[1] for row in table
    )


def test_keepalive_sync_escalation_adds_label_and_comment() -> None:
    data = _run_scenario("escalation")
    events = data["events"]
    actions = [
        dispatch["client_payload"].get("action") for dispatch in events["dispatches"]
    ]
    assert actions == ["update-branch", "create-pr"]
    assert events["labelsAdded"] == [["agents:sync-required"]]
    assert len(events["comments"]) == 1
    assert "update-branch/create-pr" in events["comments"][0]["body"]
    table = _summary_table(data)
    assert any(row[0] == "Result" and "mode=sync-timeout" in row[1] for row in table)
