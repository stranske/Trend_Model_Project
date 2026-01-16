"""Tests for the Gate workflow trigger helper."""

from __future__ import annotations

import subprocess

import pytest

from scripts import trigger_gate_workflow as tgw


def test_resolve_branch_queries_pr(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_json(args: list[str]) -> object:
        calls.append(args)
        return {"headRefName": "feature-branch", "headRefOid": "abc123"}

    monkeypatch.setattr(tgw, "run_json_command", fake_run_json)

    branch = tgw.resolve_branch("123", "owner/repo")

    assert branch == "feature-branch"
    assert calls == [
        [
            "gh",
            "pr",
            "view",
            "123",
            "--repo",
            "owner/repo",
            "--json",
            "headRefName,headRefOid",
        ]
    ]


def test_trigger_gate_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], check: bool, text: bool) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(tgw, "resolve_branch", lambda pr, repo: "feature-branch")
    monkeypatch.setattr(tgw.subprocess, "run", fake_run)

    branch, followup = tgw.trigger_gate("45", "org/repo")

    assert branch == "feature-branch"
    assert calls == [
        [
            "gh",
            "workflow",
            "run",
            "pr-00-gate.yml",
            "--repo",
            "org/repo",
            "--ref",
            "feature-branch",
        ]
    ]
    assert (
        followup == "gh run list --repo org/repo --workflow pr-00-gate.yml --branch feature-branch"
    )


def test_ensure_gate_workflow_enabled_noop_when_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(tgw, "fetch_gate_workflow", lambda repo: {"state": "active"})

    def fake_enable(repo: str) -> None:
        calls.append(repo)

    monkeypatch.setattr(tgw, "enable_gate_workflow", fake_enable)

    note = tgw.ensure_gate_workflow_enabled("org/repo")

    assert note == "Gate workflow is already active."
    assert calls == []


def test_ensure_gate_workflow_enabled_enables_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(tgw, "fetch_gate_workflow", lambda repo: {"state": "disabled_manually"})

    def fake_enable(repo: str) -> None:
        calls.append(repo)

    monkeypatch.setattr(tgw, "enable_gate_workflow", fake_enable)

    note = tgw.ensure_gate_workflow_enabled("org/repo")

    assert "enabled via gh workflow enable" in note
    assert calls == ["org/repo"]


def test_ensure_gate_triggers_when_missing_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tgw, "resolve_pr_info", lambda pr, repo: ("feature-branch", "abc123"))
    monkeypatch.setattr(tgw, "fetch_latest_gate_run", lambda branch, repo: None)
    monkeypatch.setattr(
        tgw,
        "dispatch_gate",
        lambda branch, repo: "gh run list --repo org/repo --workflow pr-00-gate.yml --branch feature-branch",
    )

    triggered, branch, note, followup = tgw.ensure_gate("123", "org/repo")

    assert triggered is True
    assert branch == "feature-branch"
    assert "Dispatching Gate workflow" in note
    assert (
        followup == "gh run list --repo org/repo --workflow pr-00-gate.yml --branch feature-branch"
    )


def test_ensure_gate_skips_when_run_matches_head(monkeypatch: pytest.MonkeyPatch) -> None:
    latest_run = {
        "headSha": "abc123",
        "status": "completed",
        "conclusion": "success",
        "htmlUrl": "https://example.test/run/1",
    }
    monkeypatch.setattr(tgw, "resolve_pr_info", lambda pr, repo: ("feature-branch", "abc123"))
    monkeypatch.setattr(tgw, "fetch_latest_gate_run", lambda branch, repo: latest_run)

    triggered, branch, note, followup = tgw.ensure_gate("123", "org/repo")

    assert triggered is False
    assert branch == "feature-branch"
    assert "already completed" in note
    assert (
        followup == "gh run list --repo org/repo --workflow pr-00-gate.yml --branch feature-branch"
    )


def test_main_reports_missing_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tgw.shutil, "which", lambda _: None)

    exit_code = tgw.main(["123"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "gh CLI is required" in captured.err


def test_main_success_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tgw.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(tgw, "trigger_gate", lambda pr, repo: ("feature-branch", "gh run list ..."))

    exit_code = tgw.main(["123", "org/repo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Triggering pr-00-gate.yml for PR #123" in captured.out
    assert "gh run list ..." in captured.out


def test_main_ensure_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tgw.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(
        tgw, "ensure_gate_workflow_enabled", lambda repo: "Gate workflow is already active."
    )
    monkeypatch.setattr(tgw, "trigger_gate", lambda pr, repo: ("feature-branch", "gh run list ..."))

    exit_code = tgw.main(["123", "org/repo", "--ensure-enabled"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Gate workflow is already active." in captured.out
    assert "Triggering pr-00-gate.yml for PR #123" in captured.out


def test_format_gate_status_with_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tgw, "resolve_pr_info", lambda pr, repo: ("feature-branch", "abc123"))
    monkeypatch.setattr(tgw, "resolve_gate_workflow_state", lambda repo: "active")
    monkeypatch.setattr(
        tgw,
        "fetch_latest_gate_run",
        lambda branch, repo: {
            "status": "completed",
            "conclusion": "success",
            "headSha": "abc123",
            "createdAt": "2026-01-01T00:00:00Z",
            "htmlUrl": "https://example.test/run/1",
        },
    )

    lines = tgw.format_gate_status("123", "org/repo")

    assert lines[0] == "Gate workflow state: active"
    assert lines[1] == "PR #123 branch: feature-branch"
    assert lines[2] == "PR #123 head: abc123"
    assert "Latest Gate run: status=completed conclusion=success head=abc123 created=2026-01-01T00:00:00Z" in lines[3]


def test_main_status(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(tgw.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(
        tgw,
        "format_gate_status",
        lambda pr, repo: ["Gate workflow state: active", "Latest Gate run: none found"],
    )

    exit_code = tgw.main(["123", "org/repo", "--status"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Gate workflow state: active" in captured.out
    assert "Latest Gate run: none found" in captured.out
