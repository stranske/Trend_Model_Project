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
