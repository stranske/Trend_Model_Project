#!/usr/bin/env python3
"""Trigger the Gate workflow for a PR via GitHub CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Mapping


def run_command(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def run_json_command(args: list[str]) -> object:
    output = run_command(args)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Expected JSON output from command: {' '.join(args)}") from exc


def resolve_pr_info(pr_number: str, repo: str) -> tuple[str, str]:
    data = run_json_command(
        [
            "gh",
            "pr",
            "view",
            pr_number,
            "--repo",
            repo,
            "--json",
            "headRefName,headRefOid",
        ]
    )
    if not isinstance(data, Mapping):
        raise RuntimeError(f"Unable to resolve PR metadata for #{pr_number} in {repo}.")
    branch = data.get("headRefName")
    head_sha = data.get("headRefOid")
    if not isinstance(branch, str) or not branch:
        raise RuntimeError(f"Unable to resolve branch for PR #{pr_number} in {repo}.")
    if not isinstance(head_sha, str) or not head_sha:
        raise RuntimeError(f"Unable to resolve head SHA for PR #{pr_number} in {repo}.")
    return branch, head_sha


def resolve_branch(pr_number: str, repo: str) -> str:
    branch, _head_sha = resolve_pr_info(pr_number, repo)
    return branch


def fetch_latest_gate_run(branch: str, repo: str) -> Mapping[str, object] | None:
    runs = run_json_command(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            "pr-00-gate.yml",
            "--branch",
            branch,
            "--limit",
            "1",
            "--json",
            "headSha,status,conclusion,htmlUrl,createdAt",
        ]
    )
    if not isinstance(runs, list) or not runs:
        return None
    latest = runs[0]
    return latest if isinstance(latest, Mapping) else None


def fetch_gate_workflow(repo: str) -> Mapping[str, object]:
    data = run_json_command(
        [
            "gh",
            "workflow",
            "view",
            "pr-00-gate.yml",
            "--repo",
            repo,
            "--json",
            "name,state,path",
        ]
    )
    if not isinstance(data, Mapping):
        raise RuntimeError(f"Unable to resolve Gate workflow metadata for {repo}.")
    return data


def resolve_gate_workflow_state(repo: str) -> str:
    workflow = fetch_gate_workflow(repo)
    state = workflow.get("state")
    if not isinstance(state, str) or not state:
        raise RuntimeError(f"Unable to resolve Gate workflow state for {repo}.")
    return state.strip().lower()


def enable_gate_workflow(repo: str) -> None:
    subprocess.run(
        ["gh", "workflow", "enable", "pr-00-gate.yml", "--repo", repo],
        check=True,
        text=True,
    )


def ensure_gate_workflow_enabled(repo: str) -> str:
    state = resolve_gate_workflow_state(repo)
    if state == "active":
        return "Gate workflow is already active."
    enable_gate_workflow(repo)
    return f"Gate workflow was {state}; enabled via gh workflow enable."


def dispatch_gate(branch: str, repo: str) -> str:
    subprocess.run(
        ["gh", "workflow", "run", "pr-00-gate.yml", "--repo", repo, "--ref", branch],
        check=True,
        text=True,
    )
    return f"gh run list --repo {repo} --workflow pr-00-gate.yml --branch {branch}"


def should_dispatch_gate(
    latest_run: Mapping[str, object] | None, head_sha: str
) -> tuple[bool, str]:
    if latest_run is None:
        return True, "Dispatching Gate workflow: no existing runs found."

    run_head = latest_run.get("headSha")
    if not isinstance(run_head, str):
        return True, "Dispatching Gate workflow: latest run missing head SHA."

    status = str(latest_run.get("status") or "").strip().lower()
    conclusion = str(latest_run.get("conclusion") or "").strip().lower()
    url = latest_run.get("htmlUrl")
    url_suffix = f" ({url})" if isinstance(url, str) and url else ""
    short_sha = head_sha[:7]

    if run_head == head_sha and status in {"queued", "in_progress"}:
        return False, f"Gate workflow already running for {short_sha}{url_suffix}"
    if run_head == head_sha and status == "completed":
        if conclusion:
            return (
                False,
                f"Gate workflow already completed ({conclusion}) for {short_sha}{url_suffix}",
            )
        return False, f"Gate workflow already completed for {short_sha}{url_suffix}"
    if run_head == head_sha and conclusion:
        return False, f"Gate workflow already reported {conclusion} for {short_sha}{url_suffix}"

    return True, "Dispatching Gate workflow: latest run does not match head SHA."


def ensure_gate(pr_number: str, repo: str) -> tuple[bool, str, str, str]:
    branch, head_sha = resolve_pr_info(pr_number, repo)
    latest_run = fetch_latest_gate_run(branch, repo)
    should_dispatch, note = should_dispatch_gate(latest_run, head_sha)
    if not should_dispatch:
        followup = f"gh run list --repo {repo} --workflow pr-00-gate.yml --branch {branch}"
        return False, branch, note, followup
    followup = dispatch_gate(branch, repo)
    return True, branch, note, followup


def trigger_gate(pr_number: str, repo: str) -> tuple[str, str]:
    branch = resolve_branch(pr_number, repo)
    followup = dispatch_gate(branch, repo)
    return branch, followup


def format_gate_status(pr_number: str, repo: str) -> list[str]:
    branch, head_sha = resolve_pr_info(pr_number, repo)
    state = resolve_gate_workflow_state(repo)
    latest_run = fetch_latest_gate_run(branch, repo)
    lines = [
        f"Gate workflow state: {state}",
        f"PR #{pr_number} branch: {branch}",
        f"PR #{pr_number} head: {head_sha}",
    ]
    if latest_run is None:
        lines.append("Latest Gate run: none found")
        return lines

    status = str(latest_run.get("status") or "unknown").strip().lower() or "unknown"
    conclusion = str(latest_run.get("conclusion") or "unknown").strip().lower() or "unknown"
    run_head = str(latest_run.get("headSha") or "unknown").strip() or "unknown"
    created_at = str(latest_run.get("createdAt") or "unknown").strip() or "unknown"
    url = latest_run.get("htmlUrl")
    url_suffix = f" ({url})" if isinstance(url, str) and url else ""
    lines.append(
        f"Latest Gate run: status={status} conclusion={conclusion} head={run_head} created={created_at}{url_suffix}"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trigger the Gate workflow for a PR via GitHub CLI."
    )
    parser.add_argument("pr_number", help="Pull request number.")
    parser.add_argument(
        "repo", nargs="?", default="stranske/Trend_Model_Project", help="owner/repo"
    )
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Only dispatch Gate if no matching run exists for the latest commit.",
    )
    parser.add_argument(
        "--ensure-enabled",
        action="store_true",
        help="Enable Gate workflow if it is disabled before dispatching.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print Gate workflow state and latest run details for the PR.",
    )
    args = parser.parse_args(argv)

    if shutil.which("gh") is None:
        print("gh CLI is required but not found in PATH.", file=sys.stderr)
        return 1

    try:
        if args.status:
            for line in format_gate_status(args.pr_number, args.repo):
                print(line)
            return 0
        if args.ensure_enabled:
            print(ensure_gate_workflow_enabled(args.repo))
        if args.ensure:
            triggered, branch, note, followup = ensure_gate(args.pr_number, args.repo)
        else:
            triggered = True
            branch, followup = trigger_gate(args.pr_number, args.repo)
            note = "Dispatched Gate workflow."
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        return exc.returncode or 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.ensure and not triggered:
        print(
            f"Gate workflow already present for PR #{args.pr_number} (branch: {branch}) in {args.repo}"
        )
        print(note)
    else:
        print(
            f"Triggering pr-00-gate.yml for PR #{args.pr_number} (branch: {branch}) in {args.repo}"
        )
        print(note)
    print("You can monitor runs with:")
    print(f"  {followup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
