#!/usr/bin/env python3
"""Trigger the Gate workflow for a PR via GitHub CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def run_command(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def resolve_branch(pr_number: str, repo: str) -> str:
    output = run_command(
        ["gh", "pr", "view", pr_number, "--repo", repo, "--json", "headRefName", "-q", ".headRefName"]
    )
    if not output:
        raise RuntimeError(f"Unable to resolve branch for PR #{pr_number} in {repo}.")
    return output


def trigger_gate(pr_number: str, repo: str) -> tuple[str, str]:
    branch = resolve_branch(pr_number, repo)
    subprocess.run(
        ["gh", "workflow", "run", "pr-00-gate.yml", "--repo", repo, "--ref", branch],
        check=True,
        text=True,
    )
    return branch, f"gh run list --repo {repo} --workflow pr-00-gate.yml --branch {branch}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trigger the Gate workflow for a PR via GitHub CLI."
    )
    parser.add_argument("pr_number", help="Pull request number.")
    parser.add_argument("repo", nargs="?", default="stranske/Trend_Model_Project", help="owner/repo")
    args = parser.parse_args(argv)

    if shutil.which("gh") is None:
        print("gh CLI is required but not found in PATH.", file=sys.stderr)
        return 1

    try:
        branch, followup = trigger_gate(args.pr_number, args.repo)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        return exc.returncode or 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Triggering pr-00-gate.yml for PR #{args.pr_number} (branch: {branch}) in {args.repo}")
    print("Dispatched. You can monitor runs with:")
    print(f"  {followup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
