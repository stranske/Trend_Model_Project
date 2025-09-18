#!/usr/bin/env python3
"""Repository health self-check helper.

This script verifies that critical repository automation prerequisites are in
place.  It is intended to be executed from a scheduled GitHub Actions workflow
and will open (or update) an actionable issue whenever a requirement is
missing.

The checks performed are:

* `SERVICE_BOT_PAT` secret exists and can author (and delete) a commit comment.
* Required repository labels are present (`agent:codex`, `agent:copilot`, and at
  least one `risk:*` label).
* The default branch has branch protection enabled with a required "gate"
  status check.

When checks fail the script creates or updates an issue titled "Repository
health self-check failure".  When the checks later pass the issue is closed.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests


API_VERSION = "2022-11-28"
DEFAULT_TIMEOUT = 15
ISSUE_TITLE = "Repository health self-check failure"


class CheckError(RuntimeError):
    """Raised when a check fails in an unexpected way."""


@dataclass
class CheckOutcome:
    """Represents the result of running a repository health check."""

    ok: bool
    details: Optional[str] = None


def _now_utc() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _build_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
    }


def _gh_request(
    method: str,
    url: str,
    *,
    token: str,
    expected: Iterable[int] | int,
    session: requests.Session,
    **kwargs,
) -> requests.Response:
    if isinstance(expected, int):
        expected_set = {expected}
    else:
        expected_set = set(expected)
    headers = kwargs.pop("headers", {})
    headers = {**_build_headers(token), **headers}
    response = session.request(
        method,
        url,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
        **kwargs,
    )
    if response.status_code not in expected_set:
        raise CheckError(
            f"GitHub API call to {url} failed: {response.status_code} {response.text.strip()}"
        )
    return response


def _gather_paginated(
    url: str,
    *,
    token: str,
    session: requests.Session,
) -> List[dict]:
    results: List[dict] = []
    next_url: Optional[str] = url
    while next_url:
        response = _gh_request("GET", next_url, token=token, expected=200, session=session)
        results.extend(response.json())
        next_url = response.links.get("next", {}).get("url")
    return results


def _check_service_bot(
    *,
    repo: str,
    branch: str,
    service_bot_pat: Optional[str],
    github_token: str,
    session: requests.Session,
    api_base: str,
) -> CheckOutcome:
    if not service_bot_pat:
        return CheckOutcome(False, "Secret SERVICE_BOT_PAT is not configured.")

    try:
        _gh_request(
            "GET",
            f"{api_base}/user",
            token=service_bot_pat,
            expected=200,
            session=session,
        )
    except CheckError as exc:  # pragma: no cover - error handling
        return CheckOutcome(False, f"SERVICE_BOT_PAT is invalid: {exc}")

    # Discover the latest commit on the default branch.
    commit_resp = _gh_request(
        "GET",
        f"{api_base}/repos/{repo}/commits/{branch}",
        token=github_token,
        expected=200,
        session=session,
    )
    commit_sha = commit_resp.json()["sha"]

    comment_resp = _gh_request(
        "POST",
        f"{api_base}/repos/{repo}/commits/{commit_sha}/comments",
        token=service_bot_pat,
        expected=201,
        session=session,
        json={"body": "Repository health self-check token probe."},
    )
    comment_id = comment_resp.json().get("id")
    if not comment_id:
        return CheckOutcome(False, "SERVICE_BOT_PAT comment probe did not return a comment id.")

    _gh_request(
        "DELETE",
        f"{api_base}/repos/{repo}/comments/{comment_id}",
        token=service_bot_pat,
        expected={200, 202, 204, 404},
        session=session,
    )

    return CheckOutcome(True)


def _check_labels(
    *,
    repo: str,
    github_token: str,
    session: requests.Session,
    api_base: str,
) -> CheckOutcome:
    labels = _gather_paginated(
        f"{api_base}/repos/{repo}/labels?per_page=100",
        token=github_token,
        session=session,
    )
    label_names = {label["name"].lower() for label in labels}
    missing = []
    for required in ("agent:codex", "agent:copilot"):
        if required.lower() not in label_names:
            missing.append(f"Missing required label '{required}'.")
    if not any(name.startswith("risk:") for name in label_names):
        missing.append("Missing a 'risk:*' label (e.g. risk:low).")

    if missing:
        return CheckOutcome(False, " ".join(missing))
    return CheckOutcome(True)


def _check_branch_protection(
    *,
    repo: str,
    branch: str,
    github_token: str,
    session: requests.Session,
    api_base: str,
) -> CheckOutcome:
    try:
        response = _gh_request(
            "GET",
            f"{api_base}/repos/{repo}/branches/{branch}/protection",
            token=github_token,
            expected={200, 403, 404},
            session=session,
        )
    except CheckError as exc:  # pragma: no cover - error handling, not logging
        return CheckOutcome(False, f"Unable to inspect branch protection: {exc}")

    if response.status_code == 404:
        return CheckOutcome(False, f"Branch protection is not enabled for {branch}.")
    if response.status_code == 403:
        return CheckOutcome(False, "GITHUB_TOKEN lacks access to branch protection settings.")

    data = response.json()
    required_status_checks = data.get("required_status_checks") or {}
    contexts = required_status_checks.get("contexts") or []
    checks = required_status_checks.get("checks") or []

    def _has_gate(value: str) -> bool:
        return "gate" in value.lower()

    gate_in_contexts = any(_has_gate(ctx) for ctx in contexts)
    gate_in_checks = any(_has_gate(item.get("context", "")) for item in checks)

    if not (gate_in_contexts or gate_in_checks):
        return CheckOutcome(
            False,
            f"Branch protection for {branch} does not require the gate status check.",
        )
    return CheckOutcome(True)


def _discover_default_branch(
    *,
    repo: str,
    github_token: str,
    session: requests.Session,
    api_base: str,
) -> str:
    repo_resp = _gh_request(
        "GET",
        f"{api_base}/repos/{repo}",
        token=github_token,
        expected=200,
        session=session,
    )
    return repo_resp.json()["default_branch"]


def _find_open_issue(
    *,
    repo: str,
    github_token: str,
    session: requests.Session,
    api_base: str,
) -> Optional[dict]:
    url = f"{api_base}/repos/{repo}/issues?state=open&per_page=100"
    issues = _gather_paginated(url, token=github_token, session=session)
    for issue in issues:
        if issue.get("pull_request"):
            continue
        if issue.get("title") == ISSUE_TITLE:
            return issue
    return None


def _create_issue(
    *,
    repo: str,
    github_token: str,
    session: requests.Session,
    api_base: str,
    body: str,
) -> None:
    _gh_request(
        "POST",
        f"{api_base}/repos/{repo}/issues",
        token=github_token,
        expected=201,
        session=session,
        json={"title": ISSUE_TITLE, "body": body},
    )


def _post_issue_comment(
    *,
    repo: str,
    issue_number: int,
    github_token: str,
    session: requests.Session,
    api_base: str,
    body: str,
) -> None:
    _gh_request(
        "POST",
        f"{api_base}/repos/{repo}/issues/{issue_number}/comments",
        token=github_token,
        expected=201,
        session=session,
        json={"body": body},
    )


def _close_issue(
    *,
    repo: str,
    issue_number: int,
    github_token: str,
    session: requests.Session,
    api_base: str,
) -> None:
    _gh_request(
        "PATCH",
        f"{api_base}/repos/{repo}/issues/{issue_number}",
        token=github_token,
        expected=200,
        session=session,
        json={"state": "closed"},
    )


def _format_issue_body(failures: List[str], run_url: Optional[str]) -> str:
    lines = ["Repository health self-check detected the following issues:", ""]
    lines.extend(f"- {failure}" for failure in failures)
    lines.append("")
    lines.append(f"Timestamp: {_now_utc()}")
    if run_url:
        lines.append(f"Workflow run: {run_url}")
    return "\n".join(lines)


def _format_success_comment(run_url: Optional[str]) -> str:
    lines = [
        "Repository health self-check is passing again.",
        f"Timestamp: {_now_utc()}",
    ]
    if run_url:
        lines.append(f"Workflow run: {run_url}")
    return "\n".join(lines)


def _format_failure_comment(failures: List[str], run_url: Optional[str]) -> str:
    lines = ["Repository health self-check is still failing with:", ""]
    lines.extend(f"- {failure}" for failure in failures)
    lines.append("")
    lines.append(f"Timestamp: {_now_utc()}")
    if run_url:
        lines.append(f"Workflow run: {run_url}")
    return "\n".join(lines)


def main() -> int:
    api_base = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY")
    github_token = os.environ.get("GITHUB_TOKEN")
    service_bot_pat = os.environ.get("SERVICE_BOT_PAT")

    if not repo:
        print("GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2
    if not github_token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2

    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    run_id = os.environ.get("GITHUB_RUN_ID")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    run_url: Optional[str] = None
    if run_id and run_attempt:
        run_url = f"{server_url}/{repo}/actions/runs/{run_id}/attempts/{run_attempt}"
    elif run_id:
        run_url = f"{server_url}/{repo}/actions/runs/{run_id}"

    session = requests.Session()

    try:
        branch = os.environ.get("DEFAULT_BRANCH")
        if not branch:
            branch = _discover_default_branch(
                repo=repo,
                github_token=github_token,
                session=session,
                api_base=api_base,
            )

        outcomes = []
        service_result = _check_service_bot(
            repo=repo,
            branch=branch,
            service_bot_pat=service_bot_pat,
            github_token=github_token,
            session=session,
            api_base=api_base,
        )
        outcomes.append(("SERVICE_BOT_PAT", service_result))

        label_result = _check_labels(
            repo=repo,
            github_token=github_token,
            session=session,
            api_base=api_base,
        )
        outcomes.append(("labels", label_result))

        protection_result = _check_branch_protection(
            repo=repo,
            branch=branch,
            github_token=github_token,
            session=session,
            api_base=api_base,
        )
        outcomes.append(("branch protection", protection_result))

        failures: List[str] = []
        for check_name, outcome in outcomes:
            if not outcome.ok:
                detail = outcome.details or "Unknown failure"
                failures.append(f"{check_name}: {detail}")

        existing_issue = _find_open_issue(
            repo=repo,
            github_token=github_token,
            session=session,
            api_base=api_base,
        )

        if failures:
            body = _format_issue_body(failures, run_url)
            if existing_issue:
                _post_issue_comment(
                    repo=repo,
                    issue_number=existing_issue["number"],
                    github_token=github_token,
                    session=session,
                    api_base=api_base,
                    body=_format_failure_comment(failures, run_url),
                )
            else:
                _create_issue(
                    repo=repo,
                    github_token=github_token,
                    session=session,
                    api_base=api_base,
                    body=body,
                )
            print("Self-check detected failures:")
            for failure in failures:
                print(f" - {failure}")
            return 1

        if existing_issue:
            _post_issue_comment(
                repo=repo,
                issue_number=existing_issue["number"],
                github_token=github_token,
                session=session,
                api_base=api_base,
                body=_format_success_comment(run_url),
            )
            _close_issue(
                repo=repo,
                issue_number=existing_issue["number"],
                github_token=github_token,
                session=session,
                api_base=api_base,
            )

        print("Repository health self-check passed.")
        return 0

    except CheckError as exc:
        print(f"Repository health self-check failed with API error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
