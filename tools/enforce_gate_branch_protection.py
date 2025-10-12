#!/usr/bin/env python3
"""Ensure the default branch requires the Gate workflow."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence

import requests

API_ROOT = os.getenv("GITHUB_API_URL", "https://api.github.com")
DEFAULT_CONTEXT = "Gate / gate"


class BranchProtectionError(RuntimeError):
    """Raised when the GitHub API reports an unrecoverable error."""


class BranchProtectionMissingError(BranchProtectionError):
    """Raised when the repository has no branch protection configured."""


@dataclass
class StatusCheckState:
    strict: bool
    contexts: List[str]

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "StatusCheckState":
        raw_contexts = payload.get("contexts")
        if isinstance(raw_contexts, Iterable) and not isinstance(
            raw_contexts, (str, bytes)
        ):
            contexts = [str(context) for context in raw_contexts]
        else:
            contexts = []
        return cls(strict=bool(payload.get("strict")), contexts=sorted(contexts))


def _build_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "trend-model-branch-protection",
        }
    )
    return session


def _status_checks_url(repo: str, branch: str) -> str:
    return (
        f"{API_ROOT}/repos/{repo}/branches/{branch}/protection/required_status_checks"
    )


def fetch_status_checks(
    session: requests.Session, repo: str, branch: str
) -> StatusCheckState:
    response = session.get(_status_checks_url(repo, branch), timeout=30)
    if response.status_code == 404:
        raise BranchProtectionMissingError(
            "Required status checks are not enabled for this branch. Configure the base protection rule first."
        )
    if response.status_code >= 400:
        raise BranchProtectionError(
            f"Failed to fetch status checks for {branch}: {response.status_code} {response.text}"
        )
    return StatusCheckState.from_api(response.json())


def update_status_checks(
    session: requests.Session,
    repo: str,
    branch: str,
    *,
    contexts: Sequence[str],
    strict: bool,
) -> StatusCheckState:
    payload: dict[str, Any] = {"contexts": sorted(contexts), "strict": strict}
    response = session.patch(_status_checks_url(repo, branch), json=payload, timeout=30)
    if response.status_code >= 400:
        raise BranchProtectionError(
            f"Failed to update status checks for {branch}: {response.status_code} {response.text}"
        )
    return StatusCheckState.from_api(response.json())


def bootstrap_branch_protection(
    session: requests.Session,
    repo: str,
    branch: str,
    *,
    contexts: Sequence[str],
    strict: bool,
) -> StatusCheckState:
    payload: dict[str, Any] = {
        "required_status_checks": {
            "strict": strict,
            "contexts": sorted(set(contexts)),
        },
        "enforce_admins": True,
        "required_pull_request_reviews": None,
        "restrictions": None,
        "required_linear_history": True,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "lock_branch": False,
        "allow_fork_syncing": True,
        "required_conversation_resolution": True,
    }

    response = session.put(
        f"{API_ROOT}/repos/{repo}/branches/{branch}/protection",
        json=payload,
        timeout=30,
    )

    if response.status_code >= 400:
        raise BranchProtectionError(
            f"Failed to create branch protection for {branch}: {response.status_code} {response.text}"
        )

    data: Mapping[str, Any] = {}
    if response.content:
        parsed = response.json()
        if isinstance(parsed, Mapping):
            data = parsed

    status_payload = data.get("required_status_checks")
    if not isinstance(status_payload, Mapping):
        status_payload = payload["required_status_checks"]
    return StatusCheckState.from_api(status_payload)


def parse_contexts(values: Iterable[str] | None) -> List[str]:
    if not values:
        return [DEFAULT_CONTEXT]
    cleaned: List[str] = []
    for value in values:
        candidate = value.strip()
        if not candidate:
            continue
        cleaned.append(candidate)
    if not cleaned:
        return [DEFAULT_CONTEXT]
    # Preserve duplicates to highlight mistakes during diffing; dedupe later.
    return cleaned


def normalise_contexts(contexts: Sequence[str]) -> List[str]:
    unique = {context for context in contexts if context}
    return sorted(unique)


def format_contexts(contexts: Sequence[str]) -> str:
    if not contexts:
        return "(none)"
    return ", ".join(contexts)


def require_token() -> str:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise BranchProtectionError(
            "Set the GITHUB_TOKEN (or GH_TOKEN) environment variable with a token that can manage branch protection."
        )
    return token


def diff_contexts(
    current: Sequence[str], desired: Sequence[str]
) -> tuple[list[str], list[str]]:
    current_set = set(current)
    desired_set = set(desired)
    to_add = sorted(desired_set - current_set)
    to_remove = sorted(current_set - desired_set)
    return to_add, to_remove


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ensure the default branch requires the Gate workflow status check.",
    )
    parser.add_argument(
        "--repo",
        default=os.getenv("GITHUB_REPOSITORY"),
        help="Repository in the form owner/name. Defaults to the GITHUB_REPOSITORY env var.",
    )
    parser.add_argument(
        "--branch",
        default=os.getenv("DEFAULT_BRANCH", "main"),
        help="Branch to inspect/update. Defaults to DEFAULT_BRANCH or 'main'.",
    )
    parser.add_argument(
        "--context",
        dest="contexts",
        action="append",
        help="Status check context to require. May be passed multiple times. Defaults to 'Gate / gate'.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the changes instead of performing a dry run.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with a non-zero status if changes would be required without applying them.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help=(
            "Keep additional required contexts instead of pruning them. The default behaviour removes contexts that do not match"
            " the provided list."
        ),
    )

    args = parser.parse_args(argv)

    if args.apply and args.check:
        parser.error("--check cannot be combined with --apply.")

    if not args.repo:
        parser.error("--repo is required when GITHUB_REPOSITORY is not set.")

    desired_contexts = normalise_contexts(parse_contexts(args.contexts))

    token = require_token()
    session = _build_session(token)

    try:
        current_state = fetch_status_checks(session, args.repo, args.branch)
    except BranchProtectionMissingError:
        print(f"Repository: {args.repo}")
        print(f"Branch:     {args.branch}")
        print("Current contexts: (none)")
        label = "Target contexts" if args.no_clean else "Desired contexts"
        print(f"{label}: {format_contexts(desired_contexts)}")
        print("Current 'require up to date': False")
        print("Desired 'require up to date': True")

        if args.apply:
            try:
                created_state = bootstrap_branch_protection(
                    session,
                    args.repo,
                    args.branch,
                    contexts=desired_contexts,
                    strict=True,
                )
            except BranchProtectionError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1

            print("Created branch protection rule.")
            print(f"New contexts: {format_contexts(created_state.contexts)}")
            print(f"'Require up to date' enabled: {created_state.strict}")
            return 0

        print("Would create branch protection.")
        if args.check:
            return 1
        return 0
    except BranchProtectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    target_contexts = (
        normalise_contexts(list(current_state.contexts) + list(desired_contexts))
        if args.no_clean
        else desired_contexts
    )

    to_add, to_remove = diff_contexts(current_state.contexts, target_contexts)
    strict_change = not current_state.strict

    print(f"Repository: {args.repo}")
    print(f"Branch:     {args.branch}")
    print(f"Current contexts: {format_contexts(current_state.contexts)}")
    label = "Target contexts" if args.no_clean else "Desired contexts"
    print(f"{label}: {format_contexts(target_contexts)}")
    print(f"Current 'require up to date': {current_state.strict}")
    print("Desired 'require up to date': True")

    no_changes_required = (
        not to_add and (args.no_clean or not to_remove) and not strict_change
    )

    if args.check or not args.apply:
        if no_changes_required:
            print("No changes required.")
            return 0

        if to_add:
            print(f"Would add contexts: {format_contexts(to_add)}")
        if not args.no_clean and to_remove:
            print(f"Would remove contexts: {format_contexts(to_remove)}")
        if strict_change:
            print("Would enable 'require branches to be up to date'.")

        if args.check:
            return 1

        print("Re-run with --apply to enforce the configuration.")
        return 0

    try:
        updated_state = update_status_checks(
            session,
            args.repo,
            args.branch,
            contexts=target_contexts,
            strict=True,
        )
    except BranchProtectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Update successful.")
    print(f"New contexts: {format_contexts(updated_state.contexts)}")
    print(f"'Require up to date' enabled: {updated_state.strict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
