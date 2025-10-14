#!/usr/bin/env python3
"""Ensure the default branch requires the Gate workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import requests

DEFAULT_API_ROOT = "https://api.github.com"


def resolve_api_root(explicit: str | None = None) -> str:
    """Return the GitHub API root, normalising trailing slashes."""

    candidate = (explicit or os.getenv("GITHUB_API_URL") or DEFAULT_API_ROOT).strip()
    if not candidate:
        return DEFAULT_API_ROOT
    return candidate.rstrip("/")


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


def _status_checks_url(repo: str, branch: str, *, api_root: str) -> str:
    return (
        f"{api_root}/repos/{repo}/branches/{branch}/protection/required_status_checks"
    )


def _branch_url(repo: str, branch: str, *, api_root: str) -> str:
    return f"{api_root}/repos/{repo}/branches/{branch}"


def _state_from_branch_payload(payload: Mapping[str, Any]) -> StatusCheckState:
    protection = payload.get("protection")
    if not isinstance(protection, Mapping) or not protection.get("enabled"):
        raise BranchProtectionMissingError(
            "Branch protection is disabled for this branch."
        )

    status_checks = protection.get("required_status_checks")
    if not isinstance(status_checks, Mapping):
        raise BranchProtectionMissingError(
            "Branch protection does not require any status checks yet."
        )

    raw_contexts = status_checks.get("contexts") or []
    if isinstance(raw_contexts, Iterable) and not isinstance(
        raw_contexts, (str, bytes)
    ):
        contexts = [str(context) for context in raw_contexts]
    else:
        contexts = []

    # Prefer the 'strict' field if present, for consistency with StatusCheckState.from_api
    if "strict" in status_checks:
        strict = bool(status_checks.get("strict"))
    else:
        enforcement_level = status_checks.get("enforcement_level")
        strict = enforcement_level in {"non_admins", "everyone"}

    return StatusCheckState(strict=strict, contexts=sorted(contexts))


def fetch_status_checks(
    session: requests.Session,
    repo: str,
    branch: str,
    *,
    api_root: str = DEFAULT_API_ROOT,
) -> StatusCheckState:
    response = session.get(
        _status_checks_url(repo, branch, api_root=api_root), timeout=30
    )
    if response.status_code == 404:
        raise BranchProtectionMissingError(
            "Required status checks are not enabled for this branch. Configure the base protection rule first."
        )
    if response.status_code == 403:
        branch_response = session.get(
            _branch_url(repo, branch, api_root=api_root), timeout=30
        )
        if branch_response.status_code == 404:
            raise BranchProtectionMissingError(
                "Branch does not exist; cannot inspect protection status."
            )
        if branch_response.status_code >= 400:
            raise BranchProtectionError(
                "Failed to inspect branch protection via branch endpoint: "
                f"{branch_response.status_code} {branch_response.text}"
            )
        return _state_from_branch_payload(branch_response.json())
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
    api_root: str = DEFAULT_API_ROOT,
) -> StatusCheckState:
    payload: dict[str, Any] = {"contexts": sorted(contexts), "strict": strict}
    response = session.patch(
        _status_checks_url(repo, branch, api_root=api_root),
        json=payload,
        timeout=30,
    )
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
    api_root: str = DEFAULT_API_ROOT,
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
        f"{api_root}/repos/{repo}/branches/{branch}/protection",
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


def require_token(explicit: str | None = None) -> str:
    if explicit:
        candidate = explicit.strip()
        if candidate:
            return candidate

    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise BranchProtectionError(
            "Set the GITHUB_TOKEN (or GH_TOKEN) environment variable with a token that can manage branch protection."
        )
    return token


def _write_snapshot(path: str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


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
        "--token",
        help=(
            "Personal access token to use when calling the GitHub API. Defaults to the GITHUB_TOKEN or GH_TOKEN environment "
            "variables."
        ),
    )
    parser.add_argument(
        "--api-url",
        help=(
            "GitHub API base URL. Defaults to the GITHUB_API_URL environment variable or https://api.github.com when unset."
        ),
    )
    parser.add_argument(
        "--snapshot",
        help="Write a JSON snapshot of the current and desired branch protection state to this file.",
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

    snapshot: dict[str, Any] | None = None
    if args.snapshot:
        now = datetime.now(UTC).replace(microsecond=0)
        snapshot = {
            "repository": args.repo,
            "branch": args.branch,
            "mode": "apply" if args.apply else "check" if args.check else "inspect",
            "generated_at": now.isoformat().replace("+00:00", "Z"),
            "changes_applied": False,
        }

    api_root = resolve_api_root(args.api_url)
    token = require_token(args.token)
    session = _build_session(token)

    try:
        current_state = fetch_status_checks(
            session, args.repo, args.branch, api_root=api_root
        )
        if snapshot is not None:
            snapshot["current"] = {
                "strict": current_state.strict,
                "contexts": list(current_state.contexts),
            }
    except BranchProtectionMissingError:
        print(f"Repository: {args.repo}")
        print(f"Branch:     {args.branch}")
        print("Current contexts: (none)")
        label = "Target contexts" if args.no_clean else "Desired contexts"
        print(f"{label}: {format_contexts(desired_contexts)}")
        print("Current 'require up to date': False")
        print("Desired 'require up to date': True")

        if snapshot is not None:
            snapshot.update(
                {
                    "current": None,
                    "desired": {"strict": True, "contexts": list(desired_contexts)},
                    "changes_required": True,
                }
            )

        if args.apply:
            try:
                created_state = bootstrap_branch_protection(
                    session,
                    args.repo,
                    args.branch,
                    contexts=desired_contexts,
                    strict=True,
                    api_root=api_root,
                )
            except BranchProtectionError as exc:
                print(f"error: {exc}", file=sys.stderr)
                if snapshot is not None:
                    snapshot["error"] = str(exc)
                    _write_snapshot(args.snapshot, snapshot)
                return 1

            print("Created branch protection rule.")
            print(f"New contexts: {format_contexts(created_state.contexts)}")
            print(f"'Require up to date' enabled: {created_state.strict}")
            if snapshot is not None:
                snapshot["changes_applied"] = True
                snapshot["after"] = {
                    "strict": created_state.strict,
                    "contexts": list(created_state.contexts),
                }
                _write_snapshot(args.snapshot, snapshot)
            return 0

        print("Would create branch protection.")
        if snapshot is not None:
            _write_snapshot(args.snapshot, snapshot)
        if args.check:
            return 1
        return 0
    except BranchProtectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if snapshot is not None:
            snapshot["error"] = str(exc)
            snapshot.setdefault(
                "desired", {"strict": True, "contexts": list(desired_contexts)}
            )
            _write_snapshot(args.snapshot, snapshot)
        return 1

    target_contexts = (
        normalise_contexts(list(current_state.contexts) + list(desired_contexts))
        if args.no_clean
        else desired_contexts
    )

    to_add, to_remove = diff_contexts(current_state.contexts, target_contexts)
    strict_change = not current_state.strict

    if snapshot is not None:
        snapshot["desired"] = {"strict": True, "contexts": list(target_contexts)}

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
    changes_required = not no_changes_required

    if snapshot is not None:
        snapshot["changes_required"] = changes_required

    if args.check or not args.apply:
        if no_changes_required:
            print("No changes required.")
            if snapshot is not None:
                _write_snapshot(args.snapshot, snapshot)
            return 0

        if to_add:
            print(f"Would add contexts: {format_contexts(to_add)}")
        if not args.no_clean and to_remove:
            print(f"Would remove contexts: {format_contexts(to_remove)}")
        if strict_change:
            print("Would enable 'require branches to be up to date'.")

        if snapshot is not None:
            _write_snapshot(args.snapshot, snapshot)

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
            api_root=api_root,
        )
    except BranchProtectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if snapshot is not None:
            snapshot["error"] = str(exc)
            snapshot["changes_required"] = True
            _write_snapshot(args.snapshot, snapshot)
        return 1

    print("Update successful.")
    print(f"New contexts: {format_contexts(updated_state.contexts)}")
    print(f"'Require up to date' enabled: {updated_state.strict}")
    if snapshot is not None:
        snapshot["changes_required"] = changes_required
        snapshot["changes_applied"] = bool(args.apply and changes_required)
        snapshot["after"] = {
            "strict": updated_state.strict,
            "contexts": list(updated_state.contexts),
        }
        _write_snapshot(args.snapshot, snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
