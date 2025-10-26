#!/usr/bin/env python3
"""Ensure the default branch requires the Gate and Health 45 Agents Guard workflows."""

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


DEFAULT_CONTEXTS = (
    "Gate / gate",
    "Health 45 Agents Guard / Enforce agents workflow protections",
)

DEFAULT_CONFIG_PATH = Path(".github/config/required-contexts.json")


class BranchProtectionError(RuntimeError):
    """Raised when the GitHub API reports an unrecoverable error."""


class BranchProtectionMissingError(BranchProtectionError):
    """Raised when the repository has no branch protection configured."""


@dataclass
class StatusCheckState:
    strict: bool | None
    contexts: List[str]

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "StatusCheckState":
        return _state_from_status_payload(payload)


def _state_from_status_payload(
    payload: Mapping[str, Any], *, default_strict: bool | None = False
) -> StatusCheckState:
    """Normalise a required status checks payload into a ``StatusCheckState``."""

    raw_contexts = payload.get("contexts")
    if isinstance(raw_contexts, Iterable) and not isinstance(
        raw_contexts, (str, bytes)
    ):
        contexts = [str(context) for context in raw_contexts]
    else:
        contexts = []

    if "strict" in payload:
        strict_value = payload.get("strict")
        strict = None if strict_value is None else bool(strict_value)
    else:
        strict = default_strict

    return StatusCheckState(strict=strict, contexts=sorted(contexts))


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

    # The branch metadata API omits the ``strict`` flag. Treat it as disabled so
    # missing configuration is still surfaced to the caller.
    return _state_from_status_payload(status_checks, default_strict=None)


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
    payload: dict[str, Any] = {
        "contexts": normalise_contexts(contexts),
        "strict": strict,
    }
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
            "contexts": normalise_contexts(contexts),
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


def load_required_contexts(
    config_path: str | os.PathLike[str] | None = None,
) -> List[str]:
    """Return contexts defined in the shared configuration file."""

    candidate = Path(
        config_path or os.getenv("REQUIRED_CONTEXTS_FILE") or DEFAULT_CONFIG_PATH
    )
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

    if isinstance(payload, Mapping):
        contexts_value = payload.get("required_contexts") or payload.get("contexts")
    else:
        contexts_value = payload

    contexts: List[str] = []
    if isinstance(contexts_value, Iterable) and not isinstance(
        contexts_value, (str, bytes)
    ):
        for item in contexts_value:
            if isinstance(item, str):
                candidate_value = item.strip()
                if candidate_value:
                    contexts.append(candidate_value)
    return contexts


def parse_contexts(
    values: Iterable[str] | None, *, config_path: str | os.PathLike[str] | None = None
) -> List[str]:
    if not values:
        contexts = load_required_contexts(config_path)
        return contexts or list(DEFAULT_CONTEXTS)
    cleaned: List[str] = []
    for value in values:
        candidate = value.strip()
        if not candidate:
            continue
        cleaned.append(candidate)
    if not cleaned:
        return list(DEFAULT_CONTEXTS)
    # Preserve duplicates to highlight mistakes during diffing; dedupe later.
    return cleaned


def normalise_contexts(contexts: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for context in contexts:
        if not context or context in seen:
            continue
        seen.add(context)
        ordered.append(context)
    return ordered


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

    to_add: list[str] = []
    seen_add: set[str] = set()
    for context in desired:
        if context in current_set or context in seen_add:
            continue
        seen_add.add(context)
        to_add.append(context)

    to_remove: list[str] = []
    seen_remove: set[str] = set()
    for context in current:
        if context in desired_set or context in seen_remove:
            continue
        seen_remove.add(context)
        to_remove.append(context)

    return to_add, to_remove


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure the default branch requires the Gate and Health 45 Agents Guard workflow"
            " status checks."
        ),
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
        "--config",
        help=(
            "Path to the required contexts configuration file. Defaults to "
            "'.github/config/required-contexts.json' when present."
        ),
    )
    parser.add_argument(
        "--context",
        dest="contexts",
        action="append",
        help=(
            "Status check context to require. May be passed multiple times. Defaults to"
            " 'Gate / gate' and 'Health 45 Agents Guard / Enforce agents workflow protections'."
        ),
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
        "--require-strict",
        action="store_true",
        help=(
            "Treat an unknown 'require branches to be up to date' setting as drift."
            " Combine with --check to fail when the workflow token cannot confirm"
            " strict enforcement."
        ),
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

    desired_contexts = normalise_contexts(
        parse_contexts(args.contexts, config_path=args.config)
    )

    snapshot: dict[str, Any] | None = None
    if args.snapshot:
        now = datetime.now(UTC).replace(microsecond=0)
        snapshot = {
            "repository": args.repo,
            "branch": args.branch,
            "mode": "apply" if args.apply else "check" if args.check else "inspect",
            "generated_at": now.isoformat().replace("+00:00", "Z"),
            "changes_applied": False,
            "require_strict": bool(args.require_strict),
            "no_clean": bool(args.no_clean),
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
            snapshot["require_strict"] = bool(args.require_strict)
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
                    "require_strict": bool(args.require_strict),
                    "strict_unknown": False,
                }
            )
            snapshot["to_add"] = list(desired_contexts)
            snapshot["to_remove"] = []

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
            snapshot.setdefault("to_add", list(desired_contexts))
            snapshot.setdefault("to_remove", [])
            _write_snapshot(args.snapshot, snapshot)
        return 1

    target_contexts = (
        normalise_contexts(list(desired_contexts) + list(current_state.contexts))
        if args.no_clean
        else desired_contexts
    )

    to_add, to_remove = diff_contexts(current_state.contexts, target_contexts)
    strict_is_unknown = current_state.strict is None
    strict_change = current_state.strict is False

    if snapshot is not None:
        snapshot["desired"] = {"strict": True, "contexts": list(target_contexts)}
        snapshot["strict_unknown"] = strict_is_unknown
        snapshot["require_strict"] = bool(args.require_strict)
        snapshot["to_add"] = list(to_add)
        snapshot["to_remove"] = list(to_remove)

    print(f"Repository: {args.repo}")
    print(f"Branch:     {args.branch}")
    print(f"Current contexts: {format_contexts(current_state.contexts)}")
    label = "Target contexts" if args.no_clean else "Desired contexts"
    print(f"{label}: {format_contexts(target_contexts)}")
    if strict_is_unknown:
        print(
            "Current 'require up to date': (unknown - supply BRANCH_PROTECTION_TOKEN to verify)"
        )
    else:
        print(f"Current 'require up to date': {current_state.strict}")
    print("Desired 'require up to date': True")

    if strict_is_unknown:
        if args.require_strict:
            print(
                "Strict enforcement could not be confirmed and --require-strict was "
                "requested; treating this as drift. Supply BRANCH_PROTECTION_TOKEN "
                "to verify with admin scope."
            )
        else:
            print(
                "Strict enforcement could not be confirmed with the default token. "
                "The check will pass, but rerun with BRANCH_PROTECTION_TOKEN to audit."
            )

    if args.require_strict and strict_is_unknown:
        strict_change = True

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
