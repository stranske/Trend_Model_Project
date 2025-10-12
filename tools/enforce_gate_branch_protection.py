#!/usr/bin/env python3
"""Ensure the default branch requires the Gate workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence

API_ROOT = os.getenv("GITHUB_API_URL", "https://api.github.com")
DEFAULT_CONTEXT = "Gate / gate"


class BranchProtectionError(RuntimeError):
    """Raised when the GitHub API reports an unrecoverable error."""


class BranchProtectionMissingError(BranchProtectionError):
    """Raised when no branch protection rule is configured."""


@dataclass
class HttpResponse:
    status_code: int
    text: str
    _payload: dict[str, object] | None = None

    def json(self) -> dict[str, object]:
        if self._payload is None:
            if not self.text:
                self._payload = {}
            else:
                try:
                    self._payload = json.loads(self.text)
                except json.JSONDecodeError:
                    self._payload = {}
        return self._payload


class _ResponseProtocol(Protocol):
    status_code: int
    text: str

    def json(self) -> dict: ...


class _SessionProtocol(Protocol):
    def get(self, url: str, *, timeout: float | None = None) -> _ResponseProtocol: ...

    def patch(
        self, url: str, *, json: dict[str, object], timeout: float | None = None
    ) -> _ResponseProtocol: ...

    def put(
        self, url: str, *, json: dict[str, object], timeout: float | None = None
    ) -> _ResponseProtocol: ...


class BranchProtectionSession:
    """Minimal HTTP client for the GitHub branch protection API."""

    def __init__(self, token: str) -> None:
        self._token = token

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        request = urllib.request.Request(url=url, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("User-Agent", "trend-model-branch-protection")

        data_bytes: bytes | None = None
        if payload is not None:
            data_bytes = json.dumps(payload).encode("utf-8")
            request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(
                request, data=data_bytes, timeout=timeout or 30
            ) as response:
                raw = response.read().decode("utf-8")
                status_code = response.getcode()
        except urllib.error.HTTPError as exc:  # pragma: no cover - network failure path
            raw = exc.read().decode("utf-8", errors="replace")
            status_code = exc.code
        except urllib.error.URLError as exc:  # pragma: no cover - network failure path
            reason = getattr(exc, "reason", exc)
            raise BranchProtectionError(
                f"Network error communicating with GitHub: {reason}"
            ) from exc

        payload_dict: dict[str, object]
        if raw:
            try:
                payload_dict = json.loads(raw)
            except json.JSONDecodeError:
                payload_dict = {}
        else:
            payload_dict = {}

        return HttpResponse(status_code=status_code, text=raw, _payload=payload_dict)

    def get(self, url: str, *, timeout: float | None = None) -> HttpResponse:
        return self._request("GET", url, timeout=timeout)

    def patch(
        self, url: str, *, json: dict[str, object], timeout: float | None = None
    ) -> HttpResponse:
        return self._request("PATCH", url, payload=json, timeout=timeout)

    def put(
        self, url: str, *, json: dict[str, object], timeout: float | None = None
    ) -> HttpResponse:
        return self._request("PUT", url, payload=json, timeout=timeout)


@dataclass
class StatusCheckState:
    strict: bool
    contexts: List[str]

    @classmethod
    def from_api(cls, payload: dict[str, object]) -> "StatusCheckState":
        raw_contexts = payload.get("contexts")
        contexts: list[str]
        if isinstance(raw_contexts, list):
            contexts = [
                context
                for context in raw_contexts
                if isinstance(context, str) and context
            ]
        else:
            contexts = []

        if not contexts and isinstance(payload.get("checks"), list):
            contexts = [
                check.get("context")
                for check in payload["checks"]
                if isinstance(check, dict)
                and isinstance(check.get("context"), str)
                and check.get("context")
            ]

        return cls(strict=bool(payload.get("strict")), contexts=sorted(contexts))


def _build_session(token: str) -> BranchProtectionSession:
    return BranchProtectionSession(token)


def _status_checks_url(repo: str, branch: str) -> str:
    return (
        f"{API_ROOT}/repos/{repo}/branches/{branch}/protection/required_status_checks"
    )


def _protection_url(repo: str, branch: str) -> str:
    return f"{API_ROOT}/repos/{repo}/branches/{branch}/protection"


def fetch_status_checks(
    session: _SessionProtocol, repo: str, branch: str
) -> StatusCheckState:
    response = session.get(_status_checks_url(repo, branch), timeout=30)
    if response.status_code == 404:
        raise BranchProtectionMissingError(
            "Required status checks are not enabled for this branch."
        )
    if response.status_code >= 400:
        raise BranchProtectionError(
            f"Failed to fetch status checks for {branch}: {response.status_code} {response.text}"
        )
    return StatusCheckState.from_api(response.json())


def update_status_checks(
    session: _SessionProtocol,
    repo: str,
    branch: str,
    *,
    contexts: Sequence[str],
    strict: bool,
) -> StatusCheckState:
    payload = {"contexts": sorted(contexts), "strict": strict}
    response = session.patch(_status_checks_url(repo, branch), json=payload, timeout=30)
    if response.status_code >= 400:
        raise BranchProtectionError(
            f"Failed to update status checks for {branch}: {response.status_code} {response.text}"
        )
    return StatusCheckState.from_api(response.json())


def bootstrap_branch_protection(
    session: _SessionProtocol,
    repo: str,
    branch: str,
    *,
    contexts: Sequence[str],
    strict: bool,
) -> StatusCheckState:
    payload = {
        "required_status_checks": {
            "strict": strict,
            "contexts": sorted(contexts),
        },
        "enforce_admins": False,
        "required_pull_request_reviews": None,
        "restrictions": None,
    }
    response = session.put(_protection_url(repo, branch), json=payload, timeout=30)
    if response.status_code >= 400:
        raise BranchProtectionError(
            f"Failed to bootstrap branch protection for {branch}: {response.status_code} {response.text}"
        )

    data = response.json()
    checks = data.get("required_status_checks") if isinstance(data, dict) else {}
    contexts_payload: Sequence[str] = []
    if isinstance(checks, dict):
        contexts_payload = checks.get("contexts") or []
        if not contexts_payload and checks.get("checks"):
            contexts_payload = [
                check.get("context")
                for check in checks["checks"]
                if isinstance(check, dict) and check.get("context")
            ]
        strict_value = checks.get("strict", strict)
    else:
        strict_value = strict

    parsed_contexts = normalise_contexts(contexts_payload or list(contexts))
    return StatusCheckState(strict=bool(strict_value), contexts=parsed_contexts)


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
        help=(
            "Exit with status 1 when the branch protection rule drifts from the desired"
            " configuration. Implies a dry run."
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

    if not args.repo:
        parser.error("--repo is required when GITHUB_REPOSITORY is not set.")
    if args.apply and args.check:
        parser.error("--check cannot be combined with --apply.")

    desired_contexts = normalise_contexts(parse_contexts(args.contexts))

    try:
        token = require_token()
        session = _build_session(token)
        missing_protection = False
        try:
            current_state = fetch_status_checks(session, args.repo, args.branch)
        except BranchProtectionMissingError:
            missing_protection = True
            current_state = StatusCheckState(strict=False, contexts=[])
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

    if not args.apply:
        drift_detected = bool(
            to_add
            or (not args.no_clean and to_remove)
            or strict_change
            or missing_protection
        )
        if not drift_detected:
            print("No changes required.")
        else:
            if to_add:
                print(f"Would add contexts: {format_contexts(to_add)}")
            if not args.no_clean and to_remove:
                print(f"Would remove contexts: {format_contexts(to_remove)}")
            if strict_change:
                print("Would enable 'require branches to be up to date'.")
            if missing_protection:
                print("Would create branch protection with the desired settings.")
            print("Re-run with --apply to enforce the configuration.")
        return 1 if args.check and drift_detected else 0

    try:
        if missing_protection:
            updated_state = bootstrap_branch_protection(
                session,
                args.repo,
                args.branch,
                contexts=target_contexts,
                strict=True,
            )
        else:
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

    if missing_protection:
        print("Created branch protection rule.")
    else:
        print("Update successful.")
    print(f"New contexts: {format_contexts(updated_state.contexts)}")
    print(f"'Require up to date' enabled: {updated_state.strict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
