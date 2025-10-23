#!/usr/bin/env python3
"""Disable retired GitHub Actions workflows that still appear as active.

This helper is intended for maintenance sweeps such as Issue #2823 where
superseded workflow definitions were deleted from the repository but the
GitHub Actions inventory still lists the legacy entries as *active*. The
script enumerates every workflow via the REST API, compares the set against the
canonical on-disk inventory, and disables any stray definitions. Use the
companion workflow ``maint-47-disable-legacy-workflows.yml`` to run this inside
GitHub Actions with the appropriate permissions.

The default allowlist mirrors ``.github/workflows``. Additional display names
or filenames can be supplied via the ``EXTRA_ALLOWLIST`` environment variable or
command-line ``--allow`` flag when temporary exceptions are required.

Requirements
------------
- ``GITHUB_REPOSITORY`` and ``GITHUB_TOKEN`` environment variables must be set.
- The token needs ``actions: write`` scope to call the disable endpoint.
- Optional: ``GITHUB_STEP_SUMMARY`` for richer reporting inside CI.

Usage
-----
The simplest invocation runs with the defaults::

    python tools/disable_legacy_workflows.py

Provide ``--dry-run`` (or set ``DRY_RUN=true``) to preview the operations
without disabling anything. Extra allowlisted names can be passed via multiple
``--allow`` flags or ``EXTRA_ALLOWLIST`` (comma-separated).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence, Set

API_VERSION = "2022-11-28"

# Canonical workflow inventory (filenames under .github/workflows/)
CANONICAL_WORKFLOW_FILES: Set[str] = {
    "agents-63-chatgpt-issue-sync.yml",
    "agents-63-codex-issue-bridge.yml",
    "agents-64-verify-agent-assignment.yml",
    "agents-70-orchestrator.yml",
    "agents-71-codex-belt-dispatcher.yml",
    "agents-72-codex-belt-worker.yml",
    "agents-73-codex-belt-conveyor.yml",
    "agents-75-keepalive-on-gate.yml",
    "agents-guard.yml",
    "health-40-repo-selfcheck.yml",
    "health-41-repo-health.yml",
    "health-42-actionlint.yml",
    "health-43-ci-signature-guard.yml",
    "health-44-gate-branch-protection.yml",
    "maint-45-cosmetic-repair.yml",
    "maint-46-post-ci.yml",
    "maint-47-disable-legacy-workflows.yml",
    "maint-coverage-guard.yml",
    "maint-keepalive.yml",
    "pr-00-gate.yml",
    "reusable-10-ci-python.yml",
    "reusable-12-ci-docker.yml",
    "reusable-16-agents.yml",
    "reusable-18-autofix.yml",
    "selftest-reusable-ci.yml",
}

# Display names expected to remain active in the Actions UI.
CANONICAL_WORKFLOW_NAMES: Set[str] = {
    "Agents 63 ChatGPT Issue Sync",
    "Agents 63 Codex Issue Bridge",
    "Agents 64 Verify Agent Assignment",
    "Agents 70 Orchestrator",
    "Agents 71 Codex Belt Dispatcher",
    "Agents 72 Codex Belt Worker",
    "Agents 73 Codex Belt Conveyor",
    "Agents 75 Keepalive On Gate",
    "Health 45 Agents Guard",
    "Gate",
    "Health 40 Repo Selfcheck",
    "Health 41 Repo Health",
    "Health 42 Actionlint",
    "Health 43 CI Signature Guard",
    "Health 44 Gate Branch Protection",
    "Maint 45 Cosmetic Repair",
    "Maint 46 Post CI",
    "Maint 47 Disable Legacy Workflows",
    "Maint Coverage Guard",
    "Maint Keepalive Heartbeat",
    "Reusable CI",
    "Reusable Docker Smoke",
    "Reusable 16 Agents",
    "Reusable 18 Autofix",
    "Selftest: Reusables",
}


class WorkflowAPIError(RuntimeError):
    """Raised when the GitHub API call fails."""

    def __init__(
        self,
        *,
        status_code: int,
        reason: str,
        url: str,
        body: str,
    ) -> None:
        super().__init__(
            f"GitHub API request failed ({status_code} {reason}) for {url}: {body}"
        )
        self.status_code = status_code
        self.reason = reason
        self.url = url
        self.body = body


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report targeted workflows without disabling them",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="NAME",
        help="Additional workflow display names or filenames to keep enabled",
    )
    return parser.parse_args(list(argv))


def _build_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "workflow-cleanup-script",
        "X-GitHub-Api-Version": API_VERSION,
    }


def _http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: bytes | None = None,
) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, data=data, method=method)
    for key, value in headers.items():
        request.add_header(key, value)
    if data is not None and "Content-Type" not in headers:
        request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request) as response:
            payload = response.read()
            info = {k: v for k, v in response.info().items()}
            return payload, info
    except urllib.error.HTTPError as exc:  # pragma: no cover - exercised via tests
        body = exc.read().decode("utf-8", errors="replace")
        raise WorkflowAPIError(
            status_code=exc.code,
            reason=str(exc.reason or ""),
            url=url,
            body=body,
        ) from exc


def _extract_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for segment in link_header.split(","):
        parts = [part.strip() for part in segment.split(";")]
        if len(parts) < 2:
            continue
        url_part = parts[0]
        params = parts[1:]
        if any(p.strip() == 'rel="next"' for p in params):
            return url_part.strip("<>")
    return None


def _list_all_workflows(
    base_url: str, headers: dict[str, str]
) -> list[dict[str, object]]:
    workflows: list[dict[str, object]] = []
    url: str | None = f"{base_url}?per_page=100"
    while url:
        payload_bytes, info = _http_request("GET", url, headers=headers)
        payload_raw = json.loads(payload_bytes.decode("utf-8"))
        if isinstance(payload_raw, dict):
            items = payload_raw.get("workflows", [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        workflows.append(item)
        url = _extract_next_link(info.get("Link"))
    return workflows


def _normalize_allowlist(raw_values: Iterable[str]) -> Set[str]:
    values: Set[str] = set()
    for value in raw_values:
        if not value:
            continue
        for piece in value.split(","):
            cleaned = piece.strip()
            if cleaned:
                values.add(cleaned)
    return values


def _workflow_key(workflow: dict[str, object]) -> str:
    name = str(workflow.get("name") or "?").strip()
    path = Path(str(workflow.get("path") or "")).name
    return f"{name} ({path})"


def disable_legacy_workflows(
    *,
    repository: str,
    token: str,
    dry_run: bool = False,
    extra_allow: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Disable workflows that are no longer part of the canonical inventory."""

    headers = _build_headers(token)
    base_url = f"https://api.github.com/repos/{repository}/actions/workflows"
    workflows = _list_all_workflows(base_url, headers)

    extra_allow_set = _normalize_allowlist(extra_allow)
    summary: dict[str, list[str]] = {
        "disabled": [],
        "kept": [],
        "skipped": [],
    }

    for workflow in workflows:
        name = str(workflow.get("name") or "").strip()
        path_name = Path(str(workflow.get("path") or "")).name
        state = str(workflow.get("state") or "").strip()

        allowed = (
            path_name in CANONICAL_WORKFLOW_FILES
            or name in CANONICAL_WORKFLOW_NAMES
            or name in extra_allow_set
            or path_name in extra_allow_set
        )

        key = _workflow_key(workflow)
        if allowed:
            summary["kept"].append(key)
            continue

        if state.lower().startswith("disabled"):
            summary["skipped"].append(key)
            continue

        if dry_run:
            summary["skipped"].append(f"(dry-run) {key}")
            continue

        disable_url = f"{base_url}/{workflow['id']}/disable"
        try:
            _http_request("PUT", disable_url, headers=headers, data=None)
        except WorkflowAPIError as exc:
            if (
                exc.status_code == 422
                and "unable to disable this workflow" in exc.body.lower()
            ):
                summary["skipped"].append(f"(unsupported) {key}")
                continue
            raise
        summary["disabled"].append(key)

    return summary


def _write_summary(summary: dict[str, list[str]]) -> None:
    output: str | None = os.environ.get("GITHUB_STEP_SUMMARY")
    if not output:
        return
    lines = ["### Legacy workflow disable report", ""]
    for category in ("disabled", "kept", "skipped"):
        entries = summary.get(category, [])
        lines.append(f"- **{category.capitalize()}**: {len(entries)}")
        if entries:
            for entry in sorted(entries):
                lines.append(f"  - {entry}")
    lines.append("")
    with Path(output).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository or not token:
        missing = []
        if not repository:
            missing.append("GITHUB_REPOSITORY")
        if not token:
            missing.append("GITHUB_TOKEN")
        print(
            f"Missing required environment variable(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    extra_allow = list(args.allow)
    extra_allow.extend(
        sorted(_normalize_allowlist([os.environ.get("EXTRA_ALLOWLIST", "")]))
    )

    try:
        summary = disable_legacy_workflows(
            repository=repository,
            token=token,
            dry_run=(args.dry_run or os.environ.get("DRY_RUN", "").lower() == "true"),
            extra_allow=extra_allow,
        )
    except WorkflowAPIError as exc:  # pragma: no cover - defensive
        print(str(exc), file=sys.stderr)
        return 1

    disabled = len(summary.get("disabled", []))
    kept = len(summary.get("kept", []))
    skipped = len(summary.get("skipped", []))

    print(
        f"Legacy workflow cleanup complete: disabled={disabled}, kept={kept}, skipped={skipped}"
    )
    for category in ("disabled", "kept", "skipped"):
        for entry in sorted(summary.get(category, [])):
            print(f"[{category}] {entry}")

    _write_summary(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
