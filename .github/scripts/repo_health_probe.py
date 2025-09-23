#!/usr/bin/env python3
"""Lightweight repository health probe used by the nightly workflow.

The script intentionally keeps runtime dependencies to the Python standard
library so it can execute in a fresh GitHub Actions runner without any
additional setup.  Results are emitted as JSON so downstream workflow steps can
render both a Markdown summary and actionable failure messages.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from http.client import HTTPResponse
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GITHUB_API = os.environ.get("GITHUB_API_URL", "https://api.github.com")


@dataclass
class CheckResult:
    """Represents the status of a single probe."""

    name: str
    ok: bool
    description: str
    details: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "description": self.description,
            "details": self.details,
        }


def _auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "repo-health-probe",
    }


def _next_link(links: Optional[str]) -> Optional[str]:
    if not links:
        return None
    for part in links.split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        url_part, rel_part = section[0].strip(), section[1].strip()
        if rel_part == 'rel="next"':
            return url_part.strip("<>")
    return None


def _paginated_get(url: str, token: str) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    headers = _auth_headers(token)
    next_url: Optional[str] = url
    while next_url:
        req = Request(next_url, headers=headers)
        response = cast(HTTPResponse, urlopen(req, timeout=10))
        try:
            payload = response.read().decode("utf-8")
            data: Any = json.loads(payload)
            link = response.headers.get("Link")
        finally:
            response.close()

        if isinstance(data, dict):
            # GitHub secrets/variables APIs wrap results in an envelope.
            if "secrets" in data:
                results.extend(cast(Sequence[Dict[str, object]], data.get("secrets", [])))
            elif "variables" in data:
                results.extend(cast(Sequence[Dict[str, object]], data.get("variables", [])))
            else:
                # Unexpected payload – normalise to list where possible.
                results.append(cast(Dict[str, object], data))  # pragma: no cover - defensive branch.
        elif isinstance(data, list):
            results.extend(cast(Sequence[Dict[str, object]], data))
        else:  # pragma: no cover - defensive branch.
            results.append({"value": data})
        next_url = _next_link(link)
    return results


def _collect_labels(repo: str, token: str) -> List[str]:
    base_url = f"{GITHUB_API}/repos/{repo}/labels?per_page=100"
    payload = _paginated_get(base_url, token)
    names: List[str] = []
    for entry in payload:
        name_value = entry.get("name")
        names.append(name_value if isinstance(name_value, str) else "")
    return names


def _collect_secrets(repo: str, token: str) -> List[str]:
    base_url = f"{GITHUB_API}/repos/{repo}/actions/secrets"
    payload = _paginated_get(base_url, token)
    names: List[str] = []
    for entry in payload:
        name_value = entry.get("name")
        names.append(name_value if isinstance(name_value, str) else "")
    return names


def _collect_variables(repo: str, token: str) -> List[str]:
    base_url = f"{GITHUB_API}/repos/{repo}/actions/variables"
    payload = _paginated_get(base_url, token)
    names: List[str] = []
    for entry in payload:
        name_value = entry.get("name")
        names.append(name_value if isinstance(name_value, str) else "")
    return names


def _label_checks(labels: Iterable[str]) -> List[CheckResult]:
    names = list(labels)
    checks = [
        (
            "agent-prefix",
            "At least one `agent:*` label exists",
            any(name.startswith("agent:") for name in names),
            "Add labels such as `agent:codex` or `agent:copilot`.",
        ),
        (
            "priority-prefix",
            "At least one `priority:*` label exists",
            any(name.startswith("priority:") for name in names),
            "Create priority labels like `priority:p0`.",
        ),
        (
            "workflows-label",
            "`workflows` label exists",
            "workflows" in names,
            "Add a `workflows` label for automation triage.",
        ),
        (
            "tech-coverage-label",
            "`tech:coverage` label exists",
            "tech:coverage" in names,
            "Add the `tech:coverage` label used by coverage tooling.",
        ),
    ]
    return [
        CheckResult(name=slug, ok=is_ok, description=desc, details=None if is_ok else guidance)
        for slug, desc, is_ok, guidance in checks
    ]


def _secret_checks(secrets: Iterable[str]) -> List[CheckResult]:
    names = set(secrets)
    checks = [
        (
            "service-bot-pat",
            "Repository secret `SERVICE_BOT_PAT` is configured",
            "SERVICE_BOT_PAT" in names,
            "Add `SERVICE_BOT_PAT` so automation can authenticate as the service bot.",
        ),
    ]
    return [
        CheckResult(name=slug, ok=is_ok, description=desc, details=None if is_ok else guidance)
        for slug, desc, is_ok, guidance in checks
    ]


def _variable_checks(variables: Iterable[str]) -> List[CheckResult]:
    # Currently no hard requirements, but keep the function for parity.
    return []


def run_probe(repo: str, token: str) -> Dict[str, Any]:
    checks: List[Dict[str, object]] = []
    errors: List[str] = []
    results: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
        "errors": errors,
    }

    try:
        label_names = _collect_labels(repo, token)
        results["labels"] = sorted(label_names)
        checks.extend(check.as_dict() for check in _label_checks(label_names))
    except HTTPError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list labels ({exc.code}): {exc.reason}")
    except URLError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list labels: {exc.reason}")

    try:
        secret_names = _collect_secrets(repo, token)
        results["secrets"] = sorted(secret_names)
        checks.extend(check.as_dict() for check in _secret_checks(secret_names))
    except HTTPError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list secrets ({exc.code}): {exc.reason}")
    except URLError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list secrets: {exc.reason}")

    try:
        variable_names = _collect_variables(repo, token)
        results["variables"] = sorted(variable_names)
        checks.extend(check.as_dict() for check in _variable_checks(variable_names))
    except HTTPError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list variables ({exc.code}): {exc.reason}")
    except URLError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list variables: {exc.reason}")

    failures = [check for check in checks if not bool(check.get("ok"))]
    results["ok"] = not failures and not errors
    results["failures"] = failures
    return results


def main() -> int:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "repo-health-report.json"

    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")

    if not repo:
        print("GITHUB_REPOSITORY is not set", file=sys.stderr)
        return 2
    if not token:
        print("GITHUB_TOKEN is not set", file=sys.stderr)
        return 2

    report = run_probe(repo=repo, token=token)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", False) else 1


if __name__ == "__main__":
    sys.exit(main())
