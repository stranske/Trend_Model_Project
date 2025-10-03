from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from http.client import HTTPResponse
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GITHUB_API = os.environ.get("GITHUB_API_URL", "https://api.github.com")


@dataclass
class CheckResult:
    """Represents the status of a single governance probe."""

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


@dataclass
class ProbeSummary:
    """Structured aggregation of probe + lint status."""

    status: str
    summary_markdown: str
    issue_lines: List[str]


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
            if "secrets" in data:
                results.extend(
                    cast(Sequence[Dict[str, object]], data.get("secrets", []))
                )
            elif "variables" in data:
                results.extend(
                    cast(Sequence[Dict[str, object]], data.get("variables", []))
                )
            else:
                results.append(cast(Dict[str, object], data))
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
        CheckResult(
            name=slug, ok=is_ok, description=desc, details=None if is_ok else guidance
        )
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
        CheckResult(
            name=slug, ok=is_ok, description=desc, details=None if is_ok else guidance
        )
        for slug, desc, is_ok, guidance in checks
    ]


def _variable_checks(variables: Iterable[str]) -> List[CheckResult]:
    names = set(variables)
    checks = [
        (
            "ops-health-issue",
            "Repository variable `OPS_HEALTH_ISSUE` is configured",
            "OPS_HEALTH_ISSUE" in names,
            "Add `OPS_HEALTH_ISSUE` (issue number) so repo-health updates can be posted.",
        ),
    ]
    return [
        CheckResult(
            name=slug, ok=is_ok, description=desc, details=None if is_ok else guidance
        )
        for slug, desc, is_ok, guidance in checks
    ]


def _aggregate_checks(
    *,
    labels: Sequence[str],
    secrets: Sequence[str],
    variables: Sequence[str],
    errors: List[str],
) -> Dict[str, Any]:
    checks: List[Dict[str, object]] = []
    checks.extend(check.as_dict() for check in _label_checks(labels))
    checks.extend(check.as_dict() for check in _secret_checks(secrets))
    checks.extend(check.as_dict() for check in _variable_checks(variables))

    failures = [check for check in checks if not check.get("ok")]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labels": sorted(labels),
        "secrets": sorted(secrets),
        "variables": sorted(variables),
        "checks": checks,
        "errors": errors,
        "failures": failures,
        "ok": not failures and not errors,
    }


def run_probe(
    *,
    repo: str,
    token: str,
    fetch_labels: Optional[Callable[[str, str], Sequence[str]]] = None,
    fetch_secrets: Optional[Callable[[str, str], Sequence[str]]] = None,
    fetch_variables: Optional[Callable[[str, str], Sequence[str]]] = None,
) -> Dict[str, Any]:
    fetch_labels = fetch_labels or _collect_labels
    fetch_secrets = fetch_secrets or _collect_secrets
    fetch_variables = fetch_variables or _collect_variables

    errors: List[str] = []
    label_names: Sequence[str] = []
    secret_names: Sequence[str] = []
    variable_names: Sequence[str] = []

    try:
        label_names = fetch_labels(repo, token)
    except HTTPError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list labels ({exc.code}): {exc.reason}")
    except URLError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list labels: {exc.reason}")

    try:
        secret_names = fetch_secrets(repo, token)
    except HTTPError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list secrets ({exc.code}): {exc.reason}")
    except URLError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list secrets: {exc.reason}")

    try:
        variable_names = fetch_variables(repo, token)
    except HTTPError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list variables ({exc.code}): {exc.reason}")
    except URLError as exc:  # pragma: no cover - network errors in CI only.
        errors.append(f"Failed to list variables: {exc.reason}")

    return _aggregate_checks(
        labels=list(label_names),
        secrets=list(secret_names),
        variables=list(variable_names),
        errors=errors,
    )


def run_probe_from_sources(
    *, labels: Sequence[str], secrets: Sequence[str], variables: Sequence[str]
) -> Dict[str, Any]:
    return _aggregate_checks(
        labels=list(labels), secrets=list(secrets), variables=list(variables), errors=[]
    )


def build_summary(
    report: Mapping[str, Any],
    *,
    actionlint_ok: bool,
    issue_id_present: bool,
) -> ProbeSummary:
    failures: Sequence[Mapping[str, Any]] = cast(
        Sequence[Mapping[str, Any]], report.get("failures", []) or []
    )
    errors: Sequence[str] = cast(Sequence[str], report.get("errors", []) or [])

    issue_lines: List[str] = []

    if not actionlint_ok:
        issue_lines.append(
            "Workflow lint (`actionlint`) failed. Inspect the job logs for errors."
        )

    for failure in failures:
        description = str(failure.get("description") or "Repository configuration check failed.")
        details = failure.get("details")
        if details:
            issue_lines.append(f"{description} — {details}")
        else:
            issue_lines.append(description)

    for api_error in errors:
        issue_lines.append(f"API error: {api_error}")

    status = "success" if not issue_lines else "failure"

    summary_lines: List[str] = ["## Repo health nightly checks"]
    summary_lines.append("")

    if status == "success":
        summary_lines.append("- ✅ Workflow lint (`actionlint`) succeeded.")
        summary_lines.append("- ✅ Required labels, variables, and secrets are present.")
    else:
        summary_lines.append("- ❌ Issues detected during the nightly probe:")
        summary_lines.extend(f"  - {line}" for line in issue_lines)
        if not issue_id_present:
            summary_lines.append(
                "- ⚠️ `OPS_HEALTH_ISSUE` is unset; Ops issue update was skipped."
            )

    return ProbeSummary(
        status=status,
        summary_markdown="\n".join(summary_lines),
        issue_lines=list(issue_lines),
    )


def _append_step_summary(summary: str, step_summary_path: Optional[str]) -> None:
    if not step_summary_path:
        return
    with open(step_summary_path, "a", encoding="utf-8") as handle:
        handle.write(summary)
        if not summary.endswith("\n"):
            handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Repository health governance probe")
    parser.add_argument(
        "output",
        nargs="?",
        default="repo-health-report.json",
        help="Path to write the JSON report (default: repo-health-report.json)",
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        help="Offline fixture JSON with optional labels, secrets, variables arrays.",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write a markdown summary next to the report (output + .md).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")

    if args.fixtures:
        with open(args.fixtures, "r", encoding="utf-8") as fh:
            fixture_data = json.load(fh)
        report = run_probe_from_sources(
            labels=fixture_data.get("labels", []),
            secrets=fixture_data.get("secrets", []),
            variables=fixture_data.get("variables", []),
        )
    else:
        if not repo:
            print("GITHUB_REPOSITORY is not set", file=sys.stderr)
            return 2
        if not token:
            print("GITHUB_TOKEN is not set", file=sys.stderr)
            return 2
        report = run_probe(repo=repo, token=token)

    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if args.write_summary:
        summary = build_summary(report, actionlint_ok=True, issue_id_present=True)
        markdown_path = f"{output_path}.md"
        with open(markdown_path, "w", encoding="utf-8") as md:
            md.write(summary.summary_markdown + "\n")
        _append_step_summary(summary.summary_markdown + "\n", summary_path)

    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", False) else 1


if __name__ == "__main__":
    sys.exit(main())
