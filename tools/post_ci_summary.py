"""Utilities for rendering the consolidated post-CI status comment.

This module is imported by the maint-30-post-ci-summary workflow to gather
workflow/job metadata, parse coverage artifacts, and upsert the single
"Automated Status Summary" pull-request comment.  The public helpers are also
covered by unit tests so we can spot formatting regressions quickly.
"""
from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib import error, parse, request


@dataclass
class Requirement:
    label: str
    patterns: Sequence[str]


@dataclass
class WorkflowConfig:
    name: str
    workflow_file: str
    requirements: Sequence[Requirement]


@dataclass
class JobRecord:
    name: str
    conclusion: Optional[str]
    status: Optional[str]
    html_url: Optional[str]


@dataclass
class WorkflowSummary:
    config: WorkflowConfig
    run_id: Optional[int]
    html_url: Optional[str]
    conclusion: Optional[str]
    status: Optional[str]
    jobs: Sequence[JobRecord]


@dataclass
class CoverageDetails:
    avg_latest: Optional[float] = None
    avg_delta: Optional[float] = None
    worst_latest: Optional[float] = None
    worst_delta: Optional[float] = None
    table_markdown: Optional[str] = None


@dataclass
class FailureSnapshot:
    issues: Sequence[dict]


MARKER_HEADING = "### Automated Status Summary"

API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "trend-model-post-ci-summary",
}


def badge(state: Optional[str]) -> str:
    if not state:
        return "⏳"
    s = state.lower()
    if s == "success":
        return "✅"
    if s in {"failure", "cancelled", "timed_out", "action_required"}:
        return "❌"
    if s == "skipped":
        return "⏭️"
    return "⏳"


def normalize_state(state: Optional[str]) -> str:
    return (state or "").lower() or "missing"


def combine_states(states: Iterable[str]) -> str:
    normalized = [normalize_state(s) for s in states if s]
    if not normalized:
        return "missing"

    failure_states = {"failure", "cancelled", "timed_out", "action_required"}
    pending_states = {"in_progress", "queued", "waiting", "pending"}

    for candidate in normalized:
        if candidate in failure_states:
            return candidate
    for candidate in normalized:
        if candidate in pending_states:
            return candidate
    if all(candidate == "skipped" for candidate in normalized):
        return "skipped"
    if any(candidate == "success" for candidate in normalized):
        return "success"
    return normalized[0]


def _match_jobs(jobs: Sequence[JobRecord], patterns: Sequence[str]) -> List[JobRecord]:
    matched: List[JobRecord] = []
    for job in jobs:
        for pattern in patterns:
            if fnmatch.fnmatch(job.name, pattern):
                matched.append(job)
                break
    return matched


def summarize_requirements(summary: WorkflowSummary) -> List[str]:
    lines: List[str] = []
    for requirement in summary.config.requirements:
        matches = _match_jobs(summary.jobs, requirement.patterns)
        if not matches:
            state = "missing"
        else:
            state = combine_states(job.conclusion or job.status for job in matches)
        lines.append(f"{requirement.label}: {badge(state)} {state}")
    return lines


def _priority(state: Optional[str]) -> int:
    normalized = normalize_state(state)
    if normalized in {"failure", "timed_out", "cancelled", "action_required"}:
        return 0
    if normalized == "success":
        return 1
    if normalized == "skipped":
        return 2
    return 3


def render_job_table(summaries: Sequence[WorkflowSummary]) -> str:
    rows: List[str] = []
    for summary in summaries:
        if not summary.jobs:
            state = summary.conclusion or summary.status or "missing"
            b = badge(state)
            label = f"{summary.config.name} (workflow)"
            link = summary.html_url or ""
            link_part = f"[details]({link})" if link else ""
            rows.append(f"| {label} | {b} {state} | {link_part} |")
            continue
        for job in summary.jobs:
            state = job.conclusion or job.status or "missing"
            b = badge(state)
            label = f"{summary.config.name} — {job.name}"
            if normalize_state(state) in {"failure", "timed_out", "cancelled", "action_required"}:
                label = f"**{label}**"
            link = job.html_url or ""
            link_part = f"[logs]({link})" if link else ""
            rows.append(f"| {label} | {b} {state} | {link_part} |")
    rows.sort(key=lambda line: _priority(_extract_state_from_row(line)))
    header = "| Job | Result | Logs |\n|-----|--------|------|"
    return "\n".join([header, *rows]) if rows else "| Job | Result | Logs |\n|-----|--------|------|\n| _No workflow runs found_ | ⏳ pending | |"


def _extract_state_from_row(row: str) -> str:
    parts = row.split("|")
    if len(parts) < 4:
        return ""
    result = parts[2].strip()
    if " " in result:
        return result.split(" ", 1)[1]
    return result


def _safe_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_delta(delta: Optional[float]) -> Optional[str]:
    if delta is None:
        return None
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}pp"


def render_coverage_section(details: CoverageDetails) -> Optional[str]:
    if not details.table_markdown and details.avg_latest is None and details.worst_latest is None:
        return None

    lines: List[str] = ["### Coverage (soft gate)"]
    headline_parts: List[str] = []
    if details.avg_latest is not None:
        avg_text = f"{details.avg_latest:.2f}%"
        delta = _format_delta(details.avg_delta)
        if delta:
            avg_text = f"{avg_text} ({delta} vs prev)"
        headline_parts.append(f"**Avg:** {avg_text}")
    if details.worst_latest is not None:
        worst_text = f"{details.worst_latest:.2f}%"
        delta = _format_delta(details.worst_delta)
        if delta:
            worst_text = f"{worst_text} ({delta} vs prev)"
        headline_parts.append(f"**Worst:** {worst_text}")
    if headline_parts:
        lines.append(" | ".join(headline_parts))
        lines.append("")
    if details.table_markdown:
        lines.append(details.table_markdown.strip())
    return "\n".join(lines)


def render_failure_section(snapshot: Optional[FailureSnapshot]) -> Optional[str]:
    if not snapshot or not snapshot.issues:
        return None
    lines = ["### Open Failure Signatures", "| Issue | Occurrences | Last Seen | URL |", "|---|---:|---|---|"]
    for issue in snapshot.issues:
        number = issue.get("number")
        occurrences = issue.get("occurrences") or ""
        last_seen = issue.get("last_seen") or ""
        url = issue.get("url")
        issue_label = f"#{str(number)}" if number is not None else ""
        link = f"[link]({url})" if url else ""
        lines.append(f"| {issue_label} | {occurrences} | {last_seen} | {link} |")
    return "\n".join(lines)


def build_comment_body(head_sha: str, summaries: Sequence[WorkflowSummary], coverage: CoverageDetails, snapshot: Optional[FailureSnapshot], trigger_name: str) -> str:
    lines: List[str] = [MARKER_HEADING, f"**Head SHA:** {head_sha}", f"**Triggering workflow:** {trigger_name}"]

    requirement_lines: List[str] = []
    for summary in summaries:
        requirement_lines.extend(summarize_requirements(summary))
    if requirement_lines:
        lines.append(f"**Required:** {', '.join(requirement_lines)}")
    lines.append("")
    lines.append(render_job_table(summaries))
    lines.append("")

    coverage_section = render_coverage_section(coverage)
    if coverage_section:
        lines.append(coverage_section)
        lines.append("")

    failure_section = render_failure_section(snapshot)
    if failure_section:
        lines.append(failure_section)
        lines.append("")

    lines.append("_Updated automatically; refreshes after CI/Docker reruns._")
    return "\n".join(lines)


def _github_request(method: str, api_url: str, token: str, endpoint: str, params: Optional[dict] = None, body: Optional[dict] = None) -> tuple[int, str, dict]:
    url = f"{api_url.rstrip('/')}{endpoint}"
    if params:
        query = parse.urlencode(params)
        if query:
            url = f"{url}?{query}"
    data_bytes: Optional[bytes] = None
    headers = dict(API_HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        data_bytes = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data_bytes, headers=headers, method=method.upper())
    try:
        with request.urlopen(req) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            text = resp.read().decode(charset)
            return resp.getcode(), text, dict(resp.headers.items())
    except error.HTTPError as exc:
        charset = exc.headers.get_content_charset() if exc.headers else "utf-8"
        text = exc.read().decode(charset, errors="replace")
        return exc.code, text, dict(exc.headers.items() if exc.headers else {})


def _github_json(method: str, api_url: str, token: str, endpoint: str, params: Optional[dict] = None, body: Optional[dict] = None) -> dict:
    status, text, _ = _github_request(method, api_url, token, endpoint, params=params, body=body)
    if status >= 400:
        raise RuntimeError(f"GitHub API call failed ({status}): {text}")
    if not text:
        return {}
    return json.loads(text)


def _paginated_get(api_url: str, token: str, endpoint: str, params: Optional[dict] = None) -> List[dict]:
    results: List[dict] = []
    page = 1
    while True:
        page_params = dict(params or {}, page=page, per_page=100)
        payload = _github_json("GET", api_url, token, endpoint, params=page_params)
        items = payload.get("jobs") or payload.get("comments") or payload.get("workflow_runs") or []
        if not isinstance(items, list):
            break
        results.extend(items)
        if len(items) < 100:
            break
        page += 1
    return results


def fetch_jobs_for_run(api_url: str, token: str, owner: str, repo: str, run_id: int) -> List[JobRecord]:
    jobs_payload = _paginated_get(api_url, token, f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs")
    jobs: List[JobRecord] = []
    for raw in jobs_payload:
        jobs.append(
            JobRecord(
                name=raw.get("name") or "<unknown>",
                conclusion=raw.get("conclusion"),
                status=raw.get("status"),
                html_url=raw.get("html_url"),
            )
        )
    return jobs


def fetch_latest_run(api_url: str, token: str, owner: str, repo: str, workflow_file: str, head_sha: str) -> Optional[dict]:
    runs = _github_json(
        "GET",
        api_url,
        token,
        f"/repos/{owner}/{repo}/actions/workflows/{workflow_file}/runs",
        params={"head_sha": head_sha, "event": "pull_request", "per_page": 1},
    ).get("workflow_runs", [])
    if not runs:
        return None
    return runs[0]


def build_workflow_summary(api_url: str, token: str, owner: str, repo: str, config: WorkflowConfig, run_payload: Optional[dict]) -> WorkflowSummary:
    if not run_payload:
        return WorkflowSummary(config=config, run_id=None, html_url=None, conclusion=None, status=None, jobs=[])
    run_id = run_payload.get("id")
    jobs = fetch_jobs_for_run(api_url, token, owner, repo, run_id) if run_id is not None else []
    return WorkflowSummary(
        config=config,
        run_id=run_id,
        html_url=run_payload.get("html_url"),
        conclusion=run_payload.get("conclusion"),
        status=run_payload.get("status"),
        jobs=jobs,
    )


def load_text_file(base_dir: Path, *candidates: str) -> Optional[str]:
    for relative in candidates:
        path = base_dir / relative
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def load_json_file(base_dir: Path, *candidates: str) -> Optional[dict]:
    for relative in candidates:
        path = base_dir / relative
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
    return None


def load_coverage_details(base_dir: Path) -> CoverageDetails:
    summary = load_text_file(base_dir, "coverage_summary.md", "coverage-summary/coverage_summary.md")
    trend = load_json_file(base_dir, "coverage-trend.json", "coverage-trend/coverage-trend.json") or {}
    history = load_json_file(base_dir, "coverage-trend-history.json", "coverage-trend-history/coverage-trend-history.json")
    details = CoverageDetails(table_markdown=summary)
    details.avg_latest = _safe_float(trend.get("avg_coverage"))
    details.worst_latest = _safe_float(trend.get("worst_job_coverage"))
    if isinstance(history, list) and len(history) >= 2:
        prev = history[-2]
        if isinstance(prev, dict):
            details.avg_delta = _delta(details.avg_latest, _safe_float(prev.get("avg_coverage")))
            details.worst_delta = _delta(details.worst_latest, _safe_float(prev.get("worst_job_coverage")))
    return details


def _delta(latest: Optional[float], previous: Optional[float]) -> Optional[float]:
    if latest is None or previous is None:
        return None
    return round(latest - previous, 2)


def load_failure_snapshot(base_dir: Path) -> Optional[FailureSnapshot]:
    payload = load_json_file(base_dir, "ci_failures_snapshot.json", "ci-failures-snapshot/ci_failures_snapshot.json")
    if not payload:
        return None
    issues = payload.get("issues")
    if not isinstance(issues, list):
        return None
    filtered = [issue for issue in issues if isinstance(issue, dict)]
    return FailureSnapshot(issues=filtered)


WORKFLOW_CONFIGS: Sequence[WorkflowConfig] = (
    WorkflowConfig(
        name="CI",
        workflow_file="pr-10-ci-python.yml",
        requirements=(
            Requirement("CI tests", ("main / tests", "main / tests / *")),
            Requirement("CI workflow-automation", ("workflow / automation-tests", "workflow / automation-tests / *")),
            Requirement("CI style", ("main / style", "main / style / *")),
            Requirement("CI gate", ("gate / all-required-green",)),
        ),
    ),
    WorkflowConfig(
        name="Docker",
        workflow_file="pr-12-docker-smoke.yml",
        requirements=(
            Requirement("Docker lint", ("lint", "lint / *")),
            Requirement("Docker smoke", ("smoke", "smoke / *")),
        ),
    ),
)


def find_existing_comment(api_url: str, token: str, owner: str, repo: str, pr_number: int) -> Optional[dict]:
    comments = _paginated_get(api_url, token, f"/repos/{owner}/{repo}/issues/{pr_number}/comments")
    for comment in comments:
        body = comment.get("body")
        if isinstance(body, str) and MARKER_HEADING in body:
            return comment
    return None


def upsert_comment(api_url: str, token: str, owner: str, repo: str, pr_number: int, body: str) -> None:
    existing = find_existing_comment(api_url, token, owner, repo, pr_number)
    if existing:
        _github_json(
            "PATCH",
            api_url,
            token,
            f"/repos/{owner}/{repo}/issues/comments/{existing['id']}",
            body={"body": body},
        )
    else:
        _github_json(
            "POST",
            api_url,
            token,
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            body={"body": body},
        )


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository or "/" not in repository:
        raise RuntimeError("GITHUB_REPOSITORY is required")
    owner, repo = repository.split("/", 1)
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")

    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise RuntimeError("GITHUB_EVENT_PATH is required")
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    run = event.get("workflow_run") or {}
    head_sha = run.get("head_sha") or ""
    trigger_name = run.get("name") or ""
    pulls = run.get("pull_requests") or []
    if not pulls:
        print("No pull request found for workflow run; skipping comment update.")
        return
    pr_number = pulls[0].get("number")
    if not isinstance(pr_number, int):
        raise RuntimeError("Unable to determine PR number from workflow_run payload")

    triggered_config = None
    for config in WORKFLOW_CONFIGS:
        if config.name == trigger_name:
            triggered_config = config
            break

    summaries: List[WorkflowSummary] = []
    for config in WORKFLOW_CONFIGS:
        run_payload = None
        if triggered_config and triggered_config is config:
            run_payload = run
        else:
            run_payload = fetch_latest_run(api_url, token, owner, repo, config.workflow_file, head_sha)
        summaries.append(build_workflow_summary(api_url, token, owner, repo, config, run_payload))

    artifacts_dir = Path(os.environ.get("SUMMARY_ARTIFACTS_DIR", "summary_artifacts"))
    coverage = load_coverage_details(artifacts_dir)
    snapshot = load_failure_snapshot(artifacts_dir)

    body = build_comment_body(head_sha, summaries, coverage, snapshot, trigger_name)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "comment_preview.md").write_text(body, encoding="utf-8")

    upsert_comment(api_url, token, owner, repo, pr_number, body)
    print("Updated automated status summary comment.")


if __name__ == "__main__":
    main()
