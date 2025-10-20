"""Maintain the rolling coverage baseline alert issue."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from urllib import error, parse, request

API_HEADER_ACCEPT = "application/vnd.github+json"
MARKER_START = "<!-- coverage-guard-metadata:"
MARKER_END = "-->"
DEFAULT_ISSUE_TITLE = "[coverage] baseline breach"
DEFAULT_ISSUE_LABEL = "health:coverage"
DEFAULT_TOP_LIMIT = 5


@dataclass
class BaselineConfig:
    baseline: Optional[float]
    warn_drop: float
    recovery_days: int


@dataclass
class CoverageSnapshot:
    current: float
    baseline: float
    delta: float


@dataclass
class FileCoverage:
    path: str
    percent: float
    covered: int
    total: int
    missing: int


class CoverageGuardError(RuntimeError):
    """Raised for unrecoverable workflow errors."""


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def load_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        print(f"Failed to parse JSON from {path}: {exc}", file=sys.stderr)
        return None


def load_baseline(path: Path) -> BaselineConfig:
    data = load_json(path) or {}
    baseline = _to_float(data.get("line"))
    warn_drop = _to_float(data.get("warn_drop")) or 1.0
    recovery_days_raw = data.get("recovery_days")
    recovery_days = _to_int(recovery_days_raw) if recovery_days_raw is not None else None
    if recovery_days is None or recovery_days <= 0:
        recovery_days = 3
    return BaselineConfig(baseline=baseline, warn_drop=warn_drop, recovery_days=recovery_days)


def load_snapshot(trend_path: Path, config: BaselineConfig) -> Optional[CoverageSnapshot]:
    data = load_json(trend_path)
    if not isinstance(data, Mapping):
        print(f"Coverage trend data missing at {trend_path}", file=sys.stderr)
        return None
    current = _to_float(data.get("current"))
    if current is None:
        print("Coverage trend data does not include a current value.", file=sys.stderr)
        return None
    baseline = config.baseline
    if baseline is None:
        baseline = _to_float(data.get("baseline"))
    if baseline is None:
        print("Unable to determine coverage baseline for comparison.", file=sys.stderr)
        return None
    delta = current - baseline
    return CoverageSnapshot(current=current, baseline=baseline, delta=delta)


def parse_links(header_value: str | None) -> dict[str, str]:
    links: dict[str, str] = {}
    if not header_value:
        return links
    parts = [segment.strip() for segment in header_value.split(",") if segment.strip()]
    for part in parts:
        if part.count("<") != 1 or part.count(">") != 1:
            continue
        url_part, *params = part.split(";")
        url = url_part.strip()[1:-1]
        rel = None
        for param in params:
            key, _, value = param.strip().partition("=")
            if key == "rel":
                rel = value.strip('"')
                break
        if rel and url:
            links[rel] = url
    return links


def github_request(
    method: str,
    url_or_path: str,
    token: str,
    *,
    params: Optional[dict[str, Any]] = None,
    payload: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, Mapping[str, str]]:
    base_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        url = url_or_path
    else:
        url = base_url + url_or_path
    if params:
        query = parse.urlencode(params, doseq=True)
        separator = "&" if parse.urlparse(url).query else "?"
        url = f"{url}{separator}{query}"
    data_bytes = None
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data_bytes, method=method.upper())
    req.add_header("Accept", API_HEADER_ACCEPT)
    req.add_header("User-Agent", "coverage-guard")
    req.add_header("Authorization", f"Bearer {token}")
    if data_bytes is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req) as resp:
            text = resp.read().decode("utf-8") if resp.length != 0 else ""
            body = json.loads(text) if text else None
            return body, resp.headers
    except error.HTTPError as exc:  # pragma: no cover - network failure
        details = exc.read().decode("utf-8", "ignore")
        print(
            f"GitHub API request failed: {exc.code} {exc.reason}: {details}",
            file=sys.stderr,
        )
        raise CoverageGuardError("GitHub API request failed") from exc


def list_issues(repo: str, token: str, label: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    url = f"/repos/{repo}/issues"
    params = {"state": "all", "labels": label, "per_page": 100, "sort": "created", "direction": "asc"}
    while url:
        body, headers = github_request("GET", url, token, params=params)
        params = None
        if isinstance(body, list):
            issues.extend(item for item in body if isinstance(item, Mapping))
        links = parse_links(headers.get("Link"))
        url = links.get("next")
    return issues


def find_issue(repo: str, token: str, label: str, title: str) -> Optional[dict[str, Any]]:
    candidates = [issue for issue in list_issues(repo, token, label) if str(issue.get("title", "")).startswith(title)]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.get("created_at", ""))
    return candidates[0]


def create_issue(repo: str, token: str, title: str, body: str, label: str) -> dict[str, Any]:
    payload = {"title": title, "body": body, "labels": [label]}
    issue, _ = github_request("POST", f"/repos/{repo}/issues", token, payload=payload)
    if not isinstance(issue, Mapping):  # pragma: no cover - defensive
        raise CoverageGuardError("Failed to create coverage guard issue")
    return dict(issue)


def update_issue(repo: str, token: str, number: int, *, body: Optional[str] = None, state: Optional[str] = None) -> None:
    payload: dict[str, Any] = {}
    if body is not None:
        payload["body"] = body
    if state is not None:
        payload["state"] = state
    if not payload:
        return
    github_request("PATCH", f"/repos/{repo}/issues/{number}", token, payload=payload)


def post_comment(repo: str, token: str, number: int, body: str) -> None:
    github_request("POST", f"/repos/{repo}/issues/{number}/comments", token, payload={"body": body})


def read_metadata(body: str) -> dict[str, Any]:
    if not body:
        return {}
    start = body.find(MARKER_START)
    if start == -1:
        return {}
    start += len(MARKER_START)
    end = body.find(MARKER_END, start)
    if end == -1:
        return {}
    raw = body[start:end].strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, Mapping):
        return dict(data)
    return {}


def compose_issue_body(config: BaselineConfig, metadata: Mapping[str, Any]) -> str:
    meta_json = json.dumps(metadata, sort_keys=True)
    lines = [
        MARKER_START,
        meta_json,
        MARKER_END,
        "",
        "## Coverage baseline monitor",
    ]
    if config.baseline is not None:
        lines.append(f"- Baseline: {config.baseline:.2f}%")
    else:
        lines.append("- Baseline: unavailable")
    lines.append(f"- Soft drop threshold: {config.warn_drop:.2f} pts")
    lines.append(f"- Recovery window: {config.recovery_days} consecutive days")
    lines.extend(
        [
            "",
            "This issue tracks code coverage when it falls below the configured baseline.",
            "Updates are posted once per day while the repository remains below baseline.",
            "",
            "_Managed automatically by `.github/workflows/maint-coverage-guard.yml`._",
        ]
    )
    return "\n".join(lines) + "\n"


def compute_top_files(data: Optional[Mapping[str, Any]], limit: int) -> list[FileCoverage]:
    if not isinstance(data, Mapping):
        return []
    files = data.get("files")
    if not isinstance(files, Mapping):
        return []
    entries: list[FileCoverage] = []
    for path, payload in files.items():
        if not isinstance(path, str) or not isinstance(payload, Mapping):
            continue
        summary = payload.get("summary")
        summary_map = summary if isinstance(summary, Mapping) else {}
        total = _to_int(summary_map.get("num_statements"))
        covered = _to_int(summary_map.get("covered_lines"))
        missing = _to_int(summary_map.get("missing_lines"))
        percent = _to_float(summary_map.get("percent_covered"))
        if total is None:
            executed = payload.get("executed_lines")
            if isinstance(executed, Iterable):
                total = len(list(executed))
        if total is None or total == 0:
            continue
        if covered is None:
            covered = total - (missing or 0)
        if missing is None:
            missing = max(total - (covered or 0), 0)
        if percent is None:
            percent = (covered or 0) / total * 100 if total else 0.0
        entries.append(
            FileCoverage(
                path=path,
                percent=float(percent),
                covered=int(covered or 0),
                total=int(total),
                missing=int(missing or 0),
            )
        )
    if not entries:
        return []
    nonzero = [item for item in entries if item.missing > 0]
    if nonzero:
        nonzero.sort(key=lambda item: (-item.missing, item.percent, item.path))
        return nonzero[:limit]
    entries.sort(key=lambda item: (-item.total, item.path))
    return entries[:limit]


def format_top_files(files: list[FileCoverage]) -> list[str]:
    if not files:
        return ["_Top changed files unavailable; coverage data missing._"]
    lines = ["**Top changed files**"]
    for index, item in enumerate(files, start=1):
        lines.append(
            f"{index}. `{item.path}` — {item.percent:.2f}% "
            f"({item.covered}/{item.total} covered, {item.missing} missing)"
        )
    return lines


def build_update_comment(
    snapshot: CoverageSnapshot,
    config: BaselineConfig,
    *,
    below_baseline: bool,
    date: dt.date,
    run_url: str,
    recovery_progress: Optional[str],
    top_files: list[FileCoverage],
) -> str:
    lines = [f"### {date.isoformat()}"]
    lines.append(f"- Current coverage: {snapshot.current:.2f}%")
    lines.append(f"- Baseline coverage: {snapshot.baseline:.2f}%")
    lines.append(f"- Delta vs baseline: {snapshot.delta:+.2f} pts")
    if run_url:
        lines.append(f"- Source run: {run_url}")
    status_text = "Below baseline" if below_baseline else "At or above baseline"
    lines.append(f"- Status: {status_text}")
    if recovery_progress:
        lines.append(f"- Recovery: {recovery_progress}")
    lines.append("")
    lines.extend(format_top_files(top_files))
    return "\n".join(lines)


def build_recovered_comment(snapshot: CoverageSnapshot, config: BaselineConfig, date: dt.date) -> str:
    return (
        "✅ Coverage recovered above baseline for "
        f"{config.recovery_days} consecutive days.\n\n"
        f"Current coverage: {snapshot.current:.2f}% (baseline {snapshot.baseline:.2f}%).\n"
        "Closing this issue. The guard will reopen it automatically if coverage drops again."
    )


def ensure_issue(repo: str, token: str, config: BaselineConfig, issue_title: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    issue = find_issue(repo, token, label, issue_title)
    metadata: dict[str, Any]
    if issue is None:
        metadata = {"recovery": 0, "last_status": "none", "last_updated": None}
        body = compose_issue_body(config, metadata)
        issue = create_issue(repo, token, issue_title, body, label)
        print(f"Created coverage guard issue #{issue['number']}")
    else:
        metadata = read_metadata(issue.get("body", ""))
        if not metadata:
            metadata = {"recovery": 0, "last_status": "none", "last_updated": None}
    return issue, metadata


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain coverage guard issue state.")
    parser.add_argument("--repo", required=True, help="Repository in owner/name format")
    parser.add_argument("--trend-path", type=Path, default=Path("coverage-trend.json"))
    parser.add_argument("--coverage-path", type=Path, default=Path("coverage.json"))
    parser.add_argument("--baseline-path", type=Path, default=Path("config/coverage-baseline.json"))
    parser.add_argument("--issue-title", default=DEFAULT_ISSUE_TITLE)
    parser.add_argument("--label", default=DEFAULT_ISSUE_LABEL)
    parser.add_argument("--recovery-days", type=int, default=None)
    parser.add_argument("--top-limit", type=int, default=DEFAULT_TOP_LIMIT)
    parser.add_argument("--run-url", default="", help="Workflow run URL for context")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    config = load_baseline(args.baseline_path)
    if args.recovery_days is not None and args.recovery_days > 0:
        config = BaselineConfig(
            baseline=config.baseline,
            warn_drop=config.warn_drop,
            recovery_days=args.recovery_days,
        )

    snapshot = load_snapshot(args.trend_path, config)
    if snapshot is None:
        print("No coverage snapshot available; skipping issue update.")
        return 0

    coverage_data = load_json(args.coverage_path)
    top_files = compute_top_files(coverage_data, args.top_limit)

    issue, metadata = ensure_issue(args.repo, token, config, args.issue_title, args.label)
    issue_number = int(issue["number"])
    is_open = issue.get("state") == "open"

    today = dt.datetime.utcnow().date()
    below_baseline = snapshot.current < snapshot.baseline
    run_url = args.run_url or os.environ.get("COVERAGE_RUN_URL", "")

    recovery_count = _to_int(metadata.get("recovery")) or 0
    last_status = str(metadata.get("last_status") or "none")

    if below_baseline:
        if issue.get("state") != "open":
            update_issue(args.repo, token, issue_number, state="open")
            issue["state"] = "open"
            print(f"Reopened coverage guard issue #{issue_number}")
        metadata.update({"recovery": 0, "last_status": "below", "last_updated": today.isoformat()})
        body = compose_issue_body(config, metadata)
        update_issue(args.repo, token, issue_number, body=body)
        comment = build_update_comment(
            snapshot,
            config,
            below_baseline=True,
            date=today,
            run_url=run_url,
            recovery_progress=None,
            top_files=top_files,
        )
        post_comment(args.repo, token, issue_number, comment)
        print(f"Posted coverage drop update to issue #{issue_number}")
        return 0

    # Coverage at or above baseline
    if not is_open:
        print(
            f"Coverage is at or above baseline and issue #{issue_number} is already closed; no update needed.",
        )
        return 0

    recovery = recovery_count + 1 if last_status == "above" or last_status == "recovery" else 1
    metadata.update({"recovery": recovery, "last_status": "above", "last_updated": today.isoformat()})
    body = compose_issue_body(config, metadata)
    update_issue(args.repo, token, issue_number, body=body)

    progress = f"{recovery}/{config.recovery_days} days above baseline"
    comment = build_update_comment(
        snapshot,
        config,
        below_baseline=False,
        date=today,
        run_url=run_url,
        recovery_progress=progress,
        top_files=top_files,
    )
    post_comment(args.repo, token, issue_number, comment)
    print(f"Posted recovery update to issue #{issue_number} (streak {recovery})")

    if recovery >= config.recovery_days and is_open:
        recovered_comment = build_recovered_comment(snapshot, config, today)
        post_comment(args.repo, token, issue_number, recovered_comment)
        update_issue(args.repo, token, issue_number, state="closed")
        print(f"Closed coverage guard issue #{issue_number} after recovery streak")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
