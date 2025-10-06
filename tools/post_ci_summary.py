"""Helpers for building the consolidated post-CI summary comment.

This module is used by the maint-30-post-ci-summary workflow and is unit tested
so regressions can be caught without executing the workflow on GitHub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableSequence, Sequence
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class JobRecord:
    name: str
    state: str | None
    url: str | None
    highlight: bool


@dataclass(frozen=True)
class RunRecord:
    key: str
    display_name: str
    present: bool
    state: str | None
    attempt: int | None
    label: str
    url: str | None


DEFAULT_REQUIRED_JOB_GROUPS = [
    {"label": "CI python", "patterns": [r"^ci / python(?: /|$)"]},
]


def _badge(state: str | None) -> str:
    if not state:
        return "⏳"
    normalized = state.lower()
    if normalized == "success":
        return "✅"
    if normalized in {"failure", "cancelled", "timed_out", "action_required"}:
        return "❌"
    if normalized == "skipped":
        return "⏭️"
    if normalized in {"in_progress", "queued", "waiting", "requested"}:
        return "⏳"
    return "⏳"


def _display_state(state: str | None) -> str:
    if not state:
        return "pending"
    text = str(state).strip()
    if not text:
        return "pending"
    return text.replace("_", " ").lower()


def _priority(state: str | None) -> int:
    normalized = (state or "").lower()
    if normalized in {"failure", "cancelled", "timed_out", "action_required"}:
        return 0
    if normalized in {"in_progress", "queued", "waiting", "requested"}:
        return 1
    if normalized == "success":
        return 2
    if normalized == "skipped":
        return 3
    return 4


def _combine_states(states: Iterable[str | None]) -> str:
    lowered: List[str] = [s.lower() for s in states if isinstance(s, str) and s]
    if not lowered:
        return "missing"
    for candidate in ("failure", "cancelled", "timed_out", "action_required"):
        if candidate in lowered:
            return candidate
    for candidate in ("in_progress", "queued", "waiting", "requested"):
        if candidate in lowered:
            return candidate
    if all(state == "skipped" for state in lowered):
        return "skipped"
    if "success" in lowered:
        return "success"
    return lowered[0]


def _load_required_groups(env_value: str | None) -> List[Mapping[str, str]]:
    if not env_value:
        return DEFAULT_REQUIRED_JOB_GROUPS.copy()
    try:
        parsed = json.loads(env_value)
    except json.JSONDecodeError:
        return DEFAULT_REQUIRED_JOB_GROUPS.copy()
    if not isinstance(parsed, list):
        return DEFAULT_REQUIRED_JOB_GROUPS.copy()
    result: List[Mapping[str, str]] = []
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or item.get("name") or "").strip()
        patterns = item.get("patterns")
        if not label or not isinstance(patterns, Sequence):
            continue
        cleaned = [p for p in patterns if isinstance(p, str) and p]
        if not cleaned:
            continue
        result.append({"label": label, "patterns": cleaned})
    return result or DEFAULT_REQUIRED_JOB_GROUPS.copy()


def _build_job_rows(runs: Sequence[Mapping[str, object]]) -> List[JobRecord]:
    rows: List[JobRecord] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        present = bool(run.get("present"))
        if not present:
            continue
        display = str(
            run.get("displayName")
            or run.get("display_name")
            or run.get("key")
            or "workflow"
        )
        jobs = run.get("jobs")
        if not isinstance(jobs, Sequence):
            continue
        for job in jobs:
            if not isinstance(job, Mapping):
                continue
            name = str(job.get("name") or "").strip()
            if not name:
                continue
            state = job.get("conclusion") or job.get("status")
            state_str = str(state) if state is not None else None
            highlight = bool(
                state_str
                and state_str.lower()
                in {"failure", "cancelled", "timed_out", "action_required"}
            )
            label = f"{display} / {name}"
            if highlight:
                label = f"**{label}**"
            rows.append(
                JobRecord(
                    name=label,
                    state=state_str,
                    url=str(job.get("html_url")) if job.get("html_url") else None,
                    highlight=highlight,
                )
            )
    rows.sort(key=lambda record: (_priority(record.state), record.name))
    return rows


def _format_jobs_table(rows: Sequence[JobRecord]) -> List[str]:
    header = [
        "| Workflow / Job | Result | Logs |",
        "|----------------|--------|------|",
    ]
    if not rows:
        return header + ["| _(no jobs reported)_ | ⏳ pending | — |"]
    body = []
    for record in rows:
        state_display = _display_state(record.state)
        link = f"[logs]({record.url})" if record.url else "—"
        body.append(
            f"| {record.name} | {_badge(record.state)} {state_display} | {link} |"
        )
    return header + body


def _collect_required_segments(
    runs: Sequence[Mapping[str, object]],
    groups: Sequence[Mapping[str, Sequence[str]]],
) -> List[str]:
    import re

    segments: List[str] = []
    run_lookup = {run.get("key"): run for run in runs if isinstance(run, Mapping)}
    ci_run = run_lookup.get("ci")
    if isinstance(ci_run, Mapping) and ci_run.get("present"):
        jobs = ci_run.get("jobs")
        job_list = jobs if isinstance(jobs, Sequence) else []
        for group in groups:
            label = group.get("label")
            patterns = group.get("patterns") or []
            regexes = []
            for pattern in patterns:
                try:
                    regexes.append(re.compile(pattern))
                except re.error:
                    continue
            if not label or not regexes:
                continue
            matched_states = []
            for job in job_list:
                if not isinstance(job, Mapping):
                    continue
                name = str(job.get("name") or "")
                if any(regex.search(name) for regex in regexes):
                    matched_states.append(job.get("conclusion") or job.get("status"))
            state = _combine_states(matched_states)
            segments.append(f"{label}: {_badge(state)} {_display_state(state)}")
    else:
        segments.append("CI: ⏳ pending")

    docker_run = run_lookup.get("docker")
    if isinstance(docker_run, Mapping) and docker_run.get("present"):
        state = docker_run.get("conclusion") or docker_run.get("status") or None
        segments.append(f"Docker: {_badge(state)} {_display_state(state)}")
    else:
        segments.append("Docker: ⏳ pending")
    return segments


def _format_latest_runs(runs: Sequence[Mapping[str, object]]) -> str:
    parts: List[str] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        display = str(
            run.get("displayName")
            or run.get("display_name")
            or run.get("key")
            or "workflow"
        )
        if not run.get("present"):
            parts.append(f"{display}: pending")
            continue
        run_id = run.get("id")
        attempt = run.get("run_attempt")
        attempt_suffix = (
            f" (attempt {attempt})" if isinstance(attempt, int) and attempt > 1 else ""
        )
        label = f"{display} (#{run_id}{attempt_suffix})" if run_id else display
        url = run.get("html_url")
        if url:
            parts.append(f"[{label}]({url})")
        else:
            parts.append(label)
    return " · ".join(parts)


def _format_coverage_lines(stats: Mapping[str, object] | None) -> List[str]:
    if not isinstance(stats, Mapping):
        return []

    def fmt_percent(value: object) -> str | None:
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return None

    def fmt_delta(value: object) -> str | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        sign = "+" if number > 0 else ""
        return f"{sign}{number:.2f} pp"

    lines: List[str] = []
    avg_latest = fmt_percent(stats.get("avg_latest"))
    avg_delta = fmt_delta(stats.get("avg_delta"))
    avg_parts = [
        part for part in (avg_latest, f"Δ {avg_delta}" if avg_delta else None) if part
    ]
    if avg_parts:
        lines.append(f"- Coverage (jobs): {' | '.join(avg_parts)}")

    worst_latest = fmt_percent(stats.get("worst_latest"))
    worst_delta = fmt_delta(stats.get("worst_delta"))
    worst_parts = [
        part
        for part in (worst_latest, f"Δ {worst_delta}" if worst_delta else None)
        if part
    ]
    if worst_parts:
        lines.append(f"- Coverage (worst job): {' | '.join(worst_parts)}")

    history_len = stats.get("history_len")
    if isinstance(history_len, int):
        lines.append(f"- Coverage history entries: {history_len}")
    return lines


def build_summary_comment(
    *,
    runs: Sequence[Mapping[str, object]],
    head_sha: str | None,
    coverage_stats: Mapping[str, object] | None,
    coverage_section: str | None,
    required_groups_env: str | None,
) -> str:
    rows = _build_job_rows(runs)
    job_table_lines = _format_jobs_table(rows)
    groups = _load_required_groups(required_groups_env)
    required_segments = _collect_required_segments(runs, groups)
    latest_runs_line = _format_latest_runs(runs)
    coverage_lines = _format_coverage_lines(coverage_stats)

    coverage_block: List[str] = []
    coverage_section_clean = (coverage_section or "").strip()
    if coverage_lines:
        coverage_block.append("### Coverage Overview")
        coverage_block.append("\n".join(coverage_lines))
    if coverage_section_clean:
        if not coverage_block:
            coverage_block.append("### Coverage Overview")
        coverage_block.append(coverage_section_clean)

    body_parts: MutableSequence[str] = [
        "<!-- post-ci-summary:do-not-edit -->",
        "### Automated Status Summary",
    ]
    if head_sha:
        body_parts.append(f"**Head SHA:** {head_sha}")
    if latest_runs_line:
        body_parts.append(f"**Latest Runs:** {latest_runs_line}")
    if required_segments:
        body_parts.append(f"**Required:** {', '.join(required_segments)}")
    body_parts.append("")
    body_parts.extend(job_table_lines)
    body_parts.append("")
    body_parts.extend(part for part in coverage_block if part)
    if coverage_block:
        body_parts.append("")
    body_parts.append(
        "_Updated automatically; will refresh on subsequent CI/Docker completions._"
    )

    return "\n".join(part for part in body_parts if part is not None)


def _load_json_from_env(value: str | None) -> Mapping[str, object] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def main() -> None:
    runs_value = os.environ.get("RUNS_JSON", "[]")
    try:
        runs = json.loads(runs_value)
    except json.JSONDecodeError:
        runs = []
    if not isinstance(runs, list):
        runs = []

    head_sha = os.environ.get("HEAD_SHA") or None
    coverage_stats = _load_json_from_env(os.environ.get("COVERAGE_STATS"))
    coverage_section = os.environ.get("COVERAGE_SECTION")
    required_groups_env = os.environ.get("REQUIRED_JOB_GROUPS_JSON")

    body = build_summary_comment(
        runs=runs,
        head_sha=head_sha,
        coverage_stats=coverage_stats,
        coverage_section=coverage_section,
        required_groups_env=required_groups_env,
    )

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        Path(output_path).write_text(f"body<<EOF\n{body}\nEOF\n", encoding="utf-8")
    else:
        print(body)


if __name__ == "__main__":
    main()
