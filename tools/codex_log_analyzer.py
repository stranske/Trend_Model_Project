#!/usr/bin/env python3
"""Codex Log Analyzer — Extract progress signals from Codex session logs.

This tool analyzes Codex session logs to determine actual task completion
status, even when checkboxes weren't explicitly updated. It bridges the gap
between what Codex did and what the automation can detect.

Usage:
    python tools/codex_log_analyzer.py --pr 4112 --log-file session.log
    python tools/codex_log_analyzer.py --pr-body body.md --tasks tasks.json

The tool can:
1. Parse PR body to extract tasks and acceptance criteria
2. Analyze session logs for evidence of completion
3. Suggest checkbox updates
4. Identify blockers that need resolution
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Patterns for detecting work evidence in logs
EVIDENCE_PATTERNS = {
    "file_created": re.compile(
        r"(?:created?|wrote?|added?)\s+(?:file\s+)?['\"]?([^\s'\"]+\.(py|yml|yaml|js|ts|md))['\"]?",
        re.I,
    ),
    "file_modified": re.compile(
        r"(?:modified?|updated?|changed?|edited?)\s+(?:file\s+)?['\"]?([^\s'\"]+\.(py|yml|yaml|js|ts|md))['\"]?",
        re.I,
    ),
    "test_added": re.compile(r"(?:added?|created?|wrote?)\s+(?:a\s+)?test", re.I),
    "test_passed": re.compile(
        r"(?:test(?:s)?\s+pass(?:ed)?|all\s+tests?\s+pass)", re.I
    ),
    "test_failed": re.compile(r"(?:test(?:s)?\s+fail(?:ed)?|FAILED)", re.I),
    "commit_made": re.compile(r"(?:committed?|commit\s+\w{7,40})", re.I),
    "error_encountered": re.compile(r"(?:error|exception|traceback|failed):", re.I),
    "implementation_done": re.compile(
        r"(?:implemented?|completed?|finished?|done)\s+(?:the\s+)?(?:task|feature|fix)",
        re.I,
    ),
    "documentation_added": re.compile(
        r"(?:added?|updated?|wrote?)\s+(?:the\s+)?(?:documentation|docs|docstring)",
        re.I,
    ),
}

# Keywords that suggest task completion
COMPLETION_KEYWORDS = [
    "implemented",
    "completed",
    "done",
    "finished",
    "added",
    "created",
    "fixed",
    "resolved",
    "working",
    "passes",
    "verified",
    "confirmed",
]

# Keywords that suggest blockers
BLOCKER_KEYWORDS = [
    "error",
    "failed",
    "cannot",
    "unable",
    "blocked",
    "missing",
    "undefined",
    "not found",
    "exception",
    "traceback",
]


@dataclass
class TaskProgress:
    """Progress status for a single task."""

    task_text: str
    status: str  # complete, partial, attempted_failed, not_started
    evidence: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class LogAnalysis:
    """Complete analysis of a Codex session."""

    tasks_completed: list[TaskProgress] = field(default_factory=list)
    tasks_attempted: list[TaskProgress] = field(default_factory=list)
    tasks_not_started: list[TaskProgress] = field(default_factory=list)
    blockers_encountered: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tests_added: int = 0
    commits_made: int = 0
    suggested_next_action: str = ""
    checkbox_updates_needed: list[str] = field(default_factory=list)


def extract_checkboxes(markdown: str) -> list[dict[str, Any]]:
    """Extract checkbox items from markdown."""
    pattern = re.compile(r"^(\s*)[-*]\s*\[([ xX])\]\s*(.+)$", re.MULTILINE)
    checkboxes = []
    for match in pattern.finditer(markdown):
        indent, checked, text = match.groups()
        checkboxes.append(
            {
                "text": text.strip(),
                "checked": checked.lower() == "x",
                "indent": len(indent),
                "full_match": match.group(0),
            }
        )
    return checkboxes


def extract_tasks_and_acceptance(body: str) -> tuple[list[str], list[str]]:
    """Extract tasks and acceptance criteria from PR body."""
    tasks: list[str] = []
    acceptance: list[str] = []

    # Look for Tasks section
    tasks_match = re.search(
        r"(?:^|\n)#+\s*Tasks?\s*\n(.*?)(?=\n#+|\n<!-- |$)",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    if tasks_match:
        for cb in extract_checkboxes(tasks_match.group(1)):
            tasks.append(cb["text"])

    # Look for Acceptance Criteria section
    ac_match = re.search(
        r"(?:^|\n)#+\s*Acceptance\s*(?:Criteria)?\s*\n(.*?)(?=\n#+|\n<!-- |$)",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    if ac_match:
        for cb in extract_checkboxes(ac_match.group(1)):
            acceptance.append(cb["text"])

    return tasks, acceptance


def find_evidence_for_task(task: str, log_content: str) -> tuple[str, list[str], float]:
    """Search log for evidence that a task was completed.

    Returns: (status, evidence_list, confidence)
    """
    task_lower = task.lower()
    evidence: list[str] = []
    confidence = 0.0

    # Extract key terms from task
    task_terms = set(re.findall(r"\b\w{4,}\b", task_lower))
    task_terms -= {"that", "this", "with", "from", "should", "must", "when", "where"}

    lines = log_content.split("\n")
    relevant_lines: list[str] = []

    for line in lines:
        line_lower = line.lower()
        # Check if line mentions task-related terms
        matching_terms = task_terms & set(re.findall(r"\b\w{4,}\b", line_lower))
        if len(matching_terms) >= 2:
            relevant_lines.append(line)

    if not relevant_lines:
        return "not_started", [], 0.0

    # Check for completion signals
    completion_found = False
    for line in relevant_lines:
        line_lower = line.lower()
        for keyword in COMPLETION_KEYWORDS:
            if keyword in line_lower:
                completion_found = True
                evidence.append(f"Found '{keyword}' in: {line[:100]}...")
                confidence += 0.2

    # Check for blockers
    blocker_found = False
    for line in relevant_lines:
        line_lower = line.lower()
        for keyword in BLOCKER_KEYWORDS:
            if keyword in line_lower:
                blocker_found = True
                evidence.append(f"Found blocker '{keyword}' in: {line[:100]}...")
                confidence -= 0.1

    # Check for file modifications
    files_touched: list[str] = []
    for name, pattern in EVIDENCE_PATTERNS.items():
        for match in pattern.finditer(log_content):
            if name in ("file_created", "file_modified"):
                files_touched.append(match.group(1))
                evidence.append(f"File {name.replace('_', ' ')}: {match.group(1)}")
                confidence += 0.15

    # Determine status
    confidence = min(1.0, max(0.0, confidence))

    if completion_found and not blocker_found and confidence >= 0.5:
        return "complete", evidence, confidence
    elif completion_found and blocker_found:
        return "partial", evidence, confidence
    elif blocker_found:
        return "attempted_failed", evidence, confidence
    elif relevant_lines:
        return "partial", evidence, confidence

    return "not_started", [], 0.0


def analyze_log(
    log_content: str,
    tasks: list[str],
    acceptance_criteria: list[str],
) -> LogAnalysis:
    """Analyze a Codex session log against tasks and acceptance criteria."""
    analysis = LogAnalysis()

    # Analyze each task
    for task in tasks:
        status, evidence, confidence = find_evidence_for_task(task, log_content)
        progress = TaskProgress(
            task_text=task,
            status=status,
            evidence=evidence,
            confidence=confidence,
        )

        if status == "complete":
            analysis.tasks_completed.append(progress)
            analysis.checkbox_updates_needed.append(task)
        elif status in ("partial", "attempted_failed"):
            analysis.tasks_attempted.append(progress)
        else:
            analysis.tasks_not_started.append(progress)

    # Analyze acceptance criteria similarly
    for ac in acceptance_criteria:
        status, evidence, confidence = find_evidence_for_task(ac, log_content)
        if status == "complete" and confidence >= 0.6:
            analysis.checkbox_updates_needed.append(ac)

    # Extract global evidence
    for name, pattern in EVIDENCE_PATTERNS.items():
        for match in pattern.finditer(log_content):
            if name == "file_created":
                analysis.files_created.append(match.group(1))
            elif name == "file_modified":
                analysis.files_modified.append(match.group(1))
            elif name == "test_added":
                analysis.tests_added += 1
            elif name == "commit_made":
                analysis.commits_made += 1
            elif name == "error_encountered":
                # Extract error context
                start = max(0, match.start() - 50)
                end = min(len(log_content), match.end() + 200)
                error_context = log_content[start:end].strip()
                if error_context not in analysis.blockers_encountered:
                    analysis.blockers_encountered.append(error_context[:300])

    # Deduplicate files
    analysis.files_created = list(set(analysis.files_created))
    analysis.files_modified = list(set(analysis.files_modified))

    # Generate suggested next action
    if analysis.blockers_encountered:
        analysis.suggested_next_action = f"Resolve blockers before continuing: {len(analysis.blockers_encountered)} errors found"
    elif analysis.tasks_not_started:
        next_task = analysis.tasks_not_started[0].task_text
        analysis.suggested_next_action = f"Start next task: {next_task}"
    elif analysis.tasks_attempted:
        next_task = analysis.tasks_attempted[0].task_text
        analysis.suggested_next_action = f"Complete partially done task: {next_task}"
    else:
        analysis.suggested_next_action = (
            "All tasks appear complete — verify and close PR"
        )

    return analysis


def analyze_pr_changes(
    file_changes: list[dict[str, Any]], tasks: list[str]
) -> LogAnalysis:
    """Analyze PR file changes to infer task completion.

    This is used when we have PR diff data instead of session logs.
    """
    analysis = LogAnalysis()

    # Build a pseudo-log from file changes
    pseudo_log_parts = []
    for change in file_changes:
        filename = change.get("fileName", change.get("filename", ""))
        patch = change.get("patch", "")

        if filename:
            analysis.files_modified.append(filename)
            pseudo_log_parts.append(f"Modified file: {filename}")

            # Check patch content for evidence
            if "+def test_" in patch or "+async def test_" in patch:
                analysis.tests_added += 1
                pseudo_log_parts.append("Added test function")

            if "+class " in patch:
                pseudo_log_parts.append("Added class definition")

            if "documentation" in filename.lower() or filename.endswith(".md"):
                pseudo_log_parts.append("Documentation updated")

    pseudo_log = "\n".join(pseudo_log_parts)

    # Now analyze tasks against this pseudo-log
    for task in tasks:
        status, evidence, confidence = find_evidence_for_task(task, pseudo_log)

        # Boost confidence based on file patterns
        task_lower = task.lower()
        for filename in analysis.files_modified:
            if "test" in task_lower and "test" in filename.lower():
                confidence += 0.3
            if "documentation" in task_lower and filename.endswith(".md"):
                confidence += 0.3

        confidence = min(1.0, confidence)

        progress = TaskProgress(
            task_text=task,
            status=status if confidence < 0.5 else "complete",
            evidence=evidence,
            files_modified=[
                f for f in analysis.files_modified if _file_relates_to_task(f, task)
            ],
            confidence=confidence,
        )

        if progress.status == "complete":
            analysis.tasks_completed.append(progress)
            analysis.checkbox_updates_needed.append(task)
        elif progress.status in ("partial", "attempted_failed"):
            analysis.tasks_attempted.append(progress)
        else:
            analysis.tasks_not_started.append(progress)

    return analysis


def _file_relates_to_task(filename: str, task: str) -> bool:
    """Check if a filename is likely related to a task."""
    task_lower = task.lower()
    filename_lower = filename.lower()

    # Extract meaningful words from task
    words = re.findall(r"\b\w{4,}\b", task_lower)

    for word in words:
        if word in filename_lower:
            return True

    return False


def format_analysis_report(analysis: LogAnalysis) -> str:
    """Format analysis as a markdown report."""
    lines = [
        "# Codex Log Analysis Report",
        "",
        "## Summary",
        f"- Tasks completed: {len(analysis.tasks_completed)}",
        f"- Tasks attempted: {len(analysis.tasks_attempted)}",
        f"- Tasks not started: {len(analysis.tasks_not_started)}",
        f"- Files modified: {len(analysis.files_modified)}",
        f"- Tests added: {analysis.tests_added}",
        f"- Blockers found: {len(analysis.blockers_encountered)}",
        "",
    ]

    if analysis.checkbox_updates_needed:
        lines.extend(
            [
                "## Checkbox Updates Recommended",
                "",
                "The following items appear complete based on evidence:",
                "",
            ]
        )
        for item in analysis.checkbox_updates_needed:
            lines.append(f"- [ ] → [x] {item}")
        lines.append("")

    if analysis.tasks_completed:
        lines.extend(
            [
                "## Completed Tasks",
                "",
            ]
        )
        for task in analysis.tasks_completed:
            lines.append(f"### {task.task_text}")
            lines.append(
                f"- Status: **{task.status}** (confidence: {task.confidence:.0%})"
            )
            if task.evidence:
                lines.append("- Evidence:")
                for ev in task.evidence[:3]:
                    lines.append(f"  - {ev}")
            lines.append("")

    if analysis.blockers_encountered:
        lines.extend(
            [
                "## Blockers Encountered",
                "",
            ]
        )
        for i, blocker in enumerate(analysis.blockers_encountered[:5], 1):
            lines.append(f"{i}. ```{blocker[:200]}...```")
        lines.append("")

    lines.extend(
        [
            "## Suggested Next Action",
            "",
            analysis.suggested_next_action,
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, help="Path to Codex session log")
    parser.add_argument("--pr-body", type=Path, help="Path to PR body markdown")
    parser.add_argument("--pr-changes", type=Path, help="Path to PR file changes JSON")
    parser.add_argument("--tasks", type=Path, help="Path to tasks JSON array")
    parser.add_argument(
        "--acceptance", type=Path, help="Path to acceptance criteria JSON array"
    )
    parser.add_argument("--output", type=Path, help="Output file for report")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of markdown"
    )
    args = parser.parse_args()

    # Load tasks and acceptance criteria
    tasks: list[str] = []
    acceptance: list[str] = []

    if args.tasks:
        raw_tasks = json.loads(args.tasks.read_text())
        # Handle both list of strings and list of dicts
        tasks = [t["text"] if isinstance(t, dict) else str(t) for t in raw_tasks]

    if args.acceptance:
        raw_acceptance = json.loads(args.acceptance.read_text())
        acceptance = [
            a["text"] if isinstance(a, dict) else str(a) for a in raw_acceptance
        ]

    if args.pr_body:
        body = args.pr_body.read_text()
        extracted_tasks, extracted_ac = extract_tasks_and_acceptance(body)
        tasks = tasks or extracted_tasks
        acceptance = acceptance or extracted_ac

    # Analyze
    if args.log_file:
        log_content = args.log_file.read_text()
        analysis = analyze_log(log_content, tasks, acceptance)
    elif args.pr_changes:
        changes = json.loads(args.pr_changes.read_text())
        analysis = analyze_pr_changes(changes, tasks)
    else:
        print("Error: Must provide either --log-file or --pr-changes", file=sys.stderr)
        return 1

    # Output
    if args.json:
        output = json.dumps(
            {
                "tasks_completed": [t.__dict__ for t in analysis.tasks_completed],
                "tasks_attempted": [t.__dict__ for t in analysis.tasks_attempted],
                "tasks_not_started": [t.__dict__ for t in analysis.tasks_not_started],
                "blockers_encountered": analysis.blockers_encountered,
                "files_created": analysis.files_created,
                "files_modified": analysis.files_modified,
                "tests_added": analysis.tests_added,
                "commits_made": analysis.commits_made,
                "suggested_next_action": analysis.suggested_next_action,
                "checkbox_updates_needed": analysis.checkbox_updates_needed,
            },
            indent=2,
        )
    else:
        output = format_analysis_report(analysis)

    if args.output:
        args.output.write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
