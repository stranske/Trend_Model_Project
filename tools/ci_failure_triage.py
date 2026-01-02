#!/usr/bin/env python3
"""CI Failure Triage Tool — Analyze CI failures and suggest fixes.

This tool takes CI failure logs and provides:
1. Classification of the error type
2. Root cause analysis
3. Suggested fix steps
4. Links to relevant playbooks

Can work with or without LLM (LLM provides better suggestions).

Usage:
    # From a log file
    python tools/ci_failure_triage.py --log-file ci_output.log --job-name "python ci (3.11)"

    # From stdin (piped from gh)
    gh run view 12345 --log-failed | python tools/ci_failure_triage.py --job-name "tests"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Check for optional LangChain
LANGCHAIN_AVAILABLE = False
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TriageResult:
    """Result of CI failure triage."""

    job_name: str
    error_type: str
    root_cause: str
    suggested_fix: str
    relevant_files: list[str] = field(default_factory=list)
    playbook_link: Optional[str] = None
    confidence: float = 0.0
    raw_error: str = ""


# Known error patterns and their fixes
ERROR_PATTERNS = {
    "mypy": {
        "pattern": re.compile(r"error:\s*(.+)\s*\[(\w+)\]", re.MULTILINE),
        "type": "type_error",
        "playbook": "docs/autofix_type_hygiene.md",
        "fix_template": "Add type annotation or # type: ignore[{code}] comment",
    },
    "pytest_failed": {
        "pattern": re.compile(r"FAILED\s+(\S+)::\S+", re.MULTILINE),
        "type": "test_failure",
        "playbook": "docs/checks.md",
        "fix_template": "Fix failing test in {file}",
    },
    "import_error": {
        "pattern": re.compile(
            r"(?:ImportError|ModuleNotFoundError):\s*(.+)", re.MULTILINE
        ),
        "type": "import_error",
        "playbook": "DEPENDENCY_QUICKSTART.md",
        "fix_template": "Install missing module or fix import path",
    },
    "syntax_error": {
        "pattern": re.compile(r"SyntaxError:\s*(.+)", re.MULTILINE),
        "type": "syntax_error",
        "playbook": None,
        "fix_template": "Fix syntax error: {details}",
    },
    "coverage": {
        "pattern": re.compile(
            r"(?:Coverage failure|FAIL Required coverage|coverage.*below)",
            re.IGNORECASE,
        ),
        "type": "coverage_failure",
        "playbook": "docs/checks.md",
        "fix_template": "Add tests to increase coverage above threshold",
    },
    "ruff": {
        "pattern": re.compile(r"Found \d+ error", re.IGNORECASE),
        "type": "lint_error",
        "playbook": "docs/checks.md",
        "fix_template": "Run 'ruff check --fix' or fix manually",
    },
    "timeout": {
        "pattern": re.compile(r"(?:timed? ?out|timeout|exceeded.*time)", re.IGNORECASE),
        "type": "timeout",
        "playbook": None,
        "fix_template": "Optimize slow test or increase timeout",
    },
    "assertion": {
        "pattern": re.compile(r"AssertionError:\s*(.+)?", re.MULTILINE),
        "type": "assertion_error",
        "playbook": None,
        "fix_template": "Fix assertion: expected value doesn't match actual",
    },
}


def extract_error_context(log: str, max_lines: int = 100) -> str:
    """Extract the most relevant error context from a log."""
    lines = log.split("\n")

    # Find lines with error indicators
    error_indices = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(
            keyword in line_lower
            for keyword in ["error", "failed", "exception", "traceback"]
        ):
            error_indices.append(i)

    if not error_indices:
        # No obvious errors, return last N lines
        return "\n".join(lines[-max_lines:])

    # Get context around first error
    first_error = error_indices[0]
    start = max(0, first_error - 10)
    end = min(len(lines), first_error + max_lines - 10)

    return "\n".join(lines[start:end])


def extract_files_from_log(log: str) -> list[str]:
    """Extract file paths mentioned in log."""
    # Pattern for Python file paths
    pattern = re.compile(r"(?:^|[\s:])([a-zA-Z_][\w/]*\.py)(?::\d+)?", re.MULTILINE)
    files = set()
    for match in pattern.finditer(log):
        filepath = match.group(1)
        if "/" in filepath or filepath.startswith("test_"):
            files.add(filepath)
    return sorted(files)


def pattern_based_triage(job_name: str, log: str) -> TriageResult:
    """Triage using pattern matching (no LLM)."""
    error_context = extract_error_context(log)
    files = extract_files_from_log(log)

    # Try each pattern
    for name, config in ERROR_PATTERNS.items():
        match = config["pattern"].search(error_context)
        if match:
            details = match.group(1) if match.lastindex else ""

            return TriageResult(
                job_name=job_name,
                error_type=config["type"],
                root_cause=f"{name}: {details[:200]}" if details else name,
                suggested_fix=config["fix_template"].format(
                    code=(
                        match.group(2)
                        if match.lastindex and match.lastindex >= 2
                        else "unknown"
                    ),
                    file=files[0] if files else "unknown",
                    details=details[:100] if details else "see log",
                ),
                relevant_files=files[:5],
                playbook_link=config["playbook"],
                confidence=0.7,
                raw_error=error_context[:500],
            )

    # No pattern matched
    return TriageResult(
        job_name=job_name,
        error_type="unknown",
        root_cause="Could not determine root cause from patterns",
        suggested_fix="Review the full log output",
        relevant_files=files[:5],
        playbook_link=None,
        confidence=0.3,
        raw_error=error_context[:500],
    )


def llm_triage(job_name: str, log: str, model: str = "gpt-4o-mini") -> TriageResult:
    """Triage using LLM for better analysis."""
    if not LANGCHAIN_AVAILABLE:
        result = pattern_based_triage(job_name, log)
        result.suggested_fix += " (LLM not available for enhanced analysis)"
        return result

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        result = pattern_based_triage(job_name, log)
        result.suggested_fix += " (OPENAI_API_KEY not set)"
        return result

    error_context = extract_error_context(log, max_lines=150)
    files = extract_files_from_log(log)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a CI/CD debugging expert. Analyze the failure log and provide a concise diagnosis.

Output format (use exactly these labels):
ERROR_TYPE: [one of: type_error, test_failure, import_error, syntax_error, coverage_failure, lint_error, timeout, assertion_error, build_error, unknown]
ROOT_CAUSE: [one sentence describing the actual problem]
SUGGESTED_FIX: [specific actionable steps to fix, maximum 2-3 sentences]
CONFIDENCE: [0.0-1.0 based on how clear the error is]

Be concise. Focus on the actual error, not warnings or info messages.""",
            ),
            (
                "human",
                """Job: {job_name}

Log excerpt:
```
{log_content}
```

Analyze this failure.""",
            ),
        ]
    )

    try:
        llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
        response = llm.invoke(
            prompt.format(job_name=job_name, log_content=error_context)
        )

        # Parse response
        content = response.content

        error_type = _extract_field(content, "ERROR_TYPE", "unknown")
        root_cause = _extract_field(content, "ROOT_CAUSE", "Could not determine")
        suggested_fix = _extract_field(content, "SUGGESTED_FIX", "Review log manually")
        confidence = float(_extract_field(content, "CONFIDENCE", "0.5"))

        # Map error type to playbook
        playbook = None
        for config in ERROR_PATTERNS.values():
            if config["type"] == error_type:
                playbook = config["playbook"]
                break

        return TriageResult(
            job_name=job_name,
            error_type=error_type,
            root_cause=root_cause,
            suggested_fix=suggested_fix,
            relevant_files=files[:5],
            playbook_link=playbook,
            confidence=confidence,
            raw_error=error_context[:500],
        )

    except Exception as e:
        result = pattern_based_triage(job_name, log)
        result.suggested_fix += f" (LLM failed: {e})"
        return result


def _extract_field(content: str, field: str, default: str) -> str:
    """Extract a labeled field from LLM response."""
    pattern = re.compile(rf"{field}:\s*(.+?)(?:\n|$)", re.IGNORECASE)
    match = pattern.search(content)
    if match:
        return match.group(1).strip()
    return default


def triage_ci_failure(
    job_name: str, log: str, use_llm: bool = True, model: str = "gpt-4o-mini"
) -> TriageResult:
    """Main entry point for CI failure triage.

    Args:
        job_name: Name of the failing CI job
        log: The log content
        use_llm: Whether to use LLM for enhanced analysis
        model: OpenAI model to use

    Returns:
        TriageResult with analysis
    """
    if use_llm:
        return llm_triage(job_name, log, model)
    return pattern_based_triage(job_name, log)


def format_triage_report(result: TriageResult) -> str:
    """Format triage result as markdown."""
    lines = [
        f"# CI Failure Triage: {result.job_name}",
        "",
        f"**Error Type:** {result.error_type}",
        f"**Confidence:** {result.confidence:.0%}",
        "",
        "## Root Cause",
        result.root_cause,
        "",
        "## Suggested Fix",
        result.suggested_fix,
    ]

    if result.relevant_files:
        lines.extend(
            [
                "",
                "## Relevant Files",
            ]
        )
        for f in result.relevant_files:
            lines.append(f"- `{f}`")

    if result.playbook_link:
        lines.extend(
            [
                "",
                "## Playbook",
                f"See [{result.playbook_link}]({result.playbook_link}) for more guidance.",
            ]
        )

    if result.raw_error:
        lines.extend(
            [
                "",
                "## Error Context",
                "```",
                result.raw_error[:300] + ("..." if len(result.raw_error) > 300 else ""),
                "```",
            ]
        )

    return "\n".join(lines)


def format_triage_comment(result: TriageResult) -> str:
    """Format triage result as a PR comment."""
    badge = (
        "🔴" if result.confidence >= 0.7 else "🟡" if result.confidence >= 0.4 else "⚪"
    )

    lines = [
        f"### {badge} CI Failure Analysis: `{result.job_name}`",
        "",
        f"**Error:** {result.error_type}",
        "",
        f"**Likely cause:** {result.root_cause}",
        "",
        f"**Suggested fix:** {result.suggested_fix}",
    ]

    if result.relevant_files:
        files_str = ", ".join(f"`{f}`" for f in result.relevant_files[:3])
        lines.append(f"\n**Files to check:** {files_str}")

    if result.playbook_link:
        lines.append(f"\n📚 [Relevant playbook]({result.playbook_link})")

    return "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, help="Path to CI log file")
    parser.add_argument("--job-name", default="unknown", help="Name of the CI job")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for analysis")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--comment-format", action="store_true", help="Output as PR comment"
    )
    args = parser.parse_args()

    if args.log_file:
        log = args.log_file.read_text()
    else:
        log = sys.stdin.read()

    result = triage_ci_failure(
        args.job_name, log, use_llm=args.use_llm, model=args.model
    )

    if args.json:
        output = {
            "job_name": result.job_name,
            "error_type": result.error_type,
            "root_cause": result.root_cause,
            "suggested_fix": result.suggested_fix,
            "relevant_files": result.relevant_files,
            "playbook_link": result.playbook_link,
            "confidence": result.confidence,
        }
        print(json.dumps(output, indent=2))
    elif args.comment_format:
        print(format_triage_comment(result))
    else:
        print(format_triage_report(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
