#!/usr/bin/env python3
"""LangChain-enhanced task/scope extraction for keepalive automation.

This module provides LLM-backed extraction of tasks, scope, and acceptance
criteria from PR bodies that may have malformed markdown. It falls back
gracefully to regex-based extraction when LLM is unavailable.

Features:
- Tolerant parsing of varied markdown formats
- Semantic understanding of task completion evidence
- Integration with existing keepalive infrastructure

Usage:
    # As a library
    from tools.langchain_task_extractor import extract_structured_scope
    result = extract_structured_scope(pr_body)
    
    # CLI for testing
    python tools/langchain_task_extractor.py --body body.md --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Check for optional LangChain dependencies
LANGCHAIN_AVAILABLE = False
try:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ExtractedTask:
    """A single extracted task with metadata."""

    text: str
    is_complete: bool
    confidence: float = 1.0
    notes: str = ""


@dataclass
class ScopeExtraction:
    """Complete extraction result."""

    scope: str = ""
    tasks: list[ExtractedTask] = field(default_factory=list)
    acceptance_criteria: list[ExtractedTask] = field(default_factory=list)
    extraction_method: str = "regex"  # "regex" or "llm"
    extraction_notes: str = ""
    raw_sections: dict[str, str] = field(default_factory=dict)


# Pydantic models for LangChain (only defined if available)
if LANGCHAIN_AVAILABLE:

    class TaskModel(BaseModel):
        """Pydantic model for a single task."""

        text: str = Field(description="The task description text")
        is_complete: bool = Field(
            description="Whether the task is marked complete (checkbox checked)"
        )
        confidence: float = Field(
            default=1.0, description="Confidence in extraction (0-1)"
        )

    class ScopeExtractionModel(BaseModel):
        """Pydantic model for complete extraction."""

        scope: str = Field(description="The scope/objective description")
        tasks: list[TaskModel] = Field(
            default_factory=list, description="List of tasks"
        )
        acceptance_criteria: list[TaskModel] = Field(
            default_factory=list, description="List of acceptance criteria"
        )
        extraction_notes: str = Field(
            default="", description="Notes about extraction quality or issues"
        )


def _regex_extract_checkboxes(content: str) -> list[ExtractedTask]:
    """Extract checkboxes using regex patterns."""
    tasks = []

    # Pattern for standard checkboxes
    pattern = re.compile(r"^\s*[-*+]\s*\[([ xX])\]\s*(.+)$", re.MULTILINE)

    for match in pattern.finditer(content):
        checkbox, text = match.groups()
        tasks.append(
            ExtractedTask(
                text=text.strip(),
                is_complete=checkbox.lower() == "x",
                confidence=1.0,
            )
        )

    # Also try numbered lists with checkboxes
    numbered_pattern = re.compile(r"^\s*\d+[.)]\s*\[([ xX])\]\s*(.+)$", re.MULTILINE)
    for match in numbered_pattern.finditer(content):
        checkbox, text = match.groups()
        if not any(t.text == text.strip() for t in tasks):
            tasks.append(
                ExtractedTask(
                    text=text.strip(),
                    is_complete=checkbox.lower() == "x",
                    confidence=1.0,
                )
            )

    return tasks


def _regex_extract_section(body: str, section_names: list[str]) -> str:
    """Extract a section by name using regex."""
    # Build pattern for section header
    names_pattern = "|".join(re.escape(name) for name in section_names)

    # Try markdown headers
    header_pattern = re.compile(
        rf"(?:^|\n)#+\s*(?:{names_pattern})\s*\n(.*?)(?=\n#+|\n<!--|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = header_pattern.search(body)
    if match:
        return match.group(1).strip()

    # Try bold headers
    bold_pattern = re.compile(
        rf"(?:^|\n)\*\*(?:{names_pattern})\*\*:?\s*\n(.*?)(?=\n\*\*|\n#+|\n<!--|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = bold_pattern.search(body)
    if match:
        return match.group(1).strip()

    return ""


def regex_extract_scope(body: str) -> ScopeExtraction:
    """Extract scope, tasks, and acceptance criteria using regex."""
    result = ScopeExtraction(extraction_method="regex")

    # Handle status summary block
    summary_match = re.search(
        r"<!-- auto-status-summary:start -->(.*?)<!-- auto-status-summary:end -->",
        body,
        re.DOTALL,
    )
    if summary_match:
        body_to_parse = summary_match.group(1)
    else:
        body_to_parse = body

    # Extract scope
    scope_content = _regex_extract_section(
        body_to_parse, ["Scope", "Why", "Background", "Context", "Overview"]
    )
    result.scope = scope_content
    result.raw_sections["scope"] = scope_content

    # Extract tasks
    tasks_content = _regex_extract_section(
        body_to_parse, ["Tasks", "Task", "Task List", "To Do", "Todo", "Implementation"]
    )
    result.raw_sections["tasks"] = tasks_content
    result.tasks = _regex_extract_checkboxes(tasks_content)

    # Extract acceptance criteria
    ac_content = _regex_extract_section(
        body_to_parse,
        ["Acceptance Criteria", "Acceptance", "Definition of Done", "Done Criteria"],
    )
    result.raw_sections["acceptance"] = ac_content
    result.acceptance_criteria = _regex_extract_checkboxes(ac_content)

    # Add notes about extraction quality
    issues = []
    if not result.scope:
        issues.append("No scope section found")
    if not result.tasks:
        issues.append("No tasks found")
    if not result.acceptance_criteria:
        issues.append("No acceptance criteria found")

    result.extraction_notes = "; ".join(issues) if issues else "Clean extraction"

    return result


def llm_extract_scope(body: str, model: str = "gpt-4o-mini") -> ScopeExtraction:
    """Extract scope, tasks, and acceptance criteria using LLM.

    Falls back to regex if LangChain is not available or API fails.
    """
    if not LANGCHAIN_AVAILABLE:
        result = regex_extract_scope(body)
        result.extraction_notes = "LangChain not available, used regex fallback"
        return result

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        result = regex_extract_scope(body)
        result.extraction_notes = "OPENAI_API_KEY not set, used regex fallback"
        return result

    try:
        parser = PydanticOutputParser(pydantic_object=ScopeExtractionModel)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are extracting structured information from a GitHub PR or Issue body.
Your job is to find and extract:
1. **Scope**: The objective or purpose of the work
2. **Tasks**: A list of tasks/to-do items
3. **Acceptance Criteria**: Criteria for completion

For tasks and acceptance criteria:
- Determine if each checkbox is checked (complete) or unchecked
- A checkbox is checked if it shows [x] or [X]
- A checkbox is unchecked if it shows [ ]
- Items without checkboxes should be treated as unchecked tasks

Be tolerant of:
- Different heading formats (# Header, **Header**, Header:, #### Header)
- Nested lists
- Missing sections (return empty list/string for missing sections)
- Malformed checkboxes
- Content inside HTML comments or markdown blocks

{format_instructions}""",
                ),
                ("human", "{body}"),
            ]
        )

        llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
        chain = prompt | llm | parser

        llm_result = chain.invoke(
            {"body": body, "format_instructions": parser.get_format_instructions()}
        )

        # Convert to our dataclass
        result = ScopeExtraction(
            scope=llm_result.scope,
            tasks=[
                ExtractedTask(
                    text=t.text, is_complete=t.is_complete, confidence=t.confidence
                )
                for t in llm_result.tasks
            ],
            acceptance_criteria=[
                ExtractedTask(
                    text=t.text, is_complete=t.is_complete, confidence=t.confidence
                )
                for t in llm_result.acceptance_criteria
            ],
            extraction_method="llm",
            extraction_notes=llm_result.extraction_notes,
        )

        return result

    except Exception as e:
        # Fall back to regex on any error
        result = regex_extract_scope(body)
        result.extraction_notes = f"LLM extraction failed ({e}), used regex fallback"
        return result


def extract_structured_scope(
    body: str, use_llm: bool = True, model: str = "gpt-4o-mini"
) -> ScopeExtraction:
    """Extract scope, tasks, and acceptance criteria.

    Args:
        body: The PR/Issue body markdown
        use_llm: Whether to attempt LLM extraction (falls back to regex)
        model: OpenAI model to use if using LLM

    Returns:
        ScopeExtraction with all extracted data
    """
    if use_llm:
        return llm_extract_scope(body, model)
    return regex_extract_scope(body)


def compare_extractions(body: str) -> dict[str, Any]:
    """Compare regex vs LLM extraction for debugging."""
    regex_result = regex_extract_scope(body)
    llm_result = llm_extract_scope(body)

    return {
        "regex": {
            "scope_length": len(regex_result.scope),
            "tasks_count": len(regex_result.tasks),
            "tasks_complete": sum(1 for t in regex_result.tasks if t.is_complete),
            "acceptance_count": len(regex_result.acceptance_criteria),
            "acceptance_complete": sum(
                1 for a in regex_result.acceptance_criteria if a.is_complete
            ),
            "notes": regex_result.extraction_notes,
        },
        "llm": {
            "scope_length": len(llm_result.scope),
            "tasks_count": len(llm_result.tasks),
            "tasks_complete": sum(1 for t in llm_result.tasks if t.is_complete),
            "acceptance_count": len(llm_result.acceptance_criteria),
            "acceptance_complete": sum(
                1 for a in llm_result.acceptance_criteria if a.is_complete
            ),
            "notes": llm_result.extraction_notes,
        },
        "differences": {
            "tasks_count_diff": len(llm_result.tasks) - len(regex_result.tasks),
            "tasks_complete_diff": (
                sum(1 for t in llm_result.tasks if t.is_complete)
                - sum(1 for t in regex_result.tasks if t.is_complete)
            ),
        },
    }


def format_extraction_report(extraction: ScopeExtraction) -> str:
    """Format extraction result as markdown report."""
    lines = [
        f"# Extraction Report (method: {extraction.extraction_method})",
        "",
        "## Scope",
        extraction.scope or "_No scope found_",
        "",
        f"## Tasks ({len(extraction.tasks)} total)",
    ]

    for task in extraction.tasks:
        checkbox = "[x]" if task.is_complete else "[ ]"
        conf = f" (conf: {task.confidence:.0%})" if task.confidence < 1.0 else ""
        lines.append(f"- {checkbox} {task.text}{conf}")

    if not extraction.tasks:
        lines.append("_No tasks found_")

    lines.extend(
        [
            "",
            f"## Acceptance Criteria ({len(extraction.acceptance_criteria)} total)",
        ]
    )

    for ac in extraction.acceptance_criteria:
        checkbox = "[x]" if ac.is_complete else "[ ]"
        conf = f" (conf: {ac.confidence:.0%})" if ac.confidence < 1.0 else ""
        lines.append(f"- {checkbox} {ac.text}{conf}")

    if not extraction.acceptance_criteria:
        lines.append("_No acceptance criteria found_")

    lines.extend(
        [
            "",
            "## Extraction Notes",
            extraction.extraction_notes or "_No notes_",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, help="Path to PR body markdown file")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM extraction")
    parser.add_argument("--compare", action="store_true", help="Compare regex vs LLM")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.body:
        body = args.body.read_text()
    else:
        print("Reading from stdin...")
        body = sys.stdin.read()

    if args.compare:
        comparison = compare_extractions(body)
        print(json.dumps(comparison, indent=2))
        return 0

    extraction = extract_structured_scope(body, use_llm=args.use_llm, model=args.model)

    if args.json:
        output = {
            "scope": extraction.scope,
            "tasks": [
                {"text": t.text, "complete": t.is_complete, "confidence": t.confidence}
                for t in extraction.tasks
            ],
            "acceptance_criteria": [
                {"text": a.text, "complete": a.is_complete, "confidence": a.confidence}
                for a in extraction.acceptance_criteria
            ],
            "method": extraction.extraction_method,
            "notes": extraction.extraction_notes,
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_extraction_report(extraction))

    return 0


if __name__ == "__main__":
    sys.exit(main())
