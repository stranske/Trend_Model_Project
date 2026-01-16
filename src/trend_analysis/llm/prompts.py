"""Prompt templates for NL config patch generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import yaml

SECTION_SYSTEM = "SYSTEM PROMPT"
SECTION_CONFIG = "CURRENT CONFIG"
SECTION_SCHEMA = "ALLOWED SCHEMA"
SECTION_SAFETY = "SAFETY RULES"
SECTION_USER = "USER INSTRUCTION"
SECTION_RETRY_ERROR = "PREVIOUS ERROR"
SECTION_RESULT_SYSTEM = "RESULT SUMMARY SYSTEM PROMPT"
SECTION_RESULT_OUTPUT = "ANALYSIS OUTPUT"
SECTION_RESULT_METRICS = "METRIC CATALOG"
SECTION_RESULT_QUESTIONS = "RESULT QUESTIONS"
SECTION_RESULT_RULES = "RESULT SAFETY RULES"

DEFAULT_SYSTEM_PROMPT = """You are a configuration assistant for Trend Model.
Your task is to read the user instruction and current configuration, then emit
a ConfigPatch JSON object that updates the config safely and minimally.

Return ONLY a valid JSON object that conforms exactly to the ConfigPatch schema.
Include only operations that are necessary to implement the instruction.
Never add keys outside the ConfigPatch schema or output non-JSON content.
Do not invent keys; if the instruction or config mentions unknown or extraneous
keys, flag them explicitly in the summary and return empty operations.
If asked to target unknown keys or unsafe content, return empty operations and
explain the refusal in the summary without echoing the unsafe request.
"""

DEFAULT_RESULT_SYSTEM_PROMPT = """You are a results explanation assistant for Trend Model.
Your task is to summarize the analysis output and answer result questions in
plain language using only the provided metrics and output snippets.

Every claim must cite specific metric values with explicit source references
like "[from out_sample_stats]" that match the provided metric catalog.
If a metric is missing or unavailable, say so explicitly and do not guess.
Include this disclaimer verbatim at the end of every response:
"This is analytical output, not financial advice. Always verify metrics independently."
"""

DEFAULT_SAFETY_RULES = (
    "Use only keys that exist in the allowed schema or current config.",
    "Do not invent new keys or aliases; unknown keys must be explicitly flagged in the summary.",
    "Reject and flag extraneous keys in the instruction or config; do not pass them through.",
    "Do not include any keys beyond operations, risk_flags, and summary.",
    "If the instruction asks for unknown keys or unsafe content, return no operations and explain why in summary.",
    "Never output markdown, prose, or any content that is not a single JSON object.",
    "Flag risky changes in risk_flags when appropriate (constraints, leverage, validations).",
    "Keep patch operations minimal and ordered in the sequence they should apply.",
    "Never include secrets, credentials, or unsafe content in any field.",
)

DEFAULT_RESULT_RULES = (
    "Use only metrics present in the analysis output and metric catalog.",
    "Cite every claim with the metric name and source reference in brackets.",
    "If a question requests unavailable data, respond with unavailability.",
    "Do not infer or estimate values beyond what is provided.",
    "Keep summaries concise and grounded in quantitative evidence.",
)


def format_config_for_prompt(config: Any) -> str:
    """Render a config mapping or excerpt as YAML for prompt injection."""

    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False).strip()


def build_config_patch_prompt(
    *,
    current_config: str,
    allowed_schema: str,
    instruction: str,
    system_prompt: str | None = None,
    safety_rules: Iterable[str] | None = None,
) -> str:
    """Build the prompt text for ConfigPatch generation."""

    system_text = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
    rules = list(safety_rules or DEFAULT_SAFETY_RULES)
    safety_text = "\n".join(f"- {rule}" for rule in rules)
    sections = [
        _format_section(SECTION_SYSTEM, system_text),
        _format_section(SECTION_CONFIG, current_config.strip()),
        _format_section(SECTION_SCHEMA, allowed_schema.strip()),
        _format_section(SECTION_SAFETY, safety_text),
        _format_section(SECTION_USER, instruction.strip()),
    ]
    return "\n\n".join(sections).strip()


def build_retry_prompt(
    *,
    current_config: str,
    allowed_schema: str,
    instruction: str,
    error_message: str,
    system_prompt: str | None = None,
    safety_rules: Iterable[str] | None = None,
) -> str:
    """Build the retry prompt with previous parsing error context."""

    base_prompt = build_config_patch_prompt(
        current_config=current_config,
        allowed_schema=allowed_schema,
        instruction=instruction,
        system_prompt=system_prompt,
        safety_rules=safety_rules,
    )
    retry_note = (
        f"{error_message}\n\n"
        "Return ONLY a valid JSON object that matches the ConfigPatch schema."
    )
    sections = [
        base_prompt,
        _format_section(SECTION_RETRY_ERROR, retry_note),
    ]
    return "\n\n".join(sections).strip()


def build_result_summary_prompt(
    *,
    analysis_output: str,
    metric_catalog: str,
    questions: str,
    system_prompt: str | None = None,
    safety_rules: Iterable[str] | None = None,
) -> str:
    """Build the prompt text for analysis result summaries and Q&A."""

    system_text = (system_prompt or DEFAULT_RESULT_SYSTEM_PROMPT).strip()
    rules = list(safety_rules or DEFAULT_RESULT_RULES)
    safety_text = "\n".join(f"- {rule}" for rule in rules)
    sections = [
        _format_section(SECTION_RESULT_SYSTEM, system_text),
        _format_section(SECTION_RESULT_OUTPUT, analysis_output.strip()),
        _format_section(SECTION_RESULT_METRICS, metric_catalog.strip()),
        _format_section(SECTION_RESULT_RULES, safety_text),
        _format_section(SECTION_RESULT_QUESTIONS, questions.strip()),
    ]
    return "\n\n".join(sections).strip()


def _format_section(title: str, body: str) -> str:
    return f"## {title}\n{body}".strip()


__all__ = [
    "SECTION_SYSTEM",
    "SECTION_CONFIG",
    "SECTION_SCHEMA",
    "SECTION_SAFETY",
    "SECTION_USER",
    "SECTION_RETRY_ERROR",
    "SECTION_RESULT_SYSTEM",
    "SECTION_RESULT_OUTPUT",
    "SECTION_RESULT_METRICS",
    "SECTION_RESULT_QUESTIONS",
    "SECTION_RESULT_RULES",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_RESULT_SYSTEM_PROMPT",
    "DEFAULT_SAFETY_RULES",
    "DEFAULT_RESULT_RULES",
    "format_config_for_prompt",
    "build_config_patch_prompt",
    "build_retry_prompt",
    "build_result_summary_prompt",
]
