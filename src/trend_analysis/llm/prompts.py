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

DEFAULT_SYSTEM_PROMPT = """You are a configuration assistant for Trend Model.
Your task is to read the user instruction and current configuration, then emit
a ConfigPatch JSON object that updates the config safely and minimally.

Return ONLY a valid JSON object that conforms exactly to the ConfigPatch schema.
Include only operations that are necessary to implement the instruction.
Never add keys outside the ConfigPatch schema or output non-JSON content.
"""

DEFAULT_SAFETY_RULES = (
    "Use only keys that exist in the allowed schema or current config.",
    "Do not invent new keys; if a request targets an unknown key, call it out in the summary.",
    "Do not include any keys beyond operations, risk_flags, and summary.",
    "Flag risky changes in risk_flags when appropriate (constraints, leverage, validations).",
    "Keep patch operations minimal and ordered in the sequence they should apply.",
    "Never include secrets, credentials, or unsafe content in any field.",
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


def _format_section(title: str, body: str) -> str:
    return f"## {title}\n{body}".strip()


__all__ = [
    "SECTION_SYSTEM",
    "SECTION_CONFIG",
    "SECTION_SCHEMA",
    "SECTION_SAFETY",
    "SECTION_USER",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_SAFETY_RULES",
    "format_config_for_prompt",
    "build_config_patch_prompt",
]
