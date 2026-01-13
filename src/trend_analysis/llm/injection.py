"""Prompt injection detection helpers for NL config updates."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_BLOCK_SUMMARY = "Unsafe instruction blocked by prompt-injection guard."


@dataclass(frozen=True)
class InjectionMatch:
    reason: str
    pattern: str


_PATTERNS: tuple[InjectionMatch, ...] = (
    InjectionMatch(
        reason="override_instructions",
        pattern=r"\b(ignore|disregard|bypass|override)\b.*\b(instructions?|rules|policies)\b",
    ),
    InjectionMatch(
        reason="system_prompt_exfil",
        pattern=r"\b(reveal|show|print|display|expose)\b.*\b(system prompt|developer message|hidden instructions?)\b",
    ),
    InjectionMatch(
        reason="system_prompt_reference",
        pattern=r"\b(system prompt|developer message|hidden instructions?)\b",
    ),
    InjectionMatch(
        reason="explicit_jailbreak",
        pattern=r"\b(prompt injection|jailbreak)\b",
    ),
    InjectionMatch(
        reason="tool_execution",
        pattern=r"\b(run|execute)\b.*\b(shell|bash|command|curl|wget|python)\b",
    ),
)


def detect_prompt_injection(instruction: str) -> list[str]:
    """Return matched injection reasons for a given instruction."""

    if not instruction:
        return []
    text = " ".join(instruction.lower().split())
    matches: list[str] = []
    for entry in _PATTERNS:
        if re.search(entry.pattern, text):
            matches.append(entry.reason)
    return matches


__all__ = ["DEFAULT_BLOCK_SUMMARY", "detect_prompt_injection"]
