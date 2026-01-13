"""Safety checks for NL prompt handling."""

from __future__ import annotations

import re

_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|override)\b[\s\S]{0,80}\b(instruction|instructions|rules)\b",
        re.IGNORECASE,
    ),
    "system_prompt_leak": re.compile(
        r"\b(show|reveal|leak|print|dump)\b[\s\S]{0,80}\b(system prompt|system message|developer message)\b",
        re.IGNORECASE,
    ),
    "role_override": re.compile(
        r"\b(you are|act as)\b[\s\S]{0,80}\b(system|developer)\b",
        re.IGNORECASE,
    ),
    "tool_execution": re.compile(
        r"\b(run|execute)\b[\s\S]{0,80}\b(shell|bash|terminal|command)\b",
        re.IGNORECASE,
    ),
    "jailbreak_keyword": re.compile(r"\b(jailbreak|do anything now|dan)\b", re.IGNORECASE),
}


def detect_prompt_injection(instruction: str) -> list[str]:
    """Return matching injection pattern names for a user instruction."""

    matches: list[str] = []
    if not isinstance(instruction, str):
        return matches
    for name, pattern in _INJECTION_PATTERNS.items():
        if pattern.search(instruction):
            matches.append(name)
    return matches


def guard_instruction(instruction: str) -> None:
    """Raise if the instruction appears to contain prompt injection attempts."""

    matches = detect_prompt_injection(instruction)
    if matches:
        joined = ", ".join(matches)
        raise ValueError(f"SecurityError: Prompt injection detected: {joined}")


__all__ = ["detect_prompt_injection", "guard_instruction"]
