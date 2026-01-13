"""Prompt injection detection helpers for NL config updates."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

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


def detect_prompt_injection_payload(
    *, instruction: str, current_config: Any | None = None
) -> list[str]:
    """Detect prompt-injection patterns across user-controlled inputs."""

    matches = set(detect_prompt_injection(instruction))
    if current_config is None:
        return sorted(matches)
    for text in _iter_text_values(current_config):
        for reason in detect_prompt_injection(text):
            matches.add(f"config_{reason}")
    return sorted(matches)


def _iter_text_values(payload: Any) -> Iterable[str]:
    if isinstance(payload, str):
        yield payload
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(key, str):
                yield key
            yield from _iter_text_values(value)
        return
    if isinstance(payload, (list, tuple, set)):
        for item in payload:
            yield from _iter_text_values(item)
        return


__all__ = [
    "DEFAULT_BLOCK_SUMMARY",
    "detect_prompt_injection",
    "detect_prompt_injection_payload",
]
