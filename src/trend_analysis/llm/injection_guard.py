"""Prompt injection detection helpers for NL config instructions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class PromptInjectionPattern:
    label: str
    regex: re.Pattern[str]


_PATTERNS: tuple[PromptInjectionPattern, ...] = (
    PromptInjectionPattern(
        label="ignore_instructions",
        regex=re.compile(
            r"\b(ignore|disregard|override)\b.*\b(instruction|instructions|system|developer|previous)\b",
            re.IGNORECASE,
        ),
    ),
    PromptInjectionPattern(
        label="system_prompt",
        regex=re.compile(
            r"\b(system prompt|developer message|hidden prompt|initial prompt)\b",
            re.IGNORECASE,
        ),
    ),
    PromptInjectionPattern(
        label="reveal_prompt",
        regex=re.compile(
            r"\b(reveal|show|print|leak|expose)\b.*\b(prompt|instructions|system)\b",
            re.IGNORECASE,
        ),
    ),
    PromptInjectionPattern(
        label="role_play",
        regex=re.compile(
            r"\b(act as|pretend to be|simulate)\b.*\b(system|developer|assistant)\b",
            re.IGNORECASE,
        ),
    ),
    PromptInjectionPattern(
        label="jailbreak",
        regex=re.compile(r"\bjailbreak\b", re.IGNORECASE),
    ),
    PromptInjectionPattern(
        label="assistant_identity",
        regex=re.compile(r"\byou are (chatgpt|an? ai|a language model)\b", re.IGNORECASE),
    ),
)


def detect_prompt_injection(instruction: str | None) -> list[str]:
    """Return labels for prompt injection patterns found in the instruction."""

    if not instruction:
        return []
    return [pattern.label for pattern in _PATTERNS if pattern.regex.search(instruction)]


def prompt_injection_summary(matches: Iterable[str]) -> str:
    """Return a generic refusal summary without echoing the instruction."""

    _ = list(matches)
    return "Refused: suspected prompt injection. No changes applied."


__all__ = ["detect_prompt_injection", "prompt_injection_summary", "PromptInjectionPattern"]
