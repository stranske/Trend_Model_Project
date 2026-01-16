"""Prompt injection detection helpers for NL config updates."""

from __future__ import annotations

import base64
import html
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote_plus

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
        pattern=r"\b(reveal|show|print|display|expose)\b.*\b(system[-\s]?prompt|developer[-\s]?message|hidden instructions?)\b",
    ),
    InjectionMatch(
        reason="system_prompt_reference",
        pattern=r"\b(system[-\s]?prompt|developer[-\s]?message|hidden instructions?)\b",
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

_COMPACT_TRIGGERS: tuple[InjectionMatch, ...] = (
    InjectionMatch(
        reason="override_instructions",
        pattern="ignorepreviousinstructions|disregardinstructions|overrideinstructions|bypassinstructions",
    ),
    InjectionMatch(
        reason="system_prompt_exfil",
        pattern=(
            "revealsystemprompt|showsystemprompt|printsystemprompt|displaysystemprompt|exposesystemprompt"
            "|revealdevelopermessage|showdevelopermessage|printdevelopermessage|displaydevelopermessage"
            "|exposehiddeninstructions"
        ),
    ),
    InjectionMatch(
        reason="system_prompt_reference",
        pattern="systemprompt|developermessage|hiddeninstructions",
    ),
    InjectionMatch(
        reason="explicit_jailbreak",
        pattern="promptinjection|jailbreak",
    ),
    InjectionMatch(
        reason="tool_execution",
        pattern="runbash|executeshell|runcommand|runcurl|runwget|runpython",
    ),
)


def detect_prompt_injection(instruction: str) -> list[str]:
    """Return matched injection reasons for a given instruction."""

    if not instruction:
        return []
    matches: list[str] = []
    _extend_unique(matches, _detect_patterns(_normalize_text(instruction), _PATTERNS))
    _extend_unique(matches, _detect_patterns(_compact_text(instruction), _COMPACT_TRIGGERS))
    for decoded in _iter_decoded_variants(instruction):
        _extend_unique(matches, _detect_patterns(_normalize_text(decoded), _PATTERNS))
        _extend_unique(matches, _detect_patterns(_compact_text(decoded), _COMPACT_TRIGGERS))
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


def _detect_patterns(text: str, patterns: Iterable[InjectionMatch]) -> list[str]:
    reasons: list[str] = []
    for entry in patterns:
        if re.search(entry.pattern, text):
            reasons.append(entry.reason)
    return reasons


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _iter_decoded_variants(text: str) -> Iterable[str]:
    html_decoded = html.unescape(text)
    if html_decoded and html_decoded != text:
        yield html_decoded
    decoded = unquote_plus(text)
    if decoded and decoded != text:
        yield decoded
    base64_decoded = _maybe_decode_base64(text)
    if base64_decoded:
        yield base64_decoded
    hex_decoded = _maybe_decode_hex(text)
    if hex_decoded:
        yield hex_decoded


def _maybe_decode_base64(text: str) -> str | None:
    candidate = "".join(text.split())
    if len(candidate) < 16:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", candidate):
        return None
    try:
        decoded = base64.b64decode(candidate, validate=True)
    except Exception:
        return None
    try:
        decoded_text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not decoded_text or not any(ch.isalpha() for ch in decoded_text):
        return None
    return decoded_text


def _maybe_decode_hex(text: str) -> str | None:
    candidate = "".join(text.split())
    if len(candidate) < 16 or len(candidate) % 2:
        return None
    if not re.fullmatch(r"[A-Fa-f0-9]+", candidate):
        return None
    try:
        decoded_text = bytes.fromhex(candidate).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    if not decoded_text or not any(ch.isalpha() for ch in decoded_text):
        return None
    return decoded_text


__all__ = [
    "DEFAULT_BLOCK_SUMMARY",
    "detect_prompt_injection",
    "detect_prompt_injection_payload",
]
