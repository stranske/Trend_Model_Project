"""Prompt injection detection helpers for NL config updates."""

from __future__ import annotations

import base64
import codecs
import html
import re
import unicodedata
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
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.lower().split())


def _compact_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"[^a-z0-9]+", "", normalized.lower())


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _iter_decoded_variants(text: str) -> Iterable[str]:
    seen: set[str] = set()
    second_pass = (
        html.unescape(unquote_plus(text)),
        unquote_plus(html.unescape(text)),
        unquote_plus(unquote_plus(text)),
    )
    for decoded in (html.unescape(text), unquote_plus(text)):
        if decoded and decoded != text and decoded not in seen:
            seen.add(decoded)
            yield decoded
        for nested in (
            _maybe_decode_base64(decoded),
            _maybe_decode_hex(decoded),
            _maybe_decode_rot13(decoded),
            _maybe_decode_unicode_escape(decoded),
        ):
            if nested and nested not in seen:
                seen.add(nested)
                yield nested
    for decoded in second_pass:
        if decoded and decoded != text and decoded not in seen:
            seen.add(decoded)
            yield decoded
        for nested in (
            _maybe_decode_base64(decoded),
            _maybe_decode_hex(decoded),
            _maybe_decode_rot13(decoded),
            _maybe_decode_unicode_escape(decoded),
        ):
            if nested and nested not in seen:
                seen.add(nested)
                yield nested
    for decoded_variant in (
        _maybe_decode_base64(text),
        _maybe_decode_hex(text),
        _maybe_decode_rot13(text),
        _maybe_decode_unicode_escape(text),
    ):
        if decoded_variant and decoded_variant not in seen:
            seen.add(decoded_variant)
            yield decoded_variant


def _maybe_decode_base64(text: str) -> str | None:
    candidate = "".join(text.split())
    lowered = candidate.lower()
    if "base64," in lowered:
        candidate = candidate[lowered.index("base64,") + len("base64,") :]
        lowered = candidate.lower()
    elif lowered.startswith("base64:"):
        candidate = candidate[len("base64:") :]
        lowered = candidate.lower()
    if len(candidate) < 16:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/_=-]+", candidate):
        return None
    padded = candidate + ("=" * ((4 - (len(candidate) % 4)) % 4))
    try:
        if "-" in candidate or "_" in candidate:
            decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
        else:
            decoded = base64.b64decode(padded, validate=True)
    except Exception:
        return None
    try:
        decoded_text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not decoded_text or not any(ch.isalpha() for ch in decoded_text):
        return None
    return decoded_text


def _decode_hex_candidate(candidate: str) -> str | None:
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


def _maybe_decode_hex(text: str) -> str | None:
    candidate = "".join(text.split())
    if candidate.lower().startswith("0x"):
        candidate = candidate[2:]
    decoded_text = _decode_hex_candidate(candidate)
    if decoded_text:
        return decoded_text
    if "0x" in text.lower():
        hex_pairs = re.findall(r"0x([0-9a-fA-F]{2})", text)
        if hex_pairs and len(hex_pairs) * 2 >= 16:
            return _decode_hex_candidate("".join(hex_pairs))
    return None


def _maybe_decode_rot13(text: str) -> str | None:
    if not text or not re.search(r"[A-Za-z]", text):
        return None
    decoded_text = codecs.decode(text, "rot_13")
    if decoded_text == text:
        return None
    if not decoded_text or not any(ch.isalpha() for ch in decoded_text):
        return None
    return decoded_text


def _maybe_decode_unicode_escape(text: str) -> str | None:
    if not text or not re.search(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}", text):
        return None
    try:
        decoded_text = codecs.decode(text, "unicode_escape")
    except Exception:
        return None
    if decoded_text == text:
        return None
    if not decoded_text or not any(ch.isalpha() for ch in decoded_text):
        return None
    return decoded_text


__all__ = [
    "DEFAULT_BLOCK_SUMMARY",
    "detect_prompt_injection",
    "detect_prompt_injection_payload",
]
