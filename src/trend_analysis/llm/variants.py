"""Variant label normalization and validation."""

from __future__ import annotations

from collections.abc import Iterable

VARIANT_LABELS = ("conservative", "baseline", "aggressive")

_CANONICAL_LABELS = {label.casefold(): label for label in VARIANT_LABELS}


def normalize_variant_label(label: str) -> str:
    """Normalize a variant label to its canonical form."""

    stripped = label.strip()
    if not stripped:
        raise ValueError("label must be a non-empty string")
    normalized = stripped.casefold()
    if normalized not in _CANONICAL_LABELS:
        raise ValueError(
            "label must be one of: " + ", ".join(sorted(_CANONICAL_LABELS.values()))
        )
    return _CANONICAL_LABELS[normalized]


def find_duplicate_variant_labels(labels: Iterable[str]) -> list[str]:
    """Return canonical labels that appear more than once (case-insensitive)."""

    counts: dict[str, int] = {}
    for label in labels:
        key = label.casefold()
        counts[key] = counts.get(key, 0) + 1
    duplicates = {key for key, count in counts.items() if count > 1}
    if not duplicates:
        return []
    ordered: list[str] = []
    for label in VARIANT_LABELS:
        if label.casefold() in duplicates:
            ordered.append(label)
    for key in sorted(duplicates):
        canonical = _CANONICAL_LABELS.get(key, key)
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


def ensure_unique_variant_labels(labels: Iterable[str]) -> None:
    """Raise when labels collide case-insensitively."""

    duplicates = find_duplicate_variant_labels(labels)
    if duplicates:
        raise ValueError(
            "variants must have unique labels (case-insensitive): "
            + ", ".join(duplicates)
        )
