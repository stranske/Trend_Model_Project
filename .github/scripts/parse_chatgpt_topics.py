#!/usr/bin/env python3
"""Parse ChatGPT topics from input.txt and produce topics.json."""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

INPUT_PATH = Path("input.txt")
OUTPUT_PATH = Path("topics.json")


def _load_text() -> str:
    try:
        text = INPUT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:  # pragma: no cover - guardrail for workflow execution
        raise SystemExit("No input.txt found to parse.") from exc
    if not text:
        raise SystemExit("No topic content provided.")
    return text


def _split_numbered_items(text: str) -> list[dict[str, str | list[str]]]:
    numbered_pattern = re.compile(r"^\s*\d+[\).]\s+", re.MULTILINE)
    items: list[dict[str, str | list[str]]] = []
    current: dict[str, str | list[str]] | None = None
    for raw_line in text.splitlines():
        if numbered_pattern.match(raw_line):
            title = numbered_pattern.sub("", raw_line, count=1).strip()
            if current:
                items.append(current)
            current = {"title": title, "lines": []}
        else:
            if current is None:
                continue
            current.setdefault("lines", []).append(raw_line.rstrip("\n"))
    if current:
        items.append(current)
    if not items:
        raise SystemExit("No numbered topics were found in the provided text.")
    return items


def _parse_sections(raw_lines: list[str]) -> tuple[list[str], dict[str, list[str]], list[str]]:
    section_aliases: dict[str, set[str]] = {
        "why": {"why"},
        "tasks": {"tasks"},
        "acceptance_criteria": {"acceptance criteria", "acceptance criteria."},
        "implementation_notes": {"implementation notes", "implementation note", "notes"},
    }

    labels: list[str] = []
    remaining: list[str] = []
    label_found = False
    for line in raw_lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if not label_found and lowered.startswith("labels"):
            label_found = True
            _, _, remainder = stripped.partition(":")
            if remainder:
                parts = re.split(r"[;,]", remainder)
                labels.extend([part.strip() for part in parts if part.strip()])
            continue
        remaining.append(line)

    sections: dict[str, list[str]] = {key: [] for key in section_aliases}
    extras: list[str] = []
    current_section: str | None = None

    for line in remaining:
        stripped = line.strip()
        if stripped == "":
            if current_section:
                sections[current_section].append("")
            continue
        normalized = re.sub(r"[^a-z0-9 ]+", " ", stripped.lower()).strip()
        normalized = normalized.rstrip(":").strip()
        matched_section = None
        for key, aliases in section_aliases.items():
            if normalized in aliases:
                matched_section = key
                break
        if matched_section:
            current_section = matched_section
            continue
        if current_section:
            sections[current_section].append(line)
        else:
            extras.append(line)

    return labels, sections, extras


def _join_section(lines: list[str]) -> str:
    return "\n".join(lines).strip()


def parse_topics() -> list[dict[str, object]]:
    text = _load_text()
    items = _split_numbered_items(text)

    parsed: list[dict[str, object]] = []
    for item in items:
        raw_lines = list(item.get("lines", []))
        labels, sections, extras = _parse_sections(raw_lines)
        data = {
            "title": item["title"],
            "labels": labels,
            "sections": {key: _join_section(value) for key, value in sections.items()},
            "extras": _join_section(extras),
        }
        normalized_title = re.sub(r"\s+", " ", item["title"].strip().lower())
        data["guid"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, normalized_title))
        parsed.append(data)

    return parsed


def main() -> None:
    parsed = parse_topics()
    OUTPUT_PATH.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"Parsed {len(parsed)} topic(s).")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        raise
