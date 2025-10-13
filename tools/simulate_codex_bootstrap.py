#!/usr/bin/env python3
"""Simulate the reusable agents bootstrap selection logic.

This helper mirrors the logic in the "Find Ready Issues" step inside
`.github/workflows/reusable-16-agents.yml`.  It demonstrates that the
workflow emits valid JSON that `fromJson(...)` can safely parse when
selecting the first issue for Codex bootstrapping.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class ReadyIssues:
    """Container for the outputs emitted by the GitHub Script step."""

    numbers: List[int]

    @property
    def issue_numbers_output(self) -> str:
        """Return the comma-separated list used by legacy consumers."""

        return ",".join(str(num) for num in self.numbers)

    @property
    def issue_numbers_json(self) -> str:
        """Return the JSON array string forwarded to `fromJson(...)`."""

        return json.dumps(self.numbers)

    @property
    def first_issue(self) -> str:
        """Return the first issue number or an empty string when absent."""

        return str(self.numbers[0]) if self.numbers else ""

    def evaluate_from_json(self) -> str:
        """Reproduce the workflow expression to fetch the first issue.

        The reusable workflow uses
        `fromJson(steps.ready.outputs.issue_numbers_json)[0]` to access the
        first issue number.  This helper mirrors the behaviour in Python to
        confirm the payload parses successfully.
        """

        parsed = json.loads(self.issue_numbers_json)
        if not parsed:
            return ""
        first = parsed[0]
        return str(first)


def parse_issue_numbers(raw_values: Iterable[str]) -> List[int]:
    numbers: List[int] = []
    for raw in raw_values:
        text = raw.strip()
        if not text:
            continue
        try:
            numbers.append(int(text))
        except ValueError as exc:  # pragma: no cover - argparse guards input
            raise argparse.ArgumentTypeError(
                f"Invalid issue number '{raw}': {exc}"
            ) from exc
    return numbers


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the Codex bootstrap readiness step by emitting the same "
            "outputs as the GitHub Actions workflow."
        )
    )
    parser.add_argument(
        "issues",
        metavar="ISSUE",
        nargs="*",
        help=(
            "Open issue numbers that would be detected by the workflow. "
            "Pass nothing to simulate an empty queue."
        ),
    )
    args = parser.parse_args(argv)

    numbers = parse_issue_numbers(args.issues)
    ready = ReadyIssues(numbers)

    print("Ready issues:", ready.numbers)
    print("issue_numbers output:", ready.issue_numbers_output or "<empty>")
    print("issue_numbers_json output:", ready.issue_numbers_json)
    print("first_issue output:", ready.first_issue or "<empty>")

    expression_value = ready.evaluate_from_json()
    if expression_value:
        print(
            "fromJson(steps.ready.outputs.issue_numbers_json)[0] =>",
            expression_value,
        )
    else:
        print("fromJson(...) result: <empty queue>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
