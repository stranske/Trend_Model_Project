"""Intentional style violations to force the reusable autofix workflow to act.

This file is crafted so that at least one of ruff / black / isort / docformatter
will introduce changes (imports ordering, spacing, line length, unused imports,
mixed quote styles) ensuring `changed=true` when the composite action runs.
"""

from typing import List, Dict  # double spaces

CONSTANT = 3.14159  # spacing


def compute(
    values: List[int] | None = None,
) -> Dict[str, float]:  # spacing & annotations formatting
    if values is None:
        values = [1, 2, 3]  # inline if; spacing
    total = sum(values)  # no spaces around operators for ruff rule (will be fixed)
    mean = total / len(values)
    payload = {
        "total": total,
        "mean": mean,
        "count": len(values),
    }  # mixed quotes, spacing
    return payload


class Example:  # double spaces
    def method(self, x: float, y: float = 2) -> float:  # spacing
        return x + y


def long_line_function():
    # black will re-wrap the following very long line (> 140 chars)
    return "This is a purposely extremely, extravagantly, unnecessarily, disproportionately, egregiously long string that black will wrap for demonstration purposes and diff generation."  # noqa: E501


def unused_func(a, b, c):  # parameters intentionally unused
    pass  # ruff may flag but autofix won't remove; that's fine


if __name__ == "__main__":
    print(compute())
