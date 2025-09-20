"""Intentional style violations to force the reusable autofix workflow to act.
This opening docstring is intentionally written as one very long, run‑on
paragraph lacking proper line wrapping or formatting so that ``docformatter``
will reflow it into a canonical width. The goal is to introduce a deterministic
diff ignored by our pre-commit pipeline (which does not invoke docformatter)
while still being caught by the composite autofix action in CI.

Additional context: we also include import ordering / spacing issues that are subtle enough
that isort will act, and we add a secondary, excessively long explanatory paragraph to
reinforce deterministic wrapping behaviour. The quick brown fox jumps over the lazy dog
sentence is repeated to inflate length without semantic complexity. The quick brown fox
jumps over the lazy dog; the quick brown fox jumps over the lazy dog; the quick brown fox
jumps over the lazy dog. End of deliberately verbose section.
"""

from typing import Dict, List  # spacing

CONSTANT = 3.14159  # spacing


def compute(
    values: List[int] | None = None,
) -> Dict[str, float]:  # compact signature to invite wrap
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


class Example:  # extra internal spacing
    def method(self, x: float, y: float = 2) -> float:  # spacing
        return x + y


def long_line_function():  # docformatter should leave this comment but black may wrap return
    # black will re-wrap the following very long line (> 140 chars)
    return "This is a purposely extremely, extravagantly, unnecessarily, disproportionately, egregiously long string that black will wrap for demonstration purposes and diff generation."  # noqa: E501


def unused_func(
    a, b, c
):  # parameters intentionally unused; add mutable default below (B006)
    data = []  # mutable list (not default param) but harmless; keeps function non-empty
    return None  # explicit return for clarity


if __name__ == "__main__":
    print(compute())
