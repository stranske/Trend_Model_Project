"""CI Probe Module

This module intentionally contains multiple small issues to exercise the
repository's validation layers (lint, type-check, tests) when opened as a PR.

DO NOT FIX in the probe PR: the failures are the point. A follow-up PR can
resolve them to demonstrate the green path.
"""


UNUSED_CONSTANT = (
    42  # This constant is never used (might not always be flagged but okay)
)

# Intentionally unused variable with an excessively long line kept inside a string so black does not wrap it automatically but flake8 should still measure length
LONG_NO_WRAP = """THIS_IS_A_PURPOSEFULLY_LONG_STRING_USED_TO_EXCEED_THE_STANDARD_LINE_LENGTH_LIMIT_SO_THAT_THE_LINTER_E501_RULE_IS_TRIGGERED_WITHOUT_BLACK_REFORMATTING_BECAUSE_IT_RESIDES_INSIDE_A_TRIPLE_QUOTED_STRING_LITERAL_AND_SHOULD_REMAIN_LONG"""  # noqa: E501

# E501: overly long line below ( > 120 chars ) .................................................................................................


def compute_value(x: int, y: str) -> int:
    """Return x squared plus length of y, but contains a deliberate type error usage."""
    # F821: reference to undefined variable 'z'
    return x * x + len(y)


def wrong_return_type(flag: bool) -> str:
    """Declared to return str but actually returns an int; mypy should complain under strict mode."""
    if flag:
        return 123  # type: ignore[return-value]
    return 456  # type: ignore[return-value]


# Deliberate shadowing & unused variable pattern
for i in range(0):  # pragma: no cover
    i = 5  # noqa: F841 (local variable assigned but not used)

# A function with mismatched annotation vs usage


def expects_list(values: list[int]) -> int:
    """Incorrectly calls with a str in probe harness (added in docstring)."""
    return sum(values)


# NOTE: Additional potential failure triggers (commented out for controlled scope):
# 1. SyntaxError example (kept disabled to avoid stopping rest of lint):
# def broken_func(:  # noqa
#     pass
# 2. Import of missing module to trigger import-error:
# import definitely_not_a_real_module  # noqa: F401

__all__ = [
    "compute_value",
    "wrong_return_type",
    "expects_list",
]
