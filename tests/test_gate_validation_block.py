"""Temporary failing test to verify Gate branch protection.

This file is committed intentionally to force the Gate workflow to fail on
the validation pull request. It will be removed once the failing state has
been recorded for the acceptance evidence bundle.
"""


def test_gate_validation_failure() -> None:
    raise AssertionError("Intentional Gate validation failure")
