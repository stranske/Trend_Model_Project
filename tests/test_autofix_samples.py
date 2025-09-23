from __future__ import annotations

import runpy

import pytest

from trend_analysis import (
    _autofix_trigger_sample,
    _autofix_violation_case2,
    _autofix_violation_case3,
)


def test_trigger_sample_functions_behave_as_documented() -> None:
    assert _autofix_trigger_sample.badly_formatted_function(3, 7) == 10
    assert _autofix_trigger_sample.another_func([1, 2], [3, 4]) == [4, 6]
    assert _autofix_trigger_sample.Demo().method(1.5) == 3.0
    assert "overly verbose" in _autofix_trigger_sample.long_line()


def test_violation_case2_compute_and_helpers() -> None:
    payload = _autofix_violation_case2.compute([2, 4, 6])
    assert payload == {"total": 12, "mean": 4.0, "count": 3}

    # Exercise the default branch and the intentionally "unused" helper.
    default_payload = _autofix_violation_case2.compute()
    assert default_payload == {"total": 6, "mean": 2.0, "count": 3}

    assert _autofix_violation_case2.Example().method(3.5, 0.5) == 4.0
    assert "extravagantly" in _autofix_violation_case2.long_line_function()
    assert _autofix_violation_case2.unused_func(1, 2, 3) is None


def test_violation_case2_module_invocation(capsys: pytest.CaptureFixture[str]) -> None:
    module = _autofix_violation_case2.__name__
    # Importing via runpy ensures the module level ``__name__`` branch executes
    # without spawning a subprocess, keeping the test hermetic.
    runpy.run_module(module, run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip() == str(_autofix_violation_case2.compute())


def test_violation_case3_exposes_expected_behaviour() -> None:
    assert _autofix_violation_case3.compute_sum(5, 7) == 12
    assert _autofix_violation_case3.list_builder([1, 2, 3]) == [1, 2, 3]
    assert _autofix_violation_case3.ambiguous_types([1, 2], [4, 6]) == [5, 8]

    container = _autofix_violation_case3.SomeContainer([2, 3, 5])
    assert container.total() == 10
