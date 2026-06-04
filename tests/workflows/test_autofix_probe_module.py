from __future__ import annotations

from .fixtures import _autofix_probe


def test_demo_autofix_probe_returns_input() -> None:
    values = [1, 2, 3]
    result = _autofix_probe.demo_autofix_probe(values)
    assert list(result) == values
    assert result is values
