from __future__ import annotations

from types import SimpleNamespace

import pytest

from trend import compat_entrypoints
from trend import cli as trend_cli


def _assert_help_exit(func, argv: list[str]) -> None:
    try:
        result = func(argv)
    except SystemExit as exc:
        assert exc.code == 0
    else:
        assert result == 0


def test_compat_entrypoints_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trend_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    wrappers = [
        compat_entrypoints.trend_analysis,
        compat_entrypoints.trend_multi_analysis,
        compat_entrypoints.trend_model,
        compat_entrypoints.trend_run,
        compat_entrypoints.trend_quick_report,
    ]

    for wrapper in wrappers:
        _assert_help_exit(wrapper, ["--help"])

    _assert_help_exit(compat_entrypoints.trend_model, ["run", "--help"])
    _assert_help_exit(compat_entrypoints.trend_app, ["--help"])
