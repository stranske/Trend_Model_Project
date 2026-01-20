from __future__ import annotations

from types import SimpleNamespace

import pytest

from trend import cli as trend_cli
from trend import compat_entrypoints


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


def _assert_warned(
    capsys: pytest.CaptureFixture[str],
    func,
    argv: list[str],
    expected_warning: str,
) -> None:
    try:
        result = func(argv)
    except SystemExit as exc:
        assert exc.code == 0
    else:
        assert result == 0
    stderr = capsys.readouterr().err
    assert expected_warning in stderr


def test_compat_entrypoints_emit_deprecation_warnings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        trend_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    warnings = [
        (
            compat_entrypoints.trend_analysis,
            "Warning: 'trend-analysis' is deprecated; use 'trend run' instead.",
        ),
        (
            compat_entrypoints.trend_multi_analysis,
            "Warning: 'trend-multi-analysis' is deprecated; use 'trend run' instead.",
        ),
        (
            compat_entrypoints.trend_model,
            "Warning: 'trend-model' is deprecated; use 'trend' instead.",
        ),
        (
            compat_entrypoints.trend_app,
            "Warning: 'trend-app' is deprecated; use 'trend app' instead.",
        ),
        (
            compat_entrypoints.trend_run,
            "Warning: 'trend-run' is deprecated; use 'trend report' instead.",
        ),
        (
            compat_entrypoints.trend_quick_report,
            "Warning: 'trend-quick-report' is deprecated; use 'trend quick-report' instead.",
        ),
    ]

    for wrapper, expected in warnings:
        _assert_warned(capsys, wrapper, ["--help"], expected)
