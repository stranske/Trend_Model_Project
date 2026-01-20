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


def test_compat_entrypoints_map_to_trend_subcommands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str]] = []

    def fake_main(argv: list[str] | None = None, *, prog: str = "trend") -> int:
        calls.append((list(argv or []), prog))
        return 0

    monkeypatch.setattr(trend_cli, "main", fake_main)

    compat_entrypoints.trend_analysis(["--help"])
    compat_entrypoints.trend_multi_analysis(["--help"])
    compat_entrypoints.trend_model(["--check"])
    compat_entrypoints.trend_model(["gui", "--help"])
    compat_entrypoints.trend_model(["run", "-c", "cfg.yml", "-i", "data.csv"])
    compat_entrypoints.trend_app(["--help"])
    compat_entrypoints.trend_run(["--output", "out.html"])
    compat_entrypoints.trend_run([])
    compat_entrypoints.trend_quick_report(["--help"])

    assert calls == [
        (["run", "--help"], "trend"),
        (["run", "--help"], "trend"),
        (["check"], "trend-model"),
        (["app", "--help"], "trend-model"),
        (["run", "-c", "cfg.yml", "--returns", "data.csv"], "trend-model"),
        (["app", "--help"], "trend"),
        (["report", "--output", "out.html"], "trend"),
        (["report", "--output", "reports/trend_report.html"], "trend"),
        (["quick-report", "--help"], "trend"),
    ]


def test_trend_run_artefacts_skip_default_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str]] = []

    def fake_main(argv: list[str] | None = None, *, prog: str = "trend") -> int:
        calls.append((list(argv or []), prog))
        return 0

    monkeypatch.setattr(trend_cli, "main", fake_main)

    compat_entrypoints.trend_run(["--artefacts", "outdir"])
    compat_entrypoints.trend_run(["--artefacts=otherdir"])

    assert calls == [
        (["report", "--out", "outdir"], "trend"),
        (["report", "--out=otherdir"], "trend"),
    ]
