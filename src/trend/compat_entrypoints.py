from __future__ import annotations

import sys
from collections.abc import Sequence

from trend import cli as trend_cli

DEFAULT_TREND_RUN_OUTPUT = "reports/trend_report.html"


def _warn_deprecated(old: str, new: str) -> None:
    print(f"Warning: '{old}' is deprecated; use '{new}' instead.", file=sys.stderr)


def _drop_detailed(args: list[str]) -> tuple[list[str], bool]:
    cleaned: list[str] = []
    removed = False
    for arg in args:
        if arg == "--detailed" or arg.startswith("--detailed="):
            removed = True
            continue
        cleaned.append(arg)
    return cleaned, removed


def _translate_trend_model_run_args(args: list[str]) -> list[str]:
    translated: list[str] = []
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token in {"-i", "--input"}:
            translated.append("--returns")
            if idx + 1 < len(args):
                translated.append(args[idx + 1])
                idx += 2
                continue
            idx += 1
            continue
        if token.startswith("--input="):
            translated.append("--returns=" + token.split("=", 1)[1])
            idx += 1
            continue
        translated.append(token)
        idx += 1
    return translated


def _translate_trend_run_args(args: list[str]) -> tuple[list[str], bool, bool]:
    translated: list[str] = []
    output_seen = False
    out_dir_seen = False
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--artefacts":
            out_dir_seen = True
            translated.append("--out")
            if idx + 1 < len(args):
                translated.append(args[idx + 1])
                idx += 2
                continue
            idx += 1
            continue
        if token.startswith("--artefacts="):
            out_dir_seen = True
            translated.append("--out=" + token.split("=", 1)[1])
            idx += 1
            continue
        if token == "-o":
            output_seen = True
            translated.append("--output")
            if idx + 1 < len(args):
                translated.append(args[idx + 1])
                idx += 2
                continue
            idx += 1
            continue
        if token.startswith("-o="):
            output_seen = True
            translated.append("--output=" + token.split("=", 1)[1])
            idx += 1
            continue
        if token == "--output":
            output_seen = True
            translated.append(token)
            if idx + 1 < len(args):
                translated.append(args[idx + 1])
                idx += 2
                continue
            idx += 1
            continue
        if token.startswith("--output="):
            output_seen = True
            translated.append(token)
            idx += 1
            continue
        translated.append(token)
        idx += 1
    return translated, output_seen, out_dir_seen


def trend_analysis(argv: Sequence[str] | None = None) -> int:
    _warn_deprecated("trend-analysis", "trend run")
    args = list(argv) if argv is not None else list(sys.argv[1:])
    cleaned, removed = _drop_detailed(args)
    if removed:
        print(
            "Note: '--detailed' output is not supported in 'trend run'.",
            file=sys.stderr,
        )
    return trend_cli.main(["run", *cleaned])


def trend_multi_analysis(argv: Sequence[str] | None = None) -> int:
    _warn_deprecated("trend-multi-analysis", "trend run")
    args = list(argv) if argv is not None else list(sys.argv[1:])
    cleaned, removed = _drop_detailed(args)
    if removed:
        print(
            "Note: '--detailed' output is not supported in 'trend run'.",
            file=sys.stderr,
        )
    return trend_cli.main(["run", *cleaned])


def trend_model(argv: Sequence[str] | None = None) -> int:
    _warn_deprecated("trend-model", "trend")
    args = list(argv) if argv is not None else list(sys.argv[1:])
    if "--check" in args:
        return trend_cli.main(["check"])
    if not args:
        return trend_cli.main(args)
    command, rest = args[0], args[1:]
    if command == "gui":
        return trend_cli.main(["app", *rest])
    if command == "run":
        translated = _translate_trend_model_run_args(rest)
        return trend_cli.main(["run", *translated])
    return trend_cli.main(args)


def trend_app(argv: Sequence[str] | None = None) -> int:
    _warn_deprecated("trend-app", "trend app")
    args = list(argv) if argv is not None else list(sys.argv[1:])
    return trend_cli.main(["app", *args])


def trend_run(argv: Sequence[str] | None = None) -> int:
    _warn_deprecated("trend-run", "trend report")
    args = list(argv) if argv is not None else list(sys.argv[1:])
    translated, output_seen, out_dir_seen = _translate_trend_run_args(args)
    if not output_seen and not out_dir_seen:
        translated.extend(["--output", DEFAULT_TREND_RUN_OUTPUT])
    return trend_cli.main(["report", *translated])


def trend_quick_report(argv: Sequence[str] | None = None) -> int:
    _warn_deprecated("trend-quick-report", "trend quick-report")
    args = list(argv) if argv is not None else list(sys.argv[1:])
    return trend_cli.main(["quick-report", *args])
