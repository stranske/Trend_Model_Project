from __future__ import annotations

import re
from pathlib import Path

ACTIVE_DOCS = (
    Path("Agents.md"),
    Path("DOCKER_QUICKSTART.md"),
    *Path("docs").rglob("*.md"),
)
# Audit records preserve historical names as evidence and are not usage guides.
EXCLUDED_DOC_ROOTS = (Path("docs/archive"), Path("docs/audits"), Path("docs/keepalive"))
# Split ``trend_analysis/`` literals below so this file does not self-match the
# legacy-surface scanner while still asserting retired runner names.
REMOVED_RUNNERS = (
    "trend_analysis." + "run_analysis",
    "trend_analysis." + "run_multi_analysis",
)
CURRENT_WEIGHTING_GUIDES = (
    Path("docs/config.md"),
    Path("docs/UserGuide.md"),
    Path("docs/phase-2/Agents.md"),
    Path("docs/phase-3/MonteCarlo.md"),
)
CURRENT_SIGNAL_GUIDES = (Path("docs/TrendSignalSettings.md"),)


def _active_doc_texts() -> list[tuple[Path, str]]:
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in ACTIVE_DOCS
        if path.exists() and not any(path.is_relative_to(root) for root in EXCLUDED_DOC_ROOTS)
    ]


def test_active_docs_do_not_reference_removed_runners() -> None:
    offenders: list[str] = []
    for path in ACTIVE_DOCS:
        if not path.exists() or any(path.is_relative_to(root) for root in EXCLUDED_DOC_ROOTS):
            continue
        text = path.read_text(encoding="utf-8")
        for runner in REMOVED_RUNNERS:
            if runner in text:
                offenders.append(f"{path}: {runner}")

    assert not offenders, "Active documentation references removed runner modules:\n" + "\n".join(
        offenders
    )


def test_active_docs_do_not_describe_removed_runner_symbols() -> None:
    semantic_tokens = ("pipeline._run_" + "analysis(", "run_" + "analysis.py")
    offenders = [
        f"{path}: {token}"
        for path, text in _active_doc_texts()
        for token in semantic_tokens
        if token in text
    ]

    assert not offenders, "Active documentation describes removed runner symbols:\n" + "\n".join(
        offenders
    )


def test_active_docs_do_not_advertise_removed_executable_aliases() -> None:
    retired_executable = "trend-" + "model"
    command_pattern = re.compile(rf"(?m)^\s*(?:\$\s*)?{re.escape(retired_executable)}(?:\s|$)")
    offenders = [str(path) for path, text in _active_doc_texts() if command_pattern.search(text)]

    assert (
        not offenders
    ), "Active documentation advertises a retired executable alias:\n" + "\n".join(offenders)


def test_streamlit_page_index_matches_current_pages() -> None:
    index_text = Path("docs/INDEX.md").read_text(encoding="utf-8")
    section = index_text.split("**`streamlit_app/` folder:**", maxsplit=1)[1].split(
        "**`tests/` folder:**", maxsplit=1
    )[0]
    documented = set(re.findall(r"`([1-9][^`]+\.py)`", section))
    actual = {path.name for path in Path("streamlit_app/pages").glob("*.py")}

    assert documented == actual


def test_current_guides_use_canonical_weighting_shape() -> None:
    forbidden = (
        "weighting_" + "scheme:",
        "portfolio.weighting." + "method",
        "weighting:\n    " + "method:",
        "weighting: " + "equal",
    )
    offenders = [
        f"{path}: {token}"
        for path in CURRENT_WEIGHTING_GUIDES
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]

    assert not offenders, "Current guides use removed weighting shapes:\n" + "\n".join(offenders)


def test_current_guides_use_canonical_signal_shape() -> None:
    forbidden = (
        "trend_" + "window",
        "trend_" + "lag",
        "trend_" + "min_periods",
        "trend_" + "zscore",
        "trend_" + "vol_adjust",
        "trend_" + "vol_target",
        "signals:\n  " + "trend:",
    )
    offenders = [
        f"{path}: {token}"
        for path in CURRENT_SIGNAL_GUIDES
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]

    assert not offenders, "Current guides use removed signal shapes:\n" + "\n".join(offenders)
