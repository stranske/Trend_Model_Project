from __future__ import annotations

from pathlib import Path

ACTIVE_DOCS = (
    Path("Agents.md"),
    Path("DOCKER_QUICKSTART.md"),
    *Path("docs").rglob("*.md"),
)
EXCLUDED_DOC_ROOTS = (Path("docs/archive"), Path("docs/keepalive"))
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
