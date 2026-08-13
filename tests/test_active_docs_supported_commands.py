from __future__ import annotations

from pathlib import Path


ACTIVE_DOCS = (
    Path("Agents.md"),
    Path("DOCKER_QUICKSTART.md"),
    *Path("docs").rglob("*.md"),
)
EXCLUDED_DOC_ROOTS = (Path("docs/archive"), Path("docs/keepalive"))
REMOVED_RUNNERS = ("trend_analysis.run_analysis", "trend_analysis.run_multi_analysis")


def test_active_docs_do_not_reference_removed_runners() -> None:
    offenders: list[str] = []
    for path in ACTIVE_DOCS:
        if not path.exists() or any(path.is_relative_to(root) for root in EXCLUDED_DOC_ROOTS):
            continue
        text = path.read_text(encoding="utf-8")
        for runner in REMOVED_RUNNERS:
            if runner in text:
                offenders.append(f"{path}: {runner}")

    assert not offenders, "Active documentation references removed runner modules:\n" + "\n".join(offenders)
