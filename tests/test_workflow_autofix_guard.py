"""Workflow guard regression tests for autofix automation."""

from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List

import yaml

WORKFLOWS = pathlib.Path(".github/workflows")


def _load_yaml(name: str) -> Dict[str, Any]:
    with (WORKFLOWS / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _guarded_follow_up_steps(
    steps: List[Dict[str, Any]], guard_id: str = "guard"
) -> List[str]:
    """Return the names of steps after ``guard_id`` lacking guard
    conditions."""
    missing: List[str] = []
    try:
        guard_index = next(
            index for index, step in enumerate(steps) if step.get("id") == guard_id
        )
    except (
        StopIteration
    ) as exc:  # pragma: no cover - defensive: workflow must define guard
        raise AssertionError(f"Guard step '{guard_id}' missing") from exc

    for step in steps[guard_index + 1 :]:
        condition = step.get("if")
        # Summary/always steps are allowed to run regardless so they can document the skip.
        if isinstance(condition, str) and "always()" in condition:
            continue
        if condition is None or "steps.guard.outputs.skip" not in str(condition):
            missing.append(step.get("name", "<unnamed>"))
    return missing


def test_autofix_workflow_uses_repo_commit_prefix() -> None:
    data = _load_yaml("maint-32-autofix.yml")
    prefix_expr = data.get("env", {}).get("COMMIT_PREFIX", "")
    assert "AUTOFIX_COMMIT_PREFIX" in prefix_expr
    assert "chore(autofix):" in prefix_expr


def test_reusable_autofix_guard_applies_to_all_steps() -> None:
    data = _load_yaml("reusable-92-autofix.yml")
    steps = data["jobs"]["autofix"]["steps"]
    missing = _guarded_follow_up_steps(steps)
    assert not missing, f"Reusable autofix steps missing guard condition: {missing}"


def _extract_trivial_keywords(script: str) -> set[str]:
    match = re.search(r"TRIVIAL_KEYWORDS\s*\|\|\s*'([^']+)'", script)
    if not match:
        raise AssertionError(
            "Default TRIVIAL_KEYWORDS clause missing from autofix workflow"
        )
    return {token.strip() for token in match.group(1).split(",") if token.strip()}


def test_autofix_trivial_keywords_cover_lint_type_and_tests() -> None:
    data = _load_yaml("maint-32-autofix.yml")
    failure_step = next(
        step for step in data["jobs"]["context"]["steps"] if step.get("id") == "failure"
    )
    script = failure_step["with"]["script"]
    keywords = _extract_trivial_keywords(script)
    expected = {"lint", "mypy", "test"}
    missing = expected.difference(keywords)
    assert not missing, f"Autofix trivial keywords missing expected tokens: {missing}"
    assert "label" in keywords, "Label failures should remain autofix-eligible"
