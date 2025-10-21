"""Workflow guard regression tests for autofix automation."""

from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List

import yaml

WORKFLOWS = pathlib.Path(".github/workflows")
GITHUB_SCRIPTS = pathlib.Path(".github/scripts")


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


WORKFLOW_FILE = "maint-46-post-ci.yml"
HELPER_FILE = "maint-post-ci.js"


def test_autofix_workflow_uses_repo_commit_prefix() -> None:
    data = _load_yaml(WORKFLOW_FILE)
    prefix_expr = data.get("env", {}).get("COMMIT_PREFIX", "")
    assert "AUTOFIX_COMMIT_PREFIX" in prefix_expr
    assert "chore(autofix):" in prefix_expr


def test_reusable_autofix_guard_applies_to_all_steps() -> None:
    data = _load_yaml("reusable-18-autofix.yml")
    steps = data["jobs"]["autofix"]["steps"]
    missing = _guarded_follow_up_steps(steps)
    assert not missing, f"Reusable autofix steps missing guard condition: {missing}"


def _load_helper(name: str) -> str:
    helper_path = GITHUB_SCRIPTS / name
    assert helper_path.exists(), f"Expected helper script to exist: {name}"
    return helper_path.read_text(encoding="utf-8")


def _extract_trivial_keywords(source: str) -> set[str]:
    patterns = (
        r"TRIVIAL_KEYWORDS\s*\|\|\s*'([^']+)'",
        r"AUTOFIX_TRIVIAL_KEYWORDS\s*\|\|\s*'([^']+)'",
    )
    match = None
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            break
    if not match:
        raise AssertionError(
            "Default AUTOFIX_TRIVIAL_KEYWORDS clause missing from autofix helper"
        )
    return {token.strip() for token in match.group(1).split(",") if token.strip()}


def test_autofix_trivial_keywords_cover_lint_type_and_tests() -> None:
    data = _load_yaml(WORKFLOW_FILE)
    failure_step = next(
        step for step in data["jobs"]["context"]["steps"] if step.get("id") == "failure"
    )
    script = failure_step["with"]["script"]
    assert "require('./.github/scripts/maint-post-ci.js')" in script

    helper_source = _load_helper(HELPER_FILE)
    keywords = _extract_trivial_keywords(helper_source)
    expected = {"lint", "mypy", "test"}
    missing = expected.difference(keywords)
    assert not missing, f"Autofix trivial keywords missing expected tokens: {missing}"
    assert "label" in keywords, "Label failures should remain autofix-eligible"
