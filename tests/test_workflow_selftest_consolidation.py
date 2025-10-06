from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_DIR = Path(".github/workflows")


def test_only_expected_selftest_workflows() -> None:
    selftest_workflows = sorted(
        path.name for path in WORKFLOW_DIR.glob("*selftest*.yml")
    )
    assert selftest_workflows == [
        "maint-90-selftest.yml",
        "reusable-99-selftest.yml",
    ], (
        "Unexpected self-test workflows detected. Expected only maint-90-selftest.yml "
        "and reusable-99-selftest.yml."
    )


def test_selftest_caller_configuration() -> None:
    caller_path = WORKFLOW_DIR / "maint-90-selftest.yml"
    assert caller_path.exists(), "Maint 90 Selftest workflow is missing"

    caller = yaml.safe_load(caller_path.read_text(encoding="utf-8"))

    # GitHub Actions workflows use the ``on`` key, which YAML 1.1 parsers (including
    # PyYAML) historically coerce to the boolean ``True``.  Attempt to retrieve the
    # trigger block using both spellings so the assertion remains stable regardless
    # of the loader semantics.
    triggers = caller.get("on") or caller.get(True)
    assert triggers, "Maint 90 Selftest workflow must define triggers"
    assert (
        "workflow_dispatch" in triggers
    ), "Selftest workflow must allow manual dispatch"
    allowed_triggers = {"workflow_dispatch", "schedule"}
    unexpected_triggers = set(triggers) - allowed_triggers
    assert not unexpected_triggers, (
        "Maint 90 Selftest should only expose workflow_dispatch and an optional "
        f"schedule trigger (unexpected: {sorted(unexpected_triggers)})"
    )

    jobs = caller.get("jobs")
    assert jobs and "selftest" in jobs, "Maint 90 Selftest must define a 'selftest' job"
    job = jobs["selftest"]
    assert (
        job.get("uses") == "./.github/workflows/reusable-99-selftest.yml"
    ), "Maint 90 Selftest must delegate to reusable-99-selftest.yml via the 'uses' key"
    assert job.get("secrets") == "inherit", "Selftest job should inherit secrets"
