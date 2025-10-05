import pathlib

ALLOWED_PREFIXES = ("pr-", "maint-", "agents-", "reusable-")
WORKFLOW_DIR = pathlib.Path(".github/workflows")


def _workflow_paths():
    return sorted(WORKFLOW_DIR.glob("*.yml"))


def test_workflow_slugs_follow_wfv1_prefixes():
    non_compliant = [
        path.name
        for path in _workflow_paths()
        if not path.name.startswith(ALLOWED_PREFIXES)
    ]
    assert (
        not non_compliant
    ), f"Non-compliant workflow slug(s) detected outside {ALLOWED_PREFIXES}: {non_compliant}"


def test_archive_directories_removed():
    assert not (
        WORKFLOW_DIR / "archive"
    ).exists(), (
        ".github/workflows/archive/ should be removed (tracked in ARCHIVE_WORKFLOWS.md)"
    )
    legacy_dir = pathlib.Path("Old/.github/workflows")
    assert not legacy_dir.exists(), "Old/.github/workflows/ should remain deleted"


def test_inventory_doc_lists_all_workflows():
    audit_doc = pathlib.Path("WORKFLOW_AUDIT_TEMP.md").read_text(encoding="utf-8")
    missing = [
        path.name for path in _workflow_paths() if f"`{path.name}`" not in audit_doc
    ]
    assert not missing, f"Workflow inventory missing entries for: {missing}"


def test_workflow_names_match_filename_convention():
    mismatches = {}
    for path in _workflow_paths():
        expected = EXPECTED_NAMES.get(path.name)
        assert expected, f"Missing expected name mapping for {path.name}"
        data = path.read_text(encoding="utf-8").splitlines()
        name_line = next((line for line in data if line.startswith("name:")), None)
        assert name_line is not None, f"Workflow {path.name} missing name field"
        actual = name_line.split(":", 1)[1].strip()
        if actual != expected:
            mismatches[path.name] = actual
    assert not mismatches, f"Workflow name mismatch detected: {mismatches}"
EXPECTED_NAMES = {
    "agents-70-orchestrator.yml": "Agents 70 Orchestrator",
    "maint-02-repo-health.yml": "Maint 02 Repo Health",
    "maint-30-post-ci-summary.yml": "Maint 30 Post CI Summary",
    "maint-32-autofix.yml": "Maint 32 Autofix",
    "maint-33-check-failure-tracker.yml": "Maint 33 Check Failure Tracker",
    "maint-36-actionlint.yml": "Maint 36 Actionlint",
    "maint-40-ci-signature-guard.yml": "Maint 40 CI Signature Guard",
    "maint-90-selftest.yml": "Maint 90 Selftest",
    "pr-10-ci-python.yml": "PR 10 CI Python",
    "pr-12-docker-smoke.yml": "PR 12 Docker Smoke",
    "reusable-70-agents.yml": "Reusable 70 Agents",
    "reusable-90-ci-python.yml": "Reusable 90 CI Python",
    "reusable-92-autofix.yml": "Reusable 92 Autofix",
    "reusable-94-legacy-ci-python.yml": "Reusable 94 Legacy CI Python",
    "reusable-99-selftest.yml": "Reusable 99 Selftest",
}

