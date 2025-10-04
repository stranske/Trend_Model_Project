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
