import pathlib

ALLOWED_PREFIXES = (
    "pr-",
    "maint-",
    "agents-",
    "reusable-",
    "reuse-",
    "autofix",
    "enforce-",
    "health-",
    "selftest-",
)
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


def test_inventory_docs_list_all_workflows():
    docs = {
        "docs/ci/WORKFLOWS.md": pathlib.Path("docs/ci/WORKFLOWS.md").read_text(
            encoding="utf-8"
        ),
        "docs/WORKFLOW_GUIDE.md": pathlib.Path("docs/WORKFLOW_GUIDE.md").read_text(
            encoding="utf-8"
        ),
    }

    def _listed(contents: str, slug: str) -> bool:
        options = (
            f"`{slug}`",
            f"`.github/workflows/{slug}`",
        )
        return any(option in contents for option in options)

    missing_by_doc = {
        doc_name: [
            path.name for path in _workflow_paths() if not _listed(contents, path.name)
        ]
        for doc_name, contents in docs.items()
    }
    failures = {doc: names for doc, names in missing_by_doc.items() if names}
    assert not failures, f"Workflow inventory missing entries: {failures}"


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


def test_chatgpt_issue_sync_workflow_present_and_intact():
    path = WORKFLOW_DIR / "agents-63-chatgpt-issue-sync.yml"
    assert (
        path.exists()
    ), "agents-63-chatgpt-issue-sync.yml must remain in the workflow inventory"
    text = path.read_text(encoding="utf-8")
    assert (
        ".github/scripts/decode_raw_input.py" in text
    ), "Workflow must normalize input using decode_raw_input.py"
    assert (
        ".github/scripts/parse_chatgpt_topics.py" in text
    ), "Workflow must parse topics via parse_chatgpt_topics.py"
    assert (
        "github.rest.issues.create" in text
    ), "Workflow must create or update GitHub issues"


EXPECTED_NAMES = {
    "agents-63-codex-issue-bridge.yml": "Agents 63 Codex Issue Bridge",
    "agents-63-chatgpt-issue-sync.yml": "Agents 63 ChatGPT Issue Sync",
    "agents-64-verify-agent-assignment.yml": "Agents 64 Verify Agent Assignment",
    "agents-70-orchestrator.yml": "Agents 70 Orchestrator",
    "pr-02-autofix.yml": "PR 02 Autofix",
    "health-40-repo-selfcheck.yml": "Health 40 Repo Selfcheck",
    "health-41-repo-health.yml": "Health 41 Repo Health",
    "health-42-actionlint.yml": "Health 42 Actionlint",
    "health-43-ci-signature-guard.yml": "Health 43 CI Signature Guard",
    "health-44-gate-branch-protection.yml": "Health 44 Gate Branch Protection",
    "maint-45-cosmetic-repair.yml": "Maint 45 Cosmetic Repair",
    "maint-46-post-ci.yml": "Maint 46 Post CI",
    "pr-00-gate.yml": "Gate",
    "reusable-10-ci-python.yml": "Reusable CI",
    "reusable-12-ci-docker.yml": "Reusable Docker Smoke",
    "reusable-16-agents.yml": "Reusable 16 Agents",
    "reusable-18-autofix.yml": "Reusable 18 Autofix",
    "selftest-81-reusable-ci.yml": "Selftest 81 Reusable CI",
    "selftest-runner.yml": "Selftest Runner",
}
