import pathlib
import yaml

WORKFLOWS = pathlib.Path(".github/workflows")
MARKER = "<!-- merge-manager-rationale -->"


def _load_yaml(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_merge_manager_exists():
    mm = WORKFLOWS / "merge-manager.yml"
    assert mm.exists(), "merge-manager.yml must exist as unified merge policy workflow"
    data = mm.read_text(encoding="utf-8")
    assert (
        MARKER in data
    ), "merge-manager rationale marker missing (single-comment invariant)"


def test_legacy_merge_workflows_archived():
    # legacy files should not sit in active workflows dir
    for legacy in ("autoapprove.yml", "enable-automerge.yml"):
        assert not (
            WORKFLOWS / legacy
        ).exists(), f"Legacy workflow {legacy} still active; must be archived"
    # archived copies should exist under Old/.github/workflows
    for legacy in ("autoapprove.yml", "enable-automerge.yml"):
        archived = pathlib.Path("Old/.github/workflows") / legacy
        assert (
            archived.exists()
        ), f"Archived workflow {legacy} missing under Old/.github/workflows"


def test_merge_manager_core_steps_present():
    content = (WORKFLOWS / "merge-manager.yml").read_text(encoding="utf-8")
    # Basic heuristic checks for critical steps / actions
    assert "actions/checkout" in content, "checkout step missing"
    assert (
        "peter-evans/enable-pull-request-automerge" in content
    ), "auto-merge enable action missing"
    # Accept either direct GitHub REST createReview usage (github-script) or gh cli invocation
    assert (
        "createReview" in content
        or "gh api repos/${{ github.repository }}/pulls/${{ github.event.pull_request.number }}/reviews"
        in content
    ), "approval invocation missing (expected createReview call or gh api command)"
    assert "<!-- merge-manager-rationale -->" in content, "rationale marker missing"


def test_commit_prefix_is_quoted():
    data = _load_yaml(WORKFLOWS / "merge-manager.yml")
    env = data.get("env", {})
    prefix = env.get("COMMIT_PREFIX")
    assert isinstance(prefix, str) and prefix.startswith(
        "chore("
    ), "COMMIT_PREFIX must be a quoted string starting with 'chore('"
