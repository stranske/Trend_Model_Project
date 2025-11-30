<!-- bootstrap for codex on issue #1669 -->

## Issue #1669 – Workflow hygiene follow-up

### Focus areas
- Finish renaming or retiring any workflows that still used legacy slugs (`ci.yml`, `gate.yml`, `cleanup-codex-bootstrap.yml`, etc.).
- Remove the archived copies under `Old/.github/workflows/` and `.github/workflows/archive/`; log the disposition in `ARCHIVE_WORKFLOWS.md`.
- Produce an authoritative inventory of the remaining workflows with triggers + consumers so future audits have a fast starting point.

### Acceptance criteria
1. Actions tab lists only `pr-*`, `maint-*`, `agents-*`, and `reusable-*` workflows.
2. Historical archive directories are gone; `ARCHIVE_WORKFLOWS.md` documents where to find replacements in git history.
3. `docs/ci/WORKFLOW_SYSTEM.md`, `.github/workflows/README.md`, and `docs/WORKFLOW_GUIDE.md` reflect the cleaned inventory and reference the new guard test coverage.
4. Guard tests under `tests/test_workflow_*.py` enforce the naming rules and pass locally (`pytest tests/test_workflow_*.py`).
