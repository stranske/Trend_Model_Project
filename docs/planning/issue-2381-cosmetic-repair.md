# Issue #2381 – Cosmetic Test Repair Mode Planning

## Scope and Key Constraints
- **Workflow scope**: Introduce a GitHub Actions workflow (`.github/workflows/cosmetic-repair.yml`) that is **manual dispatch only** (`workflow_dispatch`) and never runs on push, PR, or schedule triggers.
- **Permissions**: The workflow must request the minimal permissions required to push branches and open PRs (`contents: write`, `pull-requests: write`).
- **Execution environment**: Jobs should run on `ubuntu-latest` with the project’s standard Python toolchain setup (reuse existing helper composite/actions if available; otherwise bootstrap Python 3.11 with `pip` caching consistent with other CI flows).
- **Repository hygiene**: Cosmetic repairs must only touch files identified as strictly safe for mechanical adjustment (formatting, numeric tolerances, seed drift fixtures). Guard rails should prevent edits to core logic modules.
- **Branching strategy**: Workflow-created branches should follow a predictable prefix (e.g., `autofix/cosmetic-repair-*`) and be auto-deleted after merge/close to reduce branch clutter.
- **Idempotency**: Re-running the workflow without new drift should result in a no-op (detect clean tree and exit without creating redundant PRs).
- **Observability**: Log summary of detected issues and produced patches; expose artifacts or job summary for reviewers.

## Acceptance Criteria / Definition of Done
1. **Workflow functionality**
   - Manual dispatch runs `pytest -q` (or a focused subset if runtime constraints demand) and captures failures.
   - On detecting known cosmetic failure signatures, the workflow executes `scripts/ci_cosmetic_repair.py` to adjust tolerances/formatting.
   - The script commits changes to a temporary branch, opens a PR labeled `testing` and `autofix:applied`, and includes a guard comment in each touched file indicating the change was auto-generated.
2. **Repair script behavior**
   - Enumerates allowed repair patterns (e.g., reformatting expected-output files, updating numeric tolerances within predefined bounds, refreshing seeded fixtures) and refuses to run for unrecognized diffs.
   - Produces deterministic output (stable sorting, consistent formatting) to keep reruns reproducible.
   - Provides a dry-run mode for local verification and unit tests.
3. **Safety checks**
   - If the script cannot fully repair the detected drift, it exits with a clear message and non-zero status so the workflow surfaces the failure.
   - Unit tests cover representative cosmetic fixes and negative cases (attempted changes outside the allowed scope).
4. **Documentation**
   - Add README/operations notes describing how to trigger the workflow, what kinds of fixes it applies, and reviewer expectations.
   - Update any automation inventories to list the new workflow and script.
5. **Developer experience**
   - Provide instructions for running the repair script locally (including prerequisites) and how to simulate the workflow behavior.

## Initial Task Checklist
- [ ] Audit existing CI helpers to reuse environment bootstrap logic (actions, scripts, or caching patterns).
- [ ] Design repairable-case inventory (identify fixture files, tolerance definitions, formatting targets) and codify guard comment conventions.
- [ ] Implement `scripts/ci_cosmetic_repair.py` with CLI flags for `--dry-run`, `--apply`, and logging verbosity.
- [ ] Write targeted unit tests for the repair script under `tests/` (include at least one success and one refusal scenario).
- [ ] Create `.github/workflows/cosmetic-repair.yml` with dispatch inputs (e.g., optional branch name suffix, dry-run toggle) and integrate the script + pytest run.
- [ ] Ensure the workflow sets the required labels when opening PRs and posts a concise job summary (detected issues, files changed).
- [ ] Document usage in `docs/ops/cosmetic-repair.md` (or equivalent) and link from contributor guides if necessary.
- [ ] Validate idempotency by running the script twice on synthetic drift and confirming the second run exits cleanly.
- [ ] Perform end-to-end dry run (locally or via workflow) to confirm branch naming, labeling, and guard comments meet expectations.
