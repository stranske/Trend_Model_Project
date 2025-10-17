# Automated Autofix & Type Hygiene Pipeline

This repository includes an extended **autofix** workflow that standardises style and performs *lightweight* type hygiene automatically on pull requests. The entrypoint is `.github/workflows/pr-02-autofix.yml` (PR trigger), which shares the same reusable composite used by the post-CI follower `maint-46-post-ci.yml`.

## What It Does (Scope)
1. Code formatting & style
- Early `ruff check --fix --exit-zero` sweep that runs before the heavier composite so trivial whitespace/import issues are cleaned even when later phases short-circuit. The step installs the pinned Ruff version when `.github/workflows/autofix-versions.env` is present and otherwise falls back to the latest release.【F:.github/workflows/reusable-18-autofix.yml†L121-L148】
  - Full composite run covering `ruff`, `black`, `isort`, and `docformatter` with both safe and targeted lint passes.【F:.github/actions/autofix/action.yml†L34-L110】
   - `black` (code formatting)
   - `isort` (import sorting where unambiguous)
   - `docformatter` (docstring wrapping)
2. Type stub installation
   - Runs `mypy --install-types --non-interactive` to fetch missing third‑party stubs where available.
3. Targeted type hygiene
   - Executes `scripts/auto_type_hygiene.py` which injects `# type: ignore[import-untyped]` ONLY for a small allowlist of third‑party imports (default: `yaml`).
   - Idempotent: re-running does not duplicate ignores.
4. Warm mypy cache
   - Invokes a non-blocking mypy run so subsequent CI/type checks are faster.

If any step produces changes, the workflow auto‑commits them back to the PR branch with a conventional message (e.g. `chore(autofix): style + type hygiene`).

## Result Labels & Status Comment
- Same-repo branches that receive an autofix commit automatically gain the `autofix:applied` label; forked PRs receive `autofix:patch` when an artifact is uploaded.【F:.github/workflows/reusable-18-autofix.yml†L187-L281】
- Runs that finish clean (no diff) toggle `autofix:clean`, while any unresolved diagnostics append `autofix:debt` alongside the primary outcome label.【F:.github/workflows/reusable-18-autofix.yml†L283-L371】
- Every execution updates a single status comment with an **Autofix result** block that lists the applied labels so reviewers can confirm the outcome at a glance.【F:.github/workflows/reusable-18-autofix.yml†L187-L297】【F:scripts/build_autofix_pr_comment.py†L218-L253】

## What It Intentionally Does NOT Do
- It does **not** attempt deep structural refactors or resolve complex type inference issues.
- It does **not** add stubs for internal modules or apply `# type: ignore` broadly.
- It does **not** silence genuine semantic/type errors (e.g. wrong argument counts, incompatible assignments).
- It does **not** enforce exhaustive strict mypy modes across legacy areas not yet migrated.

This narrow scope keeps the automation safe, deterministic, and low‑noise.

## Extending the Allowlist
The allowlist for untyped third‑party imports lives in `scripts/auto_type_hygiene.py`:
```python
ALLOWLIST = {"yaml"}
```
To add another untyped module (e.g. `fastapi` if desired):
1. Edit the set to include the module root name.
2. Run:
   ```bash
   python scripts/auto_type_hygiene.py --check
   ```
3. Commit the resulting changes (if any).

Prefer adding *only* modules that reliably lack published stubs or where partial typing would otherwise generate persistent noise.

## Local Developer Workflow
During active development:
```bash
./scripts/dev_check.sh --changed --fix   # ultra-fast sanity (2-5s)
./scripts/validate_fast.sh --fix         # adaptive validation (5-30s)
./scripts/run_tests.sh                   # full test suite (15-25s)
```
Before pushing a feature branch:
```bash
./scripts/validate_fast.sh --fix
./scripts/run_tests.sh
```
If CI leaves an autofix commit on your branch, **pull/rebase** before adding further changes.

## When To Escalate Beyond Automation
Open a focused PR (or issue) for:
- Introducing or refactoring complex protocols / generics.
- Replacing dynamic imports with explicit optional dependency shims.
- Tightening mypy configuration (e.g. enabling `disallow-any-generics`).
- Broad import hygiene sweeps beyond the allowlist.

## Design Principles
| Principle | Rationale |
|-----------|-----------|
| Idempotent | Re-running produces no further diffs when clean. |
| Minimal Surface | Only low-risk fixes applied automatically. |
| Deterministic | Output stable across environments. |
| Transparent | All mutations committed with explicit chore message. |
| Extensible | Allowlist easily adjusted with single source of truth. |

## Troubleshooting
| Symptom | Likely Cause | Action |
|---------|--------------|-------|
| Repeated autofix commits | Unformatted notebooks (black lacks jupyter extra) | Install `black[jupyter]` locally or exclude notebooks. |
| New mypy errors after a rebase | Upstream typing tightened | Resolve manually; avoid blanket `# type: ignore` unless justified. |
| Missing ignore for known untyped lib | Not in allowlist | Add to `ALLOWLIST` in script, run autofix locally. |
| CI autofix skipped | No diff produced | Confirm local environment replicates tool versions. |

## Verification scenarios

Run these quick checks whenever the PR-02 autofix lane changes to confirm Issue #2649’s safeguards remain in place:

### Same-repo opt-in
1. Open a branch in the main repository with a deliberate lint issue (for example, reorder an import) and add the `autofix` label.
2. Verify the **PR 02 Autofix** workflow triggers exactly one `apply` job and cancels any superseded runs when you push extra commits or re-run the workflow from the UI.
3. Confirm the comment updated in place under the `<!-- autofix-status: DO NOT EDIT -->` marker shows the applied commit link and an "Autofix result" section.

### Fork opt-in
1. From a fork, open a PR with the `autofix` label and a trivially fixable lint issue.
2. Ensure the workflow uploads an `autofix-patch-pr-<number>` artifact, applies the `autofix:patch` label, and the status comment explains how to apply the patch locally.
3. Download and apply the patch with `git am` to confirm it replays cleanly, then push manually to complete the fix.

### Label outcomes
1. Re-run the workflow on a clean branch (no staged lint issues) and verify the status comment reports “No changes required” with the `autofix:clean` label.【F:.github/workflows/reusable-18-autofix.yml†L283-L297】
2. Introduce a trivial lint (e.g. reorder imports) and confirm the rerun pushes a commit, applies `autofix:applied`, and lists the label in the comment.【F:.github/workflows/reusable-18-autofix.yml†L187-L209】
3. If the run leaves residual diagnostics, expect `autofix:debt` to accompany either result label.【F:.github/workflows/reusable-18-autofix.yml†L299-L371】

### Label gating sanity check
1. Remove the `autofix` label (or open a fresh PR without it) and trigger the workflow via the **Re-run** button.
2. Confirm the `apply` job is skipped and no new comments are posted, demonstrating the label gate is working as expected.

## Future Enhancements (Optional Backlog)
- Add metrics: record autofix delta lines per run.
- Dry-run mode for PR comments instead of commits (toggle by label).
- Expand hygiene to detect *unused* ignores and prune them automatically.

---
Maintainers: keep this doc aligned with any future changes to `.github/actions/autofix/action.yml` or `scripts/auto_type_hygiene.py` so contributors understand the automation contract.
