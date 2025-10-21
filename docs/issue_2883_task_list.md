# Issue #2883 – Autofix entry point consolidation

## Scope / key constraints
- Limit changes to Maint 46 Post CI, the reusable autofix composite, supporting scripts, and associated documentation/tests.
- Preserve existing automation behaviour for non-autofix flows (Gate, failure tracker, keepalive) while simplifying the autofix path.
- Keep the diff surface minimal—formatting, lint fixes, and deterministic comment output only.

## Acceptance criteria / definition of done
- Exactly one workflow path can perform autofix actions on pull requests.
- The selected path is gated by a single opt-in label and emits one consolidated summary comment per run.
- Duplicate autofix commits do not occur for the same PR event.

## Work plan
- [x] Inventory current workflows to confirm Maint 46 is the sole autofix entry point and identify any legacy files to remove.
- [x] Refine Maint 46 Post CI to gate reusable autofix invocations on the `autofix:clean` opt-in label and enforce dry-run handling for forks.
- [x] Update the reusable autofix workflow inputs to require explicit `dry_run` and allowed file globs, ensuring fork patches remain read-only.
- [x] Normalize documentation and tooling references to use the `autofix:clean` label without fallback aliases.
- [x] Guarantee the consolidated status comment always lists files touched and the checks re-run, even when no changes occur.
- [x] Extend and adjust workflow/comment unit tests to exercise the unified label contract and summary output.
- [x] Validate acceptance criteria via targeted pytest runs and repository audit, then document completion in this task list.

> ✅ Acceptance criteria satisfied — Maint 46 Post CI is now the lone opt-in autofix path, label-gated, and exercises a dry-run-safe reusable workflow with comprehensive reporting.
