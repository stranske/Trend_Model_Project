# Branching and workflow triggers

When guards or health workflows specify explicit branch filters, list the current default branch (`phase-3`). Keep `main` only where a workflow intentionally supports the remaining compatibility path.

During any default-branch rename, audit the workflow triggers under `.github/workflows/` and update branch lists accordingly before completing the rename.
