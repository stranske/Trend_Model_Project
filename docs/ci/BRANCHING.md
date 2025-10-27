# Branching and workflow triggers

When guards or health workflows specify explicit branch filters, list both the current default branch (`main` today) and `phase-2-dev`. This keeps protections active if the default branch changes.

During any default-branch rename, audit the workflow triggers under `.github/workflows/` and update branch lists accordingly before completing the rename. Always restore both the default branch and `phase-2-dev` in the trigger list once the rename is complete.
